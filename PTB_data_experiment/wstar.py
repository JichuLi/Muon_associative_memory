"""Reference W* solver and geometry summaries for v2 settings.

The v2 policy is deliberately conservative about skipping: only settings whose
softmax optimum is known to be non-finite, such as hard copy, are skipped.
Zero-support targets are still attempted and the resulting reference is marked
as approximate unless the gradient criterion certifies stationarity.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from core import Problem
from resources import LaptopGuard, atomic_json
from targets import Corpus, digest


HERE = Path(__file__).resolve().parent


def minimizer_dir(case_id):
    return HERE/"results"/"v2"/case_id/"minimizer"


def _cosine(A, B):
    a = A.double()
    b = B.double()
    denom = torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    if float(denom) == 0:
        return math.nan
    return float((a*b).sum()/denom)


def _stable_rank(s):
    values = s.double()
    if len(values) == 0 or float(values[0]) == 0:
        return 0.0
    return float(values.square().sum()/(values[0]**2))


def _spectral_summaries(s):
    values = s.double()
    if len(values) == 0 or float(values[0]) == 0:
        return {
            "weight_nuclear_effective_rank": 0.0,
            "weight_rank_rel_ge_1e-2": 0,
            "weight_rank_rel_ge_1e-3": 0,
            "weight_rank_rel_ge_1e-4": 0,
            "weight_rank_rel_ge_1e-6": 0,
        }
    fro2 = float(values.square().sum())
    nuclear = float(values.sum())
    rel = values/values[0]
    return {
        "weight_nuclear_effective_rank": nuclear*nuclear/max(fro2,1e-30),
        "weight_rank_rel_ge_1e-2": int((rel>=1e-2).sum()),
        "weight_rank_rel_ge_1e-3": int((rel>=1e-3).sum()),
        "weight_rank_rel_ge_1e-4": int((rel>=1e-4).sum()),
        "weight_rank_rel_ge_1e-6": int((rel>=1e-6).sum()),
    }


def should_attempt_reference(target, case, config):
    spec = case["target"]
    if case["id"] in set(config["wstar"].get("skip_case_ids", [])):
        return False, "skipped_definitely_no_finite_optimum"
    if spec["kind"] == "copy" and float(spec.get("association", 0)) >= 1.0:
        return False, "skipped_definitely_no_finite_optimum"
    return True, "attempt_reference"


def lbfgs_oracle(problem):
    return {
        "U": problem.U.double(),
        "E": problem.E.double(),
        "P": problem.P.double(),
        "bias": problem.bias.double(),
        "column_mass": problem.column_mass.double(),
        "pi": problem.pi64,
    }


def differentiable_loss(oracle, W):
    z = oracle["U"]@(W@oracle["E"])+oracle["bias"][:, None]
    z = z-z.amax(0, keepdim=True).detach()
    logz = torch.logsumexp(z, dim=0)
    expected = (oracle["P"]*z).sum(0)
    ce = logz*oracle["column_mass"]-expected
    return oracle["pi"]@ce


def refine_lbfgs(problem, steps, goal, guard, progress_path=None, case_id=None):
    problem.reset()
    oracle = lbfgs_oracle(problem)
    W = torch.zeros((problem.d, problem.d), device=problem.device, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [W],
        lr=1.0,
        max_iter=1,
        max_eval=int(problem.config["wstar"].get("lbfgs_max_eval_per_step", 8)),
        history_size=int(problem.config["wstar"].get("lbfgs_history_size", 20)),
        tolerance_grad=goal,
        tolerance_change=float(problem.config["wstar"].get("lbfgs_tolerance_change", 1e-9)),
        line_search_fn=problem.config["wstar"].get("lbfgs_line_search", "strong_wolfe"),
    )
    best = dict(loss=math.inf, step=-1, grad_fro=math.inf, W=W.detach().clone(), status="started")
    previous_loss = math.inf
    no_gain = 0
    trace = []
    patience = int(problem.config["wstar"].get("lbfgs_stall_patience", 12))
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss_tensor = differentiable_loss(oracle, W)
        if not bool(torch.isfinite(loss_tensor)):
            best["status"] = "nonfinite_loss"
            break
        loss_tensor.backward()
        if W.grad is None or not bool(torch.isfinite(W.grad).all()):
            best["status"] = "nonfinite_gradient"
            break
        grad_fro = float(torch.linalg.vector_norm(W.grad.detach()))
        loss = float(loss_tensor.detach())
        trace.append((step, loss, grad_fro))
        if loss < best["loss"]:
            best.update(loss=loss, step=step, grad_fro=grad_fro, W=W.detach().clone(), status="running")
        if grad_fro <= goal:
            best["status"] = "solved"
            break
        if previous_loss-loss < max(1e-9, 1e-7*max(1.0, abs(previous_loss))):
            no_gain += 1
        else:
            no_gain = 0
        previous_loss = loss
        if no_gain >= patience:
            best["status"] = "stalled"
            break
        if step == steps:
            break
        def closure():
            optimizer.zero_grad(set_to_none=True)
            value = differentiable_loss(oracle, W)
            value.backward()
            return value
        try:
            optimizer.step(closure)
        except RuntimeError as error:
            best["status"] = "lbfgs_runtime_error"
            best["error"] = str(error)[:240]
            break
        if not bool(torch.isfinite(W).all()):
            best["status"] = "nonfinite_weights"
            break
        if step % problem.config["resources"]["guard_every_steps"] == 0:
            if progress_path is not None:
                atomic_json(
                    progress_path,
                    {
                        "case_id": case_id,
                        "method": "lbfgs",
                        "step": step,
                        "loss": loss,
                        "gradient_fro": grad_fro,
                        "best_loss": best["loss"],
                        "best_step": best["step"],
                        "updated": time.time(),
                    },
                    best_effort=True,
                )
            guard.checkpoint()
    return best, trace


@torch.no_grad()
def properties(problem, W):
    ev = problem.evaluate(W)
    s = torch.linalg.svdvals(W.double())
    B = problem.B.double()
    H0B = problem.Ainv@B@problem.Cinv
    q = ev["q"]
    marginal = q@problem.pi
    result = {
        "reference_loss": float(ev["loss"]),
        "gradient_fro": float(torch.linalg.vector_norm(ev["G"])),
        "weight_fro": float(torch.linalg.vector_norm(W)),
        "weight_operator": float(s[0]) if len(s) else 0.0,
        "weight_nuclear": float(s.sum()),
        "weight_stable_rank": _stable_rank(s),
        "weight_singular_values": [float(x) for x in s.cpu().numpy()],
        "alignment_with_B": _cosine(W, B),
        "alignment_with_H0invB": _cosine(W, H0B),
        "prediction_marginal_tv": float(abs(marginal-problem.p).sum()/2),
        "prediction_entropy": float(problem.pi64@(-(q*q.clamp_min(1e-30).log()).sum(0)).double()),
    }
    result.update(_spectral_summaries(s))
    for name, ids, mass in zip(["head", "middle", "tail"], problem.buckets, problem.bucket_mass):
        result[f"ce_{name}"] = float(problem.pi64[ids]@ev["ce"][ids])/max(mass, 1e-30)
        result[f"mass_{name}"] = mass
    return result


def solve_wstar_for_case(config, case, signature, force=False, guard_enabled=True):
    outdir = minimizer_dir(case["id"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir/"wstar.json"
    expected_solver = config["wstar"].get("solver", "torch_lbfgs_strong_wolfe")
    if path.exists() and not force:
        data = json.loads(path.read_text(encoding="utf-8"))
        skipped = str(data.get("status", "")).startswith("skipped_")
        solver_matches = skipped or data.get("solver") == expected_solver
        if data.get("protocol_signature") == signature and solver_matches:
            return data
    torch.set_num_threads(config["resources"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(config["resources"]["gpu_allocator_fraction"])
    guard = LaptopGuard(HERE, config["resources"], enabled=guard_enabled)
    corpus = Corpus(case.get("corpus", "ptb"), guard)
    target = corpus.target(case["target"], case["target_seeds"][0])
    eligible, reason = should_attempt_reference(target, case, config)
    target_has_zero_support = bool(float(target.P.min()) <= 0.0)
    base = {
        "case_id": case["id"],
        "protocol_signature": signature,
        "target_seed": case["target_seeds"][0],
        "rep_seed": case["rep_seeds"][0],
        "target_min_probability": float(target.P.min()),
        "target_has_zero_support": target_has_zero_support,
        "target_sha256": digest(target.P),
        "pi_sha256": digest(target.pi),
        "wstar_policy": config["wstar"].get("skip_policy"),
        "updated": time.time(),
    }
    if not eligible:
        result = {
            **base,
            "status": reason,
            "reference_loss": None,
            "explanation": (
                "This setting is treated as having no finite softmax minimizer "
                "under the zero-bias protocol, so no W* reference line is drawn."
            ),
        }
        atomic_json(path, result)
        return result
    problem = Problem(target, case, case["rep_seeds"][0], config, device)
    started = time.time()
    candidates = []
    progress_path = outdir/"wstar_progress.json"
    best, trace = refine_lbfgs(
        problem,
        int(config["wstar"].get("reference_steps", 80)),
        float(config["wstar"].get("gradient_norm_goal", 1e-5)),
        guard,
        progress_path=progress_path,
        case_id=case["id"],
    )
    candidates.append({
        "method": "lbfgs",
        "loss": best["loss"],
        "step": best["step"],
        "gradient_fro": best["grad_fro"],
        "status": best["status"],
        "error": best.get("error"),
        "trace": [{"step": int(s), "loss": float(l), "gradient_fro": float(g)} for s, l, g in trace],
        "W": best["W"].to(problem.dtype),
    })
    guard.checkpoint(force=True)
    winner = min(candidates, key=lambda item: item["loss"])
    props = properties(problem, winner["W"])
    candidate_summary = [
        {k: v for k, v in item.items() if k != "W"}
        for item in candidates
    ]
    if props["gradient_fro"] <= float(config["wstar"].get("gradient_norm_goal", 1e-5)):
        status = "solved"
    elif target_has_zero_support:
        status = "attempted_zero_support_approximate"
    else:
        status = "approximate"
    result = {
        **base,
        **props,
        "status": status,
        "solver": "torch_lbfgs_strong_wolfe",
        "reference_method": winner["method"],
        "candidate_summaries": candidate_summary,
        "seconds": time.time()-started,
        "device": device,
    }
    atomic_json(path, result)
    return result


if __name__ == "__main__":
    import argparse
    from run_case_v2 import physics_signature

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", default=str(HERE/"config/study_v2.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    case = next(c for c in config["cases"] if c["id"] == args.case)
    signature = physics_signature(config, case)
    print(json.dumps(solve_wstar_for_case(config, case, signature, args.force, not args.no_guard), indent=2))
