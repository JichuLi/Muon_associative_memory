"""V2 single-setting worker with legacy reuse and two selection criteria."""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch

from core import Problem
from plots_v2 import case_result_dirs, plot_case_v2, score_trace
from resources import BudgetExhausted, LaptopGuard, atomic_json
from storage import Store, canonical_hash, unpack_arrays
from targets import Corpus
from training import train
from wstar import solve_wstar_for_case


HERE = Path(__file__).resolve().parent


def physics_signature(config, case):
    keys = [
        "protocol_version",
        "d",
        "steps",
        "dtype",
        "tf32",
        "polar_rtol",
        "ray_bisections",
        "ray_max_expansions",
        "ray_max_fro",
    ]
    physics = {
        "corpus": case.get("corpus", "ptb"),
        "target": case["target"],
        "bias": case["bias"],
        "geometry": case["geometry"],
    }
    if case["target"]["kind"] == "prototype" and case["target"].get("residual", 0) == 0:
        physics["initial_gradient_accumulation"] = "float64_v1"
    return canonical_hash({"config": {k: config[k] for k in keys}, "case": physics})[:20]


def run_id(signature, target_seed, rep_seed, method, lr):
    rounded = None if lr is None else float(f"{lr:.12g}")
    return canonical_hash([signature, target_seed, rep_seed, method, rounded])[:28]


def canonical_lr(lr):
    return None if lr is None else float(f"{float(lr):.12g}")


def legacy_paths(config):
    return [HERE/path for path in config.get("storage", {}).get("legacy_databases", [])]


def copy_legacy_run(store, config, identity):
    if store.run(identity) is not None:
        return True
    for path in legacy_paths(config):
        if not path.exists():
            continue
        legacy = sqlite3.connect(path)
        row = legacy.execute("SELECT * FROM runs WHERE run_id=?", (identity,)).fetchone()
        if row:
            store.connection.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
            drow = legacy.execute("SELECT * FROM diagnostics WHERE run_id=?", (identity,)).fetchone()
            if drow:
                store.connection.execute("INSERT OR IGNORE INTO diagnostics VALUES (?,?,?,?)", drow)
            store.connection.commit()
            legacy.close()
            return True
        legacy.close()
    return False


def completed_trace(store, case, signature, method, lr):
    rows = store.connection.execute(
        "SELECT metadata,trace FROM runs WHERE case_id=?", (case["id"],)
    ).fetchall()
    out = []
    for raw, blob in rows:
        meta = json.loads(raw)
        if meta.get("protocol_signature") != signature:
            continue
        if meta["method"] != method or meta["lr"] != lr:
            continue
        if meta["target_seed"] not in case["target_seeds"] or meta["rep_seed"] not in case["rep_seeds"]:
            continue
        if meta["status"] != "completed":
            continue
        out.append((meta, unpack_arrays(blob)))
    return out


def required_repeats(case):
    return len(case["target_seeds"])*len(case["rep_seeds"])


def criterion_score(meta, trace, criterion_name, criterion):
    if criterion_name == "auc":
        return float(meta["auc"])
    return score_trace(trace, criterion)


def lattice_filter(config, method, values):
    lower, upper, number = config["lr_log10_grids"][method]
    spacing = (upper-lower)/(number-1)
    filtered = {}
    for lr, score in values.items():
        position = (math.log10(lr)-lower)/spacing
        if abs(position-round(position)) < 1e-8:
            filtered[lr] = score
    return filtered


def best_lr(values):
    return min(values, key=lambda lr: (values[lr], abs(math.log10(lr))))


def lr_scores(store, case, signature, config, method, criterion_name, coarse_only=False):
    criterion = config["selection"]["criteria"][criterion_name]
    rows = store.connection.execute(
        "SELECT metadata,trace FROM runs WHERE case_id=?", (case["id"],)
    ).fetchall()
    by_lr = {}
    for raw, blob in rows:
        meta = json.loads(raw)
        if meta.get("protocol_signature") != signature or meta["method"] != method:
            continue
        if meta["target_seed"] not in case["target_seeds"] or meta["rep_seed"] not in case["rep_seeds"]:
            continue
        if meta["status"] != "completed":
            continue
        by_lr.setdefault(meta["lr"], []).append(criterion_score(meta, unpack_arrays(blob), criterion_name, criterion))
    values = {
        lr: float(np.mean(scores)) if len(scores) == required_repeats(case) else math.inf
        for lr, scores in by_lr.items()
    }
    return lattice_filter(config, method, values) if coarse_only else values


def line_search_score(store, case, signature, config, method, criterion_name):
    rows = completed_trace(store, case, signature, method, None)
    if len(rows) != required_repeats(case):
        return math.inf
    criterion = config["selection"]["criteria"][criterion_name]
    return float(np.mean([criterion_score(meta, trace, criterion_name, criterion) for meta, trace in rows]))


def execute(case, config, guard_enabled=True, skip_wstar=False, force_wstar=False):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    torch.set_num_threads(config["resources"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(config["resources"]["gpu_allocator_fraction"])
    guard = LaptopGuard(HERE, config["resources"], enabled=guard_enabled)
    store = Store(HERE/config["storage"]["database"])
    signature = physics_signature(config, case)
    output_dirs = case_result_dirs(case["id"])
    atomic_json(
        output_dirs["metadata"]/"case_specification.json",
        {"case": case, "protocol_signature": signature, "zero_bias": case["bias"]["kind"] == "zero"},
    )
    store.connection.execute(
        "INSERT OR REPLACE INTO cases(case_id,specification,status,updated) VALUES (?,?,?,?)",
        (case["id"], json.dumps(case), "running", time.time()),
    )
    store.connection.commit()
    corpus = Corpus(case.get("corpus", "ptb"), guard)
    targets = {}
    metadata = {
        "protocol_signature": signature,
        "case": case,
        "torch": torch.__version__,
        "device": device,
        "targets": {},
        "representations": {},
        "started": time.time(),
        "legacy_databases": [str(p) for p in legacy_paths(config)],
    }
    status_path = HERE/"results"/"current_v2.json"

    def target_for(seed):
        if seed not in targets:
            targets[seed] = corpus.target(case["target"], seed)
            stats = targets[seed].statistics()
            metadata["targets"][str(seed)] = stats
            atomic_json(output_dirs["metadata"]/f"target_seed_{seed}.json", stats)
        return targets[seed]

    def compute(requests, deep=False):
        for ts in case["target_seeds"]:
            target = target_for(ts)
            for rs in case["rep_seeds"]:
                needed = []
                seen_identities = set()
                for method, lrs in requests.items():
                    for lr in lrs:
                        lr = canonical_lr(lr)
                        identity = run_id(signature, ts, rs, method, lr)
                        copy_legacy_run(store, config, identity)
                        missing = (
                            not store.has_diagnostics(identity, version=2)
                            if deep else store.run(identity) is None
                        )
                        if missing and identity not in seen_identities:
                            needed.append((method, lr, identity))
                            seen_identities.add(identity)
                if not needed:
                    continue
                problem = Problem(target, case, rs, config, device)
                metadata["representations"][f"{ts}/{rs}"] = {
                    "original_hashes": problem.feature_hashes,
                    "initial_hessian_condition": problem.initial_cond,
                }
                atomic_json(
                    output_dirs["metadata"]/f"representation_target{ts}_rep{rs}.json",
                    metadata["representations"][f"{ts}/{rs}"],
                )
                for method, lr, identity in needed:
                    def beat(step, loss):
                        atomic_json(
                            status_path,
                            {
                                "case_id": case["id"],
                                "family": case["family"],
                                "worker_pid": os.getpid(),
                                "target_seed": ts,
                                "rep_seed": rs,
                                "method": method,
                                "lr": lr,
                                "step": step,
                                "loss": loss,
                                "deep": deep,
                                "updated": time.time(),
                            },
                            best_effort=True,
                        )

                    capture = deep or method.endswith("_ls")
                    result, arrays, diagnostics = train(problem, method, lr, config, guard, capture, beat)
                    result.update(
                        run_id=identity,
                        case_id=case["id"],
                        protocol_signature=signature,
                        target_seed=ts,
                        rep_seed=rs,
                        method=method,
                        lr=lr,
                        steps=config["steps"],
                        d=config["d"],
                    )
                    if deep:
                        original = store.run(identity)
                        if original is None:
                            raise AssertionError("Selected diagnostic run has no underlying sweep result")
                        error = float(np.max(abs(arrays["loss"]-original[1]["loss"])))
                        result["selected_replay_max_abs_error"] = error
                        if error > 2e-4:
                            raise AssertionError(f"Selected replay differs from tuning trajectory by {error}")
                        store.save_diagnostics(identity, case["id"], result, diagnostics)
                    else:
                        store.save_run(identity, case["id"], ts, rs, method, lr, result, arrays)
                        if capture and result["status"] == "completed":
                            store.save_diagnostics(identity, case["id"], result, diagnostics)
                    print(
                        f"{case['id']} t{ts}/r{rs} {method} lr={lr} "
                        f"{'diagnostic' if deep else 'candidate'}: "
                        f"AUC={result['auc']:.8g}, CE={result['final_loss']:.7g}, "
                        f"{result['seconds']:.1f}s",
                        flush=True,
                    )
                    if (HERE/"STOP_AFTER_RUN").exists() or result["status"] == "budget_exhausted":
                        store.close()
                        return False
                del problem
                if device == "cuda":
                    torch.cuda.empty_cache()
                guard.checkpoint(force=True)
        return True

    rays = {m: [None] for m in case["methods"] if m.endswith("_ls")}
    grids = {
        m: [float(v) for v in np.logspace(*config["lr_log10_grids"][m])]
        for m in case["methods"] if m.endswith("_const")
    }
    if not compute({**rays, **grids}):
        return False

    flat = set()
    criteria_names = list(config["selection"]["criteria"])
    for _ in range(config["max_boundary_extensions"]):
        more = {}
        for method in grids:
            criterion_values = {
                name: lr_scores(store, case, signature, config, method, name, coarse_only=True)
                for name in criteria_names
            }
            finite_values = [score for values in criterion_values.values() for score in values.values() if math.isfinite(score)]
            if not finite_values:
                raise RuntimeError(f"No complete finite candidate for {method}")
            if max(finite_values)-min(finite_values) < 1e-12:
                flat.add(method)
                continue
            for values in criterion_values.values():
                winner = best_lr(values)
                if winner == min(values):
                    more.setdefault(method, []).append(min(values)/10**config["boundary_log10_increment"])
                elif winner == max(values):
                    more.setdefault(method, []).append(max(values)*10**config["boundary_log10_increment"])
        more = {m: sorted(set(lrs)) for m, lrs in more.items()}
        if not more:
            break
        if not compute(more):
            return False

    refinement = {}
    for method in grids:
        if method in flat:
            continue
        lrs = []
        for criterion_name in criteria_names:
            values = lr_scores(store, case, signature, config, method, criterion_name, coarse_only=True)
            winner = best_lr(values)
            lrs.extend(winner*10**shift for shift in config["refinement_offsets"])
        refinement[method] = sorted(set(float(x) for x in lrs))
    if not compute(refinement):
        return False

    selections = {}
    selected_requests = {}
    for criterion_name in criteria_names:
        selections[criterion_name] = {}
        for method in case["methods"]:
            if method.endswith("_ls"):
                value = line_search_score(store, case, signature, config, method, criterion_name)
                if criterion_name == "auc":
                    selections[criterion_name][method] = {"lr": None}
                else:
                    selections[criterion_name][method] = {"lr": None, criterion_name: value}
                selected_requests.setdefault(method, set()).add(None)
            else:
                values = lr_scores(store, case, signature, config, method, criterion_name)
                winner = best_lr(values)
                boundary = winner in [min(values), max(values)] and method not in flat
                if boundary:
                    raise RuntimeError(f"{method} remains boundary-optimal under {criterion_name}; grid needs review")
                selections[criterion_name][method] = {
                    "lr": winner,
                    criterion_name: values[winner],
                    "candidates": len(values),
                    "grid_min": min(values),
                    "grid_max": max(values),
                    "refinement_offsets": list(config["refinement_offsets"]),
                    "flat_objective": method in flat,
                    "grid_boundary": boundary,
                }
                selected_requests.setdefault(method, set()).add(winner)
    selection = selections["auc"]
    selection_late = selections.get("late", {})
    plot_selection = {"criteria": selections, "criteria_definition": config["selection"]["criteria"]}
    selected_requests = {method: sorted(lrs, key=lambda x: -math.inf if x is None else x) for method, lrs in selected_requests.items()}
    atomic_json(output_dirs["selection"]/"selection.json", selection)
    atomic_json(output_dirs["selection"]/"selection_late.json", selection_late)
    metadata.update(
        selection=selection,
        selection_late=selection_late,
        resource_sleep_seconds=guard.sleep_seconds,
        maximum_gpu_temperature=guard.peak_temp,
    )
    store.connection.execute(
        "UPDATE cases SET status=?,metadata=?,selection=?,updated=? WHERE case_id=?",
        ("diagnostics", json.dumps(metadata), json.dumps(selection), time.time(), case["id"]),
    )
    store.connection.commit()
    if not compute(selected_requests, deep=True):
        return False

    wstar_result = None
    if config.get("wstar", {}).get("enabled") and not skip_wstar:
        wstar_result = solve_wstar_for_case(config, case, signature, force=force_wstar, guard_enabled=guard_enabled)
        metadata["wstar"] = {k: v for k, v in wstar_result.items() if k != "weight_singular_values"}
    plot_case_v2(store, case, signature, plot_selection, config)
    metadata.update(completed=time.time(), resource_sleep_seconds=guard.sleep_seconds, maximum_gpu_temperature=guard.peak_temp)
    store.connection.execute(
        "UPDATE cases SET status=?,metadata=?,updated=? WHERE case_id=?",
        ("completed", json.dumps(metadata), time.time(), case["id"]),
    )
    store.connection.commit()
    store.close()
    atomic_json(status_path, {"case_id": case["id"], "worker_pid": os.getpid(), "status": "completed", "updated": time.time()})
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", default=str(HERE/"config/study_v2.json"))
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--skip-wstar", action="store_true")
    parser.add_argument("--force-wstar", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    case = next(c for c in config["cases"] if c["id"] == args.case)
    try:
        done = execute(case, config, not args.no_guard, args.skip_wstar, args.force_wstar)
        raise SystemExit(0 if done else 3)
    except Exception as error:
        import traceback

        store = Store(HERE/config["storage"]["database"])
        store.connection.execute(
            "UPDATE cases SET status=?,error=?,updated=? WHERE case_id=?",
            (
                "budget_deferred" if isinstance(error, BudgetExhausted) else "error",
                traceback.format_exc(),
                time.time(),
                case["id"],
            ),
        )
        store.connection.commit()
        store.close()
        if isinstance(error, BudgetExhausted):
            raise SystemExit(3)
        raise


if __name__ == "__main__":
    main()
