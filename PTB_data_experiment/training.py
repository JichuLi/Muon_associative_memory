"""One trajectory. All ordinary candidates run to the same 200-update horizon."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from resources import BudgetExhausted


@torch.no_grad()
def train(problem,method,lr,config,guard,deep=False,heartbeat=None):
    W = problem.reset()
    rows,details,spectra,probes = [],[],[],[]
    started = time.monotonic()
    status = "completed"
    diagnostic_steps = set(config["diagnostic_steps"])
    same_state_steps = set(config["same_state_steps"])
    for step in range(config["steps"]+1):
        ev = problem.evaluate(W)
        loss = ev["loss"]
        row = dict(step=step,loss=loss,grad_fro=float(torch.linalg.vector_norm(ev["G"])),
                   eta=math.nan,eta_fro=math.nan,curvature=math.nan,slope=math.nan,
                   initial_curvature=math.nan,
                   line_evals=0.,bracketed=math.nan,ray_loss=math.nan,bias_gain=0.)
        if not math.isfinite(loss) or not bool(torch.isfinite(ev["G"]).all()):
            status = "nonfinite"
            rows.append(row)
            break
        if loss<0:
            status = "invalid_ce"
            rows.append(row)
            break
        if deep and (step in diagnostic_steps or step%config["detailed_every"]==0):
            info,s,local = problem.diagnostics(W,ev,same_state=step in same_state_steps)
            details.append(dict(step=step,**info))
            if step in diagnostic_steps:
                spectra.append((step,s))
            probes.extend(dict(step=step,**probe) for probe in local)
        if step<config["steps"]:
            if problem.learnable_bias:
                problem.bias -= problem.case["bias"]["lr"]*ev["gb"]
                ev = problem.evaluate(W)
                row["bias_gain"] = loss-ev["loss"]
                row["grad_fro"] = float(torch.linalg.vector_norm(ev["G"]))
            D,s = problem.direction(ev["G"],method)
            if method.endswith("_ls"):
                W,info = problem.ray(W,ev,D)
                for key in ["eta","eta_fro","curvature","initial_curvature","slope","line_evals","bracketed","ray_loss"]:
                    row[key] = float(info[key])
                row["ray_terminal_derivative"] = info["terminal_derivative"]
                row["ray_relative_bracket_width"] = info["relative_bracket_width"]
            else:
                norm = float(torch.linalg.vector_norm(D))
                row["eta"],row["eta_fro"] = lr,lr*norm
                if deep and (step in diagnostic_steps or step%config["detailed_every"]==0):
                    geom,_ = problem.direction_geometry(ev,D)
                    row["curvature"],row["slope"] = geom["curvature"],geom["slope"]
                    row["initial_curvature"] = geom["initial_curvature"]
                W += lr*D
        rows.append(row)
        if step%config["resources"]["guard_every_steps"]==0:
            if heartbeat:
                heartbeat(step,loss)
            try:
                guard.checkpoint()
            except BudgetExhausted:
                status="budget_exhausted"
                break
    keys = sorted(set().union(*(r.keys() for r in rows)))
    arrays = {k:np.asarray([r.get(k,math.nan) for r in rows],dtype=np.float64) for k in keys}
    complete = status=="completed" and len(rows)==config["steps"]+1
    metadata = dict(status=status if complete else status if status!="completed" else "incomplete",
                    auc=float(np.log(np.maximum(arrays["loss"],1e-30)).mean()) if complete else math.inf,
                    auc_numerical_floor=1e-30,
                    final_loss=float(arrays["loss"][-1]),records=len(rows),
                    seconds=time.monotonic()-started,deep=deep,
                    max_loss_increase=float(np.diff(arrays["loss"]).max()) if len(rows)>1 else 0.)
    if method.endswith("_ls"):
        mask = arrays["step"]<config["steps"]
        metadata["unbracketed_updates"] = int(np.sum(arrays["bracketed"][mask]==0))
        metadata["ray_replay_max_abs_error"] = float(np.nanmax(abs(arrays["ray_loss"][:-1]-arrays["loss"][1:])))
    deep_arrays = {}
    if deep:
        metadata["diagnostics_version"]=2
        metadata["heldout_description"]=problem.heldout[3] if problem.heldout is not None else None
        for key in sorted(set().union(*(r.keys() for r in details))):
            deep_arrays["detail_"+key] = np.asarray([r.get(key,math.nan) for r in details],dtype=np.float64)
        if spectra:
            deep_arrays["spectrum_steps"] = np.array([x[0] for x in spectra])
            deep_arrays["gradient_singular_values"] = np.stack([x[1] for x in spectra])
        # Candidate labels remain metadata; numeric arrays stay compressed.
        deep_arrays["probe_step"] = np.array([r["step"] for r in probes],dtype=np.int32)
        for key in ["eta","eta_fro","ray_gain","curvature","initial_curvature","slope","terminal_derivative","bracketed"]:
            deep_arrays["probe_"+key] = np.array([r[key] for r in probes],dtype=np.float64)
        metadata["probe_candidates"] = [r["candidate"] for r in probes]
        deep_arrays.update({"trace_"+k:v for k,v in arrays.items()})
    del W
    return metadata,arrays,deep_arrays
