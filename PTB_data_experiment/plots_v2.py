"""V2 figures: legacy plotting style with per-setting folders and late selection."""
from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots import COLORS,NAMES,defaults
from storage import unpack_arrays

HERE = Path(__file__).resolve().parent
REFERENCE_STATUSES = {"solved","approximate","attempted_zero_support_approximate"}


def case_figure_dirs(case_id):
    root = HERE/"figures"/"v2"/case_id
    dirs = dict(root=root,loss=root/"loss",dynamics=root/"dynamics",
                selection=root/"selection",minimizer=root/"minimizer")
    for path in dirs.values():
        path.mkdir(parents=True,exist_ok=True)
    return dirs


def case_result_dirs(case_id):
    root = HERE/"results"/"v2"/case_id
    dirs = dict(root=root,selection=root/"selection",minimizer=root/"minimizer",
                metadata=root/"metadata")
    for path in dirs.values():
        path.mkdir(parents=True,exist_ok=True)
    return dirs


def label(method,selection):
    lr = selection[method]["lr"]
    return NAMES[method]+(f" (LR*={lr:.6g})" if lr is not None else "")


def band(ax,x,values,method,text):
    values = np.stack(values)
    masked = np.ma.masked_invalid(values)
    mean = masked.mean(axis=0).filled(np.nan)
    sd = masked.std(axis=0).filled(np.nan)
    valid = np.isfinite(mean)
    x,mean,sd = np.asarray(x)[valid],mean[valid],sd[valid]
    line = ax.plot(x,mean,label=text,color=COLORS.get(method),lw=1.8,
                   linestyle="--" if method.endswith("_ls") else "-")[0]
    ax.fill_between(x,np.maximum(mean-sd,1e-30),mean+sd,
                    color=line.get_color(),alpha=.10,linewidth=0)


def score_trace(trace,criterion):
    steps = trace["step"].astype(int)
    loss = np.maximum(trace["loss"].astype(float),1e-30)
    mask = (steps>=int(criterion["start"])) & (steps<=int(criterion["end"]))
    if not np.any(mask):
        return math.inf
    if criterion["kind"]=="mean_log_ce":
        return float(np.log(loss[mask]).mean())
    if criterion["kind"]=="mean_ce":
        return float(loss[mask].mean())
    raise ValueError(f"Unknown criterion: {criterion}")


def _same_lr(left,right):
    if left is None or right is None:
        return left is None and right is None
    left,right = float(left),float(right)
    return abs(left-right)<=max(1e-12,1e-10*max(abs(left),abs(right),1.0))


def _wstar(case_id):
    path = case_result_dirs(case_id)["minimizer"]/"wstar.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_selections(selection):
    if "criteria" in selection:
        return selection["criteria"]
    return {"auc":selection}


def _scalar_float(mapping,key):
    try:
        value = float(mapping.get(key))
    except (TypeError,ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value,digits=4):
    try:
        value = float(value)
    except (TypeError,ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def _spectral_scalars(singular_values):
    s = np.asarray(singular_values,dtype=float)
    s = s[np.isfinite(s) & (s>0)]
    if not s.size:
        return {}
    op = float(s[0])
    fro2 = float(np.sum(s*s))
    nuclear = float(np.sum(s))
    result = {
        "weight_nuclear_effective_rank": nuclear*nuclear/max(fro2,1e-30),
    }
    rel = s/max(op,1e-30)
    for name,threshold in [("1e-2",1e-2),("1e-3",1e-3),("1e-4",1e-4),("1e-6",1e-6)]:
        result[f"weight_rank_rel_ge_{name}"] = int(np.sum(rel>=threshold))
    return result


def _criterion_title(name,criterion):
    if name=="auc":
        return "Constants selected by mean(log CE), steps 0-200; representation seed 0"
    return f"Constants selected by mean CE, steps {criterion['start']}-{criterion['end']}; representation seed 0"


def _criterion_ylabel(name,criterion):
    if name=="auc":
        return "Mean log training CE"
    if criterion["kind"]=="mean_ce":
        return f"Mean training CE, steps {criterion['start']}-{criterion['end']}"
    return criterion["kind"]


def _load_selected(store,case,signature,selection):
    raw = store.connection.execute(
        "SELECT run_id,metadata,trace FROM runs WHERE case_id=?",(case["id"],)).fetchall()
    selected = {m:[] for m in case["methods"]}
    detail = {m:[] for m in case["methods"]}
    scores = {m:{} for m in case["methods"] if m.endswith("_const")}
    summaries = []
    for identity,metadata,blob in raw:
        meta = json.loads(metadata)
        if meta.get("protocol_signature")!=signature:
            continue
        if meta["rep_seed"] not in case["rep_seeds"] or meta["target_seed"] not in case["target_seeds"]:
            continue
        method = meta["method"]
        if method.endswith("_const"):
            scores[method].setdefault(meta["lr"],[]).append((meta,unpack_arrays(blob)))
        if method not in selection:
            continue
        if _same_lr(meta["lr"],selection[method]["lr"]) and meta["status"]=="completed":
            selected[method].append(unpack_arrays(blob))
            summaries.append(meta)
            d = store.connection.execute("SELECT arrays,metadata FROM diagnostics WHERE run_id=?",(identity,)).fetchone()
            if d:
                values = unpack_arrays(d[0])
                values["_metadata"] = json.loads(d[1])
                detail[method].append(values)
    return selected,detail,scores,summaries


def _score_candidate(meta,trace,name,criterion):
    if name=="auc":
        return float(meta["auc"])
    return score_trace(trace,criterion)


def _plot_one_selection(store,case,signature,selection,criterion_name,criterion,config):
    figdirs = case_figure_dirs(case["id"])
    outdirs = case_result_dirs(case["id"])
    selected,detail,scores,summaries = _load_selected(store,case,signature,selection)
    title = f"{case['id']} | d=256 | 200 updates"
    subtitle = _criterion_title(criterion_name,criterion)
    suffix = f"{criterion_name}_selected"
    wstar = _wstar(case["id"])

    fig,ax = plt.subplots(figsize=(10.5,6.5),layout="constrained")
    for method,curves in selected.items():
        if curves:
            band(ax,curves[0]["step"],[np.maximum(c["loss"],1e-30) for c in curves],
                 method,label(method,selection))
    if wstar and wstar.get("status") in REFERENCE_STATUSES and wstar.get("reference_loss") is not None:
        loss = float(wstar["reference_loss"])
        status = wstar.get("status")
        text = f"L(W*)={loss:.6g}" if status == "solved" else f"L(W*) ref ({status})={loss:.6g}"
        ax.axhline(loss,color="black",lw=1.2,ls=":",label=text)
    elif wstar and str(wstar.get("status","")).startswith("skipped"):
        ax.text(.98,.04,"No finite W* reference",transform=ax.transAxes,
                ha="right",va="bottom",fontsize=9,color="#555555")
    ax.set(xlabel="Parameter updates",ylabel="Training CE (nats, log scale)",
           xlim=(0,config["steps"]),yscale="log")
    ax.set_title(title+"\n"+subtitle,fontsize=12)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5)
    fig.savefig(figdirs["loss"]/f"loss_vs_step_{suffix}.png",dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(15,8.5),layout="constrained")
    settings = [
        ("eta","Native update coefficient / LR","log",False),
        ("eta_fro","Frobenius update length","log",False),
        ("grad_fro","Gradient Frobenius norm","log",False),
        ("prediction_entropy","Prediction conditional entropy","linear",True),
        ("prediction_marginal_tv","Prediction vs target marginal TV","log",True),
        ("grad_top32_energy","Gradient energy in first 32 modes","linear",True)]
    for ax,(key,ylabel,scale,deep) in zip(axes.flat,settings):
        source = detail if deep else selected
        for method,curves in source.items():
            if not curves:
                continue
            x = curves[0]["detail_step"] if deep else curves[0]["step"]
            name = "detail_"+key if deep else key
            band(ax,x,[c[name] for c in curves],method,label(method,selection))
        ax.set(xlabel="Updates",ylabel=ylabel,xlim=(0,config["steps"]),yscale=scale)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+"\nNative LR coefficients use each optimizer's own direction normalization",fontsize=12)
    fig.savefig(figdirs["dynamics"]/f"optimizer_dynamics_{suffix}.png",dpi=145)
    plt.close(fig)

    fig,axes = plt.subplots(1,len(scores),figsize=(4.3*len(scores),4),layout="constrained",squeeze=False)
    for ax,(method,values) in zip(axes.flat,scores.items()):
        pairs = sorted((lr,float(np.mean([_score_candidate(m,t,criterion_name,criterion) for m,t in group])))
                       for lr,group in values.items())
        finite = [(lr,v) for lr,v in pairs if np.isfinite(v)]
        ax.plot([r[0] for r in finite],[r[1] for r in finite],"o-",color=COLORS.get(method),ms=4)
        chosen = selection[method]["lr"]
        if chosen is not None:
            best = next((value for lr,value in pairs if _same_lr(lr,chosen)),math.nan)
            if np.isfinite(best):
                ax.scatter([chosen],[best],marker="*",s=140,c="black",zorder=5)
        ax.set(xscale="log",xlabel="Constant LR",ylabel=_criterion_ylabel(criterion_name,criterion),
               title=f"{NAMES[method]}\nLR* = {chosen:.6g}")
        ax.grid(alpha=.2)
    fig.suptitle(case["id"]+f" | {criterion_name} LR search, boundary extensions and refinement")
    fig.savefig(figdirs["selection"]/f"lr_sweeps_{criterion_name}.png",dpi=140)
    plt.close(fig)

    fig,axes = plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
    for ax,group in zip(axes,["head","middle","tail"]):
        for method,curves in detail.items():
            if curves:
                band(ax,curves[0]["detail_step"],[c["detail_ce_"+group] for c in curves],
                     method,label(method,selection))
        ax.set(xlabel="Updates",ylabel="Within-bucket CE",title=group,xlim=(0,config["steps"]))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(title+" | Context buckets defined by the setting's frequency rank")
    fig.savefig(figdirs["dynamics"]/f"frequency_ce_{suffix}.png",dpi=135)
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for method,curves in detail.items():
        if not curves or "trace_initial_curvature" not in curves[0]:
            continue
        t = curves[0]["trace_step"]
        curvature = [c["trace_curvature"] for c in curves]
        drift = [np.divide(c["trace_curvature"],c["trace_initial_curvature"],
                           out=np.full_like(c["trace_curvature"],np.nan),
                           where=c["trace_initial_curvature"]>0) for c in curves]
        band(axes[0,0],t,curvature,method,label(method,selection))
        band(axes[0,1],t,drift,method,label(method,selection))
        band(axes[1,0],curves[0]["detail_step"],
             [c.get("detail_grad_nuclear_rank",np.zeros_like(c["detail_step"])) for c in curves],
             method,label(method,selection))
        band(axes[1,1],curves[0]["detail_step"],
             [c["detail_marginal_gradient_norm_ratio"] for c in curves],method,label(method,selection))
    for ax,ylabel in zip(axes.flat,["Curvature along a unit Frobenius direction",
        "Current / initial curvature in the same direction",
        "Gradient nuclear effective rank","Marginal-gradient norm / full-gradient norm"]):
        ax.set(xlabel="Updates",ylabel=ylabel,xlim=(0,config["steps"]))
        ax.grid(alpha=.2)
    axes[0,0].set_yscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].axhline(1,color="#888888",lw=.8)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+"\nCurvature drift tests whether fixed initial geometry remains relevant")
    fig.savefig(figdirs["dynamics"]/f"geometry_{suffix}.png",dpi=140)
    plt.close(fig)

    available = {m:cs for m,cs in detail.items() if cs and "detail_heldout_ce" in cs[0]}
    if available:
        fig,ax = plt.subplots(figsize=(10,5.5),layout="constrained")
        for method,curves in available.items():
            band(ax,curves[0]["detail_step"],[c["detail_heldout_ce"] for c in curves],
                 method,label(method,selection))
        description = next(iter(available.values()))[0]["_metadata"].get("heldout_description","held-out counts")
        ax.set(xlabel="Updates",ylabel="Held-out CE (not used for LR selection)",xlim=(0,config["steps"]),
               title=title+"\n"+str(description))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(figdirs["dynamics"]/f"heldout_{suffix}.png",dpi=140)
        plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(14.5,8),layout="constrained")
    steps = list(config.get("diagnostic_steps",[0,20,100,120,150,200]))[:6]
    for ax,step in zip(axes.flat,steps):
        for method,curves in detail.items():
            if not curves:
                continue
            values = []
            for c in curves:
                ids = np.where(c["spectrum_steps"]==step)[0]
                if len(ids):
                    s = c["gradient_singular_values"][ids[0]].astype(float)
                    values.append(s/max(s[0],1e-30))
            if values:
                band(ax,np.arange(1,len(values[0])+1),values,method,label(method,selection))
        ax.set(xlabel="Singular-value index",ylabel="Singular value / largest",title=f"Update {step}",
               xscale="log",yscale="log",ylim=(1e-9,1.2))
        ax.axhline(1e-7,color="#888888",ls=":",lw=.8)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(steps):]:
        ax.axis("off")
    axes[0,0].legend(fontsize=7)
    fig.suptitle(case["id"]+" | Gradient singular-value shape; dotted line is the polar cutoff")
    fig.savefig(figdirs["dynamics"]/f"gradient_spectra_{suffix}.png",dpi=140)
    plt.close(fig)

    csv_name = "selected.csv" if criterion_name=="auc" else f"selected_{criterion_name}.csv"
    with (outdirs["selection"]/csv_name).open("w",newline="",encoding="utf-8") as f:
        fields = ["method","target_seed","rep_seed","lr","auc","final_loss","seconds"]
        if criterion_name!="auc":
            fields.append(criterion_name)
            for row in summaries:
                run = next((curves[0] for method,curves in selected.items()
                            if curves and method==row["method"] and _same_lr(row["lr"],selection[method]["lr"])),None)
                if run is not None:
                    row[criterion_name] = score_trace(run,criterion)
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def _write_wstar_properties(case,wstar):
    outdir = case_result_dirs(case["id"])["minimizer"]
    scalar = {}
    if wstar:
        for key,value in wstar.items():
            if isinstance(value,(str,int,float,bool)) or value is None:
                scalar[key] = value
        scalar.update(_spectral_scalars(wstar.get("weight_singular_values",[])))
    if not scalar:
        scalar = {"case_id":case["id"],"status":"missing","reference_loss":None}
    with (outdir/"wstar_properties.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(scalar))
        writer.writeheader()
        writer.writerow(scalar)


def _annotate_bars(ax,bars,values,digits=3,log=False):
    for bar,value in zip(bars,values):
        if value is None or not math.isfinite(float(value)):
            continue
        height = float(value)
        y = height*1.08 if log else height
        ax.text(bar.get_x()+bar.get_width()/2,y,_fmt(height,digits),
                ha="center",va="bottom",fontsize=8)


def _plot_wstar_scalar_properties(case,wstar,figdir,singular_values):
    derived = _spectral_scalars(singular_values)
    fig,axes = plt.subplots(2,2,figsize=(12.5,8.2),layout="constrained")

    ax = axes[0,0]
    ax.axis("off")
    lines = [
        f"status: {wstar.get('status')}",
        f"reference method: {wstar.get('reference_method',wstar.get('solver','NA'))}",
        f"L(W*) ref: {_fmt(wstar.get('reference_loss'),6)}",
        f"||grad L(W*)||_F: {_fmt(wstar.get('gradient_fro'),4)}",
        f"target min probability: {_fmt(wstar.get('target_min_probability'),4)}",
        f"target has zero support: {wstar.get('target_has_zero_support')}",
    ]
    ax.text(.02,.98,"\n".join(lines),transform=ax.transAxes,va="top",
            family="monospace",fontsize=10)
    ax.set_title("Reference quality",loc="left")

    norm_items = [
        ("Frobenius","weight_fro"),
        ("Operator","weight_operator"),
        ("Nuclear","weight_nuclear"),
    ]
    norm_labels = [name for name,key in norm_items if _scalar_float(wstar,key) is not None]
    norm_values = [_scalar_float(wstar,key) for name,key in norm_items if _scalar_float(wstar,key) is not None]
    ax = axes[0,1]
    if norm_values:
        bars = ax.bar(norm_labels,norm_values,color=["#146c8d","#398968","#c98c13"][:len(norm_values)])
        ax.set(yscale="log",ylabel="Value",title="W* matrix norms")
        ax.grid(axis="y",alpha=.2)
        _annotate_bars(ax,bars,norm_values,3,log=True)
    else:
        ax.axis("off")

    rank_items = [
        ("stable",_scalar_float(wstar,"weight_stable_rank")),
        ("nuclear eff",derived.get("weight_nuclear_effective_rank")),
        ("rel >=1e-2",derived.get("weight_rank_rel_ge_1e-2")),
        ("rel >=1e-3",derived.get("weight_rank_rel_ge_1e-3")),
        ("rel >=1e-4",derived.get("weight_rank_rel_ge_1e-4")),
    ]
    rank_labels = [name for name,value in rank_items if value is not None and float(value)>0]
    rank_values = [float(value) for name,value in rank_items if value is not None and float(value)>0]
    ax = axes[1,0]
    if rank_values:
        bars = ax.bar(rank_labels,rank_values,color="#7b5ab6")
        ax.set(yscale="log",ylabel="Rank / effective rank",title="Spectral rank summaries")
        ax.grid(axis="y",alpha=.2)
        ax.tick_params(axis="x",rotation=20)
        _annotate_bars(ax,bars,rank_values,3,log=True)
    else:
        ax.axis("off")

    ax = axes[1,1]
    align_items = [
        ("cos(W*, B)","alignment_with_B"),
        ("cos(W*, H0invB)","alignment_with_H0invB"),
    ]
    labels = [name for name,key in align_items if _scalar_float(wstar,key) is not None]
    values = [_scalar_float(wstar,key) for name,key in align_items if _scalar_float(wstar,key) is not None]
    if values:
        bars = ax.bar(labels,values,color=["#a5516f","#495a68"][:len(values)])
        ax.axhline(0,color="#555555",lw=.8)
        ax.set(ylim=(-1,1),ylabel="Cosine",title="Alignment")
        ax.grid(axis="y",alpha=.2)
        _annotate_bars(ax,bars,values,3)
    else:
        ax.axis("off")
    text = (
        f"prediction entropy: {_fmt(wstar.get('prediction_entropy'),4)}\n"
        f"prediction marginal TV: {_fmt(wstar.get('prediction_marginal_tv'),4)}\n"
        f"bucket CE head/middle/tail: "
        f"{_fmt(wstar.get('ce_head'),4)} / {_fmt(wstar.get('ce_middle'),4)} / {_fmt(wstar.get('ce_tail'),4)}"
    )
    ax.text(.02,.04,text,transform=ax.transAxes,va="bottom",
            family="monospace",fontsize=9,
            bbox=dict(boxstyle="round,pad=.35",fc="white",ec="#dddddd",alpha=.92))

    fig.suptitle(f"{case['id']} | W* scalar properties")
    fig.savefig(figdir/"wstar_scalar_properties.png",dpi=150)
    plt.close(fig)


def _plot_wstar(case,wstar):
    figdir = case_figure_dirs(case["id"])["minimizer"]
    _write_wstar_properties(case,wstar)
    if not wstar or wstar.get("status") not in REFERENCE_STATUSES:
        return
    singular_values = np.asarray(wstar.get("weight_singular_values",[]),dtype=float)
    _plot_wstar_scalar_properties(case,wstar,figdir,singular_values)
    if singular_values.size and np.isfinite(singular_values).any():
        fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
        idx = np.arange(1,len(singular_values)+1)
        positive = singular_values>0
        axes[0].plot(idx[positive],singular_values[positive],color="#272d35")
        axes[0].set(xscale="log",yscale="log",xlabel="Singular-value index",ylabel="Singular value")
        axes[0].grid(alpha=.2)
        axes[1].plot(idx[positive],singular_values[positive]/max(singular_values[0],1e-30),color="#146c8d")
        axes[1].set(xscale="log",yscale="log",xlabel="Singular-value index",ylabel="Singular value / largest")
        axes[1].grid(alpha=.2)
        stable_rank = _scalar_float(wstar,"weight_stable_rank")
        if stable_rank is not None and stable_rank>0:
            for ax in axes:
                ax.axvline(stable_rank,color="#c44e52",ls=":",lw=1.1)
            axes[1].text(.03,.08,
                         f"stable rank={_fmt(stable_rank,4)}\n"
                         f"||W||F={_fmt(wstar.get('weight_fro'),4)}\n"
                         f"||W||op={_fmt(wstar.get('weight_operator'),4)}",
                         transform=axes[1].transAxes,va="bottom",
                         family="monospace",fontsize=9,
                         bbox=dict(boxstyle="round,pad=.35",fc="white",ec="#dddddd",alpha=.92))
        fig.suptitle(f"{case['id']} | W* singular values; stable rank=||W||_F^2/||W||_op^2")
        fig.savefig(figdir/"wstar_singular_values.png",dpi=150)
        plt.close(fig)
    if all(k in wstar for k in ["ce_head","ce_middle","ce_tail"]):
        fig,ax = plt.subplots(figsize=(6.2,4.2),layout="constrained")
        ax.bar(["head","middle","tail"],[wstar["ce_head"],wstar["ce_middle"],wstar["ce_tail"]],
               color=["#146c8d","#398968","#c98c13"])
        ax.set(ylabel="Within-bucket CE at W*",title=f"{case['id']} | W* bucket losses")
        ax.grid(axis="y",alpha=.2)
        fig.savefig(figdir/"wstar_bucket_ce.png",dpi=150)
        plt.close(fig)
    candidates = wstar.get("candidate_summaries",[])
    if candidates:
        fig,axes = plt.subplots(1,2,figsize=(11.5,4.3),layout="constrained")
        for candidate in candidates:
            trace = candidate.get("trace",[])
            if not trace:
                continue
            method = candidate.get("method","unknown")
            color = COLORS.get(method+"_ls",COLORS.get(method+"_const"))
            axes[0].plot([r["step"] for r in trace],[r["loss"] for r in trace],label=method,color=color)
            axes[1].plot([r["step"] for r in trace],[r["gradient_fro"] for r in trace],label=method,color=color)
        axes[0].set(xlabel="Reference descent step",ylabel="CE loss")
        axes[1].set(xlabel="Reference descent step",ylabel="Gradient Frobenius norm",yscale="log")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{case['id']} | W* reference solver traces")
        fig.savefig(figdir/"wstar_solver_traces.png",dpi=150)
        plt.close(fig)


def plot_case_v2(store,case,signature,selection,config):
    defaults()
    selections = _normalise_selections(selection)
    for criterion_name,chosen in selections.items():
        if criterion_name not in config["selection"]["criteria"]:
            continue
        _plot_one_selection(store,case,signature,chosen,criterion_name,
                            config["selection"]["criteria"][criterion_name],config)
    _plot_wstar(case,_wstar(case["id"]))
