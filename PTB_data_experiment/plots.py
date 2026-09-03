"""Consistent scientific figures. Best constant LR is explicitly labelled everywhere."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"gd_ls":"#c65342","muon_ls":"#146c8d","gd_const":"#c98c13",
          "ngd_const":"#398968","muon_const":"#7b5aa6","h0_ls":"#272d35","h0_const":"#8b8c35",
          "signgd_ls":"#b05c7a","signgd_const":"#7f4f24"}
NAMES = {"gd_ls":"GD, line search","muon_ls":"Muon, line search",
         "gd_const":"GD, best constant","ngd_const":"NGD, best constant",
         "muon_const":"Muon, best constant","h0_ls":"Fixed H0 inverse GD, line search",
         "h0_const":"Fixed H0 inverse GD, best constant",
         "signgd_ls":"SignGD, line search","signgd_const":"SignGD, best constant"}


def defaults():
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "font.family":"DejaVu Sans","savefig.facecolor":"white"})


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
    line = ax.plot(x,mean,label=text,color=COLORS[method],lw=1.8,
                   linestyle="--" if method.endswith("_ls") else "-")[0]
    ax.fill_between(x,mean-sd,mean+sd,color=line.get_color(),alpha=.10,linewidth=0)


def plot_case(store,case,signature,selection):
    defaults()
    folder = HERE/"figures"/case["id"]
    folder.mkdir(parents=True,exist_ok=True)
    raw = store.connection.execute(
        "SELECT run_id,metadata,trace FROM runs WHERE case_id=?",(case["id"],)).fetchall()
    if case.get("baseline_case_id"):
        parent=store.connection.execute("SELECT specification,selection,status FROM cases WHERE case_id=?",
                                       (case["baseline_case_id"],)).fetchone()
        if parent and parent[2]=="completed":
            original=json.loads(parent[0])
            case={**case,"methods":original["methods"]+case["methods"]}
            selection={**json.loads(parent[1]),**selection}
            raw+=store.connection.execute("SELECT run_id,metadata,trace FROM runs WHERE case_id=?",
                                          (case["baseline_case_id"],)).fetchall()
    from storage import unpack_arrays
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
            scores[method].setdefault(meta["lr"],[]).append(meta["auc"])
        chosen = selection[method]["lr"]
        if meta["lr"]==chosen and meta["status"]=="completed":
            selected[method].append(unpack_arrays(blob))
            summaries.append(meta)
            d = store.connection.execute("SELECT arrays,metadata FROM diagnostics WHERE run_id=?",(identity,)).fetchone()
            if d:
                values=unpack_arrays(d[0])
                values["_metadata"]=json.loads(d[1])
                detail[method].append(values)
    title = f"{case['id']} | d=256 | 200 updates"
    subtitle = "Constants selected by mean(log CE), steps 0-200; representation seed 0"
    fig,ax = plt.subplots(figsize=(10.5,6.5),layout="constrained")
    for method,curves in selected.items():
        band(ax,curves[0]["step"],[np.maximum(c["loss"],1e-30) for c in curves],method,label(method,selection))
    ax.set(xlabel="Parameter updates",ylabel="Training CE (nats, log scale)",xlim=(0,200),yscale="log")
    ax.set_title(title+"\n"+subtitle,fontsize=12)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5)
    fig.savefig(folder/"convergence.png",dpi=160)
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
        ax.set(xlabel="Updates",ylabel=ylabel,xlim=(0,200),yscale=scale)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+"\nNative LR coefficients use each optimizer's own direction normalization",fontsize=12)
    fig.savefig(folder/"dynamics.png",dpi=145)
    plt.close(fig)

    fig,axes = plt.subplots(1,len(scores),figsize=(4.3*len(scores),4),layout="constrained",squeeze=False)
    for ax,(method,values) in zip(axes.flat,scores.items()):
        pairs = sorted((lr,float(np.mean(v))) for lr,v in values.items())
        finite = [(lr,v) for lr,v in pairs if np.isfinite(v)]
        ax.plot([r[0] for r in finite],[r[1] for r in finite],"o-",color=COLORS[method],ms=4)
        chosen = selection[method]["lr"]
        best = next(value for lr,value in pairs if lr==chosen)
        ax.scatter([chosen],[best],marker="*",s=140,c="black",zorder=5)
        ax.set(xscale="log",xlabel="Constant LR",ylabel="Mean log training CE",
               title=f"{NAMES[method]}\nLR* = {chosen:.6g}")
        ax.grid(alpha=.2)
    fig.suptitle(case["id"]+" | AUC-only coarse search, boundary extensions and refinement")
    fig.savefig(folder/"lr_sweeps.png",dpi=140)
    plt.close(fig)

    fig,axes = plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
    for ax,group in zip(axes,["head","middle","tail"]):
        for method,curves in detail.items():
            if curves:
                band(ax,curves[0]["detail_step"],[c["detail_ce_"+group] for c in curves],
                     method,label(method,selection))
        ax.set(xlabel="Updates",ylabel="Within-bucket CE",title=group,xlim=(0,200))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(title+" | Context buckets defined by the setting's frequency rank")
    fig.savefig(folder/"frequency_dynamics.png",dpi=135)
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for method,curves in detail.items():
        if not curves or "trace_initial_curvature" not in curves[0]:
            continue
        t=curves[0]["trace_step"]
        curvature=[c["trace_curvature"] for c in curves]
        drift=[np.divide(c["trace_curvature"],c["trace_initial_curvature"],
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
        "Gradient nuclear effective rank", "Marginal-gradient norm / full-gradient norm"]):
        ax.set(xlabel="Updates",ylabel=ylabel,xlim=(0,200))
        ax.grid(alpha=.2)
    axes[0,0].set_yscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].axhline(1,color="#888888",lw=.8)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+"\nCurvature drift tests whether fixed initial geometry remains relevant")
    fig.savefig(folder/"geometry_dynamics.png",dpi=140)
    plt.close(fig)

    # Same-state comparisons isolate the update direction from the trajectory.
    sources=[m for m in ["gd_ls","muon_ls","gd_const","muon_const"] if detail.get(m)]
    if sources:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
        candidate_names={"gd_ls":"GD direction","muon_ls":"Muon direction","h0_ls":"H0 inverse direction",
            "input":"Input inverse only","output":"Output inverse only","diagonal":"Diagonal H0 inverse",
            "wrong_basis":"Wrong input eigenbasis","reference_zero":"Zero-bias H0 inverse",
            "reference_unigram":"Unigram-bias H0 inverse"}
        for ax,source in zip(axes.flat,sources):
            curves=detail[source]
            candidates=list(dict.fromkeys(curves[0]["_metadata"]["probe_candidates"]))
            for candidate in candidates:
                values=[]
                for c in curves:
                    mask=np.asarray(c["_metadata"]["probe_candidates"])==candidate
                    steps=c["probe_step"][mask]
                    values.append(c["probe_ray_gain"][mask])
                stack=np.stack(values)
                mean,sd=stack.mean(0),stack.std(0)
                line=ax.plot(steps,mean,"o-",label=candidate_names[candidate],
                             color=COLORS.get(candidate),lw=1.3,ms=4)[0]
                ax.fill_between(steps,np.maximum(0,mean-sd),mean+sd,
                                color=line.get_color(),alpha=.10)
            ax.set(title=label(source,selection),xlabel="State taken after this many updates",
                   ylabel="Actual one-step CE reduction after ray search")
            ax.set_yscale("symlog",linthresh=1e-7)
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(case["id"]+" | Candidate directions evaluated at identical parameter states")
        fig.savefig(folder/"same_state.png",dpi=140)
        plt.close(fig)

    available={m:cs for m,cs in detail.items() if cs and "detail_heldout_ce" in cs[0]}
    if available:
        fig,ax=plt.subplots(figsize=(10,5.5),layout="constrained")
        for method,curves in available.items():
            band(ax,curves[0]["detail_step"],[c["detail_heldout_ce"] for c in curves],
                 method,label(method,selection))
        description=next(iter(available.values()))[0]["_metadata"].get("heldout_description","held-out counts")
        ax.set(xlabel="Updates",ylabel="Held-out CE (not used for LR selection)",xlim=(0,200),
               title=title+"\n"+str(description))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(folder/"heldout.png",dpi=140)
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,step in zip(axes.flat,[0,20,100,200]):
        for method,curves in detail.items():
            if not curves:
                continue
            values=[]
            for c in curves:
                ids=np.where(c["spectrum_steps"]==step)[0]
                if len(ids):
                    s=c["gradient_singular_values"][ids[0]].astype(float)
                    values.append(s/max(s[0],1e-30))
            if values:
                band(ax,np.arange(1,len(values[0])+1),values,method,label(method,selection))
        ax.set(xlabel="Singular-value index",ylabel="Singular value / largest",title=f"Update {step}",
               xscale="log",yscale="log",ylim=(1e-9,1.2))
        ax.axhline(1e-7,color="#888888",ls=":",lw=.8)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(case["id"]+" | Gradient singular-value shape; dotted line is the polar cutoff")
    fig.savefig(folder/"gradient_spectra.png",dpi=140)
    plt.close(fig)

    # Keep summaries compact and interoperable; underlying full traces remain in SQLite.
    import csv
    with (folder/"selected.csv").open("w",newline="",encoding="utf-8") as f:
        fields = ["method","target_seed","rep_seed","lr","auc","final_loss","seconds"]
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
