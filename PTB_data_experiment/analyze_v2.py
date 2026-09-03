"""Consolidate completed v2 settings into CSV and Markdown summaries."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from plots_v2 import score_trace
from storage import Store, unpack_arrays


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE/"results"/"analysis_v2"

MUON_METHODS = {"muon_const", "muon_ls"}
H0_METHODS = {"h0_const", "h0_ls"}
SIGN_METHODS = {"signgd_const", "signgd_ls"}
SCALAR_NO_H0_METHODS = {"gd_const", "gd_ls", "ngd_const", "signgd_const", "signgd_ls"}


def same_lr(left, right):
    if left is None or right is None:
        return left is None and right is None
    left = float(left)
    right = float(right)
    return abs(left-right) <= max(1e-12, 1e-10*max(abs(left), abs(right), 1.0))


def selections(metadata, raw_selection):
    return {
        "auc": json.loads(raw_selection),
        "late": metadata.get("selection_late", {}),
    }


def wstar(case_id):
    path = HERE/"results"/"v2"/case_id/"minimizer"/"wstar.json"
    if not path.exists():
        return {"status": "missing", "reference_loss": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": data.get("status"),
        "reference_loss": data.get("reference_loss"),
        "gradient_fro": data.get("gradient_fro"),
        "reference_method": data.get("reference_method", data.get("winning_method")),
        "seconds": data.get("seconds"),
        "target_has_zero_support": data.get("target_has_zero_support"),
        "weight_fro": data.get("weight_fro"),
        "weight_stable_rank": data.get("weight_stable_rank"),
        "alignment_with_B": data.get("alignment_with_B"),
        "alignment_with_H0invB": data.get("alignment_with_H0invB"),
        "prediction_marginal_tv": data.get("prediction_marginal_tv"),
        "prediction_entropy": data.get("prediction_entropy"),
    }


def selected_method_rows(store, case, signature, selection, config):
    rows = []
    raw_runs = store.connection.execute(
        "SELECT run_id,metadata,trace FROM runs WHERE case_id=?", (case["id"],)
    ).fetchall()
    for criterion_name, criterion in config["selection"]["criteria"].items():
        chosen = selection.get(criterion_name, {})
        for method, picked in chosen.items():
            matches = []
            for run_id, raw, blob in raw_runs:
                meta = json.loads(raw)
                if meta.get("protocol_signature") != signature:
                    continue
                if meta.get("status") != "completed":
                    continue
                if meta.get("method") != method or not same_lr(meta.get("lr"), picked.get("lr")):
                    continue
                if meta.get("target_seed") not in case["target_seeds"] or meta.get("rep_seed") not in case["rep_seeds"]:
                    continue
                trace = unpack_arrays(blob)
                losses = trace["loss"].astype(float)
                matches.append({
                    "run_id": run_id,
                    "score": float(meta["auc"]) if criterion_name == "auc" else score_trace(trace, criterion),
                    "auc_score": float(meta["auc"]),
                    "late_score": score_trace(trace, config["selection"]["criteria"]["late"]),
                    "final_loss": float(losses[-1]),
                    "lr": meta.get("lr"),
                })
            if matches:
                rows.append({
                    "case_id": case["id"],
                    "family": case["family"],
                    "criterion": criterion_name,
                    "method": method,
                    "lr": picked.get("lr"),
                    "score": float(np.mean([m["score"] for m in matches])),
                    "auc_score": float(np.mean([m["auc_score"] for m in matches])),
                    "late_score": float(np.mean([m["late_score"] for m in matches])),
                    "final_loss": float(np.mean([m["final_loss"] for m in matches])),
                    "runs": len(matches),
                })
    return rows


def best(rows, methods=None):
    subset = [row for row in rows if methods is None or row["method"] in methods]
    if not subset:
        return None
    return min(subset, key=lambda row: row["score"])


def summary_rows(config, store):
    method_rows = []
    case_rows = []
    for case in config["cases"]:
        status_row = store.connection.execute(
            "SELECT status,metadata,selection,error FROM cases WHERE case_id=?", (case["id"],)
        ).fetchone()
        status = status_row[0] if status_row else "pending"
        error = status_row[3] if status_row and status_row[3] else ""
        signature = None
        selection = None
        metadata = {}
        if status_row and status_row[1]:
            metadata = json.loads(status_row[1])
            signature = metadata.get("protocol_signature")
        if status_row and status_row[2]:
            selection = selections(metadata, status_row[2])
        w = wstar(case["id"])
        if status == "completed" and signature and selection:
            rows = selected_method_rows(store, case, signature, selection, config)
            for row in rows:
                row.update({
                    "wstar_status": w["status"],
                    "wstar_loss": w["reference_loss"],
                    "final_minus_wstar": (
                        row["final_loss"]-float(w["reference_loss"])
                        if w.get("reference_loss") is not None and math.isfinite(float(w["reference_loss"]))
                        else None
                    ),
                })
            method_rows.extend(rows)
            for criterion_name in config["selection"]["criteria"]:
                local = [row for row in rows if row["criterion"] == criterion_name]
                best_any = best(local)
                best_muon = best(local, MUON_METHODS)
                best_nonmuon = best([row for row in local if row["method"] not in MUON_METHODS])
                best_h0 = best(local, H0_METHODS)
                best_sign = best(local, SIGN_METHODS)
                best_scalar = best(local, SCALAR_NO_H0_METHODS)
                record = {
                    "case_id": case["id"],
                    "family": case["family"],
                    "status": status,
                    "criterion": criterion_name,
                    "best_method": best_any["method"] if best_any else None,
                    "best_score": best_any["score"] if best_any else None,
                    "best_final_loss": best_any["final_loss"] if best_any else None,
                    "best_muon_method": best_muon["method"] if best_muon else None,
                    "best_muon_score": best_muon["score"] if best_muon else None,
                    "best_muon_final_loss": best_muon["final_loss"] if best_muon else None,
                    "best_nonmuon_method": best_nonmuon["method"] if best_nonmuon else None,
                    "best_nonmuon_score": best_nonmuon["score"] if best_nonmuon else None,
                    "nonmuon_minus_muon": (
                        best_nonmuon["score"]-best_muon["score"] if best_nonmuon and best_muon else None
                    ),
                    "best_scalar_no_h0_method": best_scalar["method"] if best_scalar else None,
                    "best_scalar_no_h0_score": best_scalar["score"] if best_scalar else None,
                    "scalar_no_h0_minus_muon": (
                        best_scalar["score"]-best_muon["score"] if best_scalar and best_muon else None
                    ),
                    "best_h0_method": best_h0["method"] if best_h0 else None,
                    "best_h0_score": best_h0["score"] if best_h0 else None,
                    "h0_minus_muon": (
                        best_h0["score"]-best_muon["score"] if best_h0 and best_muon else None
                    ),
                    "best_signgd_method": best_sign["method"] if best_sign else None,
                    "best_signgd_score": best_sign["score"] if best_sign else None,
                    "signgd_minus_muon": (
                        best_sign["score"]-best_muon["score"] if best_sign and best_muon else None
                    ),
                    "wstar_status": w["status"],
                    "wstar_loss": w["reference_loss"],
                    "best_final_minus_wstar": (
                        best_any["final_loss"]-float(w["reference_loss"])
                        if best_any and w.get("reference_loss") is not None and math.isfinite(float(w["reference_loss"]))
                        else None
                    ),
                    "muon_final_minus_wstar": (
                        best_muon["final_loss"]-float(w["reference_loss"])
                        if best_muon and w.get("reference_loss") is not None and math.isfinite(float(w["reference_loss"]))
                        else None
                    ),
                    "wstar_reference_method": w.get("reference_method"),
                    "wstar_gradient_fro": w.get("gradient_fro"),
                    "wstar_seconds": w.get("seconds"),
                    "wstar_weight_fro": w.get("weight_fro"),
                    "wstar_weight_stable_rank": w.get("weight_stable_rank"),
                    "wstar_alignment_with_B": w.get("alignment_with_B"),
                    "wstar_alignment_with_H0invB": w.get("alignment_with_H0invB"),
                    "error": error.splitlines()[-1] if error else "",
                }
                case_rows.append(record)
        else:
            for criterion_name in config["selection"]["criteria"]:
                case_rows.append({
                    "case_id": case["id"],
                    "family": case["family"],
                    "status": status,
                    "criterion": criterion_name,
                    "wstar_status": w["status"],
                    "wstar_loss": w["reference_loss"],
                    "error": error.splitlines()[-1] if error else "",
                })
    return case_rows, method_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
    else:
        fields = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=6):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}g}"


def write_markdown(config, case_rows):
    completed = sorted({row["case_id"] for row in case_rows if row.get("status") == "completed"})
    lines = [
        "# V2 Consolidated Analysis",
        "",
        f"Generated at Unix time `{int(time.time())}`.",
        "",
        f"Completed settings: {len(completed)}/{len(config['cases'])}.",
        "",
        "Positive `nonmuon_minus_muon` means the best Muon method has a lower selection score than the best non-Muon method under that criterion. "
        "Negative `h0_minus_muon` means H0 is lower than Muon.",
        "",
        "## Completed Settings",
        "",
    ]
    if not completed:
        lines.append("No setting has completed yet.")
    for cid in completed:
        lines.append(f"### `{cid}`")
        local = [row for row in case_rows if row["case_id"] == cid and row.get("status") == "completed"]
        for row in local:
            lines.append(
                f"- `{row['criterion']}`: best `{row.get('best_method')}` score {fmt(row.get('best_score'))}; "
                f"Muon `{row.get('best_muon_method')}` score {fmt(row.get('best_muon_score'))}; "
                f"non-Muon minus Muon {fmt(row.get('nonmuon_minus_muon'))}; "
                f"H0 minus Muon {fmt(row.get('h0_minus_muon'))}; "
                f"SignGD minus Muon {fmt(row.get('signgd_minus_muon'))}; "
                f"W* `{row.get('wstar_status')}` loss {fmt(row.get('wstar_loss'))}."
            )
        lines.append("")
    lines.extend([
        "## Output Files",
        "",
        "- `summary_by_case.csv`: one row per completed or pending setting and selection criterion.",
        "- `selected_methods.csv`: one row per selected optimizer curve in each completed setting.",
        "",
    ])
    return "\n".join(lines)


def main():
    config = json.loads((HERE/"config"/"study_v2.json").read_text(encoding="utf-8"))
    store = Store(HERE/config["storage"]["database"])
    case_rows, method_rows = summary_rows(config, store)
    store.close()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS_DIR/"summary_by_case.csv", case_rows)
    write_csv(ANALYSIS_DIR/"selected_methods.csv", method_rows)
    (ANALYSIS_DIR/"README.md").write_text(write_markdown(config, case_rows), encoding="utf-8")
    print(f"Wrote {ANALYSIS_DIR/'summary_by_case.csv'}")
    print(f"Wrote {ANALYSIS_DIR/'selected_methods.csv'}")
    print(f"Wrote {ANALYSIS_DIR/'README.md'}")


if __name__ == "__main__":
    main()
