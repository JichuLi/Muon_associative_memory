"""Compact v2 dashboard and machine-readable summaries."""
from __future__ import annotations

import csv
import html
import json
import math
import time
from pathlib import Path

from plots_v2 import score_trace
from storage import Store, unpack_arrays


HERE = Path(__file__).resolve().parent


def _same_lr(left, right):
    if left is None or right is None:
        return left is None and right is None
    left, right = float(left), float(right)
    return abs(left-right) <= max(1e-12, 1e-10*max(abs(left), abs(right), 1.0))


def _selections(metadata, raw_selection):
    return {
        "auc": json.loads(raw_selection),
        "late": metadata.get("selection_late", {}),
    }


def _criterion_rows(store, case, signature, selection, config):
    rows = []
    for criterion_name, chosen in selection.items():
        if criterion_name not in config["selection"]["criteria"]:
            continue
        for method, picked in chosen.items():
            values = []
            traces = store.connection.execute(
                "SELECT metadata,trace FROM runs WHERE case_id=?", (case["id"],)
            ).fetchall()
            for raw, blob in traces:
                meta = json.loads(raw)
                if meta.get("protocol_signature") != signature:
                    continue
                if meta["method"] != method or not _same_lr(meta["lr"], picked["lr"]) or meta["status"] != "completed":
                    continue
                trace = unpack_arrays(blob)
                values.append({
                    "auc_score": float(meta["auc"]),
                    "late_score": score_trace(trace, config["selection"]["criteria"]["late"]),
                    "final_loss": float(trace["loss"][-1]),
                })
            if values:
                rows.append({
                    "criterion": criterion_name,
                    "method": method,
                    "lr": picked["lr"],
                    "auc_score": sum(v["auc_score"] for v in values)/len(values),
                    "late_score": sum(v["late_score"] for v in values)/len(values),
                    "final_loss": sum(v["final_loss"] for v in values)/len(values),
                })
    return rows


def _best(rows, criterion, methods):
    candidates = [row for row in rows if row["criterion"] == criterion and row["method"] in methods]
    if not candidates:
        return None
    key = "auc_score" if criterion == "auc" else "late_score"
    return min(candidates, key=lambda row: row[key])


def _wstar(case_id):
    path = HERE/"results"/"v2"/case_id/"minimizer"/"wstar.json"
    if not path.exists():
        return {"status": "missing", "reference_loss": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"status": data.get("status"), "reference_loss": data.get("reference_loss")}


def update_dashboard_v2(config, store):
    records = []
    scalar_methods = ["gd_const", "ngd_const", "signgd_const", "gd_ls", "signgd_ls"]
    muon_methods = ["muon_const", "muon_ls"]
    h0_methods = ["h0_const", "h0_ls"]
    for case in config["cases"]:
        row = store.connection.execute(
            "SELECT status,metadata,selection,error FROM cases WHERE case_id=?", (case["id"],)
        ).fetchone()
        record = {
            "case_id": case["id"],
            "family": case["family"],
            "stage": case["stage"],
            "status": row[0] if row else "pending",
        }
        if row and row[1]:
            metadata = json.loads(row[1])
            record["signature"] = metadata.get("protocol_signature")
            record["initial_hessian_condition"] = next(
                iter(metadata.get("representations", {}).values()), {}
            ).get("initial_hessian_condition")
            if row[2]:
                selection = _selections(metadata, row[2])
                metric_rows = _criterion_rows(store, case, metadata.get("protocol_signature"), selection, config)
                for criterion in config["selection"]["criteria"]:
                    best_scalar = _best(metric_rows, criterion, scalar_methods)
                    best_muon = _best(metric_rows, criterion, muon_methods)
                    best_h0 = _best(metric_rows, criterion, h0_methods)
                    if best_scalar and best_muon:
                        key = "auc_score" if criterion == "auc" else "late_score"
                        record[f"{criterion}_best_scalar"] = best_scalar["method"]
                        record[f"{criterion}_best_muon"] = best_muon["method"]
                        record[f"{criterion}_scalar_minus_muon"] = best_scalar[key]-best_muon[key]
                    if best_h0 and best_muon:
                        key = "auc_score" if criterion == "auc" else "late_score"
                        record[f"{criterion}_best_h0"] = best_h0["method"]
                        record[f"{criterion}_h0_minus_muon"] = best_h0[key]-best_muon[key]
        if row and row[3]:
            record["error"] = row[3].splitlines()[-1]
        wstar = _wstar(case["id"])
        record["wstar_status"] = wstar["status"]
        record["wstar_loss"] = wstar["reference_loss"]
        records.append(record)
    keys = sorted(set().union(*(r.keys() for r in records)))
    overview = HERE/"results"/"overview_v2.csv"
    overview.parent.mkdir(parents=True, exist_ok=True)
    with overview.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    done = sum(r["status"] == "completed" for r in records)
    total_runs = store.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    progress = {
        "completed_cases": done,
        "planned_cases": len(records),
        "saved_candidate_runs": total_runs,
        "updated": time.time(),
        "by_status": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
    }
    (HERE/"results"/"progress_v2.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    html_rows = []
    for r in records:
        cid = r["case_id"]
        figures = ""
        if r["status"] == "completed":
            links = []
            for label, relpath in [
                ("AUC loss", f"figures/v2/{cid}/loss/loss_vs_step_auc_selected.png"),
                ("Late loss", f"figures/v2/{cid}/loss/loss_vs_step_late_selected.png"),
                ("Dynamics", f"figures/v2/{cid}/dynamics/optimizer_dynamics_auc_selected.png"),
                ("LR sweep", f"figures/v2/{cid}/selection/lr_sweeps_auc.png"),
                ("W*", f"figures/v2/{cid}/minimizer/wstar_singular_values.png"),
            ]:
                if (HERE/relpath).exists():
                    links.append(f'<a href="{relpath}">{label}</a>')
            figures = " · ".join(links)
        html_rows.append(
            f"<tr><td>{html.escape(cid)}</td><td>{html.escape(str(r['family']))}</td>"
            f"<td>{html.escape(str(r['status']))}</td>"
            f"<td>{html.escape(str(r.get('auc_scalar_minus_muon', '')))}</td>"
            f"<td>{html.escape(str(r.get('late_scalar_minus_muon', '')))}</td>"
            f"<td>{html.escape(str(r.get('wstar_status', '')))}</td><td>{figures}</td>"
            f"<td>{html.escape(r.get('error', ''))}</td></tr>"
        )
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
    <title>PTB Muon Mechanism Study v2</title>
    <style>body{{font:15px system-ui;margin:36px;max-width:1500px;color:#203040}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dde3ea;text-align:left}}
    a{{color:#146c8d}}input{{padding:8px;width:420px}}p,li{{line-height:1.7}}</style>
    <h1>PTB Muon Mechanism Study v2</h1>
    <p>PTB only · one seed · reused legacy-compatible traces · all settings use GD/NGD/Muon/H0/SignGD · two selection standards.</p>
    <p>Completed {done}/{len(records)} settings. Saved {total_runs} run rows. <a href="results/overview_v2.csv">CSV summary</a> ·
    <a href="config/study_v2.json">v2 config</a> · <a href="legacy/mechanism_v1_20260902/REPORT.html">legacy report</a></p>
    <input id="search" placeholder="Filter settings">
    <table><thead><tr><th>Setting</th><th>Family</th><th>Status</th><th>AUC scalar−Muon</th>
    <th>Late scalar−Muon</th><th>W*</th><th>Figures</th><th>Error</th></tr></thead>
    <tbody>{''.join(html_rows)}</tbody></table>
    <script>document.getElementById('search').oninput=function(){{
    for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase());
    }}</script></html>"""
    (HERE/"REPORT_V2.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    config = json.loads((HERE/"config/study_v2.json").read_text(encoding="utf-8"))
    store = Store(HERE/config["storage"]["database"])
    update_dashboard_v2(config, store)
    store.close()
