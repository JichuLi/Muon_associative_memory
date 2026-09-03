"""Inventory reusable legacy traces and estimate remaining v2 runtime."""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from pathlib import Path

import numpy as np

from run_case_v2 import physics_signature
from storage import Store


HERE = Path(__file__).resolve().parent


def legacy_connections(config):
    for rel in config.get("storage", {}).get("legacy_databases", []):
        path = HERE/rel
        if path.exists():
            yield path, sqlite3.connect(path)


def observed_seconds(config):
    values = {}
    for _, con in legacy_connections(config):
        for raw in con.execute("SELECT metadata FROM runs"):
            meta = json.loads(raw[0])
            if meta.get("status") == "completed" and meta.get("seconds"):
                values.setdefault(meta["method"], []).append(float(meta["seconds"]))
        con.close()
    med = {method: statistics.median(items) for method, items in values.items() if items}
    # SignGD has the same full-softmax evaluation cost; its direction is cheap.
    if "gd_const" in med:
        med.setdefault("signgd_const", med["gd_const"])
    if "gd_ls" in med:
        med.setdefault("signgd_ls", med["gd_ls"])
    return med


def legacy_count(config, case_id, signature, method):
    count = 0
    for _, con in legacy_connections(config):
        for raw in con.execute("SELECT metadata FROM runs WHERE case_id=? AND method=?", (case_id, method)):
            meta = json.loads(raw[0])
            if meta.get("protocol_signature") == signature and meta.get("status") == "completed":
                count += 1
        con.close()
    return count


def current_count(store, case_id, signature, method):
    count = 0
    for raw in store.connection.execute("SELECT metadata FROM runs WHERE case_id=? AND method=?", (case_id, method)):
        meta = json.loads(raw[0])
        if meta.get("protocol_signature") == signature and meta.get("status") == "completed":
            count += 1
    return count


def expected_candidates(config, method):
    if method.endswith("_ls"):
        return 1
    coarse = int(config["lr_log10_grids"][method][2])
    # Two criteria can refine two different coarse winners; often they coincide.
    return coarse + len(config["refinement_offsets"])*1.5


def wstar_expected_minutes(case):
    kind = case["target"]["kind"]
    if kind == "copy" and float(case["target"].get("association", 0)) >= 1:
        return 0.1
    return 3.0


def inventory(config):
    store = Store(HERE/config["storage"]["database"])
    med = observed_seconds(config)
    default_seconds = {
        "gd_ls": 75.0,
        "muon_ls": 80.0,
        "h0_ls": 75.0,
        "signgd_ls": 75.0,
        "gd_const": 7.0,
        "ngd_const": 7.0,
        "muon_const": 9.0,
        "h0_const": 7.0,
        "signgd_const": 7.0,
    }
    rows = []
    total_seconds = 0.0
    for case in config["cases"]:
        signature = physics_signature(config, case)
        reusable = 0
        current = 0
        expected = 0.0
        missing_methods = []
        for method in case["methods"]:
            have = current_count(store, case["id"], signature, method)
            reuse = legacy_count(config, case["id"], signature, method)
            need = expected_candidates(config, method)
            current += have
            reusable += reuse
            if have + reuse < need:
                missing_methods.append(method)
                missing = max(0.0, need-have-reuse)
                expected += missing*med.get(method, default_seconds[method])
        expected += 60.0*wstar_expected_minutes(case)
        total_seconds += expected
        rows.append({
            "case_id": case["id"],
            "family": case["family"],
            "signature": signature,
            "current_run_rows": current,
            "legacy_reusable_run_rows": reusable,
            "missing_methods": " ".join(missing_methods),
            "estimated_seconds_remaining": round(expected, 1),
            "estimated_minutes_remaining": round(expected/60, 2),
        })
    store.close()
    return rows, total_seconds


def main():
    config = json.loads((HERE/"config/study_v2.json").read_text(encoding="utf-8"))
    rows, total_seconds = inventory(config)
    outdir = HERE/"results"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/"inventory_v2.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "planned_cases": len(rows),
        "estimated_hours_remaining": total_seconds/3600,
        "estimated_hours_with_25pct_buffer": total_seconds*1.25/3600,
        "rows": rows,
    }
    (outdir/"inventory_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
