"""Refresh v2 W* references and figures without rerunning optimizer traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plots_v2 import case_result_dirs, plot_case_v2
from run_case_v2 import physics_signature
from storage import Store
from wstar import solve_wstar_for_case


HERE = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_row(store, case_id):
    row = store.connection.execute(
        "SELECT status,metadata,selection FROM cases WHERE case_id=?", (case_id,)
    ).fetchone()
    if row is None:
        return None
    return {"status": row[0], "metadata": row[1], "selection": row[2]}


def load_selection(case_id, row, config):
    dirs = case_result_dirs(case_id)
    metadata = json.loads(row["metadata"] or "{}")
    auc = json.loads(row["selection"] or "{}")
    late = metadata.get("selection_late")
    late_path = dirs["selection"]/"selection_late.json"
    if not late and late_path.exists():
        late = load_json(late_path)
    return {
        "criteria": {"auc": auc, "late": late or {}},
        "criteria_definition": config["selection"]["criteria"],
    }


def compact_wstar(result):
    return {k: v for k, v in result.items() if k != "weight_singular_values"}


def refresh_case(store, config, case, force_wstar, guard_enabled):
    row = case_row(store, case["id"])
    if row is None or row["status"] != "completed":
        return "skipped_not_completed"
    signature = physics_signature(config, case)
    wstar = None
    if config.get("wstar", {}).get("enabled"):
        wstar = solve_wstar_for_case(
            config, case, signature, force=force_wstar, guard_enabled=guard_enabled
        )
    metadata = json.loads(row["metadata"] or "{}")
    if wstar is not None:
        metadata["wstar"] = compact_wstar(wstar)
    plot_case_v2(store, case, signature, load_selection(case["id"], row, config), config)
    store.connection.execute(
        "UPDATE cases SET metadata=? WHERE case_id=?",
        (json.dumps(metadata), case["id"]),
    )
    store.connection.commit()
    return wstar.get("status") if wstar is not None else "plotted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE/"config"/"study_v2.json"))
    parser.add_argument("--case")
    parser.add_argument("--force-wstar", action="store_true")
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    wanted = {args.case} if args.case else None
    store = Store(HERE/config["storage"]["database"])
    try:
        for case in config["cases"]:
            if wanted is not None and case["id"] not in wanted:
                continue
            status = refresh_case(store, config, case, args.force_wstar, not args.no_guard)
            print(f"{case['id']}: {status}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
