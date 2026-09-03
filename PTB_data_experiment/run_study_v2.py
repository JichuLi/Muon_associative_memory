"""V2 scheduler: one worker at a time, with legacy reuse inside each case."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resources import BudgetExhausted, LaptopGuard, atomic_json, process_alive
from storage import Store
from summarize_v2 import update_dashboard_v2
from write_readme_v2 import write_readme


HERE = Path(__file__).resolve().parent


def completed_with_v2_selection(row, case):
    if not row or row[0] != "completed" or not row[2]:
        return False
    try:
        metadata = json.loads(row[1]) if row[1] else {}
        selection = json.loads(row[2])
    except json.JSONDecodeError:
        return False
    if any(method not in selection for method in case["methods"]):
        return False
    late = metadata.get("selection_late", {})
    return all(method in late for method in case["methods"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE/"config/study_v2.json"))
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--skip-wstar", action="store_true")
    parser.add_argument("--force-wstar", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    lock = HERE/"results"/"controller_v2.json"
    if lock.exists():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        if process_alive(previous.get("pid", 0)):
            raise RuntimeError(f"Controller {previous['pid']} is still live; do not start a second worker")
        if process_alive(previous.get("worker_pid", 0)):
            raise RuntimeError(f"Previous worker {previous['worker_pid']} is still live")
    atomic_json(lock, {"pid": os.getpid(), "started": time.time(), "status": "running"})
    controller_resources = {**config["resources"], "gpu_work_duty": 1.0, "battery_work_duty": 1.0}
    guard = LaptopGuard(HERE, controller_resources)
    store = Store(HERE/config["storage"]["database"])
    todo = [case for case in config["cases"] if not args.cases or case["id"] in args.cases]
    for case in todo:
        if (HERE/"STOP_AFTER_RUN").exists():
            break
        row = store.connection.execute(
            "SELECT status,metadata,selection,error FROM cases WHERE case_id=?", (case["id"],)
        ).fetchone()
        if completed_with_v2_selection(row, case):
            continue
        if row and row[0] == "error" and not args.retry_errors:
            continue
        try:
            guard.checkpoint(force=True)
        except BudgetExhausted:
            break
        logpath = HERE/"results"/"logs"/f"{case['id']}.v2.log"
        logpath.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE/"run_case_v2.py"), "--case", case["id"], "--config", str(args.config)]
        if args.skip_wstar:
            cmd.append("--skip-wstar")
        if args.force_wstar:
            cmd.append("--force-wstar")
        with logpath.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(
                cmd,
                cwd=HERE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            )
            atomic_json(
                lock,
                {"pid": os.getpid(), "worker_pid": child.pid, "case_id": case["id"], "status": "running", "updated": time.time()},
            )
            print(f"Started {case['id']} (PID {child.pid}).", flush=True)
            while child.poll() is None:
                time.sleep(5)
            print(f"Finished {case['id']}: exit {child.returncode}", flush=True)
            if child.returncode == 3:
                break
        update_dashboard_v2(config, store)
        write_readme(config)
    update_dashboard_v2(config, store)
    write_readme(config)
    counts = dict(store.connection.execute("SELECT status,COUNT(*) FROM cases GROUP BY status").fetchall())
    status = "all_requested_completed" if all(
        store.connection.execute("SELECT status FROM cases WHERE case_id=?", (case["id"],)).fetchone() == ("completed",)
        for case in todo
    ) else "unfinished"
    atomic_json(lock, {"pid": os.getpid(), "status": status, "counts": counts, "ended": time.time()})
    store.close()


if __name__ == "__main__":
    main()
