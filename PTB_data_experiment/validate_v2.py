"""Lightweight v2 configuration validation without running optimizer traces."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from resources import LaptopGuard, atomic_json
from targets import Corpus


HERE = Path(__file__).resolve().parent


def validate(config):
    guard = LaptopGuard(HERE, config["resources"], enabled=False)
    corpus_cache = {}
    errors = []
    rows = []
    expected_methods = set(config["cases"][0]["methods"])
    for case in config["cases"]:
        cid = case["id"]
        if case["bias"]["kind"] != "zero":
            errors.append(f"{cid}: bias is not zero")
        if case.get("corpus", "ptb") != "ptb":
            errors.append(f"{cid}: corpus is not PTB")
        if set(case["methods"]) != expected_methods:
            errors.append(f"{cid}: optimizer set differs from first case")
        corpus_name = case.get("corpus", "ptb")
        if corpus_name not in corpus_cache:
            corpus_cache[corpus_name] = Corpus(corpus_name, guard)
        target = corpus_cache[corpus_name].target(case["target"], case["target_seeds"][0])
        column_error = float(abs(target.P.sum(0, dtype=np.float64)-1).max())
        if not np.isfinite(target.P).all():
            errors.append(f"{cid}: target contains nonfinite probabilities")
        if target.P.min() < -1e-8:
            errors.append(f"{cid}: target contains negative probabilities")
        if column_error > 3e-6:
            errors.append(f"{cid}: target column normalization error {column_error}")
        rows.append({
            "case_id": cid,
            "family": case["family"],
            "target_kind": case["target"]["kind"],
            "zero_bias": case["bias"]["kind"] == "zero",
            "corpus": corpus_name,
            "min_probability": float(target.P.min()),
            "max_column_sum_error": column_error,
            "target_marginal_tv_from_source": float(abs(target.p-corpus_cache[corpus_name].p).sum()/2)
            if len(target.p) == len(corpus_cache[corpus_name].p) else None,
        })
    return {
        "ok": not errors,
        "errors": errors,
        "planned_cases": len(config["cases"]),
        "optimizer_count": len(expected_methods),
        "expected_methods": sorted(expected_methods),
        "rows": rows,
        "updated": time.time(),
    }


def main():
    config = json.loads((HERE/"config"/"study_v2.json").read_text(encoding="utf-8"))
    result = validate(config)
    atomic_json(HERE/"results"/"validation_v2.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
