"""Create the v2 per-setting directory layout before running experiments."""
from __future__ import annotations

import json
from pathlib import Path

from plots_v2 import case_figure_dirs, case_result_dirs
from resources import atomic_json
from write_readme_v2 import FIGURE_RECIPES, MINIMIZER_FIGURES, write_readme


HERE = Path(__file__).resolve().parent


def planned_figures(case, config):
    fig_root = HERE/"figures"/"v2"/case["id"]
    items = []
    for criterion_name in config["selection"]["criteria"]:
        for pattern, metric, how in FIGURE_RECIPES:
            relpath = (fig_root/pattern.format(criterion=criterion_name)).relative_to(HERE).as_posix()
            items.append({
                "path": relpath,
                "criterion": criterion_name,
                "metric": metric,
                "how_plotted": how,
                "optional": "heldout" in relpath,
            })
    for filename, metric, how in MINIMIZER_FIGURES:
        relpath = (fig_root/filename).relative_to(HERE).as_posix()
        items.append({
            "path": relpath,
            "criterion": None,
            "metric": metric,
            "how_plotted": how,
            "optional": False,
        })
    return items


def prepare_layout(config):
    for case in config["cases"]:
        figure_dirs = case_figure_dirs(case["id"])
        result_dirs = case_result_dirs(case["id"])
        for path in [*figure_dirs.values(), *result_dirs.values()]:
            path.mkdir(parents=True, exist_ok=True)
        atomic_json(
            result_dirs["metadata"]/"case_specification.json",
            {"case": case, "zero_bias": case["bias"]["kind"] == "zero"},
        )
        atomic_json(
            result_dirs["metadata"]/"figure_manifest.json",
            {"case_id": case["id"], "planned_figures": planned_figures(case, config)},
        )
    write_readme(config)


def main():
    config = json.loads((HERE/"config"/"study_v2.json").read_text(encoding="utf-8"))
    prepare_layout(config)
    print(f"Prepared layout for {len(config['cases'])} v2 settings.")


if __name__ == "__main__":
    main()
