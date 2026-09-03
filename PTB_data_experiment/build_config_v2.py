"""Build the PTB-only v2 mechanism-study configuration.

The v2 protocol keeps the old numerical physics for reusable trajectories,
adds SignGD and complete H0 runs everywhere, and adds two selection criteria.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent

ALL_METHODS = [
    "gd_ls",
    "muon_ls",
    "h0_ls",
    "signgd_ls",
    "gd_const",
    "ngd_const",
    "muon_const",
    "h0_const",
    "signgd_const",
]


def case(case_id, family, target, geometry=None, role="ptb_v2", estimated_seconds=1200):
    return {
        "id": case_id,
        "family": family,
        "stage": 1,
        "target": target,
        "bias": {"kind": "zero"},
        "geometry": geometry or {"kind": "original"},
        "target_seeds": [0],
        "rep_seeds": [0],
        "methods": list(ALL_METHODS),
        "role": role,
        "required": True,
        "estimated_seconds": estimated_seconds,
        "diagnostics_preconditioners": case_id in {
            "ptb",
            "hard_copy",
            "soft_copy_05",
            "permuted_entries",
            "frequency_matched_random_onehot",
            "class_block_soft_target",
        },
    }


def configuration():
    started = int(time.time())
    cases = [
        case("ptb", "natural_conditional", {"kind": "ptb", "rho": 0.001}, estimated_seconds=450),
        case("sampled_onehot", "finite_support", {"kind": "sample_k", "k": 1}, estimated_seconds=450),
        case("hard_copy", "copy_am", {"kind": "copy", "association": 1}, estimated_seconds=450),
        case("soft_copy_05", "copy_am", {"kind": "copy", "association": 0.5}, estimated_seconds=450),
        case("rewired", "null_no_association", {"kind": "rewire", "mix": 1}, estimated_seconds=450),
        case("independent", "null_no_association", {"kind": "association", "alpha": 0}, estimated_seconds=450),
        case("flat_context", "context_frequency", {"kind": "context_frequency", "gamma": 0}, estimated_seconds=450),
        case("permuted_entries", "matrix_geometry", {"kind": "ptb", "rho": 0.001},
             {"kind": "entry_permutation", "fraction": 1}, estimated_seconds=450),
        case("onehot_mix_0.5", "finite_support", {"kind": "onehot_mix", "alpha": 0.5}, estimated_seconds=450),
        case("association_0.1", "association_strength", {"kind": "association", "alpha": 0.1}, estimated_seconds=450),
        case("rewire_mix_0.5", "null_no_association", {"kind": "rewire", "mix": 0.5}, estimated_seconds=450),
        case("context_gamma0.5", "context_frequency", {"kind": "context_frequency", "gamma": 0.5}, estimated_seconds=450),
        case("onehot_mix_0.25", "finite_support", {"kind": "onehot_mix", "alpha": 0.25}, estimated_seconds=450),
        case("onehot_mix_0.75", "finite_support", {"kind": "onehot_mix", "alpha": 0.75}, estimated_seconds=450),
        case("sample_k4", "finite_support", {"kind": "sample_k", "k": 4}, estimated_seconds=450),
        case("sample_k16", "finite_support", {"kind": "sample_k", "k": 16}, estimated_seconds=450),
        case("association_0.5", "association_strength", {"kind": "association", "alpha": 0.5}, estimated_seconds=450),
        case("context_gamma2", "context_frequency", {"kind": "context_frequency", "gamma": 2}, estimated_seconds=450),
        case("copy_assoc_0.1", "copy_am", {"kind": "copy", "association": 0.1}, estimated_seconds=450),
        case("prototype_32", "shared_structure", {"kind": "prototype", "k": 32, "residual": 0}, estimated_seconds=450),
        case("argmax_ptb", "finite_support_new", {"kind": "argmax"}, estimated_seconds=900),
        case("top2_renormalized", "finite_support_new", {"kind": "top_k", "k": 2, "background": 0.0}, estimated_seconds=900),
        case("top8_renormalized", "finite_support_new", {"kind": "top_k", "k": 8, "background": 0.0}, estimated_seconds=900),
        case("frequency_matched_random_onehot", "random_control_new",
             {"kind": "frequency_matched_onehot"}, estimated_seconds=900),
        case("frequency_matched_random_soft_k4", "random_control_new",
             {"kind": "frequency_matched_soft_k", "k": 4, "bin_size": 8}, estimated_seconds=900),
        case("random_context_column_permutation", "alignment_new",
             {"kind": "context_column_permutation"}, estimated_seconds=900),
        case("anti_frequency_entropy_alignment", "alignment_new",
             {"kind": "entropy_context_alignment", "mode": "high_frequency_low_entropy"}, estimated_seconds=900),
        case("head_background_soft_target", "background_new",
             {"kind": "background_mix", "bucket": "head", "count": 1000, "background": 0.3}, estimated_seconds=900),
        case("tail_background_soft_target", "background_new",
             {"kind": "background_mix", "bucket": "tail", "count": 2000, "background": 0.3}, estimated_seconds=900),
        case("class_block_soft_target", "shared_structure_new",
             {"kind": "class_block_soft", "blocks": 32, "background": 0.05}, estimated_seconds=900),
    ]
    return {
        "schema_version": 2,
        # Keep this physics tag unchanged so old compatible GD/NGD/Muon/H0 traces are reusable.
        "protocol_version": "mechanism-v1",
        "study_version": "ptb-v2-signgd-h0-wstar",
        "objective": (
            "PTB-only mechanism map for which conditional distributions amplify Muon's benefit; "
            "reuse old compatible traces, add SignGD, complete H0, late-window selection, and W* references."
        ),
        "d": 256,
        "steps": 200,
        "vocab_size": 5000,
        "normalize_features": True,
        "dtype": "float32",
        "loss_accumulation": "float64",
        "tf32": False,
        "polar_rtol": 1e-7,
        "momentum": 0,
        "weight_decay": 0,
        "selection": {
            "criteria": {
                "auc": {"kind": "mean_log_ce", "start": 0, "end": 200},
                "late": {"kind": "mean_ce", "start": 120, "end": 150},
            },
            "constant_lr_selected_independently_per_method_and_criterion": True,
            "line_search_has_no_lr_selection": True,
        },
        "lr_log10_grids": {
            "gd_const": [1, 5, 9],
            "ngd_const": [-1, 3, 9],
            "muon_const": [-2, 2, 9],
            "h0_const": [-3, 2, 11],
            "signgd_const": [-3, 1, 9],
        },
        "refinement_offsets": [-0.375, -0.25, -0.125, 0.125, 0.25, 0.375],
        "max_boundary_extensions": 8,
        "boundary_log10_increment": 0.5,
        "ray_bisections": 20,
        "ray_max_expansions": 48,
        "ray_max_fro": 1e12,
        "diagnostic_steps": [0, 20, 100, 120, 150, 200],
        "same_state_steps": [0, 20, 100, 120, 150],
        "detailed_every": 20,
        "wstar": {
            "enabled": True,
            "attempt_all_except_definitely_unbounded": True,
            "skip_case_ids": ["hard_copy"],
            "skip_policy": (
                "Only skip settings whose target is known to have no finite minimizer "
                "under the zero-bias softmax model. Other targets get a limited-budget "
                "reference attempt and are marked approximate if the solver cannot certify stationarity."
            ),
            "solver": "torch_lbfgs_strong_wolfe",
            "reference_steps": 80,
            "lbfgs_history_size": 20,
            "lbfgs_max_eval_per_step": 8,
            "lbfgs_line_search": "strong_wolfe",
            "lbfgs_tolerance_change": 1e-9,
            "lbfgs_stall_patience": 12,
            "gradient_norm_goal": 1e-5,
        },
        "outputs": {
            "case_root": "results/v2/<case_id>",
            "figure_root": "figures/v2/<case_id>",
            "selection": "results/v2/<case_id>/selection",
            "minimizer": "results/v2/<case_id>/minimizer",
            "metadata": "results/v2/<case_id>/metadata",
            "loss_figures": "figures/v2/<case_id>/loss",
            "dynamics_figures": "figures/v2/<case_id>/dynamics",
            "selection_figures": "figures/v2/<case_id>/selection",
            "minimizer_figures": "figures/v2/<case_id>/minimizer",
        },
        "resources": {
            "cpu_threads": 2,
            "gpu_work_duty": 0.7,
            "battery_work_duty": 0.4,
            "monitor_interval_seconds": 30,
            "guard_every_steps": 20,
            "soft_temperature_c": 76,
            "hard_temperature_c": 82,
            "min_free_commit_gb": 1,
            "min_free_disk_gb": 12,
            "battery_pause_percent": 25,
            "gpu_allocator_fraction": 0.32,
            "deadline_unix": started + 7*24*3600,
        },
        "storage": {
            "database": "results/study_v2.sqlite",
            "legacy_databases": ["legacy/mechanism_v1_20260902/results/study.sqlite"],
            "save_candidate_weights": False,
            "save_dense_target_per_case": False,
            "save_all_candidate_loss_curves": True,
            "maximum_expected_output_gb": 4,
        },
        "cases": cases,
        "legacy": {
            "path": "legacy/mechanism_v1_20260902",
            "contains": ["results", "figures", "REPORT.html"],
            "reuse_policy": "copy compatible run rows and diagnostics by physics signature and run_id",
        },
    }


def main():
    config = configuration()
    path = HERE/"config/study_v2.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
