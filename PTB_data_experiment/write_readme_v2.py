"""Generate the detailed v2 experiment README from the canonical config."""
from __future__ import annotations

import json
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


SETTING_DESCRIPTIONS = {
    "ptb": (
        "PTB default conditional. Start from the empirical PTB bigram conditional Q_PTB(y|x), "
        "then keep the legacy smoothing recipe P(y|x) = 0.999 Q_PTB(y|x) + 0.001 p(y). "
        "The context weights pi(x) and output marginal p(y) are the PTB training-set unigram marginals. "
        "This is the main natural-language baseline and is intentionally kept compatible with legacy traces."
    ),
    "sampled_onehot": (
        "For every context x, draw one fixed next-token label y from the empirical PTB conditional Q_PTB(.|x), "
        "then set P(y|x)=1 for that sampled label and zero for all other outputs. This keeps only one sampled "
        "association per context, so the target is deterministic but the labels still come from the empirical bigram law."
    ),
    "hard_copy": (
        "Identity-copy associative-memory target: P(y|x)=1 when the output token index y equals the context token index x. "
        "The background probability is zero and every column is deterministic. This setting is marked as having no finite "
        "softmax minimizer under the zero-bias protocol, so W* is skipped."
    ),
    "soft_copy_05": (
        "Soft identity-copy target with association strength 0.5. The exact recipe is "
        "P(.|x)=0.5 delta_x + 0.5 pi(.), where pi is the PTB context unigram vector reused as the output-side background. "
        "This tests whether a diagonal copy component is enough to favor Muon even when half the mass is independent."
    ),
    "rewired": (
        "Marginal-preserving rewire null. The observed PTB bigram events are expanded into individual (y,x) tokens, "
        "all y labels are randomly permuted across the original x positions, then the shuffled counts are normalized "
        "back into conditionals. The output and context marginals are preserved by construction, but local bigram "
        "association is destroyed up to finite-count noise. The final v1-compatible recipe uses a 0.001 p(y) smoothing mixture."
    ),
    "independent": (
        "Exact independent-target null. Every context has the same conditional distribution P(y|x)=p(y). "
        "This removes all x-to-y association while preserving the PTB output heavy tail exactly."
    ),
    "flat_context": (
        "Same smoothed PTB conditional columns as the default setting, but the context weights are flattened: "
        "pi(x) is replaced by a uniform distribution over the 5000 retained contexts. This separates conditional "
        "structure from the natural heavy-tailed context-frequency weighting."
    ),
    "permuted_entries": (
        "The target data are the PTB default, but the Muon matrix geometry is scrambled. A fixed random permutation "
        "of the d by d parameter entries is applied before taking Muon's polar factor, and inverted afterward. "
        "GD, NGD, H0, and SignGD still see the physical gradient; this isolates whether Muon's advantage depends on "
        "the meaningful matrix arrangement of gradient entries."
    ),
    "onehot_mix_0.5": (
        "A 50/50 mixture of the fixed sampled-one-hot target and the unsmoothed empirical PTB conditional: "
        "P=(1-alpha) P_onehot + alpha Q_PTB with alpha=0.5. This reintroduces empirical association while keeping "
        "a strong deterministic sampled component."
    ),
    "association_0.1": (
        "Low-association empirical mixture: P=0.1 Q_PTB + 0.9 p(y). The output heavy tail remains, but only ten "
        "percent of the conditional association comes from PTB bigrams."
    ),
    "rewire_mix_0.5": (
        "Half natural PTB and half marginal-preserving rewire. The recipe is the average of the smoothed PTB default "
        "and the smoothed rewired-count null, so marginals are close to PTB while association is partially disrupted."
    ),
    "context_gamma0.5": (
        "PTB default conditional columns with context weights transformed to pi(x)^0.5 and renormalized. "
        "This softens the context heavy tail and gives more relative weight to mid/tail contexts."
    ),
    "onehot_mix_0.25": (
        "One-hot/empirical mixture with alpha=0.25: 75 percent fixed sampled-one-hot target plus 25 percent "
        "unsmoothed empirical PTB conditional. This keeps the target close to deterministic while adding real bigram structure."
    ),
    "onehot_mix_0.75": (
        "One-hot/empirical mixture with alpha=0.75: 25 percent fixed sampled-one-hot target plus 75 percent "
        "unsmoothed empirical PTB conditional. This is closer to PTB while retaining exact zeros from the one-hot component."
    ),
    "sample_k4": (
        "For each context, draw four fixed next-token labels from Q_PTB(.|x) and place equal probability 1/4 on them. "
        "This creates a sparse multi-label conditional target with empirical sampling but no dense PTB probabilities."
    ),
    "sample_k16": (
        "For each context, draw sixteen fixed next-token labels from Q_PTB(.|x) and place equal probability 1/16 on them. "
        "Compared with k=4, this is less deterministic and closer to a small empirical support approximation."
    ),
    "association_0.5": (
        "Medium-association empirical mixture: P=0.5 Q_PTB + 0.5 p(y). It preserves half of the empirical conditional "
        "association while retaining a strong independent background."
    ),
    "context_gamma2": (
        "PTB default conditional columns with context weights transformed to pi(x)^2 and renormalized. "
        "This exaggerates the context heavy tail and strongly emphasizes frequent contexts."
    ),
    "copy_assoc_0.1": (
        "Weak identity-copy target: P(.|x)=0.1 delta_x + 0.9 pi(.). It tests whether a small diagonal copy component "
        "is enough to create a Muon-favorable low-rank or high-alignment training geometry."
    ),
    "prototype_32": (
        "Shared conditional prototype target. Contexts are clustered into 32 groups using data-only Hellinger geometry "
        "on sqrt PTB conditionals. Every context in a group receives the same pi-weighted prototype conditional, with "
        "residual=0. This makes p(y|x) shared across many contexts instead of idiosyncratic per token."
    ),
    "argmax_ptb": (
        "Deterministic natural argmax target. For every context, choose the most likely empirical PTB next token and set "
        "that label to probability one. This keeps natural labels but discards all uncertainty and non-argmax alternatives."
    ),
    "top2_renormalized": (
        "Sparse natural top-k target with k=2. For each context, keep only the two largest empirical PTB probabilities, "
        "zero out the rest, and renormalize the column. No background is added."
    ),
    "top8_renormalized": (
        "Sparse natural top-k target with k=8. It keeps more local uncertainty than top2 but still removes almost all "
        "tail alternatives in each context. No background is added."
    ),
    "frequency_matched_random_onehot": (
        "Frequency-matched random deterministic control. Tokens are sorted by the average of context frequency pi and "
        "output frequency p, divided into tiny adjacent frequency bins, and labels are randomly permuted within each bin. "
        "This keeps output-label frequencies close to the input frequency profile while removing real bigram semantics."
    ),
    "frequency_matched_random_soft_k4": (
        "Frequency-matched random sparse control. For each context, choose four labels by repeated within-bin frequency "
        "permutations using bins of width 8, then assign probability 1/4 to each chosen label. It has sparse support like "
        "sample_k4 but much weaker marginal drift."
    ),
    "random_context_column_permutation": (
        "Column-permutation alignment control. The full smoothed PTB conditional columns are kept intact as a multiset, "
        "but columns are randomly reassigned to contexts. This preserves the distribution of conditional entropies and "
        "column shapes while breaking the match between each context embedding and its original conditional."
    ),
    "anti_frequency_entropy_alignment": (
        "Entropy-frequency alignment intervention. PTB conditional columns are sorted by entropy, contexts are sorted by "
        "frequency, and low-entropy conditionals are assigned to high-frequency contexts. This intentionally couples "
        "frequent contexts with sharper conditionals."
    ),
    "head_background_soft_target": (
        "Head-background soft target. The default smoothed PTB conditional is mixed with a context-independent background "
        "distribution supported on the top 1000 output tokens by p(y): P=0.7 Q_smooth + 0.3 p_head(y)."
    ),
    "tail_background_soft_target": (
        "Tail-background soft target. The default smoothed PTB conditional is mixed with a context-independent background "
        "distribution supported on the 2000 least frequent output tokens by p(y): P=0.7 Q_smooth + 0.3 p_tail(y)."
    ),
    "class_block_soft_target": (
        "Frequency-block class target. Outputs are partitioned into 32 approximately equal p-mass frequency blocks. "
        "For each context, find the frequency block containing its PTB argmax token and use the p-normalized distribution "
        "inside that block as the main target; then mix in 0.05 p(y) so the target has full support."
    ),
}


FIGURE_RECIPES = [
    (
        "loss/loss_vs_step_{criterion}_selected.png",
        "raw training cross-entropy loss versus optimizer update step",
        "Curves use the LR selected by the named criterion. A dotted L(W*) reference line is drawn whenever an independent W* reference loss is available; the legend records whether it is `solved` or approximate.",
    ),
    (
        "dynamics/optimizer_dynamics_{criterion}_selected.png",
        "eta, Frobenius update length, gradient norm, prediction entropy, prediction marginal TV, and top-32 gradient spectral energy",
        "The first three panels come from the full trace; the last three come from selected-run diagnostic snapshots.",
    ),
    (
        "dynamics/frequency_ce_{criterion}_selected.png",
        "within-bucket CE for head, middle, and tail context-frequency buckets",
        "Contexts are sorted by target pi; head is the first 20 percent, middle the next 40 percent, and tail the last 40 percent.",
    ),
    (
        "dynamics/geometry_{criterion}_selected.png",
        "directional curvature, current-to-initial curvature ratio, gradient nuclear effective rank, and marginal-gradient ratio",
        "Curvature is measured along the actual update direction at each step; nuclear effective rank is computed from gradient singular values.",
    ),
    (
        "dynamics/gradient_spectra_{criterion}_selected.png",
        "gradient singular values normalized by the largest singular value at diagnostic steps",
        "The dotted line is the polar cutoff rtol=1e-7 used by Muon.",
    ),
    (
        "dynamics/heldout_{criterion}_selected.png",
        "held-out CE dynamics when a held-out count probe is attached",
        "This file is optional and appears only for cases that attach held-out evaluation counts.",
    ),
    (
        "selection/lr_sweeps_{criterion}.png",
        "constant-LR sweep score for each constant optimizer",
        "The y-axis is the named selection score; the star marks the selected constant LR for that optimizer and criterion. "
        "For `auc`, this is exactly the legacy mean(log CE) score.",
    ),
]

MINIMIZER_FIGURES = [
    (
        "minimizer/wstar_scalar_properties.png",
        "scalar properties of the reference W*",
        "Shows reference loss, gradient norm, matrix norms, stable rank, nuclear effective rank, threshold ranks, alignments, prediction entropy, prediction marginal TV, and bucket CE summaries.",
    ),
    (
        "minimizer/wstar_singular_values.png",
        "singular values of the reference W*",
        "Two panels show raw singular values and values normalized by the largest singular value; the vertical dotted line marks stable rank ||W||_F^2/||W||_op^2.",
    ),
    (
        "minimizer/wstar_bucket_ce.png",
        "head/middle/tail context-bucket CE at W*",
        "Uses the same context-frequency buckets as the frequency dynamics plot.",
    ),
    (
        "minimizer/wstar_solver_traces.png",
        "reference solver loss and gradient norm versus reference descent step",
        "Shows the configured L-BFGS reference attempt used for the reported W* candidate.",
    ),
]


def status_label(path):
    return "generated" if path.exists() else "pending"


def rel(path):
    return path.relative_to(HERE).as_posix()


def target_formula(case):
    return json.dumps(case["target"], ensure_ascii=False, indent=2)


def criterion_sentence(name, criterion):
    if criterion["kind"] == "mean_log_ce":
        metric = "the legacy AUC rule, mean log training CE"
    elif criterion["kind"] == "mean_ce":
        metric = "mean raw training CE"
    else:
        metric = criterion["kind"]
    return f"`{name}` selects by {metric} over inclusive steps {criterion['start']}..{criterion['end']}."


def case_readme_block(case, config):
    cid = case["id"]
    fig_root = HERE/"figures"/"v2"/cid
    result_root = HERE/"results"/"v2"/cid
    lines = [
        f"### `{cid}`",
        "",
        SETTING_DESCRIPTIONS.get(cid, "No hand-written description has been added yet; inspect the exact recipe below."),
        "",
        "Exact target recipe:",
        "",
        "```json",
        target_formula(case),
        "```",
        "",
        f"Family: `{case['family']}`. Geometry: `{case['geometry']['kind']}`. Bias: `{case['bias']['kind']}`. "
        f"Target seeds: `{case['target_seeds']}`. Representation seeds: `{case['rep_seeds']}`.",
        "",
        f"Per-setting results root: `{rel(result_root)}`.",
        "",
        "Expected figures for this setting:",
    ]
    for criterion in config["selection"]["criteria"]:
        for pattern, metric, how in FIGURE_RECIPES:
            filename = pattern.format(criterion=criterion)
            path = fig_root/filename
            lines.append(f"- `{rel(path)}` [{status_label(path)}]: {metric}. {how}")
    for filename, metric, how in MINIMIZER_FIGURES:
        path = fig_root/filename
        lines.append(f"- `{rel(path)}` [{status_label(path)}]: {metric}. {how}")
    wstar_path = result_root/"minimizer"/"wstar.json"
    if wstar_path.exists():
        try:
            wstar = json.loads(wstar_path.read_text(encoding="utf-8"))
            lines.append("")
            lines.append(
                f"W* status: `{wstar.get('status')}`; reference loss: `{wstar.get('reference_loss')}`; "
                f"target has zero support: `{wstar.get('target_has_zero_support')}`."
            )
        except json.JSONDecodeError:
            lines.append("")
            lines.append("W* status file exists but could not be parsed.")
    return "\n".join(lines)


def build_readme(config):
    methods = ", ".join(f"`{m}`" for m in config["cases"][0]["methods"])
    lines = [
        "# PTB Mechanism Study v2",
        "",
        "This README is generated from `config/study_v2.json` by `write_readme_v2.py`. "
        "Run the generator again after experiments finish to refresh generated/pending figure status labels.",
        "",
        f"Generated at Unix time `{int(time.time())}`.",
        "",
        "## Scope",
        "",
        "The v2 study is PTB-only and every setting uses a zero-bias softmax model. "
        "Old v1 outputs are archived under `legacy/mechanism_v1_20260902/`; compatible legacy trajectories are reused by "
        "matching the numerical protocol signature and run id. The v2 additions are SignGD, complete H0 optimizer runs, "
        "two best-selection criteria, per-setting output folders, and W* reference attempts.",
        "",
        "## Directory Layout",
        "",
        "- `README_V2_CN.md`: detailed Chinese README with setup derivations, figure metrics, setting recipes, and Muon-benefit summary.",
        "- `INTEGRATED_ANALYSIS_V2.md`: cross-setting synthesis of the completed run.",
        "- `config/study_v2.json`: canonical experiment specification.",
        "- `results/study_v2.sqlite`: compressed run traces, diagnostics, case status, and selections.",
        "- `results/v2/<case_id>/metadata/`: case specification, target statistics, and representation metadata.",
        "- `results/v2/<case_id>/selection/`: legacy-style AUC `selection.json`, plus selected-curve CSV files and the separate `selection_late.json`.",
        "- `results/v2/<case_id>/minimizer/`: W* JSON and scalar W* property CSV.",
        "- `figures/v2/<case_id>/loss/`: loss-vs-step figures.",
        "- `figures/v2/<case_id>/dynamics/`: optimizer, frequency, geometry, spectra, and optional held-out dynamics.",
        "- `figures/v2/<case_id>/selection/`: constant-LR sweep figures.",
        "- `figures/v2/<case_id>/minimizer/`: W* property figures.",
        "- `legacy/mechanism_v1_20260902/`: archived v1 results, figures, and report.",
        "",
        "## Optimizers",
        "",
        f"Every setting runs the same optimizer list: {methods}.",
        "",
        "- `gd_const` and `gd_ls`: full-batch gradient descent direction `D=-G`; either best constant LR or exact ray line search.",
        "- `ngd_const`: normalized gradient direction `D=-G/||G||_F` with best constant LR.",
        "- `muon_const` and `muon_ls`: Muon direction `D=-polar(G)` with the configured polar cutoff; either best constant LR or exact ray line search.",
        "- `h0_const` and `h0_ls`: fixed initial-Hessian inverse direction `D=-A0^{-1} G C0^{-1}`; either best constant LR or exact ray line search.",
        "- `signgd_const` and `signgd_ls`: elementwise sign direction `D=-sign(G)`; either best constant LR or exact ray line search.",
        "",
        "The line-search methods have no selected LR. For constant methods, each criterion chooses an LR independently.",
        "",
        "## Selection Criteria",
        "",
    ]
    for name, criterion in config["selection"]["criteria"].items():
        lines.append(f"- {criterion_sentence(name, criterion)}")
    lines.extend([
        "",
        "`auc` is intentionally identical to the legacy selection rule and uses `metadata['auc']` from the stored run. "
        "`late` is the new second rule and is written separately as `selection_late.json`.",
        "",
        "The loss figures plot raw CE versus step. They do not plot log loss or relative gap curves. "
        "The horizontal reference line is `L(W*)` from the independent W* reference attempt, not the best observed optimizer loss. The legend records whether the reference is certified `solved` or approximate.",
        "",
        "## W* Policy",
        "",
        "W* is attempted with a limited L-BFGS budget for every setting except those explicitly known to lack a finite minimizer under the zero-bias softmax protocol. "
        "Currently `hard_copy` is skipped for that reason. Zero-support targets such as sampled one-hot, argmax, and top-k are still attempted; "
        "if stationarity cannot be certified, the status is recorded as `attempted_zero_support_approximate` or `approximate`.",
        "",
        "Recorded W* properties include reference loss, gradient norm, Frobenius/operator/nuclear norms, stable rank, singular values, "
        "alignment with `B`, alignment with `H0^{-1}B`, prediction marginal TV, prediction entropy, and head/middle/tail CE.",
        "",
        "## Figure Catalog",
        "",
        "Each setting uses the same planned figure set. A file marked `pending` means the experiment or that optional metric has not produced it yet.",
        "",
    ])
    for pattern, metric, how in FIGURE_RECIPES:
        lines.append(f"- `{pattern}`: {metric}. {how}")
    for filename, metric, how in MINIMIZER_FIGURES:
        lines.append(f"- `{filename}`: {metric}. {how}")
    lines.extend([
        "",
        "`same_state` probes are not plotted in v2.",
        "",
        "## Commands",
        "",
        "```powershell",
        "python build_config_v2.py",
        "python inventory_v2.py",
        "python run_study_v2.py",
        "python write_readme_v2.py",
        "```",
        "",
        "Run selected cases only:",
        "",
        "```powershell",
        "python run_study_v2.py --cases ptb sample_k4 class_block_soft_target",
        "```",
        "",
        "Run the training traces first and solve W* references later:",
        "",
        "```powershell",
        "python run_study_v2.py --skip-wstar",
        "python wstar.py --case ptb",
        "```",
        "",
        "## Settings",
        "",
    ])
    for case in config["cases"]:
        lines.append(case_readme_block(case, config))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_readme(config=None):
    if config is None:
        config = json.loads((HERE/"config"/"study_v2.json").read_text(encoding="utf-8"))
    text = build_readme(config)
    root = HERE/"README_V2.md"
    root.write_text(text, encoding="utf-8")
    out = HERE/"results"/"README_V2.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return root, out


def main():
    root, out = write_readme()
    print(f"Wrote {root}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
