from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class StudyConfig:
    n_q: int
    n_k: int
    n_v: int
    n_noise: int
    m: int
    batch_size: int
    eval_every: int
    eval_repeats: int
    zipf_alpha: float
    n_heads: int
    aux_lr: float


@dataclass(frozen=True)
class RunSpec:
    group: str
    label: str
    variant: str
    train_scope: str
    optimizer: str
    lr: float
    seed: int
    seq_len: int
    d_model: int
    steps: int
    grad_clip: float

    @property
    def name(self) -> str:
        safe_lr = f"{self.lr:g}".replace(".", "p").replace("-", "m")
        safe_label = self.label.lower().replace("-", "_").replace("+", "p")
        return (
            f"{self.group}_{safe_label}_seq{self.seq_len}_d{self.d_model}_"
            f"{self.optimizer}_lr{safe_lr}_seed{self.seed}"
        )


PRESETS = {
    "debug": {
        "cfg": StudyConfig(
            n_q=64,
            n_k=64,
            n_v=256,
            n_noise=32,
            m=4,
            batch_size=64,
            eval_every=100,
            eval_repeats=2,
            zipf_alpha=1.1,
            n_heads=4,
            aux_lr=3e-4,
        ),
        "arch_steps": 200,
        "opt_steps": 250,
        "arch_seeds": [0],
        "opt_seeds": [0],
        "seq_lens": [64],
        "d_models": [64],
        "optimizer_lrs": [1e-4, 1e-3],
    },
    "main": {
        "cfg": StudyConfig(
            n_q=128,
            n_k=128,
            n_v=512,
            n_noise=64,
            m=7,
            batch_size=128,
            eval_every=250,
            eval_repeats=4,
            zipf_alpha=1.1,
            n_heads=4,
            aux_lr=3e-4,
        ),
        "arch_steps": 2500,
        "opt_steps": 3500,
        "arch_seeds": [0, 1],
        "opt_seeds": [0, 1, 2],
        "seq_lens": [16, 32, 64],
        "d_models": [64, 128, 256],
        "optimizer_lrs": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    },
}


ARCHITECTURES = [
    ("1L-attn", "one_layer_attn", "all"),
    ("2L-attn", "two_layer_attn", "all"),
    ("2L-full", "two_layer_full", "all"),
    ("4L-full", "full_transformer", "all"),
    ("2L-no-pos", "two_layer_no_pos", "all"),
    ("2L-fixed-embed", "two_layer_attn", "no_embedding_train"),
    ("2L-embed-only", "two_layer_attn", "embedding_only"),
    ("2L-attn-only", "two_layer_attn", "attention_only"),
    ("2L-qk-only", "two_layer_attn", "qk_only"),
    ("2L-freeze-QK", "two_layer_attn", "no_qk_train"),
]


def specs_for(preset: dict, only: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if only in {"all", "arch"}:
        for label, variant, scope in ARCHITECTURES:
            for seq_len in preset["seq_lens"]:
                for d_model in preset["d_models"]:
                    for seed in preset["arch_seeds"]:
                        specs.append(
                            RunSpec(
                                group="arch",
                                label=label,
                                variant=variant,
                                train_scope=scope,
                                optimizer="adamw",
                                lr=1e-3,
                                seed=seed,
                                seq_len=seq_len,
                                d_model=d_model,
                                steps=preset["arch_steps"],
                                grad_clip=1.0,
                            )
                        )
    if only in {"all", "opt"}:
        for optimizer in ["adamw", "muon", "ngd", "sgd"]:
            for lr in preset["optimizer_lrs"]:
                for seed in preset["opt_seeds"]:
                    specs.append(
                        RunSpec(
                            group="opt",
                            label=optimizer,
                            variant="two_layer_attn",
                            train_scope="all",
                            optimizer=optimizer,
                            lr=lr,
                            seed=seed,
                            seq_len=32,
                            d_model=128,
                            steps=preset["opt_steps"],
                            grad_clip=0.0,
                        )
                    )
    return specs


def run_command(cmd: list[str], run_dir: Path) -> None:
    print(" ".join(cmd), flush=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        t0 = time.perf_counter()
        last_ping = 0.0
        while proc.poll() is None:
            elapsed = time.perf_counter() - t0
            if elapsed - last_ping >= 20:
                print(f"running {run_dir.name}: {elapsed:.0f}s", flush=True)
                last_ping = elapsed
            time.sleep(1)
        if proc.returncode != 0:
            print(f"failed: {run_dir.name}", flush=True)
            if stderr_path.exists():
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                print("\n".join(tail), flush=True)
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_one(train_py: Path, out_dir: Path, cfg: StudyConfig, spec: RunSpec, device: str, force: bool) -> None:
    run_dir = out_dir / "runs" / spec.name
    if (run_dir / "summary.csv").exists() and not force:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study_spec.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    lr_args = ["--optimizer", spec.optimizer, "--lr", str(spec.lr)]
    if spec.optimizer == "muon":
        lr_args = ["--optimizer", "muon", "--lr", str(cfg.aux_lr), "--muon-lr", str(spec.lr), "--aux-lr", str(cfg.aux_lr)]
    cmd = [
        sys.executable,
        str(train_py),
        "--out-dir",
        str(run_dir),
        "--n-q",
        str(cfg.n_q),
        "--n-k",
        str(cfg.n_k),
        "--n-v",
        str(cfg.n_v),
        "--n-noise",
        str(cfg.n_noise),
        "--m",
        str(cfg.m),
        "--seq-len",
        str(spec.seq_len),
        "--zipf-alpha",
        str(cfg.zipf_alpha),
        "--batch-size",
        str(cfg.batch_size),
        "--steps",
        str(spec.steps),
        "--d-model",
        str(spec.d_model),
        "--n-heads",
        str(cfg.n_heads),
        "--variant",
        spec.variant,
        "--train-scope",
        spec.train_scope,
        "--eval-repeats",
        str(cfg.eval_repeats),
        "--eval-every",
        str(cfg.eval_every),
        "--grad-clip",
        str(spec.grad_clip),
        "--device",
        device,
        "--seed",
        str(spec.seed),
        *lr_args,
    ]
    run_command(cmd, run_dir)
    if (run_dir / "summary.csv").exists():
        row = pd.read_csv(run_dir / "summary.csv").tail(1).iloc[0]
        print(
            f"done {spec.name}: acc={row['eval_acc']:.4f} loss={row['eval_loss']:.4f} "
            f"margin={row['eval_margin']:.3f}",
            flush=True,
        )


def run_study(args: argparse.Namespace, preset: dict) -> None:
    train_py = Path(__file__).with_name("train.py")
    all_specs = specs_for(preset, args.only)
    if args.limit:
        all_specs = all_specs[: args.limit]
    for i, spec in enumerate(all_specs, 1):
        print(f"\n[{i}/{len(all_specs)}] {spec.name}", flush=True)
        run_one(train_py, args.out_dir, preset["cfg"], spec, args.device, args.force)


def auc_and_t80(metrics: pd.DataFrame) -> tuple[float, float]:
    metrics = metrics.sort_values("step")
    auc = float(
        ((metrics["eval_acc"].shift(-1) + metrics["eval_acc"]) / 2 * (metrics["step"].shift(-1) - metrics["step"]))
        .dropna()
        .sum()
        / max(float(metrics["step"].max()), 1.0)
    )
    hit = metrics[metrics["eval_acc"] >= 0.80]
    t80 = float(hit["step"].iloc[0]) if not hit.empty else float("nan")
    return auc, t80


def collect(
    out_dir: Path,
    run_roots: list[tuple[Path, set[str] | None]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_rows = []
    roots = run_roots or [(out_dir, None)]
    summary_paths = []
    for root, groups in roots:
        for path in sorted((root / "runs").glob("*/summary.csv")):
            summary_paths.append((path, groups))
    for summary_path, groups in sorted(summary_paths, key=lambda item: str(item[0])):
        run_dir = summary_path.parent
        spec = json.loads((run_dir / "study_spec.json").read_text(encoding="utf-8"))
        if "grad_clip" not in spec:
            spec["grad_clip"] = 1.0 if spec.get("group") == "arch" else float("nan")
        if groups is not None and spec.get("group") not in groups:
            continue
        summary = pd.read_csv(summary_path).tail(1).copy()
        metrics_path = run_dir / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        auc, t80 = auc_and_t80(metrics)
        for key, value in spec.items():
            summary[f"spec_{key}"] = value
            metrics[f"spec_{key}"] = value
        run_id = f"{run_dir.parent.parent.name}/{run_dir.name}" if run_roots else run_dir.name
        summary.insert(0, "run", run_id)
        summary["auc_acc"] = auc
        summary["t80_step"] = t80
        rows.append(summary)
        metrics.insert(0, "run", run_id)
        curve_rows.append(metrics)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False), pd.concat(curve_rows, ignore_index=True, sort=False)


def sem(series: pd.Series) -> float:
    return float(series.sem()) if len(series) > 1 else 0.0


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def aggregate_architecture(df: pd.DataFrame, curves: pd.DataFrame, agg: Path) -> pd.DataFrame:
    arch = df[df["spec_group"] == "arch"].copy()
    if arch.empty:
        return pd.DataFrame()
    out = (
        arch.groupby(["spec_label", "spec_seq_len", "spec_d_model"])
        .agg(
            acc_mean=("eval_acc", "mean"),
            acc_sem=("eval_acc", sem),
            full_vocab_mean=("eval_full_vocab_acc", "mean"),
            full_vocab_sem=("eval_full_vocab_acc", sem),
            loss_mean=("eval_loss", "mean"),
            margin_mean=("eval_margin", "mean"),
            head_mean=("eval_head_acc", "mean"),
            tail_mean=("eval_tail_acc", "mean"),
            auc_mean=("auc_acc", "mean"),
            t80_median=("t80_step", "median"),
            seeds=("spec_seed", "nunique"),
            params_trainable=("params_trainable", "first"),
            grad_clip=("spec_grad_clip", "first"),
        )
        .reset_index()
    )
    save_table(out, agg / "architecture_summary.csv")

    fig = agg / "figures"
    anchor = curves[
        (curves["spec_group"] == "arch")
        & (curves["spec_seq_len"] == 32)
        & (curves["spec_d_model"] == 128)
    ]
    if not anchor.empty:
        plt.figure(figsize=(8, 5))
        for label, group in anchor.groupby("spec_label"):
            line = group.groupby("step")["eval_acc"].agg(["mean", sem]).reset_index()
            plt.plot(line["step"], line["mean"], label=label)
            plt.fill_between(line["step"], line["mean"] - line["sem"], line["mean"] + line["sem"], alpha=0.13)
        plt.xlabel("step")
        plt.ylabel("eval_acc")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig / "architecture_curves_seq32_d128.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for label, group in anchor.groupby("spec_label"):
            line = group.groupby("step")["eval_full_vocab_acc"].agg(["mean", sem]).reset_index()
            plt.plot(line["step"], line["mean"], label=label)
            plt.fill_between(line["step"], line["mean"] - line["sem"], line["mean"] + line["sem"], alpha=0.13)
        plt.xlabel("step")
        plt.ylabel("eval_full_vocab_acc")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig / "architecture_full_vocab_curves_seq32_d128.png", dpi=180)
        plt.close()

    for d_model in sorted(out["spec_d_model"].unique()):
        sub = out[out["spec_d_model"] == d_model]
        plt.figure(figsize=(8, 5))
        for label, group in sub.groupby("spec_label"):
            group = group.sort_values("spec_seq_len")
            plt.plot(group["spec_seq_len"], group["acc_mean"], marker="o", label=label)
            plt.fill_between(
                group["spec_seq_len"],
                group["acc_mean"] - group["acc_sem"],
                group["acc_mean"] + group["acc_sem"],
                alpha=0.13,
            )
        plt.xlabel("sequence length")
        plt.ylabel("final eval_acc")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig / f"architecture_seq_sweep_d{int(d_model)}.png", dpi=180)
        plt.close()

    for seq_len in sorted(out["spec_seq_len"].unique()):
        sub = out[out["spec_seq_len"] == seq_len]
        plt.figure(figsize=(8, 5))
        for label, group in sub.groupby("spec_label"):
            group = group.sort_values("spec_d_model")
            plt.plot(group["spec_d_model"], group["acc_mean"], marker="o", label=label)
            plt.fill_between(
                group["spec_d_model"],
                group["acc_mean"] - group["acc_sem"],
                group["acc_mean"] + group["acc_sem"],
                alpha=0.13,
            )
        plt.xlabel("d_model")
        plt.ylabel("final eval_acc")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig / f"architecture_dim_sweep_seq{int(seq_len)}.png", dpi=180)
        plt.close()
    return out


def aggregate_optimizer(df: pd.DataFrame, curves: pd.DataFrame, agg: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    opt = df[df["spec_group"] == "opt"].copy()
    if opt.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_lr = (
        opt.groupby(["spec_optimizer", "spec_lr"])
        .agg(
            acc_mean=("eval_acc", "mean"),
            acc_sem=("eval_acc", sem),
            full_vocab_mean=("eval_full_vocab_acc", "mean"),
            full_vocab_sem=("eval_full_vocab_acc", sem),
            loss_mean=("eval_loss", "mean"),
            margin_mean=("eval_margin", "mean"),
            head_mean=("eval_head_acc", "mean"),
            tail_mean=("eval_tail_acc", "mean"),
            auc_mean=("auc_acc", "mean"),
            t80_median=("t80_step", "median"),
            seeds=("spec_seed", "nunique"),
            grad_clip=("spec_grad_clip", "first"),
        )
        .reset_index()
    )
    by_lr = by_lr.sort_values(["spec_optimizer", "spec_lr"])
    best = (
        by_lr.sort_values(["acc_mean", "auc_mean", "margin_mean"], ascending=False)
        .groupby("spec_optimizer")
        .head(1)
        .sort_values("spec_optimizer")
    )
    save_table(by_lr, agg / "optimizer_by_lr.csv")
    save_table(best, agg / "optimizer_best_over_lr.csv")

    fig = agg / "figures"
    plt.figure(figsize=(8, 5))
    for optimizer, group in by_lr.groupby("spec_optimizer"):
        group = group.sort_values("spec_lr")
        plt.semilogx(group["spec_lr"], group["acc_mean"], marker="o", label=optimizer)
        plt.fill_between(group["spec_lr"], group["acc_mean"] - group["acc_sem"], group["acc_mean"] + group["acc_sem"], alpha=0.13)
    plt.xlabel("learning rate")
    plt.ylabel("final eval_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig / "optimizer_lr_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for optimizer, group in by_lr.groupby("spec_optimizer"):
        group = group.sort_values("spec_lr")
        plt.semilogx(group["spec_lr"], group["full_vocab_mean"], marker="o", label=optimizer)
        plt.fill_between(
            group["spec_lr"],
            group["full_vocab_mean"] - group["full_vocab_sem"],
            group["full_vocab_mean"] + group["full_vocab_sem"],
            alpha=0.13,
        )
    plt.xlabel("learning rate")
    plt.ylabel("final eval_full_vocab_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig / "optimizer_full_vocab_lr_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    xs = range(len(best))
    plt.bar(list(xs), best["acc_mean"], yerr=best["acc_sem"], capsize=5)
    plt.xticks(list(xs), best["spec_optimizer"])
    plt.ylabel("best-over-LR final eval_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig / "optimizer_best_over_lr_acc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    xs = range(len(best))
    plt.bar(list(xs), best["full_vocab_mean"], yerr=best["full_vocab_sem"], capsize=5)
    plt.xticks(list(xs), best["spec_optimizer"])
    plt.ylabel("best-over-LR final eval_full_vocab_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig / "optimizer_best_over_lr_full_vocab_acc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for _, row in best.iterrows():
        sub = curves[
            (curves["spec_group"] == "opt")
            & (curves["spec_optimizer"] == row["spec_optimizer"])
            & (curves["spec_lr"] == row["spec_lr"])
        ]
        line = sub.groupby("step")["eval_acc"].agg(["mean", sem]).reset_index()
        plt.plot(line["step"], line["mean"], label=f"{row['spec_optimizer']} lr={row['spec_lr']:g}")
        plt.fill_between(line["step"], line["mean"] - line["sem"], line["mean"] + line["sem"], alpha=0.13)
    plt.xlabel("step")
    plt.ylabel("eval_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig / "optimizer_curves_best_lr.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for _, row in best.iterrows():
        sub = curves[
            (curves["spec_group"] == "opt")
            & (curves["spec_optimizer"] == row["spec_optimizer"])
            & (curves["spec_lr"] == row["spec_lr"])
        ]
        line = sub.groupby("step")["eval_full_vocab_acc"].agg(["mean", sem]).reset_index()
        plt.plot(line["step"], line["mean"], label=f"{row['spec_optimizer']} lr={row['spec_lr']:g}")
        plt.fill_between(line["step"], line["mean"] - line["sem"], line["mean"] + line["sem"], alpha=0.13)
    plt.xlabel("step")
    plt.ylabel("eval_full_vocab_acc")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig / "optimizer_full_vocab_curves_best_lr.png", dpi=180)
    plt.close()
    return by_lr, best


def write_report(agg: Path, raw: pd.DataFrame, arch: pd.DataFrame, opt_by_lr: pd.DataFrame, opt_best: pd.DataFrame) -> None:
    lines = [
        "# LKAR Empirical Research Report",
        "",
        "This report separates two questions:",
        "",
        "1. Architecture/pathway: which Transformer structures and trainable parameter subsets can express and learn the associative-memory composition?",
        "2. Optimizer: within one fixed architecture, which optimizer learns it fastest and most reliably?",
        "",
        f"Completed runs: {len(raw)}",
        "",
    ]
    if not arch.empty:
        anchor = arch[(arch["spec_seq_len"] == 32) & (arch["spec_d_model"] == 128)].sort_values("acc_mean", ascending=False)
        lines += [
            "## Architecture Anchor: seq_len=32, d_model=128",
            "",
            "Architecture and parameter-pathway ablations are compared with fixed AdamW lr=1e-3.",
            "",
            "```text",
            anchor.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "```",
            "",
        ]
    if not opt_best.empty:
        lines += [
            "## Optimizer Best Over LR",
            "",
            "All optimizers use the same LR candidate set in this study. Muon uses the candidate as matrix-update LR and keeps auxiliary AdamW LR fixed.",
            "",
            "```text",
            opt_best.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "```",
            "",
            "## Optimizer All LR Means",
            "",
            "```text",
            opt_by_lr.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "```",
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `raw_runs.csv`: one row per completed run.",
        "- `architecture_summary.csv`: architecture x sequence length x model dimension aggregate.",
        "- `optimizer_by_lr.csv`: optimizer x learning-rate aggregate.",
        "- `optimizer_best_over_lr.csv`: optimizer comparison after LR selection.",
        "- Main accuracy columns: `acc_mean` is candidate-value accuracy; `full_vocab_mean` is true full-vocabulary top-1 accuracy.",
        "- `figures/`: unified comparison plots.",
        "",
    ]
    (agg / "RESEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(out_dir: Path, run_roots: list[tuple[Path, set[str] | None]] | None = None) -> None:
    agg = out_dir / "aggregate"
    fig = agg / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    raw, curves = collect(out_dir, run_roots)
    if raw.empty:
        return
    raw.to_csv(agg / "raw_runs.csv", index=False)
    curves.to_csv(agg / "curves.csv", index=False)
    arch = aggregate_architecture(raw, curves, agg)
    opt_by_lr, opt_best = aggregate_optimizer(raw, curves, agg)
    write_report(agg, raw, arch, opt_by_lr, opt_best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="main")
    parser.add_argument("--only", choices=["all", "arch", "opt", "aggregate"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge-from", default="")
    parser.add_argument("--merge-arch-from", default="")
    parser.add_argument("--merge-opt-from", default="")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preset = PRESETS[args.preset]
    if args.only != "aggregate":
        run_study(args, preset)
    merge_roots: list[tuple[Path, set[str] | None]] = []
    merge_roots.extend((Path(p), None) for p in args.merge_from.split(",") if p)
    merge_roots.extend((Path(p), {"arch"}) for p in args.merge_arch_from.split(",") if p)
    merge_roots.extend((Path(p), {"opt"}) for p in args.merge_opt_from.split(",") if p)
    aggregate(args.out_dir, merge_roots or None)


if __name__ == "__main__":
    main()
