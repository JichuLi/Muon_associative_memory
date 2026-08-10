# LKAR Empirical Benchmark

Clean empirical benchmark for **Latent-Key Associative Recall** with real
causal Transformer next-token training.

## Task

Each episode is a sequence:

```text
[BOS] [K_1] [V_1] ... [K_m] [V_m] [noise ...] [Q] -> predict [V*]
```

The hidden cross-episode memory is fixed:

```text
Q -> K*
```

The in-context dictionary is resampled every episode:

```text
K_i -> V_i
```

So the model must combine parametric memory `Q -> K*` with context memory
`K -> V`.

## Main Study

The current study has two separated blocks.

Architecture/pathway block:

- architecture variants: `1L-attn`, `2L-attn`, `2L-full`, `4L-full`,
  `2L-no-pos`
- parameter-pathway ablations on `2L-attn`: `2L-fixed-embed`,
  `2L-embed-only`, `2L-attn-only`, `2L-qk-only`, `2L-freeze-QK`
- `seq_len = [16, 32, 64]`
- `d_model = [64, 128, 256]`
- fixed optimizer family: AdamW
- fixed AdamW `lr = 1e-3`
- `grad_clip = 1.0`

Optimizer block:

- fixed architecture: `2L-attn`
- fixed scale: `seq_len = 32`, `d_model = 128`
- optimizers: AdamW, Muon, NGD, SGD
- identical log-scale LR candidate set for all optimizers
- compared by best-over-LR
- `grad_clip = 0.0`

The shortest sequence uses `m=7`, so:

```text
1 BOS + 7 key tokens + 7 value tokens + 1 query token = 16 tokens
```

Random candidate baseline is `1/7 = 14.3%`.

## Run

```powershell
python .\lkar_empirical_benchmark\study.py `
  --out-dir .\lkar_empirical_benchmark\results\main_v3 `
  --preset main `
  --only all
```

Resume is automatic: completed runs with `summary.csv` are skipped unless
`--force` is passed.

## Outputs

```text
results/main_v3/
  runs/
  aggregate/
    raw_runs.csv
    curves.csv
    architecture_summary.csv
    optimizer_by_lr.csv
    optimizer_best_over_lr.csv
    RESEARCH_REPORT.md
    figures/
```

Primary metrics:

- `eval_acc`: accuracy among candidate values in the current context.
- `eval_full_vocab_acc`: true top-1 accuracy over the entire vocabulary.
- `eval_loss`: answer-token next-token cross entropy.
- `eval_margin`: correct candidate logit minus strongest decoy candidate logit.
- `eval_head_acc`: accuracy on the most frequent query quartile.
- `eval_tail_acc`: accuracy on the rarest query quartile.
- `auc_acc`: area under the accuracy curve.
- `t80_step`: first step reaching `eval_acc >= 0.80`.
