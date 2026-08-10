from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    n_q: int = 128
    n_k: int = 128
    n_v: int = 512
    n_noise: int = 64
    m: int = 8
    noise_tokens: int = 16
    seq_len: int = 0
    zipf_alpha: float = 1.1
    batch_size: int = 128
    steps: int = 1200
    lr: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    muon_lr: float = 0.02
    aux_lr: float = 3e-4
    muon_momentum: float = 0.95
    grad_clip: float = 1.0
    d_model: int = 128
    n_heads: int = 4
    variant: str = "two_layer_no_mlp"
    train_scope: str = "all"
    eval_repeats: int = 4
    eval_every: int = 100
    seed: int = 0
    device: str = "auto"


PRESETS = {
    "smoke": dict(
        n_q=64,
        n_k=64,
        n_v=256,
        n_noise=32,
        m=4,
        noise_tokens=8,
        batch_size=128,
        steps=300,
        d_model=64,
        n_heads=4,
        eval_every=50,
        eval_repeats=4,
    ),
    "core": dict(),
}


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def zipf_probs(n: int, alpha: float, device: torch.device) -> torch.Tensor:
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    probs = torch.ones_like(ranks) if alpha == 0 else ranks.pow(-alpha)
    return probs / probs.sum()


def make_permutation(n_q: int, n_k: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randperm(n_k, generator=gen, device=device)[:n_q]


@dataclass(frozen=True)
class Vocab:
    n_q: int
    n_k: int
    n_v: int
    n_noise: int

    @property
    def bos(self) -> int:
        return 0

    @property
    def q0(self) -> int:
        return 1

    @property
    def k0(self) -> int:
        return self.q0 + self.n_q

    @property
    def v0(self) -> int:
        return self.k0 + self.n_k

    @property
    def noise0(self) -> int:
        return self.v0 + self.n_v

    @property
    def size(self) -> int:
        return self.noise0 + self.n_noise


@dataclass
class Batch:
    tokens: torch.Tensor
    target: torch.Tensor
    values: torch.Tensor
    target_slot: torch.Tensor
    q: torch.Tensor


class Sampler:
    def __init__(self, cfg: Config, vocab: Vocab, pi: torch.Tensor, probs: torch.Tensor, device: torch.device):
        self.cfg = cfg
        self.vocab = vocab
        self.pi = pi
        self.probs = probs
        self.device = device
        min_prompt_len = 1 + 2 * cfg.m + 1
        if cfg.seq_len:
            if cfg.seq_len < min_prompt_len:
                raise ValueError(f"seq_len={cfg.seq_len} is too short; need at least {min_prompt_len}")
            self.prompt_len = cfg.seq_len
            self.noise_tokens = cfg.seq_len - min_prompt_len
        else:
            self.noise_tokens = cfg.noise_tokens
            self.prompt_len = min_prompt_len + cfg.noise_tokens
        base_gap = self.noise_tokens // (cfg.m + 1)
        extra = self.noise_tokens % (cfg.m + 1)
        self.gaps = [base_gap + (1 if i < extra else 0) for i in range(cfg.m + 1)]

    def queries(self, batch_size: int) -> torch.Tensor:
        return torch.multinomial(self.probs, batch_size, replacement=True)

    def unique_decoys(self, correct: torch.Tensor) -> torch.Tensor:
        batch = correct.numel()
        out = torch.randint(self.cfg.n_k, (batch, self.cfg.m - 1), device=self.device)
        for _ in range(12):
            bad = out.eq(correct[:, None])
            for j in range(1, out.shape[1]):
                bad[:, j] |= out[:, j, None].eq(out[:, :j]).any(dim=1)
            if not bool(bad.any()):
                return out
            out[bad] = torch.randint(self.cfg.n_k, (int(bad.sum()),), device=self.device)
        for row in bad.any(dim=1).nonzero(as_tuple=False).flatten().tolist():
            perm = torch.randperm(self.cfg.n_k, device=self.device)
            out[row] = perm[perm != correct[row]][: self.cfg.m - 1]
        return out

    def unique_values(self, batch_size: int) -> torch.Tensor:
        out = torch.randint(self.cfg.n_v, (batch_size, self.cfg.m), device=self.device)
        for _ in range(12):
            bad = torch.zeros_like(out, dtype=torch.bool)
            for j in range(1, self.cfg.m):
                bad[:, j] |= out[:, j, None].eq(out[:, :j]).any(dim=1)
            if not bool(bad.any()):
                return out + self.vocab.v0
            out[bad] = torch.randint(self.cfg.n_v, (int(bad.sum()),), device=self.device)
        for row in bad.any(dim=1).nonzero(as_tuple=False).flatten().tolist():
            out[row] = torch.randperm(self.cfg.n_v, device=self.device)[: self.cfg.m]
        return out + self.vocab.v0

    def batch(self, batch_size: int, uniform_eval: bool = False, repeats: int = 1) -> Batch:
        if uniform_eval:
            q = torch.arange(self.cfg.n_q, device=self.device).repeat_interleave(repeats)
            batch_size = q.numel()
        else:
            q = self.queries(batch_size)
        correct = self.pi[q]
        decoys = self.unique_decoys(correct)
        keys = torch.cat([correct[:, None], decoys], dim=1)
        perm = torch.argsort(torch.rand(batch_size, self.cfg.m, device=self.device), dim=1)
        keys = keys.gather(1, perm)
        target_slot = perm.eq(0).to(torch.long).argmax(dim=1)
        values = self.unique_values(batch_size)
        target = values.gather(1, target_slot[:, None]).squeeze(1)

        tokens = torch.empty(batch_size, self.prompt_len, dtype=torch.long, device=self.device)
        tokens[:, 0] = self.vocab.bos
        cursor = 1
        for j in range(self.cfg.m):
            gap = self.gaps[j]
            if gap:
                tokens[:, cursor : cursor + gap] = torch.randint(
                    self.cfg.n_noise, (batch_size, gap), device=self.device
                ) + self.vocab.noise0
                cursor += gap
            tokens[:, cursor] = keys[:, j] + self.vocab.k0
            cursor += 1
            tokens[:, cursor] = values[:, j]
            cursor += 1
        gap = self.gaps[self.cfg.m]
        if gap:
            tokens[:, cursor : cursor + gap] = torch.randint(
                self.cfg.n_noise, (batch_size, gap), device=self.device
            ) + self.vocab.noise0
            cursor += gap
        tokens[:, cursor] = q + self.vocab.q0
        return Batch(tokens=tokens, target=target, values=values, target_slot=target_slot, q=q)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:t, :t], torch.finfo(scores.dtype).min)
        y = scores.softmax(dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, mlp: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)) if mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int, variant: str):
        super().__init__()
        variants = {
            "one_layer": (1, False, True, True),
            "one_layer_attn": (1, False, True, True),
            "two_layer_no_mlp": (2, False, True, True),
            "two_layer_attn": (2, False, True, True),
            "two_layer": (2, True, True, True),
            "two_layer_full": (2, True, True, True),
            "full_transformer": (4, True, True, True),
            "two_layer_no_pos": (2, False, False, True),
            "two_layer_untied": (2, False, True, False),
        }
        if variant not in variants:
            raise ValueError(f"unknown variant: {variant}")
        n_layers, mlp, use_pos, tie_head = variants[variant]
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.blocks = nn.ModuleList([Block(d_model, n_heads, max_len, mlp) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_head:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens)
        if self.pos is not None:
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def polar(g: torch.Tensor) -> torch.Tensor:
    if g.ndim != 2:
        return g / g.norm().clamp_min(1e-12)
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(1e-12)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class NGD:
    def __init__(self, params: Iterable[nn.Parameter], lr: float, wd: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.wd = wd

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    @torch.no_grad()
    def step(self) -> None:
        total = None
        for p in self.params:
            if p.grad is not None:
                val = p.grad.pow(2).sum()
                total = val if total is None else total + val
        if total is None:
            return
        denom = total.sqrt().clamp_min(1e-12)
        for p in self.params:
            if p.grad is not None:
                if self.wd:
                    p.mul_(1.0 - self.lr * self.wd)
                p.add_(p.grad / denom, alpha=-self.lr)


class MuonHybrid:
    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], muon_lr: float, aux_lr: float, wd: float, momentum: float):
        matrix, aux = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            (matrix if name.startswith("blocks.") and p.ndim == 2 else aux).append(p)
        self.matrix = matrix
        self.muon_lr = muon_lr
        self.wd = wd
        self.momentum = momentum
        self.buffers = {id(p): torch.zeros_like(p) for p in matrix}
        self.aux = torch.optim.AdamW(aux, lr=aux_lr, weight_decay=wd) if aux else None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.matrix:
            p.grad = None if set_to_none else torch.zeros_like(p)
        if self.aux:
            self.aux.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for p in self.matrix:
            if p.grad is None:
                continue
            if self.wd:
                p.mul_(1.0 - self.muon_lr * self.wd)
            buf = self.buffers[id(p)]
            buf.mul_(self.momentum).add_(p.grad)
            p.add_(polar(buf), alpha=-self.muon_lr)
        if self.aux:
            self.aux.step()


def make_optimizer(model: GPT, cfg: Config):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "ngd":
        return NGD(model.parameters(), cfg.lr, cfg.weight_decay)
    if cfg.optimizer == "muon":
        return MuonHybrid(model.named_parameters(), cfg.muon_lr, cfg.aux_lr, cfg.weight_decay, cfg.muon_momentum)
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def apply_train_scope(model: GPT, scope: str) -> None:
    if scope == "all":
        return
    if scope in {"freeze_qk", "no_qk_train"}:
        for name, p in model.named_parameters():
            if ".attn.q_proj." in name or ".attn.k_proj." in name:
                p.requires_grad_(False)
        return
    if scope in {"freeze_embeddings", "no_embedding_train", "fixed_embeddings"}:
        for name, p in model.named_parameters():
            if name.startswith("tok.") or name.startswith("pos.") or name.startswith("head."):
                p.requires_grad_(False)
        return
    if scope in {"embedding_only", "emb_only"}:
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if name.startswith("tok.") or name.startswith("pos.") or name.startswith("head."):
                p.requires_grad_(True)
        return
    if scope == "qk_only":
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if ".attn.q_proj." in name or ".attn.k_proj." in name:
                p.requires_grad_(True)
        return
    if scope in {"attn_only", "attention_only"}:
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if ".attn." in name:
                p.requires_grad_(True)
        return
    raise ValueError(f"unknown train_scope: {scope}")


def parameter_counts(model: GPT) -> dict[str, int]:
    return {
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def metrics(logits: torch.Tensor, batch: Batch, vocab: Vocab) -> dict[str, float]:
    last = logits[:, -1, :]
    cand = last.gather(1, batch.values)
    pred = cand.argmax(dim=1)
    full_pred = last.argmax(dim=1)
    target_score = cand.gather(1, batch.target_slot[:, None]).squeeze(1)
    masked = cand.clone()
    masked.scatter_(1, batch.target_slot[:, None], -torch.inf)
    return {
        "loss": float(F.cross_entropy(last, batch.target).detach().cpu()),
        "acc": float(pred.eq(batch.target_slot).float().mean().detach().cpu()),
        "full_vocab_acc": float(full_pred.eq(batch.target).float().mean().detach().cpu()),
        "margin": float((target_score - masked.max(dim=1).values).mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: GPT, sampler: Sampler, vocab: Vocab, cfg: Config) -> dict[str, float]:
    model.eval()
    batch = sampler.batch(cfg.n_q * cfg.eval_repeats, uniform_eval=True, repeats=cfg.eval_repeats)
    logits = model(batch.tokens)
    out = {f"eval_{k}": v for k, v in metrics(logits, batch, vocab).items()}
    cand = logits[:, -1, :].gather(1, batch.values)
    ok = cand.argmax(dim=1).eq(batch.target_slot).float().view(cfg.n_q, cfg.eval_repeats).mean(dim=1)
    order = torch.argsort(sampler.probs, descending=True)
    bins = torch.chunk(order, 4)
    out["eval_head_acc"] = float(ok[bins[0]].mean().cpu())
    out["eval_tail_acc"] = float(ok[bins[-1]].mean().cpu())
    model.train()
    return out


def budget(cfg: Config, prompt_len: int) -> dict[str, float | int]:
    return {
        "prompt_len": prompt_len,
        "train_episodes": cfg.batch_size * cfg.steps,
        "train_tokens": cfg.batch_size * cfg.steps * (prompt_len + 1),
        "eval_episodes_per_eval": cfg.n_q * cfg.eval_repeats,
        "random_baseline": 1.0 / cfg.m,
    }


def train(cfg: Config, out_dir: Path) -> pd.DataFrame:
    device = choose_device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    vocab = Vocab(cfg.n_q, cfg.n_k, cfg.n_v, cfg.n_noise)
    pi = make_permutation(cfg.n_q, cfg.n_k, cfg.seed + 37, device)
    sampler = Sampler(cfg, vocab, pi, zipf_probs(cfg.n_q, cfg.zipf_alpha, device), device)
    model = GPT(vocab.size, sampler.prompt_len, cfg.d_model, cfg.n_heads, cfg.variant).to(device)
    apply_train_scope(model, cfg.train_scope)
    counts = parameter_counts(model)
    opt = make_optimizer(model, cfg)
    rows = []
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step > 0:
            batch = sampler.batch(cfg.batch_size)
            loss = F.cross_entropy(model(batch.tokens)[:, -1, :], batch.target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            row = evaluate(model, sampler, vocab, cfg)
            row.update(
                step=step,
                variant=cfg.variant,
                train_scope=cfg.train_scope,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                seq_len=sampler.prompt_len,
                n_q=cfg.n_q,
                n_k=cfg.n_k,
                n_v=cfg.n_v,
                m=cfg.m,
                optimizer=cfg.optimizer,
                lr=cfg.lr,
                muon_lr=cfg.muon_lr,
                grad_clip=cfg.grad_clip,
                seed=cfg.seed,
                elapsed_s=time.perf_counter() - t0,
                **counts,
                **budget(cfg, sampler.prompt_len),
            )
            rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), "pi_star": pi.cpu()}, out_dir / "model.pt")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    final = df.tail(1)
    final.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_metadata.json").write_text(
        json.dumps({"config": asdict(cfg), "budget": budget(cfg, sampler.prompt_len), "parameters": counts}, indent=2),
        encoding="utf-8",
    )
    return df


def plot_single(df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for col, ylabel in [
        ("eval_acc", "candidate accuracy"),
        ("eval_full_vocab_acc", "full-vocab accuracy"),
        ("eval_loss", "loss"),
        ("eval_margin", "candidate margin"),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(df["step"], df[col], marker="o")
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{col}.png", dpi=160)
        plt.close()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    for key, val in PRESETS.get(args.preset, {}).items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, val)
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["custom", "smoke", "core"], default="custom")
    p.add_argument("--out-dir", type=Path, required=True)
    for field, value in Config().__dict__.items():
        if field in {"variant", "optimizer", "device", "train_scope"}:
            p.add_argument(f"--{field.replace('_', '-')}", type=str, default=value)
        elif isinstance(value, int):
            p.add_argument(f"--{field.replace('_', '-')}", type=int, default=value)
        else:
            p.add_argument(f"--{field.replace('_', '-')}", type=float, default=value)
    return p


parser = build_parser()


def main() -> None:
    args = apply_preset(parser.parse_args())
    cfg = Config(
        n_q=args.n_q,
        n_k=args.n_k,
        n_v=args.n_v,
        n_noise=args.n_noise,
        m=args.m,
        noise_tokens=args.noise_tokens,
        seq_len=args.seq_len,
        zipf_alpha=args.zipf_alpha,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        aux_lr=args.aux_lr,
        muon_momentum=args.muon_momentum,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        n_heads=args.n_heads,
        variant=args.variant,
        train_scope=args.train_scope,
        eval_repeats=args.eval_repeats,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
    )
    df = train(cfg, args.out_dir)
    plot_single(df, args.out_dir)
    print(df.tail(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
