"""Non-closure baselines for controlled path QA.

These baselines deliberately answer the endpoint query without writing a dense
closure field that is read once by q^T M.  They are meant to separate the
single-read memory bottleneck from endpoint prediction itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from generic_closure_writer import (
    ClosureTextTokenizer,
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    PathQAExample,
    accuracy_from_logits,
    collate_examples,
    exact_dict_answer,
    iter_minibatches,
)


@dataclass(frozen=True)
class StrongerBaselineConfig:
    seed: int = 9401
    num_entities: int = 48
    num_relations: int = 4
    max_path_len: int = 32
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    train_steps: int = 3000
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    curriculum: str = "staged"
    lr_schedule: str = "constant"
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    eval_n: int = 512
    eval_batch_size: int = 16
    base_distractors: int = 6
    distractors_per_hop: int = 3
    same_relation_branch_prob: float = 0.25
    max_seq_len: int = 900
    scratchpad_loss_weight: float = 1.0
    torch_threads: int = 4


def closure_cfg(cfg: StrongerBaselineConfig) -> ClosureWriterConfig:
    return ClosureWriterConfig(
        seed=cfg.seed,
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        max_path_len=cfg.max_path_len,
        key_dim=max(8, cfg.d_model),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        train_steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        curriculum=cfg.curriculum,
        lr_schedule=cfg.lr_schedule,
        warmup_frac=cfg.warmup_frac,
        min_lr_ratio=cfg.min_lr_ratio,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
        base_distractors=cfg.base_distractors,
        distractors_per_hop=cfg.distractors_per_hop,
        same_relation_branch_prob=cfg.same_relation_branch_prob,
        max_seq_len=cfg.max_seq_len,
        torch_threads=cfg.torch_threads,
    )


class RelativeBiasEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float, max_relative_distance: int = 128) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model) // int(n_heads)
        self.max_relative_distance = int(max_relative_distance)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2 * self.max_relative_distance + 1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        qkv = self.qkv(self.norm1(x)).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        pos = torch.arange(L, device=x.device)
        rel = (pos.view(L, 1) - pos.view(1, L)).clamp(-self.max_relative_distance, self.max_relative_distance)
        scores = scores + self.rel_bias[:, rel + self.max_relative_distance].unsqueeze(0)
        scores = scores.masked_fill(~mask.view(B, 1, 1, L), -1.0e9)
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.dropout(self.out(y))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class RelativeBiasDirectTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: StrongerBaselineConfig) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            RelativeBiasEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_entities)

    def encode(self, input_ids: torch.Tensor, mask: torch.Tensor, read_pos: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        h = x[torch.arange(x.shape[0], device=x.device), read_pos.long()]
        return self.norm(h)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode(batch["input_ids"], batch["mask"], batch["read_pos"])
        return self.head(h)


class HopSupervisedScratchpadTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: StrongerBaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RelativeBiasDirectTransformer(vocab_size, cfg)
        self.answer_head = nn.Linear(cfg.d_model, cfg.num_entities)
        self.hop_head = nn.Linear(cfg.d_model, cfg.max_path_len * cfg.num_entities)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encoder.encode(batch["input_ids"], batch["mask"], batch["read_pos"])
        hop_logits = self.hop_head(h).view(-1, self.cfg.max_path_len, self.cfg.num_entities)
        return {"answer_logits": self.answer_head(h), "hop_logits": hop_logits}


def hop_targets(examples: Sequence[PathQAExample], cfg: StrongerBaselineConfig, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros((len(examples), cfg.max_path_len), dtype=torch.long, device=device)
    mask = torch.zeros((len(examples), cfg.max_path_len), dtype=torch.bool, device=device)
    for i, ex in enumerate(examples):
        hops = tuple(int(x) for x in ex.path_nodes[1:])
        for j, node in enumerate(hops[: cfg.max_path_len]):
            target[i, j] = node
            mask[i, j] = True
    return target, mask


def set_frontier_answer(ex: PathQAExample) -> Optional[int]:
    frontier = {int(ex.source)}
    for rel in ex.relations:
        nxt = {int(t) for s, r, t in ex.edges if int(r) == int(rel) and int(s) in frontier}
        frontier = nxt
        if not frontier:
            return None
    return next(iter(frontier)) if len(frontier) == 1 else None


def graph_recurrent_logits(examples: Sequence[PathQAExample], cfg: StrongerBaselineConfig, device: torch.device | str) -> torch.Tensor:
    logits = torch.zeros((len(examples), cfg.num_entities), dtype=torch.float32, device=device)
    for i, ex in enumerate(examples):
        frontier = {int(ex.source)}
        for rel in ex.relations:
            nxt = {int(t) for s, r, t in ex.edges if int(r) == int(rel) and int(s) in frontier}
            frontier = nxt
            if not frontier:
                break
        if frontier:
            for node in frontier:
                logits[i, int(node)] = 10.0
    return logits


def symbolic_accuracy(examples: Sequence[PathQAExample], fn) -> Tuple[int, int]:
    return sum(int(fn(ex) == ex.target) for ex in examples), len(examples)


def train_choices_for_step(cfg: StrongerBaselineConfig, step: int) -> Tuple[int, ...]:
    if cfg.curriculum == "mixed":
        return (1, 2, 3)
    if step <= max(1, int(0.25 * cfg.train_steps)):
        return (1,)
    if step <= max(1, int(0.55 * cfg.train_steps)):
        return (1, 2)
    return (1, 2, 3)


def lr_for_step(cfg: StrongerBaselineConfig, step: int) -> float:
    base = float(cfg.learning_rate)
    if cfg.lr_schedule == "constant":
        return base
    total = max(1, int(cfg.train_steps))
    warmup = max(1, int(float(cfg.warmup_frac) * total))
    if step <= warmup:
        return base * float(step) / float(warmup)
    progress = (float(step) - float(warmup)) / max(1.0, float(total - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return base * (float(cfg.min_lr_ratio) + (1.0 - float(cfg.min_lr_ratio)) * cosine)


def train_models(cfg: StrongerBaselineConfig, device: torch.device | str = "cpu") -> Tuple[RelativeBiasDirectTransformer, HopSupervisedScratchpadTransformer, ClosureTextTokenizer, Dict[str, object]]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    if cfg.curriculum not in {"staged", "mixed"}:
        raise ValueError(f"unknown curriculum {cfg.curriculum!r}")
    if cfg.lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    base_cfg = closure_cfg(cfg)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    rel = RelativeBiasDirectTransformer(tok.vocab_size, cfg).to(device)
    scratch = HopSupervisedScratchpadTransformer(tok.vocab_size, cfg).to(device)
    params = list(rel.parameters()) + list(scratch.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gen = ControlledDenseGraphTextQAGenerator(base_cfg, seed=cfg.seed + 101)
    rng = random.Random(cfg.seed + 202)
    t0 = time.perf_counter()
    last: Dict[str, float] = {}

    for step in range(1, int(cfg.train_steps) + 1):
        for group in opt.param_groups:
            group["lr"] = lr_for_step(cfg, step)
        lengths = [rng.choice(train_choices_for_step(cfg, step)) for _ in range(cfg.batch_size)]
        examples = [gen.make_example(L) for L in lengths]
        batch = collate_examples(examples, tok, base_cfg, device=device, rng=rng)
        target = batch["target"]
        h_target, h_mask = hop_targets(examples, cfg, device)
        rel_logits = rel(batch)
        scratch_out = scratch(batch)
        loss_rel = F.cross_entropy(rel_logits, target)
        loss_scratch_answer = F.cross_entropy(scratch_out["answer_logits"], target)
        if bool(h_mask.any()):
            loss_hop = F.cross_entropy(scratch_out["hop_logits"][h_mask], h_target[h_mask])
        else:
            loss_hop = torch.zeros((), dtype=loss_rel.dtype, device=device)
        loss = loss_rel + loss_scratch_answer + float(cfg.scratchpad_loss_weight) * loss_hop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if step == 1 or step % max(1, cfg.train_steps // 6) == 0 or step == cfg.train_steps:
            with torch.no_grad():
                c_rel, n = accuracy_from_logits(rel_logits, target)
                c_scratch, _ = accuracy_from_logits(scratch_out["answer_logits"], target)
                hop_pred = scratch_out["hop_logits"].argmax(dim=-1)
                hop_acc = float(hop_pred[h_mask].eq(h_target[h_mask]).float().mean().item()) if bool(h_mask.any()) else float("nan")
            last = {
                "step": float(step),
                "loss_relative_direct": float(loss_rel.detach().item()),
                "loss_scratchpad_answer": float(loss_scratch_answer.detach().item()),
                "loss_scratchpad_hop": float(loss_hop.detach().item()),
                "train_batch_acc_relative_direct": c_rel / max(1, n),
                "train_batch_acc_scratchpad_answer": c_scratch / max(1, n),
                "train_batch_acc_scratchpad_hop": hop_acc,
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "elapsed_sec": time.perf_counter() - t0,
            }
            print(json.dumps({"stronger_baseline_train_progress": last}, sort_keys=True), flush=True)

    meta = {
        "config": asdict(cfg),
        "final_train_snapshot": last,
        "num_parameters": {
            "relative_bias_direct": sum(p.numel() for p in rel.parameters()),
            "hop_supervised_scratchpad": sum(p.numel() for p in scratch.parameters()),
        },
        "elapsed_train_sec": time.perf_counter() - t0,
    }
    return rel, scratch, tok, meta


@torch.no_grad()
def evaluate_by_length(
    cfg: StrongerBaselineConfig,
    rel: RelativeBiasDirectTransformer,
    scratch: HopSupervisedScratchpadTransformer,
    tok: ClosureTextTokenizer,
    *,
    lengths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    device = torch.device(device)
    rel.eval()
    scratch.eval()
    base_cfg = closure_cfg(cfg)
    rows: List[Dict[str, float]] = []
    rng = random.Random(cfg.seed + 303)
    for L in lengths:
        if int(L) > int(cfg.max_path_len):
            continue
        gen = ControlledDenseGraphTextQAGenerator(base_cfg, seed=cfg.seed + 1000 + int(L))
        examples = gen.make_examples(int(L), cfg.eval_n)
        row_counts: Dict[str, List[int]] = {}

        def add(name: str, c: int, n: int) -> None:
            if name not in row_counts:
                row_counts[name] = [0, 0]
            row_counts[name][0] += int(c)
            row_counts[name][1] += int(n)

        add("dp_bfs_oracle", *symbolic_accuracy(examples, exact_dict_answer))
        add("iterative_pointer", *symbolic_accuracy(examples, set_frontier_answer))

        for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
            batch = collate_examples(batch_examples, tok, base_cfg, device=device, rng=rng)
            target = batch["target"]
            rel_logits = rel(batch)
            scratch_out = scratch(batch)
            graph_logits = graph_recurrent_logits(batch_examples, cfg, device)
            add("relative_transformer", *accuracy_from_logits(rel_logits, target))
            add("scratchpad_answer", *accuracy_from_logits(scratch_out["answer_logits"], target))
            h_target, h_mask = hop_targets(batch_examples, cfg, device)
            hop_pred = scratch_out["hop_logits"].argmax(dim=-1)
            if bool(h_mask.any()):
                add("scratchpad_hop", int(hop_pred[h_mask].eq(h_target[h_mask]).sum().item()), int(h_mask.sum().item()))
            add("graph_recurrent", *accuracy_from_logits(graph_logits, target))

        attempts = [ex.attempts for ex in examples]
        row: Dict[str, float] = {
            "length": float(L),
            "n": float(len(examples)),
            "ambiguous_rejection_mean_attempts": float(sum(attempts) / max(1, len(attempts))),
            "ambiguous_rejection_max_attempts": float(max(attempts) if attempts else 0),
        }
        for name, (c, n) in sorted(row_counts.items()):
            row[f"{name}_acc"] = float(c / n) if n else float("nan")
            row[f"{name}_n"] = float(n)
        rows.append(row)
        printable = {k: row[k] for k in row if k.endswith("_acc") or k in {"length", "n"}}
        print(json.dumps({"stronger_baseline_eval_by_length": printable}, sort_keys=True), flush=True)
    return rows


def write_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "STRONGER_BASELINES_RESULTS.json"
    csv_path = out_dir / "STRONGER_BASELINES_RESULTS.csv"
    report_path = out_dir / "STRONGER_BASELINES_REPORT.md"
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    cols = [
        "length", "n", "relative_transformer_acc", "scratchpad_answer_acc", "scratchpad_hop_acc",
        "graph_recurrent_acc", "iterative_pointer_acc", "dp_bfs_oracle_acc",
    ]
    lines = [
        "# Stronger non-closure baselines",
        "",
        "These baselines answer endpoint queries without emitting a closure field read by q^T M.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, float("nan"))
            if col in {"length", "n"}:
                vals.append(str(int(value)))
            else:
                vals.append("NA" if isinstance(value, float) and math.isnan(value) else f"{float(value):.3f}")
        lines.append("| " + " | ".join(vals) + " |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(cfg: StrongerBaselineConfig, out_dir: Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    t0 = time.perf_counter()
    rel, scratch, tok, meta = train_models(cfg, device=device)
    rows = evaluate_by_length(cfg, rel, scratch, tok, device=device)
    meta = dict(meta)
    meta["elapsed_total_sec"] = time.perf_counter() - t0
    paths = write_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate non-closure baselines.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=StrongerBaselineConfig.seed)
    p.add_argument("--num-entities", type=int, default=StrongerBaselineConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=StrongerBaselineConfig.num_relations)
    p.add_argument("--max-path-len", type=int, default=StrongerBaselineConfig.max_path_len)
    p.add_argument("--d-model", type=int, default=StrongerBaselineConfig.d_model)
    p.add_argument("--heads", type=int, default=StrongerBaselineConfig.n_heads)
    p.add_argument("--layers", type=int, default=StrongerBaselineConfig.n_layers)
    p.add_argument("--train-steps", type=int, default=StrongerBaselineConfig.train_steps)
    p.add_argument("--batch-size", type=int, default=StrongerBaselineConfig.batch_size)
    p.add_argument("--eval-n", type=int, default=StrongerBaselineConfig.eval_n)
    p.add_argument("--eval-batch-size", type=int, default=StrongerBaselineConfig.eval_batch_size)
    p.add_argument("--learning-rate", type=float, default=StrongerBaselineConfig.learning_rate)
    p.add_argument("--curriculum", type=str, choices=["staged", "mixed"], default=StrongerBaselineConfig.curriculum)
    p.add_argument("--lr-schedule", type=str, choices=["constant", "cosine"], default=StrongerBaselineConfig.lr_schedule)
    p.add_argument("--warmup-frac", type=float, default=StrongerBaselineConfig.warmup_frac)
    p.add_argument("--min-lr-ratio", type=float, default=StrongerBaselineConfig.min_lr_ratio)
    p.add_argument("--scratchpad-loss-weight", type=float, default=StrongerBaselineConfig.scratchpad_loss_weight)
    p.add_argument("--threads", type=int, default=StrongerBaselineConfig.torch_threads)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrongerBaselineConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        max_path_len=args.max_path_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        curriculum=args.curriculum,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        min_lr_ratio=args.min_lr_ratio,
        scratchpad_loss_weight=args.scratchpad_loss_weight,
        torch_threads=args.threads,
    )
    result = run_experiment(cfg, Path(args.out_dir), device=args.device)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
