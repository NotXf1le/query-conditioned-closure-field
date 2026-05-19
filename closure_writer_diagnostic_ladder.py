"""Diagnostic ladder for closure-field writing.

This experiment separates four questions that the original closure-writer
baseline conflated:

* can a model write a known value under a known key?
* can it write a gold target from controlled text?
* can it ground a one-hop fact and write it into the field?
* can it construct a multi-hop closure field?

The script intentionally reuses the existing controlled generator,
HolographicClosureField, and tokenizer so its outputs are comparable with the
main closure-writer results.
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
    HolographicClosureField,
    PathQAExample,
    TransformerBackbone,
    accuracy_from_logits,
    iter_minibatches,
)


LADDER_RESULT_JSON = "CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.json"
LADDER_RESULT_CSV = "CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.csv"
LADDER_REPORT = "CLOSURE_WRITER_DIAGNOSTIC_LADDER_REPORT.md"
DEFAULT_RUNGS = "direct_qv_write,gold_target_write,one_hop_fact_write,multi_hop_closure_write"
VALID_RUNGS = {"direct_qv_write", "gold_target_write", "one_hop_fact_write", "multi_hop_closure_write"}
VALID_VARIANTS = {"baseline", "key_conditioned", "tied_key"}


@dataclass(frozen=True)
class LadderConfig:
    seed: int = 9701
    num_entities: int = 48
    num_relations: int = 4
    max_path_len: int = 32
    key_dim: int = 96
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_steps: int = 500
    batch_size: int = 128
    eval_n: int = 256
    eval_batch_size: int = 64
    base_distractors: int = 6
    distractors_per_hop: int = 3
    same_relation_branch_prob: float = 0.25
    max_seq_len: int = 900
    curriculum: str = "staged"
    lr_schedule: str = "constant"
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    torch_threads: int = 4
    writer_variant: str = "baseline"
    ladder_rungs: Tuple[str, ...] = ("direct_qv_write", "gold_target_write", "one_hop_fact_write", "multi_hop_closure_write")
    eval_lengths: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def closure_cfg(cfg: LadderConfig) -> ClosureWriterConfig:
    return ClosureWriterConfig(
        seed=cfg.seed,
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        max_path_len=cfg.max_path_len,
        key_dim=cfg.key_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        train_steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
        base_distractors=cfg.base_distractors,
        distractors_per_hop=cfg.distractors_per_hop,
        same_relation_branch_prob=cfg.same_relation_branch_prob,
        max_seq_len=cfg.max_seq_len,
        curriculum=cfg.curriculum,
        lr_schedule=cfg.lr_schedule,
        warmup_frac=cfg.warmup_frac,
        min_lr_ratio=cfg.min_lr_ratio,
        torch_threads=cfg.torch_threads,
    )


def parse_csv_list(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def rank_one_target_memory(query_key: torch.Tensor, target: torch.Tensor, num_entities: int) -> torch.Tensor:
    """Return exact rank-one memory M=q⊗one_hot(target)."""
    bsz, key_dim = query_key.shape
    mem = torch.zeros(bsz, key_dim, int(num_entities), dtype=query_key.dtype, device=query_key.device)
    mem.scatter_add_(2, target.view(bsz, 1, 1).expand(bsz, key_dim, 1), query_key.unsqueeze(-1))
    return mem


def collate_ladder_examples(
    examples: Sequence[PathQAExample],
    tok: ClosureTextTokenizer,
    cfg: ClosureWriterConfig,
    *,
    rung: str,
    device: torch.device | str,
) -> Dict[str, torch.Tensor | List[PathQAExample]]:
    seqs: List[List[int]] = []
    sources: List[int] = []
    lengths: List[int] = []
    targets: List[int] = []
    q_rels: List[List[int]] = []
    include_gold_target = rung == "gold_target_write"

    for ex in examples:
        tokens = [tok.tok("<bos>")]
        if rung != "direct_qv_write":
            for s, r, t in ex.edges:
                tokens.extend([tok.tok("<fact>"), tok.ent(s), tok.rel(r), tok.ent(t), tok.tok(";")])
            tokens.extend([tok.tok("<query>"), tok.ent(ex.source), tok.tok("follow")])
            for j, r in enumerate(ex.relations):
                if j > 0:
                    tokens.append(tok.tok("then"))
                tokens.append(tok.rel(r))
        tokens.append(tok.tok("answer"))
        if include_gold_target:
            tokens.append(tok.ent(ex.target))
        tokens.append(tok.tok("<read>"))
        if len(tokens) > cfg.max_seq_len:
            raise RuntimeError(f"sequence too long: {len(tokens)} > {cfg.max_seq_len}")
        seqs.append(tokens)
        sources.append(int(ex.source))
        lengths.append(len(ex.relations))
        targets.append(int(ex.target))
        q_rels.append((list(ex.relations) + [0] * cfg.max_path_len)[: cfg.max_path_len])

    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    read_pos = torch.zeros(len(seqs), dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
        read_pos[i] = len(seq) - 1

    dev = torch.device(device)
    return {
        "input_ids": ids.to(dev),
        "mask": mask.to(dev),
        "read_pos": read_pos.to(dev),
        "source": torch.tensor(sources, dtype=torch.long, device=dev),
        "q_rels": torch.tensor(q_rels, dtype=torch.long, device=dev),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=dev),
        "target": torch.tensor(targets, dtype=torch.long, device=dev),
        "examples": list(examples),
    }


class LadderTransformerWriter(nn.Module):
    def __init__(self, vocab_size: int, cfg: LadderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ccfg = closure_cfg(cfg)
        self.backbone = TransformerBackbone(vocab_size, ccfg)
        self.direct_feature = nn.Sequential(
            nn.Linear(cfg.key_dim + cfg.num_entities, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.key_fusion = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.key_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.memory_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.key_dim * cfg.num_entities),
        )

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, rung: str) -> Dict[str, torch.Tensor]:
        q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
        if rung == "direct_qv_write":
            one_hot = F.one_hot(batch["target"], num_classes=self.cfg.num_entities).to(q.dtype)
            h = self.direct_feature(torch.cat([q, one_hot], dim=-1))
        else:
            h = self.backbone(batch["input_ids"], batch["mask"], batch["read_pos"])
            if self.cfg.writer_variant == "key_conditioned":
                h = self.key_fusion(torch.cat([h, q.to(h.dtype)], dim=-1))
        mem = self.memory_head(h).view(-1, self.cfg.key_dim, self.cfg.num_entities)
        return {"logits": field.read(mem, q), "memory": mem, "query_key": q}


class LadderMLPWriter(nn.Module):
    def __init__(self, vocab_size: int, cfg: LadderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.direct_feature = nn.Sequential(
            nn.Linear(cfg.key_dim + cfg.num_entities, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.key_fusion = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.key_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.key_dim * cfg.num_entities),
        )

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, rung: str) -> Dict[str, torch.Tensor]:
        q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
        if rung == "direct_qv_write":
            one_hot = F.one_hot(batch["target"], num_classes=self.cfg.num_entities).to(q.dtype)
            h = self.direct_feature(torch.cat([q, one_hot], dim=-1))
        else:
            x = self.emb(batch["input_ids"])
            m = batch["mask"].to(x.dtype).unsqueeze(-1)
            h = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            if self.cfg.writer_variant == "key_conditioned":
                h = self.key_fusion(torch.cat([h, q.to(h.dtype)], dim=-1))
        mem = self.net(h).view(-1, self.cfg.key_dim, self.cfg.num_entities)
        return {"logits": field.read(mem, q), "memory": mem, "query_key": q}


class LadderDirectEndpoint(nn.Module):
    def __init__(self, vocab_size: int, cfg: LadderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(vocab_size, closure_cfg(cfg))
        self.direct_feature = nn.Sequential(
            nn.Linear(cfg.key_dim + cfg.num_entities, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_entities))

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, rung: str) -> torch.Tensor:
        if rung == "direct_qv_write":
            q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
            one_hot = F.one_hot(batch["target"], num_classes=self.cfg.num_entities).to(q.dtype)
            h = self.direct_feature(torch.cat([q, one_hot], dim=-1))
        else:
            h = self.backbone(batch["input_ids"], batch["mask"], batch["read_pos"])
        return self.head(h)


def apply_tied_key_embeddings(model: nn.Module, tok: ClosureTextTokenizer, field: HolographicClosureField, cfg: LadderConfig) -> None:
    emb = getattr(getattr(model, "backbone", None), "emb", None)
    if emb is None:
        emb = getattr(model, "emb", None)
    if emb is None:
        return
    width = min(int(cfg.d_model), int(cfg.key_dim))
    with torch.no_grad():
        emb.weight.zero_()
        for ent in range(cfg.num_entities):
            emb.weight[tok.ent(ent), :width] = field.entity_code[ent, :width].to(emb.weight.dtype)
        rel_code = field.relpos_code[:, :, :width].mean(dim=0)
        for rel in range(cfg.num_relations):
            emb.weight[tok.rel(rel), :width] = rel_code[rel].to(emb.weight.dtype)


def train_choices_for_step(cfg: LadderConfig, step: int) -> Tuple[int, ...]:
    if cfg.curriculum == "mixed":
        return (1, 2, 3)
    if step < max(2, cfg.train_steps // 3):
        return (1,)
    if step < max(3, 2 * cfg.train_steps // 3):
        return (1, 2)
    return (1, 2, 3)


def lr_for_step(cfg: LadderConfig, step: int) -> float:
    if cfg.lr_schedule == "constant":
        return float(cfg.learning_rate)
    warmup = max(1, int(cfg.train_steps * cfg.warmup_frac))
    if step <= warmup:
        return float(cfg.learning_rate) * step / warmup
    progress = (step - warmup) / max(1, cfg.train_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(cfg.learning_rate) * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def rung_train_length(cfg: LadderConfig, rung: str, step: int, rng: random.Random) -> int:
    if rung in {"gold_target_write", "one_hop_fact_write"}:
        return 1
    if rung == "direct_qv_write":
        return rng.choice(train_choices_for_step(cfg, step))
    return rng.choice(train_choices_for_step(cfg, step))


def train_models(
    cfg: LadderConfig,
    device: torch.device | str = "cpu",
) -> Tuple[LadderTransformerWriter, LadderMLPWriter, LadderDirectEndpoint, HolographicClosureField, ClosureTextTokenizer, Dict[str, object]]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    if cfg.writer_variant not in VALID_VARIANTS:
        raise ValueError(f"unknown writer_variant {cfg.writer_variant!r}")
    unknown = set(cfg.ladder_rungs) - VALID_RUNGS
    if unknown:
        raise ValueError(f"unknown ladder rungs: {sorted(unknown)}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    ccfg = closure_cfg(cfg)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(dev)
    transformer = LadderTransformerWriter(tok.vocab_size, cfg).to(dev)
    mlp = LadderMLPWriter(tok.vocab_size, cfg).to(dev)
    direct = LadderDirectEndpoint(tok.vocab_size, cfg).to(dev)
    models: List[nn.Module] = [transformer, mlp, direct]
    if cfg.writer_variant == "tied_key":
        for model in models:
            apply_tied_key_embeddings(model, tok, field, cfg)
    params: List[nn.Parameter] = []
    for model in models:
        params.extend(list(model.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed + 101)
    gen = ControlledDenseGraphTextQAGenerator(ccfg, seed=cfg.seed + 202)
    t0 = time.perf_counter()
    snapshots: List[Dict[str, float]] = []

    for step in range(1, int(cfg.train_steps) + 1):
        for group in opt.param_groups:
            group["lr"] = lr_for_step(cfg, step)
        rung = rng.choice(list(cfg.ladder_rungs))
        lengths = [rung_train_length(cfg, rung, step, rng) for _ in range(cfg.batch_size)]
        examples = [gen.make_example(L) for L in lengths]
        batch = collate_ladder_examples(examples, tok, ccfg, rung=rung, device=dev)
        target = batch["target"]
        tw_logits = transformer(batch, field, rung=rung)["logits"]
        mlp_logits = mlp(batch, field, rung=rung)["logits"]
        direct_logits = direct(batch, field, rung=rung)
        loss_tw = F.cross_entropy(tw_logits, target)
        loss_mlp = F.cross_entropy(mlp_logits, target)
        loss_direct = F.cross_entropy(direct_logits, target)
        loss = loss_tw + loss_mlp + loss_direct
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if cfg.writer_variant == "tied_key":
            for model in models:
                apply_tied_key_embeddings(model, tok, field, cfg)
        if step == 1 or step % max(1, cfg.train_steps // 6) == 0 or step == cfg.train_steps:
            c_tw, n = accuracy_from_logits(tw_logits.detach(), target)
            c_mlp, _ = accuracy_from_logits(mlp_logits.detach(), target)
            c_dir, _ = accuracy_from_logits(direct_logits.detach(), target)
            snap = {
                "step": float(step),
                "rung": rung,
                "loss_transformer_writer": float(loss_tw.detach().item()),
                "loss_mlp_writer": float(loss_mlp.detach().item()),
                "loss_direct_endpoint": float(loss_direct.detach().item()),
                "train_batch_acc_transformer_writer": c_tw / max(1, n),
                "train_batch_acc_mlp_writer": c_mlp / max(1, n),
                "train_batch_acc_direct_endpoint": c_dir / max(1, n),
                "learning_rate": float(opt.param_groups[0]["lr"]),
            }
            snapshots.append(snap)
            print(json.dumps({"ladder_train_progress": snap}, sort_keys=True), flush=True)

    meta = {
        "suite": "closure_writer_diagnostic_ladder",
        "config": asdict(cfg),
        "num_parameters": {
            "transformer_writer": sum(p.numel() for p in transformer.parameters()),
            "mlp_writer": sum(p.numel() for p in mlp.parameters()),
            "direct_endpoint": sum(p.numel() for p in direct.parameters()),
        },
        "train": {"snapshots": snapshots, "elapsed_train_sec": time.perf_counter() - t0},
    }
    return transformer, mlp, direct, field, tok, meta


def eval_lengths_for_rung(cfg: LadderConfig, rung: str) -> Tuple[int, ...]:
    if rung in {"gold_target_write", "one_hop_fact_write"}:
        return (1,)
    if rung == "direct_qv_write":
        return tuple(x for x in cfg.eval_lengths if x <= cfg.max_path_len)
    return tuple(x for x in cfg.eval_lengths if x >= 2 and x <= cfg.max_path_len)


@torch.no_grad()
def evaluate_ladder(
    cfg: LadderConfig,
    transformer: LadderTransformerWriter,
    mlp: LadderMLPWriter,
    direct: LadderDirectEndpoint,
    field: HolographicClosureField,
    tok: ClosureTextTokenizer,
    *,
    device: torch.device | str,
) -> List[Dict[str, float | str]]:
    transformer.eval(); mlp.eval(); direct.eval(); field.eval()
    ccfg = closure_cfg(cfg)
    rows: List[Dict[str, float | str]] = []
    dev = torch.device(device)
    for rung in cfg.ladder_rungs:
        for length in eval_lengths_for_rung(cfg, rung):
            gen = ControlledDenseGraphTextQAGenerator(ccfg, seed=cfg.seed + 1000 + int(length) + 17 * len(rung))
            examples = gen.make_examples(int(length), cfg.eval_n)
            counts: Dict[str, List[int]] = {}

            def add(name: str, c: int, n: int) -> None:
                counts.setdefault(name, [0, 0])
                counts[name][0] += int(c)
                counts[name][1] += int(n)

            for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
                batch = collate_ladder_examples(batch_examples, tok, ccfg, rung=rung, device=dev)
                target = batch["target"]
                tw = transformer(batch, field, rung=rung)
                mlp_out = mlp(batch, field, rung=rung)
                direct_logits = direct(batch, field, rung=rung)
                q = tw["query_key"]
                oracle_mem = rank_one_target_memory(q, target, cfg.num_entities)
                oracle_logits = 10.0 * field.read(oracle_mem, q)
                add("transformer_writer", *accuracy_from_logits(tw["logits"], target))
                add("mlp_writer", *accuracy_from_logits(mlp_out["logits"], target))
                add("direct_endpoint", *accuracy_from_logits(direct_logits, target))
                add("oracle_full_closure", *accuracy_from_logits(oracle_logits, target))

            row: Dict[str, float | str] = {
                "rung": rung,
                "length": float(length),
                "n": float(len(examples)),
                "writer_variant": cfg.writer_variant,
            }
            for name, (c, n) in sorted(counts.items()):
                row[f"{name}_acc"] = float(c / n) if n else float("nan")
                row[f"{name}_n"] = float(n)
            rows.append(row)
            printable = {k: row[k] for k in row if k.endswith("_acc") or k in {"rung", "length", "writer_variant", "n"}}
            print(json.dumps({"ladder_eval": printable}, sort_keys=True), flush=True)
    return rows


def write_results(rows: List[Dict[str, float | str]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / LADDER_RESULT_JSON
    csv_path = out_dir / LADDER_RESULT_CSV
    report_path = out_dir / LADDER_REPORT
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Closure Writer Diagnostic Ladder",
        "",
        f"Writer variant: `{meta.get('config', {}).get('writer_variant', 'NA') if isinstance(meta.get('config'), dict) else 'NA'}`.",
        "",
        "This ladder separates direct field writing, gold-target writing, one-hop fact writing, and multi-hop closure writing.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(cfg: LadderConfig, out_dir: Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    t0 = time.perf_counter()
    transformer, mlp, direct, field, tok, meta = train_models(cfg, device=device)
    rows = evaluate_ladder(cfg, transformer, mlp, direct, field, tok, device=device)
    meta = dict(meta)
    meta["elapsed_total_sec"] = time.perf_counter() - t0
    paths = write_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run closure-writer diagnostic ladder.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=LadderConfig.seed)
    p.add_argument("--train-steps", type=int, default=LadderConfig.train_steps)
    p.add_argument("--batch-size", type=int, default=LadderConfig.batch_size)
    p.add_argument("--eval-n", type=int, default=LadderConfig.eval_n)
    p.add_argument("--eval-batch-size", type=int, default=LadderConfig.eval_batch_size)
    p.add_argument("--d-model", type=int, default=LadderConfig.d_model)
    p.add_argument("--layers", type=int, default=LadderConfig.n_layers)
    p.add_argument("--heads", type=int, default=LadderConfig.n_heads)
    p.add_argument("--key-dim", type=int, default=LadderConfig.key_dim)
    p.add_argument("--num-entities", type=int, default=LadderConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=LadderConfig.num_relations)
    p.add_argument("--learning-rate", type=float, default=LadderConfig.learning_rate)
    p.add_argument("--curriculum", choices=["staged", "mixed"], default=LadderConfig.curriculum)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default=LadderConfig.lr_schedule)
    p.add_argument("--warmup-frac", type=float, default=LadderConfig.warmup_frac)
    p.add_argument("--min-lr-ratio", type=float, default=LadderConfig.min_lr_ratio)
    p.add_argument("--threads", type=int, default=LadderConfig.torch_threads)
    p.add_argument("--ladder-rungs", type=str, default=DEFAULT_RUNGS)
    p.add_argument("--writer-variant", choices=sorted(VALID_VARIANTS), default=LadderConfig.writer_variant)
    p.add_argument("--eval-lengths", type=str, default="1,2,3,4,6,8,12,16,24,32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rungs = parse_csv_list(args.ladder_rungs)
    unknown = set(rungs) - VALID_RUNGS
    if unknown:
        raise SystemExit(f"unknown ladder rungs: {sorted(unknown)}")
    out_dir = Path(args.out_dir)
    cfg = LadderConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
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
        torch_threads=args.threads,
        ladder_rungs=rungs,
        writer_variant=args.writer_variant,
        eval_lengths=parse_int_list(args.eval_lengths),
    )
    result = run_experiment(cfg, out_dir, device=args.device)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
