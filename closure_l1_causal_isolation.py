"""L=1 causal isolation for externally keyed closure-field writing.

The original closure writer fails already at L=1.  This script separates
pure rank-one field writing, target/value mapping, text grounding, distractor
selection, key conditioning, and field-supervised objectives.
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
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from closure_writer_diagnostic_ladder import parse_int_list, rank_one_target_memory
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


L1_RESULT_JSON = "CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.json"
L1_RESULT_CSV = "CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.csv"
L1_REPORT = "CLOSURE_L1_CAUSAL_ISOLATION_REPORT.md"

VALID_CONDITIONS = {
    "direct_qv_write",
    "gold_target_text_write",
    "id_only_write",
    "one_fact_no_distractor",
    "l1_no_distractor",
    "l1_with_distractors",
    "teacher_forced_q",
    "teacher_forced_v",
    "teacher_forced_qv",
    "field_supervised_mse",
    "field_supervised_read_ce",
}
VALID_FIELD_LOSSES = {"none", "mse", "multi_read_ce"}
VALID_INPUT_MODES = {"text", "id_only"}
VALID_DISTRACTOR_MODES = {"full", "none", "one_fact"}
VALID_KEY_CODEBOOKS = {"random_sign", "orthogonalized", "learned_key_projection"}


@dataclass(frozen=True)
class L1IsolationConfig:
    seed: int = 12001
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
    train_steps: int = 3000
    batch_size: int = 128
    eval_n: int = 512
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
    condition: str = "l1_with_distractors"
    field_loss: str = "none"
    input_mode: str = "text"
    distractor_mode: str = "full"
    key_codebook: str = "random_sign"
    log_every: int = 500
    eval_lengths: Tuple[int, ...] = (1,)


def closure_cfg(cfg: L1IsolationConfig) -> ClosureWriterConfig:
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


def condition_defaults(condition: str) -> Tuple[str, str, str]:
    if condition == "id_only_write":
        return "none", "id_only", "none"
    if condition in {"one_fact_no_distractor"}:
        return "none", "text", "one_fact"
    if condition in {"l1_no_distractor", "direct_qv_write", "gold_target_text_write", "teacher_forced_v", "teacher_forced_qv"}:
        return "none", "text", "none"
    if condition == "field_supervised_mse":
        return "mse", "text", "full"
    if condition == "field_supervised_read_ce":
        return "multi_read_ce", "text", "full"
    return "none", "text", "full"


def orthogonal_rows(count: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    mat = torch.randn(count, dim, generator=g)
    if count <= dim:
        rows: List[torch.Tensor] = []
        for i in range(count):
            v = mat[i].clone()
            for prev in rows:
                v = v - torch.dot(v, prev) / torch.clamp(torch.dot(prev, prev), min=1e-8) * prev
            norm = torch.norm(v)
            if float(norm) < 1e-6:
                v = torch.zeros(dim)
                v[i % dim] = 1.0
                norm = torch.norm(v)
            rows.append(v / norm)
        return torch.stack(rows, dim=0) * math.sqrt(float(dim))
    return F.normalize(mat, dim=-1) * math.sqrt(float(dim))


def apply_key_codebook(field: HolographicClosureField, cfg: L1IsolationConfig) -> None:
    if cfg.key_codebook == "random_sign":
        return
    if cfg.key_codebook == "learned_key_projection":
        # The learned projection condition keeps the same read interface but
        # exposes trainable writer-side projections; the fixed field remains a
        # comparable random-sign target.
        return
    if cfg.key_codebook != "orthogonalized":
        raise ValueError(f"unknown key_codebook {cfg.key_codebook!r}")
    with torch.no_grad():
        field.entity_code.copy_(orthogonal_rows(cfg.num_entities, cfg.key_dim, cfg.seed + 31))
        field.length_code.copy_(orthogonal_rows(cfg.max_path_len + 1, cfg.key_dim, cfg.seed + 32))
        rp = orthogonal_rows(cfg.max_path_len * cfg.num_relations, cfg.key_dim, cfg.seed + 33)
        field.relpos_code.copy_(rp.view(cfg.max_path_len, cfg.num_relations, cfg.key_dim))


def make_field(cfg: L1IsolationConfig, device: torch.device | str) -> Tuple[HolographicClosureField, ClosureTextTokenizer, ControlledDenseGraphTextQAGenerator]:
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0)
    apply_key_codebook(field, cfg)
    field = field.to(torch.device(device))
    gen = ControlledDenseGraphTextQAGenerator(closure_cfg(cfg), seed=cfg.seed + 202)
    return field, tok, gen


def strip_to_gold_edges(ex: PathQAExample, *, one_fact: bool) -> PathQAExample:
    edges = tuple((ex.path_nodes[i], ex.relations[i], ex.path_nodes[i + 1]) for i in range(len(ex.relations)))
    if one_fact:
        edges = edges[:1]
    return PathQAExample(
        source=ex.source,
        relations=ex.relations,
        target=ex.target,
        path_nodes=ex.path_nodes,
        edges=edges,
        attempts=ex.attempts,
    )


def condition_uses_gold_target(condition: str) -> bool:
    return condition in {
        "direct_qv_write",
        "gold_target_text_write",
        "id_only_write",
        "teacher_forced_v",
        "teacher_forced_qv",
    }


def condition_uses_query_key(condition: str) -> bool:
    return condition in {"gold_target_text_write", "teacher_forced_q", "teacher_forced_qv"}


def build_tokens(ex: PathQAExample, tok: ClosureTextTokenizer, cfg: L1IsolationConfig, condition: str) -> List[int]:
    tokens = [tok.tok("<bos>")]
    if condition != "direct_qv_write":
        for s, r, t in ex.edges:
            tokens.extend([tok.tok("<fact>"), tok.ent(s), tok.rel(r), tok.ent(t), tok.tok(";")])
        tokens.extend([tok.tok("<query>"), tok.ent(ex.source), tok.tok("follow"), tok.rel(ex.relations[0])])
    tokens.append(tok.tok("answer"))
    if condition_uses_gold_target(condition):
        tokens.append(tok.ent(ex.target))
    tokens.append(tok.tok("<read>"))
    if len(tokens) > cfg.max_seq_len:
        raise RuntimeError(f"sequence too long: {len(tokens)} > {cfg.max_seq_len}")
    return tokens


def build_condition_batch(
    cfg: L1IsolationConfig,
    tok: ClosureTextTokenizer,
    gen: ControlledDenseGraphTextQAGenerator,
    condition: str,
    *,
    batch_size: int,
    device: torch.device | str,
) -> Dict[str, torch.Tensor | List[PathQAExample]]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    field_loss, input_mode, distractor_mode = condition_defaults(condition)
    del field_loss
    if cfg.input_mode != "text":
        input_mode = cfg.input_mode
    if cfg.distractor_mode != "full":
        distractor_mode = cfg.distractor_mode
    if condition == "id_only_write":
        input_mode = "id_only"
    examples = [gen.make_example(1) for _ in range(int(batch_size))]
    if distractor_mode in {"none", "one_fact"}:
        examples = [strip_to_gold_edges(ex, one_fact=distractor_mode == "one_fact") for ex in examples]

    seqs = [build_tokens(ex, tok, cfg, condition) for ex in examples]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    read_pos = torch.zeros(len(seqs), dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
        read_pos[i] = len(seq) - 1

    dev = torch.device(device)
    q_rels = torch.zeros(len(examples), cfg.max_path_len, dtype=torch.long, device=dev)
    q_rels[:, 0] = torch.tensor([ex.relations[0] for ex in examples], dtype=torch.long, device=dev)
    out: Dict[str, torch.Tensor | List[PathQAExample]] = {
        "input_ids": ids.to(dev),
        "mask": mask.to(dev),
        "read_pos": read_pos.to(dev),
        "source": torch.tensor([ex.source for ex in examples], dtype=torch.long, device=dev),
        "q_rels": q_rels,
        "lengths": torch.ones(len(examples), dtype=torch.long, device=dev),
        "target": torch.tensor([ex.target for ex in examples], dtype=torch.long, device=dev),
        "examples": examples,
    }
    if input_mode == "id_only":
        out["id_features"] = torch.tensor(
            [[ex.source, ex.relations[0], ex.target] for ex in examples],
            dtype=torch.long,
            device=dev,
        )
    return out


class L1FieldWriter(nn.Module):
    def __init__(self, vocab_size: int, cfg: L1IsolationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ccfg = closure_cfg(cfg)
        self.backbone = TransformerBackbone(vocab_size, ccfg)
        self.id_source = nn.Embedding(cfg.num_entities, cfg.d_model)
        self.id_rel = nn.Embedding(cfg.num_relations, cfg.d_model)
        self.id_target = nn.Embedding(cfg.num_entities, cfg.d_model)
        self.direct_feature = nn.Sequential(nn.Linear(cfg.key_dim + cfg.num_entities, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        self.key_fusion = nn.Sequential(nn.Linear(cfg.d_model + cfg.key_dim, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        self.value_fusion = nn.Sequential(nn.Linear(cfg.d_model + cfg.num_entities, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        self.learned_key_projection = nn.Sequential(nn.Linear(cfg.key_dim, cfg.key_dim), nn.Tanh())
        self.memory_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.key_dim * cfg.num_entities),
        )
        self.direct_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_entities))

    def hidden(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, condition: str) -> Tuple[torch.Tensor, torch.Tensor]:
        q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
        if self.cfg.key_codebook == "learned_key_projection":
            q_for_writer = self.learned_key_projection(q)
        else:
            q_for_writer = q
        if condition in {"direct_qv_write", "teacher_forced_qv"}:
            one_hot = F.one_hot(batch["target"], num_classes=self.cfg.num_entities).to(q.dtype)
            return self.direct_feature(torch.cat([q_for_writer, one_hot], dim=-1)), q
        if "id_features" in batch:
            ids = batch["id_features"]
            h = self.id_source(ids[:, 0]) + self.id_rel(ids[:, 1]) + self.id_target(ids[:, 2])
        else:
            h = self.backbone(batch["input_ids"], batch["mask"], batch["read_pos"])
        if condition_uses_query_key(condition):
            h = self.key_fusion(torch.cat([h, q_for_writer.to(h.dtype)], dim=-1))
        if condition == "teacher_forced_v":
            one_hot = F.one_hot(batch["target"], num_classes=self.cfg.num_entities).to(h.dtype)
            h = self.value_fusion(torch.cat([h, one_hot], dim=-1))
        return h, q

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, condition: str) -> Dict[str, torch.Tensor]:
        h, q = self.hidden(batch, field, condition)
        memory = self.memory_head(h).view(-1, self.cfg.key_dim, self.cfg.num_entities)
        return {
            "memory": memory,
            "query_key": q,
            "logits": field.read(memory, q),
            "direct_logits": self.direct_head(h),
        }


def lr_for_step(cfg: L1IsolationConfig, step: int) -> float:
    if cfg.lr_schedule == "constant":
        return float(cfg.learning_rate)
    warmup = max(1, int(cfg.train_steps * cfg.warmup_frac))
    if step <= warmup:
        return float(cfg.learning_rate) * step / warmup
    progress = (step - warmup) / max(1, cfg.train_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(cfg.learning_rate) * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def random_source_key(field: HolographicClosureField, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    source = (batch["source"] + 1) % field.num_entities
    return field.key(source, batch["q_rels"], batch["lengths"])


def field_supervision_loss(cfg: L1IsolationConfig, field: HolographicClosureField, out: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    q = out["query_key"]
    oracle = rank_one_target_memory(q, target, cfg.num_entities)
    if cfg.field_loss == "mse":
        return F.mse_loss(out["memory"], oracle)
    if cfg.field_loss == "multi_read_ce":
        wrong_q = q.roll(shifts=1, dims=0)
        wrong_logits = field.read(out["memory"], wrong_q)
        wrong_prob = F.softmax(wrong_logits, dim=-1).gather(1, target.view(-1, 1)).squeeze(1)
        return -torch.log1p(-wrong_prob.clamp(max=0.999)).mean()
    return torch.zeros((), dtype=out["memory"].dtype, device=out["memory"].device)


def train_model(cfg: L1IsolationConfig, device: torch.device | str) -> Tuple[L1FieldWriter, HolographicClosureField, ClosureTextTokenizer, ControlledDenseGraphTextQAGenerator, Dict[str, object]]:
    if cfg.condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown condition {cfg.condition!r}")
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    field, tok, gen = make_field(cfg, dev)
    model = L1FieldWriter(tok.vocab_size, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    checkpoints = {1, int(cfg.train_steps)}
    checkpoints.update(x for x in [100, 500, 1000, 3000, 10000] if x <= int(cfg.train_steps))
    if cfg.log_every > 0:
        checkpoints.update(range(cfg.log_every, int(cfg.train_steps) + 1, cfg.log_every))
    snapshots: List[Dict[str, float | str]] = []
    last_grad_norm = 0.0
    t0 = time.perf_counter()
    for step in range(1, int(cfg.train_steps) + 1):
        for group in opt.param_groups:
            group["lr"] = lr_for_step(cfg, step)
        batch = build_condition_batch(cfg, tok, gen, cfg.condition, batch_size=cfg.batch_size, device=dev)
        target = batch["target"]
        out = model(batch, field, cfg.condition)
        out["memory"].retain_grad()
        loss_writer = F.cross_entropy(out["logits"], target)
        loss_direct = F.cross_entropy(out["direct_logits"], target)
        loss_field = field_supervision_loss(cfg, field, out, target)
        loss = loss_writer + loss_direct + loss_field
        opt.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = float(out["memory"].grad.detach().norm().cpu().item()) if out["memory"].grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step in checkpoints:
            c_writer, n = accuracy_from_logits(out["logits"].detach(), target)
            c_direct, _ = accuracy_from_logits(out["direct_logits"].detach(), target)
            snap = {
                "step": float(step),
                "condition": cfg.condition,
                "loss_writer": float(loss_writer.detach().cpu().item()),
                "loss_direct": float(loss_direct.detach().cpu().item()),
                "loss_field": float(loss_field.detach().cpu().item()),
                "train_batch_acc_writer": float(c_writer / max(1, n)),
                "train_batch_acc_direct": float(c_direct / max(1, n)),
                "grad_norm": last_grad_norm,
                "learning_rate": float(opt.param_groups[0]["lr"]),
            }
            snapshots.append(snap)
            print(json.dumps({"l1_train_progress": snap}, sort_keys=True), flush=True)
    meta = {
        "suite": "closure_l1_causal_isolation",
        "config": asdict(cfg),
        "num_parameters": {"field_writer": sum(p.numel() for p in model.parameters())},
        "train": {"snapshots": snapshots, "elapsed_train_sec": time.perf_counter() - t0, "last_grad_norm": last_grad_norm},
    }
    return model, field, tok, gen, meta


@torch.no_grad()
def evaluate_model(
    cfg: L1IsolationConfig,
    model: L1FieldWriter,
    field: HolographicClosureField,
    tok: ClosureTextTokenizer,
    gen: ControlledDenseGraphTextQAGenerator,
    meta: Dict[str, object],
    *,
    device: torch.device | str,
) -> List[Dict[str, float | str]]:
    model.eval()
    dev = torch.device(device)
    rows: List[Dict[str, float | str]] = []
    for length in cfg.eval_lengths:
        if int(length) != 1:
            continue
        counts: Dict[str, List[float]] = {
            "writer": [0.0, 0.0],
            "direct": [0.0, 0.0],
            "oracle": [0.0, 0.0],
            "wrong_old": [0.0, 0.0],
            "field_mse": [0.0, 0.0],
        }
        for _ in range(0, int(cfg.eval_n), int(cfg.eval_batch_size)):
            n_batch = min(int(cfg.eval_batch_size), int(cfg.eval_n) - int(counts["writer"][1]))
            if n_batch <= 0:
                break
            batch = build_condition_batch(cfg, tok, gen, cfg.condition, batch_size=n_batch, device=dev)
            target = batch["target"]
            out = model(batch, field, cfg.condition)
            q = out["query_key"]
            oracle_mem = rank_one_target_memory(q, target, cfg.num_entities)
            oracle_logits = 10.0 * field.read(oracle_mem, q)
            c, n = accuracy_from_logits(out["logits"], target)
            counts["writer"][0] += c; counts["writer"][1] += n
            c, n = accuracy_from_logits(out["direct_logits"], target)
            counts["direct"][0] += c; counts["direct"][1] += n
            c, n = accuracy_from_logits(oracle_logits, target)
            counts["oracle"][0] += c; counts["oracle"][1] += n
            wrong_logits = field.read(out["memory"], random_source_key(field, batch))
            counts["wrong_old"][0] += int(wrong_logits.argmax(dim=-1).eq(target).sum().item())
            counts["wrong_old"][1] += int(target.numel())
            counts["field_mse"][0] += float(F.mse_loss(out["memory"], oracle_mem, reduction="sum").cpu().item())
            counts["field_mse"][1] += float(out["memory"].numel())
        row: Dict[str, float | str] = {
            "condition": cfg.condition,
            "length": float(length),
            "n": counts["writer"][1],
            "field_loss": cfg.field_loss,
            "input_mode": cfg.input_mode,
            "distractor_mode": cfg.distractor_mode,
            "key_codebook": cfg.key_codebook,
            "train_steps": float(cfg.train_steps),
            "grad_norm": float(meta.get("train", {}).get("last_grad_norm", 0.0)) if isinstance(meta.get("train"), dict) else 0.0,
        }
        row["transformer_writer_acc"] = counts["writer"][0] / max(1.0, counts["writer"][1])
        row["transformer_writer_n"] = counts["writer"][1]
        row["correct_key_read_acc"] = row["transformer_writer_acc"]
        row["correct_key_read_n"] = counts["writer"][1]
        row["direct_endpoint_acc"] = counts["direct"][0] / max(1.0, counts["direct"][1])
        row["direct_endpoint_n"] = counts["direct"][1]
        row["oracle_full_closure_acc"] = counts["oracle"][0] / max(1.0, counts["oracle"][1])
        row["oracle_full_closure_n"] = counts["oracle"][1]
        row["wrong_key_old_target_rate"] = counts["wrong_old"][0] / max(1.0, counts["wrong_old"][1])
        row["wrong_key_old_target_n"] = counts["wrong_old"][1]
        row["field_mse"] = counts["field_mse"][0] / max(1.0, counts["field_mse"][1])
        rows.append(row)
        print(json.dumps({"l1_eval": row}, sort_keys=True), flush=True)
    return rows


def write_results(rows: List[Dict[str, float | str]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / L1_RESULT_JSON
    csv_path = out_dir / L1_RESULT_CSV
    report_path = out_dir / L1_REPORT
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(
        "# L=1 Causal Isolation\n\n"
        f"Condition: `{meta.get('config', {}).get('condition', 'NA') if isinstance(meta.get('config'), dict) else 'NA'}`.\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(cfg: L1IsolationConfig, out_dir: Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    t0 = time.perf_counter()
    model, field, tok, gen, meta = train_model(cfg, device=device)
    rows = evaluate_model(cfg, model, field, tok, gen, meta, device=device)
    meta = dict(meta)
    meta["elapsed_total_sec"] = time.perf_counter() - t0
    paths = write_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run L=1 causal isolation diagnostics for closure-field writing.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=L1IsolationConfig.seed)
    p.add_argument("--train-steps", type=int, default=L1IsolationConfig.train_steps)
    p.add_argument("--batch-size", type=int, default=L1IsolationConfig.batch_size)
    p.add_argument("--eval-n", type=int, default=L1IsolationConfig.eval_n)
    p.add_argument("--eval-batch-size", type=int, default=L1IsolationConfig.eval_batch_size)
    p.add_argument("--d-model", type=int, default=L1IsolationConfig.d_model)
    p.add_argument("--layers", type=int, default=L1IsolationConfig.n_layers)
    p.add_argument("--heads", type=int, default=L1IsolationConfig.n_heads)
    p.add_argument("--key-dim", type=int, default=L1IsolationConfig.key_dim)
    p.add_argument("--num-entities", type=int, default=L1IsolationConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=L1IsolationConfig.num_relations)
    p.add_argument("--learning-rate", type=float, default=L1IsolationConfig.learning_rate)
    p.add_argument("--curriculum", choices=["staged", "mixed"], default=L1IsolationConfig.curriculum)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default=L1IsolationConfig.lr_schedule)
    p.add_argument("--warmup-frac", type=float, default=L1IsolationConfig.warmup_frac)
    p.add_argument("--min-lr-ratio", type=float, default=L1IsolationConfig.min_lr_ratio)
    p.add_argument("--threads", type=int, default=L1IsolationConfig.torch_threads)
    p.add_argument("--condition", choices=sorted(VALID_CONDITIONS), default=L1IsolationConfig.condition)
    p.add_argument("--field-loss", choices=sorted(VALID_FIELD_LOSSES), default="")
    p.add_argument("--input-mode", choices=sorted(VALID_INPUT_MODES), default="")
    p.add_argument("--distractor-mode", choices=sorted(VALID_DISTRACTOR_MODES), default="")
    p.add_argument("--key-codebook", choices=sorted(VALID_KEY_CODEBOOKS), default=L1IsolationConfig.key_codebook)
    p.add_argument("--log-every", type=int, default=L1IsolationConfig.log_every)
    p.add_argument("--eval-lengths", type=str, default="1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    field_loss, input_mode, distractor_mode = condition_defaults(args.condition)
    if args.field_loss:
        field_loss = args.field_loss
    if args.input_mode:
        input_mode = args.input_mode
    if args.distractor_mode:
        distractor_mode = args.distractor_mode
    cfg = L1IsolationConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        learning_rate=args.learning_rate,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        curriculum=args.curriculum,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        min_lr_ratio=args.min_lr_ratio,
        torch_threads=args.threads,
        condition=args.condition,
        field_loss=field_loss,
        input_mode=input_mode,
        distractor_mode=distractor_mode,
        key_codebook=args.key_codebook,
        log_every=args.log_every,
        eval_lengths=parse_int_list(args.eval_lengths),
    )
    result = run_experiment(cfg, Path(args.out_dir), device=args.device)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
