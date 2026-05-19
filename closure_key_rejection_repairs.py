"""Wrong-key rejection repairs for rank-one closure memories.

The anti-shortcut suite shows that a rank-one memory can be causally used while
still returning the old target for correlated wrong keys. This script isolates
the read side with exact rank-one writes and tests simple acceptance mechanisms:
plain linear read, a contrastively trained key gate, a fixed threshold/null gate,
and a margin gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    PathQAExample,
    accuracy_from_logits,
    iter_minibatches,
)
from closure_writer_diagnostic_ladder import rank_one_target_memory


REPAIR_RESULT_JSON = "CLOSURE_KEY_REJECTION_REPAIRS_RESULTS.json"
REPAIR_RESULT_CSV = "CLOSURE_KEY_REJECTION_REPAIRS_RESULTS.csv"
REPAIR_REPORT = "CLOSURE_KEY_REJECTION_REPAIRS_REPORT.md"
VALID_REPAIRS = {"linear", "contrastive", "threshold_null", "margin_gate"}


@dataclass(frozen=True)
class RepairConfig:
    seed: int = 9901
    num_entities: int = 48
    num_relations: int = 4
    max_path_len: int = 32
    key_dim: int = 128
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    curriculum: str = "staged"
    lr_schedule: str = "constant"
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    train_steps: int = 200
    batch_size: int = 128
    eval_n: int = 512
    eval_batch_size: int = 64
    base_distractors: int = 6
    distractors_per_hop: int = 3
    same_relation_branch_prob: float = 0.25
    repair_variant: str = "linear"
    pipeline: str = "exact_rank_one"
    wrong_key_families: Tuple[str, ...] = ("random_source", "reversed", "prefix", "shuffled")
    calibration_split: str = "fixed_threshold"
    contrastive_weight: float = 1.0
    reject_threshold: float = 0.25
    accept_margin: float = 0.25
    learning_rate: float = 5e-2
    torch_threads: int = 4
    eval_lengths: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


class KeyGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, key_dot: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale * key_dot + self.bias)


def closure_cfg(cfg: RepairConfig) -> ClosureWriterConfig:
    return ClosureWriterConfig(
        seed=cfg.seed,
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        max_path_len=cfg.max_path_len,
        key_dim=cfg.key_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        train_steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
        base_distractors=cfg.base_distractors,
        distractors_per_hop=cfg.distractors_per_hop,
        same_relation_branch_prob=cfg.same_relation_branch_prob,
        curriculum=cfg.curriculum,
        lr_schedule=cfg.lr_schedule,
        warmup_frac=cfg.warmup_frac,
        min_lr_ratio=cfg.min_lr_ratio,
        torch_threads=cfg.torch_threads,
    )


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def parse_name_list(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def padded_relations(examples: Sequence[PathQAExample], cfg: RepairConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.tensor([ex.source for ex in examples], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ex.relations) for ex in examples], dtype=torch.long, device=device)
    rels = torch.zeros(len(examples), cfg.max_path_len, dtype=torch.long, device=device)
    for i, ex in enumerate(examples):
        rels[i, : len(ex.relations)] = torch.tensor(ex.relations, dtype=torch.long, device=device)
    return source, rels, lengths


def shuffled_relations(q_rels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = q_rels.clone()
    for i, length in enumerate(lengths.tolist()):
        if length > 1:
            vals = out[i, :length].clone()
            out[i, :length] = torch.cat([vals[1:], vals[:1]])
    return out


def wrong_keys(
    field: HolographicClosureField,
    source: torch.Tensor,
    q_rels: torch.Tensor,
    lengths: torch.Tensor,
    families: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    wanted = set(families or ("random_source", "reversed", "prefix", "shuffled"))
    rand_source = (source + 1) % field.num_entities
    rev_rels = q_rels.clone()
    for i, length in enumerate(lengths.tolist()):
        if length > 1:
            rev_rels[i, :length] = torch.flip(rev_rels[i, :length], dims=[0])
    prefix_len = torch.clamp(lengths - 1, min=1)
    swapped_rels = q_rels.clone()
    swapped_rels[:, 0] = (swapped_rels[:, 0] + 1) % field.num_relations
    out = {
        "random_source": field.key(rand_source, q_rels, lengths),
        "reversed": field.key(source, rev_rels, lengths),
        "prefix": field.key(source, q_rels, prefix_len),
        "shuffled": field.key(source, shuffled_relations(q_rels, lengths), lengths),
        "swapped": field.key(source, swapped_rels, lengths),
    }
    unknown = wanted.difference(out.keys())
    if unknown:
        raise ValueError(f"unknown wrong-key families: {sorted(unknown)}")
    return {k: v for k, v in out.items() if k in wanted}


def key_dot(read_key: torch.Tensor, stored_key: torch.Tensor) -> torch.Tensor:
    return (read_key * stored_key).sum(dim=-1)


def binary_score_metrics(positives: torch.Tensor, negatives: torch.Tensor, threshold: float) -> Dict[str, float]:
    pos = positives.detach().flatten().float().cpu()
    neg = negatives.detach().flatten().float().cpu()
    if pos.numel() == 0 or neg.numel() == 0:
        auroc = float("nan")
    else:
        cmp = pos.view(-1, 1) - neg.view(1, -1)
        auroc = float(((cmp > 0).float() + 0.5 * (cmp == 0).float()).mean().item())
    fpr = float((neg >= float(threshold)).float().mean().item()) if neg.numel() else float("nan")
    fnr = float((pos < float(threshold)).float().mean().item()) if pos.numel() else float("nan")
    return {"auroc": auroc, "fpr": fpr, "fnr": fnr, "threshold": float(threshold)}


def select_validation_gate(positives: torch.Tensor, negatives: torch.Tensor, candidates: torch.Tensor) -> Dict[str, float]:
    pos = positives.detach().flatten().float().cpu()
    neg = negatives.detach().flatten().float().cpu()
    cand = candidates.detach().flatten().float().cpu()
    if cand.numel() == 0:
        if pos.numel() or neg.numel():
            vals = torch.cat([x for x in [pos, neg] if x.numel()])
            cand = torch.linspace(float(vals.min().item()), float(vals.max().item()), steps=41)
        else:
            cand = torch.tensor([0.0])
    best: Optional[Dict[str, float]] = None
    for t in cand.tolist():
        metrics = binary_score_metrics(pos, neg, float(t))
        objective = metrics["fpr"] + metrics["fnr"]
        tie_break = metrics["fpr"]
        current = {"threshold": float(t), "objective": float(objective), **metrics}
        if best is None or (objective, tie_break, abs(float(t))) < (best["objective"], best["fpr"], abs(best["threshold"])):
            best = current
    assert best is not None
    return best


def margin_scores(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def candidate_thresholds(positives: torch.Tensor, negatives: torch.Tensor, fallback: float) -> torch.Tensor:
    vals = [x.detach().flatten().float().cpu() for x in [positives, negatives] if x.numel()]
    if not vals:
        return torch.tensor([float(fallback)])
    all_vals = torch.cat(vals)
    lo = float(all_vals.min().item())
    hi = float(all_vals.max().item())
    if abs(hi - lo) < 1e-8:
        return torch.tensor([float(fallback), lo])
    grid = torch.linspace(lo, hi, steps=81)
    return torch.unique(torch.cat([grid, torch.tensor([float(fallback)])]).sort().values)


def train_contrastive_gate(cfg: RepairConfig, field: HolographicClosureField, device: torch.device) -> Tuple[KeyGate, Dict[str, float]]:
    gate = KeyGate().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=cfg.learning_rate)
    gen = ControlledDenseGraphTextQAGenerator(closure_cfg(cfg), seed=cfg.seed + 404)
    snapshots: List[Dict[str, float]] = []
    rng = random.Random(cfg.seed + 505)
    for step in range(1, int(cfg.train_steps) + 1):
        lengths = [rng.choice((1, 2, 3)) for _ in range(cfg.batch_size)]
        examples = [gen.make_example(L) for L in lengths]
        source, q_rels, lens = padded_relations(examples, cfg, device)
        correct = field.key(source, q_rels, lens)
        wrong = wrong_keys(field, source, q_rels, lens, cfg.wrong_key_families)
        wrong_stack = torch.cat([v for v in wrong.values()], dim=0)
        correct_dot = key_dot(correct, correct)
        wrong_dot = key_dot(wrong_stack, correct.repeat(len(wrong), 1))
        logits = torch.cat([gate.scale * correct_dot + gate.bias, gate.scale * wrong_dot + gate.bias], dim=0)
        labels = torch.cat([torch.ones_like(correct_dot), torch.zeros_like(wrong_dot)], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels) * float(cfg.contrastive_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step == cfg.train_steps or step % max(1, cfg.train_steps // 4) == 0:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pred = prob >= 0.5
                acc = pred.eq(labels.bool()).float().mean().item()
            snapshots.append({"step": float(step), "loss": float(loss.detach().item()), "gate_acc": float(acc)})
    return gate, {
        "gate_scale": float(gate.scale.detach().cpu().item()),
        "gate_bias": float(gate.bias.detach().cpu().item()),
        "last_gate_acc": float(snapshots[-1]["gate_acc"] if snapshots else float("nan")),
    }


def accept_mask(
    cfg: RepairConfig,
    variant: str,
    logits: torch.Tensor,
    read_key: torch.Tensor,
    stored_key: torch.Tensor,
    gate: KeyGate | None,
) -> torch.Tensor:
    if variant == "linear":
        return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    dots = key_dot(read_key, stored_key)
    if variant == "contrastive":
        if gate is None:
            raise ValueError("contrastive repair requires a trained gate")
        return gate(dots) >= 0.5
    if variant == "threshold_null":
        return dots >= float(cfg.reject_threshold)
    if variant == "margin_gate":
        top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
        return margin >= float(cfg.accept_margin)
    raise ValueError(variant)


def repaired_accuracy(logits: torch.Tensor, target: torch.Tensor, accept: torch.Tensor) -> Tuple[int, int]:
    pred = logits.argmax(dim=-1)
    correct = pred.eq(target) & accept
    return int(correct.sum().item()), int(target.numel())


def build_learned_pipeline(cfg: RepairConfig, field: HolographicClosureField, *, device: torch.device):
    from learned_grounding_closure import LearnedGroundingTextTokenizer, train_learned_extractor_writer

    ccfg = closure_cfg(cfg)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations)
    writer, train_meta = train_learned_extractor_writer(
        ccfg,
        tok,
        field,
        device=device,
        train_steps=cfg.train_steps,
        batch_size=cfg.batch_size,
    )
    return writer, tok, train_meta


@torch.no_grad()
def learned_pipeline_scores(
    cfg: RepairConfig,
    field: HolographicClosureField,
    writer,
    tok,
    *,
    seed_offset: int,
    eval_n: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    from learned_grounding_closure import collate_learned_grounding_examples

    writer.eval(); field.eval()
    ccfg = closure_cfg(cfg)
    rng = random.Random(cfg.seed + seed_offset + 91)
    threshold_pos: List[torch.Tensor] = []
    threshold_neg: List[torch.Tensor] = []
    margin_pos: List[torch.Tensor] = []
    margin_neg: List[torch.Tensor] = []
    for length in cfg.eval_lengths:
        gen = ControlledDenseGraphTextQAGenerator(ccfg, seed=cfg.seed + seed_offset + int(length))
        examples = gen.make_examples(int(length), int(eval_n))
        for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
            batch = collate_learned_grounding_examples(batch_examples, tok, ccfg, device=device, rng=rng)
            out = writer(batch, field)
            q = out["query_key"]
            mem = out["memory"]
            scale = float(getattr(writer, "output_scale", 1.0))
            threshold_pos.append(key_dot(q, q).detach().cpu())
            margin_pos.append(margin_scores(out["logits"]).detach().cpu())
            for wrong_q in wrong_keys(field, batch["source"], batch["q_rels"], batch["lengths"], cfg.wrong_key_families).values():
                wrong_logits = scale * field.read(mem, wrong_q)
                threshold_neg.append(key_dot(wrong_q, q).detach().cpu())
                margin_neg.append(margin_scores(wrong_logits).detach().cpu())
    empty = torch.empty(0)
    return {
        "threshold_pos": torch.cat(threshold_pos) if threshold_pos else empty,
        "threshold_neg": torch.cat(threshold_neg) if threshold_neg else empty,
        "margin_pos": torch.cat(margin_pos) if margin_pos else empty,
        "margin_neg": torch.cat(margin_neg) if margin_neg else empty,
    }


def calibrate_learned_pipeline(cfg: RepairConfig, field: HolographicClosureField, writer, tok, *, device: torch.device) -> Dict[str, float]:
    scores = learned_pipeline_scores(
        cfg,
        field,
        writer,
        tok,
        seed_offset=30000,
        eval_n=max(4, min(int(cfg.eval_n), 256)),
        device=device,
    )
    threshold = select_validation_gate(
        scores["threshold_pos"],
        scores["threshold_neg"],
        candidate_thresholds(scores["threshold_pos"], scores["threshold_neg"], cfg.reject_threshold),
    )
    margin = select_validation_gate(
        scores["margin_pos"],
        scores["margin_neg"],
        candidate_thresholds(scores["margin_pos"], scores["margin_neg"], cfg.accept_margin),
    )
    if cfg.repair_variant == "margin_gate":
        chosen = margin
        selected_reject = float(cfg.reject_threshold)
        selected_margin = float(margin["threshold"])
    else:
        chosen = threshold
        selected_reject = float(threshold["threshold"])
        selected_margin = float(cfg.accept_margin)
    return {
        "selected_reject_threshold": selected_reject,
        "selected_accept_margin": selected_margin,
        "validation_auroc": float(chosen["auroc"]),
        "validation_fpr": float(chosen["fpr"]),
        "validation_fnr": float(chosen["fnr"]),
        "threshold_validation_auroc": float(threshold["auroc"]),
        "threshold_validation_fpr": float(threshold["fpr"]),
        "threshold_validation_fnr": float(threshold["fnr"]),
        "margin_validation_auroc": float(margin["auroc"]),
        "margin_validation_fpr": float(margin["fpr"]),
        "margin_validation_fnr": float(margin["fnr"]),
    }


@torch.no_grad()
def evaluate_learned_repairs(
    cfg: RepairConfig,
    field: HolographicClosureField,
    writer,
    tok,
    calibration: Dict[str, float],
    *,
    device: torch.device,
) -> List[Dict[str, float | str]]:
    from learned_grounding_closure import collate_learned_grounding_examples

    writer.eval(); field.eval()
    rows: List[Dict[str, float | str]] = []
    ccfg = closure_cfg(cfg)
    rng = random.Random(cfg.seed + 6060)
    for length in cfg.eval_lengths:
        gen = ControlledDenseGraphTextQAGenerator(ccfg, seed=cfg.seed + 40000 + int(length))
        examples = gen.make_examples(int(length), cfg.eval_n)
        counts: Dict[str, List[int]] = {}
        positive_scores: List[torch.Tensor] = []
        negative_scores: List[torch.Tensor] = []

        def add(name: str, value: int, denom: int) -> None:
            counts.setdefault(name, [0, 0])
            counts[name][0] += int(value)
            counts[name][1] += int(denom)

        for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
            batch = collate_learned_grounding_examples(batch_examples, tok, ccfg, device=device, rng=rng)
            out = writer(batch, field)
            target = batch["target"]
            q = out["query_key"]
            mem = out["memory"]
            logits = out["logits"]
            scale = float(getattr(writer, "output_scale", 1.0))
            correct_accept = accept_mask(cfg, cfg.repair_variant, logits, q, q, None)
            c, n = repaired_accuracy(logits, target, correct_accept)
            add("correct_key_answer", c, n)
            add("correct_key_reject", int((~correct_accept).sum().item()), int(correct_accept.numel()))
            if cfg.repair_variant == "margin_gate":
                positive_scores.append(margin_scores(logits).detach().cpu())
            else:
                positive_scores.append(key_dot(q, q).detach().cpu())

            for wrong_name, wrong_q in wrong_keys(field, batch["source"], batch["q_rels"], batch["lengths"], cfg.wrong_key_families).items():
                wrong_logits = scale * field.read(mem, wrong_q)
                accept = accept_mask(cfg, cfg.repair_variant, wrong_logits, wrong_q, q, None)
                pred = wrong_logits.argmax(dim=-1)
                add(f"{wrong_name}_old_target", int((pred.eq(target) & accept).sum().item()), int(target.numel()))
                add(f"{wrong_name}_reject", int((~accept).sum().item()), int(accept.numel()))
                if cfg.repair_variant == "margin_gate":
                    negative_scores.append(margin_scores(wrong_logits).detach().cpu())
                else:
                    negative_scores.append(key_dot(wrong_q, q).detach().cpu())

        row: Dict[str, float | str] = {
            "length": float(length),
            "n": float(len(examples)),
            "repair_variant": cfg.repair_variant,
            "pipeline": cfg.pipeline,
            "pipeline_implementation": "learned_extractor_memory",
            "wrong_key_families": ",".join(cfg.wrong_key_families),
            "calibration_split": cfg.calibration_split,
            "selected_reject_threshold": float(calibration["selected_reject_threshold"]),
            "selected_accept_margin": float(calibration["selected_accept_margin"]),
            "validation_auroc": float(calibration["validation_auroc"]),
            "validation_fpr": float(calibration["validation_fpr"]),
            "validation_fnr": float(calibration["validation_fnr"]),
        }
        wrong_old_num = wrong_old_den = wrong_rej_num = wrong_rej_den = 0
        for name, (num, den) in sorted(counts.items()):
            row[f"{name}_rate"] = float(num / den) if den else float("nan")
            row[f"{name}_n"] = float(den)
            if name.endswith("_old_target"):
                wrong_old_num += num; wrong_old_den += den
            if name.endswith("_reject") and not name.startswith("correct_key"):
                wrong_rej_num += num; wrong_rej_den += den
        row["wrong_key_old_target_rate"] = float(wrong_old_num / wrong_old_den) if wrong_old_den else float("nan")
        row["wrong_key_old_target_n"] = float(wrong_old_den)
        row["wrong_key_reject_rate"] = float(wrong_rej_num / wrong_rej_den) if wrong_rej_den else float("nan")
        row["wrong_key_reject_n"] = float(wrong_rej_den)
        if positive_scores and negative_scores:
            selected = float(calibration["selected_accept_margin"] if cfg.repair_variant == "margin_gate" else calibration["selected_reject_threshold"])
            metrics = binary_score_metrics(torch.cat(positive_scores), torch.cat(negative_scores), selected)
            row["key_gate_auroc"] = metrics["auroc"]
            row["key_gate_fpr"] = metrics["fpr"]
            row["key_gate_fnr"] = metrics["fnr"]
            row["key_gate_threshold"] = metrics["threshold"]
        rows.append(row)
        printable = {k: row[k] for k in row if k in {"length", "repair_variant", "pipeline_implementation", "correct_key_answer_rate", "wrong_key_old_target_rate", "wrong_key_reject_rate", "correct_key_reject_rate"}}
        print(json.dumps({"repair_eval": printable}, sort_keys=True), flush=True)
    return rows


@torch.no_grad()
def evaluate_repairs(cfg: RepairConfig, field: HolographicClosureField, gate: KeyGate | None, *, device: torch.device) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    ccfg = closure_cfg(cfg)
    for length in cfg.eval_lengths:
        gen = ControlledDenseGraphTextQAGenerator(ccfg, seed=cfg.seed + 1000 + int(length))
        examples = gen.make_examples(int(length), cfg.eval_n)
        counts: Dict[str, List[int]] = {}
        correct_dots: List[torch.Tensor] = []
        wrong_dots: List[torch.Tensor] = []

        def add(name: str, value: int, denom: int) -> None:
            counts.setdefault(name, [0, 0])
            counts[name][0] += int(value)
            counts[name][1] += int(denom)

        for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
            source, q_rels, lens = padded_relations(batch_examples, cfg, device)
            target = torch.tensor([ex.target for ex in batch_examples], dtype=torch.long, device=device)
            q = field.key(source, q_rels, lens)
            mem = rank_one_target_memory(q, target, cfg.num_entities)
            logits = field.read(mem, q)
            correct_dots.append(key_dot(q, q).detach().cpu())
            correct_accept = accept_mask(cfg, cfg.repair_variant, logits, q, q, gate)
            c, n = repaired_accuracy(logits, target, correct_accept)
            add("correct_key_answer", c, n)
            add("correct_key_reject", int((~correct_accept).sum().item()), int(correct_accept.numel()))

            for wrong_name, wrong_q in wrong_keys(field, source, q_rels, lens, cfg.wrong_key_families).items():
                wrong_logits = field.read(mem, wrong_q)
                accept = accept_mask(cfg, cfg.repair_variant, wrong_logits, wrong_q, q, gate)
                pred = wrong_logits.argmax(dim=-1)
                wrong_dots.append(key_dot(wrong_q, q).detach().cpu())
                add(f"{wrong_name}_old_target", int((pred.eq(target) & accept).sum().item()), int(target.numel()))
                add(f"{wrong_name}_reject", int((~accept).sum().item()), int(accept.numel()))

        row: Dict[str, float | str] = {
            "length": float(length),
            "n": float(len(examples)),
            "repair_variant": cfg.repair_variant,
            "pipeline": cfg.pipeline,
            "pipeline_implementation": "exact_rank_one_oracle_memory",
            "wrong_key_families": ",".join(cfg.wrong_key_families),
            "calibration_split": cfg.calibration_split,
            "selected_reject_threshold": float(cfg.reject_threshold),
            "selected_accept_margin": float(cfg.accept_margin),
            "validation_auroc": float("nan"),
            "validation_fpr": float("nan"),
            "validation_fnr": float("nan"),
        }
        wrong_old_num = wrong_old_den = wrong_rej_num = wrong_rej_den = 0
        for name, (num, den) in sorted(counts.items()):
            row[f"{name}_rate"] = float(num / den) if den else float("nan")
            row[f"{name}_n"] = float(den)
            if name.endswith("_old_target"):
                wrong_old_num += num; wrong_old_den += den
            if name.endswith("_reject") and not name.startswith("correct_key"):
                wrong_rej_num += num; wrong_rej_den += den
        row["wrong_key_old_target_rate"] = float(wrong_old_num / wrong_old_den) if wrong_old_den else float("nan")
        row["wrong_key_old_target_n"] = float(wrong_old_den)
        row["wrong_key_reject_rate"] = float(wrong_rej_num / wrong_rej_den) if wrong_rej_den else float("nan")
        row["wrong_key_reject_n"] = float(wrong_rej_den)
        if correct_dots and wrong_dots:
            metrics = binary_score_metrics(torch.cat(correct_dots), torch.cat(wrong_dots), cfg.reject_threshold)
            row["key_gate_auroc"] = metrics["auroc"]
            row["key_gate_fpr"] = metrics["fpr"]
            row["key_gate_fnr"] = metrics["fnr"]
            row["key_gate_threshold"] = metrics["threshold"]
        rows.append(row)
        printable = {k: row[k] for k in row if k in {"length", "repair_variant", "correct_key_answer_rate", "wrong_key_old_target_rate", "wrong_key_reject_rate", "correct_key_reject_rate"}}
        print(json.dumps({"repair_eval": printable}, sort_keys=True), flush=True)
    return rows


def write_results(rows: List[Dict[str, float | str]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPAIR_RESULT_JSON
    csv_path = out_dir / REPAIR_RESULT_CSV
    report_path = out_dir / REPAIR_REPORT
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text("# Closure Key Rejection Repairs\n\nExact rank-one writes with read-side rejection variants.\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(cfg: RepairConfig, out_dir: Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    if cfg.repair_variant not in VALID_REPAIRS:
        raise ValueError(f"unknown repair_variant {cfg.repair_variant!r}")
    if cfg.pipeline not in {"exact_rank_one", "learned_grounding"}:
        raise ValueError(f"unknown pipeline {cfg.pipeline!r}")
    if cfg.calibration_split not in {"validation", "fixed_threshold"}:
        raise ValueError(f"unknown calibration_split {cfg.calibration_split!r}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(dev)
    t0 = time.perf_counter()
    gate = None
    gate_meta: Dict[str, float] = {}
    pipeline_implementation = "exact_rank_one_oracle_memory"
    calibration_meta: Dict[str, float] = {
        "selected_reject_threshold": float(cfg.reject_threshold),
        "selected_accept_margin": float(cfg.accept_margin),
        "validation_auroc": float("nan"),
        "validation_fpr": float("nan"),
        "validation_fnr": float("nan"),
    }
    train_meta: Dict[str, object] = {}
    if cfg.pipeline == "learned_grounding":
        if cfg.repair_variant == "contrastive":
            raise ValueError("learned_grounding pipeline supports threshold_null, margin_gate, or linear repairs; contrastive gate is exact-only")
        writer, tok, train_meta = build_learned_pipeline(cfg, field, device=dev)
        if cfg.calibration_split == "validation" and cfg.repair_variant in {"threshold_null", "margin_gate"}:
            calibration_meta = calibrate_learned_pipeline(cfg, field, writer, tok, device=dev)
            cfg = replace(
                cfg,
                reject_threshold=float(calibration_meta["selected_reject_threshold"]),
                accept_margin=float(calibration_meta["selected_accept_margin"]),
            )
        rows = evaluate_learned_repairs(cfg, field, writer, tok, calibration_meta, device=dev)
        pipeline_implementation = "learned_extractor_memory"
    else:
        if cfg.repair_variant == "contrastive":
            gate, gate_meta = train_contrastive_gate(cfg, field, dev)
        rows = evaluate_repairs(cfg, field, gate, device=dev)
    meta = {
        "suite": "closure_key_rejection_repairs",
        "config": asdict(cfg),
        "pipeline_implementation": pipeline_implementation,
        "calibration": calibration_meta,
        "gate": gate_meta,
        "train": train_meta,
        "elapsed_total_sec": time.perf_counter() - t0,
    }
    paths = write_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate wrong-key rejection repairs for rank-one closure reads.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=RepairConfig.seed)
    p.add_argument("--train-steps", type=int, default=RepairConfig.train_steps)
    p.add_argument("--batch-size", type=int, default=RepairConfig.batch_size)
    p.add_argument("--eval-n", type=int, default=RepairConfig.eval_n)
    p.add_argument("--eval-batch-size", type=int, default=RepairConfig.eval_batch_size)
    p.add_argument("--d-model", type=int, default=RepairConfig.d_model)
    p.add_argument("--layers", type=int, default=RepairConfig.n_layers)
    p.add_argument("--heads", type=int, default=RepairConfig.n_heads)
    p.add_argument("--key-dim", type=int, default=RepairConfig.key_dim)
    p.add_argument("--num-entities", type=int, default=RepairConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=RepairConfig.num_relations)
    p.add_argument("--curriculum", choices=["staged", "mixed"], default=RepairConfig.curriculum)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default=RepairConfig.lr_schedule)
    p.add_argument("--warmup-frac", type=float, default=RepairConfig.warmup_frac)
    p.add_argument("--min-lr-ratio", type=float, default=RepairConfig.min_lr_ratio)
    p.add_argument("--repair-variant", choices=sorted(VALID_REPAIRS), default=RepairConfig.repair_variant)
    p.add_argument("--pipeline", choices=["exact_rank_one", "learned_grounding"], default=RepairConfig.pipeline)
    p.add_argument("--wrong-key-families", type=str, default=",".join(RepairConfig.wrong_key_families))
    p.add_argument("--calibration-split", choices=["validation", "fixed_threshold"], default=RepairConfig.calibration_split)
    p.add_argument("--contrastive-weight", type=float, default=RepairConfig.contrastive_weight)
    p.add_argument("--reject-threshold", type=float, default=RepairConfig.reject_threshold)
    p.add_argument("--accept-margin", type=float, default=RepairConfig.accept_margin)
    p.add_argument("--learning-rate", type=float, default=RepairConfig.learning_rate)
    p.add_argument("--threads", type=int, default=RepairConfig.torch_threads)
    p.add_argument("--eval-lengths", type=str, default="1,2,3,4,6,8,12,16,24,32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RepairConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        curriculum=args.curriculum,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        min_lr_ratio=args.min_lr_ratio,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        repair_variant=args.repair_variant,
        pipeline=args.pipeline,
        wrong_key_families=parse_name_list(args.wrong_key_families),
        calibration_split=args.calibration_split,
        contrastive_weight=args.contrastive_weight,
        reject_threshold=args.reject_threshold,
        accept_margin=args.accept_margin,
        learning_rate=args.learning_rate,
        torch_threads=args.threads,
        eval_lengths=parse_int_list(args.eval_lengths),
    )
    result = run_experiment(cfg, Path(args.out_dir), device=args.device)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
