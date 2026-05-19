"""Permutation-aware relation grounding diagnostics.

Canonical relation accuracy can be zero even when a model has learned a
systematic, permuted relation code. This script trains the existing learned
grounding + explicit closure model under ablation settings and reports both
canonical and optimal-permutation relation metrics.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from generic_closure_writer import ClosureWriterConfig, ControlledDenseGraphTextQAGenerator
from learned_grounding_closure import (
    HolographicClosureField,
    LearnedGroundingTextTokenizer,
    collate_learned_grounding_examples,
)
from grounding_ablation_suite import train_custom_learned_extractor_writer


PERM_RESULT_JSON = "GROUNDING_PERMUTATION_DIAGNOSTICS_RESULTS.json"
PERM_RESULT_CSV = "GROUNDING_PERMUTATION_DIAGNOSTICS_RESULTS.csv"
PERM_REPORT = "GROUNDING_PERMUTATION_DIAGNOSTICS_REPORT.md"
VALID_CONDITIONS = {"supervised", "answer_only_no_slot", "extraction_weight_zero"}


def confusion_from_predictions(pred: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor], num_classes: int) -> torch.Tensor:
    p = pred.detach().cpu().reshape(-1)
    y = labels.detach().cpu().reshape(-1)
    if mask is not None:
        m = mask.detach().cpu().reshape(-1).bool()
        p = p[m]
        y = y[m]
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, got in zip(y.tolist(), p.tolist()):
        if 0 <= int(true) < num_classes and 0 <= int(got) < num_classes:
            conf[int(true), int(got)] += 1
    return conf


def permutation_aligned_accuracy(conf: torch.Tensor) -> Tuple[float, List[int]]:
    n = int(conf.sum().item())
    if n == 0:
        return float("nan"), []
    r = int(conf.shape[0])
    best = -1
    best_perm: Tuple[int, ...] = tuple(range(r))
    for perm in itertools.permutations(range(r)):
        score = sum(int(conf[i, perm[i]].item()) for i in range(r))
        if score > best:
            best = score
            best_perm = tuple(int(x) for x in perm)
    return float(best / n), list(best_perm)


def mutual_information_from_confusion(conf: torch.Tensor) -> float:
    total = float(conf.sum().item())
    if total <= 0:
        return float("nan")
    pxy = conf.to(torch.float64) / total
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    mi = 0.0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            p = float(pxy[i, j].item())
            if p > 0:
                mi += p * math.log(p / max(1e-12, float(px[i, 0].item()) * float(py[0, j].item())))
    return float(mi / max(1e-12, math.log(conf.shape[0])))


def canonical_accuracy(conf: torch.Tensor) -> float:
    total = int(conf.sum().item())
    if total == 0:
        return float("nan")
    return float(conf.diag().sum().item() / total)


def condition_weights(condition: str, default_extraction_weight: float) -> float:
    if condition == "supervised":
        return float(default_extraction_weight)
    if condition in {"answer_only_no_slot", "extraction_weight_zero"}:
        return 0.0
    raise ValueError(condition)


@torch.no_grad()
def evaluate_permutation(
    cfg: ClosureWriterConfig,
    tok: LearnedGroundingTextTokenizer,
    field: HolographicClosureField,
    writer,
    *,
    condition: str,
    eval_lengths: Sequence[int],
    device: torch.device,
) -> List[Dict[str, float | str]]:
    writer.eval()
    rows: List[Dict[str, float | str]] = []
    for length in eval_lengths:
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 3000 + int(length))
        examples = gen.make_examples(int(length), cfg.eval_n)
        fact_conf = torch.zeros(cfg.num_relations, cfg.num_relations, dtype=torch.long)
        query_conf = torch.zeros(cfg.num_relations, cfg.num_relations, dtype=torch.long)
        answer_correct = answer_n = 0
        for start in range(0, len(examples), cfg.eval_batch_size):
            batch_examples = examples[start : start + cfg.eval_batch_size]
            batch = collate_learned_grounding_examples(batch_examples, tok, cfg, device=device)
            out = writer(batch, field)
            pred = out["logits"].argmax(dim=-1)
            answer_correct += int(pred.eq(batch["target"]).sum().item())
            answer_n += int(pred.numel())
            ext = out["extractor"]
            fact_pred = ext["fact_relation_logits"].argmax(dim=-1)
            query_pred = ext["query_relation_logits"].argmax(dim=-1)
            fact_conf += confusion_from_predictions(fact_pred, batch["fact_relation_label"], batch["fact_mask"], cfg.num_relations)
            query_conf += confusion_from_predictions(query_pred, batch["q_rels"], batch["query_relation_mask"], cfg.num_relations)
        fact_perm_acc, fact_perm = permutation_aligned_accuracy(fact_conf)
        query_perm_acc, query_perm = permutation_aligned_accuracy(query_conf)
        rows.append({
            "condition": condition,
            "length": float(length),
            "n": float(answer_n),
            "answer_acc": float(answer_correct / max(1, answer_n)),
            "answer_n": float(answer_n),
            "fact_relation_canonical_acc": canonical_accuracy(fact_conf),
            "fact_relation_canonical_n": float(fact_conf.sum().item()),
            "fact_relation_permutation_acc": fact_perm_acc,
            "fact_relation_permutation_n": float(fact_conf.sum().item()),
            "fact_relation_mutual_info": mutual_information_from_confusion(fact_conf),
            "query_relation_canonical_acc": canonical_accuracy(query_conf),
            "query_relation_canonical_n": float(query_conf.sum().item()),
            "query_relation_permutation_acc": query_perm_acc,
            "query_relation_permutation_n": float(query_conf.sum().item()),
            "query_relation_mutual_info": mutual_information_from_confusion(query_conf),
            "fact_relation_best_permutation": json.dumps(fact_perm),
            "query_relation_best_permutation": json.dumps(query_perm),
        })
    return rows


def write_results(rows: List[Dict[str, float | str]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / PERM_RESULT_JSON
    csv_path = out_dir / PERM_RESULT_CSV
    report_path = out_dir / PERM_REPORT
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text("# Grounding Permutation Diagnostics\n\nCanonical and permutation-aligned relation metrics.\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    if args.condition not in VALID_CONDITIONS:
        raise ValueError(args.condition)
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = ClosureWriterConfig(
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
        curriculum=args.curriculum,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        min_lr_ratio=args.min_lr_ratio,
        torch_threads=args.threads,
        max_seq_len=args.max_seq_len,
    )
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=args.relation_aliases)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(device)
    extraction_weight = condition_weights(args.condition, args.extraction_weight)
    t0 = time.perf_counter()
    writer, train_meta = train_custom_learned_extractor_writer(
        cfg,
        tok,
        field,
        device=device,
        train_steps=args.train_steps,
        batch_size=cfg.batch_size,
        learning_rate=args.learning_rate,
        extraction_weight=extraction_weight,
        answer_weight=args.answer_weight,
        use_mlp_extractor=args.use_mlp_extractor,
    )
    rows = evaluate_permutation(
        cfg,
        tok,
        field,
        writer,
        condition=args.condition,
        eval_lengths=tuple(int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()),
        device=device,
    )
    meta = {
        "suite": "grounding_permutation_diagnostics",
        "condition": args.condition,
        "config": asdict(cfg),
        "extraction_weight": extraction_weight,
        "train": train_meta,
        "elapsed_total_sec": time.perf_counter() - t0,
    }
    paths = write_results(rows, meta, Path(args.out_dir))
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run permutation-aware grounding diagnostics.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--condition", choices=sorted(VALID_CONDITIONS), default="answer_only_no_slot")
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--num-entities", type=int, default=48)
    p.add_argument("--num-relations", type=int, default=4)
    p.add_argument("--curriculum", choices=["staged", "mixed"], default="staged")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--extraction-weight", type=float, default=2.0)
    p.add_argument("--relation-aliases", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=2200)
    p.add_argument("--use-mlp-extractor", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval-lengths", type=str, default="1,2,3,4,6,8,12,16,24,32")
    return p.parse_args()


def main() -> None:
    result = run_experiment(parse_args())
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
