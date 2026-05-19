"""Held-out relation-order split challenge.

This script tests whether the learned extractor + explicit closure writer can
handle relation orders that were deliberately absent from training.  It is not a
new memory mechanism; it is a stronger evaluation split for the controlled
learned-grounding closure pipeline.

Split rules:
  adjacent_pair_holdout: training examples exclude any adjacent relation pair in
      {(0,1), (1,2), (2,3 mod R)}; evaluation examples require at least one.
  trigram_holdout: training examples exclude exact relation trigrams
      {(0,1,2), (1,2,3 mod R)}; evaluation examples require at least one.
  repeated_relation_holdout: training examples exclude adjacent repeats; eval
      requires at least one adjacent repeat.

The final answer is still exactly one HolographicClosureField read.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from generic_closure_writer import (
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    accuracy_from_logits,
    exact_dict_answer,
    raw_onehop_answer,
    symbolic_accuracy,
    iter_minibatches,
)
from structured_transition_closure import SemiringClosureWriter
from learned_grounding_closure import (
    LearnedGroundingTextTokenizer,
    LearnedExtractorClosureWriter,
    collate_learned_grounding_examples,
    extractor_supervision_loss,
    extractor_slot_metrics,
)


def heldout_pairs(num_relations: int) -> set[Tuple[int, int]]:
    return {(0 % num_relations, 1 % num_relations), (1 % num_relations, 2 % num_relations), (2 % num_relations, 3 % num_relations)}


def heldout_trigrams(num_relations: int) -> set[Tuple[int, int, int]]:
    return {(0 % num_relations, 1 % num_relations, 2 % num_relations), (1 % num_relations, 2 % num_relations, 3 % num_relations)}


def has_heldout_pattern(rels: Sequence[int], split: str, num_relations: int) -> bool:
    r = tuple(int(x) for x in rels)
    if split == "adjacent_pair_holdout":
        hp = heldout_pairs(num_relations)
        return any((r[i], r[i + 1]) in hp for i in range(max(0, len(r) - 1)))
    if split == "trigram_holdout":
        ht = heldout_trigrams(num_relations)
        return any((r[i], r[i + 1], r[i + 2]) in ht for i in range(max(0, len(r) - 2)))
    if split == "repeated_relation_holdout":
        return any(r[i] == r[i + 1] for i in range(max(0, len(r) - 1)))
    raise ValueError(f"unknown split {split}")


def sample_relations_for_split(
    length: int,
    split: str,
    num_relations: int,
    want_heldout: bool,
    rng: random.Random,
) -> Tuple[int, ...]:
    L = int(length)
    R = int(num_relations)
    if L < 1:
        raise ValueError(f"length must be positive, got {length}")
    if R < 1:
        raise ValueError(f"num_relations must be positive, got {num_relations}")

    if want_heldout:
        rels = [rng.randrange(R) for _ in range(L)]
        if split == "adjacent_pair_holdout":
            if L < 2:
                raise ValueError("adjacent_pair_holdout requires length >= 2 for held-out examples")
            pos = rng.randrange(L - 1)
            a, b = rng.choice(tuple(sorted(heldout_pairs(R))))
            rels[pos], rels[pos + 1] = a, b
        elif split == "trigram_holdout":
            if L < 3:
                raise ValueError("trigram_holdout requires length >= 3 for held-out examples")
            pos = rng.randrange(L - 2)
            a, b, c = rng.choice(tuple(sorted(heldout_trigrams(R))))
            rels[pos], rels[pos + 1], rels[pos + 2] = a, b, c
        elif split == "repeated_relation_holdout":
            if L < 2:
                raise ValueError("repeated_relation_holdout requires length >= 2 for held-out examples")
            pos = rng.randrange(L - 1)
            r = rng.randrange(R)
            rels[pos], rels[pos + 1] = r, r
        else:
            raise ValueError(f"unknown split {split}")
        out = tuple(int(x) for x in rels)
        if not has_heldout_pattern(out, split, R):
            raise RuntimeError(f"failed to construct held-out relation sequence for {split}")
        return out

    rels: List[int] = []
    if split == "adjacent_pair_holdout":
        blocked = heldout_pairs(R)
        for pos in range(L):
            choices = list(range(R))
            if pos > 0:
                choices = [r for r in choices if (rels[-1], r) not in blocked]
            if not choices:
                raise ValueError(f"cannot sample train-like adjacent-pair sequence with num_relations={R}")
            rels.append(rng.choice(choices))
    elif split == "trigram_holdout":
        blocked = heldout_trigrams(R)
        for pos in range(L):
            choices = list(range(R))
            if pos > 1:
                choices = [r for r in choices if (rels[-2], rels[-1], r) not in blocked]
            if not choices:
                raise ValueError(f"cannot sample train-like trigram sequence with num_relations={R}")
            rels.append(rng.choice(choices))
    elif split == "repeated_relation_holdout":
        for pos in range(L):
            choices = list(range(R))
            if pos > 0:
                choices = [r for r in choices if r != rels[-1]]
            if not choices:
                raise ValueError(f"cannot sample train-like repeated-relation sequence with num_relations={R}")
            rels.append(rng.choice(choices))
    else:
        raise ValueError(f"unknown split {split}")

    out = tuple(int(x) for x in rels)
    if has_heldout_pattern(out, split, R):
        raise RuntimeError(f"failed to construct train-like relation sequence for {split}")
    return out


def make_split_examples(
    gen: ControlledDenseGraphTextQAGenerator,
    length: int,
    n: int,
    *,
    split: str,
    want_heldout: bool,
    num_relations: int,
    rng: random.Random,
):
    out = []
    for _ in range(int(n)):
        rels = sample_relations_for_split(length, split, num_relations, want_heldout, rng)
        ex = gen.make_example(int(length), relations=rels)
        if has_heldout_pattern(ex.relations, split, num_relations) != bool(want_heldout):
            raise RuntimeError(f"generated example does not match split={split}, want_heldout={want_heldout}")
        out.append(ex)
    return out


def make_filtered_examples(
    gen: ControlledDenseGraphTextQAGenerator,
    length: int,
    n: int,
    predicate: Callable[[Sequence[int]], bool],
    *,
    max_tries: int = 200000,
):
    out = []
    tries = 0
    while len(out) < int(n) and tries < max_tries:
        tries += 1
        ex = gen.make_example(int(length))
        if predicate(ex.relations):
            out.append(ex)
    if len(out) < int(n):
        raise RuntimeError(f"only generated {len(out)}/{n} filtered examples for L={length}")
    return out


def train_split_model(cfg: ClosureWriterConfig, tok: LearnedGroundingTextTokenizer, field: HolographicClosureField, args: argparse.Namespace):
    device = torch.device(args.device)
    writer = LearnedExtractorClosureWriter(tok, cfg, output_scale=args.output_scale, hard_eval_extraction=not args.soft_eval_extraction).to(device)
    opt = torch.optim.AdamW(writer.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 1701)
    rng = random.Random(cfg.seed + 1702)
    snapshots: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        if step <= max(1, int(0.20 * args.train_steps)):
            choices = (1,)
        elif step <= max(1, int(0.50 * args.train_steps)):
            choices = (1, 2)
        else:
            choices = (1, 2, 3)
        examples = [
            gen.make_example(
                L,
                relations=sample_relations_for_split(L, args.split, cfg.num_relations, False, rng),
            )
            for L in (rng.choice(choices) for _ in range(args.batch_size))
        ]
        batch = collate_learned_grounding_examples(
            examples, tok, cfg, device=device, rng=rng,
            relation_aliases=(rng.random() < args.alias_train_prob),
            extra_text_noise=(rng.random() < args.noise_train_prob),
            fact_order_shuffle=(rng.random() < args.fact_shuffle_train_prob),
        )
        out = writer(batch, field)
        answer_loss = F.cross_entropy(out["logits"], batch["target"])
        extract_loss, stats = extractor_supervision_loss(out, batch)
        loss = args.answer_weight * answer_loss + args.extraction_weight * extract_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg.grad_clip)
        opt.step()
        if step == 1 or step % max(1, args.train_steps // 6) == 0 or step == args.train_steps:
            c, n = accuracy_from_logits(out["logits"].detach(), batch["target"])
            snap = {
                "step": float(step), "loss": float(loss.detach().item()),
                "answer_loss": float(answer_loss.detach().item()),
                "extractor_loss": float(extract_loss.detach().item()),
                "train_batch_acc": float(c / max(1, n)),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
            snap.update({k: float(v) for k, v in stats.items()})
            snapshots.append(snap)
            print(json.dumps({"heldout_order_train_progress": snap}, sort_keys=True), flush=True)
    return writer, {"snapshots": snapshots, "elapsed_train_sec": time.perf_counter() - t0}


@torch.no_grad()
def evaluate_split(cfg: ClosureWriterConfig, tok: LearnedGroundingTextTokenizer, field: HolographicClosureField, writer: LearnedExtractorClosureWriter, args: argparse.Namespace):
    device = torch.device(args.device)
    writer.eval(); field.eval()
    from generic_closure_writer import ClosureTextTokenizer, collate_examples as collate_generic_examples
    generic_tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    dp = SemiringClosureWriter(generic_tok, cfg, learn_relation_match=False, write_prefixes=False, output_scale=20.0).to(device)
    dp.eval()
    rng = random.Random(cfg.seed + 1803)
    rows: List[Dict[str, float]] = []

    # L=1 cannot satisfy pair/trigram/repeat holdouts.  We still report trainlike.
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    for L in lengths:
        gen_hard = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 1900 + L)
        gen_trainlike = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 2100 + L)
        if L == 1 or (args.split == "trigram_holdout" and L < 3):
            hard_examples = []
        else:
            hard_examples = make_split_examples(
                gen_hard,
                L,
                cfg.eval_n,
                split=args.split,
                want_heldout=True,
                num_relations=cfg.num_relations,
                rng=rng,
            )
        trainlike_examples = make_split_examples(
            gen_trainlike,
            L,
            cfg.eval_n,
            split=args.split,
            want_heldout=False,
            num_relations=cfg.num_relations,
            rng=rng,
        )

        def eval_group(name: str, examples):
            counts: Dict[str, List[int]] = {}
            def add(metric: str, c: int, n: int):
                counts.setdefault(metric, [0, 0])
                counts[metric][0] += int(c); counts[metric][1] += int(n)
            if not examples:
                return {}
            add(f"{name}_exact_dict", *symbolic_accuracy(examples, exact_dict_answer))
            add(f"{name}_raw_onehop", *symbolic_accuracy(examples, raw_onehop_answer))
            for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
                batch = collate_learned_grounding_examples(batch_examples, tok, cfg, device=device, rng=rng)
                out = writer(batch, field)
                add(f"{name}_learned_extractor_writer", *accuracy_from_logits(out["logits"], batch["target"]))
                for k, (c, n) in extractor_slot_metrics(out, batch).items():
                    add(f"{name}_extractor_{k}", c, n)
                generic_batch = collate_generic_examples(batch_examples, generic_tok, cfg, device=device, rng=rng)
                add(f"{name}_dp_parser_semiring_writer", *accuracy_from_logits(dp(generic_batch, field)["logits"], generic_batch["target"]))
                for variant, metric in [("no_facts", "no_facts"), ("wrong_facts", "wrong_facts"), ("reversed_order", "reversed_order"), ("shuffled_order", "shuffled_order")]:
                    if variant == "wrong_facts":
                        wrong_gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 2300 + L)
                        wrong = make_split_examples(
                            wrong_gen,
                            L,
                            len(batch_examples),
                            split=args.split,
                            want_heldout=(name == "hard"),
                            num_relations=cfg.num_relations,
                            rng=rng,
                        )
                        vb = collate_learned_grounding_examples(batch_examples, tok, cfg, variant=variant, wrong_fact_examples=wrong, device=device, rng=rng)
                    else:
                        vb = collate_learned_grounding_examples(batch_examples, tok, cfg, variant=variant, device=device, rng=rng)
                    logits = writer(vb, field)["logits"]
                    if variant in {"reversed_order", "shuffled_order"}:
                        mask = vb["control_changed"]
                        if bool(mask.any()):
                            add(f"{name}_{metric}", *accuracy_from_logits(logits, vb["target"], mask=mask))
                    else:
                        add(f"{name}_{metric}", *accuracy_from_logits(logits, vb["target"]))
            return {f"{k}_acc": float(c / n) if n else float("nan") for k, (c, n) in counts.items()} | {f"{k}_n": float(n) for k, (c, n) in counts.items()}

        row: Dict[str, float] = {"length": float(L), "hard_n": float(len(hard_examples)), "trainlike_n": float(len(trainlike_examples))}
        row.update(eval_group("hard", hard_examples))
        row.update(eval_group("trainlike", trainlike_examples))
        rows.append(row)
        print(json.dumps({"heldout_order_eval_by_length": {k: v for k, v in row.items() if k.endswith("_acc") or k in {"length", "hard_n", "trainlike_n"}}}, sort_keys=True), flush=True)
    return rows


def write_split_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "HELDOUT_RELATION_ORDER_RESULTS.json"
    csv_path = out_dir / "HELDOUT_RELATION_ORDER_RESULTS.csv"
    report_path = out_dir / "HELDOUT_RELATION_ORDER_REPORT.md"
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    lines = [
        "# Held-out relation-order split",
        "",
        "Training excludes the held-out relation-order pattern. Evaluation reports train-like examples and hard examples that require the held-out pattern. Results are by length only.",
        "",
        "Split: `" + str(meta.get("split")) + "`",
        "",
    ]
    cols = ["length", "hard_n", "hard_learned_extractor_writer_acc", "hard_dp_parser_semiring_writer_acc", "hard_raw_onehop_acc", "hard_no_facts_acc", "hard_wrong_facts_acc", "hard_reversed_order_acc", "hard_shuffled_order_acc", "trainlike_learned_extractor_writer_acc"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, float("nan"))
            if c in {"length", "hard_n"}:
                vals.append(str(int(v)) if v == v else "NA")
            else:
                vals.append("NA" if v != v else f"{float(v):.3f}")
        lines.append("| " + " | ".join(vals) + " |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def parse_args():
    p = argparse.ArgumentParser(description="Run held-out relation-order compositional split.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--split", type=str, default="adjacent_pair_holdout", choices=["adjacent_pair_holdout", "trigram_holdout", "repeated_relation_holdout"])
    p.add_argument("--seed", type=int, default=8801)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lengths", type=str, default="1,2,3,4,6,8,12,16,24,32")
    p.add_argument("--num-entities", type=int, default=48)
    p.add_argument("--num-relations", type=int, default=4)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=2200)
    p.add_argument("--base-distractors", type=int, default=6)
    p.add_argument("--distractors-per-hop", type=int, default=3)
    p.add_argument("--same-relation-branch-prob", type=float, default=0.25)
    p.add_argument("--relation-aliases", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--extraction-weight", type=float, default=2.0)
    p.add_argument("--alias-train-prob", type=float, default=0.35)
    p.add_argument("--noise-train-prob", type=float, default=0.15)
    p.add_argument("--fact-shuffle-train-prob", type=float, default=0.20)
    p.add_argument("--output-scale", type=float, default=30.0)
    p.add_argument("--soft-eval-extraction", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ClosureWriterConfig(
        seed=args.seed, num_entities=args.num_entities, num_relations=args.num_relations,
        key_dim=args.key_dim, d_model=args.d_model, train_steps=args.train_steps,
        batch_size=args.batch_size, eval_n=args.eval_n, eval_batch_size=args.eval_batch_size,
        torch_threads=args.threads, same_relation_branch_prob=args.same_relation_branch_prob,
        max_seq_len=args.max_seq_len, base_distractors=args.base_distractors,
        distractors_per_hop=args.distractors_per_hop,
    )
    device = torch.device(args.device)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=args.relation_aliases)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(device)
    t0 = time.perf_counter()
    writer, train_meta = train_split_model(cfg, tok, field, args)
    rows = evaluate_split(cfg, tok, field, writer, args)
    meta = {"suite": "heldout_relation_order", "split": args.split, "config": asdict(cfg), "train": train_meta, "args": vars(args), "elapsed_total_sec": time.perf_counter() - t0}
    paths = write_split_results(rows, meta, out_dir)
    print(json.dumps({"status": "done", "paths": paths, "elapsed_total_sec": meta["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
