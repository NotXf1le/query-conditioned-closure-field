"""Research-suite runner for learned grounding plus structured closure.

This file does not introduce a new claim.  It exposes the knobs needed for
paper-grade ablations around the learned-grounding closure pipeline:

    learned text grounding -> transition tensor A[r,s,t]
    -> explicit semiring path closure -> m_closure
    -> exactly one HolographicClosureField read

The main purpose is reproducibility and falsification: many seeds, larger eval
sets, supervised/unsupervised grounding ablations, key-size sweeps, ontology
size sweeps, distractor-density sweeps, and curriculum/noise/alias controls.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from generic_closure_writer import (
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    accuracy_from_logits,
)
from structured_transition_closure import SemiringClosureWriter
from learned_grounding_closure import (
    LearnedGroundingTextTokenizer,
    LearnedExtractorClosureWriter,
    collate_learned_grounding_examples,
    extractor_supervision_loss,
    evaluate_learned_grounding,
    write_learned_grounding_results,
)


def train_custom_learned_extractor_writer(
    cfg: ClosureWriterConfig,
    tok: LearnedGroundingTextTokenizer,
    field: HolographicClosureField,
    *,
    device: torch.device | str = "cpu",
    train_steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    extraction_weight: float = 2.0,
    answer_weight: float = 1.0,
    alias_train_prob: float = 0.35,
    noise_train_prob: float = 0.15,
    fact_shuffle_train_prob: float = 0.20,
    use_mlp_extractor: bool = False,
    hard_eval_extraction: bool = True,
    output_scale: float = 30.0,
    normalize_frontier: bool = True,
    write_prefixes: bool = False,
) -> Tuple[LearnedExtractorClosureWriter, Dict[str, object]]:
    """Train the learned-grounding writer with explicit ablation knobs.

    extraction_weight=0 gives the answer-only weak-supervision ablation.  This
    often fails; that is useful evidence, not an error.
    """
    device = torch.device(device)
    writer = LearnedExtractorClosureWriter(
        tok,
        cfg,
        write_prefixes=write_prefixes,
        output_scale=output_scale,
        normalize_frontier=normalize_frontier,
        hard_eval_extraction=hard_eval_extraction,
        use_mlp_extractor=use_mlp_extractor,
    ).to(device)
    opt = torch.optim.AdamW(writer.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 8101)
    rng = random.Random(cfg.seed + 8202)
    t0 = time.perf_counter()
    snapshots: List[Dict[str, float]] = []
    for step in range(1, int(train_steps) + 1):
        if step <= max(1, int(0.20 * train_steps)):
            choices = (1,)
        elif step <= max(1, int(0.50 * train_steps)):
            choices = (1, 2)
        else:
            choices = (1, 2, 3)
        examples = [gen.make_example(rng.choice(choices)) for _ in range(int(batch_size))]
        batch = collate_learned_grounding_examples(
            examples,
            tok,
            cfg,
            device=device,
            rng=rng,
            relation_aliases=(rng.random() < alias_train_prob),
            extra_text_noise=(rng.random() < noise_train_prob),
            fact_order_shuffle=(rng.random() < fact_shuffle_train_prob),
        )
        out = writer(batch, field)
        answer_loss = F.cross_entropy(out["logits"], batch["target"])
        extract_loss, extract_stats = extractor_supervision_loss(out, batch)
        loss = float(answer_weight) * answer_loss + float(extraction_weight) * extract_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg.grad_clip)
        opt.step()
        if step == 1 or step % max(1, train_steps // 6) == 0 or step == train_steps:
            c, n = accuracy_from_logits(out["logits"].detach(), batch["target"])
            snap: Dict[str, float] = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "answer_loss": float(answer_loss.detach().item()),
                "extractor_loss": float(extract_loss.detach().item()),
                "train_batch_acc": float(c / max(1, n)),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
            for k, v in extract_stats.items():
                snap[k] = float(v)
            snapshots.append(snap)
            print(json.dumps({"grounding_ablation_train_progress": snap}, sort_keys=True), flush=True)
    meta = {
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "answer_weight": float(answer_weight),
        "extraction_weight": float(extraction_weight),
        "alias_train_prob": float(alias_train_prob),
        "noise_train_prob": float(noise_train_prob),
        "fact_shuffle_train_prob": float(fact_shuffle_train_prob),
        "use_mlp_extractor": bool(use_mlp_extractor),
        "hard_eval_extraction": bool(hard_eval_extraction),
        "output_scale": float(output_scale),
        "normalize_frontier": bool(normalize_frontier),
        "write_prefixes": bool(write_prefixes),
        "snapshots": snapshots,
        "elapsed_train_sec": float(time.perf_counter() - t0),
    }
    return writer, meta


def run_grounding_ablation(cfg: ClosureWriterConfig, out_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=args.relation_aliases)
    field = HolographicClosureField(
        cfg.num_entities,
        cfg.num_relations,
        cfg.key_dim,
        cfg.max_path_len,
        seed=cfg.seed + 11,
        read_scale=1.0,
    ).to(device)
    learned, train_meta = train_custom_learned_extractor_writer(
        cfg,
        tok,
        field,
        device=device,
        train_steps=args.train_steps,
        batch_size=cfg.batch_size,
        learning_rate=args.learning_rate,
        extraction_weight=args.extraction_weight,
        answer_weight=args.answer_weight,
        alias_train_prob=args.alias_train_prob,
        noise_train_prob=args.noise_train_prob,
        fact_shuffle_train_prob=args.fact_shuffle_train_prob,
        use_mlp_extractor=args.use_mlp_extractor,
        hard_eval_extraction=not args.soft_eval_extraction,
        output_scale=args.output_scale,
        normalize_frontier=not args.no_normalize_frontier,
        write_prefixes=args.write_prefixes,
    )

    from generic_closure_writer import ClosureTextTokenizer
    generic_tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    dp = SemiringClosureWriter(generic_tok, cfg, learn_relation_match=False, write_prefixes=False, output_scale=20.0).to(device)
    rows = evaluate_learned_grounding(cfg, tok, field, learned, dp, device=device)
    meta = {
        "suite": "grounding_ablation",
        "config": asdict(cfg),
        "relation_aliases": int(args.relation_aliases),
        "ablation": train_meta,
        "num_parameters": {"learned_extractor_writer": sum(p.numel() for p in learned.parameters())},
    }
    paths = write_learned_grounding_results(rows, meta, out_dir)
    (out_dir / "RUN_META_GROUNDING_ABLATION.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run grounding ablations: learned extractor + explicit closure + one read.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=7601)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
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
    p.add_argument("--use-mlp-extractor", action="store_true")
    p.add_argument("--soft-eval-extraction", action="store_true")
    p.add_argument("--no-normalize-frontier", action="store_true")
    p.add_argument("--write-prefixes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ClosureWriterConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
        d_model=args.d_model,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        torch_threads=args.threads,
        same_relation_branch_prob=args.same_relation_branch_prob,
        max_seq_len=args.max_seq_len,
        base_distractors=args.base_distractors,
        distractors_per_hop=args.distractors_per_hop,
    )
    t0 = time.perf_counter()
    result = run_grounding_ablation(cfg, out_dir, args)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": time.perf_counter() - t0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
