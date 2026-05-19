"""Anti-shortcut and causal-control suite.

This file does not introduce a new architecture.  It trains the controlled
pipeline (learned slot grounding + explicit transition/semiring closure + one
HolographicClosureField read) and then performs stronger falsification tests:

* permutation-aligned relation grounding diagnostics;
* relation/query counterfactuals;
* fact/edge counterfactuals;
* minimal pairs;
* memory-state interventions;
* closure-field key exactness checks;
* text-scanned slot positions instead of externally supplied role positions;
* gold-edge deletion and distractor deletion;
* same-multiset order sensitivity and repeated-relation stress.

The point is to distinguish a real closure mechanism from shortcuts.  Answer-only
training can learn a latent relation coding or a generator shortcut; therefore
no-slot-supervision success is reported as a diagnostic, not as the central
interpretable-grounding claim.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from generic_closure_writer import (
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    PathQAExample,
    accuracy_from_logits,
    exact_dict_answer,
    iter_minibatches,
    raw_onehop_answer,
    symbolic_accuracy,
    _different_permutation,
)
from learned_grounding_closure import (
    LearnedGroundingTextTokenizer,
    LearnedExtractorClosureWriter,
    collate_learned_grounding_examples,
    extractor_slot_metrics,
    extractor_supervision_loss,
    train_learned_extractor_writer,
)

Edge = Tuple[int, int, int]


def endpoint_set(source: int, relations: Sequence[int], edges: Sequence[Edge]) -> set[int]:
    frontier = {int(source)}
    for r in relations:
        nxt: set[int] = set()
        rr = int(r)
        for s, er, t in edges:
            if int(er) == rr and int(s) in frontier:
                nxt.add(int(t))
        frontier = nxt
        if not frontier:
            break
    return frontier


def unused_entities(ex: PathQAExample, cfg: ClosureWriterConfig, rng: random.Random, k: int, avoid: Sequence[int] = ()) -> List[int]:
    used = set(int(x) for x in ex.path_nodes) | set(int(x) for x in avoid)
    pool = [e for e in range(cfg.num_entities) if e not in used]
    rng.shuffle(pool)
    if len(pool) < k:
        # Fall back to allowing non-path entities that may already appear in distractors.
        pool = [e for e in range(cfg.num_entities) if e not in set(int(x) for x in avoid)]
        rng.shuffle(pool)
    if len(pool) < k:
        raise RuntimeError("not enough entities for counterfactual branch")
    return pool[:k]


def make_query_counterfactual(
    ex: PathQAExample,
    new_relations: Sequence[int],
    cfg: ClosureWriterConfig,
    rng: random.Random,
    *,
    max_tries: int = 50,
) -> Optional[PathQAExample]:
    """Keep source and most facts, alter the query relation sequence, and add a
    unique branch so the counterfactual has a different deterministic endpoint.

    This tests whether the model follows relation identity/order rather than the
    original source or answer prior.  At the first changed relation, we remove
    competing outgoing edges with that relation from the pivot node, then add a
    new branch for the counterfactual suffix.
    """
    old = tuple(int(r) for r in ex.relations)
    new = tuple(int(r) for r in new_relations)
    if len(old) != len(new) or new == old:
        return None
    L = len(old)
    first_diff = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None)
    if first_diff is None:
        return None
    pivot = int(ex.path_nodes[first_diff])
    for _ in range(max_tries):
        branch_nodes = unused_entities(ex, cfg, rng, L - first_diff, avoid=[ex.target])
        full_nodes = tuple(list(ex.path_nodes[: first_diff + 1]) + branch_nodes)
        edges = []
        # Remove potentially competing edges at the counterfactual divergence.
        for s, r, t in ex.edges:
            if int(s) == pivot and int(r) == int(new[first_diff]):
                continue
            edges.append((int(s), int(r), int(t)))
        for j in range(first_diff, L):
            edges.append((int(full_nodes[j]), int(new[j]), int(full_nodes[j + 1])))
        target = int(full_nodes[-1])
        if target == ex.target:
            continue
        cf = PathQAExample(
            source=int(ex.source),
            relations=new,
            target=target,
            path_nodes=full_nodes,
            edges=tuple(dict.fromkeys(edges)),
            attempts=ex.attempts,
        )
        if exact_dict_answer(cf) == cf.target and cf.target != ex.target:
            return cf
    return None


def make_relation_swap_counterfactual(ex: PathQAExample, cfg: ClosureWriterConfig, rng: random.Random) -> Optional[PathQAExample]:
    if len(ex.relations) < 1 or cfg.num_relations < 2:
        return None
    positions = list(range(len(ex.relations)))
    rng.shuffle(positions)
    for pos in positions:
        old_rel = int(ex.relations[pos])
        rels = list(ex.relations)
        choices = [r for r in range(cfg.num_relations) if r != old_rel]
        rng.shuffle(choices)
        for r in choices:
            rels[pos] = int(r)
            cf = make_query_counterfactual(ex, rels, cfg, rng)
            if cf is not None:
                return cf
    return None


def make_same_multiset_order_counterfactual(ex: PathQAExample, cfg: ClosureWriterConfig, rng: random.Random) -> Optional[PathQAExample]:
    if len(ex.relations) < 2:
        return None
    rels = _different_permutation(ex.relations, rng)
    if tuple(rels) == tuple(ex.relations):
        return None
    return make_query_counterfactual(ex, rels, cfg, rng)


def make_fact_swap_counterfactual(ex: PathQAExample, cfg: ClosureWriterConfig, rng: random.Random) -> Optional[PathQAExample]:
    """Keep query relations fixed but change a causal gold edge/suffix so the
    endpoint changes.  This tests fact causality rather than query sensitivity.
    """
    L = len(ex.relations)
    if L < 1:
        return None
    positions = list(range(L))
    rng.shuffle(positions)
    for pos in positions:
        pivot = int(ex.path_nodes[pos])
        rel = int(ex.relations[pos])
        for _ in range(50):
            new_suffix = unused_entities(ex, cfg, rng, L - pos, avoid=[ex.target])
            nodes = tuple(list(ex.path_nodes[: pos + 1]) + new_suffix)
            edges: List[Edge] = []
            # Remove outgoing edges from pivot on the causal relation so the old
            # path is genuinely broken for the query.
            for s, r, t in ex.edges:
                if int(s) == pivot and int(r) == rel:
                    continue
                edges.append((int(s), int(r), int(t)))
            for j in range(pos, L):
                edges.append((int(nodes[j]), int(ex.relations[j]), int(nodes[j + 1])))
            cf = PathQAExample(
                source=int(ex.source),
                relations=tuple(int(r) for r in ex.relations),
                target=int(nodes[-1]),
                path_nodes=nodes,
                edges=tuple(dict.fromkeys(edges)),
                attempts=ex.attempts,
            )
            if cf.target != ex.target and exact_dict_answer(cf) == cf.target:
                return cf
    return None


def make_gold_edge_deleted(ex: PathQAExample, rng: random.Random) -> Optional[PathQAExample]:
    if not ex.relations:
        return None
    positions = list(range(len(ex.relations)))
    rng.shuffle(positions)
    for pos in positions:
        edge = (int(ex.path_nodes[pos]), int(ex.relations[pos]), int(ex.path_nodes[pos + 1]))
        edges = tuple(e for e in ex.edges if tuple(map(int, e)) != edge)
        broken = PathQAExample(
            source=int(ex.source), relations=tuple(int(r) for r in ex.relations),
            target=int(ex.target), path_nodes=tuple(int(x) for x in ex.path_nodes),
            edges=edges, attempts=ex.attempts,
        )
        if exact_dict_answer(broken) != ex.target:
            return broken
    return None


def make_distractor_deleted(ex: PathQAExample) -> PathQAExample:
    gold = {(int(ex.path_nodes[i]), int(ex.relations[i]), int(ex.path_nodes[i + 1])) for i in range(len(ex.relations))}
    edges = tuple(e for e in ex.edges if tuple(map(int, e)) in gold)
    return PathQAExample(
        source=int(ex.source), relations=tuple(int(r) for r in ex.relations),
        target=int(ex.target), path_nodes=tuple(int(x) for x in ex.path_nodes),
        edges=edges, attempts=ex.attempts,
    )


def has_adjacent_repeat(ex: PathQAExample) -> bool:
    return any(int(ex.relations[i]) == int(ex.relations[i + 1]) for i in range(len(ex.relations) - 1))


def make_repeated_relation_examples(cfg: ClosureWriterConfig, length: int, n: int, seed: int) -> List[PathQAExample]:
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=seed)
    out: List[PathQAExample] = []
    tries = 0
    while len(out) < n and tries < max(5000, 400 * n):
        tries += 1
        ex = gen.make_example(length)
        if length < 2 or has_adjacent_repeat(ex):
            out.append(ex)
    return out


def clone_with_scanned_slots(batch: Dict[str, torch.Tensor], tok: LearnedGroundingTextTokenizer, cfg: ClosureWriterConfig) -> Dict[str, torch.Tensor]:
    """Recompute role-slot positions from input token IDs only.

    This removes hidden dependence on PathQAExample metadata for slot positions.
    It is still a controlled text scanner, not an open-domain parser.
    """
    ids = batch["input_ids"].detach().cpu()
    B, _ = ids.shape
    fact_s_pos_all: List[List[int]] = []
    fact_r_pos_all: List[List[int]] = []
    fact_t_pos_all: List[List[int]] = []
    q_source_pos_all: List[int] = []
    q_rel_pos_all: List[List[int]] = []
    fact_tok = tok.tok("<fact>")
    query_tok = tok.tok("<query>")
    none_tok = tok.tok("<none>")
    answer_tok = tok.tok("answer")
    pad_tok = tok.pad_id
    for b in range(B):
        seq = [int(x) for x in ids[b].tolist()]
        try:
            stop = seq.index(pad_tok)
            seq = seq[:stop]
        except ValueError:
            pass
        fs: List[int] = []
        fr: List[int] = []
        ft: List[int] = []
        i = 0
        while i < len(seq):
            if seq[i] == fact_tok:
                if i + 1 < len(seq) and seq[i + 1] != none_tok and i + 3 < len(seq):
                    fs.append(i + 1); fr.append(i + 2); ft.append(i + 3)
                i += 4
            else:
                i += 1
        if query_tok not in seq:
            raise RuntimeError("<query> not found while scanning slots")
        qidx = seq.index(query_tok)
        q_source_pos_all.append(qidx + 2)  # <query> from eX follow ...
        qrels: List[int] = []
        j = qidx + 4
        while j < len(seq) and seq[j] != answer_tok:
            tid = seq[j]
            if tok.rel_label_from_token(tid) is not None:
                qrels.append(j)
            j += 1
        fact_s_pos_all.append(fs); fact_r_pos_all.append(fr); fact_t_pos_all.append(ft); q_rel_pos_all.append(qrels)

    max_facts = max((len(x) for x in fact_s_pos_all), default=0)
    dev = batch["input_ids"].device
    scanned = dict(batch)
    for name, src in [("fact_source_pos", fact_s_pos_all), ("fact_relation_pos", fact_r_pos_all), ("fact_target_pos", fact_t_pos_all)]:
        t = torch.zeros((B, max_facts), dtype=torch.long)
        for b, vals in enumerate(src):
            if vals:
                t[b, : len(vals)] = torch.tensor(vals, dtype=torch.long)
        scanned[name] = t.to(dev)
    qrp = torch.zeros((B, cfg.max_path_len), dtype=torch.long)
    for b, vals in enumerate(q_rel_pos_all):
        vals = vals[: cfg.max_path_len]
        if vals:
            qrp[b, : len(vals)] = torch.tensor(vals, dtype=torch.long)
    scanned["query_source_pos"] = torch.tensor(q_source_pos_all, dtype=torch.long, device=dev)
    scanned["query_relation_pos"] = qrp.to(dev)
    # Keep labels/masks from the original batch for metrics.  Positions are the intervention.
    return scanned


def confusion_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor], R: int) -> torch.Tensor:
    pred = logits.argmax(dim=-1).detach().cpu().reshape(-1)
    lab = labels.detach().cpu().reshape(-1)
    if mask is not None:
        m = mask.detach().cpu().reshape(-1).bool()
        pred = pred[m]; lab = lab[m]
    conf = torch.zeros(R, R, dtype=torch.long)
    for t, p in zip(lab.tolist(), pred.tolist()):
        if 0 <= int(t) < R and 0 <= int(p) < R:
            conf[int(t), int(p)] += 1
    return conf


def best_permutation_accuracy(conf: torch.Tensor) -> Tuple[float, List[int]]:
    R = int(conf.shape[0])
    total = int(conf.sum().item())
    if total == 0:
        return float("nan"), []
    if R <= 9:
        best = -1
        best_perm: Tuple[int, ...] = tuple(range(R))
        for perm in itertools.permutations(range(R)):
            score = sum(int(conf[i, perm[i]].item()) for i in range(R))
            if score > best:
                best = score; best_perm = perm
        return float(best / total), list(best_perm)
    # Greedy fallback for very large relation ontologies.
    remaining = set(range(R)); score = 0; perm = [-1] * R
    for i in range(R):
        if not remaining:
            break
        j = max(remaining, key=lambda x: int(conf[i, x].item()))
        perm[i] = int(j); score += int(conf[i, j].item()); remaining.remove(j)
    return float(score / total), perm


def entropy_and_maxfreq(targets: Sequence[int], num_entities: int) -> Tuple[float, float, float]:
    counts = [0] * num_entities
    for t in targets:
        counts[int(t)] += 1
    n = max(1, len(targets))
    probs = [c / n for c in counts if c > 0]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    norm_ent = ent / max(1e-12, math.log(num_entities))
    return norm_ent, max(counts) / n, min(counts) / n


@torch.no_grad()
def evaluate_anti_shortcut(
    cfg: ClosureWriterConfig,
    tok: LearnedGroundingTextTokenizer,
    field: HolographicClosureField,
    writer: LearnedExtractorClosureWriter,
    *,
    lengths: Sequence[int],
    device: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    writer.eval(); field.eval()
    rng = random.Random(cfg.seed + 9911)
    rows: List[Dict[str, float]] = []
    all_fact_rel_conf = torch.zeros(cfg.num_relations, cfg.num_relations, dtype=torch.long)
    all_query_rel_conf = torch.zeros(cfg.num_relations, cfg.num_relations, dtype=torch.long)

    for L in lengths:
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 10000 + int(L))
        examples = gen.make_examples(int(L), cfg.eval_n)
        cf_relation = [make_relation_swap_counterfactual(ex, cfg, rng) for ex in examples]
        cf_fact = [make_fact_swap_counterfactual(ex, cfg, rng) for ex in examples]
        cf_order = [make_same_multiset_order_counterfactual(ex, cfg, rng) for ex in examples]
        edge_deleted = [make_gold_edge_deleted(ex, rng) for ex in examples]
        distractor_deleted = [make_distractor_deleted(ex) for ex in examples]
        repeated = make_repeated_relation_examples(cfg, int(L), cfg.eval_n, seed=cfg.seed + 11000 + int(L)) if int(L) >= 2 else []

        counts: Dict[str, List[int]] = {}
        sums: Dict[str, List[float]] = {}
        def add(name: str, c: int, n: int) -> None:
            counts.setdefault(name, [0, 0]); counts[name][0] += int(c); counts[name][1] += int(n)
        def add_rate_numer(name: str, numer: int, denom: int) -> None:
            add(name, numer, denom)
        def add_sum(name: str, value: float, n: int = 1) -> None:
            sums.setdefault(name, [0.0, 0.0]); sums[name][0] += float(value); sums[name][1] += float(n)

        add("exact_dict", *symbolic_accuracy(examples, exact_dict_answer))
        add("raw_onehop", *symbolic_accuracy(examples, raw_onehop_answer))

        # Main examples and memory/closure interventions.
        for batch_examples in iter_minibatches(examples, cfg.eval_batch_size):
            batch = collate_learned_grounding_examples(batch_examples, tok, cfg, device=device, rng=rng)
            out = writer(batch, field)
            logits = out["logits"]
            add("learned_extractor_writer", *accuracy_from_logits(logits, batch["target"]))
            for k, (c, n) in extractor_slot_metrics(out, batch).items():
                add(f"extractor_{k}", c, n)

            ext = out["extractor"]
            if batch["fact_mask"].numel() and bool(batch["fact_mask"].any()):
                all_fact_rel_conf += confusion_from_logits(ext["fact_relation_logits"], batch["fact_relation_label"], batch["fact_mask"], cfg.num_relations)
            all_query_rel_conf += confusion_from_logits(ext["query_relation_logits"], batch["q_rels"], batch["query_relation_mask"], cfg.num_relations)

            scanned = clone_with_scanned_slots(batch, tok, cfg)
            scanned_out = writer(scanned, field)
            add("text_scanned_slot_positions", *accuracy_from_logits(scanned_out["logits"], scanned["target"]))

            mem = out["memory"]
            q = out["query_key"]
            scale = float(getattr(writer, "output_scale", 1.0))
            B = int(mem.shape[0])
            zero_logits = scale * field.read(torch.zeros_like(mem), q)
            add("zero_memory_old_target", *accuracy_from_logits(zero_logits, batch["target"]))
            if B > 1:
                swapped_mem_logits = scale * field.read(mem.roll(1, dims=0), q)
                swapped_key_logits = scale * field.read(mem, q.roll(1, dims=0))
                add("swapped_memory", *accuracy_from_logits(swapped_mem_logits, batch["target"]))
                add("swapped_query_key", *accuracy_from_logits(swapped_key_logits, batch["target"]))

            qrels = batch["q_rels"]
            lengths_t = batch["lengths"]
            source = batch["source"]
            targets = batch["target"]
            # Wrong-key exactness: read the same memory with keys that should not match.
            if int(L) > 1:
                rev_rels = qrels.clone()
                for b in range(B):
                    ell = int(lengths_t[b].item())
                    rev_rels[b, :ell] = torch.flip(rev_rels[b, :ell], dims=[0])
                rev_q = field.key(source, rev_rels, lengths_t)
                add("closure_reversed_key_old_target", *accuracy_from_logits(scale * field.read(mem, rev_q), targets))
                shuf_rels = qrels.clone()
                for b in range(B):
                    ell = int(lengths_t[b].item())
                    vals = [int(x) for x in shuf_rels[b, :ell].detach().cpu().tolist()]
                    vals = list(_different_permutation(vals, rng))
                    shuf_rels[b, :ell] = torch.tensor(vals, dtype=torch.long, device=device)
                shuf_q = field.key(source, shuf_rels, lengths_t)
                add("closure_shuffled_key_old_target", *accuracy_from_logits(scale * field.read(mem, shuf_q), targets))
                pref_len = torch.clamp(lengths_t - 1, min=1)
                pref_q = field.key(source, qrels, pref_len)
                add("closure_prefix_key_old_target", *accuracy_from_logits(scale * field.read(mem, pref_q), targets))
            rand_source = (source + torch.randint(1, cfg.num_entities, source.shape, device=device)) % cfg.num_entities
            rand_q = field.key(rand_source, qrels, lengths_t)
            add("closure_random_source_key_old_target", *accuracy_from_logits(scale * field.read(mem, rand_q), targets))

        # Paired counterfactuals.
        def eval_cf_group(name: str, cfs: Sequence[Optional[PathQAExample]], original: Sequence[PathQAExample]) -> None:
            pairs = [(o, c) for o, c in zip(original, cfs) if c is not None]
            add_sum(f"{name}_available", len(pairs), 1)
            if not pairs:
                return
            origs = [p[0] for p in pairs]
            mods = [p[1] for p in pairs]
            for bo, bm in zip(iter_minibatches(origs, cfg.eval_batch_size), iter_minibatches(mods, cfg.eval_batch_size)):
                b0 = collate_learned_grounding_examples(bo, tok, cfg, device=device, rng=rng)
                b1 = collate_learned_grounding_examples(bm, tok, cfg, device=device, rng=rng)
                out0 = writer(b0, field)["logits"]
                out1 = writer(b1, field)["logits"]
                add(f"{name}_original", *accuracy_from_logits(out0, b0["target"]))
                add(f"{name}_counterfactual", *accuracy_from_logits(out1, b1["target"]))
                pred0 = out0.argmax(dim=-1)
                pred1 = out1.argmax(dim=-1)
                both = pred0.eq(b0["target"]) & pred1.eq(b1["target"])
                changed = b0["target"].ne(b1["target"])
                add_rate_numer(f"{name}_both_correct", int((both & changed).sum().item()), int(changed.sum().item()))
                add_rate_numer(f"{name}_answer_changed", int(pred0.ne(pred1).sum().item()), int(pred0.numel()))
        eval_cf_group("counterfactual_relation_swap", cf_relation, examples)
        eval_cf_group("counterfactual_fact_swap", cf_fact, examples)
        eval_cf_group("same_multiset_order", cf_order, examples)

        # Edge deletion: old target should not remain confidently selected.
        valid_deleted = [x for x in edge_deleted if x is not None]
        if valid_deleted:
            for bd in iter_minibatches(valid_deleted, cfg.eval_batch_size):
                b = collate_learned_grounding_examples(bd, tok, cfg, device=device, rng=rng)
                logits = writer(b, field)["logits"]
                pred = logits.argmax(dim=-1)
                add_rate_numer("gold_edge_deletion_old_target_rate", int(pred.eq(b["target"]).sum().item()), int(pred.numel()))
                prob = F.softmax(logits, dim=-1).gather(1, b["target"].view(-1, 1)).mean().item()
                add_sum("gold_edge_deletion_old_target_prob", prob, 1)
        # Distractor deletion should preserve the answer.
        for bd in iter_minibatches(distractor_deleted, cfg.eval_batch_size):
            b = collate_learned_grounding_examples(bd, tok, cfg, device=device, rng=rng)
            add("distractor_deleted", *accuracy_from_logits(writer(b, field)["logits"], b["target"]))
        # Repeated relation hard subset.
        if repeated:
            for br in iter_minibatches(repeated, cfg.eval_batch_size):
                b = collate_learned_grounding_examples(br, tok, cfg, device=device, rng=rng)
                add("repeated_relation_subset", *accuracy_from_logits(writer(b, field)["logits"], b["target"]))

        row: Dict[str, float] = {"length": float(L), "n": float(len(examples))}
        ent, maxf, minf = entropy_and_maxfreq([ex.target for ex in examples], cfg.num_entities)
        row["target_entropy_normalized"] = ent
        row["target_max_frequency"] = maxf
        row["target_min_frequency"] = minf
        for name, (c, n) in sorted(counts.items()):
            row[f"{name}_acc"] = float(c / n) if n else float("nan")
            row[f"{name}_n"] = float(n)
        for name, (s, n) in sorted(sums.items()):
            row[name] = float(s / n) if n else float("nan")
        rows.append(row)
        printable = {k: row[k] for k in row if k.endswith("_acc") or k in {"length", "n", "counterfactual_relation_swap_available", "counterfactual_fact_swap_available"}}
        print(json.dumps({"key_selectivity_eval_by_length": printable}, sort_keys=True), flush=True)

    fact_perm_acc, fact_perm = best_permutation_accuracy(all_fact_rel_conf)
    query_perm_acc, query_perm = best_permutation_accuracy(all_query_rel_conf)
    diagnostics = {
        "fact_relation_confusion": all_fact_rel_conf.tolist(),
        "query_relation_confusion": all_query_rel_conf.tolist(),
        "fact_relation_best_permuted_acc": fact_perm_acc,
        "query_relation_best_permuted_acc": query_perm_acc,
        "fact_relation_best_permutation_true_to_pred": fact_perm,
        "query_relation_best_permutation_true_to_pred": query_perm,
    }
    return rows, diagnostics


def write_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "KEY_SELECTIVITY_RESULTS.json"
    csv_path = out_dir / "KEY_SELECTIVITY_RESULTS.csv"
    report_path = out_dir / "KEY_SELECTIVITY_REPORT.md"
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

    def fmt(v: object) -> str:
        try:
            x = float(v)
        except Exception:
            return str(v)
        if math.isnan(x):
            return "NA"
        return f"{x:.3f}"

    main_cols = [
        "length", "n", "learned_extractor_writer_acc", "text_scanned_slot_positions_acc",
        "counterfactual_relation_swap_counterfactual_acc", "counterfactual_fact_swap_counterfactual_acc",
        "counterfactual_relation_swap_both_correct_acc", "counterfactual_fact_swap_both_correct_acc",
        "same_multiset_order_counterfactual_acc", "distractor_deleted_acc", "repeated_relation_subset_acc",
    ]
    causal_cols = [
        "length", "zero_memory_old_target_acc", "swapped_memory_acc", "swapped_query_key_acc",
        "closure_reversed_key_old_target_acc", "closure_shuffled_key_old_target_acc",
        "closure_prefix_key_old_target_acc", "closure_random_source_key_old_target_acc",
        "gold_edge_deletion_old_target_rate_acc", "gold_edge_deletion_old_target_prob",
    ]
    lines: List[str] = []
    lines.append("# Anti-shortcut / causal-control results")
    lines.append("")
    lines.append("This suite trains the controlled learned-grounding closure pipeline and then applies counterfactual and memory interventions. High answer accuracy alone is not accepted as evidence of reasoning; the model must also respond correctly when facts/relations are changed, and must collapse when memory or exact keys are intervened on.")
    lines.append("")
    lines.append("## Main anti-shortcut metrics by length")
    lines.append("")
    lines.append("| " + " | ".join(main_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(main_cols)) + "|")
    for r in rows:
        vals = []
        for c in main_cols:
            vals.append(str(int(r.get(c, 0))) if c in {"length", "n"} else fmt(r.get(c, float("nan"))))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Memory and closure-key interventions by length")
    lines.append("")
    lines.append("| " + " | ".join(causal_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(causal_cols)) + "|")
    for r in rows:
        vals = []
        for c in causal_cols:
            vals.append(str(int(r.get(c, 0))) if c == "length" else fmt(r.get(c, float("nan"))))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    diag = meta.get("diagnostics", {}) if isinstance(meta.get("diagnostics", {}), dict) else {}
    lines.append("## Permutation-alignment diagnostics")
    lines.append("")
    lines.append(f"Fact relation best-permuted accuracy: `{fmt(diag.get('fact_relation_best_permuted_acc', float('nan')))}`")
    lines.append(f"Query relation best-permuted accuracy: `{fmt(diag.get('query_relation_best_permuted_acc', float('nan')))}`")
    lines.append("")
    lines.append("Interpretation: if canonical relation accuracy is low but best-permuted accuracy is high, answer-only training has likely found a latent label permutation. If both are low while answer accuracy is high, treat it as a shortcut warning, not interpretable grounding evidence.")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("Passing this suite supports a controlled causal-mechanism claim: learned grounding plus explicit transition/semiring closure plus one associative read. It does not establish open-domain reasoning or generic Transformer closure writing.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def parse_lengths(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run anti-shortcut causal controls.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=8901)
    p.add_argument("--train-steps", type=int, default=500)
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
    p.add_argument("--relation-aliases", type=int, default=3)
    p.add_argument("--base-distractors", type=int, default=6)
    p.add_argument("--distractors-per-hop", type=int, default=3)
    p.add_argument("--same-relation-branch-prob", type=float, default=0.25)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--extraction-weight", type=float, default=2.0)
    p.add_argument("--alias-train-prob", type=float, default=0.35)
    p.add_argument("--noise-train-prob", type=float, default=0.15)
    p.add_argument("--use-mlp-extractor", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
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
    writer, train_meta = train_learned_extractor_writer(
        cfg, tok, field, device=device, train_steps=args.train_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, extraction_weight=args.extraction_weight,
        alias_train_prob=args.alias_train_prob, noise_train_prob=args.noise_train_prob,
        use_mlp_extractor=args.use_mlp_extractor,
    )
    rows, diagnostics = evaluate_anti_shortcut(cfg, tok, field, writer, lengths=parse_lengths(args.lengths), device=device)
    meta = {
        "suite": "key_selectivity",
        "config": asdict(cfg),
        "args": vars(args),
        "train": train_meta,
        "diagnostics": diagnostics,
        "elapsed_total_sec": time.perf_counter() - t0,
    }
    paths = write_results(rows, meta, out_dir)
    print(json.dumps({"status": "done", "paths": paths, "elapsed_total_sec": meta["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
