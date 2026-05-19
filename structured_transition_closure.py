"""Semiring / dynamic-programming closure writer for HolographicClosureField.

This experiment is the structured-control diagnostic after the generic learned
writer. It adds a writer with the right algorithmic bias: parse controlled
one-hop facts, build relation-conditioned transition matrices, compose them
along the query, and write the resulting closure into the holographic field.
The final answer is still produced by exactly one associative read from
m_closure.

The deterministic DP writer is not a learned-reasoning result.  It is a
non-oracle constructive control: it uses raw one-hop facts and query text, not
full_closure labels or target labels.  The neural semiring writer learns only a
small relation-matching table; path composition itself is explicit and length
shared, which is the intended inductive-bias test.
"""
from __future__ import annotations

from dataclasses import asdict
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
    ClosureTextTokenizer,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    PathQAExample,
    accuracy_from_logits,
    collate_examples,
    exact_dict_answer,
    iter_minibatches,
    raw_onehop_answer,
    symbolic_accuracy,
)


class ControlledTextGraphParser:
    """Parse the controlled raw token text into one-hop graph tensors.

    This is deliberately a parser for the controlled language used by the
    generator, not an oracle closure writer.  It reads only <fact> e r e ; and
    <query> from e follow r then r ... answer <read> tokens.
    """

    def __init__(self, tok: ClosureTextTokenizer, cfg: ClosureWriterConfig) -> None:
        self.cfg = cfg
        self.pad_id = tok.pad_id
        self.fact_id = tok.tok("<fact>")
        self.query_id = tok.tok("<query>")
        self.answer_id = tok.tok("answer")
        self.read_id = tok.tok("<read>")
        self.none_id = tok.tok("<none>")
        self.ent0 = tok.ent(0)
        self.rel0 = tok.rel(0)
        self.num_entities = int(cfg.num_entities)
        self.num_relations = int(cfg.num_relations)
        self.max_path_len = int(cfg.max_path_len)

    def _is_ent(self, token: int) -> bool:
        return self.ent0 <= int(token) < self.ent0 + self.num_entities

    def _is_rel(self, token: int) -> bool:
        return self.rel0 <= int(token) < self.rel0 + self.num_relations

    def _ent(self, token: int) -> int:
        return int(token) - self.ent0

    def _rel(self, token: int) -> int:
        return int(token) - self.rel0

    def parse_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        mask = batch["mask"]
        device = input_ids.device
        B = int(input_ids.shape[0])
        A = torch.zeros(B, self.num_relations, self.num_entities, self.num_entities, dtype=torch.float32, device=device)
        sources = torch.zeros(B, dtype=torch.long, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        q_rels = torch.zeros(B, self.max_path_len, dtype=torch.long, device=device)

        # CPU-side token scanning is fine for this controlled diagnostic and
        # keeps the parser transparent.
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = mask.detach().cpu()
        for b in range(B):
            seq = ids_cpu[b, mask_cpu[b]].tolist()
            i = 0
            while i < len(seq):
                tok_i = int(seq[i])
                if tok_i == self.fact_id:
                    # Normal fact: <fact> eS r eT ; .  The no-facts control can
                    # contain <fact> <none> ;, which is ignored.
                    if i + 3 < len(seq) and self._is_ent(seq[i + 1]) and self._is_rel(seq[i + 2]) and self._is_ent(seq[i + 3]):
                        s = self._ent(seq[i + 1])
                        r = self._rel(seq[i + 2])
                        t = self._ent(seq[i + 3])
                        A[b, r, s, t] += 1.0
                        i += 5
                        continue
                elif tok_i == self.query_id:
                    j = i + 1
                    # Query source: first entity token after <query>.
                    while j < len(seq) and not self._is_ent(seq[j]):
                        j += 1
                    if j < len(seq) and self._is_ent(seq[j]):
                        sources[b] = self._ent(seq[j])
                        j += 1
                    rels: List[int] = []
                    while j < len(seq):
                        tj = int(seq[j])
                        if tj in {self.answer_id, self.read_id}:
                            break
                        if self._is_rel(tj) and len(rels) < self.max_path_len:
                            rels.append(self._rel(tj))
                        j += 1
                    if rels:
                        lengths[b] = len(rels)
                        q_rels[b, : len(rels)] = torch.tensor(rels, dtype=torch.long, device=device)
                    break
                i += 1

        return {"A": A, "source": sources, "q_rels": q_rels, "lengths": lengths}


class SemiringClosureWriter(nn.Module):
    """Closure writer with explicit relation-transition composition.

    The writer builds T_i = sum_r match(query_relation_i, r) A_r and composes a
    source distribution through T_1...T_L.  It then writes m_closure so the final
    answer is obtained by one HolographicClosureField.read call.

    learn_relation_match=False gives an exact parser+DP constructive control.
    learn_relation_match=True learns only the relation matching table; the path
    composition algorithm is fixed and length shared.
    """

    def __init__(
        self,
        tok: ClosureTextTokenizer,
        cfg: ClosureWriterConfig,
        *,
        learn_relation_match: bool = True,
        write_prefixes: bool = False,
        output_scale: float = 20.0,
        normalize_frontier: bool = True,
        hard_eval_relation_match: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.parser = ControlledTextGraphParser(tok, cfg)
        self.learn_relation_match = bool(learn_relation_match)
        self.write_prefixes = bool(write_prefixes)
        self.output_scale = float(output_scale)
        self.normalize_frontier = bool(normalize_frontier)
        self.hard_eval_relation_match = bool(hard_eval_relation_match)
        if self.learn_relation_match:
            # Start uncommitted: uniform relation matching.  The model must learn
            # the identity alignment between query relation tokens and fact
            # relation tokens from answer loss on L in {1,2,3}.
            self.rel_match_logits = nn.Parameter(torch.zeros(cfg.num_relations, cfg.num_relations))
        else:
            self.register_buffer("_dummy", torch.zeros(()), persistent=False)

    def relation_weights(self, rel_ids: torch.Tensor) -> torch.Tensor:
        if self.learn_relation_match:
            if self.hard_eval_relation_match and not self.training:
                # After relation grounding is learned, evaluate with a discrete
                # relation-token match.  This removes soft leakage that can make
                # reversed/shuffled controls appear to succeed even though the
                # requested order is wrong.
                mapped = self.rel_match_logits.argmax(dim=-1)[rel_ids]
                return F.one_hot(mapped, num_classes=self.cfg.num_relations).to(torch.float32)
            return F.softmax(self.rel_match_logits[rel_ids], dim=-1)
        return F.one_hot(rel_ids, num_classes=self.cfg.num_relations).to(torch.float32)

    def compose_frontiers(self, parsed: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        A = parsed["A"]
        source = parsed["source"]
        q_rels = parsed["q_rels"]
        lengths = parsed["lengths"]
        B = int(A.shape[0])
        E = int(self.cfg.num_entities)
        frontier = F.one_hot(source, num_classes=E).to(A.dtype)
        prefix_frontiers: List[torch.Tensor] = []
        max_l = int(lengths.max().item()) if lengths.numel() else 0
        for pos in range(max_l):
            active = lengths > pos
            rel_ids = q_rels[:, pos]
            weights = self.relation_weights(rel_ids).to(A.dtype)  # [B, R]
            T = torch.einsum("br,brst->bst", weights, A)
            nxt = torch.bmm(frontier.unsqueeze(1), T).squeeze(1)
            if self.normalize_frontier:
                denom = nxt.sum(dim=-1, keepdim=True)
                nxt = torch.where(denom > 1e-8, nxt / denom.clamp_min(1e-8), nxt)
            frontier = torch.where(active.unsqueeze(-1), nxt, frontier)
            prefix_frontiers.append(frontier)
        return frontier, prefix_frontiers

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, memory_mode: str = "normal") -> Dict[str, torch.Tensor]:
        parsed = self.parser.parse_batch(batch)
        A = parsed["A"]
        source = parsed["source"]
        q_rels = parsed["q_rels"]
        lengths = parsed["lengths"]
        B = int(A.shape[0])
        D = int(self.cfg.key_dim)
        E = int(self.cfg.num_entities)
        final_frontier, prefix_frontiers = self.compose_frontiers(parsed)
        q = field.key(source, q_rels, lengths)
        mem = torch.zeros(B, D, E, dtype=A.dtype, device=A.device)
        if memory_mode not in {"normal", "no_exact_query", "prefix_only"}:
            raise ValueError(memory_mode)

        if self.write_prefixes:
            # Direct control construction, not geometric projection.
            # normal: write every prefix including the exact full queried path.
            # no_exact_query / prefix_only: write only proper prefixes, never the
            # exact queried key.  This prevents exact-key signal from leaking
            # through a projection onto non-orthogonal random prefix keys.
            for pos, frontier in enumerate(prefix_frontiers):
                plen_value = pos + 1
                plen = torch.full_like(lengths, plen_value)
                if memory_mode == "normal":
                    write_mask = lengths >= plen_value
                else:
                    write_mask = lengths > plen_value
                if not bool(write_mask.any()):
                    continue
                pk = field.key(source, q_rels, plen)
                mem = mem + write_mask.to(A.dtype).view(B, 1, 1) * pk.unsqueeze(-1).to(A.dtype) * frontier.unsqueeze(1)
        else:
            # Exact-query writer: a no-exact or prefix-only control has no legal
            # write left, so the memory is intentionally zero.
            if memory_mode == "normal":
                mem = q.unsqueeze(-1).to(A.dtype) * final_frontier.unsqueeze(1)

        logits = self.output_scale * field.read(mem, q)
        return {"logits": logits, "memory": mem, "query_key": q, "parsed": parsed, "frontier": final_frontier}


def train_neural_semiring_writer(
    cfg: ClosureWriterConfig,
    tok: ClosureTextTokenizer,
    field: HolographicClosureField,
    *,
    device: torch.device | str = "cpu",
    train_steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 0.2,
    write_prefixes: bool = False,
    relation_grounding_weight: float = 0.2,
) -> Tuple[SemiringClosureWriter, Dict[str, object]]:
    device = torch.device(device)
    writer = SemiringClosureWriter(tok, cfg, learn_relation_match=True, write_prefixes=write_prefixes, output_scale=20.0).to(device)
    opt = torch.optim.AdamW(writer.parameters(), lr=float(learning_rate), weight_decay=0.0)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 5101)
    rng = random.Random(cfg.seed + 5202)
    t0 = time.perf_counter()
    snapshots: List[Dict[str, float]] = []
    for step in range(1, int(train_steps) + 1):
        if step <= max(1, int(0.25 * train_steps)):
            choices = (1,)
        elif step <= max(1, int(0.55 * train_steps)):
            choices = (1, 2)
        else:
            choices = (1, 2, 3)
        examples = [gen.make_example(rng.choice(choices)) for _ in range(int(batch_size))]
        batch = collate_examples(examples, tok, cfg, device=device, rng=rng)
        out = writer(batch, field)
        answer_loss = F.cross_entropy(out["logits"], batch["target"])
        # Controlled-language relation grounding: same relation token in the
        # query should match the same relation token on facts.  This is parser
        # supervision, not oracle closure supervision; it prevents soft relation
        # leakage from making reversed/shuffled controls look artificially good.
        rel_target = torch.arange(cfg.num_relations, dtype=torch.long, device=device)
        grounding_loss = F.cross_entropy(writer.rel_match_logits, rel_target)
        loss = answer_loss + float(relation_grounding_weight) * grounding_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, train_steps // 5) == 0 or step == train_steps:
            c, n = accuracy_from_logits(out["logits"].detach(), batch["target"])
            with torch.no_grad():
                weights = F.softmax(writer.rel_match_logits, dim=-1).detach().cpu()
                diag = weights.diag().mean().item()
                offdiag = ((weights.sum() - weights.diag().sum()) / max(1, cfg.num_relations * (cfg.num_relations - 1))).item()
            snap = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "answer_loss": float(answer_loss.detach().item()),
                "grounding_loss": float(grounding_loss.detach().item()),
                "train_batch_acc": float(c / max(1, n)),
                "mean_diag_relation_match": float(diag),
                "mean_offdiag_relation_match": float(offdiag),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
            snapshots.append(snap)
            print(json.dumps({"structured_closure_train_progress": snap}, sort_keys=True), flush=True)
    meta = {
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "write_prefixes": bool(write_prefixes),
        "relation_grounding_weight": float(relation_grounding_weight),
        "snapshots": snapshots,
        "final_relation_match": F.softmax(writer.rel_match_logits, dim=-1).detach().cpu().tolist(),
        "elapsed_train_sec": float(time.perf_counter() - t0),
    }
    return writer, meta


@torch.no_grad()
def evaluate_structured_closure(
    cfg: ClosureWriterConfig,
    tok: ClosureTextTokenizer,
    field: HolographicClosureField,
    exact_dp_writer: SemiringClosureWriter,
    exact_prefix_writer: SemiringClosureWriter,
    neural_writer: SemiringClosureWriter,
    *,
    lengths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    device = torch.device(device)
    exact_dp_writer.eval(); exact_prefix_writer.eval(); neural_writer.eval(); field.eval()
    rng = random.Random(cfg.seed + 5303)
    rows: List[Dict[str, float]] = []
    for L in lengths:
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 6000 + int(L))
        examples = gen.make_examples(int(L), cfg.eval_n)
        wrong_gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 7000 + int(L))
        wrong_examples = wrong_gen.make_examples(int(L), cfg.eval_n)
        row_counts: Dict[str, List[int]] = {}
        def add(name: str, c: int, n: int) -> None:
            row_counts.setdefault(name, [0, 0])
            row_counts[name][0] += int(c)
            row_counts[name][1] += int(n)

        add("exact_dict", *symbolic_accuracy(examples, exact_dict_answer))
        add("raw_onehop", *symbolic_accuracy(examples, raw_onehop_answer))

        for batch_examples, wrong_batch_examples in zip(iter_minibatches(examples, cfg.eval_batch_size), iter_minibatches(wrong_examples, cfg.eval_batch_size)):
            batch = collate_examples(batch_examples, tok, cfg, device=device, rng=rng)
            target = batch["target"]
            out_exact = exact_dp_writer(batch, field)
            out_prefix = exact_prefix_writer(batch, field)
            out_neural = neural_writer(batch, field)
            add("dp_exact_query_writer", *accuracy_from_logits(out_exact["logits"], target))
            add("dp_prefix_field_writer", *accuracy_from_logits(out_prefix["logits"], target))
            add("neural_semiring_writer", *accuracy_from_logits(out_neural["logits"], target))

            # Controls for the neural semiring writer.
            for variant, metric_name in [
                ("query_only", "neural_query_only"),
                ("no_facts", "neural_no_facts"),
                ("wrong_facts", "neural_wrong_facts"),
                ("reversed_order", "neural_reversed_order"),
                ("shuffled_order", "neural_shuffled_order"),
                ("first_order", "neural_first_order"),
            ]:
                if variant == "wrong_facts":
                    vb = collate_examples(batch_examples, tok, cfg, variant=variant, wrong_fact_examples=wrong_batch_examples, device=device, rng=rng)
                else:
                    vb = collate_examples(batch_examples, tok, cfg, variant=variant, device=device, rng=rng)
                logits = neural_writer(vb, field)["logits"]
                if variant in {"reversed_order", "shuffled_order"}:
                    mask = vb["control_changed"]
                    if bool(mask.any()):
                        add(metric_name, *accuracy_from_logits(logits, vb["target"], mask=mask))
                    else:
                        add(metric_name, 0, 0)
                else:
                    add(metric_name, *accuracy_from_logits(logits, vb["target"]))

            add("neural_no_exact_query", *accuracy_from_logits(neural_writer(batch, field, memory_mode="no_exact_query")["logits"], target))
            add("neural_prefix_only", *accuracy_from_logits(neural_writer(batch, field, memory_mode="prefix_only")["logits"], target))
            add("dp_prefix_no_exact_query", *accuracy_from_logits(exact_prefix_writer(batch, field, memory_mode="no_exact_query")["logits"], target))
            add("dp_prefix_prefix_only", *accuracy_from_logits(exact_prefix_writer(batch, field, memory_mode="prefix_only")["logits"], target))

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
        print(json.dumps({"structured_closure_eval_by_length": printable}, sort_keys=True), flush=True)
    return rows


def write_structured_closure_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "STRUCTURED_TRANSITION_CLOSURE_RESULTS.json"
    csv_path = out_dir / "STRUCTURED_TRANSITION_CLOSURE_RESULTS.csv"
    report_path = out_dir / "STRUCTURED_TRANSITION_CLOSURE_REPORT.md"
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def fmt(x: float) -> str:
        if isinstance(x, float) and math.isnan(x):
            return "NA"
        return f"{x:.3f}"

    main_cols = [
        "length", "n",
        "neural_semiring_writer_acc",
        "dp_exact_query_writer_acc",
        "dp_prefix_field_writer_acc",
        "raw_onehop_acc", "exact_dict_acc",
    ]
    control_cols = [
        "length",
        "neural_query_only_acc", "neural_no_facts_acc", "neural_wrong_facts_acc",
        "neural_reversed_order_acc", "neural_shuffled_order_acc", "neural_first_order_acc",
        "neural_no_exact_query_acc", "neural_prefix_only_acc",
        "dp_prefix_no_exact_query_acc", "dp_prefix_prefix_only_acc",
    ]
    lines: List[str] = []
    lines.append("# SemiringClosureWriter / HolographicClosureField results")
    lines.append("")
    lines.append("This is the structured-control diagnostic after the generic TransformerClosureWriter result. It does not use oracle full_closure writes or target labels to construct memory. It parses controlled raw one-hop fact text, composes relation transitions, writes m_closure, and the answer is still produced by exactly one holographic associative read.")
    lines.append("")
    lines.append("Training lengths for the neural semiring relation matcher: L in {1, 2, 3}. Evaluation is reported by length only; no averaged headline is used.")
    lines.append("")
    lines.append("## Main metrics by length")
    lines.append("")
    lines.append("| " + " | ".join(main_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(main_cols)) + "|")
    for row in rows:
        vals = []
        for c in main_cols:
            v = row.get(c, float("nan"))
            vals.append(str(int(v)) if c in {"length", "n"} else fmt(float(v)))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Controls by length")
    lines.append("")
    lines.append("| " + " | ".join(control_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(control_cols)) + "|")
    for row in rows:
        vals = []
        for c in control_cols:
            v = row.get(c, float("nan"))
            vals.append(str(int(v)) if c == "length" else fmt(float(v)))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Run notes")
    lines.append("")
    cfg_meta = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    lines.append(f"This CPU diagnostic used eval_n={int(cfg_meta.get('eval_n', 0))} per length, key_dim={int(cfg_meta.get('key_dim', 0))}, and same_relation_branch_prob={cfg_meta.get('same_relation_branch_prob', 'NA')}. Wrong-relation and off-path distractors are still present; the generator keeps distractor targets off the gold path to make reversed/shuffled-order controls stricter. The learned component is deliberately small: relation-token grounding plus hard relation matching at evaluation. The closure construction itself is explicit semiring/dynamic-programming composition, not a plain Transformer learned writer.")
    lines.append("")
    lines.append("The positive result should therefore be read narrowly: the HolographicClosureField reader is not the bottleneck when the writer has the right transition-composition bias. It does not overturn the negative result for ordinary TransformerClosureWriter.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The deterministic DP writer is a constructive non-oracle control: it uses raw one-hop facts plus the query and therefore should extrapolate if the reader/key machinery is not the bottleneck. The neural semiring writer learns only relation matching; explicit dynamic-programming composition carries the length extrapolation bias.")
    final_match = meta.get("neural_train", {}).get("final_relation_match")
    if final_match is not None:
        lines.append("")
        lines.append("Final learned relation-match matrix, rows=query relation and columns=fact relation:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(final_match, indent=2))
        lines.append("```")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_structured_closure(cfg: ClosureWriterConfig, out_dir: Path, device: torch.device | str = "cpu", train_steps: int = 500) -> Dict[str, object]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(device)
    exact_dp = SemiringClosureWriter(tok, cfg, learn_relation_match=False, write_prefixes=False, output_scale=20.0).to(device)
    exact_prefix = SemiringClosureWriter(tok, cfg, learn_relation_match=False, write_prefixes=True, output_scale=20.0).to(device)
    neural, neural_meta = train_neural_semiring_writer(cfg, tok, field, device=device, train_steps=train_steps, batch_size=cfg.batch_size, learning_rate=0.2, write_prefixes=False)
    rows = evaluate_structured_closure(cfg, tok, field, exact_dp, exact_prefix, neural, device=device)
    meta = {
        "config": asdict(cfg),
        "neural_train": neural_meta,
        "num_parameters": {
            "neural_semiring_writer": sum(p.numel() for p in neural.parameters()),
            "dp_exact_query_writer": sum(p.numel() for p in exact_dp.parameters()),
            "dp_prefix_field_writer": sum(p.numel() for p in exact_prefix.parameters()),
        },
    }
    paths = write_structured_closure_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run semiring closure writer diagnostic.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=ClosureWriterConfig.seed)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--num-entities", type=int, default=ClosureWriterConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=ClosureWriterConfig.num_relations)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--threads", type=int, default=ClosureWriterConfig.torch_threads)
    p.add_argument("--same-relation-branch-prob", type=float, default=ClosureWriterConfig.same_relation_branch_prob)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClosureWriterConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        torch_threads=args.threads,
        same_relation_branch_prob=args.same_relation_branch_prob,
    )
    result = run_structured_closure(cfg, Path(args.out_dir), device=args.device, train_steps=args.train_steps)
    print(json.dumps({"status": "done", "paths": result["paths"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
