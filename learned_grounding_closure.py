"""Learned transition extractor + explicit closure writer.

This diagnostic follows the split between generic learned field writing and
structured transition composition. It makes the text-to-transition step learned
while keeping the path composition explicit and keeping the final answer to
exactly one holographic associative read.

No oracle full_closure writes or target labels are used to construct memory.
Training may use slot-level extraction supervision for source/relation/target
roles; this tests the hybrid claim: learned grounding + structured closure
composition + single-read retrieval.
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
from structured_transition_closure import SemiringClosureWriter


class LearnedGroundingTextTokenizer:
    """Tokenizer with relation aliases and harmless noise tokens.

    Canonical entity tokens are e0..eN.  Relation tokens include canonical r0..rR
    plus aliases r0_a1, r0_a2, ... .  Aliases map to the same relation label and
    let us test whether the extractor is learning grounding rather than only the
    single canonical spelling.
    """

    def __init__(self, num_entities: int, num_relations: int, relation_aliases: int = 3) -> None:
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.relation_aliases = int(max(1, relation_aliases))
        vocab = [
            "<pad>", "<bos>", "<fact>", "<query>", "<read>", "<none>", ";",
            "from", "to", "follow", "then", "answer", "noise", "irrelevant",
            "because", "near", "not", "story", "marker",
        ]
        vocab += [f"e{i}" for i in range(self.num_entities)]
        self._entity_start = len(vocab) - self.num_entities
        self._relation_token_ids: Dict[Tuple[int, int], int] = {}
        self._relation_id_by_token: Dict[int, int] = {}
        for r in range(self.num_relations):
            for a in range(self.relation_aliases):
                name = f"r{r}" if a == 0 else f"r{r}_a{a}"
                self._relation_token_ids[(r, a)] = len(vocab)
                self._relation_id_by_token[len(vocab)] = r
                vocab.append(name)
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = list(vocab)
        self.pad_id = self.token_to_id["<pad>"]
        self.noise_ids = [self.token_to_id[t] for t in ["noise", "irrelevant", "because", "near", "not", "story", "marker"]]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tok(self, t: str) -> int:
        return self.token_to_id[t]

    def ent(self, e: int) -> int:
        return self.token_to_id[f"e{int(e)}"]

    def rel(self, r: int, alias: int = 0) -> int:
        alias = int(alias) % self.relation_aliases
        return self._relation_token_ids[(int(r), alias)]

    def rel_label_from_token(self, token_id: int) -> Optional[int]:
        return self._relation_id_by_token.get(int(token_id))

    def is_ent_token(self, token_id: int) -> bool:
        tid = int(token_id)
        return self._entity_start <= tid < self._entity_start + self.num_entities

    def ent_label_from_token(self, token_id: int) -> Optional[int]:
        if not self.is_ent_token(token_id):
            return None
        return int(token_id) - self._entity_start


def _apply_entity_permutation_to_example(ex: PathQAExample, perm: Sequence[int]) -> PathQAExample:
    p = [int(x) for x in perm]
    edges = tuple((p[s], int(r), p[t]) for s, r, t in ex.edges)
    nodes = tuple(p[x] for x in ex.path_nodes)
    return PathQAExample(
        source=p[ex.source],
        relations=tuple(int(r) for r in ex.relations),
        target=p[ex.target],
        path_nodes=nodes,
        edges=edges,
        attempts=ex.attempts,
    )


def collate_learned_grounding_examples(
    examples: Sequence[PathQAExample],
    tok: LearnedGroundingTextTokenizer,
    cfg: ClosureWriterConfig,
    *,
    variant: str = "normal",
    wrong_fact_examples: Optional[Sequence[PathQAExample]] = None,
    rng: Optional[random.Random] = None,
    device: torch.device | str = "cpu",
    relation_aliases: bool = False,
    extra_text_noise: bool = False,
    fact_order_shuffle: bool = False,
    entity_renaming: bool = False,
) -> Dict[str, torch.Tensor | List[PathQAExample]]:
    """Collate examples and expose role slot labels for learned extraction.

    The slot positions are obtained from the controlled text markers.  The model
    does not receive these labels during forward evaluation; they are used only
    for supervised grounding losses during training and for diagnostics.
    """

    rng = rng or random.Random(0)
    dev = torch.device(device)
    seqs: List[List[int]] = []
    sources: List[int] = []
    lengths: List[int] = []
    targets: List[int] = []
    q_rels_all: List[List[int]] = []
    control_changed: List[bool] = []
    fact_s_pos_all: List[List[int]] = []
    fact_r_pos_all: List[List[int]] = []
    fact_t_pos_all: List[List[int]] = []
    fact_s_lbl_all: List[List[int]] = []
    fact_r_lbl_all: List[List[int]] = []
    fact_t_lbl_all: List[List[int]] = []
    q_source_pos_all: List[int] = []
    q_rel_pos_all: List[List[int]] = []

    for i, original_ex in enumerate(examples):
        ex = original_ex
        if entity_renaming:
            perm = list(range(cfg.num_entities))
            rng.shuffle(perm)
            ex = _apply_entity_permutation_to_example(original_ex, perm)

        facts = list(ex.edges)
        rels = tuple(ex.relations)
        source = int(ex.source)
        target = int(ex.target)
        changed = entity_renaming

        if variant == "wrong_facts":
            if wrong_fact_examples is None:
                wrong_ex = examples[(i + 1) % len(examples)]
            else:
                wrong_ex = wrong_fact_examples[i]
            if entity_renaming:
                perm = list(range(cfg.num_entities))
                rng.shuffle(perm)
                wrong_ex = _apply_entity_permutation_to_example(wrong_ex, perm)
            facts = list(wrong_ex.edges)
            changed = True
        elif variant in {"query_only", "no_facts"}:
            facts = []
            changed = True
        elif variant == "reversed_order":
            rels = tuple(reversed(rels))
            changed = tuple(rels) != tuple(ex.relations)
        elif variant == "shuffled_order":
            rels = _different_permutation(rels, rng)
            changed = tuple(rels) != tuple(ex.relations)
        elif variant == "first_order":
            rels = tuple(rels[:1])
            changed = True
        elif variant == "normal":
            pass
        else:
            raise ValueError(f"unknown variant {variant}")

        if fact_order_shuffle and facts:
            rng.shuffle(facts)
            changed = True

        tokens: List[int] = [tok.tok("<bos>")]
        fs_pos: List[int] = []
        fr_pos: List[int] = []
        ft_pos: List[int] = []
        fs_lbl: List[int] = []
        fr_lbl: List[int] = []
        ft_lbl: List[int] = []

        def maybe_noise() -> None:
            if extra_text_noise:
                for _ in range(rng.randint(1, 3)):
                    tokens.append(rng.choice(tok.noise_ids))

        maybe_noise()
        if facts:
            for s, r, t in facts:
                maybe_noise()
                tokens.append(tok.tok("<fact>"))
                fs_pos.append(len(tokens)); tokens.append(tok.ent(s))
                alias = rng.randrange(tok.relation_aliases) if relation_aliases else 0
                fr_pos.append(len(tokens)); tokens.append(tok.rel(r, alias=alias))
                ft_pos.append(len(tokens)); tokens.append(tok.ent(t))
                tokens.append(tok.tok(";"))
                fs_lbl.append(int(s)); fr_lbl.append(int(r)); ft_lbl.append(int(t))
        elif variant == "no_facts":
            maybe_noise()
            tokens.extend([tok.tok("<fact>"), tok.tok("<none>"), tok.tok(";")])
        maybe_noise()

        tokens.extend([tok.tok("<query>"), tok.tok("from")])
        q_source_pos = len(tokens); tokens.append(tok.ent(source))
        tokens.append(tok.tok("follow"))
        q_rel_pos: List[int] = []
        for j, r in enumerate(rels):
            if j > 0:
                tokens.append(tok.tok("then"))
            alias = rng.randrange(tok.relation_aliases) if relation_aliases else 0
            q_rel_pos.append(len(tokens)); tokens.append(tok.rel(r, alias=alias))
        tokens.extend([tok.tok("answer"), tok.tok("<read>")])

        if len(tokens) > cfg.max_seq_len:
            raise RuntimeError(f"sequence too long: {len(tokens)} > {cfg.max_seq_len}")

        seqs.append(tokens)
        sources.append(source)
        lengths.append(len(rels))
        targets.append(target)
        padded = list(rels) + [0] * (cfg.max_path_len - len(rels))
        q_rels_all.append(padded[: cfg.max_path_len])
        control_changed.append(bool(changed))
        fact_s_pos_all.append(fs_pos); fact_r_pos_all.append(fr_pos); fact_t_pos_all.append(ft_pos)
        fact_s_lbl_all.append(fs_lbl); fact_r_lbl_all.append(fr_lbl); fact_t_lbl_all.append(ft_lbl)
        q_source_pos_all.append(q_source_pos)
        q_rel_pos_all.append(q_rel_pos + [0] * (cfg.max_path_len - len(q_rel_pos)))

    B = len(seqs)
    max_len = max(len(s) for s in seqs)
    max_facts = max((len(x) for x in fact_s_pos_all), default=0)
    ids = torch.full((B, max_len), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    read_pos = torch.zeros(B, dtype=torch.long)
    fact_s_pos = torch.zeros((B, max_facts), dtype=torch.long)
    fact_r_pos = torch.zeros((B, max_facts), dtype=torch.long)
    fact_t_pos = torch.zeros((B, max_facts), dtype=torch.long)
    fact_s_lbl = torch.zeros((B, max_facts), dtype=torch.long)
    fact_r_lbl = torch.zeros((B, max_facts), dtype=torch.long)
    fact_t_lbl = torch.zeros((B, max_facts), dtype=torch.long)
    fact_mask = torch.zeros((B, max_facts), dtype=torch.bool)

    for b, seq in enumerate(seqs):
        ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[b, : len(seq)] = True
        read_pos[b] = len(seq) - 1
        f = len(fact_s_pos_all[b])
        if f:
            fact_s_pos[b, :f] = torch.tensor(fact_s_pos_all[b], dtype=torch.long)
            fact_r_pos[b, :f] = torch.tensor(fact_r_pos_all[b], dtype=torch.long)
            fact_t_pos[b, :f] = torch.tensor(fact_t_pos_all[b], dtype=torch.long)
            fact_s_lbl[b, :f] = torch.tensor(fact_s_lbl_all[b], dtype=torch.long)
            fact_r_lbl[b, :f] = torch.tensor(fact_r_lbl_all[b], dtype=torch.long)
            fact_t_lbl[b, :f] = torch.tensor(fact_t_lbl_all[b], dtype=torch.long)
            fact_mask[b, :f] = True

    q_rel_mask = torch.zeros((B, cfg.max_path_len), dtype=torch.bool)
    for b, L in enumerate(lengths):
        q_rel_mask[b, : int(L)] = True

    return {
        "input_ids": ids.to(dev),
        "mask": mask.to(dev),
        "read_pos": read_pos.to(dev),
        "source": torch.tensor(sources, dtype=torch.long, device=dev),
        "q_rels": torch.tensor(q_rels_all, dtype=torch.long, device=dev),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=dev),
        "target": torch.tensor(targets, dtype=torch.long, device=dev),
        "control_changed": torch.tensor(control_changed, dtype=torch.bool, device=dev),
        "fact_source_pos": fact_s_pos.to(dev),
        "fact_relation_pos": fact_r_pos.to(dev),
        "fact_target_pos": fact_t_pos.to(dev),
        "fact_source_label": fact_s_lbl.to(dev),
        "fact_relation_label": fact_r_lbl.to(dev),
        "fact_target_label": fact_t_lbl.to(dev),
        "fact_mask": fact_mask.to(dev),
        "query_source_pos": torch.tensor(q_source_pos_all, dtype=torch.long, device=dev),
        "query_relation_pos": torch.tensor(q_rel_pos_all, dtype=torch.long, device=dev),
        "query_relation_mask": q_rel_mask.to(dev),
        "examples": list(examples),
    }


class SlotGroundingExtractor(nn.Module):
    """Learned slot grounding for controlled relational text.

    The grammar markers provide slot positions, while the mappings from raw token
    embeddings to entity/relation labels are learned.  This is intentionally not
    a generic open-domain parser; it is a controlled grounding module used to
    test whether learned extraction plus explicit closure composition
    extrapolates in path length.
    """

    def __init__(self, vocab_size: int, cfg: ClosureWriterConfig, *, alias_count: int = 3, use_mlp: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.alias_count = int(alias_count)
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.use_mlp = bool(use_mlp)
        if self.use_mlp:
            self.encoder = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        else:
            self.encoder = nn.Identity()
        self.entity_head = nn.Linear(cfg.d_model, cfg.num_entities)
        self.relation_head = nn.Linear(cfg.d_model, cfg.num_relations)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.emb(input_ids))

    @staticmethod
    def _gather(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.dim() == 1:
            return x.gather(1, pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1])).squeeze(1)
        return x.gather(1, pos.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @staticmethod
    def _dist(logits: torch.Tensor, hard: bool) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        if hard:
            h = F.one_hot(p.argmax(dim=-1), num_classes=logits.shape[-1]).to(p.dtype)
            return h
        return p

    def forward(self, batch: Dict[str, torch.Tensor], *, hard: bool = False) -> Dict[str, torch.Tensor]:
        h = self.encode(batch["input_ids"])
        fs_h = self._gather(h, batch["fact_source_pos"]) if batch["fact_source_pos"].shape[1] else h[:, :0, :]
        fr_h = self._gather(h, batch["fact_relation_pos"]) if batch["fact_relation_pos"].shape[1] else h[:, :0, :]
        ft_h = self._gather(h, batch["fact_target_pos"]) if batch["fact_target_pos"].shape[1] else h[:, :0, :]
        qs_h = self._gather(h, batch["query_source_pos"])
        qr_h = self._gather(h, batch["query_relation_pos"])
        fs_logits = self.entity_head(fs_h)
        fr_logits = self.relation_head(fr_h)
        ft_logits = self.entity_head(ft_h)
        qs_logits = self.entity_head(qs_h)
        qr_logits = self.relation_head(qr_h)
        return {
            "fact_source_logits": fs_logits,
            "fact_relation_logits": fr_logits,
            "fact_target_logits": ft_logits,
            "query_source_logits": qs_logits,
            "query_relation_logits": qr_logits,
            "fact_source_dist": self._dist(fs_logits, hard),
            "fact_relation_dist": self._dist(fr_logits, hard),
            "fact_target_dist": self._dist(ft_logits, hard),
            "query_source_dist": self._dist(qs_logits, hard),
            "query_relation_dist": self._dist(qr_logits, hard),
        }

    def set_token_identity_grounding(self, tok: LearnedGroundingTextTokenizer, scale: float = 12.0) -> None:
        """Initialize a near-perfect token identity mapper for deterministic tests.

        Requires d_model >= vocab_size. It is not used in the main public run.
        """
        if self.cfg.d_model < self.vocab_size:
            raise ValueError("identity grounding requires d_model >= vocab_size")
        with torch.no_grad():
            self.emb.weight.zero_()
            for tid in range(self.vocab_size):
                self.emb.weight[tid, tid] = 1.0
            if isinstance(self.encoder, nn.Identity):
                pass
            else:
                raise ValueError("identity grounding expects use_mlp=False")
            self.entity_head.weight.zero_(); self.entity_head.bias.zero_()
            self.relation_head.weight.zero_(); self.relation_head.bias.zero_()
            for e in range(self.cfg.num_entities):
                self.entity_head.weight[e, tok.ent(e)] = float(scale)
            for tid, r in tok._relation_id_by_token.items():
                self.relation_head.weight[int(r), int(tid)] = float(scale)


class LearnedExtractorClosureWriter(nn.Module):
    """Learned extractor + explicit semiring closure + one holographic read."""

    def __init__(
        self,
        tok: LearnedGroundingTextTokenizer,
        cfg: ClosureWriterConfig,
        *,
        write_prefixes: bool = False,
        output_scale: float = 30.0,
        normalize_frontier: bool = True,
        hard_eval_extraction: bool = True,
        use_mlp_extractor: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.extractor = SlotGroundingExtractor(tok.vocab_size, cfg, alias_count=tok.relation_aliases, use_mlp=use_mlp_extractor)
        self.write_prefixes = bool(write_prefixes)
        self.output_scale = float(output_scale)
        self.normalize_frontier = bool(normalize_frontier)
        self.hard_eval_extraction = bool(hard_eval_extraction)

    def _should_harden(self) -> bool:
        return bool(self.hard_eval_extraction and not self.training)

    def build_adjacency(self, ext: Dict[str, torch.Tensor], fact_mask: torch.Tensor) -> torch.Tensor:
        p_s = ext["fact_source_dist"]
        p_r = ext["fact_relation_dist"]
        p_t = ext["fact_target_dist"]
        if p_s.shape[1] == 0:
            B = int(fact_mask.shape[0])
            return torch.zeros(B, self.cfg.num_relations, self.cfg.num_entities, self.cfg.num_entities, device=fact_mask.device)
        m = fact_mask.to(p_s.dtype)
        return torch.einsum("bfs,bfr,bft,bf->brst", p_s, p_r, p_t, m)

    def compose_frontiers(self, A: torch.Tensor, q_source_dist: torch.Tensor, q_rel_dist: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        frontier = q_source_dist.to(A.dtype)
        prefix_frontiers: List[torch.Tensor] = []
        max_l = int(lengths.max().item()) if lengths.numel() else 0
        for pos in range(max_l):
            active = lengths > pos
            weights = q_rel_dist[:, pos, :].to(A.dtype)
            T = torch.einsum("br,brst->bst", weights, A)
            nxt = torch.bmm(frontier.unsqueeze(1), T).squeeze(1)
            if self.normalize_frontier:
                denom = nxt.sum(dim=-1, keepdim=True)
                nxt = torch.where(denom > 1e-8, nxt / denom.clamp_min(1e-8), nxt)
            frontier = torch.where(active.unsqueeze(-1), nxt, frontier)
            prefix_frontiers.append(frontier)
        return frontier, prefix_frontiers

    def soft_key(self, field: HolographicClosureField, q_source_dist: torch.Tensor, q_rel_dist: torch.Tensor, lengths: torch.Tensor, prefix_len: Optional[int] = None) -> torch.Tensor:
        B = int(q_source_dist.shape[0])
        if self._should_harden():
            src = q_source_dist.argmax(dim=-1)
            qrels = q_rel_dist.argmax(dim=-1)
            if prefix_len is None:
                lens = lengths
            else:
                lens = torch.full_like(lengths, int(prefix_len))
            return field.key(src, qrels, lens)
        k = q_source_dist.to(field.entity_code.dtype) @ field.entity_code
        lens = lengths if prefix_len is None else torch.full_like(lengths, int(prefix_len))
        k = k * field.length_code[lens.long().clamp(0, field.max_path_len)]
        max_l = int(lens.max().item()) if lens.numel() else 0
        for pos in range(max_l):
            active = lens > pos
            if bool(active.any()):
                rel_code = q_rel_dist[active, pos, :].to(field.relpos_code.dtype) @ field.relpos_code[pos]
                k[active] = k[active] * rel_code
        norm = k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return k / norm

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, memory_mode: str = "normal") -> Dict[str, torch.Tensor]:
        if memory_mode not in {"normal", "no_exact_query", "prefix_only"}:
            raise ValueError(memory_mode)
        hard = self._should_harden()
        ext = self.extractor(batch, hard=hard)
        A = self.build_adjacency(ext, batch["fact_mask"])
        lengths = batch["lengths"]
        final_frontier, prefix_frontiers = self.compose_frontiers(A, ext["query_source_dist"], ext["query_relation_dist"], lengths)
        q = self.soft_key(field, ext["query_source_dist"], ext["query_relation_dist"], lengths)
        B = int(A.shape[0]); D = int(self.cfg.key_dim); E = int(self.cfg.num_entities)
        mem = torch.zeros(B, D, E, dtype=A.dtype, device=A.device)
        if self.write_prefixes:
            for pos, frontier in enumerate(prefix_frontiers):
                plen_value = pos + 1
                if memory_mode == "normal":
                    write_mask = lengths >= plen_value
                else:
                    write_mask = lengths > plen_value
                if not bool(write_mask.any()):
                    continue
                pk = self.soft_key(field, ext["query_source_dist"], ext["query_relation_dist"], lengths, prefix_len=plen_value)
                mem = mem + write_mask.to(A.dtype).view(B, 1, 1) * pk.unsqueeze(-1).to(A.dtype) * frontier.unsqueeze(1)
        else:
            if memory_mode == "normal":
                mem = q.unsqueeze(-1).to(A.dtype) * final_frontier.unsqueeze(1)
        logits = self.output_scale * field.read(mem, q)
        return {"logits": logits, "memory": mem, "query_key": q, "A": A, "frontier": final_frontier, "extractor": ext}


def extractor_supervision_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    ext = out["extractor"]
    losses: List[torch.Tensor] = []
    stats: Dict[str, float] = {}
    fact_mask = batch["fact_mask"]
    if fact_mask.numel() and bool(fact_mask.any()):
        m = fact_mask.reshape(-1)
        def masked_ce(logits: torch.Tensor, labels: torch.Tensor, name: str) -> torch.Tensor:
            flat_logits = logits.reshape(-1, logits.shape[-1])[m]
            flat_labels = labels.reshape(-1)[m]
            loss = F.cross_entropy(flat_logits, flat_labels)
            pred = flat_logits.argmax(dim=-1)
            stats[f"{name}_acc"] = float(pred.eq(flat_labels).float().mean().detach().item())
            return loss
        losses.append(masked_ce(ext["fact_source_logits"], batch["fact_source_label"], "fact_source"))
        losses.append(masked_ce(ext["fact_relation_logits"], batch["fact_relation_label"], "fact_relation"))
        losses.append(masked_ce(ext["fact_target_logits"], batch["fact_target_label"], "fact_target"))
    qs_loss = F.cross_entropy(ext["query_source_logits"], batch["source"])
    stats["query_source_acc"] = float(ext["query_source_logits"].argmax(dim=-1).eq(batch["source"]).float().mean().detach().item())
    losses.append(qs_loss)
    qmask = batch["query_relation_mask"].reshape(-1)
    qr_logits = ext["query_relation_logits"].reshape(-1, ext["query_relation_logits"].shape[-1])[qmask]
    qr_labels = batch["q_rels"].reshape(-1)[qmask]
    qr_loss = F.cross_entropy(qr_logits, qr_labels)
    stats["query_relation_acc"] = float(qr_logits.argmax(dim=-1).eq(qr_labels).float().mean().detach().item())
    losses.append(qr_loss)
    total = sum(losses) / max(1, len(losses))
    return total, stats


@torch.no_grad()
def extractor_slot_metrics(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, int]]:
    ext = out["extractor"]
    metrics: Dict[str, Tuple[int, int]] = {}
    fact_mask = batch["fact_mask"]
    if fact_mask.numel() and bool(fact_mask.any()):
        m = fact_mask.reshape(-1)
        for key, label_key, name in [
            ("fact_source_logits", "fact_source_label", "fact_source"),
            ("fact_relation_logits", "fact_relation_label", "fact_relation"),
            ("fact_target_logits", "fact_target_label", "fact_target"),
        ]:
            logits = ext[key].reshape(-1, ext[key].shape[-1])[m]
            labels = batch[label_key].reshape(-1)[m]
            metrics[name] = (int(logits.argmax(dim=-1).eq(labels).sum().item()), int(labels.numel()))
    logits = ext["query_source_logits"]
    labels = batch["source"]
    metrics["query_source"] = (int(logits.argmax(dim=-1).eq(labels).sum().item()), int(labels.numel()))
    qmask = batch["query_relation_mask"].reshape(-1)
    qr_logits = ext["query_relation_logits"].reshape(-1, ext["query_relation_logits"].shape[-1])[qmask]
    qr_labels = batch["q_rels"].reshape(-1)[qmask]
    metrics["query_relation"] = (int(qr_logits.argmax(dim=-1).eq(qr_labels).sum().item()), int(qr_labels.numel()))
    return metrics


def train_learned_extractor_writer(
    cfg: ClosureWriterConfig,
    tok: LearnedGroundingTextTokenizer,
    field: HolographicClosureField,
    *,
    device: torch.device | str = "cpu",
    train_steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    extraction_weight: float = 2.0,
    alias_train_prob: float = 0.35,
    noise_train_prob: float = 0.15,
    use_mlp_extractor: bool = False,
) -> Tuple[LearnedExtractorClosureWriter, Dict[str, object]]:
    device = torch.device(device)
    writer = LearnedExtractorClosureWriter(tok, cfg, write_prefixes=False, output_scale=30.0, use_mlp_extractor=use_mlp_extractor).to(device)
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
            examples, tok, cfg, device=device, rng=rng,
            relation_aliases=(rng.random() < alias_train_prob),
            extra_text_noise=(rng.random() < noise_train_prob),
            fact_order_shuffle=(rng.random() < 0.20),
        )
        out = writer(batch, field)
        answer_loss = F.cross_entropy(out["logits"], batch["target"])
        extract_loss, extract_stats = extractor_supervision_loss(out, batch)
        loss = answer_loss + float(extraction_weight) * extract_loss
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
            print(json.dumps({"learned_grounding_train_progress": snap}, sort_keys=True), flush=True)
    meta = {
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "extraction_weight": float(extraction_weight),
        "alias_train_prob": float(alias_train_prob),
        "noise_train_prob": float(noise_train_prob),
        "use_mlp_extractor": bool(use_mlp_extractor),
        "snapshots": snapshots,
        "elapsed_train_sec": float(time.perf_counter() - t0),
    }
    return writer, meta


@torch.no_grad()
def evaluate_learned_grounding(
    cfg: ClosureWriterConfig,
    tok: LearnedGroundingTextTokenizer,
    field: HolographicClosureField,
    learned_writer: LearnedExtractorClosureWriter,
    dp_writer: SemiringClosureWriter,
    *,
    lengths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    device = torch.device(device)
    learned_writer.eval(); dp_writer.eval(); field.eval()
    rng = random.Random(cfg.seed + 8303)
    rows: List[Dict[str, float]] = []
    for L in lengths:
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 9000 + int(L))
        examples = gen.make_examples(int(L), cfg.eval_n)
        wrong_gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 10000 + int(L))
        wrong_examples = wrong_gen.make_examples(int(L), cfg.eval_n)
        row_counts: Dict[str, List[int]] = {}
        def add(name: str, c: int, n: int) -> None:
            row_counts.setdefault(name, [0, 0])
            row_counts[name][0] += int(c)
            row_counts[name][1] += int(n)

        add("exact_dict", *symbolic_accuracy(examples, exact_dict_answer))
        add("raw_onehop", *symbolic_accuracy(examples, raw_onehop_answer))

        # DP baseline needs the generic canonical tokenizer/batch. Import
        # locally to avoid confusing the learned-token collate path.
        from generic_closure_writer import ClosureTextTokenizer, collate_examples as collate_generic_examples
        generic_tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)

        for batch_examples, wrong_batch_examples in zip(iter_minibatches(examples, cfg.eval_batch_size), iter_minibatches(wrong_examples, cfg.eval_batch_size)):
            batch = collate_learned_grounding_examples(batch_examples, tok, cfg, device=device, rng=rng)
            out = learned_writer(batch, field)
            add("learned_extractor_writer", *accuracy_from_logits(out["logits"], batch["target"]))
            for k, (c, n) in extractor_slot_metrics(out, batch).items():
                add(f"extractor_{k}", c, n)

            # Deterministic parser+DP constructive baseline.
            generic_batch = collate_generic_examples(batch_examples, generic_tok, cfg, device=device, rng=rng)
            add("dp_parser_semiring_writer", *accuracy_from_logits(dp_writer(generic_batch, field)["logits"], generic_batch["target"]))

            controls = [
                ("query_only", "learned_query_only", {}),
                ("no_facts", "learned_no_facts", {}),
                ("wrong_facts", "learned_wrong_facts", {}),
                ("reversed_order", "learned_reversed_order", {}),
                ("shuffled_order", "learned_shuffled_order", {}),
                ("first_order", "learned_first_order", {}),
                ("normal", "learned_fact_order_shuffle", {"fact_order_shuffle": True}),
                ("normal", "learned_extra_text_noise", {"extra_text_noise": True}),
                ("normal", "learned_relation_aliases", {"relation_aliases": True}),
                ("normal", "learned_entity_renaming", {"entity_renaming": True}),
            ]
            for variant, metric_name, extra_kwargs in controls:
                kwargs = dict(extra_kwargs)
                if variant == "wrong_facts":
                    vb = collate_learned_grounding_examples(batch_examples, tok, cfg, variant=variant, wrong_fact_examples=wrong_batch_examples, device=device, rng=rng, **kwargs)
                else:
                    vb = collate_learned_grounding_examples(batch_examples, tok, cfg, variant=variant, device=device, rng=rng, **kwargs)
                logits = learned_writer(vb, field)["logits"]
                if variant in {"reversed_order", "shuffled_order"}:
                    mask = vb["control_changed"]
                    if bool(mask.any()):
                        add(metric_name, *accuracy_from_logits(logits, vb["target"], mask=mask))
                    else:
                        add(metric_name, 0, 0)
                else:
                    add(metric_name, *accuracy_from_logits(logits, vb["target"]))

            add("learned_no_exact_query", *accuracy_from_logits(learned_writer(batch, field, memory_mode="no_exact_query")["logits"], batch["target"]))
            add("learned_prefix_only", *accuracy_from_logits(learned_writer(batch, field, memory_mode="prefix_only")["logits"], batch["target"]))

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
        print(json.dumps({"learned_grounding_eval_by_length": printable}, sort_keys=True), flush=True)
    return rows


def write_learned_grounding_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "LEARNED_GROUNDING_CLOSURE_RESULTS.json"
    csv_path = out_dir / "LEARNED_GROUNDING_CLOSURE_RESULTS.csv"
    report_path = out_dir / "LEARNED_GROUNDING_CLOSURE_REPORT.md"
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
        "length", "n", "learned_extractor_writer_acc", "dp_parser_semiring_writer_acc", "raw_onehop_acc", "exact_dict_acc",
        "extractor_fact_source_acc", "extractor_fact_relation_acc", "extractor_fact_target_acc", "extractor_query_source_acc", "extractor_query_relation_acc",
    ]
    control_cols = [
        "length", "learned_query_only_acc", "learned_no_facts_acc", "learned_wrong_facts_acc",
        "learned_reversed_order_acc", "learned_shuffled_order_acc", "learned_first_order_acc",
        "learned_no_exact_query_acc", "learned_prefix_only_acc",
        "learned_fact_order_shuffle_acc", "learned_extra_text_noise_acc", "learned_relation_aliases_acc", "learned_entity_renaming_acc",
    ]
    lines: List[str] = []
    lines.append("# LearnedTransitionExtractor / HolographicClosureField results")
    lines.append("")
    lines.append("This diagnostic keeps explicit semiring closure construction but replaces deterministic text parsing with a learned slot-grounding extractor. The model receives raw controlled text, predicts distributions over fact source/relation/target slots and query source/relation slots, builds A[r,s,t], composes the query path, writes m_closure, and answers by exactly one HolographicClosureField read.")
    lines.append("")
    lines.append("No oracle full_closure writes or target labels are used to construct memory. Slot-level extraction labels are used as supervised grounding during training; this tests learned grounding plus structured closure composition, not generic Transformer-only closure writing.")
    lines.append("")
    lines.append("Training lengths: L in {1, 2, 3}. Evaluation is reported by length only; no averaged headline is used.")
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
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This should be read as a hybrid result. If it succeeds, it does not reverse the generic-writer negative result. It shows that the useful split is learned grounding/extraction followed by explicit differentiable closure construction and a single associative read. If it fails, the bottleneck is the learned extractor rather than the HolographicClosureField reader.")
    lines.append("")
    cfg_meta = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    lines.append(f"Run configuration: eval_n={int(cfg_meta.get('eval_n', 0))}, key_dim={int(cfg_meta.get('key_dim', 0))}, d_model={int(cfg_meta.get('d_model', 0))}, relation_aliases={meta.get('relation_aliases', 'NA')}. Chance answer accuracy is approximately 1/num_entities = {1.0 / max(1, int(cfg_meta.get('num_entities', 1))):.4f}.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_learned_grounding(cfg: ClosureWriterConfig, out_dir: Path, device: torch.device | str = "cpu", train_steps: int = 500, relation_aliases: int = 3) -> Dict[str, object]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=relation_aliases)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(device)
    learned, train_meta = train_learned_extractor_writer(cfg, tok, field, device=device, train_steps=train_steps, batch_size=cfg.batch_size)
    # DP baseline uses the generic writer tokenizer internally.
    from generic_closure_writer import ClosureTextTokenizer
    generic_tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    dp = SemiringClosureWriter(generic_tok, cfg, learn_relation_match=False, write_prefixes=False, output_scale=20.0).to(device)
    rows = evaluate_learned_grounding(cfg, tok, field, learned, dp, device=device)
    meta = {
        "config": asdict(cfg),
        "relation_aliases": int(relation_aliases),
        "train": train_meta,
        "num_parameters": {"learned_extractor_writer": sum(p.numel() for p in learned.parameters())},
    }
    paths = write_learned_grounding_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run learned-grounding closure diagnostic.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=ClosureWriterConfig.seed)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-entities", type=int, default=ClosureWriterConfig.num_entities)
    p.add_argument("--num-relations", type=int, default=ClosureWriterConfig.num_relations)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--threads", type=int, default=ClosureWriterConfig.torch_threads)
    p.add_argument("--max-seq-len", type=int, default=1600)
    p.add_argument("--relation-aliases", type=int, default=3)
    p.add_argument("--same-relation-branch-prob", type=float, default=ClosureWriterConfig.same_relation_branch_prob)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    )
    result = run_learned_grounding(cfg, Path(args.out_dir), device=args.device, train_steps=args.train_steps, relation_aliases=args.relation_aliases)
    print(json.dumps({"status": "done", "paths": result["paths"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
