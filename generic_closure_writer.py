"""Generic TransformerClosureWriter diagnostic for HolographicClosureField.

This file intentionally separates the learned closure-field writer question from
oracle/vectorized transition composition.  The learned models receive only raw
one-hop fact text plus the query text.  The
TransformerClosureWriter and MLP writer must emit a dense closure memory
m_closure.  The answer is obtained by exactly one associative read:

    logits = HolographicClosureField.read(m_closure, query_key)

Oracle closure writes are implemented only as controls/baselines and are not used
in learned model forward passes or training losses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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


Edge = Tuple[int, int, int]  # source, relation, target


@dataclass(frozen=True)
class ClosureWriterConfig:
    seed: int = 7401
    num_entities: int = 48
    num_relations: int = 4
    max_path_len: int = 32
    key_dim: int = 96
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    train_steps: int = 900
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    curriculum: str = "staged"
    lr_schedule: str = "constant"
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    eval_n: int = 256
    eval_batch_size: int = 64
    base_distractors: int = 6
    distractors_per_hop: int = 3
    same_relation_branch_prob: float = 0.25
    max_seq_len: int = 900
    torch_threads: int = 4


@dataclass(frozen=True)
class PathQAExample:
    source: int
    relations: Tuple[int, ...]
    target: int
    path_nodes: Tuple[int, ...]
    edges: Tuple[Edge, ...]
    attempts: int


class ControlledDenseGraphTextQAGenerator:
    """Controlled graph/text QA generator with distractors and ambiguity rejection.

    The query endpoint is accepted only when set-valued traversal from the query
    source through the ordered relation sequence has exactly one endpoint and it
    is the intended target.  Distractors include off-path same-relation edges,
    wrong-relation branches out of path nodes, and occasional same-relation
    branches that are kept only when they do not make the final endpoint
    ambiguous.
    """

    def __init__(self, cfg: ClosureWriterConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.entities = tuple(f"e{i}" for i in range(cfg.num_entities))
        self.relations = tuple(f"r{i}" for i in range(cfg.num_relations))

    def _endpoints(self, source: int, relations: Sequence[int], edges: Sequence[Edge]) -> set[int]:
        frontier = {int(source)}
        for r in relations:
            nxt: set[int] = set()
            rr = int(r)
            for s, er, t in edges:
                if er == rr and s in frontier:
                    nxt.add(t)
            frontier = nxt
            if not frontier:
                break
        return frontier

    def _safe_distractors(self, path_nodes: Sequence[int], relations: Sequence[int], edges: set[Edge], length: int) -> None:
        # Wrong-relation branches from true path nodes are branching distractors
        # but cannot be followed at that query step.
        for i, node in enumerate(path_nodes[:-1]):
            true_r = int(relations[i])
            wrong_relations = [r for r in range(self.cfg.num_relations) if r != true_r]
            self.rng.shuffle(wrong_relations)
            for r in wrong_relations[: max(1, min(2, len(wrong_relations)))]:
                t = self.rng.randrange(self.cfg.num_entities)
                if t != path_nodes[i + 1]:
                    edges.add((int(node), int(r), int(t)))
        # Off-path edges with arbitrary relations create dense irrelevant facts.
        desired = self.cfg.base_distractors + self.cfg.distractors_per_hop * int(length)
        path_set = set(path_nodes)
        tries = 0
        while len(edges) < int(length) + desired and tries < 10000:
            tries += 1
            s = self.rng.randrange(self.cfg.num_entities)
            if s in path_set:
                continue
            r = self.rng.randrange(self.cfg.num_relations)
            t = self.rng.randrange(self.cfg.num_entities)
            if s != t:
                edges.add((s, r, t))

    def _normalize_forced_relations(self, length: int, relations: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
        if relations is None:
            return None
        out = tuple(int(r) for r in relations)
        if len(out) != int(length):
            raise ValueError(f"relations length must equal length {int(length)}, got {len(out)}")
        if any(r < 0 or r >= self.cfg.num_relations for r in out):
            raise ValueError(f"relation ids must be in [0,{self.cfg.num_relations})")
        return out

    def make_example(self, length: int, relations: Optional[Sequence[int]] = None) -> PathQAExample:
        L = int(length)
        if L < 1 or L > self.cfg.max_path_len:
            raise ValueError(f"length must be in [1,{self.cfg.max_path_len}], got {length}")
        if L + 1 > self.cfg.num_entities:
            raise ValueError("num_entities must exceed max_path_len")
        forced_relations = self._normalize_forced_relations(L, relations)

        for attempt in range(1, 201):
            path_nodes = tuple(self.rng.sample(range(self.cfg.num_entities), L + 1))
            relations = forced_relations or tuple(self.rng.randrange(self.cfg.num_relations) for _ in range(L))
            edges: set[Edge] = set((path_nodes[i], relations[i], path_nodes[i + 1]) for i in range(L))

            self._safe_distractors(path_nodes, relations, edges, L)

            # Add occasional same-relation decoy branches from the current path
            # node.  Ambiguity rejection below keeps only cases that do not
            # create multiple final endpoints for the full query.
            for i, node in enumerate(path_nodes[:-1]):
                if self.rng.random() < self.cfg.same_relation_branch_prob:
                    t = self.rng.randrange(self.cfg.num_entities)
                    if t != path_nodes[i + 1]:
                        edges.add((int(node), int(relations[i]), int(t)))

            edge_list = list(edges)
            self.rng.shuffle(edge_list)
            endpoints = self._endpoints(path_nodes[0], relations, edge_list)
            if len(endpoints) == 1 and next(iter(endpoints)) == path_nodes[-1]:
                return PathQAExample(
                    source=int(path_nodes[0]),
                    relations=tuple(int(x) for x in relations),
                    target=int(path_nodes[-1]),
                    path_nodes=tuple(int(x) for x in path_nodes),
                    edges=tuple(edge_list),
                    attempts=attempt,
                )

        # Conservative fallback: still has branching distractors, but avoids
        # same-relation path-node decoys that can cause ambiguous endpoints.
        path_nodes = tuple(self.rng.sample(range(self.cfg.num_entities), L + 1))
        relations = forced_relations or tuple(self.rng.randrange(self.cfg.num_relations) for _ in range(L))
        edges = set((path_nodes[i], relations[i], path_nodes[i + 1]) for i in range(L))
        self._safe_distractors(path_nodes, relations, edges, L)
        edge_list = list(edges)
        self.rng.shuffle(edge_list)
        endpoints = self._endpoints(path_nodes[0], relations, edge_list)
        if len(endpoints) != 1 or next(iter(endpoints)) != path_nodes[-1]:
            # This should be rare; keep the invariant explicit.
            edge_list = [(path_nodes[i], relations[i], path_nodes[i + 1]) for i in range(L)]
        return PathQAExample(
            source=int(path_nodes[0]), relations=tuple(int(x) for x in relations),
            target=int(path_nodes[-1]), path_nodes=tuple(int(x) for x in path_nodes),
            edges=tuple(edge_list), attempts=201,
        )

    def make_examples(self, length: int, n: int, relations: Optional[Sequence[int]] = None) -> List[PathQAExample]:
        return [self.make_example(length, relations=relations) for _ in range(int(n))]


class ClosureTextTokenizer:
    def __init__(self, num_entities: int, num_relations: int) -> None:
        vocab = ["<pad>", "<bos>", "<fact>", "<query>", "<read>", "<none>", ";", "from", "to", "follow", "then", "answer"]
        vocab += [f"e{i}" for i in range(num_entities)]
        vocab += [f"r{i}" for i in range(num_relations)]
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = list(vocab)
        self.pad_id = self.token_to_id["<pad>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def ent(self, e: int) -> int:
        return self.token_to_id[f"e{int(e)}"]

    def rel(self, r: int) -> int:
        return self.token_to_id[f"r{int(r)}"]

    def tok(self, t: str) -> int:
        return self.token_to_id[t]


def _different_permutation(rels: Sequence[int], rng: random.Random) -> Tuple[int, ...]:
    rels = tuple(int(r) for r in rels)
    if len(rels) <= 1:
        return rels
    idx = list(range(len(rels)))
    for _ in range(10):
        rng.shuffle(idx)
        candidate = tuple(rels[i] for i in idx)
        if candidate != rels:
            return candidate
    # deterministic fallback: rotate
    return tuple(list(rels[1:]) + [rels[0]])


def collate_examples(
    examples: Sequence[PathQAExample],
    tok: ClosureTextTokenizer,
    cfg: ClosureWriterConfig,
    *,
    variant: str = "normal",
    wrong_fact_examples: Optional[Sequence[PathQAExample]] = None,
    rng: Optional[random.Random] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor | List[PathQAExample]]:
    rng = rng or random.Random(0)
    seqs: List[List[int]] = []
    sources: List[int] = []
    lengths: List[int] = []
    targets: List[int] = []
    q_rels: List[List[int]] = []
    control_changed: List[bool] = []

    for i, ex in enumerate(examples):
        facts = list(ex.edges)
        rels = tuple(ex.relations)
        source = int(ex.source)
        if variant == "wrong_facts":
            if wrong_fact_examples is None:
                facts = list(examples[(i + 1) % len(examples)].edges)
            else:
                facts = list(wrong_fact_examples[i].edges)
        elif variant in {"query_only", "no_facts"}:
            facts = []
        elif variant == "reversed_order":
            rels = tuple(reversed(rels))
        elif variant == "shuffled_order":
            rels = _different_permutation(rels, rng)
        elif variant == "first_order":
            rels = tuple(rels[:1])
        elif variant != "normal":
            raise ValueError(f"unknown variant {variant}")

        tokens = [tok.tok("<bos>")]
        if facts:
            for s, r, t in facts:
                tokens.extend([tok.tok("<fact>"), tok.ent(s), tok.rel(r), tok.ent(t), tok.tok(";")])
        elif variant == "no_facts":
            tokens.extend([tok.tok("<fact>"), tok.tok("<none>"), tok.tok(";")])
        # query text: source plus an ordered relation sequence.
        tokens.extend([tok.tok("<query>"), tok.tok("from"), tok.ent(source), tok.tok("follow")])
        for j, r in enumerate(rels):
            if j > 0:
                tokens.append(tok.tok("then"))
            tokens.append(tok.rel(r))
        tokens.extend([tok.tok("answer"), tok.tok("<read>")])

        if len(tokens) > cfg.max_seq_len:
            raise RuntimeError(f"sequence too long: {len(tokens)} > {cfg.max_seq_len}")
        seqs.append(tokens)
        sources.append(source)
        lengths.append(len(rels))
        targets.append(int(ex.target))
        padded_rels = list(rels) + [0] * (cfg.max_path_len - len(rels))
        q_rels.append(padded_rels[: cfg.max_path_len])
        control_changed.append(tuple(rels) != tuple(ex.relations) or variant in {"query_only", "no_facts", "wrong_facts", "first_order"})

    B = len(seqs)
    max_len = max(len(s) for s in seqs)
    ids = torch.full((B, max_len), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    read_pos = torch.zeros(B, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
        # read token is always last non-pad.
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
        "control_changed": torch.tensor(control_changed, dtype=torch.bool, device=dev),
        "examples": list(examples),
    }


class HolographicClosureField(nn.Module):
    """Order-sensitive relation-sequence keyed associative field.

    Keys are fixed random-sign holographic products:

        k(s, r_1..r_L) = ent[s] * len[L] * Π_i rel_pos[i, r_i]

    where multiplication is elementwise and the final key is normalized.  The
    position-specific relation factors make reversed/shuffled relation orders
    produce different keys.
    """

    def __init__(self, num_entities: int, num_relations: int, key_dim: int, max_path_len: int, seed: int = 0, read_scale: float = 1.0) -> None:
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        def signs(*shape: int) -> torch.Tensor:
            return torch.where(torch.rand(shape, generator=g) < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
        self.register_buffer("entity_code", signs(num_entities, key_dim), persistent=True)
        self.register_buffer("length_code", signs(max_path_len + 1, key_dim), persistent=True)
        self.register_buffer("relpos_code", signs(max_path_len, num_relations, key_dim), persistent=True)
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.key_dim = int(key_dim)
        self.max_path_len = int(max_path_len)
        self.read_scale = float(read_scale)

    def key(self, source: torch.Tensor, q_rels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        source = source.long()
        lengths = lengths.long().clamp(0, self.max_path_len)
        q_rels = q_rels.long()
        k = self.entity_code[source] * self.length_code[lengths]
        max_l = int(lengths.max().item()) if lengths.numel() else 0
        for pos in range(max_l):
            active = lengths > pos
            if bool(active.any()):
                rel = q_rels[active, pos]
                k[active] = k[active] * self.relpos_code[pos, rel]
        return k / math.sqrt(float(self.key_dim))

    def prefix_keys(self, source: torch.Tensor, q_rels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Returns [B, K, D] where K=max(lengths)-1. Rows beyond each example's
        # prefix count are zero.
        B = int(source.shape[0])
        max_prefix = max(0, int(lengths.max().item()) - 1)
        if max_prefix == 0:
            return torch.zeros(B, 0, self.key_dim, dtype=self.entity_code.dtype, device=source.device)
        out = torch.zeros(B, max_prefix, self.key_dim, dtype=self.entity_code.dtype, device=source.device)
        for plen in range(1, max_prefix + 1):
            plen_t = torch.full_like(lengths, plen)
            active = lengths > plen
            if bool(active.any()):
                out[active, plen - 1] = self.key(source[active], q_rels[active], plen_t[active])
        return out

    def write_oracle(self, source: torch.Tensor, q_rels: torch.Tensor, lengths: torch.Tensor, target: torch.Tensor, *, include_exact: bool = True, prefixes_only: bool = False) -> torch.Tensor:
        B = int(source.shape[0])
        mem = torch.zeros(B, self.key_dim, self.num_entities, dtype=self.entity_code.dtype, device=source.device)
        if include_exact and not prefixes_only:
            k = self.key(source, q_rels, lengths)
            mem.scatter_add_(2, target.view(B, 1, 1).expand(B, self.key_dim, 1), k.unsqueeze(-1))
        # Prefix writes deliberately use their true prefix endpoints if available
        # only in symbolic evaluation code; here they are zero-valued controls.
        # The learned prefix_only control projects a learned memory onto prefix
        # key directions before exact read.
        return mem

    def read(self, memory: torch.Tensor, query_key: torch.Tensor) -> torch.Tensor:
        return self.read_scale * torch.einsum("bd,bde->be", query_key.to(memory.dtype), memory)

    def project_remove_exact(self, memory: torch.Tensor, query_key: torch.Tensor) -> torch.Tensor:
        coeff = torch.einsum("bd,bde->be", query_key.to(memory.dtype), memory)
        out = memory - query_key.to(memory.dtype).unsqueeze(-1) * coeff.unsqueeze(1)
        # Avoid numerical dust preserving the exact target after subtracting an
        # oracle one-key memory.  Learned memories generally retain substantial
        # non-exact components if they used them.
        return torch.where(out.abs() < 1e-6, torch.zeros_like(out), out)

    def project_prefix_only(self, memory: torch.Tensor, prefix_keys: torch.Tensor) -> torch.Tensor:
        if prefix_keys.shape[1] == 0:
            return torch.zeros_like(memory)
        B, K, D = prefix_keys.shape
        projected = torch.zeros_like(memory)
        # Batched modified Gram-Schmidt.  K<=31 and D is small, so a Python loop
        # is fine and keeps the projection numerically transparent.
        basis = []
        for j in range(K):
            v = prefix_keys[:, j, :].to(memory.dtype)
            for b in basis:
                v = v - (v * b).sum(dim=-1, keepdim=True) * b
            norm = v.norm(dim=-1, keepdim=True)
            b = torch.where(norm > 1e-6, v / norm.clamp_min(1e-6), torch.zeros_like(v))
            basis.append(b)
            coeff = torch.einsum("bd,bde->be", b, memory)
            projected = projected + b.unsqueeze(-1) * coeff.unsqueeze(1)
        return projected


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].to(x.dtype).unsqueeze(0)


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, cfg: ClosureWriterConfig) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor, read_pos: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.emb(input_ids))
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        idx = read_pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
        return x.gather(1, idx).squeeze(1)


class TransformerClosureWriter(nn.Module):
    def __init__(self, vocab_size: int, cfg: ClosureWriterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(vocab_size, cfg)
        self.memory_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.key_dim * cfg.num_entities),
        )

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, memory_mode: str = "normal") -> Dict[str, torch.Tensor]:
        h = self.backbone(batch["input_ids"], batch["mask"], batch["read_pos"])
        mem = self.memory_head(h).view(-1, self.cfg.key_dim, self.cfg.num_entities)
        q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
        if memory_mode == "no_exact_query":
            mem = field.project_remove_exact(mem, q)
        elif memory_mode == "prefix_only":
            prefixes = field.prefix_keys(batch["source"], batch["q_rels"], batch["lengths"])
            mem = field.project_prefix_only(mem, prefixes)
        elif memory_mode != "normal":
            raise ValueError(memory_mode)
        logits = field.read(mem, q)
        return {"logits": logits, "memory": mem, "query_key": q}


class MLPClosureWriter(nn.Module):
    def __init__(self, vocab_size: int, cfg: ClosureWriterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.key_dim * cfg.num_entities),
        )

    def forward(self, batch: Dict[str, torch.Tensor], field: HolographicClosureField, *, memory_mode: str = "normal") -> Dict[str, torch.Tensor]:
        x = self.emb(batch["input_ids"])
        m = batch["mask"].to(x.dtype).unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        mem = self.net(pooled).view(-1, self.cfg.key_dim, self.cfg.num_entities)
        q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
        if memory_mode == "no_exact_query":
            mem = field.project_remove_exact(mem, q)
        elif memory_mode == "prefix_only":
            prefixes = field.prefix_keys(batch["source"], batch["q_rels"], batch["lengths"])
            mem = field.project_prefix_only(mem, prefixes)
        elif memory_mode != "normal":
            raise ValueError(memory_mode)
        return {"logits": field.read(mem, q), "memory": mem, "query_key": q}


class VanillaTransformerAnswerHead(nn.Module):
    def __init__(self, vocab_size: int, cfg: ClosureWriterConfig) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size, cfg)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_entities))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.backbone(batch["input_ids"], batch["mask"], batch["read_pos"])
        return self.head(h)


def exact_dict_answer(ex: PathQAExample, relations: Optional[Sequence[int]] = None) -> Optional[int]:
    rels = tuple(ex.relations if relations is None else relations)
    frontier = {int(ex.source)}
    for r in rels:
        nxt: set[int] = set()
        for s, er, t in ex.edges:
            if er == int(r) and s in frontier:
                nxt.add(t)
        frontier = nxt
        if not frontier:
            return None
    if len(frontier) == 1:
        return next(iter(frontier))
    return None


def raw_onehop_answer(ex: PathQAExample) -> Optional[int]:
    if not ex.relations:
        return None
    first = int(ex.relations[0])
    outs = {t for s, r, t in ex.edges if s == int(ex.source) and r == first}
    if len(outs) == 1:
        return next(iter(outs))
    return None


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    pred = logits.argmax(dim=-1)
    correct = pred.eq(target)
    if mask is not None:
        correct = correct[mask]
        denom = int(mask.sum().item())
    else:
        denom = int(target.numel())
    return int(correct.sum().item()), denom


def symbolic_accuracy(examples: Sequence[PathQAExample], fn) -> Tuple[int, int]:
    correct = 0
    for ex in examples:
        ans = fn(ex)
        correct += int(ans == ex.target)
    return correct, len(examples)


def iter_minibatches(items: Sequence[PathQAExample], batch_size: int) -> Iterable[List[PathQAExample]]:
    for i in range(0, len(items), int(batch_size)):
        yield list(items[i : i + int(batch_size)])


def train_models(cfg: ClosureWriterConfig, device: torch.device | str = "cpu") -> Tuple[TransformerClosureWriter, MLPClosureWriter, VanillaTransformerAnswerHead, HolographicClosureField, ClosureTextTokenizer, Dict[str, object]]:
    if cfg.torch_threads > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    if cfg.curriculum not in {"staged", "mixed"}:
        raise ValueError(f"unknown curriculum {cfg.curriculum!r}")
    if cfg.lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11, read_scale=1.0).to(device)
    transformer_writer = TransformerClosureWriter(tok.vocab_size, cfg).to(device)
    mlp_writer = MLPClosureWriter(tok.vocab_size, cfg).to(device)
    vanilla = VanillaTransformerAnswerHead(tok.vocab_size, cfg).to(device)
    params = list(transformer_writer.parameters()) + list(mlp_writer.parameters()) + list(vanilla.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 101)
    rng = random.Random(cfg.seed + 202)
    t0 = time.perf_counter()
    last: Dict[str, float] = {}

    def train_choices_for_step(step: int) -> Tuple[int, ...]:
        if cfg.curriculum == "mixed":
            return (1, 2, 3)
        if step <= max(1, int(0.25 * cfg.train_steps)):
            return (1,)
        if step <= max(1, int(0.55 * cfg.train_steps)):
            return (1, 2)
        return (1, 2, 3)

    def lr_for_step(step: int) -> float:
        base = float(cfg.learning_rate)
        if cfg.lr_schedule == "constant":
            return base
        total = max(1, int(cfg.train_steps))
        warmup = max(1, int(float(cfg.warmup_frac) * total))
        min_ratio = float(cfg.min_lr_ratio)
        if step <= warmup:
            return base * float(step) / float(warmup)
        if total <= warmup:
            return base
        progress = (float(step) - float(warmup)) / max(1.0, float(total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return base * (min_ratio + (1.0 - min_ratio) * cosine)

    for step in range(1, int(cfg.train_steps) + 1):
        for group in opt.param_groups:
            group["lr"] = lr_for_step(step)
        # The default staged curriculum avoids making 3-hop composition the
        # first thing learned from a cold start; stress runs can use mixed
        # lengths from the first step.
        train_choices = train_choices_for_step(step)
        lengths = [rng.choice(train_choices) for _ in range(cfg.batch_size)]
        examples = [gen.make_example(L) for L in lengths]
        batch = collate_examples(examples, tok, cfg, device=device, rng=rng)
        target = batch["target"]
        tw_logits = transformer_writer(batch, field)["logits"]
        mlp_logits = mlp_writer(batch, field)["logits"]
        vanilla_logits = vanilla(batch)
        loss_tw = F.cross_entropy(tw_logits, target)
        loss_mlp = F.cross_entropy(mlp_logits, target)
        loss_vanilla = F.cross_entropy(vanilla_logits, target)
        loss = loss_tw + loss_mlp + loss_vanilla
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if step == 1 or step % max(1, cfg.train_steps // 6) == 0 or step == cfg.train_steps:
            with torch.no_grad():
                c_tw, n = accuracy_from_logits(tw_logits, target)
                c_mlp, _ = accuracy_from_logits(mlp_logits, target)
                c_v, _ = accuracy_from_logits(vanilla_logits, target)
            last = {
                "step": float(step),
                "loss_transformer_writer": float(loss_tw.detach().item()),
                "loss_mlp_writer": float(loss_mlp.detach().item()),
                "loss_vanilla_transformer": float(loss_vanilla.detach().item()),
                "train_batch_acc_transformer_writer": c_tw / max(1, n),
                "train_batch_acc_mlp_writer": c_mlp / max(1, n),
                "train_batch_acc_vanilla_transformer": c_v / max(1, n),
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "elapsed_sec": time.perf_counter() - t0,
            }
            print(json.dumps({"train_progress": last}, sort_keys=True), flush=True)
    meta = {
        "config": asdict(cfg),
        "final_train_snapshot": last,
        "num_parameters": {
            "transformer_writer": sum(p.numel() for p in transformer_writer.parameters()),
            "mlp_writer": sum(p.numel() for p in mlp_writer.parameters()),
            "vanilla_transformer": sum(p.numel() for p in vanilla.parameters()),
        },
        "elapsed_train_sec": time.perf_counter() - t0,
    }
    return transformer_writer, mlp_writer, vanilla, field, tok, meta


@torch.no_grad()
def evaluate_by_length(
    cfg: ClosureWriterConfig,
    transformer_writer: TransformerClosureWriter,
    mlp_writer: MLPClosureWriter,
    vanilla: VanillaTransformerAnswerHead,
    field: HolographicClosureField,
    tok: ClosureTextTokenizer,
    *,
    lengths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    device = torch.device(device)
    transformer_writer.eval(); mlp_writer.eval(); vanilla.eval(); field.eval()
    rows: List[Dict[str, float]] = []
    rng = random.Random(cfg.seed + 303)
    for L in lengths:
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 1000 + int(L))
        examples = gen.make_examples(int(L), cfg.eval_n)
        wrong_gen = ControlledDenseGraphTextQAGenerator(cfg, seed=cfg.seed + 2000 + int(L))
        wrong_examples = wrong_gen.make_examples(int(L), cfg.eval_n)
        row_counts: Dict[str, List[int]] = {}
        def add(name: str, c: int, n: int) -> None:
            if name not in row_counts:
                row_counts[name] = [0, 0]
            row_counts[name][0] += int(c); row_counts[name][1] += int(n)

        # Symbolic baselines and controls.
        add("exact_dict", *symbolic_accuracy(examples, exact_dict_answer))
        add("raw_onehop", *symbolic_accuracy(examples, raw_onehop_answer))

        for batch_examples, wrong_batch_examples in zip(iter_minibatches(examples, cfg.eval_batch_size), iter_minibatches(wrong_examples, cfg.eval_batch_size)):
            batch = collate_examples(batch_examples, tok, cfg, device=device, rng=rng)
            target = batch["target"]
            # Learned models, normal.
            tw = transformer_writer(batch, field)
            mlp = mlp_writer(batch, field)
            van = vanilla(batch)
            add("transformer_writer", *accuracy_from_logits(tw["logits"], target))
            add("mlp_writer", *accuracy_from_logits(mlp["logits"], target))
            add("vanilla_transformer", *accuracy_from_logits(van, target))

            # Oracle full closure read: exact query key is written.  Control only.
            q = tw["query_key"]
            oracle_mem = torch.zeros(len(batch_examples), cfg.key_dim, cfg.num_entities, device=device)
            oracle_mem.scatter_add_(2, target.view(-1, 1, 1).expand(-1, cfg.key_dim, 1), q.unsqueeze(-1))
            oracle_logits = 10.0 * field.read(oracle_mem, q)
            add("oracle_full_closure", *accuracy_from_logits(oracle_logits, target))

            # Prefix-only/no-exact oracle controls write only proper prefix
            # closures for this query path.  They deliberately do not include
            # the exact queried path key.
            prefix_mem = torch.zeros_like(oracle_mem)
            max_l = int(batch["lengths"].max().item())
            for plen in range(1, max_l):
                active_idx = [bi for bi, exb in enumerate(batch_examples) if len(exb.relations) > plen]
                if not active_idx:
                    continue
                idx_t = torch.tensor(active_idx, dtype=torch.long, device=device)
                plen_t = torch.full((len(active_idx),), plen, dtype=torch.long, device=device)
                pk = field.key(batch["source"][idx_t], batch["q_rels"][idx_t], plen_t)
                pt = torch.tensor([batch_examples[bi].path_nodes[plen] for bi in active_idx], dtype=torch.long, device=device)
                prefix_mem[idx_t].scatter_add_(2, pt.view(-1, 1, 1).expand(-1, cfg.key_dim, 1), pk.unsqueeze(-1))
            add("oracle_no_exact_query", *accuracy_from_logits(field.read(prefix_mem, q), target))
            add("oracle_prefix_only", *accuracy_from_logits(field.read(prefix_mem, q), target))

            # Learned writer causal controls.
            for variant, metric_name in [
                ("query_only", "tw_query_only"),
                ("no_facts", "tw_no_facts"),
                ("wrong_facts", "tw_wrong_facts"),
                ("reversed_order", "tw_reversed_order"),
                ("shuffled_order", "tw_shuffled_order"),
                ("first_order", "tw_first_order"),
            ]:
                if variant == "wrong_facts":
                    vb = collate_examples(batch_examples, tok, cfg, variant=variant, wrong_fact_examples=wrong_batch_examples, device=device, rng=rng)
                else:
                    vb = collate_examples(batch_examples, tok, cfg, variant=variant, device=device, rng=rng)
                out = transformer_writer(vb, field)["logits"]
                # For order controls, report only changed controls when possible.
                if variant in {"reversed_order", "shuffled_order"}:
                    mask = vb["control_changed"]
                    if bool(mask.any()):
                        add(metric_name, *accuracy_from_logits(out, vb["target"], mask=mask))
                    else:
                        add(metric_name, 0, 0)
                else:
                    add(metric_name, *accuracy_from_logits(out, vb["target"]))

            add("tw_no_exact_query", *accuracy_from_logits(transformer_writer(batch, field, memory_mode="no_exact_query")["logits"], target))
            add("tw_prefix_only", *accuracy_from_logits(transformer_writer(batch, field, memory_mode="prefix_only")["logits"], target))

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
        print(json.dumps({"eval_by_length": printable}, sort_keys=True), flush=True)
    return rows


def write_results(rows: List[Dict[str, float]], meta: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "GENERIC_CLOSURE_WRITER_RESULTS.json"
    csv_path = out_dir / "GENERIC_CLOSURE_WRITER_RESULTS.csv"
    report_path = out_dir / "GENERIC_CLOSURE_WRITER_REPORT.md"
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

    cols = [
        "length", "n", "transformer_writer_acc", "mlp_writer_acc", "vanilla_transformer_acc",
        "raw_onehop_acc", "exact_dict_acc", "oracle_full_closure_acc",
        "tw_query_only_acc", "tw_no_facts_acc", "tw_wrong_facts_acc", "tw_reversed_order_acc",
        "tw_shuffled_order_acc", "tw_first_order_acc", "tw_no_exact_query_acc", "tw_prefix_only_acc",
        "oracle_no_exact_query_acc", "oracle_prefix_only_acc",
    ]
    lines = []
    lines.append("# TransformerClosureWriter / HolographicClosureField results")
    lines.append("")
    lines.append("This is a learned-writer test. Oracle full-closure writes are included only as controls and are not evidence of learned reasoning.")
    lines.append("")
    lines.append("Training lengths: L in {1, 2, 3}. Evaluation is reported by length only; no averaged headline is used.")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, float("nan"))
            if c in {"length", "n"}:
                vals.append(str(int(v)))
            else:
                vals.append(fmt(float(v)))
        lines.append("| " + " | ".join(vals) + " |")
    # Simple negative/positive assessment by extrapolation lengths.
    extrap = [r for r in rows if int(r["length"]) > 3]
    trainish = [r for r in rows if int(r["length"]) <= 3]
    if extrap:
        # No headline average: use explicit per-length criterion.
        failed_lengths = [int(r["length"]) for r in extrap if r.get("transformer_writer_acc", 0.0) < 0.5]
        if failed_lengths:
            lines.append("")
            lines.append("Negative result: the TransformerClosureWriter did not extrapolate on these held-out lengths: " + ", ".join(map(str, failed_lengths)) + ".")
    if trainish:
        lines.append("")
        lines.append("Final train snapshot: `" + json.dumps(meta.get("final_train_snapshot", {}), sort_keys=True) + "`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}


def run_experiment(cfg: ClosureWriterConfig, out_dir: Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    t0 = time.perf_counter()
    tw, mlp, vanilla, field, tok, meta = train_models(cfg, device=device)
    rows = evaluate_by_length(cfg, tw, mlp, vanilla, field, tok, device=device)
    meta = dict(meta)
    meta["elapsed_total_sec"] = time.perf_counter() - t0
    paths = write_results(rows, meta, out_dir)
    return {"meta": meta, "rows": rows, "paths": paths}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate TransformerClosureWriter for HolographicClosureField.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=ClosureWriterConfig.seed)
    p.add_argument("--train-steps", type=int, default=ClosureWriterConfig.train_steps)
    p.add_argument("--batch-size", type=int, default=ClosureWriterConfig.batch_size)
    p.add_argument("--eval-n", type=int, default=ClosureWriterConfig.eval_n)
    p.add_argument("--eval-batch-size", type=int, default=ClosureWriterConfig.eval_batch_size)
    p.add_argument("--d-model", type=int, default=ClosureWriterConfig.d_model)
    p.add_argument("--key-dim", type=int, default=ClosureWriterConfig.key_dim)
    p.add_argument("--num-entities", type=int, default=ClosureWriterConfig.num_entities)
    p.add_argument("--learning-rate", type=float, default=ClosureWriterConfig.learning_rate)
    p.add_argument("--layers", type=int, default=ClosureWriterConfig.n_layers)
    p.add_argument("--heads", type=int, default=ClosureWriterConfig.n_heads)
    p.add_argument("--curriculum", type=str, choices=["staged", "mixed"], default=ClosureWriterConfig.curriculum)
    p.add_argument("--lr-schedule", type=str, choices=["constant", "cosine"], default=ClosureWriterConfig.lr_schedule)
    p.add_argument("--warmup-frac", type=float, default=ClosureWriterConfig.warmup_frac)
    p.add_argument("--min-lr-ratio", type=float, default=ClosureWriterConfig.min_lr_ratio)
    p.add_argument("--threads", type=int, default=ClosureWriterConfig.torch_threads)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClosureWriterConfig(
        seed=args.seed,
        num_entities=args.num_entities,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        d_model=args.d_model,
        key_dim=args.key_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        learning_rate=args.learning_rate,
        curriculum=args.curriculum,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        min_lr_ratio=args.min_lr_ratio,
        torch_threads=args.threads,
    )
    result = run_experiment(cfg, Path(args.out_dir), device=args.device)
    print(json.dumps({"status": "done", "paths": result["paths"], "elapsed_total_sec": result["meta"]["elapsed_total_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
