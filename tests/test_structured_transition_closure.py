import math
import random
import pytest

torch = pytest.importorskip("torch")

from generic_closure_writer import (
    ClosureWriterConfig,
    ClosureTextTokenizer,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    accuracy_from_logits,
    collate_examples,
)
from structured_transition_closure import ControlledTextGraphParser, SemiringClosureWriter


def small_cfg():
    torch.set_num_threads(2)
    return ClosureWriterConfig(
        seed=8501,
        num_entities=24,
        num_relations=4,
        max_path_len=16,
        key_dim=96,
        d_model=64,
        batch_size=16,
        eval_n=32,
        base_distractors=4,
        distractors_per_hop=2,
        torch_threads=2,
    )


def test_parser_reads_fact_and_query_tokens():
    cfg = small_cfg()
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=1)
    ex = gen.make_example(3)
    batch = collate_examples([ex], tok, cfg)
    parser = ControlledTextGraphParser(tok, cfg)
    parsed = parser.parse_batch(batch)
    assert int(parsed["source"][0]) == ex.source
    assert int(parsed["lengths"][0]) == 3
    assert tuple(parsed["q_rels"][0, :3].tolist()) == ex.relations
    for s, r, t in ex.edges:
        assert float(parsed["A"][0, r, s, t]) >= 1.0


def test_exact_dp_writer_solves_long_controlled_paths():
    cfg = small_cfg()
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=9)
    writer = SemiringClosureWriter(tok, cfg, learn_relation_match=False, write_prefixes=False)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=2)
    examples = gen.make_examples(12, 24)
    batch = collate_examples(examples, tok, cfg)
    out = writer(batch, field)
    c, n = accuracy_from_logits(out["logits"], batch["target"])
    assert c == n


def test_prefix_field_writer_has_single_read_and_no_exact_control_collapses():
    cfg = small_cfg()
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=10)
    writer = SemiringClosureWriter(tok, cfg, learn_relation_match=False, write_prefixes=True)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=3)
    examples = gen.make_examples(6, 32)
    batch = collate_examples(examples, tok, cfg)
    out = writer(batch, field)
    c, n = accuracy_from_logits(out["logits"], batch["target"])
    assert c / n >= 0.90
    out_no_exact = writer(batch, field, memory_mode="no_exact_query")
    c2, n2 = accuracy_from_logits(out_no_exact["logits"], batch["target"])
    assert c2 / n2 < 0.50


def test_neural_relation_match_can_be_set_to_identity():
    cfg = small_cfg()
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=11)
    writer = SemiringClosureWriter(tok, cfg, learn_relation_match=True, write_prefixes=False)
    with torch.no_grad():
        writer.rel_match_logits.fill_(-8.0)
        for r in range(cfg.num_relations):
            writer.rel_match_logits[r, r] = 8.0
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=4)
    examples = gen.make_examples(16, 16)
    batch = collate_examples(examples, tok, cfg)
    out = writer(batch, field)
    c, n = accuracy_from_logits(out["logits"], batch["target"])
    assert c == n
