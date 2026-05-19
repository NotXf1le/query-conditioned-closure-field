from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from closure_writer_diagnostic_ladder import (
    LadderConfig,
    LadderMLPWriter,
    LadderTransformerWriter,
    collate_ladder_examples,
    rank_one_target_memory,
)
from generic_closure_writer import (
    ClosureTextTokenizer,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
)


def test_rank_one_target_memory_reads_back_target() -> None:
    cfg = LadderConfig(num_entities=12, num_relations=3, key_dim=16, max_path_len=4)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=1)
    source = torch.tensor([0, 1])
    rels = torch.zeros(2, cfg.max_path_len, dtype=torch.long)
    rels[:, 0] = torch.tensor([1, 2])
    lengths = torch.tensor([1, 1])
    target = torch.tensor([3, 4])
    q = field.key(source, rels, lengths)

    mem = rank_one_target_memory(q, target, cfg.num_entities)
    logits = field.read(mem, q)

    assert mem.shape == (2, cfg.key_dim, cfg.num_entities)
    assert logits.argmax(dim=-1).tolist() == [3, 4]


def test_key_conditioned_and_tied_writers_forward_shapes() -> None:
    for variant in ["key_conditioned", "tied_key"]:
        cfg = LadderConfig(
            num_entities=16,
            num_relations=3,
            max_path_len=4,
            key_dim=16,
            d_model=24,
            n_heads=4,
            train_steps=2,
            batch_size=2,
            eval_n=2,
            writer_variant=variant,
        )
        tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
        field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=2)
        gen = ControlledDenseGraphTextQAGenerator(cfg, seed=3)
        examples = gen.make_examples(1, 2)
        batch = collate_ladder_examples(examples, tok, cfg, rung="one_hop_fact_write", device="cpu")

        tw = LadderTransformerWriter(tok.vocab_size, cfg)
        mlp = LadderMLPWriter(tok.vocab_size, cfg)
        tw_out = tw(batch, field, rung="one_hop_fact_write")
        mlp_out = mlp(batch, field, rung="one_hop_fact_write")

        assert tw_out["logits"].shape == (2, cfg.num_entities)
        assert tw_out["memory"].shape == (2, cfg.key_dim, cfg.num_entities)
        assert tw_out["query_key"].shape == (2, cfg.key_dim)
        assert mlp_out["logits"].shape == (2, cfg.num_entities)
