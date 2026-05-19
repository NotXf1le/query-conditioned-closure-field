from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nonclosure_baseline_controls import (
    HopSupervisedScratchpadTransformer,
    RelativeBiasDirectTransformer,
    StrongerBaselineConfig,
    closure_cfg,
    graph_recurrent_logits,
)
from generic_closure_writer import (
    ClosureTextTokenizer,
    ControlledDenseGraphTextQAGenerator,
    collate_examples,
    exact_dict_answer,
)


def small_cfg() -> StrongerBaselineConfig:
    return StrongerBaselineConfig(
        seed=123,
        num_entities=36,
        num_relations=4,
        max_path_len=8,
        d_model=32,
        n_heads=4,
        n_layers=1,
        train_steps=2,
        batch_size=4,
        eval_n=4,
        eval_batch_size=4,
        max_seq_len=300,
    )


def test_stronger_baseline_forward_shapes() -> None:
    cfg = small_cfg()
    base_cfg = closure_cfg(cfg)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    gen = ControlledDenseGraphTextQAGenerator(base_cfg, seed=55)
    examples = gen.make_examples(4, 3)
    batch = collate_examples(examples, tok, base_cfg)

    rel = RelativeBiasDirectTransformer(tok.vocab_size, cfg)
    scratch = HopSupervisedScratchpadTransformer(tok.vocab_size, cfg)

    rel_logits = rel(batch)
    scratch_out = scratch(batch)

    assert rel_logits.shape == (3, cfg.num_entities)
    assert scratch_out["answer_logits"].shape == (3, cfg.num_entities)
    assert scratch_out["hop_logits"].shape == (3, cfg.max_path_len, cfg.num_entities)
    assert torch.isfinite(rel_logits).all()
    assert torch.isfinite(scratch_out["answer_logits"]).all()
    assert torch.isfinite(scratch_out["hop_logits"]).all()


def test_graph_recurrent_oracle_returns_generator_target() -> None:
    cfg = small_cfg()
    base_cfg = closure_cfg(cfg)
    gen = ControlledDenseGraphTextQAGenerator(base_cfg, seed=77)
    examples = [gen.make_example(L) for L in (1, 2, 3, 8)]

    logits = graph_recurrent_logits(examples, cfg, device="cpu")
    pred = logits.argmax(dim=-1).tolist()

    assert pred == [ex.target for ex in examples]
    assert [exact_dict_answer(ex) for ex in examples] == [ex.target for ex in examples]
