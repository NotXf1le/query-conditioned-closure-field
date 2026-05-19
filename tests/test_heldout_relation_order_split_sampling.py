from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

from generic_closure_writer import ClosureWriterConfig, ControlledDenseGraphTextQAGenerator
from heldout_relation_order_split import has_heldout_pattern, make_split_examples


def fast_cfg() -> ClosureWriterConfig:
    return ClosureWriterConfig(
        num_entities=64,
        num_relations=4,
        base_distractors=0,
        distractors_per_hop=0,
        same_relation_branch_prob=0.0,
    )


def test_adjacent_pair_trainlike_examples_are_generated_without_heldout_pairs() -> None:
    cfg = fast_cfg()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=101)

    examples = make_split_examples(
        gen,
        length=32,
        n=512,
        split="adjacent_pair_holdout",
        want_heldout=False,
        num_relations=cfg.num_relations,
        rng=random.Random(102),
    )

    assert len(examples) == 512
    assert all(not has_heldout_pattern(ex.relations, "adjacent_pair_holdout", cfg.num_relations) for ex in examples)


def test_repeated_relation_trainlike_examples_are_generated_without_adjacent_repeats() -> None:
    cfg = fast_cfg()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=201)

    examples = make_split_examples(
        gen,
        length=32,
        n=512,
        split="repeated_relation_holdout",
        want_heldout=False,
        num_relations=cfg.num_relations,
        rng=random.Random(202),
    )

    assert len(examples) == 512
    assert all(not has_heldout_pattern(ex.relations, "repeated_relation_holdout", cfg.num_relations) for ex in examples)


@pytest.mark.parametrize("split", ["adjacent_pair_holdout", "trigram_holdout", "repeated_relation_holdout"])
def test_hard_examples_are_generated_with_heldout_pattern(split: str) -> None:
    cfg = fast_cfg()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=301)

    examples = make_split_examples(
        gen,
        length=32,
        n=64,
        split=split,
        want_heldout=True,
        num_relations=cfg.num_relations,
        rng=random.Random(302),
    )

    assert len(examples) == 64
    assert all(has_heldout_pattern(ex.relations, split, cfg.num_relations) for ex in examples)
