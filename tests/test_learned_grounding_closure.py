import random
import pytest

torch = pytest.importorskip("torch")

from generic_closure_writer import (
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    HolographicClosureField,
    accuracy_from_logits,
)
from learned_grounding_closure import (
    LearnedGroundingTextTokenizer,
    collate_learned_grounding_examples,
    LearnedExtractorClosureWriter,
    extractor_supervision_loss,
    train_learned_extractor_writer,
)


def small_cfg():
    torch.set_num_threads(2)
    return ClosureWriterConfig(
        seed=9601,
        num_entities=24,
        num_relations=4,
        max_path_len=16,
        key_dim=96,
        d_model=96,
        batch_size=32,
        eval_n=32,
        eval_batch_size=16,
        base_distractors=4,
        distractors_per_hop=2,
        max_seq_len=900,
        torch_threads=2,
    )


def test_learned_grounding_collate_exposes_grounding_slots_and_labels():
    cfg = small_cfg()
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=3)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=1)
    ex = gen.make_example(3)
    batch = collate_learned_grounding_examples([ex], tok, cfg, relation_aliases=True, extra_text_noise=True, rng=random.Random(2))
    assert int(batch["source"][0]) == ex.source
    assert int(batch["target"][0]) == ex.target
    assert int(batch["lengths"][0]) == 3
    assert tuple(batch["q_rels"][0, :3].tolist()) == ex.relations
    assert int(batch["fact_mask"].sum().item()) == len(ex.edges)
    # Slot labels should be valid entity/relation IDs, independent of relation aliases.
    assert int(batch["fact_source_label"].max().item()) < cfg.num_entities
    assert int(batch["fact_target_label"].max().item()) < cfg.num_entities
    assert int(batch["fact_relation_label"].max().item()) < cfg.num_relations


def test_identity_grounded_learned_writer_solves_long_paths_with_one_read():
    cfg = small_cfg()
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=3)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=3)
    writer = LearnedExtractorClosureWriter(tok, cfg, output_scale=30.0)
    writer.extractor.set_token_identity_grounding(tok)
    writer.eval()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=4)
    examples = gen.make_examples(12, 24)
    batch = collate_learned_grounding_examples(examples, tok, cfg, relation_aliases=True, rng=random.Random(5))
    out = writer(batch, field)
    c, n = accuracy_from_logits(out["logits"], batch["target"])
    assert c == n


def test_no_facts_and_no_exact_controls_collapse_for_identity_writer():
    cfg = small_cfg()
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=2)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=6)
    writer = LearnedExtractorClosureWriter(tok, cfg, output_scale=30.0)
    writer.extractor.set_token_identity_grounding(tok)
    writer.eval()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=7)
    examples = gen.make_examples(8, 32)
    batch = collate_learned_grounding_examples(examples, tok, cfg, rng=random.Random(8))
    nofacts = collate_learned_grounding_examples(examples, tok, cfg, variant="no_facts", rng=random.Random(8))
    c0, n0 = accuracy_from_logits(writer(nofacts, field)["logits"], nofacts["target"])
    c1, n1 = accuracy_from_logits(writer(batch, field, memory_mode="no_exact_query")["logits"], batch["target"])
    assert c0 / n0 < 0.25
    assert c1 / n1 < 0.25


def test_short_training_learns_extractor_and_extrapolates_small_l8():
    cfg = small_cfg()
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=3)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=9)
    writer, meta = train_learned_extractor_writer(cfg, tok, field, train_steps=60, batch_size=48, learning_rate=3e-3, device="cpu")
    writer.eval()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=10)
    examples = gen.make_examples(8, 24)
    batch = collate_learned_grounding_examples(examples, tok, cfg, relation_aliases=True, extra_text_noise=True, rng=random.Random(11))
    out = writer(batch, field)
    c, n = accuracy_from_logits(out["logits"], batch["target"])
    loss, stats = extractor_supervision_loss(out, batch)
    assert c / n >= 0.90
    assert stats["query_source_acc"] >= 0.95
    assert stats["query_relation_acc"] >= 0.95
