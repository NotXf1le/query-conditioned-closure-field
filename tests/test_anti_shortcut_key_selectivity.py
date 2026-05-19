import random
import pytest

torch = pytest.importorskip("torch")

from generic_closure_writer import ClosureWriterConfig, ControlledDenseGraphTextQAGenerator, HolographicClosureField, exact_dict_answer
from learned_grounding_closure import LearnedGroundingTextTokenizer, collate_learned_grounding_examples, LearnedExtractorClosureWriter
from anti_shortcut_key_selectivity import (
    make_relation_swap_counterfactual,
    make_fact_swap_counterfactual,
    make_same_multiset_order_counterfactual,
    make_gold_edge_deleted,
    make_distractor_deleted,
    clone_with_scanned_slots,
    best_permutation_accuracy,
)


def test_counterfactual_constructors_change_symbolic_endpoint():
    cfg = ClosureWriterConfig(seed=123, num_entities=40, num_relations=4, max_path_len=8, base_distractors=2, distractors_per_hop=1)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=456)
    rng = random.Random(789)
    ex = gen.make_example(4, relations=(0, 1, 2, 3))
    assert exact_dict_answer(ex) == ex.target
    rel_cf = make_relation_swap_counterfactual(ex, cfg, rng)
    fact_cf = make_fact_swap_counterfactual(ex, cfg, rng)
    order_cf = make_same_multiset_order_counterfactual(ex, cfg, rng)
    assert rel_cf is not None and exact_dict_answer(rel_cf) == rel_cf.target and rel_cf.target != ex.target
    assert fact_cf is not None and exact_dict_answer(fact_cf) == fact_cf.target and fact_cf.target != ex.target
    assert order_cf is not None and exact_dict_answer(order_cf) == order_cf.target and order_cf.target != ex.target


def test_edge_deletion_breaks_and_distractor_deletion_preserves_path():
    cfg = ClosureWriterConfig(seed=124, num_entities=40, num_relations=4, max_path_len=8, base_distractors=4, distractors_per_hop=1)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=457)
    rng = random.Random(790)
    ex = gen.make_example(5)
    broken = make_gold_edge_deleted(ex, rng)
    clean = make_distractor_deleted(ex)
    assert broken is not None
    assert exact_dict_answer(broken) != ex.target
    assert exact_dict_answer(clean) == clean.target == ex.target


def test_scanned_slots_match_controlled_collate_positions():
    cfg = ClosureWriterConfig(seed=125, num_entities=40, num_relations=4, max_path_len=8, key_dim=24, d_model=24, base_distractors=2, distractors_per_hop=1)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=3)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=458)
    examples = [gen.make_example(3) for _ in range(4)]
    batch = collate_learned_grounding_examples(examples, tok, cfg, rng=random.Random(1), relation_aliases=True, extra_text_noise=True)
    scanned = clone_with_scanned_slots(batch, tok, cfg)
    for key in ["fact_source_pos", "fact_relation_pos", "fact_target_pos", "query_source_pos", "query_relation_pos"]:
        assert torch.equal(batch[key].cpu(), scanned[key].cpu())


def test_memory_zero_intervention_is_causal_with_identity_grounding():
    cfg = ClosureWriterConfig(seed=126, num_entities=40, num_relations=4, max_path_len=8, key_dim=32, d_model=96, base_distractors=2, distractors_per_hop=1)
    tok = LearnedGroundingTextTokenizer(cfg.num_entities, cfg.num_relations, relation_aliases=2)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=cfg.seed + 11)
    writer = LearnedExtractorClosureWriter(tok, cfg, output_scale=30.0, hard_eval_extraction=True)
    writer.extractor.set_token_identity_grounding(tok, scale=12.0)
    writer.eval()
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=459)
    examples = [gen.make_example(3) for _ in range(16)]
    batch = collate_learned_grounding_examples(examples, tok, cfg, rng=random.Random(2))
    with torch.no_grad():
        out = writer(batch, field)
        normal = out["logits"].argmax(dim=-1).eq(batch["target"]).float().mean().item()
        zero_logits = writer.output_scale * field.read(torch.zeros_like(out["memory"]), out["query_key"])
        zero = zero_logits.argmax(dim=-1).eq(batch["target"]).float().mean().item()
    assert normal == 1.0
    assert zero < 0.25


def test_best_permutation_accuracy_detects_label_permutation():
    conf = torch.tensor([[0, 5, 0], [0, 0, 4], [6, 0, 0]])
    acc, perm = best_permutation_accuracy(conf)
    assert acc == 1.0
    assert perm == [1, 2, 0]
