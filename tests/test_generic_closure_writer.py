from __future__ import annotations
import pytest

torch = pytest.importorskip("torch")
from generic_closure_writer import (
    ClosureWriterConfig,
    ControlledDenseGraphTextQAGenerator,
    ClosureTextTokenizer,
    HolographicClosureField,
    TransformerClosureWriter,
    collate_examples,
    exact_dict_answer,
)


def test_generator_rejects_ambiguous_endpoint() -> None:
    cfg = ClosureWriterConfig(num_entities=36, d_model=32, key_dim=32, n_layers=1, eval_n=8)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=11)
    for L in (1, 2, 3, 8, 32):
        ex = gen.make_example(L)
        assert exact_dict_answer(ex) == ex.target
        assert len(ex.relations) == L
        assert len(ex.edges) >= L


def test_generator_accepts_forced_relation_sequence() -> None:
    cfg = ClosureWriterConfig(num_entities=36, num_relations=4, d_model=32, key_dim=32, n_layers=1)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=12)

    ex = gen.make_example(4, relations=(3, 2, 1, 0))

    assert ex.relations == (3, 2, 1, 0)
    assert exact_dict_answer(ex) == ex.target
    with pytest.raises(ValueError, match="relations length"):
        gen.make_example(4, relations=(0, 1, 2))
    with pytest.raises(ValueError, match="relation ids"):
        gen.make_example(2, relations=(0, cfg.num_relations))


def test_holographic_keys_are_order_sensitive() -> None:
    cfg = ClosureWriterConfig(num_entities=36, d_model=32, key_dim=64, n_layers=1)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=123)
    src = torch.tensor([0, 0])
    rels = torch.tensor([[0, 1, 2] + [0] * (cfg.max_path_len - 3), [2, 1, 0] + [0] * (cfg.max_path_len - 3)])
    lengths = torch.tensor([3, 3])
    keys = field.key(src, rels, lengths)
    assert keys.shape == (2, cfg.key_dim)
    assert torch.isclose(keys.norm(dim=-1), torch.ones(2), atol=1e-6).all()
    assert float((keys[0] * keys[1]).sum()) < 0.5


def test_oracle_exact_key_and_no_exact_control() -> None:
    cfg = ClosureWriterConfig(num_entities=36, d_model=32, key_dim=64, n_layers=1)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=22)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=124)
    examples = gen.make_examples(3, 8)
    batch = collate_examples(examples, tok, cfg)
    q = field.key(batch["source"], batch["q_rels"], batch["lengths"])
    target = batch["target"]
    mem = torch.zeros(8, cfg.key_dim, cfg.num_entities)
    mem.scatter_add_(2, target.view(-1, 1, 1).expand(-1, cfg.key_dim, 1), q.unsqueeze(-1))
    assert torch.equal(field.read(mem, q).argmax(dim=-1), target)
    no_exact = field.project_remove_exact(mem, q)
    # Removing the exact query direction should remove the oracle evidence.
    assert no_exact.abs().max() == 0


def test_transformer_writer_single_read_shape() -> None:
    cfg = ClosureWriterConfig(num_entities=36, d_model=32, key_dim=32, n_layers=1, max_seq_len=900)
    gen = ControlledDenseGraphTextQAGenerator(cfg, seed=33)
    tok = ClosureTextTokenizer(cfg.num_entities, cfg.num_relations)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=125)
    model = TransformerClosureWriter(tok.vocab_size, cfg)
    batch = collate_examples(gen.make_examples(4, 5), tok, cfg)
    out = model(batch, field)
    assert out["memory"].shape == (5, cfg.key_dim, cfg.num_entities)
    assert out["query_key"].shape == (5, cfg.key_dim)
    assert out["logits"].shape == (5, cfg.num_entities)
    assert torch.isfinite(out["logits"]).all()
