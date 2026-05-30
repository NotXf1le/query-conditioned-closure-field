from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from closure_l1_causal_isolation import (
    L1IsolationConfig,
    VALID_CONDITIONS,
    build_condition_batch,
    field_alignment_diagnostics,
    make_field,
    run_experiment,
)
from closure_writer_diagnostic_ladder import rank_one_target_memory


def test_l1_isolation_condition_batches_cover_text_id_and_distractors() -> None:
    cfg = L1IsolationConfig(
        num_entities=12,
        num_relations=3,
        max_path_len=4,
        key_dim=16,
        d_model=24,
        n_heads=4,
        batch_size=2,
        eval_n=2,
        train_steps=2,
    )
    field, tok, gen = make_field(cfg, "cpu")

    text_batch = build_condition_batch(cfg, tok, gen, "l1_with_distractors", batch_size=2, device="cpu")
    id_batch = build_condition_batch(cfg, tok, gen, "id_only_write", batch_size=2, device="cpu")
    one_fact = build_condition_batch(cfg, tok, gen, "one_fact_no_distractor", batch_size=2, device="cpu")

    assert "input_ids" in text_batch
    assert "id_features" not in text_batch
    assert id_batch["id_features"].shape[0] == 2
    assert all(len(ex.edges) == 1 for ex in one_fact["examples"])
    assert field.key(text_batch["source"], text_batch["q_rels"], text_batch["lengths"]).shape == (2, cfg.key_dim)
    assert {"direct_qv_write", "field_supervised_mse", "field_supervised_read_ce"} <= VALID_CONDITIONS


def test_field_alignment_diagnostics_are_scale_aware() -> None:
    q = torch.tensor([[1.0, -1.0]], dtype=torch.float32) / torch.sqrt(torch.tensor(2.0))
    target = torch.tensor([1], dtype=torch.long)
    target_mem = rank_one_target_memory(q, target, num_entities=3)

    exact = field_alignment_diagnostics(target_mem, target_mem)
    scaled = field_alignment_diagnostics(3.0 * target_mem, target_mem)
    zero = field_alignment_diagnostics(torch.zeros_like(target_mem), target_mem)

    orthogonal = torch.zeros_like(target_mem)
    orthogonal[:, :, 2] = target_mem[:, :, 1]
    noise = field_alignment_diagnostics(orthogonal, target_mem)

    assert exact["canonical_cos"].item() == pytest.approx(1.0)
    assert scaled["canonical_cos"].item() == pytest.approx(1.0)
    assert scaled["field_rms"].item() == pytest.approx(3.0 * exact["field_rms"].item())
    assert zero["canonical_cos"].item() == pytest.approx(0.0)
    assert zero["field_rms"].item() == pytest.approx(0.0)
    assert abs(noise["canonical_cos"].item()) < 1e-6


def test_l1_isolation_smoke_run_writes_metrics(tmp_path) -> None:
    cfg = L1IsolationConfig(
        seed=123,
        num_entities=12,
        num_relations=3,
        max_path_len=4,
        key_dim=16,
        d_model=24,
        n_heads=4,
        train_steps=2,
        batch_size=2,
        eval_n=4,
        eval_batch_size=2,
        condition="field_supervised_mse",
        torch_threads=1,
    )

    result = run_experiment(cfg, tmp_path, device="cpu")

    assert result["paths"]["json"].endswith("CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.json")
    row = result["rows"][0]
    assert row["condition"] == "field_supervised_mse"
    assert "transformer_writer_acc" in row
    assert "field_mse" in row
    assert "canonical_cos" in row
    assert "field_rms" in row
    assert "wrong_key_old_target_rate" in row
    assert result["meta"]["train"]["snapshots"]
