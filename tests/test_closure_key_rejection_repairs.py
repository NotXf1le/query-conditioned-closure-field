from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from closure_key_rejection_repairs import (
    RepairConfig,
    accept_mask,
    binary_score_metrics,
    key_dot,
    run_experiment,
    select_validation_gate,
    wrong_keys,
)
from closure_writer_diagnostic_ladder import rank_one_target_memory
from generic_closure_writer import HolographicClosureField


def test_repair_metrics_distinguish_accept_and_reject() -> None:
    cfg = RepairConfig(num_entities=12, num_relations=3, max_path_len=4, key_dim=32, reject_threshold=0.5)
    field = HolographicClosureField(cfg.num_entities, cfg.num_relations, cfg.key_dim, cfg.max_path_len, seed=11)
    source = torch.tensor([0, 1])
    rels = torch.zeros(2, cfg.max_path_len, dtype=torch.long)
    rels[:, 0] = torch.tensor([1, 2])
    lengths = torch.tensor([1, 1])
    target = torch.tensor([3, 4])
    q = field.key(source, rels, lengths)
    mem = rank_one_target_memory(q, target, cfg.num_entities)

    correct_logits = field.read(mem, q)
    correct_accept = accept_mask(cfg, "threshold_null", correct_logits, q, q, None)
    random_source_q = wrong_keys(field, source, rels, lengths)["random_source"]
    wrong_logits = field.read(mem, random_source_q)
    wrong_accept = accept_mask(cfg, "threshold_null", wrong_logits, random_source_q, q, None)

    assert correct_accept.tolist() == [True, True]
    assert wrong_accept.dtype == torch.bool
    assert key_dot(q, q).min().item() > key_dot(random_source_q, q).max().item()


def test_binary_score_metrics_reports_auroc_and_error_rates() -> None:
    metrics = binary_score_metrics(
        positives=torch.tensor([0.9, 0.8, 0.7]),
        negatives=torch.tensor([0.3, 0.2, 0.1]),
        threshold=0.5,
    )

    assert metrics["auroc"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_validation_gate_selection_uses_validation_scores_only() -> None:
    selected = select_validation_gate(
        positives=torch.tensor([0.9, 0.8, 0.7]),
        negatives=torch.tensor([0.3, 0.2, 0.1]),
        candidates=torch.tensor([0.15, 0.5, 0.95]),
    )

    assert selected["threshold"] == 0.5
    assert selected["fpr"] == 0.0
    assert selected["fnr"] == 0.0


def test_learned_grounding_repair_path_uses_learned_memory(tmp_path) -> None:
    cfg = RepairConfig(
        seed=321,
        num_entities=12,
        num_relations=3,
        max_path_len=4,
        key_dim=16,
        d_model=24,
        n_heads=4,
        n_layers=1,
        train_steps=2,
        batch_size=2,
        eval_n=4,
        eval_batch_size=2,
        repair_variant="threshold_null",
        pipeline="learned_grounding",
        calibration_split="validation",
        torch_threads=1,
        eval_lengths=(1,),
    )

    result = run_experiment(cfg, tmp_path, device="cpu")

    assert result["meta"]["pipeline_implementation"] == "learned_extractor_memory"
    row = result["rows"][0]
    assert row["pipeline_implementation"] == "learned_extractor_memory"
    assert row["selected_reject_threshold"] == result["meta"]["calibration"]["selected_reject_threshold"]
    assert "validation_auroc" in row
