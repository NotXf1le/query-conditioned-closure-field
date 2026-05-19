from __future__ import annotations

from paper_suite.make_l1_isolation_manifest import build_manifest


def test_l1_smoke_manifest_has_one_per_public_condition() -> None:
    manifest = build_manifest(smoke=True)
    ids = [str(exp["id"]) for exp in manifest]
    l1 = [exp for exp in manifest if exp["family"] == "l1_causal_isolation"]
    repair = [exp for exp in manifest if exp["family"] == "key_rejection_repair_pipeline"]

    assert len(ids) == len(set(ids))
    assert len(l1) >= 11
    assert len(repair) >= 4
    assert {str(exp["script"]) for exp in l1} == {"closure_l1_causal_isolation.py"}
    assert {str(exp["script"]) for exp in repair} == {"closure_key_rejection_repairs.py"}


def test_l1_full_manifest_uses_k10_public_conditions() -> None:
    manifest = build_manifest(smoke=False)
    ids = [str(exp["id"]) for exp in manifest]
    assert len(ids) == len(set(ids))

    by_theme = {}
    for exp in manifest:
        key = (exp["family"], exp["theme"])
        by_theme[key] = by_theme.get(key, 0) + 1

    assert by_theme[("l1_causal_isolation", "direct_qv_write")] == 10
    assert by_theme[("l1_causal_isolation", "field_supervised_mse")] == 10
    assert by_theme[("key_rejection_repair_pipeline", "threshold_null_full_pipeline")] == 10


def test_pipeline_repair_only_manifest_uses_corrected_themes() -> None:
    manifest = build_manifest(smoke=False, pipeline_repair_only=True)
    ids = [str(exp["id"]) for exp in manifest]
    by_theme = {}
    for exp in manifest:
        by_theme[(exp["family"], exp["theme"])] = by_theme.get((exp["family"], exp["theme"]), 0) + 1
        assert exp["script"] == "closure_key_rejection_repairs.py"
        assert exp["args"]["pipeline"] == "learned_grounding"

    assert len(ids) == len(set(ids))
    assert by_theme[("key_rejection_repair_pipeline", "threshold_null_learned_pipeline")] == 10
    assert by_theme[("key_rejection_repair_pipeline", "margin_gate_learned_pipeline")] == 10
