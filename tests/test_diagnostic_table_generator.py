from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_table_generator():
    path = Path(__file__).resolve().parents[1] / "tools" / "make_diagnostic_tables.py"
    spec = importlib.util.spec_from_file_location("make_diagnostic_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table_generator_handles_missing_run_root(tmp_path) -> None:
    mod = load_table_generator()

    rows, summary = mod.load_completed_rows(tmp_path / "missing")

    assert rows == []
    assert summary["completed_result_runs"] == 0
    assert summary["error"] == "runs root not found"


def test_table_generator_loads_nonclosure_baseline_csv(tmp_path) -> None:
    mod = load_table_generator()
    run_dir = tmp_path / "runs" / "stronger"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    (run_dir / "experiment.json").write_text(
        json.dumps({
            "id": "stronger",
            "family": "nonclosure_baseline",
            "theme": "relbias_scratchpad_staged",
            "args": {"seed": 1},
        }),
        encoding="utf-8",
    )
    (run_dir / "STRONGER_BASELINES_RESULTS.csv").write_text(
        "length,n,relative_transformer_acc,relative_transformer_n\n1,4,1.0,4\n",
        encoding="utf-8",
    )

    rows, summary = mod.load_completed_rows(tmp_path / "runs")

    assert summary["completed_result_runs"] == 1
    assert len(rows) == 1
    assert rows[0]["_family"] == "nonclosure_baseline"


def test_table_generator_loads_extended_diagnostics_csv(tmp_path) -> None:
    mod = load_table_generator()
    run_dir = tmp_path / "runs" / "ladder"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    (run_dir / "experiment.json").write_text(
        json.dumps({
            "id": "ladder",
            "family": "closure_writer_ladder",
            "theme": "answer_only_baseline",
            "args": {"seed": 1},
        }),
        encoding="utf-8",
    )
    (run_dir / "CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.csv").write_text(
        "rung,length,n,writer_variant,transformer_writer_acc,transformer_writer_n,mlp_writer_acc,mlp_writer_n,direct_endpoint_acc,direct_endpoint_n,oracle_full_closure_acc,oracle_full_closure_n\n"
        "one_hop_fact_write,1,4,baseline,0.25,4,0.25,4,0.50,4,1.0,4\n",
        encoding="utf-8",
    )

    rows, summary = mod.load_completed_rows(tmp_path / "runs")

    assert summary["completed_result_runs"] == 1
    assert len(rows) == 1
    assert rows[0]["_family"] == "closure_writer_ladder"
    assert mod.diagnostic_ladder_rows(rows)


def test_table_generator_does_not_emit_extended_tables_without_extended_rows(tmp_path) -> None:
    mod = load_table_generator()
    out_dir = tmp_path / "generated"

    mod.write_outputs(out_dir, [], {}, [], {}, [], {}, [], {})

    assert not (out_dir / "table_diagnostic_ladder.tex").exists()
    assert not (out_dir / "table_key_alignment_baselines.tex").exists()
    assert not (out_dir / "table_wrong_key_repairs.tex").exists()
    assert not (out_dir / "table_permutation_grounding.tex").exists()
    assert not (out_dir / "table_l1_isolation.tex").exists()


def test_full_baseline_completeness_requires_k10_per_condition() -> None:
    mod = load_table_generator()
    rows = []
    for theme in ["long_budget_staged", "wide_deep_staged", "wide_deep_mixed_cosine"]:
        for seed in range(10):
            rows.append({"_family": "closure_stress_baseline", "_theme": theme, "_run_id": f"{theme}_{seed}"})
    stronger_rows = []
    for theme in ["relbias_scratchpad_staged", "relbias_scratchpad_mixed_cosine"]:
        for seed in range(10):
            stronger_rows.append({"_family": "nonclosure_baseline", "_theme": theme, "_run_id": f"{theme}_{seed}"})

    assert mod.has_full_closure_stress(rows)
    assert not mod.has_full_closure_stress(rows[:-1])
    assert mod.has_full_stronger_baselines(stronger_rows)
    assert not mod.has_full_stronger_baselines(stronger_rows[:-1])


def test_table_generator_loads_l1_rows(tmp_path) -> None:
    mod = load_table_generator()
    run_dir = tmp_path / "runs" / "l1"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    (run_dir / "experiment.json").write_text(
        json.dumps({
            "id": "l1",
            "family": "l1_causal_isolation",
            "theme": "direct_qv_write",
            "args": {"seed": 1},
        }),
        encoding="utf-8",
    )
    (run_dir / "CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.csv").write_text(
        "condition,length,n,transformer_writer_acc,transformer_writer_n,direct_endpoint_acc,direct_endpoint_n,field_mse,canonical_cos,field_rms,wrong_key_old_target_rate,wrong_key_old_target_n,grad_norm\n"
        "direct_qv_write,1,4,1.0,4,1.0,4,99.0,0.75,0.25,0.5,4,1.0\n",
        encoding="utf-8",
    )

    rows, summary = mod.load_completed_rows(tmp_path / "runs")

    assert summary["completed_result_runs"] == 1
    assert rows[0]["_family"] == "l1_causal_isolation"
    table_rows = mod.l1_isolation_rows(rows)
    assert table_rows
    assert "$0.7500\\pm0.0000$" in table_rows[0]
    assert "$0.2500\\pm0.0000$" in table_rows[0]
    assert "$99.0000\\pm0.0000$" not in table_rows[0]


def test_training_curve_summary_includes_100k_budget() -> None:
    mod = load_table_generator()
    rows = []
    for steps in [3000, 10000, 30000, 100000]:
        rows.append({
            "_family": "l1_training_budget",
            "_theme": f"budget_{steps}",
            "_run_id": f"budget_{steps}_seed_1",
            "train_steps": str(steps),
            "transformer_writer_acc": "0.5",
            "transformer_writer_n": "4",
            "direct_endpoint_acc": "0.5",
            "direct_endpoint_n": "4",
            "grad_norm": "1.0",
        })

    table = mod.training_curve_summary_rows(rows)

    assert [row[0] for row in table] == ["3000", "10000", "30000", "100000"]


def test_field_ablation_summary_uses_scale_aware_field_diagnostics() -> None:
    mod = load_table_generator()
    rows = [{
        "_family": "l1_field_ablation",
        "_theme": "keydim_96",
        "_run_id": "run_1",
        "transformer_writer_acc": "0.5",
        "transformer_writer_n": "4",
        "field_mse": "99.0",
        "canonical_cos": "0.75",
        "field_rms": "0.25",
        "wrong_key_old_target_rate": "0.1",
        "wrong_key_old_target_n": "4",
    }]

    table_rows = mod.field_ablation_summary_rows(rows)

    assert "$0.7500\\pm0.0000$" in table_rows[0]
    assert "$0.2500\\pm0.0000$" in table_rows[0]
    assert "$99.0000\\pm0.0000$" not in table_rows[0]


def test_pipeline_repair_table_ignores_stale_full_pipeline_rows() -> None:
    mod = load_table_generator()
    rows = [
        {
            "_family": "key_rejection_repair_pipeline",
            "_theme": "threshold_null_full_pipeline",
            "_run_id": "stale",
            "correct_key_answer_rate": "1.0",
            "correct_key_answer_n": "4",
            "wrong_key_old_target_rate": "0.0",
            "wrong_key_old_target_n": "4",
            "wrong_key_reject_rate": "1.0",
            "wrong_key_reject_n": "4",
            "correct_key_reject_rate": "0.0",
            "correct_key_reject_n": "4",
        },
        {
            "_family": "key_rejection_repair_pipeline",
            "_theme": "threshold_null_learned_pipeline",
            "_run_id": "learned",
            "pipeline_implementation": "learned_extractor_memory",
            "correct_key_answer_rate": "1.0",
            "correct_key_answer_n": "4",
            "wrong_key_old_target_rate": "0.25",
            "wrong_key_old_target_n": "4",
            "wrong_key_reject_rate": "0.75",
            "wrong_key_reject_n": "4",
            "correct_key_reject_rate": "0.0",
            "correct_key_reject_n": "4",
            "key_gate_auroc": "1.0",
            "key_gate_fpr": "0.0",
            "key_gate_fnr": "0.0",
        },
    ]

    table = mod.wrong_key_repair_v2_rows(rows)

    assert len(table) == 1
    assert "learned" in table[0][0]
