from __future__ import annotations

import json

from paper_suite.aggregate_results import find_result_csv
from paper_suite.make_baseline_controls_manifest import build_manifest as build_baseline_controls_manifest
from paper_suite.make_core_diagnostics_manifest import build_manifest
from paper_suite.make_extended_diagnostics_manifest import build_manifest as build_extended_diagnostics_manifest
from paper_suite.make_l1_isolation_manifest import build_manifest as build_l1_isolation_manifest
from paper_suite.run_manifest import is_complete_run


def write_json(path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def test_hard_stress_appendix_2_uses_resource_safe_batch_sizes() -> None:
    stress_2 = [exp for exp in build_manifest() if exp["theme"] == "hard_stress_appendix_2"]

    assert len(stress_2) == 2
    for exp in stress_2:
        assert exp["args"]["num_entities"] == 408
        assert exp["args"]["max_seq_len"] == 5200
        assert exp["args"]["batch_size"] == 16
        assert exp["args"]["eval_batch_size"] == 16


def test_manifest_adds_only_key_selectivity_experiments_without_duplicates() -> None:
    manifest = build_manifest()
    ids = [str(exp["id"]) for exp in manifest]
    anti = [exp for exp in manifest if exp["script"] == "anti_shortcut_key_selectivity.py"]

    assert len(manifest) == 340
    assert len(ids) == len(set(ids))
    assert len(anti) == 128
    assert {str(exp["script"]) for exp in anti} == {"anti_shortcut_key_selectivity.py"}


def test_baseline_controls_manifest_has_unique_k10_conditions() -> None:
    manifest = build_baseline_controls_manifest()
    ids = [str(exp["id"]) for exp in manifest]
    closure = [exp for exp in manifest if exp["family"] == "closure_stress_baseline"]
    stronger = [exp for exp in manifest if exp["family"] == "nonclosure_baseline"]

    assert len(manifest) == 50
    assert len(ids) == len(set(ids))
    assert len(closure) == 30
    assert len(stronger) == 20
    assert {str(exp["script"]) for exp in stronger} == {"nonclosure_baseline_controls.py"}


def test_extended_diagnostics_manifest_has_unique_k10_conditions() -> None:
    manifest = build_extended_diagnostics_manifest()
    ids = [str(exp["id"]) for exp in manifest]
    ladder = [exp for exp in manifest if exp["family"] == "closure_writer_ladder"]
    repairs = [exp for exp in manifest if exp["family"] == "key_rejection_repair"]
    permutation = [exp for exp in manifest if exp["family"] == "grounding_permutation"]

    assert len(manifest) == 90
    assert len(ids) == len(set(ids))
    assert len(ladder) == 30
    assert len(repairs) == 40
    assert len(permutation) == 20
    assert {str(exp["script"]) for exp in ladder} == {"closure_writer_diagnostic_ladder.py"}
    assert {str(exp["script"]) for exp in repairs} == {"closure_key_rejection_repairs.py"}
    assert {str(exp["script"]) for exp in permutation} == {"grounding_permutation_diagnostics.py"}


def test_extended_diagnostics_smoke_manifest_has_one_per_condition() -> None:
    manifest = build_extended_diagnostics_manifest(smoke=True)
    ids = [str(exp["id"]) for exp in manifest]
    ladder = [exp for exp in manifest if exp["family"] == "closure_writer_ladder"]
    repairs = [exp for exp in manifest if exp["family"] == "key_rejection_repair"]
    permutation = [exp for exp in manifest if exp["family"] == "grounding_permutation"]

    assert len(manifest) == 9
    assert len(ids) == len(set(ids))
    assert sorted({str(exp["theme"]) for exp in ladder}) == [
        "baseline",
        "key_conditioned",
        "tied_key",
    ]
    assert sorted({str(exp["theme"]) for exp in repairs}) == [
        "contrastive",
        "linear",
        "margin_gate",
        "threshold_null",
    ]
    assert sorted({str(exp["theme"]) for exp in permutation}) == [
        "answer_only_no_slot",
        "extraction_weight_zero",
    ]


def test_resume_completion_requires_final_done_event_and_expected_result_json(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp = {"id": "run", "script": "grounding_ablation_suite.py"}
    write_json(run_dir / "status.json", {"status": "done", "returncode": 0})

    assert not is_complete_run(run_dir, exp)

    (run_dir / "run.log").write_text('{"status": "done", "paths": {"json": "LEARNED_GROUNDING_CLOSURE_RESULTS.json"}}\n', encoding="utf-8")

    assert not is_complete_run(run_dir, exp)

    write_json(run_dir / "LEARNED_GROUNDING_CLOSURE_RESULTS.json", {"rows": []})

    assert is_complete_run(run_dir, exp)


def test_resume_completion_accepts_anti_shortcut_result_json(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp = {"id": "run", "script": "anti_shortcut_key_selectivity.py"}
    write_json(run_dir / "status.json", {"status": "done", "returncode": 0})
    (run_dir / "run.log").write_text(
        '{"status": "done", "paths": {"json": "KEY_SELECTIVITY_RESULTS.json"}}\n',
        encoding="utf-8",
    )

    assert not is_complete_run(run_dir, exp)

    write_json(run_dir / "KEY_SELECTIVITY_RESULTS.json", {"rows": []})

    assert is_complete_run(run_dir, exp)


def test_resume_completion_accepts_extended_diagnostics_result_jsons(tmp_path) -> None:
    for script, expected in [
        ("closure_writer_diagnostic_ladder.py", "CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.json"),
        ("closure_key_rejection_repairs.py", "CLOSURE_KEY_REJECTION_REPAIRS_RESULTS.json"),
        ("grounding_permutation_diagnostics.py", "GROUNDING_PERMUTATION_DIAGNOSTICS_RESULTS.json"),
    ]:
        run_dir = tmp_path / script
        run_dir.mkdir()
        exp = {"id": script, "script": script}
        write_json(run_dir / "status.json", {"status": "done", "returncode": 0})
        (run_dir / "run.log").write_text(f'{{"status": "done", "paths": {{"json": "{expected}"}}}}\n', encoding="utf-8")

        assert not is_complete_run(run_dir, exp)

        write_json(run_dir / expected, {"rows": []})

        assert is_complete_run(run_dir, exp)


def test_resume_completion_accepts_l1_isolation_result_json(tmp_path) -> None:
    run_dir = tmp_path / "l1"
    run_dir.mkdir()
    exp = {"id": "l1", "script": "closure_l1_causal_isolation.py"}
    expected = "CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.json"
    write_json(run_dir / "status.json", {"status": "done", "returncode": 0})
    (run_dir / "run.log").write_text(f'{{"status": "done", "paths": {{"json": "{expected}"}}}}\n', encoding="utf-8")

    assert not is_complete_run(run_dir, exp)

    write_json(run_dir / expected, {"rows": []})

    assert is_complete_run(run_dir, exp)


def test_l1_isolation_manifest_imports_without_duplicates() -> None:
    manifest = build_l1_isolation_manifest(smoke=True)
    ids = [str(exp["id"]) for exp in manifest]

    assert ids
    assert len(ids) == len(set(ids))


def test_aggregator_finds_anti_shortcut_result_csv(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "KEY_SELECTIVITY_RESULTS.csv"
    csv_path.write_text("length,n\n1,4\n", encoding="utf-8")

    assert find_result_csv(run_dir) == csv_path


def test_resume_completion_accepts_multiline_final_done_event(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp = {"id": "run", "script": "grounding_ablation_suite.py"}
    write_json(run_dir / "status.json", {"status": "done", "returncode": 0})
    write_json(run_dir / "LEARNED_GROUNDING_CLOSURE_RESULTS.json", {"rows": []})
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                '{"grounding_ablation_train_progress": {"step": 1.0}}',
                "{",
                '  "status": "done",',
                '  "paths": {',
                '    "json": "LEARNED_GROUNDING_CLOSURE_RESULTS.json"',
                "  },",
                '  "elapsed_total_sec": 1.0',
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert is_complete_run(run_dir, exp)
