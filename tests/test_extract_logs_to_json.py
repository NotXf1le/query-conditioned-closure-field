import json

from paper_suite.extract_logs_to_json import extract_runs, write_summary


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def make_run(root, name="run_a", *, with_log=True, log_text=""):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "experiment.json",
        {
            "args": {"seed": 123, "train_steps": 10},
            "family": "grounding_ablation",
            "id": name,
            "script": "grounding_ablation_suite.py",
            "theme": "unit",
        },
    )
    write_json(
        run_dir / "status.json",
        {"elapsed_sec": 1.25, "returncode": 0, "status": "done"},
    )
    if with_log:
        (run_dir / "run.log").write_text(log_text, encoding="utf-8")
    return run_dir


def test_extract_runs_collects_metadata_latest_train_eval_rows_and_final_event(tmp_path):
    make_run(
        tmp_path,
        "run_a",
        log_text="\n".join(
            [
                '{"train_progress": {"step": 1.0, "loss": 3.0}}',
                '{"grounding_ablation_train_progress": {"step": 2.0, "loss": 1.0}}',
                '{"learned_grounding_eval_by_length": {"length": 1.0, "n": 512.0, "learned_extractor_writer_acc": 1.0}}',
                '{"status": "done", "paths": {"json": "result.json"}, "elapsed_total_sec": 12.5}',
            ]
        )
        + "\n",
    )

    summary = extract_runs(tmp_path)

    assert summary["runs_root"] == str(tmp_path)
    assert summary["num_runs_seen"] == 1
    assert summary["num_runs_with_logs"] == 1
    assert summary["num_parse_warnings"] == 0
    run = summary["runs"][0]
    assert run["run_id"] == "run_a"
    assert run["family"] == "grounding_ablation"
    assert run["theme"] == "unit"
    assert run["script"] == "grounding_ablation_suite.py"
    assert run["args"] == {"seed": 123, "train_steps": 10}
    assert run["status"] == "done"
    assert run["returncode"] == 0
    assert run["elapsed_sec"] == 1.25
    assert run["last_train"] == {
        "event": "grounding_ablation_train_progress",
        "data": {"step": 2.0, "loss": 1.0},
    }
    assert run["eval_by_length"] == [
        {
            "event": "learned_grounding_eval_by_length",
            "data": {
                "length": 1.0,
                "learned_extractor_writer_acc": 1.0,
                "n": 512.0,
            },
        }
    ]
    assert run["final_event"] == {
        "elapsed_total_sec": 12.5,
        "paths": {"json": "result.json"},
        "status": "done",
    }
    assert run["parse_warnings"] == []


def test_extract_runs_records_non_json_lines_without_dropping_valid_events(tmp_path):
    make_run(
        tmp_path,
        "run_warning",
        log_text="\n".join(
            [
                "C:\\Python\\Lib\\site-packages\\torch\\nn\\modules\\transformer.py:382: UserWarning: warning text",
                '{"structured_closure_train_progress": {"step": 1.0, "loss": 0.5}}',
                '{"structured_closure_eval_by_length": {"length": 2.0, "neural_semiring_writer_acc": 1.0}}',
            ]
        )
        + "\n",
    )

    run = extract_runs(tmp_path)["runs"][0]

    assert run["last_train"]["event"] == "structured_closure_train_progress"
    assert run["eval_by_length"][0]["event"] == "structured_closure_eval_by_length"
    assert len(run["parse_warnings"]) == 1
    assert "line 1" in run["parse_warnings"][0]
    assert "non-json" in run["parse_warnings"][0]


def test_write_summary_converts_non_finite_values_to_strict_json_null(tmp_path):
    make_run(
        tmp_path,
        "run_nan",
        log_text='{"learned_grounding_eval_by_length": {"length": 1.0, "learned_reversed_order_acc": NaN, "inf_metric": Infinity}}\n',
    )
    out_dir = tmp_path / "out"

    out_path = write_summary(tmp_path, out_dir, "summary.json")

    text = out_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    data = json.loads(text)
    row = data["runs"][0]["eval_by_length"][0]["data"]
    assert row["learned_reversed_order_acc"] is None
    assert row["inf_metric"] is None


def test_missing_run_log_records_warning_and_does_not_crash(tmp_path):
    make_run(tmp_path, "missing_log", with_log=False)

    summary = extract_runs(tmp_path)

    assert summary["num_runs_seen"] == 1
    assert summary["num_runs_with_logs"] == 0
    assert summary["num_parse_warnings"] == 1
    run = summary["runs"][0]
    assert run["last_train"] is None
    assert run["eval_by_length"] == []
    assert run["final_event"] is None
    assert run["parse_warnings"] == ["missing run.log"]
