"""Extract compact JSON summaries from experiment run logs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


def sanitize_json_value(value: Any) -> Any:
    """Return a JSON-strict copy, converting NaN/Infinity floats to None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(v) for v in value]
    return value


def load_json_file(path: Path, default: JsonDict, warnings: List[str]) -> JsonDict:
    if not path.exists():
        warnings.append(f"missing {path.name}")
        return dict(default)
    try:
        return sanitize_json_value(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        warnings.append(f"malformed {path.name}: {exc}")
        return dict(default)


def _try_parse_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def iter_log_json_objects(log_path: Path, warnings: List[str]) -> Iterable[JsonDict]:
    """Yield JSON objects from a mostly-JSONL log.

    The training/eval records are line-delimited JSON, while the final status is
    sometimes pretty-printed as a multi-line JSON object.
    """
    collecting: List[str] = []
    start_line: Optional[int] = None
    last_error: Optional[str] = None

    for line_no, raw_line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if collecting:
            collecting.append(raw_line)
            obj, err = _try_parse_json("\n".join(collecting))
            if err is None:
                if isinstance(obj, dict):
                    yield sanitize_json_value(obj)
                else:
                    warnings.append(f"line {start_line}: json block is not an object")
                collecting = []
                start_line = None
                last_error = None
            else:
                last_error = err
            continue

        obj, err = _try_parse_json(line)
        if err is None:
            if isinstance(obj, dict):
                yield sanitize_json_value(obj)
            else:
                warnings.append(f"line {line_no}: json value is not an object")
            continue

        if line in {"{", "["}:
            collecting = [raw_line]
            start_line = line_no
            last_error = err
            continue

        if line.startswith("{") or line.startswith("["):
            warnings.append(f"line {line_no}: malformed json: {err}")
        else:
            preview = line[:160]
            warnings.append(f"line {line_no}: non-json: {preview}")

    if collecting:
        warnings.append(f"line {start_line}: malformed json block: {last_error}")


def parse_run_dir(run_dir: Path) -> JsonDict:
    warnings: List[str] = []
    default_meta = {
        "args": {},
        "family": "unknown",
        "id": run_dir.name,
        "script": "",
        "theme": "unknown",
    }
    meta = load_json_file(run_dir / "experiment.json", default_meta, warnings)
    status_meta = load_json_file(run_dir / "status.json", {"status": "unknown"}, warnings)

    last_train = None
    eval_by_length: List[JsonDict] = []
    final_event = None
    log_path = run_dir / "run.log"
    if not log_path.exists():
        warnings.append("missing run.log")
    else:
        for obj in iter_log_json_objects(log_path, warnings):
            if len(obj) == 1:
                event, data = next(iter(obj.items()))
                if str(event).endswith("train_progress"):
                    last_train = {"event": str(event), "data": data}
                    continue
                if str(event).endswith("eval_by_length"):
                    eval_by_length.append({"event": str(event), "data": data})
                    continue
            if "status" in obj:
                final_event = obj

    return sanitize_json_value(
        {
            "run_id": str(meta.get("id") or run_dir.name),
            "family": str(meta.get("family", "unknown")),
            "theme": str(meta.get("theme", "unknown")),
            "script": str(meta.get("script", "")),
            "args": meta.get("args", {}),
            "status": status_meta.get("status", "unknown"),
            "returncode": status_meta.get("returncode"),
            "elapsed_sec": status_meta.get("elapsed_sec"),
            "log_path": str(log_path),
            "last_train": last_train,
            "eval_by_length": eval_by_length,
            "final_event": final_event,
            "parse_warnings": warnings,
        }
    )


def extract_runs(runs_root: PathLike) -> JsonDict:
    root = Path(runs_root)
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    runs = [parse_run_dir(run_dir) for run_dir in run_dirs]
    return sanitize_json_value(
        {
            "runs_root": str(root),
            "num_runs_seen": len(runs),
            "num_runs_with_logs": sum(1 for run_dir in run_dirs if (run_dir / "run.log").exists()),
            "num_parse_warnings": sum(len(run["parse_warnings"]) for run in runs),
            "runs": runs,
        }
    )


def write_summary(runs_root: PathLike, out_dir: PathLike, out_name: str = "log_extract_summary.json") -> Path:
    summary = extract_runs(runs_root)
    out_path = Path(out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=str, default="runs_core_diagnostics")
    parser.add_argument("--out-dir", type=str, default="paper_tables_full")
    parser.add_argument("--out-name", type=str, default="log_extract_summary.json")
    args = parser.parse_args()

    out_path = write_summary(args.runs_root, args.out_dir, args.out_name)
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "out_json": str(out_path),
                "num_runs_seen": summary["num_runs_seen"],
                "num_runs_with_logs": summary["num_runs_with_logs"],
                "num_parse_warnings": summary["num_parse_warnings"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
