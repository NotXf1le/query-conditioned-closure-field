"""Aggregate CSV/JSON results from manifest runs.

This script keeps per-length rows.  It also computes seed summaries only for
run-selection diagnostics; do not use seed-averaged headline accuracy as the main
paper claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_experiment_meta(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "experiment.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"id": run_dir.name, "family": "unknown", "theme": "unknown", "args": {}}


def find_result_csv(run_dir: Path) -> Optional[Path]:
    patterns = [
        "GENERIC_CLOSURE_WRITER_RESULTS.csv",
        "STRUCTURED_TRANSITION_CLOSURE_RESULTS.csv",
        "LEARNED_GROUNDING_CLOSURE_RESULTS.csv",
        "HELDOUT_RELATION_ORDER_RESULTS.csv",
        "KEY_SELECTIVITY_RESULTS.csv",
    ]
    for pat in patterns:
        hits = list(run_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, default="runs_core_diagnostics")
    p.add_argument("--out-dir", type=str, default="paper_tables")
    args = p.parse_args()
    root = Path(args.runs_root)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for run_dir in sorted([x for x in root.iterdir() if x.is_dir()] if root.exists() else []):
        meta = load_experiment_meta(run_dir)
        csv_path = find_result_csv(run_dir)
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "unknown"}
        if not csv_path:
            rows.append({"run_id": run_dir.name, "status": str(status.get("status", "unknown")), "family": str(meta.get("family", "unknown")), "theme": str(meta.get("theme", "unknown")), "missing_results": "1"})
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rr = {str(k): str(v) for k, v in r.items()}
                rr.update({
                    "run_id": str(meta.get("id", run_dir.name)),
                    "family": str(meta.get("family", "unknown")),
                    "theme": str(meta.get("theme", "unknown")),
                    "status": str(status.get("status", "unknown")),
                    "script": str(meta.get("script", "")),
                    "args_json": json.dumps(meta.get("args", {}), sort_keys=True),
                    "result_csv": str(csv_path),
                })
                rows.append(rr)
    fieldnames = sorted({k for r in rows for k in r})
    combined_csv = out / "combined_by_length_results.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

    # Family/theme counts.
    counts: Dict[str, int] = {}
    for r in rows:
        key = r.get("family", "unknown") + "::" + r.get("theme", "unknown")
        counts[key] = counts.get(key, 0) + 1
    (out / "aggregation_summary.json").write_text(json.dumps({"num_rows": len(rows), "num_runs_seen": len(set(r.get("run_id", "") for r in rows)), "family_theme_row_counts": counts}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"combined_csv": str(combined_csv), "num_rows": len(rows), "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
