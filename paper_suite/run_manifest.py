"""Run experiments from a JSONL manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List


EXPECTED_RESULT_JSONS = {
    "generic_closure_writer.py": ["GENERIC_CLOSURE_WRITER_RESULTS.json"],
    "structured_transition_closure.py": ["STRUCTURED_TRANSITION_CLOSURE_RESULTS.json"],
    "learned_grounding_closure.py": ["LEARNED_GROUNDING_CLOSURE_RESULTS.json"],
    "grounding_ablation_suite.py": ["LEARNED_GROUNDING_CLOSURE_RESULTS.json"],
    "heldout_relation_order_split.py": ["HELDOUT_RELATION_ORDER_RESULTS.json"],
    "anti_shortcut_key_selectivity.py": ["KEY_SELECTIVITY_RESULTS.json"],
    "nonclosure_baseline_controls.py": ["STRONGER_BASELINES_RESULTS.json"],
    "closure_writer_diagnostic_ladder.py": ["CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.json"],
    "closure_key_rejection_repairs.py": ["CLOSURE_KEY_REJECTION_REPAIRS_RESULTS.json"],
    "grounding_permutation_diagnostics.py": ["GROUNDING_PERMUTATION_DIAGNOSTICS_RESULTS.json"],
    "closure_l1_causal_isolation.py": ["CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.json"],
}


def load_manifest(path: Path) -> List[Dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cli_name(k: str) -> str:
    return "--" + k.replace("_", "-")


def build_cmd(exp: Dict[str, object], out_root: Path, device: str) -> List[str]:
    script = str(exp["script"])
    out_dir = out_root / str(exp["id"])
    args = dict(exp.get("args", {}))
    # Some scripts intentionally expose fewer CLI knobs.  The manifest can keep
    # a uniform schema; the runner filters unsupported knobs.
    unsupported = {
        "generic_closure_writer.py": {"num_relations", "max_seq_len", "base_distractors", "distractors_per_hop", "same_relation_branch_prob", "relation_aliases"},
        "structured_transition_closure.py": {"eval_batch_size", "d_model", "max_seq_len", "base_distractors", "distractors_per_hop", "relation_aliases"},
    }.get(script, set())
    args = {k: v for k, v in args.items() if k not in unsupported}
    args["out_dir"] = str(out_dir)
    args["device"] = device
    cmd = [sys.executable, script]
    for k, v in args.items():
        flag = cli_name(k)
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def run_log_has_final_done_event(run_dir: Path) -> bool:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    last_status = None
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and "status" in obj:
            last_status = obj
        idx = start + max(1, int(end))
    return isinstance(last_status, dict) and last_status.get("status") == "done"


def is_complete_run(run_dir: Path, exp: Dict[str, object]) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if status.get("status") != "done":
        return False
    if not run_log_has_final_done_event(run_dir):
        return False
    script = str(exp.get("script", ""))
    expected = EXPECTED_RESULT_JSONS.get(script, [])
    if not expected:
        return False
    return any((run_dir / name).exists() for name in expected)


def run_one(exp: Dict[str, object], out_root: Path, device: str, resume: bool) -> Dict[str, object]:
    out_dir = out_root / str(exp["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    if resume and status_path.exists():
        if is_complete_run(out_dir, exp):
            return {"id": exp["id"], "status": "skipped_done", "elapsed_sec": 0.0}
    cmd = build_cmd(exp, out_root, device)
    (out_dir / "experiment.json").write_text(json.dumps(exp, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    status = {"id": exp["id"], "status": "done" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "elapsed_sec": elapsed, "log": str(log_path)}
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return status


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="configs/core_diagnostics_manifest.jsonl")
    p.add_argument("--out-root", type=str, default="runs_core_diagnostics")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--match", type=str, default="", help="substring filter over id/theme/family")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    experiments = load_manifest(Path(args.manifest))
    if args.match:
        m = args.match.lower()
        experiments = [e for e in experiments if m in (str(e.get("id", "")) + " " + str(e.get("theme", "")) + " " + str(e.get("family", ""))).lower()]
    if args.limit and args.limit > 0:
        experiments = experiments[: args.limit]
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    plan = [{"id": e["id"], "family": e["family"], "theme": e["theme"], "cmd": build_cmd(e, out_root, args.device)} for e in experiments]
    (out_root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"num_selected": len(experiments), "out_root": str(out_root), "dry_run": bool(args.dry_run)}, indent=2), flush=True)
    if args.dry_run:
        for item in plan[:20]:
            print(json.dumps(item, sort_keys=True))
        if len(plan) > 20:
            print(json.dumps({"omitted": len(plan) - 20}))
        return
    statuses = []
    if args.workers <= 1:
        for e in experiments:
            status = run_one(e, out_root, args.device, args.resume)
            statuses.append(status)
            print(json.dumps(status, sort_keys=True), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, e, out_root, args.device, args.resume) for e in experiments]
            for fut in as_completed(futs):
                status = fut.result(); statuses.append(status); print(json.dumps(status, sort_keys=True), flush=True)
    summary = {"num": len(statuses), "done": sum(s.get("status") == "done" for s in statuses), "failed": sum(s.get("status") == "failed" for s in statuses), "skipped_done": sum(s.get("status") == "skipped_done" for s in statuses)}
    (out_root / "run_summary.json").write_text(json.dumps({"summary": summary, "statuses": statuses}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
