"""Claim-validation checks for paper runs.

This is a guardrail script.  It does not prove the claim by itself, but it flags
runs that would be unsafe to cite because controls are too high, oracle controls
leak, or the expected negative/positive pattern is missing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def f(row: Dict[str, str], key: str):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=str, default="paper_tables/combined_by_length_results.csv")
    p.add_argument("--out", type=str, default="paper_tables/claim_validation.json")
    p.add_argument("--control-max", type=float, default=0.15)
    args = p.parse_args()
    path = Path(args.combined_csv)
    if not path.exists():
        raise SystemExit(f"missing {path}; run aggregate_results.py first")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    issues: List[Dict[str, object]] = []
    for r in rows:
        fam = r.get("family", "")
        L = int(float(r.get("length", "0"))) if r.get("length") else None
        rid = r.get("run_id", "")
        if fam == "generic_writer" and L and L > 3:
            # A generic transformer writer success here would be interesting, but
            # it contradicts the negative-result story and needs manual review.
            if f(r, "transformer_writer_acc") >= 0.5:
                issues.append({"run_id": rid, "length": L, "issue": "generic_writer_high_on_extrapolation", "value": f(r, "transformer_writer_acc")})
        if fam in {"learned_grounding", "grounding_ablation"}:
            for col in ["learned_query_only_acc", "learned_no_facts_acc", "learned_wrong_facts_acc", "learned_reversed_order_acc", "learned_shuffled_order_acc", "learned_no_exact_query_acc", "learned_prefix_only_acc"]:
                v = f(r, col)
                if v == v and v > args.control_max and not (col == "learned_first_order_acc" and L == 1):
                    issues.append({"run_id": rid, "length": L, "issue": f"control_high_{col}", "value": v})
        if fam == "generic_writer":
            for col in ["oracle_no_exact_query_acc", "oracle_prefix_only_acc"]:
                v = f(r, col)
                if v == v and v > args.control_max and L and L > 1:
                    issues.append({"run_id": rid, "length": L, "issue": f"oracle_control_leak_{col}", "value": v})
        if fam == "heldout_order":
            for col in ["hard_no_facts_acc", "hard_wrong_facts_acc", "hard_reversed_order_acc", "hard_shuffled_order_acc"]:
                v = f(r, col)
                if v == v and v > args.control_max:
                    issues.append({"run_id": rid, "length": L, "issue": f"split_control_high_{col}", "value": v})
        if fam == "key_selectivity":
            # These should collapse to chance/low old-target rate.
            # Counterfactual success metrics are intentionally not controls.
            for col in ["zero_memory_old_target_acc", "swapped_memory_acc", "swapped_query_key_acc", "closure_reversed_key_old_target_acc", "closure_shuffled_key_old_target_acc", "closure_prefix_key_old_target_acc", "closure_random_source_key_old_target_acc", "gold_edge_deletion_old_target_rate_acc"]:
                v = f(r, col)
                if v == v and v > args.control_max:
                    issues.append({"run_id": rid, "length": L, "issue": f"anti_shortcut_control_high_{col}", "value": v})
            for col in ["counterfactual_relation_swap_counterfactual_acc", "counterfactual_fact_swap_counterfactual_acc", "text_scanned_slot_positions_acc"]:
                v = f(r, col)
                if v == v and v < 0.75:
                    issues.append({"run_id": rid, "length": L, "issue": f"anti_shortcut_expected_success_low_{col}", "value": v})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"num_rows": len(rows), "num_issues": len(issues), "issues": issues}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "num_rows": len(rows), "num_issues": len(issues)}, indent=2))


if __name__ == "__main__":
    main()
