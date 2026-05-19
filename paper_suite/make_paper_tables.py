"""Create manuscript-ready tables from aggregated results.

The output intentionally emphasizes per-length results and control columns.  Any
seed aggregation is auxiliary and should be described as a reproducibility
summary, not the main scientific claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: List[Dict[str, object]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append("NA" if v != v else f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def first_done_by_family(rows: List[Dict[str, str]], family: str) -> List[Dict[str, str]]:
    # Choose the first completed run for compact example tables.
    candidates = [r for r in rows if r.get("family") == family and r.get("status") in {"done", "skipped_done", "unknown"}]
    if not candidates:
        return []
    run_id = sorted(set(r["run_id"] for r in candidates))[0]
    return [r for r in candidates if r["run_id"] == run_id]


def summarize_seeds(rows: List[Dict[str, str]], metric: str) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for r in rows:
        if r.get("status") not in {"done", "skipped_done", "unknown"}:
            continue
        L = r.get("length")
        v = to_float(r.get(metric, ""))
        if L is not None and v is not None:
            groups[(r.get("family", ""), r.get("theme", ""), L)].append(v)
    out = []
    for (fam, theme, L), vals in sorted(groups.items()):
        if len(vals) >= 2:
            out.append({"family": fam, "theme": theme, "length": int(float(L)), "metric": metric, "n_seeds": len(vals), "mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "min": min(vals), "max": max(vals)})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=str, default="paper_tables/combined_by_length_results.csv")
    p.add_argument("--out-dir", type=str, default="paper_tables")
    args = p.parse_args()
    rows = load_rows(Path(args.combined_csv)) if Path(args.combined_csv).exists() else []
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# Manuscript table draft", "", "Use these as drafts.  Do not report averaged headline accuracy as the primary claim; main evidence remains by length and control.", ""]

    generic = first_done_by_family(rows, "generic_writer")
    if generic:
        cols = ["length", "transformer_writer_acc", "mlp_writer_acc", "vanilla_transformer_acc", "oracle_full_closure_acc", "oracle_no_exact_query_acc", "oracle_prefix_only_acc"]
        lines += ["## Table A: generic Transformer closure writer negative result", "", markdown_table(generic, cols), ""]
    structured = first_done_by_family(rows, "structured_closure")
    if structured:
        cols = ["length", "neural_semiring_writer_acc", "dp_exact_query_writer_acc", "dp_prefix_field_writer_acc", "raw_onehop_acc"]
        lines += ["## Table B: structured semiring writer positive control", "", markdown_table(structured, cols), ""]
    learned = first_done_by_family(rows, "learned_grounding")
    if learned:
        cols = ["length", "learned_extractor_writer_acc", "dp_parser_semiring_writer_acc", "raw_onehop_acc", "extractor_fact_source_acc", "extractor_fact_relation_acc", "extractor_fact_target_acc", "extractor_query_source_acc", "extractor_query_relation_acc"]
        lines += ["## Table C: learned extractor + explicit closure", "", markdown_table(learned, cols), ""]
    split = first_done_by_family(rows, "heldout_order")
    if split:
        cols = ["length", "hard_n", "hard_learned_extractor_writer_acc", "hard_dp_parser_semiring_writer_acc", "hard_no_facts_acc", "hard_wrong_facts_acc", "hard_reversed_order_acc", "hard_shuffled_order_acc"]
        lines += ["## Table D: held-out relation-order split", "", markdown_table(split, cols), ""]
    anti = first_done_by_family(rows, "key_selectivity")
    if anti:
        cols = ["length", "learned_extractor_writer_acc", "text_scanned_slot_positions_acc", "counterfactual_relation_swap_counterfactual_acc", "counterfactual_fact_swap_counterfactual_acc", "counterfactual_relation_swap_both_correct_acc", "counterfactual_fact_swap_both_correct_acc", "zero_memory_old_target_acc", "swapped_memory_acc", "closure_random_source_key_old_target_acc"]
        lines += ["## Table E: anti-shortcut causal controls", "", markdown_table(anti, cols), ""]

    summaries = []
    for metric in ["transformer_writer_acc", "learned_extractor_writer_acc", "neural_semiring_writer_acc", "hard_learned_extractor_writer_acc", "counterfactual_relation_swap_counterfactual_acc", "counterfactual_fact_swap_counterfactual_acc"]:
        summaries.extend(summarize_seeds(rows, metric))
    (out / "seed_reproducibility_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    lines += ["## Seed reproducibility summary", "", "Written to `seed_reproducibility_summary.json` for diagnostics. Keep paper tables by length.", ""]
    md = out / "MANUSCRIPT_TABLE_DRAFTS.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"tables_md": str(md), "seed_summary_json": str(out / "seed_reproducibility_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
