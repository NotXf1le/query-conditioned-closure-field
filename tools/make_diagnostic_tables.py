"""Generate compact, traceable tables for the diagnostic manuscript.

Quantitative rows are included only when a run directory has:
  - status.json with status == "done";
  - experiment.json;
  - an existing result CSV.

The public tables intentionally avoid local version labels. Detailed provenance
stays in the completed run artifacts; this script writes a compact public
evidence summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT_CANDIDATES = [
    REPO_ROOT / "runs_core_diagnostics",
]
DEFAULT_RUNS_ROOT = next((p for p in RUNS_ROOT_CANDIDATES if p.exists()), RUNS_ROOT_CANDIDATES[0])
DEFAULT_BASELINE_CONTROLS_ROOT = REPO_ROOT / "runs_baseline_controls"
DEFAULT_EXTENDED_DIAGNOSTICS_ROOT = REPO_ROOT / "runs_extended_diagnostics"
DEFAULT_L1_ISOLATION_ROOT = REPO_ROOT / "runs_l1_isolation"
DEFAULT_PIPELINE_REPAIR_ROOT = REPO_ROOT / "runs_pipeline_repair"

RESULT_PATTERNS = [
    "GENERIC_CLOSURE_WRITER_RESULTS.csv",
    "STRUCTURED_TRANSITION_CLOSURE_RESULTS.csv",
    "LEARNED_GROUNDING_CLOSURE_RESULTS.csv",
    "HELDOUT_RELATION_ORDER_RESULTS.csv",
    "KEY_SELECTIVITY_RESULTS.csv",
    "STRONGER_BASELINES_RESULTS.csv",
    "CLOSURE_WRITER_DIAGNOSTIC_LADDER_RESULTS.csv",
    "CLOSURE_KEY_REJECTION_REPAIRS_RESULTS.csv",
    "GROUNDING_PERMUTATION_DIAGNOSTICS_RESULTS.csv",
    "CLOSURE_L1_CAUSAL_ISOLATION_RESULTS.csv",
]


def find_result_csv(run_dir: Path) -> Optional[Path]:
    for pattern in RESULT_PATTERNS:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def canonical_family(family: object) -> str:
    return str(family)


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) else out


def load_completed_rows(runs_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    completed_runs: List[Dict[str, object]] = []
    excluded_runs: List[Dict[str, object]] = []
    missing_result_runs: List[Dict[str, object]] = []

    if not runs_root.exists():
        return rows, {
            "runs_root": repo_path(runs_root),
            "completed_result_runs": 0,
            "excluded_runs": [],
            "missing_result_runs": [],
            "error": "runs root not found",
        }

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        status_path = run_dir / "status.json"
        meta_path = run_dir / "experiment.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
            "id": run_dir.name,
            "family": "unknown",
            "theme": "unknown",
            "args": {},
        }
        csv_path = find_result_csv(run_dir)
        run_info = {
            "id": meta.get("id", run_dir.name),
            "family": canonical_family(meta.get("family", "unknown")),
            "theme": meta.get("theme", "unknown"),
            "status": status.get("status", "unknown"),
            "returncode": status.get("returncode"),
            "result_csv": repo_path(csv_path) if csv_path else None,
        }
        if status.get("status") != "done":
            excluded_runs.append(run_info)
            continue
        if csv_path is None:
            missing_result_runs.append(run_info)
            continue
        completed_runs.append(run_info)
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rr: Dict[str, object] = dict(row)
                rr["_run_id"] = meta.get("id", run_dir.name)
                rr["_family"] = canonical_family(meta.get("family", "unknown"))
                rr["_source_family"] = meta.get("family", "unknown")
                rr["_theme"] = meta.get("theme", "unknown")
                rr["_seed"] = meta.get("args", {}).get("seed")
                rr["_args"] = meta.get("args", {})
                rr["_result_csv"] = repo_path(csv_path)
                rows.append(rr)

    summary = {
        "runs_root": repo_path(runs_root),
        "completed_result_runs": len(completed_runs),
        "completed_by_family": dict(Counter(str(r["family"]) for r in completed_runs)),
        "completed_by_family_theme": {
            f"{family}::{theme}": count
            for (family, theme), count in sorted(Counter((str(r["family"]), str(r["theme"])) for r in completed_runs).items())
        },
        "excluded_runs": excluded_runs,
        "missing_result_runs": missing_result_runs,
    }
    return rows, summary


def n_key_for(metric: str) -> str:
    return metric[:-4] + "_n" if metric.endswith("_acc") else f"{metric}_n"


def metric_weighted(rows: Sequence[Dict[str, object]], metric: str, n_metric: Optional[str] = None) -> Tuple[Optional[float], int]:
    n_key = n_metric or n_key_for(metric)
    numerator = 0.0
    denom = 0.0
    for row in rows:
        value = to_float(row.get(metric))
        n = to_float(row.get(n_key))
        if n is None:
            n = to_float(row.get("hard_n"))
        if n is None:
            n = to_float(row.get("n"))
        if value is None or n is None:
            continue
        numerator += value * n
        denom += n
    if denom <= 0:
        return None, 0
    return numerator / denom, int(round(denom))


def family_rows(rows: Sequence[Dict[str, object]], family: str, theme: Optional[str] = None) -> List[Dict[str, object]]:
    out = [r for r in rows if r.get("_family") == family]
    if theme is not None:
        out = [r for r in out if r.get("_theme") == theme]
    return out


def rows_by_length(rows: Sequence[Dict[str, object]], length: int) -> List[Dict[str, object]]:
    return [r for r in rows if to_float(r.get("length")) == float(length)]


def unique_seeds(rows: Sequence[Dict[str, object]]) -> int:
    return len({r.get("_seed") for r in rows if r.get("_seed") is not None})


def unique_runs(rows: Sequence[Dict[str, object]]) -> int:
    return len({r.get("_run_id") for r in rows if r.get("_run_id") is not None})


def fmt_acc(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.3f}"


def fmt_acc4(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def latex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latex_table(
    *,
    label: str,
    caption: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    align: str,
    note: Optional[str] = None,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\vspace{{2pt}}\par\footnotesize {note}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(str(c) for c in columns) + " |"]
    out.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def metric_pair(rows: Sequence[Dict[str, object]], metric: str, length: Optional[int] = None) -> Tuple[str, int]:
    group = rows_by_length(rows, length) if length is not None else list(rows)
    value, n = metric_weighted(group, metric)
    return fmt_acc(value), n


def metric_values_by_run(rows: Sequence[Dict[str, object]], metric: str, n_metric: Optional[str] = None) -> List[float]:
    values: List[float] = []
    run_ids = sorted({str(r.get("_run_id")) for r in rows if r.get("_run_id") is not None})
    for run_id in run_ids:
        run_rows = [r for r in rows if str(r.get("_run_id")) == run_id]
        value, _ = metric_weighted(run_rows, metric, n_metric=n_metric)
        if value is not None:
            values.append(value)
    return values


def metric_values_by_run_unweighted(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    values: List[float] = []
    run_ids = sorted({str(r.get("_run_id")) for r in rows if r.get("_run_id") is not None})
    for run_id in run_ids:
        vals = [to_float(r.get(metric)) for r in rows if str(r.get("_run_id")) == run_id]
        vals = [v for v in vals if v is not None]
        if vals:
            values.append(sum(vals) / len(vals))
    return values


def mean_sd_ci(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None, None
    vals = [float(v) for v in values]
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        sd = 0.0
        ci = 0.0
    else:
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        ci = 1.96 * sd / math.sqrt(len(vals))
    return mean, sd, ci, min(vals), max(vals)


def fmt_mean_ci(values: Sequence[float]) -> str:
    mean, _, ci, _, _ = mean_sd_ci(values)
    if mean is None or ci is None:
        return "NA"
    return rf"${mean:.4f}\pm{ci:.4f}$"


def seed_variance_row(label: str, rows: Sequence[Dict[str, object]], metric: str, n_metric: Optional[str] = None) -> List[str]:
    values = metric_values_by_run(rows, metric, n_metric=n_metric)
    mean, sd, ci, vmin, vmax = mean_sd_ci(values)
    return [
        latex_escape(label),
        fmt_int(len(values)),
        fmt_acc4(mean),
        fmt_acc4(sd),
        fmt_acc4(ci),
        f"{fmt_acc4(vmin)}--{fmt_acc4(vmax)}",
    ]


def diagnostic_pair(rows: Sequence[Dict[str, object]], metric_a: str, metric_b: str) -> str:
    a, _ = metric_pair(rows, metric_a)
    b, _ = metric_pair(rows, metric_b)
    return f"{a}/{b}"


def closure_mechanism_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    generic_rows = family_rows(rows, "generic_writer")
    structured_rows = family_rows(rows, "structured_closure")
    learned_rows = family_rows(rows, "learned_grounding")

    specs = [
        ("Raw one-hop", generic_rows, "raw_onehop_acc", "direct lookup baseline", "L=1 1.000"),
        ("Transformer writer", generic_rows, "transformer_writer_acc", "generic learned writer", "chance-scale"),
        ("MLP writer", generic_rows, "mlp_writer_acc", "generic learned writer", "chance-scale"),
        ("Vanilla answer head", generic_rows, "vanilla_transformer_acc", "non-memory neural baseline", "weak short-path"),
        ("Oracle exact read", generic_rows, "oracle_full_closure_acc", "read upper bound", "no-exact/prefix " + diagnostic_pair(generic_rows, "oracle_no_exact_query_acc", "oracle_prefix_only_acc")),
        ("Structured transition", structured_rows, "neural_semiring_writer_acc", "explicit closure writer", "wrong facts " + fmt_acc(metric_weighted(structured_rows, "neural_wrong_facts_acc")[0])),
        ("Learned grounding + closure", learned_rows, "learned_extractor_writer_acc", "controlled slot grounding", "no/wrong " + diagnostic_pair(learned_rows, "learned_no_facts_acc", "learned_wrong_facts_acc")),
    ]

    out: List[List[str]] = []
    for name, group, metric, mechanism, control in specs:
        all_acc, n = metric_pair(group, metric)
        l32_acc, _ = metric_pair(group, metric, length=32)
        out.append([
            latex_escape(name),
            fmt_int(unique_runs(group)),
            fmt_int(n),
            all_acc,
            l32_acc,
            latex_escape(mechanism),
            latex_escape(control or "see text"),
        ])
    return out


def extraction_weight_theme_rows(rows: Sequence[Dict[str, object]], label: str, theme: str) -> Optional[List[str]]:
    group = family_rows(rows, "grounding_ablation", theme)
    if not group:
        return None
    answer, n = metric_pair(group, "learned_extractor_writer_acc")
    fact, _ = metric_pair(group, "extractor_fact_relation_acc")
    query, _ = metric_pair(group, "extractor_query_relation_acc")
    no_wrong = diagnostic_pair(group, "learned_no_facts_acc", "learned_wrong_facts_acc")
    return [latex_escape(label), fmt_int(unique_runs(group)), fmt_int(n), answer, f"{fact}/{query}", no_wrong]


def split_row(rows: Sequence[Dict[str, object]]) -> Optional[List[str]]:
    split = family_rows(rows, "heldout_order")
    if not split:
        return None
    answer, n = metric_pair(split, "hard_learned_extractor_writer_acc")
    no_wrong = diagnostic_pair(split, "hard_no_facts_acc", "hard_wrong_facts_acc")
    rev_shuf = diagnostic_pair(split, "hard_reversed_order_acc", "hard_shuffled_order_acc")
    control = f"no/wrong {no_wrong}; rev/shuf {rev_shuf}"
    return [latex_escape("Held-out relation orders"), fmt_int(unique_runs(split)), fmt_int(n), answer, "NA", latex_escape(control)]


def grounding_and_split_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []

    learned_rows = family_rows(rows, "learned_grounding")
    answer, n = metric_pair(learned_rows, "learned_extractor_writer_acc")
    fact, _ = metric_pair(learned_rows, "extractor_fact_relation_acc")
    query, _ = metric_pair(learned_rows, "extractor_query_relation_acc")
    out.append([
        latex_escape("Supervised controlled grounding"),
        fmt_int(unique_runs(learned_rows)),
        fmt_int(n),
        answer,
        f"{fact}/{query}",
        diagnostic_pair(learned_rows, "learned_no_facts_acc", "learned_wrong_facts_acc"),
    ])

    for label, theme in [
        ("Answer-only, no slots", "answer_only_no_slot_supervision"),
        ("Extraction weight 0", "extraction_weight_sweep_0.0"),
    ]:
        row = extraction_weight_theme_rows(rows, label, theme)
        if row:
            out.append(row)

    row = split_row(rows)
    if row:
        out.append(row)
    return out


def training_protocol_rows() -> List[List[str]]:
    return [
        [
            latex_escape("Constrained closure writer"),
            r"Transformer encoder, $d=96$, 2 layers, 4 heads, FF $=4d$; 1.14M closure-writer params; sinusoidal absolute positions; no relative-position bias.",
            r"Answer cross-entropy only. The writer emits a dense closure memory $M$ read once by $q^\top M$.",
            r"AdamW, LR $10^{-3}$ constant, weight decay $10^{-4}$, grad clip 1.0.",
            r"Staged train lengths $1\rightarrow\{1,2\}\rightarrow\{1,2,3\}$; evaluated through $L=32$.",
        ],
        [
            latex_escape("Direct endpoint baseline"),
            r"Same 2-layer Transformer encoder and positional encoding as the constrained writer; 0.235M params.",
            latex_escape("Answer cross-entropy only; predicts endpoint logits directly, without the associative-memory bottleneck."),
            r"AdamW, LR $10^{-3}$ constant, same batches and curriculum as the constrained writer.",
            latex_escape("Same train/eval length protocol as the constrained writer."),
        ],
        [
            latex_escape("Structured transition writer"),
            latex_escape("Controlled transition parser plus explicit semiring/dynamic-programming closure."),
            latex_escape("Grounded one-hop relation structure; answer evaluated after one associative read."),
            latex_escape("AdamW on the small learned relation matcher; explicit transition composition does the closure step."),
            latex_escape("Same extrapolation target, evaluated through L=32."),
        ],
        [
            latex_escape("Learned grounding + explicit closure"),
            r"Slot extractor, $d=96$, key dimension 128; explicit closure over extracted relation slots.",
            latex_escape("Answer CE plus intermediate slot supervision for fact and query relations."),
            r"AdamW, LR $3\times10^{-3}$, weight decay $10^{-5}$, grad clip 1.0.",
            latex_escape("Alias/noise/fact-order robustness curricula; evaluated through L=32 and anti-shortcut probes."),
        ],
        [
            latex_escape("Architecture/training stress"),
            r"Transformer stress tests: $d=96$, 2L/4H and $d=192$, 4L/8H; closure-writer params 1.14M or 4.23M.",
            latex_escape("Same constrained closure writer, MLP writer (0.95M or 2.60M params), direct endpoint baseline (0.235M or 1.80M), and oracle controls."),
            latex_escape("AdamW, 3000 steps, batch 128; constant LR or 5% warmup cosine decay to 0.1x LR."),
            r"Staged or mixed $\{1,2,3\}$ training lengths; evaluated through $L=32$.",
        ],
    ]


def task_protocol_rows() -> List[List[str]]:
    return [
        [
            latex_escape("Candidate space"),
            latex_escape("48 entities and 4 relation labels in the main controlled task."),
            r"Uniform endpoint chance is $1/48=0.0208$.",
        ],
        [
            latex_escape("Train/eval lengths"),
            r"Training queries use lengths $1\rightarrow\{1,2\}\rightarrow\{1,2,3\}$ unless a stress condition uses mixed $\{1,2,3\}$ from step 1.",
            r"Evaluation reports $L\in\{1,2,3,4,6,8,12,16,24,32\}$.",
        ],
        [
            latex_escape("Distractors"),
            r"Each length-$L$ example contains the $L$ gold edges plus $6+3L$ off-path distractors, wrong-relation branches, and occasional same-relation decoys.",
            latex_escape("Same-relation decoys are retained only when the final ordered query endpoint remains unique."),
        ],
        [
            latex_escape("Leakage controls"),
            latex_escape("Train and evaluation generators use disjoint deterministic RNG streams by replicate and length."),
            latex_escape("Held-out split families are adjacent-pair, trigram, and repeated-relation holdouts."),
        ],
    ]


def seed_variance_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    return [
        seed_variance_row(
            "Constrained Transformer closure writer",
            family_rows(rows, "generic_writer"),
            "transformer_writer_acc",
        ),
        seed_variance_row(
            "Direct endpoint Transformer baseline",
            family_rows(rows, "generic_writer"),
            "vanilla_transformer_acc",
        ),
        seed_variance_row(
            "Structured transition closure",
            family_rows(rows, "structured_closure"),
            "neural_semiring_writer_acc",
        ),
        seed_variance_row(
            "Learned grounding + explicit closure",
            family_rows(rows, "learned_grounding"),
            "learned_extractor_writer_acc",
        ),
        seed_variance_row(
            "Anti-shortcut suite answer accuracy",
            family_rows(rows, "key_selectivity"),
            "learned_extractor_writer_acc",
        ),
        seed_variance_row(
            "Wrong-key old-target leakage",
            family_rows(rows, "key_selectivity"),
            "swapped_query_key_acc",
        ),
    ]


def stress_condition_label(theme: str) -> str:
    labels = {
        "long_budget_staged": "Long-budget staged",
        "wide_deep_staged": "Wide/deep staged",
        "wide_deep_mixed_cosine": "Wide/deep mixed+cosine",
    }
    return labels.get(theme, theme.replace("_", " "))


def stress_protocol_text(group: Sequence[Dict[str, object]]) -> str:
    if not group:
        return "NA"
    args = group[0].get("_args", {})
    if not isinstance(args, dict):
        args = {}
    d_model = args.get("d_model", "NA")
    layers = args.get("layers", "NA")
    heads = args.get("heads", "NA")
    key_dim = args.get("key_dim", "NA")
    steps = args.get("train_steps", "NA")
    batch_size = args.get("batch_size", "NA")
    curriculum = args.get("curriculum", "NA")
    schedule = args.get("lr_schedule", "NA")
    return latex_escape(f"d={d_model}, {layers}L/{heads}H, key={key_dim}; {steps} steps, batch {batch_size}; {curriculum}; {schedule}")


def baseline_stress_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    ordered_themes = ["long_budget_staged", "wide_deep_staged", "wide_deep_mixed_cosine"]
    for theme in ordered_themes:
        group = family_rows(rows, "closure_stress_baseline", theme)
        if not group:
            continue
        out.append([
            latex_escape(stress_condition_label(theme)),
            fmt_int(unique_runs(group)),
            stress_protocol_text(group),
            fmt_mean_ci(metric_values_by_run(group, "transformer_writer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "mlp_writer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "vanilla_transformer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "oracle_full_closure_acc")),
        ])
    if not out:
        out.append([latex_escape("No completed stress artifacts"), "0", "NA", "NA", "NA", "NA", "NA"])
    return out


def has_full_closure_stress(rows: Sequence[Dict[str, object]]) -> bool:
    required = ["long_budget_staged", "wide_deep_staged", "wide_deep_mixed_cosine"]
    return all(unique_runs(family_rows(rows, "closure_stress_baseline", theme)) >= 10 for theme in required)


def has_full_stronger_baselines(rows: Sequence[Dict[str, object]]) -> bool:
    required = ["relbias_scratchpad_staged", "relbias_scratchpad_mixed_cosine"]
    return all(unique_runs(family_rows(rows, "nonclosure_baseline", theme)) >= 10 for theme in required)


def stronger_baseline_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    ordered_themes = ["relbias_scratchpad_staged", "relbias_scratchpad_mixed_cosine"]
    labels = {
        "relbias_scratchpad_staged": "Rel-bias + scratchpad, staged",
        "relbias_scratchpad_mixed_cosine": "Rel-bias + scratchpad, mixed+cosine",
    }
    for theme in ordered_themes:
        group = family_rows(rows, "nonclosure_baseline", theme)
        if not group:
            continue
        out.append([
            latex_escape(labels.get(theme, theme)),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "relative_transformer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "scratchpad_answer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "scratchpad_hop_acc")),
            fmt_mean_ci(metric_values_by_run(group, "graph_recurrent_acc")),
            fmt_mean_ci(metric_values_by_run(group, "iterative_pointer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "dp_bfs_oracle_acc")),
        ])
    if not out:
        out.append([latex_escape("Pending external full-baseline run"), "0", "NA", "NA", "NA", "NA", "NA", "NA"])
    return out


def rows_with(rows: Sequence[Dict[str, object]], **matches: object) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        ok = True
        for key, value in matches.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def diagnostic_ladder_label(rung: str) -> str:
    labels = {
        "direct_qv_write": r"Direct $q,v\rightarrow M$",
        "gold_target_write": "Gold-target write",
        "one_hop_fact_write": r"One-hop fact write ($L=1$)",
        "multi_hop_closure_write": r"Multi-hop closure write ($L\geq2$)",
    }
    return labels.get(rung, rung.replace("_", " "))


def diagnostic_ladder_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    ladder = rows_with(family_rows(rows, "closure_writer_ladder"), writer_variant="baseline")
    out: List[List[str]] = []
    for rung in ["direct_qv_write", "gold_target_write", "one_hop_fact_write", "multi_hop_closure_write"]:
        group = rows_with(ladder, rung=rung)
        if not group:
            continue
        out.append([
            diagnostic_ladder_label(rung),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "transformer_writer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "mlp_writer_acc")),
            fmt_mean_ci(metric_values_by_run(group, "direct_endpoint_acc")),
            fmt_mean_ci(metric_values_by_run(group, "oracle_full_closure_acc")),
        ])
    return out


def key_alignment_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    ladder = family_rows(rows, "closure_writer_ladder")
    labels = {
        "baseline": "Answer-only baseline",
        "key_conditioned": "Key-conditioned writer",
        "tied_key": "Tied key embeddings",
    }
    out: List[List[str]] = []
    for variant in ["baseline", "key_conditioned", "tied_key"]:
        group = rows_with(ladder, writer_variant=variant)
        if not group:
            continue
        out.append([
            latex_escape(labels.get(variant, variant)),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(rows_with(group, rung="direct_qv_write"), "transformer_writer_acc")),
            fmt_mean_ci(metric_values_by_run(rows_with(group, rung="gold_target_write"), "transformer_writer_acc")),
            fmt_mean_ci(metric_values_by_run(rows_with(group, rung="one_hop_fact_write"), "transformer_writer_acc")),
            fmt_mean_ci(metric_values_by_run(rows_with(group, rung="multi_hop_closure_write"), "transformer_writer_acc")),
        ])
    return out


def wrong_key_repair_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    repairs = family_rows(rows, "key_rejection_repair")
    labels = {
        "linear": "Rank-one linear read",
        "contrastive": "+ contrastive key gate",
        "threshold_null": "+ threshold/null reject",
        "margin_gate": "+ margin gate",
    }
    out: List[List[str]] = []
    for variant in ["linear", "contrastive", "threshold_null", "margin_gate"]:
        group = rows_with(repairs, repair_variant=variant)
        if not group:
            continue
        out.append([
            latex_escape(labels.get(variant, variant)),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "correct_key_answer_rate", n_metric="correct_key_answer_n")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_old_target_rate", n_metric="wrong_key_old_target_n")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_reject_rate", n_metric="wrong_key_reject_n")),
            fmt_mean_ci(metric_values_by_run(group, "correct_key_reject_rate", n_metric="correct_key_reject_n")),
        ])
    return out


def l1_condition_label(condition: str) -> str:
    labels = {
        "direct_qv_write": r"Direct $q,v\rightarrow M$",
        "gold_target_text_write": "Text + gold target",
        "id_only_write": "ID-only write",
        "one_fact_no_distractor": "One gold fact, no distractors",
        "l1_no_distractor": r"$L=1$, no distractors",
        "l1_with_distractors": r"$L=1$, full distractors",
        "teacher_forced_q": "Teacher-forced key",
        "teacher_forced_v": "Teacher-forced value",
        "teacher_forced_qv": "Teacher-forced key+value",
        "field_supervised_mse": "Field-supervised MSE",
        "field_supervised_read_ce": "Field-supervised multi-read CE",
    }
    return labels.get(condition, condition.replace("_", " "))


def l1_isolation_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    ordered = [
        "direct_qv_write",
        "gold_target_text_write",
        "id_only_write",
        "one_fact_no_distractor",
        "l1_no_distractor",
        "l1_with_distractors",
        "teacher_forced_q",
        "teacher_forced_v",
        "teacher_forced_qv",
        "field_supervised_mse",
        "field_supervised_read_ce",
    ]
    for condition in ordered:
        group = rows_with(family_rows(rows, "l1_causal_isolation"), condition=condition)
        if not group:
            group = family_rows(rows, "l1_causal_isolation", condition)
        if not group:
            continue
        out.append([
            l1_condition_label(condition),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "transformer_writer_acc", n_metric="transformer_writer_n")),
            fmt_mean_ci(metric_values_by_run(group, "direct_endpoint_acc", n_metric="direct_endpoint_n")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "field_mse")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_old_target_rate", n_metric="wrong_key_old_target_n")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "grad_norm")),
        ])
    return out


def field_ablation_summary_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        family = str(row.get("_family", ""))
        theme = str(row.get("_theme", ""))
        if family in {"l1_field_ablation", "l1_training_budget"}:
            groups.setdefault(f"{family}:{theme}", []).append(row)
    for key in sorted(groups):
        group = groups[key]
        values = metric_values_by_run(group, "transformer_writer_acc", n_metric="transformer_writer_n")
        median = "NA"
        if values:
            ordered = sorted(values)
            mid = len(ordered) // 2
            med = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
            median = fmt_acc4(med)
        out.append([
            latex_escape(key.replace("l1_field_ablation:", "").replace("l1_training_budget:", "")),
            fmt_int(unique_runs(group)),
            fmt_acc4(max(values) if values else None),
            median,
            fmt_mean_ci(metric_values_by_run(group, "transformer_writer_acc", n_metric="transformer_writer_n")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "field_mse")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_old_target_rate", n_metric="wrong_key_old_target_n")),
        ])
    return out


def training_curve_summary_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    for theme in ["budget_3000", "budget_10000", "budget_30000"]:
        group = family_rows(rows, "l1_training_budget", theme)
        if not group:
            continue
        steps = sorted({int(float(str(r.get("train_steps", "0")))) for r in group if to_float(r.get("train_steps")) is not None})
        out.append([
            latex_escape(theme.replace("budget_", "")),
            fmt_int(unique_runs(group)),
            ",".join(str(s) for s in steps) if steps else "NA",
            fmt_mean_ci(metric_values_by_run(group, "transformer_writer_acc", n_metric="transformer_writer_n")),
            fmt_mean_ci(metric_values_by_run(group, "direct_endpoint_acc", n_metric="direct_endpoint_n")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "grad_norm")),
        ])
    return out


def wrong_key_repair_v2_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    out: List[List[str]] = []
    labels = {
        "linear_exact": "Linear exact",
        "contrastive_exact": "Contrastive exact",
        "threshold_null_exact": "Threshold/null exact",
        "margin_gate_exact": "Margin exact",
        "threshold_null_learned_pipeline": "Threshold/null learned",
        "margin_gate_learned_pipeline": "Margin learned",
    }
    for theme in [
        "linear_exact",
        "contrastive_exact",
        "threshold_null_exact",
        "margin_gate_exact",
        "threshold_null_learned_pipeline",
        "margin_gate_learned_pipeline",
    ]:
        group = family_rows(rows, "key_rejection_repair_pipeline", theme)
        if theme.endswith("_learned_pipeline"):
            group = [r for r in group if str(r.get("pipeline_implementation")) == "learned_extractor_memory"]
        if not group:
            continue
        out.append([
            latex_escape(labels.get(theme, theme)),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "correct_key_answer_rate", n_metric="correct_key_answer_n")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_old_target_rate", n_metric="wrong_key_old_target_n")),
            fmt_mean_ci(metric_values_by_run(group, "wrong_key_reject_rate", n_metric="wrong_key_reject_n")),
            fmt_mean_ci(metric_values_by_run(group, "correct_key_reject_rate", n_metric="correct_key_reject_n")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "key_gate_auroc")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "key_gate_fpr")),
            fmt_mean_ci(metric_values_by_run_unweighted(group, "key_gate_fnr")),
        ])
    return out


def permutation_grounding_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    grounding = family_rows(rows, "grounding_permutation")
    labels = {
        "answer_only_no_slot": "Answer-only/no-slot",
        "extraction_weight_zero": "Extraction weight zero",
        "supervised": "Slot supervised",
    }
    out: List[List[str]] = []
    themes = ["answer_only_no_slot", "extraction_weight_zero", "supervised"]
    for theme in themes:
        group = family_rows(rows, "grounding_permutation", theme)
        if not group:
            group = rows_with(grounding, condition=theme)
        if not group:
            continue
        fact_can = fmt_mean_ci(metric_values_by_run(group, "fact_relation_canonical_acc"))
        fact_perm = fmt_mean_ci(metric_values_by_run(group, "fact_relation_permutation_acc"))
        query_can = fmt_mean_ci(metric_values_by_run(group, "query_relation_canonical_acc"))
        query_perm = fmt_mean_ci(metric_values_by_run(group, "query_relation_permutation_acc"))
        fact_mi = fmt_mean_ci(metric_values_by_run_unweighted(group, "fact_relation_mutual_info"))
        query_mi = fmt_mean_ci(metric_values_by_run_unweighted(group, "query_relation_mutual_info"))
        out.append([
            latex_escape(labels.get(theme, theme)),
            fmt_int(unique_runs(group)),
            fmt_mean_ci(metric_values_by_run(group, "answer_acc")),
            rf"\shortstack{{{fact_can}\\{fact_perm}}}",
            rf"\shortstack{{{query_can}\\{query_perm}}}",
            rf"\shortstack{{{fact_mi}\\{query_mi}}}",
        ])
    return out


def random_sign_codes(count: int, dim: int, rng) -> List[List[int]]:
    return [[-1 if rng.random() < 0.5 else 1 for _ in range(dim)] for _ in range(count)]


def synthetic_key(source: int, relations: Sequence[int], *, key_dim: int, entity_code: Sequence[Sequence[int]], length_code: Sequence[Sequence[int]], relpos_code: Sequence[Sequence[Sequence[int]]]) -> List[float]:
    length = len(relations)
    scale = math.sqrt(float(key_dim))
    out = [entity_code[source][i] * length_code[length][i] for i in range(key_dim)]
    for pos, rel in enumerate(relations):
        code = relpos_code[pos][int(rel)]
        out = [out[i] * code[i] for i in range(key_dim)]
    return [x / scale for x in out]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def key_geometry_rows() -> List[List[str]]:
    rng = __import__("random").Random(20260514)
    rows: List[List[str]] = []
    for dim in [16, 32, 64, 96, 128, 256]:
        local_rng = __import__("random").Random(20260514 + dim)
        entity = random_sign_codes(48, dim, local_rng)
        length = random_sign_codes(33, dim, local_rng)
        relpos = [[[-1 if local_rng.random() < 0.5 else 1 for _ in range(dim)] for _ in range(4)] for _ in range(32)]
        wrong_dots: List[float] = []
        max_wrong: List[float] = []
        for _ in range(400):
            L = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
            source = rng.randrange(48)
            rels = [rng.randrange(4) for _ in range(L)]
            q = synthetic_key(source, rels, key_dim=dim, entity_code=entity, length_code=length, relpos_code=relpos)
            variants: List[List[float]] = []
            wrong_source = (source + 1 + rng.randrange(47)) % 48
            variants.append(synthetic_key(wrong_source, rels, key_dim=dim, entity_code=entity, length_code=length, relpos_code=relpos))
            if L > 1:
                variants.append(synthetic_key(source, list(reversed(rels)), key_dim=dim, entity_code=entity, length_code=length, relpos_code=relpos))
            swapped = list(rels)
            idx = rng.randrange(L)
            swapped[idx] = (swapped[idx] + 1 + rng.randrange(3)) % 4
            variants.append(synthetic_key(source, swapped, key_dim=dim, entity_code=entity, length_code=length, relpos_code=relpos))
            candidate_dots = [dot(q, v) for v in variants]
            wrong_dots.extend(candidate_dots)
            random_source_dots = [
                dot(q, synthetic_key((source + j) % 48, rels, key_dim=dim, entity_code=entity, length_code=length, relpos_code=relpos))
                for j in range(1, 17)
            ]
            max_wrong.append(max(random_source_dots))
        mean = sum(wrong_dots) / len(wrong_dots)
        sd = math.sqrt(sum((x - mean) ** 2 for x in wrong_dots) / max(1, len(wrong_dots) - 1))
        abs_sorted = sorted(abs(x) for x in wrong_dots)
        p95_abs = abs_sorted[int(0.95 * (len(abs_sorted) - 1))]
        pos_rate = sum(1 for x in wrong_dots if x > 0.0) / len(wrong_dots)
        mean_max = sum(max_wrong) / len(max_wrong)
        rows.append([str(dim), fmt_acc4(mean), fmt_acc4(sd), fmt_acc4(p95_abs), fmt_acc4(pos_rate), fmt_acc4(mean_max)])
    return rows


def key_selectivity_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    anti = family_rows(rows, "key_selectivity")
    specs = [
        ("Original answer", "learned_extractor_writer_acc", "task accuracy"),
        ("Relation swap", "counterfactual_relation_swap_counterfactual_acc", "counterfactual target"),
        ("Fact swap", "counterfactual_fact_swap_counterfactual_acc", "counterfactual target"),
        ("Same multiset order", "same_multiset_order_counterfactual_acc", "order-sensitive target"),
        ("Zero memory old target", "zero_memory_old_target_acc", "memory causality"),
        ("Swapped memory old target", "swapped_memory_acc", "example specificity"),
        ("Swapped query key old target", "swapped_query_key_acc", "wrong-key leakage"),
        ("Reversed key old target", "closure_reversed_key_old_target_acc", "wrong-key leakage"),
        ("Prefix key old target", "closure_prefix_key_old_target_acc", "wrong-key leakage"),
        ("Random-source key old target", "closure_random_source_key_old_target_acc", "wrong-key leakage"),
        ("Shuffled key old target", "closure_shuffled_key_old_target_acc", "wrong-key leakage"),
    ]
    out: List[List[str]] = []
    for probe, metric, meaning in specs:
        value, n = metric_pair(anti, metric)
        out.append([latex_escape(probe), latex_escape(meaning), value, fmt_int(n)])
    return out


def by_length_rows(
    rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    *,
    n_metric: Optional[str] = None,
    display_n_metric: Optional[str] = None,
) -> List[List[str]]:
    out: List[List[str]] = []
    lengths = sorted({
        int(float(r.get("length")))
        for r in rows
        if to_float(r.get("length")) is not None
    })
    for length in lengths:
        group = rows_by_length(rows, length)
        values: List[str] = []
        for metric in metrics:
            value, n = metric_weighted(group, metric, n_metric=n_metric)
            values.append(fmt_acc(value))
        n_source = display_n_metric or n_metric or n_key_for(metrics[0])
        _, n = metric_weighted(group, metrics[0], n_metric=n_source)
        if n <= 0:
            continue
        out.append([str(length), fmt_int(n)] + values)
    return out


def generic_writer_by_length_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    return by_length_rows(
        family_rows(rows, "generic_writer"),
        [
            "raw_onehop_acc",
            "transformer_writer_acc",
            "mlp_writer_acc",
            "vanilla_transformer_acc",
            "oracle_full_closure_acc",
            "oracle_no_exact_query_acc",
            "oracle_prefix_only_acc",
        ],
    )


def structured_closure_by_length_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    return by_length_rows(
        family_rows(rows, "structured_closure"),
        [
            "raw_onehop_acc",
            "neural_semiring_writer_acc",
            "neural_no_facts_acc",
            "neural_wrong_facts_acc",
            "neural_reversed_order_acc",
            "neural_shuffled_order_acc",
        ],
    )


def learned_grounding_by_length_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    return by_length_rows(
        family_rows(rows, "learned_grounding"),
        [
            "raw_onehop_acc",
            "learned_extractor_writer_acc",
            "extractor_fact_relation_acc",
            "extractor_query_relation_acc",
            "learned_no_facts_acc",
            "learned_wrong_facts_acc",
            "learned_reversed_order_acc",
            "learned_shuffled_order_acc",
        ],
        display_n_metric="learned_extractor_writer_n",
    )


def heldout_order_by_length_rows(rows: Sequence[Dict[str, object]]) -> List[List[str]]:
    return by_length_rows(
        family_rows(rows, "heldout_order"),
        [
            "hard_learned_extractor_writer_acc",
            "hard_extractor_fact_relation_acc",
            "hard_extractor_query_relation_acc",
            "hard_no_facts_acc",
            "hard_wrong_facts_acc",
            "hard_reversed_order_acc",
            "hard_shuffled_order_acc",
        ],
        n_metric="hard_n",
        display_n_metric="hard_n",
    )


def write_outputs(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    full_rows: Sequence[Dict[str, object]],
    full_summary: Dict[str, object],
    extended_rows: Sequence[Dict[str, object]],
    extended_summary: Dict[str, object],
    l1_rows: Sequence[Dict[str, object]] = (),
    l1_summary: Optional[Dict[str, object]] = None,
    pipeline_repair_rows: Sequence[Dict[str, object]] = (),
    pipeline_repair_summary: Optional[Dict[str, object]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    l1_summary = l1_summary or {}
    pipeline_repair_summary = pipeline_repair_summary or {}
    repair_source_rows = list(l1_rows) + list(pipeline_repair_rows)
    stress_rows = full_rows
    complete_stronger = has_full_stronger_baselines(full_rows)
    table_specs = [
        (
            "table_task_protocol",
            ["Protocol item", "Definition", "Interpretation"],
            task_protocol_rows(),
            "tab:task-protocol",
            "Controlled task distribution and chance scale for the main path-query diagnostic.",
            r"p{0.16\textwidth}p{0.46\textwidth}p{0.28\textwidth}",
            None,
        ),
        (
            "table_training_protocol",
            ["Condition", "Architecture", "Supervision/objective", "Optimization", "Length protocol"],
            training_protocol_rows(),
            "tab:training-protocol",
            "Experimental protocol details for the main diagnostic conditions, including architecture scale. The direct endpoint baseline is not forced to write a closure memory.",
            r"p{0.13\textwidth}p{0.22\textwidth}p{0.22\textwidth}p{0.17\textwidth}p{0.16\textwidth}",
            None,
        ),
        (
            "table_seed_variance",
            ["Claim metric", "K", "Mean", "Std.", "95\\% CI", "Min--max"],
            seed_variance_rows(rows),
            "tab:seed-variance",
            "Run-level variance for the main claims. K counts independent completed runs; confidence intervals are computed over run-level means, not over individual examples.",
            "lrrrrr",
            None,
        ),
        (
            "table_generic_writer_by_length",
            ["L", "N", "Raw", "Transf.", "MLP", "Vanilla", "Oracle", "No-exact", "Prefix"],
            generic_writer_by_length_rows(rows),
            "tab:generic-writer-by-length",
            "Generic closure-writing negative result by path length. Oracle exact writes the queried key and is used only as a read upper bound.",
            "rrrrrrrrr",
            None,
        ),
        (
            "table_structured_closure_by_length",
            ["L", "N", "Raw", "Structured", "No facts", "Wrong facts", "Reversed", "Shuffled"],
            structured_closure_by_length_rows(rows),
            "tab:structured-closure-by-length",
            "Structured transition/semiring closure by path length. The structured writer composes one-hop transitions before a single associative read.",
            "rrrrrrrr",
            None,
        ),
        (
            "table_learned_grounding_by_length",
            ["L", "N", "Raw", "Learned+closure", "Fact rel", "Query rel", "No facts", "Wrong facts", "Reversed", "Shuffled"],
            learned_grounding_by_length_rows(rows),
            "tab:learned-grounding-by-length",
            "Controlled learned grounding plus explicit closure by path length. Relation columns are canonical slot accuracies.",
            "rrrrrrrrrr",
            None,
        ),
        (
            "table_heldout_order_by_length",
            ["L", "N", "Learned+closure", "Fact rel", "Query rel", "No facts", "Wrong facts", "Reversed", "Shuffled"],
            heldout_order_by_length_rows(rows),
            "tab:heldout-order-by-length",
            "Held-out relation-order split results by path length, using hard split examples only.",
            "rrrrrrrrr",
            None,
        ),
        (
            "table_closure_mechanisms",
            ["Mechanism", "Runs", "N", "All-L", "L=32", "Role", "Control"],
            closure_mechanism_rows(rows),
            "tab:closure-mechanisms",
            "Closure construction and reading diagnostics. Oracle exact is a read upper bound; no-exact/prefix controls test whether exact closure information was already written.",
            "lrrrrll",
            None,
        ),
        (
            "table_grounding_and_splits",
            ["Setting", "Runs", "N", "Answer", "Fact/query rel", "Controls"],
            grounding_and_split_rows(rows),
            "tab:grounding-splits",
            "Controlled grounding, identifiability, and held-out relation-order results. Relation columns are canonical slot accuracies where defined.",
            "lrrrll",
            None,
        ),
        (
            "table_key_selectivity",
            ["Probe", "Meaning", "Rate", "N"],
            key_selectivity_rows(rows),
            "tab:key-selectivity",
            "Anti-shortcut and key-selectivity probes. Low zero/swapped-memory rates show causal memory use; high wrong-key old-target rates expose insufficient key selectivity.",
            "llrr",
            None,
        ),
        (
            "table_baseline_stress",
            ["Stress condition", "K", "Protocol", "Closure writer", "MLP writer", "Direct endpoint", "Oracle"],
            baseline_stress_rows(stress_rows),
            "tab:baseline-stress",
            "Architecture/training stress baselines. Values are mean $\\pm$ 95\\% CI across independent runs; the direct endpoint model predicts the answer without the single associative read.",
            r"lp{0.06\textwidth}p{0.25\textwidth}rrrr",
            None,
        ),
        (
            "table_key_geometry",
            ["Key dim.", "Mean dot", "Std.", r"$95\%\,|\mathrm{dot}|$", r"$\Pr(\mathrm{dot}>0)$", "Mean max sampled wrong-source dot"],
            key_geometry_rows(),
            "tab:key-geometry",
            "Deterministic random-sign key-geometry diagnostic. Positive wrong-key dots are expected under finite-dimensional rank-one keys; without cleanup or rejection, they can preserve an old target under small background logits.",
            "rrrrrr",
            None,
        ),
    ]
    if complete_stronger:
        table_specs.append((
            "table_stronger_baselines",
            ["Baseline", "K", "Rel-bias", "Scratch ans.", "Hop acc.", "Graph", "Pointer", "DP/BFS"],
            stronger_baseline_rows(full_rows),
            "tab:stronger-baselines",
            "Non-closure baselines. Values are mean $\\pm$ 95\\% CI across independent runs; graph/pointer/DP rows use iterative structured state rather than a single closure-memory read.",
            r"p{0.24\textwidth}rcccccc",
            None,
        ))
    ladder_rows = diagnostic_ladder_rows(extended_rows)
    if ladder_rows:
        table_specs.append((
            "table_diagnostic_ladder",
            ["Ladder rung", "K", "Transformer writer", "MLP writer", "Direct endpoint", "Oracle"],
            ladder_rows,
            "tab:diagnostic-ladder",
            "Diagnostic ladder for separating field writing, one-hop grounding, and multi-hop closure construction. Values are mean $\\pm$ 95\\% CI across completed runs.",
            r"p{0.22\textwidth}rcccc",
            None,
        ))
    alignment_rows = key_alignment_rows(extended_rows)
    if alignment_rows:
        table_specs.append((
            "table_key_alignment_baselines",
            ["Writer variant", "K", r"Direct $q,v$", "Gold target", r"$L=1$ fact", r"$L\geq2$ closure"],
            alignment_rows,
            "tab:key-alignment-baselines",
            "Key-alignment baselines for the Transformer field writer. These tests distinguish random-key field-writing difficulty from path-composition difficulty.",
            r"p{0.24\textwidth}rcccc",
            None,
        ))
    repair_rows = wrong_key_repair_rows(extended_rows)
    if repair_rows:
        table_specs.append((
            "table_wrong_key_repairs",
            ["Read variant", "K", "Correct-key answer", "Wrong-key old target", "Wrong-key reject", "Correct-key reject"],
            repair_rows,
            "tab:wrong-key-repairs",
            "Wrong-key rejection repairs for exact rank-one writes. A useful repair lowers old-target leakage without destroying correct-key accuracy.",
            r"p{0.24\textwidth}rcccc",
            None,
        ))
    perm_rows = permutation_grounding_rows(extended_rows)
    if perm_rows:
        table_specs.append((
            "table_permutation_grounding",
            ["Condition", "K", "Answer", r"\shortstack{Fact rel.\\canon/perm}", r"\shortstack{Query rel.\\canon/perm}", r"\shortstack{Fact/query\\MI}"],
            perm_rows,
            "tab:permutation-grounding",
            "Permutation-aware relation diagnostics for answer-only grounding conditions. Aligned accuracy separates non-canonical systematic codes from unidentified shortcuts.",
            r"p{0.22\textwidth}rcccc",
            None,
        ))
    l1_table_rows = l1_isolation_rows(l1_rows)
    if l1_table_rows:
        table_specs.append((
            "table_l1_isolation",
            ["Condition", "K", "Writer", "Direct", "Field MSE", "Wrong-key old", "Grad norm"],
            l1_table_rows,
            "tab:l1-isolation",
            "L=1 causal-isolation campaign. Rows separate pure field writing, text/value mapping, distractor selection, teacher forcing, and field-supervised objectives; values are mean $\\pm$ 95\\% CI over completed runs.",
            r"p{0.28\textwidth}rccccc",
            None,
        ))
    l1_ablation_rows = field_ablation_summary_rows(l1_rows)
    if l1_ablation_rows:
        table_specs.append((
            "table_field_ablation_summary",
            ["Ablation", "K", "Best", "Median", "Mean", "Field MSE", "Wrong-key old"],
            l1_ablation_rows,
            "tab:field-ablation-summary",
            "Key/field ablations for the L=1 bottleneck campaign. Best and median summarize run-level writer accuracy within each public condition.",
            r"p{0.24\textwidth}rrrrrr",
            None,
        ))
    l1_curve_rows = training_curve_summary_rows(l1_rows)
    if l1_curve_rows:
        table_specs.append((
            "table_training_curves_summary",
            ["Budget", "K", "Steps", "Writer", "Direct", "Grad norm"],
            l1_curve_rows,
            "tab:training-curves-summary",
            "Training-budget summary for L=1 writer runs. Full learning curves are retained in run artifacts; this table reports completed-run endpoints.",
            r"lrrccc",
            None,
        ))
    repair_rows = wrong_key_repair_v2_rows(repair_source_rows)
    if repair_rows:
        table_specs.append((
            "table_wrong_key_repair_full_pipeline",
            ["Read variant", "K", "Correct", "Old target", "Wrong reject", "False reject", "AUROC", "FPR", "FNR"],
            repair_rows,
            "tab:wrong-key-repair-full-pipeline",
            "Wrong-key repair campaign. Learned-pipeline rows are included only when they come from learned extractor memory artifacts, not metadata-only pipeline labels.",
            r"p{0.14\textwidth}rccccccc",
            None,
        ))

    expected_files = {f"{filename}.tex" for filename, *_ in table_specs}
    for old in out_dir.glob("table_*.tex"):
        if old.name not in expected_files:
            old.unlink()

    md_lines = [
        "# Diagnostic Tables",
        "",
        "All public table values come from completed valid result artifacts.",
        "",
    ]
    for filename, columns, table_rows, label, caption, align, note in table_specs:
        tex = latex_table(
            label=label,
            caption=caption,
            columns=columns,
            rows=table_rows,
            align=align,
            note=note,
        )
        (out_dir / f"{filename}.tex").write_text(tex, encoding="utf-8")
        md_lines += [f"## {filename}", "", markdown_table(columns, table_rows), ""]

    summary = dict(summary)
    summary["public_tables"] = sorted(expected_files)
    summary["key_selectivity_table_included"] = bool(family_rows(rows, "key_selectivity"))
    summary["baseline_controls_completed_result_runs"] = full_summary.get("completed_result_runs", 0)
    summary["full_closure_stress_complete"] = has_full_closure_stress(full_rows)
    summary["full_stronger_baselines_complete"] = complete_stronger
    summary["extended_diagnostics_completed_result_runs"] = extended_summary.get("completed_result_runs", 0)
    summary["extended_diagnostics_tables_included"] = sorted(
        name for name in expected_files
        if name in {
            "table_diagnostic_ladder.tex",
            "table_key_alignment_baselines.tex",
            "table_wrong_key_repairs.tex",
            "table_permutation_grounding.tex",
        }
    )
    summary["l1_isolation_completed_result_runs"] = l1_summary.get("completed_result_runs", 0)
    summary["pipeline_repair_completed_result_runs"] = pipeline_repair_summary.get("completed_result_runs", 0)
    summary["l1_and_repair_tables_included"] = sorted(
        name for name in expected_files
        if name in {
            "table_l1_isolation.tex",
            "table_field_ablation_summary.tex",
            "table_training_curves_summary.tex",
            "table_wrong_key_repair_full_pipeline.tex",
        }
    )
    (out_dir / "evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "diagnostic_tables.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--baseline-controls-runs-root", "--full-baseline-runs-root", dest="baseline_controls_runs_root", type=Path, default=DEFAULT_BASELINE_CONTROLS_ROOT)
    parser.add_argument("--extended-runs-root", dest="extended_runs_root", type=Path, default=DEFAULT_EXTENDED_DIAGNOSTICS_ROOT)
    parser.add_argument("--l1-runs-root", dest="l1_runs_root", type=Path, default=DEFAULT_L1_ISOLATION_ROOT)
    parser.add_argument("--pipeline-repair-runs-root", dest="pipeline_repair_runs_root", type=Path, default=DEFAULT_PIPELINE_REPAIR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper" / "generated")
    args = parser.parse_args()

    rows, summary = load_completed_rows(args.runs_root)
    full_rows, full_summary = load_completed_rows(args.baseline_controls_runs_root)
    extended_rows, extended_summary = load_completed_rows(args.extended_runs_root)
    l1_rows, l1_summary = load_completed_rows(args.l1_runs_root)
    pipeline_repair_rows, pipeline_repair_summary = load_completed_rows(args.pipeline_repair_runs_root)
    write_outputs(
        args.out_dir,
        rows,
        summary,
        full_rows,
        full_summary,
        extended_rows,
        extended_summary,
        l1_rows,
        l1_summary,
        pipeline_repair_rows,
        pipeline_repair_summary,
    )
    print(json.dumps({
        "completed_result_runs": summary.get("completed_result_runs", 0),
        "completed_by_family": summary.get("completed_by_family", {}),
        "baseline_controls_completed_result_runs": full_summary.get("completed_result_runs", 0),
        "extended_diagnostics_completed_result_runs": extended_summary.get("completed_result_runs", 0),
        "l1_isolation_completed_result_runs": l1_summary.get("completed_result_runs", 0),
        "pipeline_repair_completed_result_runs": pipeline_repair_summary.get("completed_result_runs", 0),
        "out_dir": str(args.out_dir),
        "baseline_controls_runs_root": repo_path(args.baseline_controls_runs_root),
        "extended_runs_root": repo_path(args.extended_runs_root),
        "l1_runs_root": repo_path(args.l1_runs_root),
        "pipeline_repair_runs_root": repo_path(args.pipeline_repair_runs_root),
        "runs_root": repo_path(args.runs_root),
        "tables": sorted(p.name for p in args.out_dir.glob("table_*.tex")),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
