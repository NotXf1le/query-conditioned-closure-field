"""Generate the L=1 isolation and key-repair manifest.

The full manifest is intentionally large and is meant for external CUDA
hardware.  Use --smoke for local contract checks.
"""
from __future__ import annotations

from typing import Dict, List
import argparse
import json
from pathlib import Path


Experiment = Dict[str, object]


L1_CONDITIONS = [
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


def base_l1_args(seed: int, *, smoke: bool) -> Dict[str, object]:
    return {
        "seed": seed,
        "d_model": 96,
        "layers": 2,
        "heads": 4,
        "key_dim": 96,
        "curriculum": "staged",
        "lr_schedule": "constant",
        "train_steps": 2 if smoke else 3000,
        "batch_size": 4 if smoke else 128,
        "eval_n": 4 if smoke else 512,
        "eval_batch_size": 4 if smoke else 64,
        "threads": 2 if smoke else 4,
        "eval_lengths": "1",
        "log_every": 1 if smoke else 500,
    }


def l1_args(seed: int, *, condition: str, smoke: bool) -> Dict[str, object]:
    args = base_l1_args(seed, smoke=smoke)
    args["condition"] = condition
    if condition == "field_supervised_mse":
        args["field_loss"] = "mse"
    if condition == "field_supervised_read_ce":
        args["field_loss"] = "multi_read_ce"
    return args


def repair_args(seed: int, *, repair_variant: str, pipeline: str, smoke: bool) -> Dict[str, object]:
    args: Dict[str, object] = {
        "seed": seed,
        "repair_variant": repair_variant,
        "pipeline": pipeline,
        "wrong_key_families": "swapped,reversed,prefix,shuffled,random_source",
        "calibration_split": "validation" if pipeline == "learned_grounding" else "fixed_threshold",
        "key_dim": 128,
        "train_steps": 2 if smoke else 500,
        "batch_size": 4 if smoke else 128,
        "eval_n": 4 if smoke else 512,
        "eval_batch_size": 4 if smoke else 64,
        "threads": 2 if smoke else 4,
        "eval_lengths": "1,2" if smoke else "1,2,3,4,6,8,12,16,24,32",
    }
    return args


def append_k(
    experiments: List[Experiment],
    *,
    smoke: bool,
    k: int,
    first_seed: int,
    family: str,
    theme: str,
    script: str,
    args_builder,
) -> None:
    for offset in range(k):
        seed = first_seed + offset
        prefix = "smoke_" if smoke else ""
        experiments.append({
            "id": f"{prefix}{family}_{theme}_seed_{seed}",
            "family": family,
            "theme": theme,
            "script": script,
            "args": args_builder(seed),
        })


def build_manifest(*, smoke: bool = False, pipeline_repair_only: bool = False) -> List[Experiment]:
    experiments: List[Experiment] = []
    k = 1 if smoke else 10

    if pipeline_repair_only:
        for idx, (theme, variant) in enumerate([
            ("threshold_null_learned_pipeline", "threshold_null"),
            ("margin_gate_learned_pipeline", "margin_gate"),
        ]):
            append_k(
                experiments,
                smoke=smoke,
                k=k,
                first_seed=18001 + 100 * idx,
                family="key_rejection_repair_pipeline",
                theme=theme,
                script="closure_key_rejection_repairs.py",
                args_builder=lambda seed, variant=variant: repair_args(seed, repair_variant=variant, pipeline="learned_grounding", smoke=smoke),
            )
        ids = [str(exp["id"]) for exp in experiments]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate experiment IDs in pipeline-repair manifest")
        return experiments

    for idx, condition in enumerate(L1_CONDITIONS):
        append_k(
            experiments,
            smoke=smoke,
            k=k,
            first_seed=12001 + 100 * idx,
            family="l1_causal_isolation",
            theme=condition,
            script="closure_l1_causal_isolation.py",
            args_builder=lambda seed, condition=condition: l1_args(seed, condition=condition, smoke=smoke),
        )

    for idx, dim in enumerate([32, 64, 96, 128, 256, 512, 1024]):
        append_k(
            experiments,
            smoke=smoke,
            k=k,
            first_seed=14001 + 100 * idx,
            family="l1_field_ablation",
            theme=f"keydim_{dim}",
            script="closure_l1_causal_isolation.py",
            args_builder=lambda seed, dim=dim: {**l1_args(seed, condition="l1_with_distractors", smoke=smoke), "key_dim": dim},
        )

    for idx, codebook in enumerate(["random_sign", "orthogonalized", "learned_key_projection"]):
        append_k(
            experiments,
            smoke=smoke,
            k=k,
            first_seed=15001 + 100 * idx,
            family="l1_field_ablation",
            theme=f"codebook_{codebook}",
            script="closure_l1_causal_isolation.py",
            args_builder=lambda seed, codebook=codebook: {
                **l1_args(seed, condition="teacher_forced_q", smoke=smoke),
                "key_codebook": codebook,
            },
        )

    for idx, steps in enumerate([3000, 10000, 30000, 100000]):
        append_k(
            experiments,
            smoke=smoke,
            k=k,
            first_seed=16001 + 100 * idx,
            family="l1_training_budget",
            theme=f"budget_{steps}",
            script="closure_l1_causal_isolation.py",
            args_builder=lambda seed, steps=steps: {
                **l1_args(seed, condition="l1_with_distractors", smoke=smoke),
                "train_steps": 2 if smoke else steps,
            },
        )

    repair_specs = [
        ("linear_exact", "linear", "exact_rank_one"),
        ("contrastive_exact", "contrastive", "exact_rank_one"),
        ("threshold_null_exact", "threshold_null", "exact_rank_one"),
        ("margin_gate_exact", "margin_gate", "exact_rank_one"),
        ("threshold_null_full_pipeline", "threshold_null", "learned_grounding"),
        ("margin_gate_full_pipeline", "margin_gate", "learned_grounding"),
    ]
    for idx, (theme, variant, pipeline) in enumerate(repair_specs):
        append_k(
            experiments,
            smoke=smoke,
            k=k,
            first_seed=17001 + 100 * idx,
            family="key_rejection_repair_pipeline",
            theme=theme,
            script="closure_key_rejection_repairs.py",
            args_builder=lambda seed, variant=variant, pipeline=pipeline: repair_args(seed, repair_variant=variant, pipeline=pipeline, smoke=smoke),
        )

    ids = [str(exp["id"]) for exp in experiments]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate experiment IDs in L=1 isolation manifest")
    return experiments


def write_manifest(experiments: List[Experiment], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(exp, sort_keys=True) for exp in experiments) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="configs/l1_isolation_manifest.jsonl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pipeline-repair-only", action="store_true")
    args = parser.parse_args()
    experiments = build_manifest(smoke=args.smoke, pipeline_repair_only=args.pipeline_repair_only)
    out = Path(args.output)
    write_manifest(experiments, out)
    print(json.dumps({
        "output": str(out),
        "num_experiments": len(experiments),
        "pipeline_repair_only": bool(args.pipeline_repair_only),
        "smoke": bool(args.smoke),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
