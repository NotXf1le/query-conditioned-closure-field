"""Generate the extended diagnostics manifest.

The manifest is intentionally separated from the core diagnostics manifest.
Heavy runs target external CUDA hardware; use --smoke for local sanity checks.
"""
from __future__ import annotations

from typing import Dict, List
import argparse
import json
from pathlib import Path


Experiment = Dict[str, object]


def ladder_args(seed: int, *, writer_variant: str, smoke: bool) -> Dict[str, object]:
    args: Dict[str, object] = {
        "seed": seed,
        "writer_variant": writer_variant,
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
        "ladder_rungs": "direct_qv_write,gold_target_write,one_hop_fact_write" if smoke else "direct_qv_write,gold_target_write,one_hop_fact_write,multi_hop_closure_write",
    }
    if not smoke:
        args["eval_lengths"] = "1,2,3,4,6,8,12,16,24,32"
    return args


def repair_args(seed: int, *, repair_variant: str, smoke: bool) -> Dict[str, object]:
    args: Dict[str, object] = {
        "seed": seed,
        "repair_variant": repair_variant,
        "key_dim": 128,
        "train_steps": 2 if smoke else 500,
        "batch_size": 4 if smoke else 128,
        "eval_n": 4 if smoke else 512,
        "eval_batch_size": 4 if smoke else 64,
        "threads": 2 if smoke else 4,
    }
    if smoke:
        args["eval_lengths"] = "1,2"
    return args


def permutation_args(seed: int, *, condition: str, smoke: bool) -> Dict[str, object]:
    return {
        "seed": seed,
        "condition": condition,
        "d_model": 96,
        "key_dim": 128,
        "train_steps": 2 if smoke else 1000,
        "batch_size": 4 if smoke else 128,
        "eval_n": 4 if smoke else 512,
        "eval_batch_size": 4 if smoke else 64,
        "threads": 2 if smoke else 4,
        "eval_lengths": "1,2" if smoke else "1,2,3,4,6,8,12,16,24,32",
    }


def build_manifest(*, smoke: bool = False) -> List[Experiment]:
    experiments: List[Experiment] = []
    k = 1 if smoke else 10

    for writer_variant, first_seed in [
        ("baseline", 10101),
        ("key_conditioned", 10201),
        ("tied_key", 10301),
    ]:
        for offset in range(k):
            seed = first_seed + offset
            experiments.append({
                "id": f"{'smoke_' if smoke else ''}ladder_{writer_variant}_seed_{seed}",
                "family": "closure_writer_ladder",
                "theme": writer_variant,
                "script": "closure_writer_diagnostic_ladder.py",
                "args": ladder_args(seed, writer_variant=writer_variant, smoke=smoke),
            })

    for repair_variant, first_seed in [
        ("linear", 10401),
        ("contrastive", 10501),
        ("threshold_null", 10601),
        ("margin_gate", 10701),
    ]:
        for offset in range(k):
            seed = first_seed + offset
            experiments.append({
                "id": f"{'smoke_' if smoke else ''}repair_{repair_variant}_seed_{seed}",
                "family": "key_rejection_repair",
                "theme": repair_variant,
                "script": "closure_key_rejection_repairs.py",
                "args": repair_args(seed, repair_variant=repair_variant, smoke=smoke),
            })

    for condition, first_seed in [
        ("answer_only_no_slot", 10801),
        ("extraction_weight_zero", 10901),
    ]:
        for offset in range(k):
            seed = first_seed + offset
            experiments.append({
                "id": f"{'smoke_' if smoke else ''}perm_{condition}_seed_{seed}",
                "family": "grounding_permutation",
                "theme": condition,
                "script": "grounding_permutation_diagnostics.py",
                "args": permutation_args(seed, condition=condition, smoke=smoke),
            })

    ids = [str(exp["id"]) for exp in experiments]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate experiment IDs in extended diagnostics manifest")
    return experiments


def write_manifest(experiments: List[Experiment], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(exp, sort_keys=True) for exp in experiments) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="configs/extended_diagnostics_manifest.jsonl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiments = build_manifest(smoke=args.smoke)
    out = Path(args.output)
    write_manifest(experiments, out)
    print(json.dumps({"output": str(out), "num_experiments": len(experiments), "smoke": bool(args.smoke)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
