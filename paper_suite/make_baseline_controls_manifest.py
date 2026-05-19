"""Generate baseline-control manifest.

The manifest combines K=10 closure-writer stress conditions with K=10
non-closure endpoint baselines.  The non-closure script also reports deterministic
iterative graph/pointer/DP controls, so each run separates endpoint prediction
from the single-read closure-memory bottleneck.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


Experiment = Dict[str, object]


def closure_stress_args(seed: int, *, d_model: int, layers: int, heads: int, key_dim: int, curriculum: str, lr_schedule: str, eval_batch_size: int = 64) -> Dict[str, object]:
    args: Dict[str, object] = {
        "seed": seed,
        "train_steps": 3000,
        "eval_n": 512,
        "eval_batch_size": eval_batch_size,
        "batch_size": 128,
        "threads": 4,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "key_dim": key_dim,
        "learning_rate": 0.001,
        "curriculum": curriculum,
        "lr_schedule": lr_schedule,
    }
    if lr_schedule == "cosine":
        args["warmup_frac"] = 0.05
        args["min_lr_ratio"] = 0.1
    return args


def stronger_args(seed: int, *, d_model: int, layers: int, heads: int, curriculum: str, lr_schedule: str, eval_batch_size: int) -> Dict[str, object]:
    args: Dict[str, object] = {
        "seed": seed,
        "train_steps": 3000,
        "eval_n": 512,
        "eval_batch_size": eval_batch_size,
        "batch_size": 128,
        "threads": 4,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "learning_rate": 0.001,
        "curriculum": curriculum,
        "lr_schedule": lr_schedule,
        "scratchpad_loss_weight": 1.0,
    }
    if lr_schedule == "cosine":
        args["warmup_frac"] = 0.05
        args["min_lr_ratio"] = 0.1
    return args


def build_manifest() -> List[Experiment]:
    experiments: List[Experiment] = []

    closure_conditions = [
        ("long_budget_staged", 9101, dict(d_model=96, layers=2, heads=4, key_dim=96, curriculum="staged", lr_schedule="constant", eval_batch_size=64)),
        ("wide_deep_staged", 9201, dict(d_model=192, layers=4, heads=8, key_dim=128, curriculum="staged", lr_schedule="constant", eval_batch_size=16)),
        ("wide_deep_mixed_cosine", 9301, dict(d_model=192, layers=4, heads=8, key_dim=128, curriculum="mixed", lr_schedule="cosine", eval_batch_size=16)),
    ]
    for theme, first_seed, kwargs in closure_conditions:
        for offset in range(10):
            seed = first_seed + offset
            experiments.append({
                "id": f"closure_stress_{theme}_seed_{seed}",
                "family": "closure_stress_baseline",
                "theme": theme,
                "script": "generic_closure_writer.py",
                "args": closure_stress_args(seed, **kwargs),
            })

    stronger_conditions = [
        ("relbias_scratchpad_staged", 9401, dict(d_model=96, layers=2, heads=4, curriculum="staged", lr_schedule="constant", eval_batch_size=16)),
        ("relbias_scratchpad_mixed_cosine", 9501, dict(d_model=96, layers=2, heads=4, curriculum="mixed", lr_schedule="cosine", eval_batch_size=16)),
    ]
    for theme, first_seed, kwargs in stronger_conditions:
        for offset in range(10):
            seed = first_seed + offset
            experiments.append({
                "id": f"nonclosure_{theme}_seed_{seed}",
                "family": "nonclosure_baseline",
                "theme": theme,
                "script": "nonclosure_baseline_controls.py",
                "args": stronger_args(seed, **kwargs),
            })

    ids = [str(exp["id"]) for exp in experiments]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate experiment IDs in baseline-control manifest")
    return experiments


def write_manifest(experiments: List[Experiment], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(exp, sort_keys=True) for exp in experiments) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="configs/baseline_controls_manifest.jsonl")
    args = parser.parse_args()
    experiments = build_manifest()
    out = Path(args.output)
    write_manifest(experiments, out)
    print(json.dumps({"output": str(out), "num_experiments": len(experiments)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
