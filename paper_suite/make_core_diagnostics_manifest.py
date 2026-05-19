"""Generate the core paper experiment manifest.

The generated manifest intentionally contains more experiments than a paper will
show.  Use it to run broad falsification/reproducibility sweeps, then select the
most informative tables for the final manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def exp(exp_id: str, family: str, theme: str, script: str, args: Dict[str, object]) -> Dict[str, object]:
    return {"id": exp_id, "family": family, "theme": theme, "script": script, "args": args}


def base_args(seed: int, train_steps: int = 500, eval_n: int = 512) -> Dict[str, object]:
    return {"seed": seed, "train_steps": train_steps, "eval_n": eval_n, "batch_size": 128, "eval_batch_size": 64, "threads": 4}


def build_manifest() -> List[Dict[str, object]]:
    E: List[Dict[str, object]] = []

    # 1. Core reproducibility: negative, structured positive, hybrid positive.
    for i, seed in enumerate(range(7401, 7413), 1):
        args = base_args(seed, train_steps=900, eval_n=512)
        E.append(exp(f"generic_writer_seed_{seed}", "generic_writer", f"core_negative_transformer_writer_seed_{i}", "generic_closure_writer.py", args))
    for i, seed in enumerate(range(7501, 7513), 1):
        args = base_args(seed, train_steps=500, eval_n=512)
        E.append(exp(f"structured_closure_seed_{seed}", "structured_closure", f"core_structured_semiring_writer_seed_{i}", "structured_transition_closure.py", args))
    for i, seed in enumerate(range(7601, 7621), 1):
        args = base_args(seed, train_steps=500, eval_n=512)
        E.append(exp(f"learned_grounding_seed_{seed}", "learned_grounding", f"core_learned_extractor_hybrid_seed_{i}", "learned_grounding_closure.py", args))

    # 2. Larger eval confirmatory seeds.
    for seed in range(7631, 7636):
        args = base_args(seed, train_steps=700, eval_n=1000)
        E.append(exp(f"grounding_large_eval_seed_{seed}", "grounding_ablation", "large_eval_confirmatory_hybrid", "grounding_ablation_suite.py", args))

    # 3. Supervision/grounding ablations.
    for seed in range(7701, 7711):
        args = base_args(seed, train_steps=700, eval_n=512)
        args.update({"extraction_weight": 0.0})
        E.append(exp(f"grounding_answer_only_no_slot_seed_{seed}", "grounding_ablation", "answer_only_no_slot_supervision", "grounding_ablation_suite.py", args))
    weights = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for w in weights:
        for seed in [7721, 7722]:
            args = base_args(seed + int(w * 1000), train_steps=500, eval_n=512)
            args.update({"extraction_weight": w})
            E.append(exp(f"grounding_extraction_weight_{str(w).replace('.', 'p')}_seed_{args['seed']}", "grounding_ablation", f"extraction_weight_sweep_{w}", "grounding_ablation_suite.py", args))

    # 4. Holographic key capacity.
    for key_dim in [16, 32, 48, 64, 96, 128, 192, 256]:
        for seed in [7801, 7802]:
            args = base_args(seed + key_dim, train_steps=500, eval_n=512)
            args.update({"key_dim": key_dim})
            E.append(exp(f"grounding_key_dim_{key_dim}_seed_{args['seed']}", "grounding_ablation", f"holographic_key_capacity_{key_dim}", "grounding_ablation_suite.py", args))

    # 5. Ontology size stress.
    for num_entities in [36, 48, 72, 96, 128]:
        for seed in [7901, 7902]:
            args = base_args(seed + num_entities, train_steps=500, eval_n=512)
            args.update({"num_entities": num_entities, "max_seq_len": 2600})
            E.append(exp(f"grounding_entity_scale_{num_entities}_seed_{args['seed']}", "grounding_ablation", f"entity_ontology_scale_{num_entities}", "grounding_ablation_suite.py", args))
    for num_relations in [2, 3, 4, 6, 8]:
        for seed in [8001, 8002]:
            args = base_args(seed + num_relations, train_steps=500, eval_n=512)
            args.update({"num_relations": num_relations, "max_seq_len": 2600})
            E.append(exp(f"grounding_relation_scale_{num_relations}_seed_{args['seed']}", "grounding_ablation", f"relation_ontology_scale_{num_relations}", "grounding_ablation_suite.py", args))

    # 6. Distractor density and branching.
    density_grid = [
        (0, 0, 0.0, "no_distractors"),
        (3, 1, 0.0, "light_distractors_no_same_relation"),
        (6, 3, 0.25, "default_distractors"),
        (10, 5, 0.50, "dense_branching"),
        (14, 7, 0.75, "very_dense_branching"),
    ]
    for base_d, per_hop, same_prob, name in density_grid:
        for seed in [8101, 8102]:
            args = base_args(seed + base_d * 13 + per_hop, train_steps=500, eval_n=512)
            args.update({"base_distractors": base_d, "distractors_per_hop": per_hop, "same_relation_branch_prob": same_prob, "max_seq_len": 3200})
            E.append(exp(f"grounding_density_{name}_seed_{args['seed']}", "grounding_ablation", f"distractor_density_{name}", "grounding_ablation_suite.py", args))

    # 7. Alias/noise/fact-order robustness during training.
    for alias_prob in [0.0, 0.2, 0.5, 0.8]:
        for noise_prob in [0.0, 0.2, 0.5]:
            args = base_args(8200 + int(alias_prob * 100) + int(noise_prob * 1000), train_steps=500, eval_n=512)
            args.update({"alias_train_prob": alias_prob, "noise_train_prob": noise_prob})
            E.append(exp(f"grounding_alias_{alias_prob}_noise_{noise_prob}", "grounding_ablation", f"alias_noise_curriculum_alias{alias_prob}_noise{noise_prob}", "grounding_ablation_suite.py", args))
    for shuffle_prob in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for seed in [8251, 8252]:
            args = base_args(seed + int(shuffle_prob * 100), train_steps=500, eval_n=512)
            args.update({"fact_shuffle_train_prob": shuffle_prob})
            E.append(exp(f"grounding_fact_order_shuffle_train_{shuffle_prob}_seed_{args['seed']}", "grounding_ablation", f"fact_order_shuffle_curriculum_{shuffle_prob}", "grounding_ablation_suite.py", args))

    # 8. Train-budget and model-size sweeps.
    for steps in [50, 100, 200, 500, 1000]:
        for seed in [8301, 8302]:
            args = base_args(seed + steps, train_steps=steps, eval_n=512)
            E.append(exp(f"grounding_train_budget_{steps}_seed_{args['seed']}", "grounding_ablation", f"train_budget_{steps}", "grounding_ablation_suite.py", args))
    for d_model in [32, 64, 96, 128, 192]:
        for seed in [8401, 8402]:
            args = base_args(seed + d_model, train_steps=500, eval_n=512)
            args.update({"d_model": d_model})
            E.append(exp(f"grounding_d_model_{d_model}_seed_{args['seed']}", "grounding_ablation", f"extractor_model_scale_{d_model}", "grounding_ablation_suite.py", args))
    for seed in range(8451, 8456):
        args = base_args(seed, train_steps=500, eval_n=512)
        args.update({"use_mlp_extractor": True})
        E.append(exp(f"grounding_mlp_extractor_seed_{seed}", "grounding_ablation", "mlp_extractor_variant", "grounding_ablation_suite.py", args))

    # 9. Relation-order heldout compositional splits.
    for split in ["adjacent_pair_holdout", "trigram_holdout", "repeated_relation_holdout"]:
        for seed in range(8501, 8511):
            args = base_args(seed + len(split), train_steps=600, eval_n=512)
            args.update({"split": split})
            E.append(exp(f"heldout_order_{split}_seed_{args['seed']}", "heldout_order", f"heldout_relation_order_{split}", "heldout_relation_order_split.py", args))

    # 10. Anti-shortcut / causal-control suite.
    # These are falsification tests for the learned-grounding closure pipeline,
    # not a new architecture: relation/fact counterfactuals, memory
    # interventions, closure-key exactness, text-scanned slots, repeated
    # relations, and same-multiset order sensitivity.
    for seed in range(8901, 8925):
        args = base_args(seed, train_steps=500, eval_n=512)
        E.append(exp(f"key_selectivity_supervised_seed_{seed}", "key_selectivity", f"anti_shortcut_supervised_seed_{seed}", "anti_shortcut_key_selectivity.py", args))

    # Answer-only/no-slot runs are shortcut and latent-coding diagnostics, not
    # the main interpretable-grounding evidence.
    for seed in range(8925, 8941):
        args = base_args(seed, train_steps=700, eval_n=512)
        args.update({"extraction_weight": 0.0})
        E.append(exp(f"key_selectivity_answer_only_seed_{seed}", "key_selectivity", f"anti_shortcut_answer_only_no_slot_seed_{seed}", "anti_shortcut_key_selectivity.py", args))

    for w in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for seed in [8941, 8942, 8943, 8944]:
            args = base_args(seed + int(w * 1000), train_steps=500, eval_n=512)
            args.update({"extraction_weight": w})
            E.append(exp(f"key_selectivity_extract_weight_{str(w).replace('.', 'p')}_seed_{args['seed']}", "key_selectivity", f"anti_shortcut_extraction_weight_{w}", "anti_shortcut_key_selectivity.py", args))

    hard_causal = [
        ("dense_branching", {"base_distractors": 10, "distractors_per_hop": 5, "same_relation_branch_prob": 0.5, "max_seq_len": 3600}),
        ("very_dense_branching", {"base_distractors": 14, "distractors_per_hop": 7, "same_relation_branch_prob": 0.75, "max_seq_len": 4400}),
        ("many_relations", {"num_relations": 6, "max_seq_len": 3200}),
        ("larger_entity_set", {"num_entities": 96, "max_seq_len": 3600}),
        ("small_key_dim", {"key_dim": 32}),
        ("alias_noise_heavy", {"alias_train_prob": 0.8, "noise_train_prob": 0.5}),
        ("mlp_extractor", {"use_mlp_extractor": True}),
    ]
    for j, (name, overrides) in enumerate(hard_causal, 1):
        for seed in range(8951, 8959):
            args = base_args(seed + j * 100, train_steps=600, eval_n=512)
            args.update(overrides)
            E.append(exp(f"key_selectivity_{name}_seed_{args['seed']}", "key_selectivity", f"anti_shortcut_hard_{name}", "anti_shortcut_key_selectivity.py", args))

    # 11. Hard stress experiments for appendix, not necessarily main table.
    stress = [
        {"num_entities": 96, "num_relations": 6, "base_distractors": 10, "distractors_per_hop": 5, "relation_aliases": 5, "max_seq_len": 4200},
        {"num_entities": 408, "num_relations": 8, "base_distractors": 14, "distractors_per_hop": 7, "relation_aliases": 5, "max_seq_len": 5200, "batch_size": 16, "eval_batch_size": 16},
        {"key_dim": 16, "num_entities": 96, "num_relations": 6, "base_distractors": 10, "distractors_per_hop": 5, "max_seq_len": 4200},
        {"extraction_weight": 0.1, "alias_train_prob": 0.8, "noise_train_prob": 0.5, "base_distractors": 10, "distractors_per_hop": 5, "max_seq_len": 4200},
        {"train_steps": 100, "num_entities": 96, "num_relations": 6, "base_distractors": 10, "distractors_per_hop": 5, "max_seq_len": 4200},
    ]
    for j, overrides in enumerate(stress, 1):
        for seed in [8601, 8602]:
            args = base_args(seed + j, train_steps=overrides.get("train_steps", 700), eval_n=512)
            args.update(overrides)
            E.append(exp(f"grounding_stress_{j}_seed_{args['seed']}", "grounding_ablation", f"hard_stress_appendix_{j}", "grounding_ablation_suite.py", args))

    return E


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="configs/core_diagnostics_manifest.jsonl")
    p.add_argument("--smoke-output", type=str, default="configs/core_diagnostics_smoke_manifest.jsonl")
    args = p.parse_args()
    experiments = build_manifest()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(e, sort_keys=True) for e in experiments) + "\n", encoding="utf-8")

    smoke: List[Dict[str, object]] = [
        exp("smoke_generic_writer", "generic_writer", "smoke_transformer_negative", "generic_closure_writer.py", {"seed": 9401, "train_steps": 2, "batch_size": 4, "eval_n": 4, "eval_batch_size": 4, "d_model": 24, "key_dim": 24, "num_entities": 40, "threads": 2}),
        exp("smoke_structured_closure", "structured_closure", "smoke_semiring_positive", "structured_transition_closure.py", {"seed": 9501, "train_steps": 2, "batch_size": 4, "eval_n": 4, "num_entities": 40, "key_dim": 24, "threads": 2}),
        exp("smoke_learned_grounding", "learned_grounding", "smoke_hybrid_positive", "learned_grounding_closure.py", {"seed": 9601, "train_steps": 2, "batch_size": 4, "eval_n": 4, "eval_batch_size": 4, "num_entities": 40, "key_dim": 24, "d_model": 24, "threads": 2, "max_seq_len": 2600}),
        exp("smoke_grounding_ablation", "grounding_ablation", "smoke_hybrid_ablation", "grounding_ablation_suite.py", {"seed": 9701, "train_steps": 2, "batch_size": 4, "eval_n": 4, "eval_batch_size": 4, "num_entities": 40, "key_dim": 24, "d_model": 24, "threads": 2, "max_seq_len": 2600}),
        exp("smoke_heldout_order", "heldout_order", "smoke_compositional_split", "heldout_relation_order_split.py", {"seed": 9801, "train_steps": 2, "batch_size": 4, "eval_n": 4, "eval_batch_size": 4, "num_entities": 40, "key_dim": 24, "d_model": 24, "threads": 2, "max_seq_len": 2600, "lengths": "2,3"}),
        exp("smoke_key_selectivity", "key_selectivity", "smoke_anti_shortcut_causal_controls", "anti_shortcut_key_selectivity.py", {"seed": 9901, "train_steps": 2, "batch_size": 4, "eval_n": 4, "eval_batch_size": 4, "num_entities": 40, "num_relations": 4, "key_dim": 24, "d_model": 96, "threads": 2, "max_seq_len": 2600, "lengths": "2,3"}),
    ]
    sout = Path(args.smoke_output); sout.parent.mkdir(parents=True, exist_ok=True)
    sout.write_text("\n".join(json.dumps(e, sort_keys=True) for e in smoke) + "\n", encoding="utf-8")
    print(json.dumps({"paper_manifest": str(out), "smoke_manifest": str(sout), "num_experiments": len(experiments), "num_smoke": len(smoke)}, indent=2))


if __name__ == "__main__":
    main()
