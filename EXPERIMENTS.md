# Experiment Index

This file is the human navigation layer for the reproducibility repo. Public
names are semantic and stable.

## Public Campaigns

| Manifest | Release run root | Role | Completed / excluded | Main outputs |
| --- | --- | --- | --- | --- |
| `configs/core_diagnostics_manifest.jsonl` | `runs_core_diagnostics` | Core mechanism diagnostics, structured positive controls, learned grounding, robustness ablations, held-out order splits, and key-selectivity probes. | 319 / 21 | `table_generic_writer_by_length.tex`, `table_structured_closure_by_length.tex`, `table_learned_grounding_by_length.tex`, `table_heldout_order_by_length.tex`, `table_closure_mechanisms.tex`, `table_grounding_and_splits.tex`, `table_key_selectivity.tex`, `table_seed_variance.tex` |
| `configs/baseline_controls_manifest.jsonl` | `runs_baseline_controls` | Architecture/training stress and non-closure endpoint/scratchpad baselines. | 50 / 0 | `table_baseline_stress.tex`, `table_stronger_baselines.tex` |
| `configs/extended_diagnostics_manifest.jsonl` | `runs_extended_diagnostics` | Diagnostic ladder, key-conditioned/tied-key alignment, permutation-aware grounding, and exact rank-one wrong-key repairs. | 90 / 0 | `table_diagnostic_ladder.tex`, `table_key_alignment_baselines.tex`, `table_permutation_grounding.tex`, `table_wrong_key_repairs.tex` |
| `configs/l1_isolation_manifest.jsonl` | `runs_l1_isolation` | L=1 causal isolation, field/key ablations, training-budget curves, and exact wrong-key repair rows. | 300 / 0 | `table_l1_isolation.tex`, `table_field_ablation_summary.tex`, `table_training_curves_summary.tex`, exact rows in `table_wrong_key_repair_full_pipeline.tex` |
| `configs/pipeline_repair_manifest.jsonl` | `runs_pipeline_repair` | Corrected learned-extractor-memory repair evaluation. | 20 / 0 | learned-pipeline rows in `table_wrong_key_repair_full_pipeline.tex` |

Generated tables can be recreated from the public evidence archive with
`tools/make_diagnostic_tables.py`.

## Smoke Manifests

| Manifest | Suggested run root | Purpose |
| --- | --- | --- |
| `configs/core_diagnostics_smoke_manifest.jsonl` | `runs_smoke_core` | Short core diagnostic smoke checks. |
| `configs/extended_diagnostics_smoke_manifest.jsonl` | `runs_smoke_extended` | Short extended-diagnostic smoke checks. |
| `configs/l1_isolation_smoke_manifest.jsonl` | `runs_smoke_l1` | Short L=1 isolation smoke checks. |
| `configs/pipeline_repair_smoke_manifest.jsonl` | `runs_smoke_pipeline_repair` | Short pipeline-repair smoke checks. |

Smoke outputs are local sanity checks and are not eligible for paper tables.

## Root Scripts

| Script | Role |
| --- | --- |
| `generic_closure_writer.py` | Generic writer/direct endpoint/oracle read diagnostics. |
| `structured_transition_closure.py` | Algorithmic transition-closure positive control. |
| `learned_grounding_closure.py` | Slot-supervised extraction feeding explicit closure. |
| `grounding_ablation_suite.py` | Robustness, grounding, and identifiability ablations. |
| `heldout_relation_order_split.py` | Held-out relation-order split evaluation. |
| `anti_shortcut_key_selectivity.py` | Anti-shortcut and causal intervention probes. |
| `nonclosure_baseline_controls.py` | Stronger endpoint/scratchpad baseline campaign. |

## Regeneration Commands

Run public campaigns from the repository root:

```powershell
python paper_suite\run_manifest.py --manifest configs\baseline_controls_manifest.jsonl --out-root runs_baseline_controls --device cuda --workers 1 --resume
python paper_suite\run_manifest.py --manifest configs\extended_diagnostics_manifest.jsonl --out-root runs_extended_diagnostics --device cuda --workers 1 --resume
python paper_suite\run_manifest.py --manifest configs\l1_isolation_manifest.jsonl --out-root runs_l1_isolation --device cuda --workers 1 --resume
python paper_suite\run_manifest.py --manifest configs\pipeline_repair_manifest.jsonl --out-root runs_pipeline_repair --device cuda --workers 1 --resume
```

Regenerate tables from extracted release evidence:

```powershell
python tools\make_diagnostic_tables.py `
  --runs-root runs_core_diagnostics `
  --baseline-controls-runs-root runs_baseline_controls `
  --extended-runs-root runs_extended_diagnostics `
  --l1-runs-root runs_l1_isolation `
  --pipeline-repair-runs-root runs_pipeline_repair `
  --out-dir generated_tables
```

Corrected learned-pipeline repair rows supersede metadata-only pipeline-label
rows that are excluded from public quantitative tables.
