# Reproducibility Contract

## Artifact Contract

A quantitative result is eligible for the paper only when its run directory
contains all three items:

- `status.json` with a completed status.
- `experiment.json` matching the manifest entry.
- The expected result CSV or JSON file.

Smoke runs, failed runs, partial runs, and metadata-only rows are excluded. The
table generator consumes only eligible rows and writes reproducibility tables.
The release manifest records the evidence blocks, source manifests, run counts,
release asset names, and checksums.

## Public Evidence Blocks

| Evidence block | Manifest | Run root | Status |
| --- | --- | --- | --- |
| Core diagnostics | `configs/core_diagnostics_manifest.jsonl` | `runs_core_diagnostics` | 319 done / 21 failed; completed only |
| Baseline controls | `configs/baseline_controls_manifest.jsonl` | `runs_baseline_controls` | 50 done / 0 failed |
| Extended diagnostics | `configs/extended_diagnostics_manifest.jsonl` | `runs_extended_diagnostics` | 90 done / 0 failed |
| L=1 isolation | `configs/l1_isolation_manifest.jsonl` | `runs_l1_isolation` | 310 done / 0 failed |
| Pipeline repair | `configs/pipeline_repair_manifest.jsonl` | `runs_pipeline_repair` | 20 done / 0 failed |

Corrected learned-pipeline repair runs supersede earlier metadata-only
pipeline-label rows. Public quantitative tables use only completed run
directories listed in the release manifest.

## Rerun Commands

Heavy campaigns are run from the repository root. Example commands:

```powershell
python paper_suite/run_manifest.py --manifest configs/baseline_controls_manifest.jsonl --out-root runs_baseline_controls --device cuda --workers 1 --resume
python paper_suite/run_manifest.py --manifest configs/extended_diagnostics_manifest.jsonl --out-root runs_extended_diagnostics --device cuda --workers 1 --resume
python paper_suite/run_manifest.py --manifest configs/l1_isolation_manifest.jsonl --out-root runs_l1_isolation --device cuda --workers 1 --resume
python paper_suite/run_manifest.py --manifest configs/pipeline_repair_manifest.jsonl --out-root runs_pipeline_repair --device cuda --workers 1 --resume
```

The current public L=1 budget sweep includes `budget_100000` runs with seeds
16301--16310.

For a local smoke check:

```powershell
python paper_suite/run_manifest.py --manifest configs/core_diagnostics_smoke_manifest.jsonl --out-root runs_smoke_core --device cpu --workers 1 --resume
```

## Evidence Table Regeneration

After extracting `query-conditioned-closure-field-evidence.zip` into the
repository root:

```powershell
python tools/make_diagnostic_tables.py `
  --runs-root runs_core_diagnostics `
  --baseline-controls-runs-root runs_baseline_controls `
  --extended-runs-root runs_extended_diagnostics `
  --l1-runs-root runs_l1_isolation `
  --pipeline-repair-runs-root runs_pipeline_repair `
  --out-dir generated_tables
```

## Release Assets

The GitHub Release contains:

- `query-conditioned-closure-field-evidence.zip`
- `SHA256SUMS.txt`

The ZIP archives are not committed to git. Their SHA256 values are tracked in
`SHA256SUMS.txt` and `artifacts/release_manifest.json`.
