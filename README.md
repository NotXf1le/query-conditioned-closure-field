# Query-Conditioned Closure-Field Writing

This repository is the reproducibility artifact for the paper
**Query-Conditioned Closure-Field Writing Under a Single-Read Bottleneck**.

The paper should be read as a mechanism analysis, not as a new general-purpose
memory architecture. The diagnostic interface asks a query-conditioned writer to
emit a dense transient field `M_q`, which is read exactly once by `q^T M_q`.
The controls separate endpoint prediction from field writing, closure
construction, closure reading, and wrong-key rejection.

## Repository Contents

- `configs/`: public JSONL manifests for reproducible experiment campaigns.
- `paper_suite/`: manifest runner, aggregation helpers, and table utilities.
- `tools/`: table generator for release evidence.
- `tests/`: unit tests for generators, manifests, and diagnostic components.
- `artifacts/release_manifest.json`: release asset contract and checksums.

The manuscript PDF, manuscript source, response documents, and local reports are
not part of this public repository.

Completed run directories are not stored in git. They are distributed as
versioned GitHub Release assets and archived through Zenodo.

## Quickstart

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Run a small CPU smoke check:

```powershell
python paper_suite/run_manifest.py --manifest configs/core_diagnostics_smoke_manifest.jsonl --out-root runs_smoke_core --device cpu --workers 1 --resume
```

Regenerate evidence tables after extracting the evidence archive into the
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

## Public Repository and DOI

Canonical artifact repository:

<https://github.com/NotXf1le/query-conditioned-closure-field>

The citable artifact is the Zenodo archive of the versioned GitHub Release.
The `v1.1.0` version DOI is assigned after Zenodo processes the GitHub Release.

Concept DOI: <https://doi.org/10.5281/zenodo.20279221>

Release assets listed in `artifacts/release_manifest.json`:

- `query-conditioned-closure-field-evidence.zip`
- `SHA256SUMS.txt`
