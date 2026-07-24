# Manuscript Results completion workflow

The Chapter 4 workflow now has two mandatory stages: rebuild an enriched master panel, then run the nonfinancial analysis sample. The panel retains excluded firms for auditability but the Results runner removes financial firms before every accrual, Shapley, switching, randomisation, time-shift, and applied-consequence calculation.

## Required raw inputs

Place the files at the configured paths:

```text
data/raw/financial_statement_full_long.csv.gz
data/raw/bctc_industry_icb.csv
data/raw/bctc_audit_annual_long.csv
```

The audit file supplies both `audit_firm` and `audit_opinion` rows. Known reused symbols are canonicalised by legal entity name:

- Container Miền Trung remains `VSM`;
- Chứng khoán VSM becomes `VSMS`;
- Gạch Ngói Từ Sơn remains `VTS`;
- Chứng khoán Việt Thành becomes `VTSC`.

## Rebuild the enriched master panel

```powershell
python .\scripts\01_build_panel.py `
  --config .\config\signal_gate.yaml
```

The resulting `data/processed/accrual_panel.csv.gz` must contain at least:

```text
icb_l1
financial_flag
auditor_group
big4_flag
audit_opinion_group
analysis_eligible
exclusion_reason
```

Financial firms remain in this master file with `exclusion_reason=financial_firm`. The Chapter 4 runner enforces the nonfinancial restriction and refuses to use an unenriched panel.

## Parallel configuration

The heavy bootstrap and simulation stages are vectorized and distributed with `ProcessPoolExecutor`. The default configuration is tuned for a Windows workstation with 32 physical cores and 63 logical threads:

- `parallel_workers: 31`
- `simulation_batch_size: 32`
- `blas_threads_per_worker: 1`

Limiting BLAS to one thread per process prevents each worker from spawning its own large OpenMP pool.

## Full Chapter 4 run

Delete legacy outputs because they predate the nonfinancial sample contract:

```powershell
Remove-Item .\artifacts\manuscript_results -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\artifacts | Out-Null
```

Then run:

```powershell
python .\scripts\31_complete_manuscript_results.py `
  --config .\config\results_completion.yaml `
  --workers 31 `
  --simulation-batch-size 32
```

The runner writes `analysis_sample_contract.json` before estimation. It records the master-panel size, nonfinancial analysis size, exclusions, and a deterministic hash of the selected issuer-year-state keys.

## Resume from compatible checkpoints

```powershell
python .\scripts\31_complete_manuscript_results.py `
  --config .\config\results_completion.yaml `
  --resume `
  --workers 31
```

Resume is permitted only when `analysis_sample_contract.json` exactly matches the current nonfinancial sample. Old checkpoints without this contract are rejected.

The runner writes base checkpoints after architecture, attribution, and switching construction. It also writes separate heavy-stage checkpoints after profit-gate sensitivity, randomisation, time-shift simulation, and applied-consequence estimation. Progress messages report the number of process tasks and completed tasks. Time-shift is split by model, reference benchmark, and donor design, producing up to 36 independent process tasks.

The runner rebuilds four model families, four historical architectures, three reference-state constructions, Shapley attribution, issuer-cluster inference, common post-audit CDF switching outputs, randomisation benchmarks, ICB Level-1 same-year peer donors, all 24 applied-consequence comparisons including Big Four, and the three confirmatory intersection-union families.
