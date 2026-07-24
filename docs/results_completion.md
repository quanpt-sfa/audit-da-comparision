# Manuscript Results completion workflow

The Chapter 4 workflow has four mandatory layers:

1. rebuild and enrich the unrestricted paired panel;
2. reapply the previously locked issuer-year population;
3. select the nonfinancial Chapter 4 analysis sample;
4. pass the locked Jones, Shapley, and randomisation method contract.

This separation prevents metadata enrichment or implementation drift from silently changing the population or estimand contracts.

## Required inputs

Place the files at the configured paths:

```text
data/raw/financial_statement_full_long.csv.gz
data/raw/bctc_industry_icb.csv
data/raw/bctc_audit_annual_long.csv
artifacts/population_lock/population_eligible_keys.csv
```

The audit file supplies both `audit_firm` and `audit_opinion` rows. Known reused symbols are canonicalised by legal entity name:

- Container Miền Trung remains `VSM`;
- Chứng khoán VSM becomes `VSMS`;
- Gạch Ngói Từ Sơn remains `VTS`;
- Chứng khoán Việt Thành becomes `VTSC`.

If the financial-statement source has no legal-name field, VSM and VTS are accepted only when every remaining row is associated with a recognised listed exchange. OTC or unresolved rows block the build.

## Rebuild enriched unrestricted and locked panels

```powershell
python .\scripts\01_build_panel.py `
  --config .\config\signal_gate.yaml
```

The script writes:

```text
data/processed/accrual_panel_unrestricted.csv.gz
data/processed/accrual_panel.csv.gz
```

`accrual_panel_unrestricted.csv.gz` contains every paired issuer-year available from the financial-statement extraction, enriched with ICB and audit metadata. `accrual_panel.csv.gz` contains only the issuer-year keys in `population_eligible_keys.csv`. Every locked issuer-year must have exactly two reporting-state rows.

The locked file must contain at least:

```text
icb_l1
financial_flag
auditor_group
big4_flag
audit_opinion_group
analysis_eligible
exclusion_reason
```

Financial firms remain observable in the unrestricted panel. The population lock and the Chapter 4 runner record any financial and unknown-industry exclusions explicitly.

## Locked method contract

The Chapter 4 completion run uses:

- no ordinary intercept in Jones-family normal-accrual regressions;
- predictor scaling without mean centring;
- `inv_assets` as the scale regressor;
- exact three-player PAT--CFO--benchmark-residual Shapley attribution;
- signed-shift reassignment within fiscal year.

The full rationale and invariants are documented in `docs/method_contract_corrections.md`.

Run the audit independently:

```powershell
python .\scripts\32_audit_method_contract.py `
  --config .\config\results_completion.yaml `
  --skip-existing-outputs
```

## Parallel configuration

The heavy bootstrap and simulation stages use `ProcessPoolExecutor`. The default configuration is tuned for a Windows workstation with 32 physical cores and 63 logical threads:

- `parallel_workers: 31`
- `simulation_batch_size: 32`
- `blas_threads_per_worker: 1`

Limiting BLAS to one thread per process prevents each worker from spawning its own OpenMP pool.

## Clean corrected Chapter 4 run

Do not invoke the legacy completion entrypoint directly for the final run. Use the guarded wrapper, which deletes legacy checkpoints, audits the method contract, runs the completion pipeline, and audits the generated bundle again:

```powershell
python .\scripts\33_run_method_corrected_results.py `
  --config .\config\results_completion.yaml `
  --clean `
  --workers 31 `
  --simulation-batch-size 32
```

The underlying runner writes `analysis_sample_contract.json` before estimation. The method audit writes `method_contract_audit.json`. Together they record the selected issuer-year-state keys and the locked estimator, attribution, and randomisation definitions.

## Resume from compatible checkpoints

Resume only through the guarded wrapper:

```powershell
python .\scripts\33_run_method_corrected_results.py `
  --config .\config\results_completion.yaml `
  --resume `
  --workers 31 `
  --simulation-batch-size 32
```

Resume is refused when architecture checkpoints lack the no-intercept fields, attribution checkpoints lack the three-player efficiency fields, randomisation checkpoints lack the fiscal-year cell declaration, or the sample contract differs.

The runner writes base checkpoints after architecture, attribution, and switching construction. It writes separate heavy-stage checkpoints after profit-gate sensitivity, randomisation, time-shift simulation, and applied-consequence estimation. Time-shift is split by model, reference benchmark, and donor design, producing up to 36 independent process tasks.
