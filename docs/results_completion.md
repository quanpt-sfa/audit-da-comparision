# Manuscript Results completion workflow

> **Authoritative submission workflow:** use scripts `34`--`35` and `artifacts/manuscript_results_final/`. Scripts `31`--`33` and `artifacts/manuscript_results/` are retained only to audit superseded outputs.

The final Chapter 4 workflow has five mandatory layers:

1. rebuild and enrich the unrestricted paired panel;
2. reapply the locked issuer-year test population;
3. retain unrestricted prior audited observations for model estimation;
4. select nonfinancial analysis and training populations separately;
5. pass the final sample, estimator, attribution, randomisation, applied-test, and supplemental-input contracts.

## Required source inputs

```text
data/raw/financial_statement_full_long.csv.gz
data/raw/bctc_industry_icb.csv
data/raw/bctc_audit_annual_long.csv
artifacts/population_lock/population_eligible_keys.csv
```

Rebuild the enriched panels:

```powershell
python .\scripts\01_build_panel.py `
  --config .\config\signal_gate.yaml
```

This must write:

```text
data/processed/accrual_panel_unrestricted.csv.gz
data/processed/accrual_panel.csv.gz
```

The unrestricted file supplies historical audited estimation observations. The locked file supplies only Chapter 4 test issuer-years. Both files are filtered to nonfinancial firms by the final runner, but only the test file is subject to the issuer-year population lock.

## Required final-analysis inputs

```text
data/processed/accrual_panel.csv.gz
data/processed/accrual_panel_unrestricted.csv.gz
artifacts/cfs_shifting_validation/concentration_cases.csv
artifacts/cfs_shifting_validation/near_zero_randomisation.csv
```

The unrestricted panel must contain audited rows in fiscal year 2015. The final workflow stops rather than silently starting estimation in 2016. The concentration and near-zero inputs are also required; an empty supplemental table is not accepted.

## Final method contract

The authoritative run uses:

- no ordinary intercept in Jones-family normal-accrual regressions;
- predictor scaling without mean centring;
- `inv_assets` as the scale regressor;
- historical-only outcome winsorisation: current-state `ta_scaled` is not clipped;
- exact two-player PAT--CFO Shapley attribution for fixed-reference benchmarks;
- a separate diagnostic for version-specific benchmark movement;
- one common complete-case population for direct switching;
- signed-shift reassignment within fiscal year;
- paired-difference regression as the primary applied state-dependence test;
- fully interacted stacked sensitivity models;
- one unique signed-DA change test per focal variable before multiplicity adjustment.

The complete specification is in `docs/final_results_contract_v2.md`.

## Focused tests

```powershell
pytest -q `
  tests\test_final_results_contract.py `
  tests\test_method_contract.py `
  tests\test_switching_complete_case.py `
  tests\test_results_completion.py `
  tests\test_results_parallel.py
```

## Pre-run audit

```powershell
python .\scripts\34_audit_final_results_contract.py `
  --config .\config\results_completion.yaml `
  --skip-existing-outputs
```

## Final clean run

```powershell
python .\scripts\35_run_final_results.py `
  --config .\config\results_completion.yaml `
  --clean `
  --workers 31 `
  --simulation-batch-size 32 `
  2>&1 | Tee-Object .\artifacts\chapter4_final_v2.log
```

Final outputs are written to:

```text
artifacts/manuscript_results_final/
```

The wrapper deletes stale final outputs, audits all required inputs, executes the full pipeline, and audits the generated bundle. Resume is intentionally disabled because the estimator, attribution estimand, applied test family, and training population changed together.

## Manuscript integration

Do not update or merge Chapter 4 from the legacy output directory. After the final post-run audit passes, regenerate Chapter 4, Discussion, Conclusion, and the Round 48 LaTeX project only from `artifacts/manuscript_results_final/`.
