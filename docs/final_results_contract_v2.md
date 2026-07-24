# Final Results contract v2

This workflow supersedes the legacy `31`--`33` Results runners for the submission bundle. It fixes the remaining code and execution-contract errors identified in the Results audit.

## Corrections

1. **Separate populations.** Test cases come from the locked nonfinancial panel. Normal-accrual estimation uses prior audited observations from the unrestricted nonfinancial history, including fiscal year 2015.
2. **No current-outcome clipping.** Historical training outcomes and regressors may be winsorised. Current-state predictors may be bounded to historical support, but current `ta_scaled` outcomes remain raw.
3. **Two-player attribution.** PAT/CFO Shapley attribution is estimated only for fixed-reference benchmarks (`audited_reference` and `pre_reference`). `version_specific` benchmark movement is written to a separate diagnostic table.
4. **Complete-case switching.** Direct PAT/CFO/TA switching uses one common complete sample. Missing values never become switches or outside-gate observations.
5. **Unique applied tests.** Signed-DA state differences create one test per focal variable, not four algebraic copies across model families. High-|DA| tests remain model-specific.
6. **Aligned estimands.** The stacked model interacts every regressor and fixed effect with reporting state. Its focal state-difference coefficient is audited against the paired-difference coefficient.
7. **Fail-closed supplemental inputs.** Concentration and near-zero diagnostics are required. The final run stops when either input is absent or empty.

## Required files

```text
data/processed/accrual_panel.csv.gz
data/processed/accrual_panel_unrestricted.csv.gz
artifacts/cfs_shifting_validation/concentration_cases.csv
artifacts/cfs_shifting_validation/near_zero_randomisation.csv
```

The unrestricted panel must contain audited observations in `training_start_year` (2015 under the locked config).

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

This command deliberately fails if 2015 audited history or either required supplemental input is unavailable.

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

Do not reuse `artifacts/manuscript_results/` for the final manuscript. That directory contains legacy three-player and stale-sample checkpoints.

## Required output invariants

The post-run audit checks that:

- historical estimation reaches fiscal year 2015;
- no ordinary intercept or feature centring is introduced;
- current test outcomes are not clipped;
- fixed-reference attribution has exactly two players and zero benchmark movement;
- direct RQ2 summaries use the complete-case denominator;
- signed-DA applied tests are unique by focal variable;
- fully interacted stacked and paired-difference coefficients agree numerically;
- supplemental inference is non-empty.
