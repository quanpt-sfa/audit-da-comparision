# Comparative predictive validity of reporting states

## Purpose

This workflow evaluates whether the audited annual reporting state improves established dimensions of accounting-information quality relative to the annual state aggregated from quarterly reports. It is separate from the discretionary-accrual Chapter 4 contract and does not use DA as a validation target.

The primary outcome is always an audited value observed in fiscal year `t+1`. Current predictors are taken separately from the unaudited quarterly-aggregate state and the audited annual state in fiscal year `t`.

## Time contract

The processed source panel covers 2015--2025.

- 2015 supplies beginning-assets support.
- Predictor years are 2016--2024.
- Audited outcomes are observed in 2017--2025.
- Expanding-window OOS evaluation begins with predictor year 2018, using 2016--2017 as the first historical training window.
- Accrual-quality robustness uses fiscal years 2017--2024 because it requires audited CFO at both `t-1` and `t+1`.

No observation from the test predictor year or a later year enters an OOS training fold.

## Main tests

### 1. Earnings persistence

```text
ROA_audited(t+1) <- ROA_state(t)
```

The focal coefficient is the persistence of current earnings. The state contrast is the audited slope minus the quarterly-aggregate slope.

### 2. Earnings informativeness about future CFO

```text
CFO_audited(t+1) <- ROA_state(t)
```

This tests whether current earnings in either reporting state map more strongly and more accurately into future audited operating cash flow.

### 3. CFO persistence

```text
CFO_audited(t+1) <- CFO_state(t)
```

This is the cash-flow persistence or cash-flow predictability benchmark.

### 4. Earnings/CFO horse race

```text
CFO_audited(t+1) <- ROA_state(t) + CFO_state(t)
```

This reports the incremental future-CFO information in current earnings and current CFO within each reporting state.

## Common-sample contract

Every state comparison is estimated on a common complete-case firm-year sample. A row is admitted only when:

- both reporting states exist in year `t`;
- both state-specific predictors are finite;
- the audited `t+1` outcome is finite;
- the firm is in the locked nonfinancial analysis population.

The workflow never compares fit statistics calculated from different state-specific samples.

## Coefficient inference

Canonical and year/industry fixed-effect specifications are estimated. The two states are represented as block-diagonal regressions in one stacked system. All state-specific regressors, intercepts and fixed effects are separated. Issuer-clustered covariance therefore provides a direct test of each audited-minus-pre coefficient difference while preserving dependence between the two rows belonging to the same firm-year.

## Out-of-sample evaluation

OOS forecasts use expanding prior-year training windows. For each fold:

1. training outcomes and predictors are winsorised at historical 1%/99% bounds;
2. current test predictors are bounded to historical support;
3. current audited test outcomes remain raw;
4. pre and audited models are evaluated on identical test observations.

Reported metrics are RMSE, MAE and OOS R-squared relative to the historical-mean forecast. RMSE and MAE state differences use issuer-cluster bootstrap inference. For loss differences, a negative audited-minus-pre estimate favours the audited state.

## Accrual-quality robustness

Working-capital accruals are constructed as total balance-sheet accruals plus depreciation, scaled by audited beginning assets. Where necessary, the workflow falls back to the direct current-asset/current-liability construction.

For each reporting state:

```text
WCA_state(t) <- CFO_audited(t-1) + CFO_state(t) + CFO_audited(t+1)
```

The surrounding cash-flow realizations are audited; only the current reporting state changes. Coefficients are reported under the same pooled specifications as the main tests. Quality metrics use leave-one-fiscal-year-out residuals, preventing the evaluated year from fitting its own mapping. Lower RMSE, MAE and residual dispersion indicate tighter cash-flow realization.

This analysis is robustness evidence, not a ground-truth measure of reporting error.

## Run

```powershell
python .\scripts\37_run_predictive_validity.py `
  --config .\config\predictive_validity.yaml `
  --clean `
  2>&1 | Tee-Object .\artifacts\predictive_validity.log
```

Smoke run with fewer bootstrap draws:

```powershell
python .\scripts\37_run_predictive_validity.py `
  --config .\config\predictive_validity.yaml `
  --clean `
  --bootstrap-draws 50
```

## Outputs

The runner writes to `artifacts/predictive_validity/`:

- `predictive_validity_cases.csv`;
- `predictive_validity_sample_manifest.csv`;
- `predictive_validity_coefficients.csv`;
- `predictive_validity_pooled_fit.csv`;
- `predictive_validity_oos_folds.csv`;
- `predictive_validity_oos_predictions.csv`;
- `predictive_validity_oos_summary.csv`;
- `predictive_validity_oos_state_differences.csv`;
- `accrual_quality_cases.csv`;
- `accrual_quality_coefficients.csv`;
- `accrual_quality_crossfit_cases.csv`;
- `accrual_quality_summary.csv`;
- `accrual_quality_state_differences.csv`;
- `predictive_validity_manifest.json`.

The manifest records the analysis population hash, settings, sample exclusions, row counts and deterministic hashes of every table.

## Tests

```powershell
pytest -q tests\test_predictive_validity.py
```
