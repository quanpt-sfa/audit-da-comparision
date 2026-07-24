# Supplemental diagnostics

This module replaces the two unproduced placeholder inputs formerly named
`concentration_cases.csv` and `near_zero_randomisation.csv`. The new workflow
derives every supplemental result from committed code and declared upstream
data.

## Inputs

- `data/processed/accrual_panel.csv.gz`
- either the cached mapped line-item table
  `artifacts/next_diagnostics/cfs_line_item_long.csv`
- or `data/raw/financial_statement_full_long.csv.gz`, from which the script
  rebuilds the mapped line-item table using
  `config/cfs_shifting_validation.yaml`

No manually supplied supplemental CSV is accepted.

## Diagnostic 1: cash-flow line-item revision concentration

For each issuer-year, preliminary and audited mapped concepts are paired.
Concepts present in only one state are retained and the absent state is treated
as zero; the output reports counts of pre-only and post-only concepts so this
assumption remains visible.

Let `m_j = |Delta x_j|` for active concept `j`, and
`w_j = m_j / sum_j(m_j)`. The observed normalized HHI is:

`NHHI = (sum_j(w_j^2) - 1/K) / (1 - 1/K)`

where `K` is the number of active concepts. The statistic is zero under equal
allocation and one under complete concentration.

The null preserves each case's total absolute revision magnitude and number of
active concepts. For every draw, it samples `K` positive revision magnitudes
with replacement from the same-fiscal-year empirical line-item pool, normalizes
them into shares, and recomputes NHHI. If a year has fewer than the configured
minimum pool size, the all-year pool is used. The case-level diagnostic is:

`excess_nhhi = observed_nhhi - E_null(NHHI)`

Issuer-cluster bootstrap inference is applied to the mean excess NHHI.

## Diagnostic 2: near-zero CFO sign shift

CFO is scaled by the absolute value of beginning assets. A firm-year enters the
near-zero population when both preliminary and audited absolute scaled CFO are
at or below the configured threshold.

Preliminary and audited absolute distances from zero are pooled within fiscal
year and assigned quantile bins. A pair is retained only when both states occupy
the same distance bin. The observed statistic is the mean paired change in the
positive-CFO indicator:

`mean[I(CFO_audited > 0) - I(CFO_pre > 0)]`

The randomisation null independently swaps the preliminary and audited state
labels within each firm-year pair. This preserves both observed values,
absolute-distance matching, fiscal year, and the paired dependence structure.

## Outputs

The runner writes:

- `cfs_revision_concentration_cases.csv`
- `near_zero_cfo_cases.csv`
- `near_zero_cfo_permutation_draws.csv`
- `supplemental_diagnostics_summary.csv`
- `supplemental_diagnostics_manifest.json`

## Run

```powershell
python .\scripts\36_run_supplemental_diagnostics.py `
  --config .\config\supplemental_diagnostics.yaml
```

To rebuild mapped line items from the raw long file:

```powershell
python .\scripts\36_run_supplemental_diagnostics.py `
  --config .\config\supplemental_diagnostics.yaml `
  --rebuild-line-items
```

The supplemental workflow is intentionally separate from the core Chapter 4
runner. A failure in raw line-item mapping must not invalidate RQ1/RQ2 results,
and core results must not silently substitute stale supplemental CSVs.
