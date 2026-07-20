# Auditor-regime heterogeneity for CFS proxy validation

## Purpose

This extension tests whether the criterion validity of abnormal-CFO scores differs between Big4 and non-Big4 audit regimes. Auditor identity is joined only after expected-CFO scores are constructed and is never used as an expected-CFO predictor.

## Primary reporting window

The primary evaluation window is fiscal years 2015-2025. This aligns the empirical sample with the reporting regime governed by Circular 200/2014. Earlier years may remain in the processed accounting panel only to form lagged variables, firm history, and expanding-window training samples. They are not reported as test or outcome years.

The standard runner creates a temporary runtime config that sets:

- `minimum_year = 2015`;
- `maximum_year = 2025`;
- `minimum_test_year = 2015`;
- `maximum_test_year = 2025`.

The upstream CFS deep-dive also starts in 2015, so the observed correction targets and the proxy-validation test years use the same period boundary.

## Verified auditor source

The project source is fixed at:

```text
data/raw/bctc_audit_annual_long.csv
```

Its metadata contract was inspected directly on 2026-07-20. The file contains 40,265 rows and these relevant fields:

- issuer-year keys: `issuer_ticker`, `year`;
- annual metadata: `period_type == annual`;
- consolidated metadata: `statement_scope == Hợp nhất`;
- audit metadata: `audit_status == audited`;
- row selector: `audit_indicator == audit_firm`;
- original audit-firm name: `audit_firm_raw`;
- equality check/fallback: `audit_value_raw`;
- provenance: `source_file`.

There are 20,129 audit-firm rows representing 20,116 issuer-years. `audit_firm_raw` and `audit_value_raw` agree on all inspected audit-firm rows. Thirteen issuer-years contain multiple auditor names and are retained as `AMBIGUOUS` rather than resolved silently.

The exact `BCTC_AUDIT_ANNUAL_LONG_V1` adapter is always preferred. Generic schema discovery remains only as a fallback if the verified source file is absent.

## Auditor normalization

Original audit-firm names are retained and normalized into:

- `BIG4`;
- `NON_BIG4`;
- `UNKNOWN`;
- `AMBIGUOUS`.

Big4 matching covers Deloitte, PwC/PricewaterhouseCoopers, EY/Ernst & Young, and KPMG. Missing auditor values are never coded as non-Big4.

## Analysis sample

The extension uses `earnings_working_capital` scores from:

- `common_primary_models`;
- `analysis_core`.

The pooled sample remains unchanged. Auditor coverage does not filter the main validation results. A diagnostic join against the prior CFS case bundle matched approximately 94.55% of EWC firm-years to a known auditor tier, confirming that the issuer-year keys are compatible.

## Outputs

The extension reports:

- exact source-contract status and metadata checks;
- analysis-window status, including rows before and after the 2015-2025 filter;
- name mapping and issuer-year auditor assignments;
- Big4/non-Big4/unknown/ambiguous coverage and outcome prevalence;
- group-specific AUC, average precision, top-decile rate and lift;
- Big4-minus-non-Big4 metric differences;
- issuer-cluster bootstrap confidence intervals;
- score-by-Big4 logistic interaction models;
- balance diagnostics;
- consecutive-year auditor-tier switches.

## Interpretation

The focal parameter is the interaction between the outcome-specific abnormal-CFO score and the Big4 indicator. A prevalence difference alone does not identify audit quality because Big4 and non-Big4 clients are selected populations. Results therefore describe audit-regime-dependent criterion validity, not a causal effect of auditor choice.
