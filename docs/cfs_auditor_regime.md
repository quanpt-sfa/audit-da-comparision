# Auditor-regime heterogeneity for CFS proxy validation

## Purpose

This extension tests whether the criterion validity of abnormal-CFO scores differs between Big4 and non-Big4 audit regimes. Auditor identity is joined only after expected-CFO scores are constructed and is never used as an expected-CFO predictor.

## Auditor source discovery

The accounting panel and financial-statement long file do not need to contain auditor identity. The loader searches, in order:

1. configured `auditor_input` or `audit_input` paths;
2. `panel_input` and `raw_input` when they contain auditor metadata;
3. explicit historical paths such as `data/bctc_audit_annual_long.csv`;
4. conservative repository globs for `bctc_audit_annual_long.csv*` and annual audit CSV files.

Two source schemas are supported:

- wide schema, where fields such as `audit_firm_name` or `auditor_name` are columns;
- long schema, where `audit_indicator` identifies `audit_firm` and a value column contains the original audit-firm name.

The source-status output records the selected file, schema type, ticker/year fields, indicator/value fields when applicable, and coverage counts. Unknown auditor names are not coded as non-Big4.

If no usable auditor source is found, the extension writes `NOT_EVALUATED`, removes stale auditor result tables, and allows the pooled CFS validation pipeline to finish. Set `fail_pipeline_if_unavailable: true` only when auditor heterogeneity is intentionally treated as a mandatory execution gate.

## Auditor normalization

Original audit-firm names are retained and normalized into:

- `BIG4`;
- `NON_BIG4`;
- `UNKNOWN`;
- `AMBIGUOUS`.

Big4 matching covers Deloitte, PwC/PricewaterhouseCoopers, EY/Ernst & Young, and KPMG. Multiple inconsistent auditor names within the same issuer-year are classified as ambiguous rather than silently selecting one.

## Analysis sample

The extension uses `earnings_working_capital` scores from:

- `common_primary_models`;
- `analysis_core`.

The pooled sample remains unchanged. Auditor coverage does not filter the main validation results.

## Outputs

The extension reports:

- auditor source and schema status;
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
