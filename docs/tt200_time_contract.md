# TT200 empirical time contract

All analyses on this branch use one explicit reporting-regime contract.

## Contract

- Source and target-construction years: 2015-2025.
- Rolling-training start: 2015.
- Theoretical out-of-sample test years: 2016-2025.
- Effective first test year: the first year at or after 2016 that satisfies the prespecified minimum training-sample requirement.
- Cross-model comparisons: the issuer-year intersection on which every prespecified comparison model is available.

The distinction is deliberate. Fiscal year 2015 is part of the TT200 source regime and can be used as the first training/warm-up year. It is not an out-of-sample test year because no earlier TT200 year exists for model training. Target construction remains independent of model availability.

## Enforced components

The contract is enforced in:

- processed signal-panel source limits;
- Bayesian signal analysis;
- OLS discretionary-accrual baselines;
- rolling calibration;
- direct CFS identity and target construction;
- CFS deep-dive diagnostics;
- line-item inventory and reconciliation;
- rolling expected-CFO models;
- common-primary and common-all comparisons;
- Big4/non-Big4 auditor-regime heterogeneity;
- completion gates and artifact status tables.

## OLS and expected-CFO training

For test year `t`, the permissible training set is:

```text
2015 <= fiscal_year <= t - 1
```

No observation before 2015 may enter coefficient estimation, winsorization bounds, firm-history construction, stacking, validation weights, or expected-CFO estimation.

## Target independence

CFS targets are constructed directly from paired preliminary and audited statement values. Missing OLS or expected-CFO folds must never convert an otherwise valid target observation into a negative label. Model coverage affects only whether a score can be evaluated.

## Artifact checks

New or updated outputs include:

- OLS rows with source, training and test contract metadata;
- `cfs_identity_window_status.csv`;
- `cfs_target_input_coverage.csv`;
- `cfs_deep_dive_window_status.csv`;
- `rolling_calibration_window_status.csv`;
- `cfs_analysis_window_status.csv`;
- `cfs_time_contract_status.csv`;
- `cfs_auditor_analysis_window_status.csv`;
- completion gate `consistent_tt200_time_contract`.

`run_next_diagnostics.py` rejects an OLS baseline that lacks contract metadata, contains training years before 2015, or contains test years outside 2016-2025. This prevents stale artifacts from silently contaminating a rerun.
