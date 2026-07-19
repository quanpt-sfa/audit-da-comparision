# Auditor-regime heterogeneity protocol

## Question

The analysis asks whether abnormal-CFO criterion validity differs between Big4 and non-Big4 audit regimes. Auditor identity is not an expected-CFO predictor. It is joined to the locked `common_primary_models + analysis_core` cases after out-of-time scores have been generated.

## Auditor mapping

The loader searches the configured auditor input, the processed panel, and then the raw long source. Column names may be configured explicitly; otherwise conservative candidate lists and schema detection are used. Raw auditor names are normalized to:

- `BIG4`: Deloitte, PwC/PricewaterhouseCoopers, EY/Ernst & Young, or KPMG;
- `NON_BIG4`: a non-empty auditor name matching no Big4 brand;
- `UNKNOWN`: no usable auditor name;
- `AMBIGUOUS`: conflicting auditor identities for one issuer-year or multiple Big4 brands in one raw label.

Missing auditor values are never coded as non-Big4. The pipeline exports raw-to-normalized mappings and firm-year resolution statuses.

## Primary heterogeneity evidence

For the prespecified `earnings_working_capital` proxy, the pipeline reports by auditor group:

- candidate prevalence;
- AUC;
- average precision;
- within-group/year top-decile rate and lift;
- Big4-minus-non-Big4 differences;
- issuer-cluster bootstrap confidence intervals.

Outcome-specific scores remain unchanged:

- absolute abnormal CFO for any candidate;
- positive abnormal CFO for CFO decreases and CFF-down candidates;
- negative abnormal CFO for CFO increases and CFI-up candidates.

## Interaction model

For each outcome, a logistic model includes standardized score, Big4, and score-by-Big4. Controls are restricted to fields already available in the locked case table. The default specification includes log lagged assets, preliminary CFO, and year, exchange, and industry fixed effects. Standard errors are clustered by issuer.

The score-by-Big4 term is the focal coefficient. The Big4 main effect alone is not interpreted as audit quality because auditor choice is endogenous.

## Selection and switches

The pipeline reports standardized mean differences for size, preliminary CFO, abnormal CFO, exchange, industry, and year composition. It also counts consecutive-year switches between Big4 and non-Big4. These diagnostics delimit, but do not eliminate, auditor-client selection.

## Outputs

- `cfs_auditor_source_status.csv`
- `cfs_auditor_name_mapping.csv`
- `cfs_auditor_firm_year.csv`
- `cfs_auditor_analysis_sample.csv.gz`
- `cfs_auditor_regime_coverage.csv`
- `cfs_auditor_regime_metrics.csv`
- `cfs_auditor_regime_metric_differences.csv`
- `cfs_auditor_regime_bootstrap.csv`
- `cfs_auditor_regime_interaction.csv`
- `cfs_auditor_regime_balance.csv`
- `cfs_auditor_switch_events.csv`
- `cfs_auditor_switch_summary.csv`
- `cfs_auditor_regime_status.csv`
- `CFS_AUDITOR_REGIME_REPORT.md`

## Interpretation

A stronger score relationship among Big4 clients indicates that the observed correction target is conditional on the audit regime producing that correction. It does not prove that Big4 auditors causally improve reporting. A null difference supports transportability of the proxy across auditor tiers. A prevalence difference without a score interaction is primarily a client-composition result.
