# Observed validation of cash-flow classification-shifting proxies

## Purpose

The diagnostics observe preliminary-to-audited cash-flow reclassifications. This stage asks whether indirect classification-shifting proxies identify those observed transitions out of time.

Observed revision is a validation outcome, not proof of managerial intent. The analysis distinguishes:

1. any identity-consistent reclassification candidate;
2. audited CFO decreases;
3. audited CFO increases;
4. CFF-dominant CFO decreases;
5. CFI-dominant CFO increases.

A proxy that predicts any candidate but cannot distinguish direction or channel is a reporting-instability screen, not a validated manipulation proxy.

## Estimation population

Expected-CFO models are estimated only on the locked research population before firm history or rolling folds are constructed:

- known non-financial ICB classification;
- HOSE, HNX or UPCoM;
- valid ticker format;
- positive lagged assets.

Unknown-industry and financial observations are excluded before fitting, not merely removed from the final evaluation sample. The pipeline writes `cfs_expected_cfo_estimation_sample_status.csv` as evidence for this gate.

## Outcome-specific scores

A single signed residual cannot rank all five outcomes. The protocol uses:

- `abs(abnormal_cfo_proxy)` for any reclassification;
- `abnormal_cfo_proxy` for audited CFO decreases and CFF-down candidates;
- `-abnormal_cfo_proxy` for audited CFO increases and CFI-up candidates.

Top-decile rates and lifts are recomputed from the outcome-specific score within each test year.

## Proxy families and baselines

All expected-CFO models use unaudited information and expanding prior-year training samples.

- `sales_level_only`: inverse assets and preliminary sales;
- `roychowdhury_sales`: inverse assets, sales and sales changes;
- `earnings_conditioned`: adds preliminary earnings;
- `earnings_working_capital`: conditions on earnings, sales and receivables movements and loss status;
- `earnings_working_capital_history`: adds the issuer's prior-only median CFO to the EWC specification.

Simple baselines are:

- raw preliminary CFO scaled by lagged assets;
- within-year CFO percentile;
- deviation from the issuer's prior-year median CFO.

The nested EWC+history model is compared with EWC on the identical `common_all_models` firm-year sample in `cfs_history_incremental_comparison.csv`.

## Common-sample rules

`common_primary_models` excludes history requirements and defines the main estimand. `common_all_models` requires both the standalone history baseline and the nested EWC+history model. AUC, average precision and lift are never compared across different firm-year samples without an explicit sample-sensitivity table.

## Fold robustness

Each expanding-window expected-CFO fold reports raw and robust errors: RMSE, 1%/99% winsorized RMSE, RMSE excluding the largest 1% absolute residuals, MAE, median absolute error, p95/p99 errors and the maximum-error issuer.

## Scale and scope

Additional scale/scope screening is waived by design. Preliminary and audited records are provided under the same source-controlled monetary unit, consolidated scope and reporting-period convention. This maintained data-design assumption is disclosed in the paper and reported as `WAIVED_BY_DESIGN`; it does not remove observations.

## Detailed CFS line items

The raw long file is scanned in chunks and mapped to conservative operating, investing and financing concepts. Line-item reconciliation is recomputed directly on:

- `common_primary_models + analysis_core`;
- `common_all_models + analysis_core`.

Main mechanism claims must use the common-primary/core outputs rather than full-universe contributor tables.

The retained source records preserve the original line-item labels and values from the preliminary and audited statements. These records were verified during data construction. Accordingly, the validation pipeline relies on source-record reconciliation and does not create a separate PDF-verification requirement or manifest.

## Completion gates

`cfs_completion_gate_status.csv` records:

- non-financial estimation population;
- nested history incremental test;
- common-primary/core source-record reconciliation;
- scale/scope waiver.

## Decision rules

- Expected-CFO results are admissible only if the estimation-population gate passes.
- Predicting CFF-downward revisions supports validation of an upward-CFO shifting proxy.
- Predicting CFI-upward revisions with the inverse score supports a bidirectional classification-reliability construct.
- The nested history model is retained only if it improves EWC on the identical all-model sample.
- A named line-item mechanism requires common-primary/core reconciliation coverage using the retained and previously verified source-record labels and values.
