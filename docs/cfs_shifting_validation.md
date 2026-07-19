# Observed validation of cash-flow classification-shifting proxies

## Purpose

The existing diagnostics observe preliminary-to-audited cash-flow reclassifications. This stage asks whether indirect classification-shifting proxies used in prior literature identify those observed transitions out of time.

Observed revision is treated as a validation outcome, not as proof of managerial intent. The analysis distinguishes:

1. any identity-consistent reclassification candidate;
2. audited CFO decreases;
3. audited CFO increases;
4. CFF-dominant CFO decreases;
5. CFI-dominant CFO increases.

A proxy that predicts any candidate but cannot distinguish direction or channel is a reporting-instability screen, not a validated manipulation proxy.

## Outcome-specific scores

A single signed residual cannot validly rank all five outcomes. The protocol therefore uses:

- `abs(abnormal_cfo_proxy)` for any reclassification;
- `abnormal_cfo_proxy` for audited CFO decreases and CFF-down candidates;
- `-abnormal_cfo_proxy` for audited CFO increases and CFI-up candidates.

Top-decile rates and lifts are recomputed from the outcome-specific score within each test year.

## Proxy families and baselines

All expected-CFO models use unaudited information and expanding prior-year training samples.

- `sales_level_only`: inverse assets and preliminary sales.
- `roychowdhury_sales`: inverse assets, sales and sales changes.
- `earnings_conditioned`: adds preliminary earnings.
- `earnings_working_capital`: conditions on preliminary earnings, sales and receivables movements and loss status.

Simple baselines are evaluated alongside the expected-CFO models:

- raw preliminary CFO scaled by lagged assets;
- within-year CFO percentile;
- deviation from the issuer's prior-year median CFO.

Incremental validity is measured relative to raw CFO on the same firm-year sample.

## Common-sample rule

Model comparisons use the intersection of firm-years available for every prespecified comparison model. Model-specific samples remain available only as coverage diagnostics. AUC, average precision and lift are not compared across samples with different prevalence.

## Fold robustness

Each expanding-window expected-CFO fold reports:

- raw RMSE;
- 1%/99% winsorized RMSE;
- RMSE after excluding the largest 1% absolute residuals;
- MAE and median absolute error;
- p95 and p99 absolute error;
- maximum absolute error and the responsible issuer.

Raw RMSE alone is not interpreted when the residual distribution is heavy-tailed.

## Sample restrictions

The protocol reports full and restricted samples for:

- HOSE/HNX/UPCoM only;
- valid ticker format;
- lagged-assets floor;
- exclusion of available scale/scope anomaly flags;
- non-financial firms when industry metadata or a financial-firm flag exists;
- an `analysis_core` sample combining all restrictions that can be evaluated.

Unavailable industry or scale/scope metadata is reported as `NOT_EVALUATED`; it is never silently assumed.

## Detailed CFS line items

The raw long file is scanned in chunks. Regex rules map source items to conservative concepts covering operating, investing and financing cash flows. The mapping explicitly separates loan recoveries from investment disposals, interest receipts from dividend receipts, debt repayments from lease-principal payments, and owner contributions from share repurchases.

The pipeline exports every unmapped or ambiguous source item. Institutional interpretations are prohibited until high-coverage mapping-review items and selected source documents are checked.

Mapped line-item changes are reconciled to aggregate CFI and CFF changes. The primary top-contributor table contains reclassification candidates only. A separate all-resolution table is retained as an audit trail.

## Decision rules

- Predicting CFF-downward revisions supports validation of an upward-CFO shifting proxy.
- Predicting CFI-upward revisions with the inverse score supports a bidirectional classification-reliability construct.
- Predicting any revision requires the absolute residual; a signed score mechanically cancels the two tails.
- Expected-CFO models must improve on raw CFO and percentile baselines on the common sample.
- A named line-item mechanism requires material reconciliation coverage and source-document confirmation.
- Poor performance after the listed/core restrictions indicates limited transportability rather than evidence against all classification revisions.
