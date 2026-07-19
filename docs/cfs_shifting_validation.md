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

## Proxy families

All proxy models use unaudited data and are estimated with expanding prior-year training samples.

- `roychowdhury_sales`: expected CFO from inverse assets, sales and sales changes.
- `earnings_conditioned`: adds preliminary earnings to separate CFO from earnings level.
- `earnings_working_capital`: conditions on preliminary earnings, sales and receivables movements and loss status.

The main validation statistics are AUC, average precision, top-decile lift and temporal stability.

## Detailed CFS line items

The raw long file is scanned in chunks. Regex rules map source items to conservative concepts covering operating, investing and financing cash flows. The pipeline exports every unmapped or ambiguous source item. Institutional interpretations are prohibited until high-coverage mapping-review items and selected source documents are checked.

Mapped line-item changes are reconciled to aggregate CFI and CFF changes. Reconciliation coverage determines whether a specific mechanism such as borrowing repayment, interest classification, loans advanced or asset purchases can be named.

## Decision rules

- Predicting CFF-downward revisions supports validation of an upward-CFO shifting proxy.
- Similar performance for CFF-downward and CFI-upward outcomes indicates generic classification instability.
- Predicting any revision but not direction supports reliability screening only.
- Poor prediction of observed revisions indicates that expected-CFO residuals and actual within-cycle reclassification are different constructs.
