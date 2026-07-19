# CFS offset and incentive deep-dive protocol

## Purpose

This stage moves beyond identifying identity-consistent CFO reclassification candidates. It asks four narrower questions:

1. Where is the CFO change offset: CFI, CFF, FX, cash-change differences, or a mixed combination?
2. Are chronic reclassifiers consistently revised in one direction or revised in both directions over time?
3. Do distress and cash-flow-reporting incentives predict audited CFO decreases, rather than merely predicting the existence of a reclassification?
4. Does the cash-flow-anchor versus balance-sheet-anchor sign reversal survive on the identical complete-case sample?

## Identification boundaries

Passing the cash-flow identities establishes internal numerical consistency. It does not prove that the vendor's semantic mapping of every cash-flow line is correct. Accordingly, the pipeline retains the term `reclassification candidate` until a stratified source-document sample is checked.

A CFI-dominant offset does not identify the underlying transaction. Related-party lending, asset purchases, deposits, and other investing cash flows require line-item data or manual verification. A CFF-dominant offset similarly does not identify interest, dividends, borrowings, or owner transactions without lower-level cash-flow items.

## Offset allocation

For each firm-year candidate, the required non-CFO offset is:

```text
required_offset = -delta_CFO
```

The observed terms are:

```text
delta_CFI + delta_CFF + delta_FX - delta_cash_change
```

Their sum should equal the required offset for an identity-consistent case. Absolute shares identify the dominant destination; cases with no component reaching the configured threshold are classified as mixed.

## Chronicity

An issuer is a chronic reclassifier when it is a candidate in at least four years and in at least 75% of its observed analysis years. Direction is classified separately:

- mostly audited CFO increase;
- mostly audited CFO decrease;
- bidirectional;
- sparse single-direction.

Bidirectionality supports a recurring reporting-process or classification-policy problem. Persistent audited CFO decreases are more consistent with an incentive-driven upward preliminary-CFO subset, but remain descriptive without a causal design.

## Incentive tests

The pipeline uses pre-audit variables only:

- loss;
- negative CFO;
- CFO close to zero;
- current ratio below one;
- low cash;
- ROA;
- short-term debt scaled by lagged assets;
- size.

It estimates firm-clustered linear probability models for candidate incidence and for an audited CFO decrease among candidates. The second outcome is the relevant test for upward preliminary-CFO shifting. Predicting candidate incidence alone is insufficient because complexity or weak reporting systems can generate two-sided reclassifications.

## Audit-quality metadata

Big4 and audit-opinion splits require a separate file keyed by `issuer_ticker` and `fiscal_year`. When that file is absent or lacks the necessary fields, the pipeline returns `NOT_EVALUATED`; it does not infer auditor identity from financial-statement data.

Any Big4 association remains descriptive until client selection and pre-audit reporting quality are addressed.

## Source-document sample

The verification sample includes:

- preliminary identity failures repaired in the audited version;
- insufficient-component cases;
- CFI-, CFF-, mixed-, upward-, downward-, chronic-, and non-chronic candidates;
- extreme CFO changes.

For each selected case, verify the reported values and labels for CFO, CFI, CFF, FX effects, net cash change, and beginning/end cash in both versions.
