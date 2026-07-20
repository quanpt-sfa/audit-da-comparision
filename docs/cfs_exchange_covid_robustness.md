# Exchange and COVID-period robustness

## Scope

These analyses use the focal `earnings_working_capital` abnormal-CFO score on the `common_primary_models` and `analysis_core` issuer-year sample. They do not refit or select the expected-CFO model using robustness outcomes.

The shared TT200 contract remains authoritative:

- source/target period: 2015-2025;
- training begins in 2015;
- theoretical out-of-sample tests begin in 2016;
- effective test years depend on the prespecified minimum-training gate.

## Exchange robustness

The analysis normalizes listing boards to `HOSE`, `HNX`, and `UPCOM` and produces:

- within-exchange prevalence, AUC, average precision, top-decile rate and lift;
- all pairwise exchange differences;
- issuer-cluster bootstrap intervals for pairwise differences;
- leave-one-exchange-out sensitivity relative to pooled results;
- score-by-HNX and score-by-UPCOM logistic interactions with HOSE as reference, issuer-clustered standard errors, year fixed effects and industry fixed effects.

The exchange analysis tests transportability. It does not estimate a causal effect of listing venue because firms select into exchanges and differ in size, age, governance and reporting capacity.

## COVID-period robustness

The primary temporal regimes are:

- pre-COVID: 2016-2019;
- COVID period: 2020-2021;
- recovery: 2022-2025.

Prespecified sensitivity windows are:

- 2020 only;
- 2020-2021;
- 2020-2022.

Outputs include regime-specific metrics, shock-minus-non-shock differences, issuer-cluster bootstrap intervals and score-by-COVID-period interactions. Full fiscal-year fixed effects absorb period-level intercept differences; the focal interaction tests whether the criterion-validity slope changes during the configured COVID period.

These are temporal-regime robustness checks, not causal pandemic effects. Enforcement intensity, reporting deadlines, firm composition, macroeconomic stress and audit practices also changed over time.

## Decision rule

A result is considered robust when:

1. within-exchange score direction is consistent;
2. no single exchange drives the pooled result in leave-one-out tests;
3. pairwise exchange intervals do not show a material reversal;
4. discrimination remains present in pre-COVID, COVID and recovery regimes;
5. COVID slope conclusions are qualitatively stable across the primary and at least one alternative window.
