# Exchange and COVID-period robustness

## Scope

These analyses use the focal `earnings_working_capital` abnormal-CFO score on the `common_primary_models` and `analysis_core` issuer-year sample. They do not refit or select the expected-CFO model using robustness outcomes.

The shared TT200 contract remains authoritative:

- source/target period: 2015-2025;
- training begins in 2015;
- no raw or lag lookback before 2015 is permitted;
- theoretical out-of-sample tests begin in 2016;
- effective test years depend on the prespecified minimum-training gate.

## Independent entrypoints

Run the exchange analysis only:

```bash
python scripts/25_analyze_exchange_robustness.py \
  --config config/cfs_shifting_validation.yaml
```

Run the COVID-period analysis only:

```bash
python scripts/26_analyze_covid_robustness.py \
  --config config/cfs_shifting_validation.yaml
```

The prior combined command remains available as a compatibility wrapper and delegates to both scripts:

```bash
python scripts/25_analyze_exchange_covid_robustness.py \
  --config config/cfs_shifting_validation.yaml
```

For smoke runs, either script accepts `--bootstrap-repetitions`. Both also accept `--bootstrap-seed`, `--case-table`, and a comma-separated `--outcomes` override. The exchange script additionally accepts `--exchanges` and `--reference-exchange`. The COVID script additionally accepts `--pre-years`, `--shock-years`, and `--recovery-years`.

## Exchange robustness

The analysis normalizes listing boards to `HOSE`, `HNX`, and `UPCOM` and produces:

- sample coverage and positive counts by exchange;
- within-exchange prevalence, AUC, average precision, top-decile rate and lift;
- all pairwise exchange differences;
- issuer-cluster bootstrap intervals for pairwise differences;
- leave-one-exchange-out sensitivity relative to pooled results;
- score-by-HNX and score-by-UPCOM logistic interactions with HOSE as reference, issuer-clustered standard errors, year fixed effects and industry fixed effects;
- a standalone exchange gate and TT200 window-status artifact.

The exchange analysis tests transportability. It does not estimate a causal effect of listing venue because firms select into exchanges and differ in size, age, governance and reporting capacity.

Primary exchange artifacts are:

- `cfs_exchange_robustness_sample.csv`;
- `cfs_exchange_sample_coverage.csv`;
- `cfs_exchange_robustness_metrics.csv`;
- `cfs_exchange_pairwise_differences.csv`;
- `cfs_exchange_cluster_bootstrap.csv`;
- `cfs_exchange_leave_one_out.csv`;
- `cfs_exchange_interactions.csv`;
- `cfs_exchange_robustness_status.csv`.

## COVID-period robustness

The primary temporal regimes are:

- pre-COVID: 2016-2019;
- COVID period: 2020-2021;
- recovery: 2022-2025.

Prespecified sensitivity windows are:

- 2020 only;
- 2020-2021;
- 2020-2022.

Outputs include sample coverage by regime, regime-specific metrics, COVID/recovery-minus-pre-COVID differences, alternative-window sensitivity, issuer-cluster bootstrap intervals and score-by-COVID-period interactions. Full fiscal-year fixed effects absorb period-level intercept differences; the focal term is `score_x_covid_shock`, which tests whether the criterion-validity slope changes during the configured COVID period. The COVID main dummy is deliberately omitted from this specification.

These are temporal-regime robustness checks, not causal pandemic effects. Enforcement intensity, reporting deadlines, firm composition, macroeconomic stress and audit practices also changed over time.

Primary COVID artifacts are:

- `cfs_covid_robustness_sample.csv`;
- `cfs_covid_sample_coverage.csv`;
- `cfs_covid_regime_metrics.csv`;
- `cfs_covid_regime_differences.csv`;
- `cfs_covid_window_sensitivity.csv`;
- `cfs_covid_cluster_bootstrap.csv`;
- `cfs_covid_interactions.csv`;
- `cfs_covid_robustness_status.csv`.

## Shared status artifacts

Each standalone script updates only its own gate while preserving the other analysis results in:

- `cfs_regime_robustness_status.csv`;
- `cfs_regime_robustness_window_status.csv`;
- `cfs_completion_gate_status.csv`.

This allows either robustness analysis to be rerun independently without erasing the status of the other analysis.

## Decision rule

A result is considered robust when:

1. within-exchange score direction is consistent;
2. no single exchange drives the pooled result in leave-one-out tests;
3. pairwise exchange intervals do not show a material reversal;
4. discrimination remains present in pre-COVID, COVID and recovery regimes;
5. COVID slope conclusions are qualitatively stable across the primary and at least one alternative window.
