from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.predictive_validity import (
    PredictiveValiditySettings,
    build_accrual_quality_cases,
    build_predictive_cases,
    run_predictive_validity,
)


def _synthetic_panel(n_issuers: int = 48, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    industries = ["Industrials", "Consumer", "Technology"]
    for issuer_index in range(n_issuers):
        ticker = f"F{issuer_index:03d}"
        latent_profitability = 0.03 + rng.normal(0, 0.01)
        assets = 100.0 + rng.uniform(0, 50)
        previous_assets = assets
        previous_current_assets = 0.50 * assets
        previous_cash = 0.08 * assets
        previous_current_liabilities = 0.30 * assets
        previous_debt = 0.10 * assets
        previous_tax = 0.02 * assets

        for year in range(2015, 2026):
            assets = previous_assets * (1.0 + rng.normal(0.05, 0.02))
            latent_profitability = (
                0.65 * latent_profitability + rng.normal(0.015, 0.007)
            )
            audited_roa = latent_profitability + rng.normal(0, 0.004)
            audited_cfo = 0.70 * audited_roa + rng.normal(0.015, 0.005)

            current_assets = 0.50 * assets + rng.normal(0, 2)
            cash = 0.08 * assets + rng.normal(0, 1)
            current_liabilities = 0.30 * assets + rng.normal(0, 2)
            debt = 0.10 * assets + rng.normal(0, 1)
            tax = 0.02 * assets + rng.normal(0, 0.3)
            depreciation = 0.03 * assets
            dca = current_assets - previous_current_assets
            dcash = cash - previous_cash
            dcl = current_liabilities - previous_current_liabilities
            dstd = debt - previous_debt
            dtax = tax - previous_tax
            working_capital_accrual = (dca - dcash) - (dcl - dstd - dtax)

            for state in ("unaudited", "audited"):
                noise_scale = 0.025 if state == "unaudited" else 0.004
                roa = audited_roa + rng.normal(0, noise_scale)
                cfo_scaled = audited_cfo + rng.normal(0, noise_scale)
                lag_assets = previous_assets if year > 2015 else np.nan
                pat = roa * lag_assets if np.isfinite(lag_assets) else np.nan
                cfo = cfo_scaled * lag_assets if np.isfinite(lag_assets) else np.nan
                state_wca = working_capital_accrual + rng.normal(
                    0, noise_scale * previous_assets
                )
                rows.append(
                    {
                        "issuer_ticker": ticker,
                        "fiscal_year": year,
                        "audit_status": state,
                        "financial_flag": False,
                        "icb_l1": industries[issuer_index % len(industries)],
                        "lag_assets": lag_assets,
                        "pat": pat,
                        "cfo": cfo,
                        "roa": roa if year > 2015 else np.nan,
                        "cfo_scaled": cfo_scaled if year > 2015 else np.nan,
                        "ta_balance_sheet": (
                            state_wca - depreciation if year > 2015 else np.nan
                        ),
                        "depreciation": depreciation if year > 2015 else np.nan,
                    }
                )

            previous_assets = assets
            previous_current_assets = current_assets
            previous_cash = cash
            previous_current_liabilities = current_liabilities
            previous_debt = debt
            previous_tax = tax
    return pd.DataFrame(rows)


def _settings() -> PredictiveValiditySettings:
    return PredictiveValiditySettings(
        minimum_train_rows=40,
        aq_minimum_train_rows=40,
        bootstrap_draws=30,
        pooled_specifications=("canonical", "year_industry_fe"),
    )


def test_future_targets_are_exact_audited_t_plus_one() -> None:
    panel = _synthetic_panel(n_issuers=8)
    settings = _settings()
    cases = build_predictive_cases(panel, settings)

    assert cases.fiscal_year.min() == 2016
    assert cases.fiscal_year.max() == 2024
    assert cases.outcome_fiscal_year.eq(cases.fiscal_year + 1).all()

    audited = panel.loc[
        panel.audit_status.eq("audited"),
        ["issuer_ticker", "fiscal_year", "roa", "cfo_scaled"],
    ].copy()
    audited["fiscal_year"] -= 1
    checked = cases.merge(
        audited,
        on=["issuer_ticker", "fiscal_year"],
        how="left",
        validate="one_to_one",
    )
    np.testing.assert_allclose(checked.future_roa_audited, checked.roa)
    np.testing.assert_allclose(
        checked.future_cfo_audited,
        checked.cfo_scaled,
    )


def test_accrual_quality_cases_require_audited_lag_and_lead_cfo() -> None:
    panel = _synthetic_panel(n_issuers=8)
    settings = _settings()
    cases = build_accrual_quality_cases(panel, settings)

    assert cases.fiscal_year.min() == 2017
    assert cases.fiscal_year.max() == 2024
    assert cases[
        [
            "wca_scaled_pre",
            "wca_scaled_audited",
            "cfo_lag_audited",
            "cfo_scaled_pre",
            "cfo_scaled_audited",
            "cfo_lead_audited",
        ]
    ].notna().all().all()
    assert not cases.duplicated(["issuer_ticker", "fiscal_year"]).any()


def test_pipeline_uses_common_samples_and_reports_state_differences() -> None:
    panel = _synthetic_panel()
    settings = _settings()
    outputs = run_predictive_validity(panel, settings)

    assert len(outputs["predictive_validity_oos_summary"]) == 8
    assert len(outputs["predictive_validity_oos_state_differences"]) == 12
    assert len(outputs["accrual_quality_summary"]) == 2
    assert len(outputs["accrual_quality_state_differences"]) == 3

    coefficients = outputs["predictive_validity_coefficients"]
    differences = coefficients.loc[
        coefficients.contrast.eq("audited_minus_pre")
    ]
    assert set(differences.test) == {
        "earnings_persistence",
        "earnings_to_future_cfo",
        "cfo_persistence",
        "earnings_cfo_horse_race",
    }

    oos = outputs["predictive_validity_oos_state_differences"]
    audited_improvements = oos.loc[
        oos.metric.eq("rmse"),
        "estimate",
    ]
    assert audited_improvements.lt(0).all()

    aq = outputs["accrual_quality_state_differences"]
    assert aq.estimate.lt(0).all()


def test_oos_predictions_never_use_same_or_future_predictor_years_for_training() -> None:
    panel = _synthetic_panel()
    settings = _settings()
    outputs = run_predictive_validity(panel, settings)
    folds = outputs["predictive_validity_oos_folds"]
    estimated = folds.loc[folds.status.eq("estimated")]

    assert not estimated.empty
    assert estimated.test_year.ge(settings.oos_test_start_year).all()
    assert estimated.train_years.ge(settings.minimum_train_years).all()
    assert estimated.train_rows.ge(settings.minimum_train_rows).all()
    assert estimated.training_outcome_bounds.notna().all()
