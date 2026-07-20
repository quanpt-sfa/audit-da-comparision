from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.cfs_robustness_runners import (
    run_covid_robustness,
    run_exchange_robustness,
)


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    rows = []
    exchanges = ["HOSE", "HNX", "UPCoM"]
    for issuer_index in range(60):
        ticker = f"F{issuer_index:03d}"
        exchange = exchanges[issuer_index % 3]
        industry = "A" if issuer_index % 2 == 0 else "B"
        for year in range(2016, 2026):
            abnormal = rng.normal()
            shock = year in {2020, 2021}
            probability = 1.0 / (
                1.0
                + np.exp(
                    -(
                        -2.0
                        + 0.75 * abs(abnormal)
                        + 0.20 * shock * abs(abnormal)
                    )
                )
            )
            any_candidate = rng.binomial(1, probability)
            down = any_candidate * rng.binomial(1, 0.5)
            up = any_candidate - down
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": year,
                    "proxy_model": "earnings_working_capital",
                    "sample_mode": "common_primary_models",
                    "sample_restriction": "analysis_core",
                    "raw_exchange": exchange,
                    "industry_name": industry,
                    "lag_assets": float(np.exp(rng.normal(25, 1))),
                    "pre_cfo_scaled": rng.normal(0.05, 0.10),
                    "abnormal_cfo_proxy": abnormal,
                    "any_candidate": any_candidate,
                    "audited_cfo_decrease": down,
                    "audited_cfo_increase": up,
                    "cff_down_candidate": down,
                    "cfi_up_candidate": up,
                }
            )
    return pd.DataFrame(rows)


def _settings() -> dict:
    return {
        "proxy_model": "earnings_working_capital",
        "sample_mode": "common_primary_models",
        "sample_restriction": "analysis_core",
        "outcomes": ["any_candidate"],
        "exchange_groups": ["HOSE", "HNX", "UPCOM"],
        "exchange_reference": "HOSE",
        "minimum_group_rows": 20,
        "minimum_group_positives": 2,
        "bootstrap_repetitions": 2,
        "bootstrap_seed": 19,
        "minimum_interaction_rows": 50,
        "minimum_interaction_positives": 2,
        "exchange_fixed_effects": ["fiscal_year", "industry_name"],
        "covid_fixed_effects": [
            "fiscal_year",
            "raw_exchange",
            "industry_name",
        ],
        "covid": {
            "pre_years": [2016, 2017, 2018, 2019],
            "primary_shock_years": [2020, 2021],
            "recovery_years": [2022, 2023, 2024, 2025],
            "alternative_shock_windows": {
                "COVID_2020_ONLY": [2020],
                "COVID_2020_2021": [2020, 2021],
                "COVID_2020_2022": [2020, 2021, 2022],
            },
        },
    }


def test_exchange_runner_is_independent_from_covid_outputs() -> None:
    tables = run_exchange_robustness(_sample(), _settings())

    assert "cfs_exchange_robustness_metrics" in tables
    assert "cfs_covid_regime_metrics" not in tables
    assert set(tables["cfs_exchange_robustness_metrics"]["group"]) == {
        "HOSE",
        "HNX",
        "UPCOM",
    }
    assert set(tables["cfs_exchange_leave_one_out"]["excluded_exchange"]) == {
        "HOSE",
        "HNX",
        "UPCOM",
    }
    assert set(tables["cfs_exchange_robustness_status"]["gate"]) == {
        "within_exchange_robustness"
    }
    assert tables["cfs_exchange_robustness_sample"]["fiscal_year"].min() == 2016


def test_covid_runner_is_independent_and_uses_identified_slope_interaction() -> None:
    tables = run_covid_robustness(_sample(), _settings())

    assert "cfs_covid_regime_metrics" in tables
    assert "cfs_exchange_robustness_metrics" not in tables
    assert set(tables["cfs_covid_regime_metrics"]["group"]) == {
        "PRE_COVID",
        "COVID_SHOCK",
        "RECOVERY",
    }
    assert set(tables["cfs_covid_robustness_sample"]["covid_regime"]) == {
        "PRE_COVID",
        "COVID_SHOCK",
        "RECOVERY",
    }
    interactions = tables["cfs_covid_interactions"]
    assert "covid_shock" not in set(interactions["term"])
    assert set(interactions.loc[interactions["focal_term"].eq(True), "term"]) == {
        "score_x_covid_shock"
    }
    assert set(tables["cfs_covid_robustness_status"]["gate"]) == {
        "covid_period_robustness"
    }
