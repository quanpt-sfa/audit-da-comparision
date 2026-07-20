from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.cfs_regime_robustness import (
    assign_covid_regime,
    normalize_exchange,
    prepare_regime_sample,
    run_regime_robustness,
)


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    rows = []
    exchanges = ["HOSE", "HNX", "UPCoM"]
    for issuer_index in range(90):
        ticker = f"F{issuer_index:03d}"
        exchange = exchanges[issuer_index % 3]
        industry = "IND_A" if issuer_index % 2 == 0 else "IND_B"
        for year in range(2016, 2026):
            abnormal = rng.normal()
            shock = year in {2020, 2021}
            exchange_effect = {"HOSE": 0.15, "HNX": 0.0, "UPCoM": -0.10}[exchange]
            probability = 1.0 / (
                1.0
                + np.exp(
                    -(
                        -2.2
                        + 0.70 * abs(abnormal)
                        + 0.20 * shock * abs(abnormal)
                        + exchange_effect
                    )
                )
            )
            any_candidate = rng.binomial(1, probability)
            down = any_candidate * rng.binomial(1, 0.52)
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
                    "cff_down_candidate": down * rng.binomial(1, 0.60),
                    "cfi_up_candidate": up * rng.binomial(1, 0.55),
                }
            )
    return pd.DataFrame(rows)


def _settings() -> dict:
    return {
        "proxy_model": "earnings_working_capital",
        "sample_mode": "common_primary_models",
        "sample_restriction": "analysis_core",
        "exchange_groups": ["HOSE", "HNX", "UPCOM"],
        "exchange_reference": "HOSE",
        "minimum_group_rows": 50,
        "minimum_group_positives": 5,
        "bootstrap_repetitions": 5,
        "bootstrap_seed": 19,
        "minimum_interaction_rows": 100,
        "minimum_interaction_positives": 5,
        "exchange_fixed_effects": ["fiscal_year", "industry_name"],
        "covid_fixed_effects": ["fiscal_year", "raw_exchange", "industry_name"],
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


def test_exchange_normalization_preserves_upcom() -> None:
    assert normalize_exchange("HOSE") == "HOSE"
    assert normalize_exchange("HSX") == "HOSE"
    assert normalize_exchange("HNX") == "HNX"
    assert normalize_exchange("UPCoM") == "UPCOM"
    assert normalize_exchange(None) == "UNKNOWN"


def test_covid_regime_assignment_uses_prespecified_windows() -> None:
    years = pd.Series([2015, 2016, 2019, 2020, 2021, 2022, 2025, 2026])
    regime = assign_covid_regime(
        years,
        [2016, 2017, 2018, 2019],
        [2020, 2021],
        [2022, 2023, 2024, 2025],
    )
    assert regime.tolist() == [
        "OUTSIDE_CONFIGURED_REGIMES",
        "PRE_COVID",
        "PRE_COVID",
        "COVID_SHOCK",
        "COVID_SHOCK",
        "RECOVERY",
        "RECOVERY",
        "OUTSIDE_CONFIGURED_REGIMES",
    ]


def test_prepare_sample_uses_focal_common_primary_rows() -> None:
    cases = _sample()
    extra = cases.iloc[[0]].copy()
    extra["proxy_model"] = "raw_cfo_level"
    cases = pd.concat([cases, extra], ignore_index=True)
    sample = prepare_regime_sample(cases, _settings())
    assert sample["proxy_model"].eq("earnings_working_capital").all()
    assert sample.duplicated(["issuer_ticker", "fiscal_year"]).sum() == 0
    assert set(sample["exchange_group"]) == {"HOSE", "HNX", "UPCOM"}


def test_full_robustness_produces_exchange_and_covid_outputs() -> None:
    tables = run_regime_robustness(_sample(), _settings())

    exchange = tables["cfs_exchange_robustness_metrics"]
    assert set(exchange["group"]) == {"HOSE", "HNX", "UPCOM"}
    assert set(exchange["outcome"]) == {
        "any_candidate",
        "audited_cfo_decrease",
        "audited_cfo_increase",
        "cff_down_candidate",
        "cfi_up_candidate",
    }

    leave_out = tables["cfs_exchange_leave_one_out"]
    assert set(leave_out["excluded_exchange"]) == {"HOSE", "HNX", "UPCOM"}

    covid = tables["cfs_covid_regime_metrics"]
    assert set(covid["group"]) == {"PRE_COVID", "COVID_SHOCK", "RECOVERY"}
    assert set(tables["cfs_covid_window_sensitivity"]["shock_window"]) == {
        "COVID_2020_ONLY",
        "COVID_2020_2021",
        "COVID_2020_2022",
    }

    exchange_focal = tables["cfs_exchange_interactions"].query("focal_term == True")
    assert {"score_x_exchange_HNX", "score_x_exchange_UPCOM"}.issubset(
        set(exchange_focal["term"])
    )
    covid_focal = tables["cfs_covid_interactions"].query("focal_term == True")
    assert set(covid_focal["term"]) == {"score_x_covid_shock"}

    status = tables["cfs_regime_robustness_status"]
    assert set(status["gate"]) == {
        "within_exchange_robustness",
        "covid_period_robustness",
    }
    assert set(status["status"]).issubset({"PASS", "PARTIALLY_EVALUATED"})
