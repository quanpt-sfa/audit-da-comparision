from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.yearly_auc_heterogeneity import run_yearly_auc_heterogeneity


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    rows: list[dict] = []
    for issuer_index in range(160):
        issuer_effect = rng.normal(scale=0.20)
        for year in range(2018, 2023):
            score = rng.normal()
            slope = 0.15 if year <= 2019 else 1.20
            probability = 1.0 / (
                1.0 + np.exp(-(-1.60 + issuer_effect + slope * abs(score)))
            )
            rows.append(
                {
                    "issuer_ticker": f"F{issuer_index:03d}",
                    "fiscal_year": year,
                    "proxy_model": "earnings_working_capital",
                    "sample_mode": "common_primary_models",
                    "sample_restriction": "analysis_core",
                    "raw_exchange": ["HOSE", "HNX", "UPCOM"][issuer_index % 3],
                    "industry_name": "A" if issuer_index % 2 == 0 else "B",
                    "lag_assets": float(np.exp(rng.normal(25.0, 0.8))),
                    "pre_cfo_scaled": rng.normal(0.05, 0.08),
                    "abnormal_cfo_proxy": score,
                    "any_candidate": rng.binomial(1, probability),
                }
            )
    return pd.DataFrame(rows)


def _settings() -> dict:
    return {
        "yearly_auc_heterogeneity": {
            "proxy_model": "earnings_working_capital",
            "sample_mode": "common_primary_models",
            "sample_restriction": "analysis_core",
            "outcomes": ["any_candidate"],
            "minimum_year_positives": 10,
            "minimum_year_negatives": 10,
            "bootstrap_repetitions": 30,
            "bootstrap_seed": 19,
            "minimum_interaction_rows": 200,
            "minimum_interaction_positives": 20,
            "interaction_max_iter": 200,
            "continuous_controls": ["lag_assets", "pre_cfo_scaled"],
            "fixed_effects": ["raw_exchange", "industry_name"],
        }
    }


def test_yearly_auc_generalized_q_and_score_year_wald_are_produced() -> None:
    tables = run_yearly_auc_heterogeneity(_sample(), _settings())

    yearly = tables["cfs_yearly_auc_metrics"]
    assert yearly["status"].eq("OK").all()
    assert set(yearly["fiscal_year"]) == {2018, 2019, 2020, 2021, 2022}

    covariance = tables["cfs_yearly_auc_covariance"]
    assert len(covariance) == 25
    assert covariance["bootstrap_draws"].min() > 0

    q_test = tables["cfs_yearly_auc_generalized_q"]
    assert len(q_test) == 1
    assert q_test.iloc[0]["df"] == 4
    assert np.isfinite(q_test.iloc[0]["chi_square"])

    interaction_test = tables["cfs_score_by_year_joint_tests"]
    assert len(interaction_test) == 1
    assert interaction_test.iloc[0]["df"] == 4
    assert np.isfinite(interaction_test.iloc[0]["chi_square"])
    assert tables["cfs_yearly_auc_heterogeneity_status"].loc[0, "status"] == "PASS"
