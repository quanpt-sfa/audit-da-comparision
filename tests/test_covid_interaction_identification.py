from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.cfs_regime_robustness_identified import (
    identified_covid_interactions,
)


def test_covid_main_dummy_is_omitted_with_full_year_fixed_effects() -> None:
    rng = np.random.default_rng(17)
    rows = []
    for issuer_index in range(80):
        for year in range(2016, 2026):
            score = rng.normal()
            probability = 1 / (
                1
                + np.exp(
                    -(
                        -2.0
                        + 0.65 * abs(score)
                        + 0.25 * abs(score) * (year in {2020, 2021})
                    )
                )
            )
            rows.append(
                {
                    "issuer_ticker": f"F{issuer_index:03d}",
                    "fiscal_year": year,
                    "raw_exchange": ["HOSE", "HNX", "UPCOM"][issuer_index % 3],
                    "industry_name": "A" if issuer_index % 2 == 0 else "B",
                    "lag_assets": float(np.exp(rng.normal(25, 1))),
                    "pre_cfo_scaled": rng.normal(0.05, 0.10),
                    "abnormal_cfo_proxy": score,
                    "any_candidate": rng.binomial(1, probability),
                }
            )
    sample = pd.DataFrame(rows)
    result = identified_covid_interactions(
        sample,
        {
            "outcomes": ["any_candidate"],
            "minimum_interaction_rows": 100,
            "minimum_interaction_positives": 5,
            "covid_fixed_effects": [
                "fiscal_year",
                "raw_exchange",
                "industry_name",
            ],
            "covid": {
                "primary_shock_years": [2020, 2021],
                "alternative_shock_windows": {
                    "COVID_2020_ONLY": [2020],
                    "COVID_2020_2022": [2020, 2021, 2022],
                },
            },
        },
        ["any_candidate"],
    )

    assert not result.empty
    assert "covid_shock" not in set(result["term"])
    assert set(result.loc[result["focal_term"].eq(True), "term"]) == {
        "score_x_covid_shock"
    }
    assert result["covid_main_dummy_included"].eq(False).all()
    assert result["identification_note"].str.contains(
        "fiscal-year fixed effects", regex=False
    ).all()
