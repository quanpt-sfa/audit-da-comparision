from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.supplemental_diagnostics import (
    SupplementalSettings,
    concentration_cases,
    near_zero_cfo_cases,
    near_zero_randomisation,
    normalized_hhi,
    supplemental_summary,
)


def test_normalized_hhi_endpoints() -> None:
    assert np.isclose(normalized_hhi([1.0, 1.0]), 0.0)
    assert normalized_hhi([1.0]) == 1.0
    assert normalized_hhi([100.0, 0.000001]) > 0.99


def _line_items() -> pd.DataFrame:
    rows = []
    for issuer, year, pre, post in [
        ("A", 2020, [10.0, 10.0], [14.0, 6.0]),
        ("B", 2020, [8.0, 12.0], [16.0, 4.0]),
        ("C", 2021, [10.0, 10.0], [11.0, 9.0]),
    ]:
        for concept, before, after in zip(["x", "y"], pre, post):
            rows.append(
                {
                    "issuer_ticker": issuer,
                    "fiscal_year": year,
                    "audit_status": "unaudited",
                    "concept": concept,
                    "value_numeric": before,
                }
            )
            rows.append(
                {
                    "issuer_ticker": issuer,
                    "fiscal_year": year,
                    "audit_status": "audited",
                    "concept": concept,
                    "value_numeric": after,
                }
            )
    return pd.DataFrame(rows)


def test_concentration_cases_are_deterministic() -> None:
    settings = SupplementalSettings(
        minimum_year=2020,
        maximum_year=2021,
        concentration_draws=50,
        concentration_pool_minimum=2,
        bootstrap_draws=20,
        seed=7,
    )
    first = concentration_cases(_line_items(), settings)
    second = concentration_cases(_line_items(), settings)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert first.observed_nhhi.between(0, 1).all()
    assert first.expected_nhhi.between(0, 1).all()
    np.testing.assert_allclose(
        first.excess_nhhi,
        first.observed_nhhi - first.expected_nhhi,
    )


def _panel() -> pd.DataFrame:
    rows = []
    values = [
        ("A", 2020, -0.005, 0.004),
        ("B", 2020, 0.006, -0.005),
        ("C", 2020, -0.007, 0.006),
        ("D", 2020, 0.008, 0.007),
    ]
    for issuer, year, pre, post in values:
        for status, cfo in [("unaudited", pre), ("audited", post)]:
            rows.append(
                {
                    "issuer_ticker": issuer,
                    "fiscal_year": year,
                    "audit_status": status,
                    "cfo": cfo * 100.0,
                    "lag_assets": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_near_zero_randomisation_is_paired_and_reproducible() -> None:
    settings = SupplementalSettings(
        minimum_year=2020,
        maximum_year=2020,
        near_zero_threshold=0.02,
        near_zero_distance_bins=1,
        near_zero_draws=100,
        bootstrap_draws=20,
        seed=11,
    )
    cases = near_zero_cfo_cases(_panel(), settings)
    assert len(cases) == 4
    assert set(cases.sign_shift) == {-1, 0, 1}

    first = near_zero_randomisation(cases, settings)
    second = near_zero_randomisation(cases, settings)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].draw == "observed"
    assert len(first) == 101

    concentration = concentration_cases(
        _line_items(),
        SupplementalSettings(
            minimum_year=2020,
            maximum_year=2021,
            concentration_draws=20,
            concentration_pool_minimum=2,
            bootstrap_draws=20,
            seed=11,
        ),
    )
    summary = supplemental_summary(concentration, cases, first, settings)
    assert set(summary.diagnostic) == {
        "cfs_revision_excess_concentration",
        "near_zero_cfo_positive_sign_shift",
    }
