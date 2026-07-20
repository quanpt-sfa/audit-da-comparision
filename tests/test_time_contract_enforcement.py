from __future__ import annotations

import pandas as pd

from audit_da.analysis_window import AnalysisWindow, window_from_section
from audit_da.cfs_proxy_window import rolling_expected_cfo_proxies


def _expected_cfo_panel() -> pd.DataFrame:
    rows = []
    for year in range(2014, 2018):
        for index in range(30):
            assets = 100.0 + index
            rows.append(
                {
                    "issuer_ticker": f"F{index:03d}",
                    "fiscal_year": year,
                    "audit_status": "unaudited",
                    "raw_exchange": "HOSE",
                    "lag_assets": assets,
                    "cfo": 10.0 + index * 0.1,
                    "pat": 5.0,
                    "revenue": 80.0 + index,
                    "drev": 1.0,
                    "drec": 0.5,
                    "inv_assets": 1.0 / assets,
                    "loss": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_expected_cfo_wrapper_excludes_pre_2015_source_rows() -> None:
    settings = {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2025,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2017,
        },
        "minimum_train_rows": 20,
        "proxy_models": {
            "sales": ["inv_assets", "pre_revenue_scaled", "pre_drev_scaled"]
        },
    }
    predictions, folds = rolling_expected_cfo_proxies(
        _expected_cfo_panel(), settings
    )

    assert not predictions.empty
    assert not folds.empty
    assert set(predictions["fiscal_year"]) == {2016, 2017}
    assert folds["source_panel_minimum_year_actual"].eq(2015).all()
    assert folds["training_start_year"].eq(2015).all()
    assert not predictions["fiscal_year"].eq(2015).any()


def test_legacy_aliases_resolve_to_same_contract() -> None:
    window = window_from_section(
        {
            "minimum_year": 2015,
            "maximum_year": 2025,
            "training_start_year": 2015,
            "minimum_test_year": 2016,
            "maximum_test_year": 2025,
        }
    )
    assert window == AnalysisWindow()


def test_source_target_and_test_masks_are_distinct() -> None:
    window = AnalysisWindow()
    years = pd.Series([2014, 2015, 2016, 2025, 2026])
    assert years[window.source_mask(years)].tolist() == [2015, 2016, 2025]
    assert years[window.target_mask(years)].tolist() == [2015, 2016, 2025]
    assert years[window.test_mask(years)].tolist() == [2016, 2025]
    assert years[window.training_mask(years, 2017)].tolist() == [2015, 2016]
