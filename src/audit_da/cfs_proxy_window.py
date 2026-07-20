from __future__ import annotations

from typing import Any

import pandas as pd

from .analysis_window import window_from_section
from .cfs_proxy_validate import rolling_expected_cfo_proxies as _legacy_rolling


def rolling_expected_cfo_proxies(
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expected-CFO folds under the shared TT200 time contract."""
    window = window_from_section(settings)
    source = panel.loc[window.source_mask(panel["fiscal_year"])].copy()
    runtime = dict(settings)
    runtime["minimum_year"] = window.source_start_year
    runtime["maximum_year"] = window.source_end_year
    runtime["training_start_year"] = window.training_start_year
    runtime["minimum_test_year"] = window.test_start_year
    runtime["maximum_test_year"] = window.test_end_year

    predictions, folds = _legacy_rolling(source, runtime)
    source_year = pd.to_numeric(source["fiscal_year"], errors="coerce").dropna()
    metadata = window.as_dict() | {
        "source_panel_minimum_year_actual": int(source_year.min())
        if not source_year.empty
        else pd.NA,
        "source_panel_maximum_year_actual": int(source_year.max())
        if not source_year.empty
        else pd.NA,
    }
    for frame in (predictions, folds):
        if frame.empty:
            continue
        for name, value in metadata.items():
            frame[name] = value

    if not predictions.empty:
        year = pd.to_numeric(predictions["fiscal_year"], errors="coerce")
        if not year.between(window.test_start_year, window.test_end_year).all():
            raise AssertionError(
                "Expected-CFO predictions contain years outside the test contract"
            )
    if not folds.empty:
        year = pd.to_numeric(folds["fiscal_year"], errors="coerce")
        if not year.between(window.test_start_year, window.test_end_year).all():
            raise AssertionError(
                "Expected-CFO folds contain years outside the test contract"
            )
    if not source_year.empty and int(source_year.min()) < window.training_start_year:
        raise AssertionError(
            "Expected-CFO source panel includes pre-contract training years"
        )
    return predictions, folds
