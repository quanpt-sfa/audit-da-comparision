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
    metadata = window.as_dict()
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
    return predictions, folds
