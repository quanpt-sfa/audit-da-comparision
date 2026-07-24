from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.predictive_validity import PredictiveValiditySettings
from audit_da.predictive_validity_parallel import (
    summarize_accrual_quality_parallel,
    summarize_oos_parallel,
)


def _prediction_frame() -> pd.DataFrame:
    rows = []
    for test_index, test in enumerate(
        [
            "earnings_persistence",
            "earnings_to_future_cfo",
            "cfo_persistence",
            "earnings_cfo_horse_race",
        ]
    ):
        for issuer in range(12):
            actual = 0.02 * issuer + 0.01 * test_index
            error_pre = 0.020 + 0.001 * (issuer % 3)
            error_audited = 0.008 + 0.0005 * (issuer % 3)
            rows.append(
                {
                    "issuer_ticker": f"F{issuer:03d}",
                    "fiscal_year": 2018 + issuer % 4,
                    "test": test,
                    "construct": test,
                    "actual": actual,
                    "prediction_benchmark": actual - 0.03,
                    "error_pre": error_pre,
                    "error_audited": error_audited,
                    "squared_error_pre": error_pre**2,
                    "squared_error_audited": error_audited**2,
                    "absolute_error_pre": abs(error_pre),
                    "absolute_error_audited": abs(error_audited),
                }
            )
    return pd.DataFrame(rows)


def _crossfit_frame() -> pd.DataFrame:
    rows = []
    for issuer in range(12):
        pre = 0.020 + 0.001 * (issuer % 3)
        audited = 0.008 + 0.0005 * (issuer % 3)
        rows.append(
            {
                "issuer_ticker": f"F{issuer:03d}",
                "fiscal_year": 2017 + issuer % 4,
                "residual_pre": pre,
                "residual_audited": audited,
                "absolute_residual_pre": abs(pre),
                "absolute_residual_audited": abs(audited),
                "squared_residual_pre": pre**2,
                "squared_residual_audited": audited**2,
            }
        )
    return pd.DataFrame(rows)


def test_parallel_oos_bootstrap_preserves_point_estimates() -> None:
    settings = PredictiveValiditySettings(bootstrap_draws=40)
    predictions = _prediction_frame()
    _, serial = summarize_oos_parallel(
        predictions,
        settings,
        workers=1,
        batch_size=8,
    )
    _, parallel = summarize_oos_parallel(
        predictions,
        settings,
        workers=2,
        batch_size=8,
    )
    keys = ["test", "metric"]
    left = serial.sort_values(keys).reset_index(drop=True)
    right = parallel.sort_values(keys).reset_index(drop=True)
    np.testing.assert_allclose(left.estimate, right.estimate, equal_nan=True)
    assert left[keys].equals(right[keys])


def test_parallel_aq_bootstrap_preserves_point_estimates() -> None:
    settings = PredictiveValiditySettings(bootstrap_draws=40)
    crossfit = _crossfit_frame()
    _, serial = summarize_accrual_quality_parallel(
        crossfit,
        settings,
        workers=1,
        batch_size=8,
    )
    _, parallel = summarize_accrual_quality_parallel(
        crossfit,
        settings,
        workers=2,
        batch_size=8,
    )
    left = serial.sort_values("metric").reset_index(drop=True)
    right = parallel.sort_values("metric").reset_index(drop=True)
    np.testing.assert_allclose(left.estimate, right.estimate)
    assert left[["metric"]].equals(right[["metric"]])
