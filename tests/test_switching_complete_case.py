import numpy as np
import pandas as pd

from audit_da.results_completion import CompletionSettings
from audit_da.results_completion.parallel import switching_tables
from audit_da.results_completion.switching_complete_case import (
    _profit_gate_complete,
    profit_gate_sensitivity,
    switching_cases,
)


def _panel() -> pd.DataFrame:
    rows = []
    for ticker, pat_pre, cfo_pre, ta_pre in (
        ("A", 1.0, 1.0, 0.10),
        ("B", 1.0, np.nan, 0.20),
        ("C", np.nan, -1.0, 0.30),
    ):
        for status, pat, cfo, ta in (
            ("unaudited", pat_pre, cfo_pre, ta_pre),
            ("audited", 1.0, -1.0, 0.15),
        ):
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": 2020,
                    "audit_status": status,
                    "pat": pat,
                    "cfo": cfo,
                    "lag_assets": 100.0,
                    "ta_scaled": ta,
                }
            )
    return pd.DataFrame(rows)


def _accrual_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_ticker": ["A", "B", "C"],
            "fiscal_year": [2020, 2020, 2020],
            "model": ["jones"] * 3,
            "architecture": ["pooled"] * 3,
            "architecture_group": ["all"] * 3,
            "benchmark": ["audited_reference"] * 3,
            "da_pre": [0.10, 0.20, -0.10],
            "da_post": [-0.10, 0.10, 0.10],
            "signed_shift": [-0.20, -0.10, 0.20],
            "reduction": [0.0, 0.1, 0.0],
        }
    )


def test_profit_gate_is_missing_when_profit_or_assets_are_missing():
    frame = pd.DataFrame(
        {
            "pat_pre": [1.0, np.nan, 1.0],
            "pat_post": [1.0, 1.0, 1.0],
            "lag_assets_pre": [100.0, 100.0, np.nan],
        }
    )
    gate = _profit_gate_complete(frame, 0.05)
    assert gate.iloc[0] == False
    assert pd.isna(gate.iloc[1])
    assert pd.isna(gate.iloc[2])


def test_direct_switching_uses_one_complete_case_population():
    settings = CompletionSettings(test_start_year=2020, test_end_year=2020)
    direct, model = switching_cases(_accrual_rows(), _panel(), settings)
    assert direct[["issuer_ticker", "fiscal_year"]].to_records(index=False).tolist() == [
        ("A", 2020)
    ]
    assert len(model) == 3
    assert model["gate_0_05"].notna().sum() == 1

    tables = switching_tables(direct, model, settings)
    direct_summary = tables["rq2_switch_summary"].query("model == 'direct'")
    assert direct_summary["denominator"].eq(1).all()


def test_gate_sensitivity_excludes_missing_model_gate_rows():
    settings = CompletionSettings(
        test_start_year=2020,
        test_end_year=2020,
        profit_thresholds=(0.05,),
    )
    direct, model = switching_cases(_accrual_rows(), _panel(), settings)
    result = profit_gate_sensitivity(direct, model, settings)
    da = result.query("model == 'jones' and outcome == 'da_sign'").iloc[0]
    assert da.switch_n == 1
