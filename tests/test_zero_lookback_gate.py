from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

from audit_da.analysis_window import AnalysisWindow


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_completion_module():
    spec = importlib.util.spec_from_file_location(
        "complete_cfs_zero_lookback_test",
        SCRIPTS_DIR / "21_complete_cfs_validation_gates.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2016, 2025],
            "source_start_year": [2015, 2015],
            "source_end_year": [2025, 2025],
            "training_start_year": [2015, 2015],
            "test_start_year": [2016, 2016],
            "test_end_year": [2025, 2025],
            "source_panel_minimum_year_actual": [2015, 2015],
        }
    )


def test_completion_gate_rejects_nonmissing_2015_lag_values():
    completion = _load_completion_module()
    settings = {"analysis_window": AnalysisWindow().as_dict()}
    panel = pd.DataFrame(
        {
            "issuer_ticker": ["A", "B"],
            "fiscal_year": [2015, 2025],
            "lag_assets": [float("nan"), 100.0],
            "drev": [float("nan"), 5.0],
            "ta_scaled": [float("nan"), 0.02],
        }
    )
    observed = panel[["issuer_ticker", "fiscal_year"]].copy()
    line_items = observed.copy()
    primary = pd.DataFrame(
        {"issuer_ticker": ["A", "B"], "fiscal_year": [2016, 2025]}
    )

    detail, gate = completion.time_contract_gate(
        settings, panel, observed, line_items, _folds(), primary
    )
    assert detail.loc[0, "status"] == "PASS"
    assert bool(detail.loc[0, "zero_pre_2015_lookback_pass"])
    assert detail.loc[0, "source_start_year_nonmissing_lag_cells"] == 0
    assert gate.loc[0, "status"] == "PASS"

    contaminated = panel.copy()
    contaminated.loc[
        contaminated["fiscal_year"].eq(2015), "lag_assets"
    ] = 90.0
    detail_bad, gate_bad = completion.time_contract_gate(
        settings, contaminated, observed, line_items, _folds(), primary
    )
    assert detail_bad.loc[0, "status"] == "FAILED"
    assert not bool(detail_bad.loc[0, "zero_pre_2015_lookback_pass"])
    assert detail_bad.loc[0, "source_start_year_nonmissing_lag_cells"] == 1
    assert gate_bad.loc[0, "status"] == "FAILED"
