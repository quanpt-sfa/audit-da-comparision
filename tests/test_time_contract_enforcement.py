from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

from audit_da.analysis_window import AnalysisWindow, window_from_section
from audit_da.cfs_proxy_window import rolling_expected_cfo_proxies


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_next_diagnostics_runtime_panel_is_source_window_only(
    tmp_path: Path,
) -> None:
    runner = _load_script_module(
        "run_next_diagnostics_contract_test", "run_next_diagnostics.py"
    )
    panel_path = tmp_path / "panel.csv"
    baseline_path = tmp_path / "baseline.csv"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "next.yaml"

    pd.DataFrame(
        {
            "issuer_ticker": ["A", "A", "A", "A"],
            "fiscal_year": [2014, 2015, 2025, 2026],
            "audit_status": ["audited"] * 4,
        }
    ).to_csv(panel_path, index=False)
    pd.DataFrame(
        {
            "issuer_ticker": ["A"],
            "fiscal_year": [2016],
            "source_start_year_contract": [2015],
            "source_end_year_contract": [2025],
            "training_start_year_contract": [2015],
            "training_min_year": [2015],
            "training_max_year": [2015],
            "test_start_year_contract": [2016],
            "test_end_year_contract": [2025],
        }
    ).to_csv(baseline_path, index=False)
    os.utime(panel_path, (1, 1))
    os.utime(baseline_path, (2, 2))

    config = {
        "paths": {
            "panel_input": str(panel_path),
            "baseline_input": str(baseline_path),
            "output_dir": str(output_dir),
        },
        "analysis_window": AnalysisWindow().as_dict(),
        "cfs_identity": {},
        "cfs_deep_dive": {},
        "calibration": {},
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    loaded, window = runner._validate_inputs(tmp_path, config_path)
    runtime_path, runtime_panel, status = runner._build_runtime_inputs(
        tmp_path, config_path, loaded, window
    )
    try:
        filtered = pd.read_csv(runtime_panel)
        assert filtered["fiscal_year"].tolist() == [2015, 2025]
        assert status.loc[0, "panel_minimum_year_after"] == 2015
        assert status.loc[0, "panel_maximum_year_after"] == 2025
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        assert runtime["analysis_window"] == AnalysisWindow().as_dict()
        assert Path(runtime["paths"]["panel_input"]) == runtime_panel.resolve()
    finally:
        runtime_path.unlink(missing_ok=True)
        runtime_panel.unlink(missing_ok=True)


def _contract_folds(source_minimum: int = 2015) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2016, 2025],
            "source_start_year": [2015, 2015],
            "source_end_year": [2025, 2025],
            "training_start_year": [2015, 2015],
            "test_start_year": [2016, 2016],
            "test_end_year": [2025, 2025],
            "source_panel_minimum_year_actual": [source_minimum, source_minimum],
        }
    )


def test_completion_gate_verifies_artifact_contract_metadata() -> None:
    completion = _load_script_module(
        "complete_cfs_contract_test", "21_complete_cfs_validation_gates.py"
    )
    source = pd.DataFrame(
        {"issuer_ticker": ["A", "B"], "fiscal_year": [2015, 2025]}
    )
    primary = pd.DataFrame(
        {"issuer_ticker": ["A", "B"], "fiscal_year": [2016, 2025]}
    )
    detail, gate = completion.time_contract_gate(
        {"analysis_window": AnalysisWindow().as_dict()},
        source,
        source,
        source,
        _contract_folds(),
        primary,
    )
    assert detail.loc[0, "status"] == "PASS"
    assert gate.loc[0, "status"] == "PASS"

    detail_bad, gate_bad = completion.time_contract_gate(
        {"analysis_window": AnalysisWindow().as_dict()},
        source,
        source,
        source,
        _contract_folds(source_minimum=2014),
        primary,
    )
    assert detail_bad.loc[0, "status"] == "FAILED"
    assert gate_bad.loc[0, "status"] == "FAILED"
