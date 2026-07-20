from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import yaml

from audit_da.analysis_window import AnalysisWindow


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


def test_runtime_config_locks_source_training_and_test_windows(tmp_path: Path) -> None:
    runner = _load_script_module(
        "run_cfs_shifting_validation_test",
        "run_cfs_shifting_validation.py",
    )
    config_path = tmp_path / "cfs_shifting_validation.yaml"
    original = {
        "paths": {
            "raw_input": "data/raw.csv",
            "panel_input": "data/panel.csv",
            "output_dir": "artifacts/out",
        },
        "cfs_shifting_validation": {
            "minimum_year": 2018,
            "maximum_year": 2025,
            "minimum_test_year": 2018,
            "maximum_test_year": 2025,
        },
    }
    config_path.write_text(
        yaml.safe_dump(original, sort_keys=False),
        encoding="utf-8",
    )

    runtime_path, runtime = runner.build_runtime_config(
        config_path,
        original,
        2015,
        2025,
        2016,
    )

    settings = runtime["cfs_shifting_validation"]
    assert settings["analysis_window"] == {
        "source_start_year": 2015,
        "source_end_year": 2025,
        "training_start_year": 2015,
        "test_start_year": 2016,
        "test_end_year": 2025,
    }
    assert settings["minimum_year"] == 2015
    assert settings["maximum_year"] == 2025
    assert settings["training_start_year"] == 2015
    assert settings["minimum_test_year"] == 2016
    assert settings["maximum_test_year"] == 2025
    loaded = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert loaded["cfs_shifting_validation"] == settings
    runtime_path.unlink()


def test_auditor_uses_source_year_for_history_and_test_year_for_cases() -> None:
    auditor_script = _load_script_module(
        "analyze_auditor_regime_test",
        "22_analyze_auditor_regime.py",
    )
    years = [2014, 2015, 2016, 2020, 2025, 2026]
    cases = pd.DataFrame(
        {
            "issuer_ticker": [f"C{i}" for i in range(len(years))],
            "fiscal_year": years,
        }
    )
    firm_year = pd.DataFrame(
        {
            "issuer_ticker": [f"C{i}" for i in range(len(years))],
            "fiscal_year": years,
            "auditor_group": ["NON_BIG4"] * len(years),
        }
    )
    settings = {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2025,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2025,
        }
    }

    cases_window, auditor_window, status = (
        auditor_script.apply_analysis_year_window(cases, firm_year, settings)
    )

    assert cases_window["fiscal_year"].tolist() == [2016, 2020, 2025]
    assert auditor_window["fiscal_year"].tolist() == [2015, 2016, 2020, 2025]
    assert status.loc[0, "source_start_year"] == 2015
    assert status.loc[0, "test_start_year"] == 2016
    assert status.loc[0, "case_minimum_year_after"] == 2016
    assert status.loc[0, "auditor_firm_year_minimum_year_after"] == 2015


def test_invalid_contract_is_rejected() -> None:
    try:
        AnalysisWindow.from_mapping(
            {
                "source_start_year": 2015,
                "source_end_year": 2025,
                "training_start_year": 2014,
                "test_start_year": 2016,
                "test_end_year": 2025,
            }
        )
    except ValueError as exc:
        assert "training_start_year" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Training before the source regime must be rejected")


def test_test_year_must_follow_training_start() -> None:
    try:
        AnalysisWindow.from_mapping(
            {
                "source_start_year": 2015,
                "source_end_year": 2025,
                "training_start_year": 2015,
                "test_start_year": 2015,
                "test_end_year": 2025,
            }
        )
    except ValueError as exc:
        assert "test_start_year" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Test must start after the first training year")
