from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import yaml


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


def test_runtime_config_locks_tt200_window(tmp_path: Path) -> None:
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
    )

    settings = runtime["cfs_shifting_validation"]
    assert settings["minimum_year"] == 2015
    assert settings["maximum_year"] == 2025
    assert settings["minimum_test_year"] == 2015
    assert settings["maximum_test_year"] == 2025
    loaded = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert loaded["cfs_shifting_validation"] == settings
    runtime_path.unlink()


def test_auditor_window_excludes_pre_2015_and_post_2025() -> None:
    auditor_script = _load_script_module(
        "analyze_auditor_regime_test",
        "22_analyze_auditor_regime.py",
    )
    years = [2014, 2015, 2020, 2025, 2026]
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

    cases_window, auditor_window, status = (
        auditor_script.apply_analysis_year_window(
            cases,
            firm_year,
            {
                "analysis_minimum_year": 2015,
                "analysis_maximum_year": 2025,
            },
        )
    )

    assert cases_window["fiscal_year"].tolist() == [2015, 2020, 2025]
    assert auditor_window["fiscal_year"].tolist() == [2015, 2020, 2025]
    assert status.loc[0, "case_rows_before"] == 5
    assert status.loc[0, "case_rows_after"] == 3
    assert status.loc[0, "case_minimum_year_after"] == 2015
    assert status.loc[0, "case_maximum_year_after"] == 2025


def test_invalid_year_window_is_rejected() -> None:
    auditor_script = _load_script_module(
        "analyze_auditor_regime_invalid_window_test",
        "22_analyze_auditor_regime.py",
    )
    empty = pd.DataFrame(columns=["issuer_ticker", "fiscal_year"])

    try:
        auditor_script.apply_analysis_year_window(
            empty,
            empty,
            {
                "analysis_minimum_year": 2026,
                "analysis_maximum_year": 2025,
            },
        )
    except ValueError as exc:
        assert "analysis_minimum_year" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Invalid year window should raise ValueError")
