from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.analysis_window import AnalysisWindow


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _validate_inputs(repo_root: Path, config_path: Path) -> tuple[dict, AnalysisWindow]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    window = AnalysisWindow.from_mapping(config.get("analysis_window"))
    panel_path = _resolve(repo_root, config["paths"]["panel_input"])
    baseline_path = _resolve(repo_root, config["paths"]["baseline_input"])

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Processed panel not found: {panel_path}. Run scripts/01_build_panel.py first."
        )
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"OLS baseline not found: {baseline_path}. Run scripts/03_run_baselines.py first."
        )
    if baseline_path.stat().st_mtime < panel_path.stat().st_mtime:
        raise RuntimeError(
            "OLS baseline is older than the processed panel and is therefore stale. "
            "Run scripts/03_run_baselines.py successfully before diagnostics.\n"
            f"Panel: {panel_path}\nBaseline: {baseline_path}"
        )

    baseline = pd.read_csv(baseline_path, low_memory=False)
    required_contract_columns = {
        "source_start_year_contract",
        "source_end_year_contract",
        "training_start_year_contract",
        "training_min_year",
        "training_max_year",
        "test_start_year_contract",
        "test_end_year_contract",
    }
    missing = sorted(required_contract_columns - set(baseline.columns))
    if missing:
        raise RuntimeError(
            "OLS baseline predates the shared TT200 time contract and must be "
            f"rebuilt. Missing metadata columns: {missing}"
        )
    if pd.to_numeric(baseline["training_min_year"], errors="coerce").lt(
        window.training_start_year
    ).any():
        raise RuntimeError("OLS baseline uses training observations before 2015")
    test_year = pd.to_numeric(baseline["fiscal_year"], errors="coerce")
    if not test_year.between(window.test_start_year, window.test_end_year).all():
        raise RuntimeError("OLS baseline contains test years outside 2016-2025")
    contract_values = {
        "source_start_year_contract": window.source_start_year,
        "source_end_year_contract": window.source_end_year,
        "training_start_year_contract": window.training_start_year,
        "test_start_year_contract": window.test_start_year,
        "test_end_year_contract": window.test_end_year,
    }
    for column, expected in contract_values.items():
        values = set(
            pd.to_numeric(baseline[column], errors="coerce").dropna().astype(int)
        )
        if values != {expected}:
            raise RuntimeError(
                f"OLS baseline {column}={sorted(values)} does not match {expected}"
            )
    return config, window


def _build_runtime_inputs(
    repo_root: Path,
    config_path: Path,
    config: dict,
    window: AnalysisWindow,
) -> tuple[Path, Path, pd.DataFrame]:
    panel_path = _resolve(repo_root, config["paths"]["panel_input"])
    panel = pd.read_csv(panel_path, low_memory=False)
    source_panel = panel.loc[window.source_mask(panel["fiscal_year"])].copy()
    if source_panel.empty:
        raise RuntimeError("No panel rows remain in the TT200 source window")

    runtime_panel = config_path.with_name(".next_diagnostics.tt200_panel.csv.gz")
    source_panel.to_csv(runtime_panel, index=False, compression="gzip")

    runtime = deepcopy(config)
    runtime["analysis_window"] = window.as_dict()
    runtime["paths"]["panel_input"] = str(runtime_panel.resolve())
    runtime.setdefault("cfs_identity", {})["minimum_year"] = window.source_start_year
    runtime["cfs_identity"]["maximum_year"] = window.source_end_year
    runtime.setdefault("cfs_deep_dive", {})["minimum_year"] = window.source_start_year
    runtime["cfs_deep_dive"]["maximum_year"] = window.source_end_year
    runtime.setdefault("calibration", {})["training_start_year"] = (
        window.training_start_year
    )
    runtime["calibration"]["minimum_test_year"] = window.test_start_year
    runtime["calibration"]["maximum_test_year"] = window.test_end_year

    runtime_path = config_path.with_name(".next_diagnostics.runtime.yaml")
    runtime_path.write_text(
        yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    status = pd.DataFrame(
        [
            {
                "status": "PASS",
                **window.as_dict(),
                "panel_rows_before": len(panel),
                "panel_rows_after_source_window": len(source_panel),
                "panel_minimum_year_after": int(source_panel["fiscal_year"].min()),
                "panel_maximum_year_after": int(source_panel["fiscal_year"].max()),
            }
        ]
    )
    return runtime_path, runtime_panel, status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the post-baseline transition and falsification diagnostics"
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    config, window = _validate_inputs(repo_root, config_path)
    runtime_path, runtime_panel, window_status = _build_runtime_inputs(
        repo_root, config_path, config, window
    )

    scripts = [
        "05_audit_tails_and_ta.py",
        "11_analyze_ta_decomposition.py",
        "12_analyze_cfs_identity.py",
        "14_analyze_cfo_tilt.py",
        "13_analyze_component_placebos.py",
        "15_analyze_cfs_deep_dive.py",
        "17_analyze_cfs_uncertainty_bridge.py",
        "16_write_cfs_deep_dive_report.py",
        "06_analyze_sign_transitions.py",
        "07_run_directional_placebos.py",
        "08_analyze_rolling_calibration.py",
        "09_analyze_model_family_discordance.py",
        "10_write_next_diagnostics_report.py",
    ]

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    output_dir = _resolve(repo_root, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    window_status.to_csv(
        output_dir / "next_diagnostics_time_contract_status.csv", index=False
    )

    try:
        for script in scripts:
            command = [
                sys.executable,
                str(scripts_dir / script),
                "--config",
                str(runtime_path),
            ]
            print("Running", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=repo_root, env=env)
    finally:
        runtime_path.unlink(missing_ok=True)
        runtime_panel.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
