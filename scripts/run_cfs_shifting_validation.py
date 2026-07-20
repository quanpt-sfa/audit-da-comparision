from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import yaml


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def build_runtime_config(
    config_path: Path,
    config: dict,
    source_start_year: int,
    source_end_year: int,
    test_start_year: int,
) -> tuple[Path, dict]:
    """Create one internally consistent runtime time contract.

    Source/target observations are restricted to the TT200 reporting regime.
    Rolling training begins at the source start and test evaluation begins no
    earlier than the next fiscal year.
    """
    if source_start_year > source_end_year:
        raise ValueError("source_start_year must not exceed source_end_year")
    if test_start_year <= source_start_year:
        raise ValueError("test_start_year must follow source_start_year")
    if test_start_year > source_end_year:
        raise ValueError("test_start_year must not exceed source_end_year")

    runtime = deepcopy(config)
    settings = runtime.setdefault("cfs_shifting_validation", {})
    contract = {
        "source_start_year": int(source_start_year),
        "source_end_year": int(source_end_year),
        "training_start_year": int(source_start_year),
        "test_start_year": int(test_start_year),
        "test_end_year": int(source_end_year),
    }
    settings["analysis_window"] = contract
    settings["minimum_year"] = contract["source_start_year"]
    settings["maximum_year"] = contract["source_end_year"]
    settings["training_start_year"] = contract["training_start_year"]
    settings["minimum_test_year"] = contract["test_start_year"]
    settings["maximum_test_year"] = contract["test_end_year"]

    runtime_path = config_path.with_name(f".{config_path.stem}.runtime.yaml")
    runtime_path.write_text(
        yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_path, runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CFS item mapping and observed shifting-proxy validation"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    parser.add_argument(
        "--source-start-year",
        type=int,
        default=2015,
        help="First TT200 source/target fiscal year (default: 2015)",
    )
    parser.add_argument(
        "--source-end-year",
        type=int,
        default=2025,
        help="Last TT200 source/target fiscal year (default: 2025)",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2016,
        help="First theoretical out-of-sample test year (default: 2016)",
    )
    parser.add_argument("--analysis-minimum-year", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-maximum-year", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_start = (
        args.analysis_minimum_year
        if args.analysis_minimum_year is not None
        else args.source_start_year
    )
    source_end = (
        args.analysis_maximum_year
        if args.analysis_maximum_year is not None
        else args.source_end_year
    )

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    required_keys = ["raw_input", "panel_input"]
    if (
        config.get("cfs_shifting_validation", {})
        .get("industry_mapping", {})
        .get("required", False)
    ):
        required_keys.append("industry_input")
    for key in required_keys:
        if key not in config.get("paths", {}):
            raise KeyError(f"Required path is not configured: paths.{key}")
        path = resolve(repo_root, config["paths"][key])
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    output = resolve(repo_root, config["paths"]["output_dir"])
    upstream_name = config.get("upstream", {}).get(
        "observed_cases_table", "cfs_offset_channel_cases"
    )
    if not (output / f"{upstream_name}.csv").exists() and not (
        output / f"{upstream_name}.csv.gz"
    ).exists():
        raise FileNotFoundError(
            f"Observed CFS cases are missing from {output}. "
            "Run the CFS deep-dive diagnostics first."
        )

    runtime_path, runtime = build_runtime_config(
        config_path,
        config,
        source_start,
        source_end,
        args.test_start_year,
    )
    contract = runtime["cfs_shifting_validation"]["analysis_window"]
    print("CFS time contract:", contract, flush=True)

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        for script in [
            "18_inventory_cfs_items.py",
            "19_validate_cfs_shifting_proxies.py",
            "21_complete_cfs_validation_gates.py",
            "22_analyze_auditor_regime.py",
            "20_write_cfs_shifting_validation_report.py",
            "23_write_auditor_regime_report.py",
            "24_write_time_contract_report.py",
        ]:
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


if __name__ == "__main__":
    main()
