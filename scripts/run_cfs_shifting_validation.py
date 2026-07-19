from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CFS item mapping and observed shifting-proxy validation"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
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

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for script in [
        "18_inventory_cfs_items.py",
        "19_validate_cfs_shifting_proxies.py",
        "21_complete_cfs_validation_gates.py",
        "20_write_cfs_shifting_validation_report.py",
    ]:
        command = [
            sys.executable,
            str(scripts_dir / script),
            "--config",
            str(config_path),
        ]
        print("Running", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
