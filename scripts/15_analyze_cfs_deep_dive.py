from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.diag_common import write_tables
from audit_da.diag_cfs_deep_dive import deep_dive_tables


def _read_artifact(output_dir: Path, name: str) -> pd.DataFrame:
    plain = output_dir / f"{name}.csv"
    compressed = output_dir / f"{name}.csv.gz"
    if compressed.exists():
        return pd.read_csv(compressed, low_memory=False)
    if plain.exists():
        return pd.read_csv(plain, low_memory=False)
    raise FileNotFoundError(
        f"Required artifact not found: {plain} or {compressed}. "
        "Run scripts 12-14 and 13 before this deep dive."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose CFS offsets, profile chronic reclassifiers, test incentive "
            "asymmetry, compare anchors on a common sample, and optionally split "
            "results by auditor quality."
        )
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    output_dir = resolve(config_path, config["paths"]["output_dir"])
    panel = pd.read_csv(
        resolve(config_path, config["paths"]["panel_input"]),
        low_memory=False,
    )
    identity_cases = _read_artifact(output_dir, "cfs_identity_cases")
    alignment_cases = _read_artifact(output_dir, "component_alignment_cases")

    metadata_value = config["paths"].get(
        "audit_metadata_input", "data/processed/audit_metadata.csv"
    )
    audit_metadata_path = resolve(config_path, metadata_value)
    tables = deep_dive_tables(
        identity_cases=identity_cases,
        alignment_cases=alignment_cases,
        panel=panel,
        settings=config["cfs_deep_dive"],
        audit_metadata_path=audit_metadata_path,
    )
    write_tables(tables, output_dir)


if __name__ == "__main__":
    main()
