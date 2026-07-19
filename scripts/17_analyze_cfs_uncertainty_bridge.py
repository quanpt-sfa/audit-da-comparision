from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.diag_common import write_tables
from audit_da.diag_cfs_uncertainty_bridge import uncertainty_bridge_tables


def _read_artifact(output_dir: Path, name: str) -> pd.DataFrame:
    compressed = output_dir / f"{name}.csv.gz"
    plain = output_dir / f"{name}.csv"
    if compressed.exists():
        return pd.read_csv(compressed, low_memory=False)
    if plain.exists():
        return pd.read_csv(plain, low_memory=False)
    raise FileNotFoundError(
        f"Required artifact not found: {compressed} or {plain}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the payoff of CFS reclassification candidates varies "
            "with cross-model and cross-benchmark measurement uncertainty."
        )
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output_dir = resolve(config_path, config["paths"]["output_dir"])

    identity_cases = _read_artifact(output_dir, "cfs_identity_cases")
    alignment_cases = _read_artifact(output_dir, "component_alignment_cases")
    tables = uncertainty_bridge_tables(
        identity_cases,
        alignment_cases,
        config["cfs_deep_dive"],
    )
    write_tables(tables, output_dir)


if __name__ == "__main__":
    main()
