from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.cfs_completion import (
    completion_gate_status,
    core_reconciliation_outputs,
    history_incremental_comparison,
)
from audit_da.diag_common import write_tables


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def remove_stale_pdf_verification_outputs(output: Path) -> None:
    """Remove legacy manifest files so prior runs cannot contaminate bundles."""
    for suffix in (".csv", ".csv.gz"):
        path = output / f"cfs_pdf_verification_manifest{suffix}"
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete the remaining executable CFS validation gates"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = dict(config["cfs_shifting_validation"])

    remove_stale_pdf_verification_outputs(output)

    panel = pd.read_csv(
        resolve(config_path, config["paths"]["panel_input"]), low_memory=False
    )
    observed = read_table(
        output,
        config.get("upstream", {}).get(
            "observed_cases_table", "cfs_offset_channel_cases"
        ),
    )
    line_items = read_table(output, "cfs_line_item_panel")
    primary_cases = read_table(
        output, "cfs_shifting_proxy_common_primary_core_cases"
    )
    all_model_cases = read_table(
        output, "cfs_shifting_proxy_common_all_core_cases"
    )
    validation = read_table(output, "cfs_shifting_proxy_validation")
    estimation_status = read_table(
        output, "cfs_expected_cfo_estimation_sample_status"
    )

    tables = core_reconciliation_outputs(
        line_items,
        observed,
        panel,
        primary_cases,
        all_model_cases,
        settings,
    )
    history = history_incremental_comparison(validation, settings)
    tables["cfs_history_incremental_comparison"] = history

    primary_reconciliation = tables.get(
        "cfs_line_item_reconciliation_cases_common_primary_core",
        pd.DataFrame(),
    )
    tables["cfs_completion_gate_status"] = completion_gate_status(
        estimation_status,
        history,
        primary_reconciliation,
    )
    write_tables(tables, output)


if __name__ == "__main__":
    main()
