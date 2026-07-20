from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import window_from_section
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
    for suffix in (".csv", ".csv.gz"):
        path = output / f"cfs_pdf_verification_manifest{suffix}"
        if path.exists():
            path.unlink()


def _year_bounds(frame: pd.DataFrame) -> tuple[object, object]:
    if frame.empty or "fiscal_year" not in frame.columns:
        return pd.NA, pd.NA
    year = pd.to_numeric(frame["fiscal_year"], errors="coerce").dropna()
    if year.empty:
        return pd.NA, pd.NA
    return int(year.min()), int(year.max())


def time_contract_gate(
    settings: dict,
    panel: pd.DataFrame,
    observed: pd.DataFrame,
    line_items: pd.DataFrame,
    folds: pd.DataFrame,
    primary_cases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = window_from_section(settings)
    panel_min, panel_max = _year_bounds(panel)
    target_min, target_max = _year_bounds(observed)
    item_min, item_max = _year_bounds(line_items)
    fold_min, fold_max = _year_bounds(folds)
    primary_min, primary_max = _year_bounds(primary_cases)

    source_ok = all(
        value is pd.NA
        or value is None
        or (window.source_start_year <= int(value) <= window.source_end_year)
        for value in (panel_min, panel_max, target_min, target_max, item_min, item_max)
    )
    test_ok = all(
        value is pd.NA
        or value is None
        or (window.test_start_year <= int(value) <= window.test_end_year)
        for value in (fold_min, fold_max, primary_min, primary_max)
    )
    status = "PASS" if source_ok and test_ok else "FAILED"
    detail = pd.DataFrame(
        [
            {
                "status": status,
                **window.as_dict(),
                "panel_minimum_year": panel_min,
                "panel_maximum_year": panel_max,
                "target_minimum_year": target_min,
                "target_maximum_year": target_max,
                "line_item_minimum_year": item_min,
                "line_item_maximum_year": item_max,
                "expected_cfo_fold_minimum_year": fold_min,
                "expected_cfo_fold_maximum_year": fold_max,
                "common_primary_minimum_year": primary_min,
                "common_primary_maximum_year": primary_max,
                "source_window_pass": source_ok,
                "test_window_pass": test_ok,
                "common_sample_rule": "intersection of prespecified model availability by issuer-year",
            }
        ]
    )
    gate = pd.DataFrame(
        [
            {
                "gate": "consistent_tt200_time_contract",
                "status": status,
                "evidence_rows": len(primary_cases),
            }
        ]
    )
    return detail, gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete the remaining executable CFS validation gates"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = dict(config["cfs_shifting_validation"])
    window = window_from_section(settings)

    remove_stale_pdf_verification_outputs(output)

    raw_panel = pd.read_csv(
        resolve(config_path, config["paths"]["panel_input"]), low_memory=False
    )
    panel = raw_panel.loc[window.source_mask(raw_panel["fiscal_year"])].copy()
    observed = read_table(
        output,
        config.get("upstream", {}).get(
            "observed_cases_table", "cfs_offset_channel_cases"
        ),
    )
    observed = observed.loc[window.target_mask(observed["fiscal_year"])].copy()
    line_items = read_table(output, "cfs_line_item_panel")
    line_items = line_items.loc[
        window.target_mask(line_items["fiscal_year"])
    ].copy()
    primary_cases = read_table(
        output, "cfs_shifting_proxy_common_primary_core_cases"
    )
    all_model_cases = read_table(
        output, "cfs_shifting_proxy_common_all_core_cases"
    )
    validation = read_table(output, "cfs_shifting_proxy_validation")
    folds = read_table(output, "cfs_expected_cfo_folds")
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
    gates = completion_gate_status(
        estimation_status,
        history,
        primary_reconciliation,
    )
    time_status, time_gate = time_contract_gate(
        settings,
        panel,
        observed,
        line_items,
        folds,
        primary_cases,
    )
    tables["cfs_time_contract_status"] = time_status
    tables["cfs_completion_gate_status"] = pd.concat(
        [gates, time_gate], ignore_index=True
    )
    write_tables(tables, output)


if __name__ == "__main__":
    main()
