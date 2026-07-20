from __future__ import annotations

import argparse

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import AnalysisWindow
from audit_da.cfs_target_finalize import finalize_cfs_target_tables
from audit_da.next_diagnostics import cfs_identity_tables, write_tables


REQUIRED_CFS_COLUMNS = {
    "cfo",
    "cfi",
    "cff",
    "net_cash_change",
    "cash_begin_cfs",
    "fx_effect",
    "cash_end_cfs",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test cash-flow statement internal identities and classify CFO corrections"
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    raw_panel = pd.read_csv(
        resolve(config_path, config["paths"]["panel_input"]),
        low_memory=False,
    )
    window = AnalysisWindow.from_mapping(config.get("analysis_window"))
    panel = raw_panel.loc[window.source_mask(raw_panel["fiscal_year"])].copy()
    baseline = baseline.loc[window.test_mask(baseline["fiscal_year"])].copy()
    missing = sorted(REQUIRED_CFS_COLUMNS - set(panel.columns))
    if missing:
        raise RuntimeError(
            "Processed panel lacks the full cash-flow-statement fields required "
            f"for identity testing: {missing}. Pull the latest branch, rebuild the "
            "panel with scripts/01_build_panel.py, then rerun OLS baselines."
        )
    settings = dict(config["cfs_identity"])
    settings["minimum_year"] = window.source_start_year
    settings["maximum_year"] = window.source_end_year
    tables = cfs_identity_tables(panel, baseline, settings)
    tables = finalize_cfs_target_tables(tables, settings)
    tables["cfs_identity_window_status"] = pd.DataFrame(
        [
            {
                "status": "PASS",
                **window.as_dict(),
                "panel_rows_before": len(raw_panel),
                "panel_rows_after_source_window": len(panel),
                "baseline_rows_in_test_window": len(baseline),
            }
        ]
    )
    write_tables(tables, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
