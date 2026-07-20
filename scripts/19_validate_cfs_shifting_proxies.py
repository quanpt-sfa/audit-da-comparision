from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import window_from_section
from audit_da.cfs_completion import (
    restrict_estimation_panel,
    restrict_to_estimation_keys,
)
from audit_da.diag_common import write_tables
from audit_da.diag_cfs_proxy_validation import run_cfs_shifting_validation
from audit_da.icb_industry import attach_icb_industry, load_icb_industry


def read_table(output: Path, name: str) -> pd.DataFrame:
    plain = output / f"{name}.csv"
    gz = output / f"{name}.csv.gz"
    if plain.exists():
        return pd.read_csv(plain, low_memory=False)
    if gz.exists():
        return pd.read_csv(gz, low_memory=False)
    raise FileNotFoundError(f"Required upstream table not found: {plain} or {gz}")


def _window_status(
    raw_panel: pd.DataFrame,
    source_panel: pd.DataFrame,
    cases: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    window = window_from_section(settings)
    case_year = pd.to_numeric(cases.get("fiscal_year"), errors="coerce")
    return pd.DataFrame(
        [
            {
                "status": "PASS",
                **window.as_dict(),
                "panel_rows_before": len(raw_panel),
                "panel_rows_after_source_window": len(source_panel),
                "panel_minimum_year_after": int(source_panel["fiscal_year"].min())
                if not source_panel.empty
                else pd.NA,
                "panel_maximum_year_after": int(source_panel["fiscal_year"].max())
                if not source_panel.empty
                else pd.NA,
                "observed_case_rows_source_window": int(
                    window.target_mask(case_year).sum()
                ),
                "training_rule": "training_start_year <= fiscal_year <= test_year - 1",
                "comparison_rule": "common issuer-year intersection across prespecified models",
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate inferred cash-flow shifting proxies against observed "
            "pre/post reclassifications"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = dict(config["cfs_shifting_validation"])
    window = window_from_section(settings)
    raw_panel = pd.read_csv(
        resolve(config_path, config["paths"]["panel_input"]), low_memory=False
    )
    panel = raw_panel.loc[window.source_mask(raw_panel["fiscal_year"])].copy()
    panel["fiscal_year"] = pd.to_numeric(
        panel["fiscal_year"], errors="coerce"
    ).astype(int)

    auxiliary: dict[str, pd.DataFrame] = {}
    industry_value = config.get("paths", {}).get("industry_input")
    if industry_value:
        industry_path = resolve(config_path, industry_value)
        require_industry = bool(
            settings.get("industry_mapping", {}).get("required", True)
        )
        if not industry_path.exists():
            if require_industry:
                raise FileNotFoundError(
                    f"Required ICB industry file not found: {industry_path}"
                )
            auxiliary["cfs_industry_mapping_status"] = pd.DataFrame(
                [
                    {
                        "industry_path": str(industry_path),
                        "status": "NOT_EVALUATED",
                        "reason": "File not found",
                    }
                ]
            )
        else:
            mapping, load_status = load_icb_industry(
                industry_path,
                settings.get("industry_mapping", {}),
            )
            panel, join_status, unmatched = attach_icb_industry(panel, mapping)
            auxiliary["cfs_industry_mapping"] = mapping
            auxiliary["cfs_industry_mapping_status"] = load_status.merge(
                join_status, how="cross", suffixes=("_load", "_join")
            )
            auxiliary["cfs_industry_unmatched_tickers"] = unmatched

    estimation_panel, estimation_status = restrict_estimation_panel(panel, settings)
    auxiliary["cfs_expected_cfo_estimation_sample_status"] = estimation_status

    cases = read_table(
        output,
        config.get("upstream", {}).get(
            "observed_cases_table", "cfs_offset_channel_cases"
        ),
    )
    case_year = pd.to_numeric(cases["fiscal_year"], errors="coerce")
    cases = cases.loc[window.target_mask(case_year)].copy()
    cases["fiscal_year"] = case_year.loc[cases.index].astype(int)

    line_items = read_table(output, "cfs_line_item_panel")
    item_year = pd.to_numeric(line_items["fiscal_year"], errors="coerce")
    line_items = line_items.loc[window.target_mask(item_year)].copy()
    line_items["fiscal_year"] = item_year.loc[line_items.index].astype(int)

    auxiliary["cfs_analysis_window_status"] = _window_status(
        raw_panel, panel, cases, settings
    )

    # Use the identical issuer-year population for fitting, prediction,
    # validation, and detailed line-item reconciliation.
    cases = restrict_to_estimation_keys(cases, estimation_panel)
    line_items = restrict_to_estimation_keys(line_items, estimation_panel)

    tables = run_cfs_shifting_validation(
        estimation_panel, cases, line_items, settings
    )
    tables.update(auxiliary)
    write_tables(tables, output)


if __name__ == "__main__":
    main()
