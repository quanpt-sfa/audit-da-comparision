from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate inferred cash-flow shifting proxies against observed pre/post reclassifications"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = dict(config["cfs_shifting_validation"])
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]), low_memory=False)

    auxiliary: dict[str, pd.DataFrame] = {}
    industry_value = config.get("paths", {}).get("industry_input")
    if industry_value:
        industry_path = resolve(config_path, industry_value)
        require_industry = bool(settings.get("industry_mapping", {}).get("required", True))
        if not industry_path.exists():
            if require_industry:
                raise FileNotFoundError(f"Required ICB industry file not found: {industry_path}")
            auxiliary["cfs_industry_mapping_status"] = pd.DataFrame([
                {"industry_path": str(industry_path), "status": "NOT_EVALUATED", "reason": "File not found"}
            ])
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

    cases = read_table(
        output,
        config.get("upstream", {}).get("observed_cases_table", "cfs_offset_channel_cases"),
    )
    line_items = read_table(output, "cfs_line_item_panel")
    tables = run_cfs_shifting_validation(panel, cases, line_items, settings)
    tables.update(auxiliary)
    write_tables(tables, output)


if __name__ == "__main__":
    main()
