from __future__ import annotations

import argparse
import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import decomposition_tables, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose DA reduction into PAT, CFO, and benchmark-reference components")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]))
    settings = config["decomposition"]
    tables = decomposition_tables(
        baseline, panel,
        cfo_dominance_grid=settings["cfo_to_pat_ratio_grid"],
        materiality_grid=settings["reduction_materiality_grid"],
        trim_fraction=float(settings["trim_fraction"]),
        asset_gap_threshold=float(settings["asset_pre_post_gap_threshold"]),
        asset_growth_multiple_threshold=float(settings["asset_growth_multiple_threshold"]),
        small_lag_assets_quantile=float(settings["small_lag_assets_quantile"]),
        repeated_candidate_min_years=int(settings["repeated_candidate_min_years"]),
    )
    write_tables(tables, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
