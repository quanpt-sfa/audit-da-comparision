from __future__ import annotations

import argparse

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import cfo_tilt_tables, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-tab corrective tilt by CFO dominance, materiality, year, and sample"
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]), low_memory=False)
    tables = cfo_tilt_tables(baseline, panel, config["cfo_tilt"])
    write_tables(tables, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
