from __future__ import annotations
import argparse
import pandas as pd
from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import sign_transition_tables, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DA pre/post sign transition matrices and expose R-hidden sign flips")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    settings = config["sign_transitions"]
    tables = sign_transition_tables(baseline, settings["sign_epsilon_grid"], settings["reduction_delta_grid"])
    write_tables(tables, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
