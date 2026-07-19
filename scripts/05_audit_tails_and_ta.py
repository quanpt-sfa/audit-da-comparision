from __future__ import annotations
import argparse
import pandas as pd
from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import ta_source_audit, tail_case_tables, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TA construction and export extreme pre/post DA cases")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]))
    output = resolve(config_path, config["paths"]["output_dir"])
    source_summary, pair = ta_source_audit(panel)
    settings = config["tail_audit"]
    tables = tail_case_tables(
        baseline, panel,
        settings["primary_model"], settings["primary_benchmark"],
        float(settings["tail_fraction"]), int(settings["manual_cases_per_side"]), int(settings["special_year"]),
    )
    tables["ta_source_summary"] = source_summary
    if {"ta_source_pre", "ta_source_post"}.issubset(pair.columns):
        tables["ta_source_pair_summary"] = pair.groupby(["ta_source_pre", "ta_source_post"], observed=True).size().rename("rows").reset_index()
    write_tables(tables, output)


if __name__ == "__main__":
    main()
