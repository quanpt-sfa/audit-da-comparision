from __future__ import annotations
import argparse
import pandas as pd
from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import directional_placebo, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run directional Jensen/permutation placebos under common DA benchmarks")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    baseline = pd.read_csv(resolve(config_path, config["paths"]["baseline_input"]))
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]))
    settings = config["placebo"]
    summary, draws = directional_placebo(
        baseline, panel, settings["models"], settings["common_benchmarks"], settings["strata_columns"],
        int(settings["minimum_stratum_size"]), int(settings["permutations"]), float(settings["trim_fraction"]),
        int(settings["random_seed"]), float(settings["identity_tolerance"]),
        conditioning_bins=settings.get("conditioning_bins", {}),
    )
    write_tables({"directional_placebo_summary": summary, "directional_placebo_draws": draws}, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
