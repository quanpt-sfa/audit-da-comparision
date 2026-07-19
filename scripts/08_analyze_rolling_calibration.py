from __future__ import annotations
import argparse
import pandas as pd
from _next_diag_common import load_config, resolve
from audit_da.next_diagnostics import rolling_calibration, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure rolling normal-accrual calibration, including robust likelihood diagnostics")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]))
    settings = config["calibration"]
    metrics, weights, residuals = rolling_calibration(
        panel, {k: list(v) for k, v in settings["candidate_models"].items()},
        int(settings["minimum_train_rows"]), int(settings["minimum_validation_rows"]),
        int(settings["minimum_test_year"]), int(settings["maximum_test_year"]),
        float(settings["winsor_lower"]), float(settings["winsor_upper"]), int(settings["random_seed"]),
        student_t_dfs=tuple(settings.get("student_t_df_grid", [3, 5, 10, 30])),
    )
    write_tables({"rolling_calibration": metrics, "rolling_stacking_weights": weights,
                  "rolling_calibration_residuals": residuals}, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
