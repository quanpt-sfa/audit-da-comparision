from __future__ import annotations

import argparse

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import AnalysisWindow
from audit_da.next_diagnostics import rolling_calibration, write_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure rolling normal-accrual calibration, including robust "
            "likelihood diagnostics"
        )
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    raw_panel = pd.read_csv(resolve(config_path, config["paths"]["panel_input"]))
    settings = config["calibration"]
    window = AnalysisWindow.from_mapping(
        config.get("analysis_window"),
        fallback={
            "training_start_year": settings.get("training_start_year", 2015),
            "test_start_year": settings.get("minimum_test_year", 2016),
            "test_end_year": settings.get("maximum_test_year", 2025),
        },
    )
    panel = raw_panel.loc[window.source_mask(raw_panel["fiscal_year"])].copy()
    metrics, weights, residuals = rolling_calibration(
        panel,
        {k: list(v) for k, v in settings["candidate_models"].items()},
        int(settings["minimum_train_rows"]),
        int(settings["minimum_validation_rows"]),
        window.test_start_year,
        window.test_end_year,
        float(settings["winsor_lower"]),
        float(settings["winsor_upper"]),
        int(settings["random_seed"]),
        student_t_dfs=tuple(
            settings.get("student_t_df_grid", [3, 5, 10, 30])
        ),
    )
    status = pd.DataFrame(
        [
            {
                "status": "PASS",
                **window.as_dict(),
                "panel_rows_before": len(raw_panel),
                "panel_rows_after_source_window": len(panel),
                "metric_minimum_test_year": int(metrics["fiscal_year"].min())
                if not metrics.empty
                else pd.NA,
                "metric_maximum_test_year": int(metrics["fiscal_year"].max())
                if not metrics.empty
                else pd.NA,
            }
        ]
    )
    write_tables(
        {
            "rolling_calibration": metrics,
            "rolling_stacking_weights": weights,
            "rolling_calibration_residuals": residuals,
            "rolling_calibration_window_status": status,
        },
        resolve(config_path, config["paths"]["output_dir"]),
    )


if __name__ == "__main__":
    main()
