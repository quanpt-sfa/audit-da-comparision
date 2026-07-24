#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else f"{number:.{digits}f}"


def _decision(rmse: pd.Series, mae: pd.Series) -> str:
    rmse_estimate = float(rmse["estimate"])
    mae_estimate = float(mae["estimate"])
    rmse_high = float(rmse["ci_high"])
    mae_high = float(mae["ci_high"])
    rmse_low = float(rmse["ci_low"])
    mae_low = float(mae["ci_low"])
    if rmse_estimate < 0 and mae_estimate < 0:
        if rmse_high < 0 and mae_high < 0:
            return "SUPPORTS_AUDITED"
        return "DIRECTIONAL_AUDITED"
    if rmse_estimate > 0 and mae_estimate > 0:
        if rmse_low > 0 and mae_low > 0:
            return "SUPPORTS_PRE"
        return "DIRECTIONAL_PRE"
    return "MIXED"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a decision-oriented report from predictive-validity outputs"
    )
    parser.add_argument(
        "--config",
        default="config/predictive_validity.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = resolve(config_path, config["paths"]["output_dir"])
    required = {
        "coefficients": output_dir / "predictive_validity_coefficients.csv",
        "oos_summary": output_dir / "predictive_validity_oos_summary.csv",
        "oos_difference": output_dir
        / "predictive_validity_oos_state_differences.csv",
        "aq_summary": output_dir / "accrual_quality_summary.csv",
        "aq_difference": output_dir / "accrual_quality_state_differences.csv",
        "sample": output_dir / "predictive_validity_sample_manifest.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Predictive-validity outputs are incomplete: " + ", ".join(missing)
        )

    coefficients = pd.read_csv(required["coefficients"])
    oos_summary = pd.read_csv(required["oos_summary"])
    oos_difference = pd.read_csv(required["oos_difference"])
    aq_summary = pd.read_csv(required["aq_summary"])
    aq_difference = pd.read_csv(required["aq_difference"])
    sample = pd.read_csv(required["sample"])

    decision_rows: list[list[str]] = []
    for test_name, group in oos_difference.groupby("test", observed=True):
        rmse = group.loc[group.metric.eq("rmse")].iloc[0]
        mae = group.loc[group.metric.eq("mae")].iloc[0]
        r2 = group.loc[group.metric.eq("r2_oos")].iloc[0]
        decision_rows.append(
            [
                str(test_name),
                _fmt(rmse.estimate),
                f"[{_fmt(rmse.ci_low)}, {_fmt(rmse.ci_high)}]",
                _fmt(mae.estimate),
                f"[{_fmt(mae.ci_low)}, {_fmt(mae.ci_high)}]",
                _fmt(r2.estimate),
                _decision(rmse, mae),
            ]
        )

    coefficient_rows: list[list[str]] = []
    canonical = coefficients.loc[
        coefficients.specification.eq("canonical")
        & coefficients.term.ne("intercept")
    ].copy()
    for (test_name, term), group in canonical.groupby(
        ["test", "term"], observed=True
    ):
        pre = group.loc[group.contrast.eq("pre")].iloc[0]
        audited = group.loc[group.contrast.eq("audited")].iloc[0]
        difference = group.loc[group.contrast.eq("audited_minus_pre")].iloc[0]
        coefficient_rows.append(
            [
                str(test_name),
                str(term),
                _fmt(pre.estimate),
                _fmt(audited.estimate),
                _fmt(difference.estimate),
                _fmt(difference.p_value),
            ]
        )

    aq_rmse = aq_difference.loc[aq_difference.metric.eq("rmse")].iloc[0]
    aq_mae = aq_difference.loc[aq_difference.metric.eq("mae")].iloc[0]
    aq_sd = aq_difference.loc[aq_difference.metric.eq("residual_sd")].iloc[0]
    aq_decision = _decision(aq_rmse, aq_mae)

    sample_rows = [
        [
            str(row.analysis),
            str(int(row.rows)),
            str(int(row.issuers)),
            _fmt(row.year_min, 0),
            _fmt(row.year_max, 0),
        ]
        for row in sample.itertuples(index=False)
    ]

    lines = [
        "# Reporting-state predictive-validity results",
        "",
        "Negative audited-minus-pre RMSE and MAE differences favour the audited annual state. Decisions require the two loss measures to point in the same direction; `SUPPORTS_*` additionally requires both 95% cluster-bootstrap intervals to exclude zero.",
        "",
        "## Out-of-sample evidence",
        "",
        _table(
            [
                "Test",
                "ΔRMSE",
                "95% CI",
                "ΔMAE",
                "95% CI",
                "ΔOOS R²",
                "Decision",
            ],
            decision_rows,
        ),
        "",
        "## Canonical coefficient comparisons",
        "",
        _table(
            ["Test", "Term", "Pre", "Audited", "Audited−pre", "p-value"],
            coefficient_rows,
        ),
        "",
        "Coefficient differences describe persistence or mapping changes; the OOS loss comparisons remain the primary predictive-validity criterion.",
        "",
        "## Accrual-quality robustness",
        "",
        _table(
            ["Metric", "Audited−pre", "95% CI", "p-value"],
            [
                [
                    "RMSE",
                    _fmt(aq_rmse.estimate),
                    f"[{_fmt(aq_rmse.ci_low)}, {_fmt(aq_rmse.ci_high)}]",
                    _fmt(aq_rmse.p_two_sided),
                ],
                [
                    "MAE",
                    _fmt(aq_mae.estimate),
                    f"[{_fmt(aq_mae.ci_low)}, {_fmt(aq_mae.ci_high)}]",
                    _fmt(aq_mae.p_two_sided),
                ],
                [
                    "Residual SD",
                    _fmt(aq_sd.estimate),
                    f"[{_fmt(aq_sd.ci_low)}, {_fmt(aq_sd.ci_high)}]",
                    _fmt(aq_sd.p_two_sided),
                ],
            ],
        ),
        "",
        f"Accrual-quality decision: **{aq_decision}**. This is robustness evidence, not a ground-truth classification of reporting error.",
        "",
        "## Samples",
        "",
        _table(
            ["Analysis", "Rows", "Issuers", "Year min", "Year max"],
            sample_rows,
        ),
        "",
    ]

    report_path = output_dir / "predictive_validity_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[predictive-validity-report] wrote {report_path}")


if __name__ == "__main__":
    main()
