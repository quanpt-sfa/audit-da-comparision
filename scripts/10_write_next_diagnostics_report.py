from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _next_diag_common import load_config, resolve


def maybe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    gz = Path(str(path) + ".gz")
    return pd.read_csv(gz) if gz.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize tail, transition, placebo, calibration, and discordance diagnostics")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    placebo = maybe_read(output / "directional_placebo_summary.csv")
    flips = maybe_read(output / "sign_flip_summary.csv")
    calibration = maybe_read(output / "rolling_calibration.csv")
    discordance = maybe_read(output / "family_discordance_summary.csv")
    sources = maybe_read(output / "ta_source_summary.csv")
    lines = ["# Next Diagnostics Report", "", "## Interpretation corrections", "",
        "- Nonzero R is not equivalent to an auditor touching the engagement; use material tolerances and raw TA shifts.",
        "- Jensen predicts non-positive placebo R only for conditionally mean-zero noise. Centered and symmetric placebos are the directional falsification tests; raw permutation is descriptive.",
        "- Common-benchmark placebos are valid because signed DA movement equals raw TA movement under a fixed normal-accrual reference.", ""]
    if not sources.empty:
        cash_share = sources.loc[sources["ta_source"].eq("cash_flow"), "share_within_status"].mean()
        lines += ["## TA construction", "", f"- Mean cash-flow-source share across statuses: {cash_share:.1%}.", ""]
    if not flips.empty:
        hidden = flips[flips["flip_category"].eq("strict_flips_hidden_inside_R_near_zero")]
        lines += ["## Sign-flip point-mass audit", "", f"- Median share of all observations hidden as sign flips inside the selected R-near-zero bands: {hidden['share_all'].median():.1%}." if not hidden.empty else "- No sign-flip summary available.", ""]
    if not placebo.empty:
        centered = placebo[placebo["placebo_type"].isin(["centered_permutation", "symmetric_sign"])]
        lines += ["## Directional placebo", "", f"- All centered/symmetric placebo means negative: {bool((centered['placebo_mean'] < 0).all()) if not centered.empty else 'not available'}.", f"- All real means exceed the 97.5% placebo quantile: {bool((centered['real_mean_reduction'] > centered['placebo_q975']).all()) if not centered.empty else 'not available'}.", ""]
    if not calibration.empty:
        ensemble = calibration[(calibration["model"] == "stacked_ensemble") & (calibration["prediction_mode"] == "conditional_existing_firm") & (calibration["target_variant"] == "raw_target")]
        lines += ["## Rolling calibration", ""]
        if not ensemble.empty:
            lines += [ensemble[["fiscal_year", "rmse", "mean_log_score", "coverage80", "coverage95"]].to_markdown(index=False), ""]
    if not discordance.empty:
        lines += ["## Model-family discordance", "", discordance.to_markdown(index=False), ""]
    lines += ["## Decision logic", "", "1. Stop and repair data if tails are concentrated in source mismatches, formula failures, or anomalous 2024 records.", "2. Retain transition typology if sign flips materially populate R-near-zero cells.", "3. Treat centered/symmetric placebo excess as the primary falsification result.", "4. Attribute the 2021-22 collapse to benchmark failure only if rolling predictive calibration deteriorates in those years.", "5. Use family-discordant firm-years as the prespecified target population for Bayesian model-uncertainty resolution."]
    report = output / "NEXT_DIAGNOSTICS_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
