from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _next_diag_common import load_config, resolve


def maybe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    gz = Path(str(path) + ".gz")
    return pd.read_csv(gz) if gz.exists() else pd.DataFrame()


def verdict(values: pd.Series, predicate) -> str:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return "NOT_EVALUATED"
    return "PASS" if bool(predicate(finite).all()) else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize decomposition, placebo, calibration, and discordance diagnostics")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    placebo = maybe_read(output / "directional_placebo_summary.csv")
    flips = maybe_read(output / "sign_flip_summary.csv")
    calibration = maybe_read(output / "rolling_calibration.csv")
    weights = maybe_read(output / "rolling_stacking_weights.csv")
    discordance = maybe_read(output / "family_discordance_summary.csv")
    sources = maybe_read(output / "ta_source_summary.csv")
    decomposition = maybe_read(output / "ta_component_decomposition_summary.csv")
    yearly_decomp = maybe_read(output / "ta_component_decomposition_by_year.csv")
    filters = maybe_read(output / "sample_filter_diagnostics.csv")

    lines = [
        "# Next Diagnostics Report", "", "## Interpretation rules", "",
        "- Nonzero R is not equivalent to an auditor touching the engagement; material tolerances and raw TA shifts remain separate.",
        "- Absolute reduction is non-additive. PAT, CFO, and benchmark effects are allocated using an exact Shapley decomposition.",
        "- Jensen predicts non-positive placebo R only for conditionally mean-zero noise. Centered and symmetric placebos are the directional falsification tests.",
        "- NaN is reported as NOT_EVALUATED and is never converted into a False empirical verdict.", "",
    ]
    if not sources.empty:
        cash_share = sources.loc[sources["ta_source"].eq("cash_flow"), "share_within_status"].mean()
        missing_share = sources.loc[sources["ta_source"].eq("missing"), "share_within_status"].mean()
        lines += ["## TA construction", "", f"- Mean cash-flow-source share: {cash_share:.1%}.",
                  f"- Mean explicitly missing-source share: {missing_share:.2%}" if np.isfinite(missing_share) else "- Missing-source share unavailable.", ""]
    if not decomposition.empty:
        primary = decomposition[(decomposition["model"] == "modified_jones") & (decomposition["benchmark"] == "audited_reference")]
        lines += ["## PAT/CFO/benchmark decomposition", ""]
        if not primary.empty:
            lines += [primary.to_markdown(index=False), ""]
    if not filters.empty:
        lines += ["## Scale and reporting-boundary flags", "", filters.to_markdown(index=False), ""]
    if not flips.empty:
        hidden = flips[flips["flip_category"].eq("strict_flips_hidden_inside_R_near_zero")]
        lines += ["## Sign-flip audit", "",
                  f"- Median share hidden as sign flips inside R-near-zero bands: {hidden['share_all'].median():.1%}." if not hidden.empty else "- No sign-flip summary available.", ""]
    if not placebo.empty:
        centered = placebo[placebo["placebo_type"].isin(["centered_permutation", "symmetric_sign"])]
        mean_negative = verdict(centered["placebo_mean"], lambda x: x < 0) if not centered.empty else "NOT_EVALUATED"
        trimmed_negative = verdict(centered["placebo_trimmed_mean"], lambda x: x < 0) if not centered.empty else "NOT_EVALUATED"
        real_above = verdict(centered["real_mean_reduction"] - centered["placebo_q975"], lambda x: x > 0) if not centered.empty else "NOT_EVALUATED"
        real_trim_above = verdict(centered["real_trimmed_mean_reduction"] - centered["placebo_trimmed_q975"], lambda x: x > 0) if not centered.empty else "NOT_EVALUATED"
        finite_status = verdict(centered["all_outputs_finite"].astype(float), lambda x: x == 1) if not centered.empty else "NOT_EVALUATED"
        lines += ["## Directional placebo", "", f"- Finite-output check: **{finite_status}**.",
                  f"- All centered/symmetric untrimmed placebo means negative: **{mean_negative}**.",
                  f"- All centered/symmetric trimmed placebo means negative: **{trimmed_negative}**.",
                  f"- Real untrimmed means exceed placebo 97.5% quantiles: **{real_above}**.",
                  f"- Real trimmed means exceed placebo trimmed 97.5% quantiles: **{real_trim_above}**.", ""]
    if not calibration.empty:
        ensemble = calibration[(calibration["model"] == "stacked_ensemble") & (calibration["prediction_mode"] == "conditional_existing_firm") & (calibration["target_variant"] == "raw_target")]
        lines += ["## Rolling calibration and robust likelihood", ""]
        cols = [c for c in ["fiscal_year", "rmse", "mean_log_score", "best_robust_log_score", "robust_minus_gaussian_log_score", "coverage80", "coverage95"] if c in ensemble]
        if not ensemble.empty:
            lines += [ensemble[cols].to_markdown(index=False), ""]
    if not weights.empty:
        diagnostic_cols = [c for c in ["fiscal_year", "model", "weight", "optimizer_success", "optimizer_message", "effective_model_count", "stacking_objective", "equal_weight_objective", "best_single_objective"] if c in weights]
        lines += ["## Stacking diagnostics", "", weights[diagnostic_cols].to_markdown(index=False), ""]
    if not discordance.empty:
        lines += ["## Model-family discordance", "", discordance.to_markdown(index=False), ""]
    if not yearly_decomp.empty:
        primary_year = yearly_decomp[(yearly_decomp["model"] == "modified_jones") & (yearly_decomp["benchmark"] == "audited_reference")]
        lines += ["## Body time series after scale/scope exclusions", "", primary_year.to_markdown(index=False), ""]
    lines += [
        "## Decision sequence", "",
        "1. Verify repeated CFO-dominant candidates against original preliminary cash-flow statements.",
        "2. Decide whether CFO corrections belong in the primary construct or a separate transition branch.",
        "3. Require finite placebo outputs and report both matched untrimmed and trimmed randomization tests.",
        "4. Prefer robust likelihood only if raw-target Student-t log scores systematically improve on Gaussian scores.",
        "5. Use model-family discordant firm-years as the prespecified Bayesian classification target.",
    ]
    report = output / "NEXT_DIAGNOSTICS_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
