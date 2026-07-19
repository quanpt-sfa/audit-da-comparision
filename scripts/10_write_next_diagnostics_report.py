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
    parser = argparse.ArgumentParser(
        description="Synthesize CFS identity, component, placebo, calibration, and discordance diagnostics"
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])

    placebo = maybe_read(output / "directional_placebo_summary.csv")
    component_placebo = maybe_read(output / "component_placebo_summary.csv")
    component_anchor = maybe_read(output / "component_anchor_diagnostics.csv")
    cfo_tilt = maybe_read(output / "cfo_tilt_contrasts.csv")
    cfs_identity = maybe_read(output / "cfs_identity_by_year.csv")
    cfs_resolution = maybe_read(output / "cfs_candidate_resolution.csv")
    cfo_year = maybe_read(output / "cfo_magnitude_by_year.csv")
    invalid_tickers = maybe_read(output / "invalid_ticker_cases.csv")
    flips = maybe_read(output / "sign_flip_summary.csv")
    calibration = maybe_read(output / "rolling_calibration.csv")
    weights = maybe_read(output / "rolling_stacking_weights.csv")
    discordance = maybe_read(output / "family_discordance_summary.csv")
    sources = maybe_read(output / "ta_source_summary.csv")
    decomposition = maybe_read(output / "ta_component_decomposition_summary.csv")
    yearly_decomp = maybe_read(output / "ta_component_decomposition_by_year.csv")
    filters = maybe_read(output / "sample_filter_diagnostics.csv")

    lines = [
        "# Next Diagnostics Report",
        "",
        "## Interpretation rules",
        "",
        "- Nonzero R is not equivalent to an auditor targeting normal accruals.",
        "- Absolute reduction is non-additive; PAT, CFO, and benchmark effects use exact Shapley allocation.",
        "- CFS identity includes CFO, CFI, CFF, FX effects, and beginning/end cash from the cash-flow statement itself.",
        "- A pass-to-pass identity with offsetting CFI/CFF is a reclassification candidate, not proof of substantive audit correction.",
        "- Component placebos test PAT and CFO separately. Centered and symmetric variants carry the directional Jensen null.",
        "- Balance-sheet-accrual anchors are reported to distinguish targeting of an independent abnormality measure from mechanical removal of preliminary-CFO noise.",
        "- NaN is reported as NOT_EVALUATED and is never converted into a False empirical verdict.",
        "",
    ]

    if not sources.empty:
        cash_share = sources.loc[
            sources["ta_source"].eq("cash_flow"), "share_within_status"
        ].mean()
        missing_share = sources.loc[
            sources["ta_source"].eq("missing"), "share_within_status"
        ].mean()
        lines += [
            "## TA construction",
            "",
            f"- Mean cash-flow-source share: {cash_share:.1%}.",
            f"- Mean explicitly missing-source share: {missing_share:.2%}"
            if np.isfinite(missing_share)
            else "- Missing-source share unavailable.",
            "",
        ]

    if not decomposition.empty:
        primary = decomposition[
            decomposition["model"].eq("modified_jones")
            & decomposition["benchmark"].eq("audited_reference")
        ]
        lines += ["## PAT/CFO/benchmark decomposition", ""]
        if not primary.empty:
            lines += [primary.to_markdown(index=False), ""]

    if not cfs_identity.empty:
        identity_columns = [
            c for c in [
                "version",
                "fiscal_year",
                "availability_share",
                "full_identity_pass_share_available",
                "section_pass_share_available",
                "rollforward_pass_share_available",
                "median_abs_full_residual_scaled",
                "p95_abs_full_residual_scaled",
            ] if c in cfs_identity
        ]
        lines += [
            "## Cash-flow statement identity",
            "",
            cfs_identity[identity_columns].to_markdown(index=False),
            "",
        ]

    if not cfs_resolution.empty:
        lines += [
            "## CFO-case resolution",
            "",
            cfs_resolution.to_markdown(index=False),
            "",
        ]

    if not cfo_year.empty:
        lines += [
            "## CFO-adjustment magnitude over time",
            "",
            cfo_year.to_markdown(index=False),
            "",
        ]

    if not cfo_tilt.empty:
        primary_tilt = cfo_tilt[
            cfo_tilt["model"].eq("modified_jones")
            & cfo_tilt["benchmark"].eq("audited_reference")
            & cfo_tilt["sample"].eq("exclude_scale_scope_flags")
            & cfo_tilt["cfo_to_pat_threshold"].eq(5.0)
        ]
        lines += [
            "## Corrective tilt inside versus outside CFO-dominant engagements",
            "",
            primary_tilt.to_markdown(index=False)
            if not primary_tilt.empty
            else "No primary CFO-tilt contrast available.",
            "",
        ]

    if not component_placebo.empty:
        directional = component_placebo[
            component_placebo["placebo_type"].isin(
                ["centered_permutation", "symmetric_sign"]
            )
        ]
        lines += ["## Component-specific placebos", ""]
        for component in ["pat", "cfo"]:
            subset = directional[directional["component"].eq(component)]
            finite_status = (
                verdict(subset["all_outputs_finite"].astype(float), lambda x: x == 1)
                if not subset.empty else "NOT_EVALUATED"
            )
            real_above = (
                verdict(
                    subset["real_trimmed_mean_component_reduction"]
                    - subset["placebo_trimmed_q975"],
                    lambda x: x > 0,
                )
                if not subset.empty else "NOT_EVALUATED"
            )
            lines += [
                f"- **{component.upper()}** finite outputs: **{finite_status}**; real trimmed component effect exceeds placebo 97.5% quantile in every specification: **{real_above}**."
            ]
        lines += ["", component_placebo.to_markdown(index=False), ""]

    if not component_anchor.empty:
        primary_anchor = component_anchor[
            component_anchor["model"].eq("modified_jones")
            & component_anchor["benchmark"].eq("audited_reference")
        ]
        lines += [
            "## Independent-anchor diagnostic",
            "",
            primary_anchor.to_markdown(index=False)
            if not primary_anchor.empty
            else "No primary anchor diagnostic available.",
            "",
        ]

    if not filters.empty:
        lines += [
            "## Scale and reporting-boundary flags",
            "",
            filters.to_markdown(index=False),
            "",
        ]

    if not flips.empty:
        hidden = flips[
            flips["flip_category"].eq(
                "strict_flips_hidden_inside_R_near_zero"
            )
        ]
        lines += [
            "## Sign-flip audit",
            "",
            f"- Median share hidden as sign flips inside R-near-zero bands: {hidden['share_all'].median():.1%}."
            if not hidden.empty else "- No sign-flip summary available.",
            "",
        ]

    if not placebo.empty:
        centered = placebo[
            placebo["placebo_type"].isin(
                ["centered_permutation", "symmetric_sign"]
            )
        ]
        mean_negative = (
            verdict(centered["placebo_mean"], lambda x: x < 0)
            if not centered.empty else "NOT_EVALUATED"
        )
        trimmed_negative = (
            verdict(centered["placebo_trimmed_mean"], lambda x: x < 0)
            if not centered.empty else "NOT_EVALUATED"
        )
        real_above = (
            verdict(
                centered["real_mean_reduction"] - centered["placebo_q975"],
                lambda x: x > 0,
            )
            if not centered.empty else "NOT_EVALUATED"
        )
        real_trim_above = (
            verdict(
                centered["real_trimmed_mean_reduction"]
                - centered["placebo_trimmed_q975"],
                lambda x: x > 0,
            )
            if not centered.empty else "NOT_EVALUATED"
        )
        finite_status = (
            verdict(centered["all_outputs_finite"].astype(float), lambda x: x == 1)
            if not centered.empty else "NOT_EVALUATED"
        )
        lines += [
            "## Whole-adjustment directional placebo",
            "",
            f"- Finite-output check: **{finite_status}**.",
            f"- All centered/symmetric untrimmed placebo means negative: **{mean_negative}**.",
            f"- All centered/symmetric trimmed placebo means negative: **{trimmed_negative}**.",
            f"- Real untrimmed means exceed placebo 97.5% quantiles: **{real_above}**.",
            f"- Real trimmed means exceed placebo trimmed 97.5% quantiles: **{real_trim_above}**.",
            "",
        ]

    if not calibration.empty:
        ensemble = calibration[
            calibration["model"].eq("stacked_ensemble")
            & calibration["prediction_mode"].eq("conditional_existing_firm")
            & calibration["target_variant"].eq("raw_target")
        ]
        columns = [
            c for c in [
                "fiscal_year",
                "rmse",
                "mean_log_score",
                "best_robust_log_score",
                "robust_minus_gaussian_log_score",
                "coverage80",
                "coverage95",
            ] if c in ensemble
        ]
        lines += ["## Rolling calibration and robust likelihood", ""]
        if not ensemble.empty:
            lines += [ensemble[columns].to_markdown(index=False), ""]

    if not weights.empty:
        columns = [
            c for c in [
                "fiscal_year",
                "model",
                "weight",
                "optimizer_success",
                "optimizer_message",
                "effective_model_count",
                "stacking_objective",
                "equal_weight_objective",
                "best_single_objective",
            ] if c in weights
        ]
        lines += [
            "## Stacking diagnostics",
            "",
            weights[columns].to_markdown(index=False),
            "",
        ]

    if not discordance.empty:
        lines += [
            "## Model-family discordance",
            "",
            discordance.to_markdown(index=False),
            "",
        ]

    if not yearly_decomp.empty:
        primary_year = yearly_decomp[
            yearly_decomp["model"].eq("modified_jones")
            & yearly_decomp["benchmark"].eq("audited_reference")
        ]
        lines += [
            "## Body time series after scale/scope exclusions",
            "",
            primary_year.to_markdown(index=False),
            "",
        ]

    if not invalid_tickers.empty:
        lines += [
            "## Identifier hygiene",
            "",
            f"- Invalid or numeric-only ticker rows requiring review: {len(invalid_tickers):,}.",
            "",
        ]

    lines += [
        "## Decision sequence",
        "",
        "1. Use CFS section and roll-forward identities to separate internally inconsistent preliminary statements from identity-consistent offsetting reclassifications.",
        "2. Check whether the positive-minus-negative share advantage survives outside CFO-dominant engagements.",
        "3. Require PAT-only corrective excess against its own centered/symmetric placebo before retaining an earnings-normalization claim.",
        "4. Compare CFO alignment on cash-flow and balance-sheet-accrual anchors. Alignment confined to the cash-flow anchor supports mechanical noise removal rather than normal-targeting.",
        "5. Use original PDFs only for identity failures, missing components, and a small sample of identity-consistent reclassification candidates.",
        "6. Defer full Bayesian engagement classification until the primary construct is resolved.",
    ]

    report = output / "NEXT_DIAGNOSTICS_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
