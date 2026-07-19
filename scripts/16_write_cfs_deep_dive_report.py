from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve


def maybe_read(output: Path, name: str) -> pd.DataFrame:
    plain = output / f"{name}.csv"
    compressed = output / f"{name}.csv.gz"
    try:
        if plain.exists():
            return pd.read_csv(plain, low_memory=False)
        if compressed.exists():
            return pd.read_csv(compressed, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the CFS offset, persistence, incentive, and audit-quality report"
    )
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])

    offset = maybe_read(output, "cfs_offset_channel_summary")
    offset_year = maybe_read(output, "cfs_offset_channel_by_year")
    chronic = maybe_read(output, "chronic_reclassifier_profiles")
    incentive = maybe_read(output, "cfs_incentive_descriptives")
    models = maybe_read(output, "cfs_incentive_models")
    anchors = maybe_read(output, "component_anchor_common_sample")
    uncertainty_year = maybe_read(output, "cfs_payoff_uncertainty_by_year")
    uncertainty_bin = maybe_read(output, "cfs_payoff_by_uncertainty_bin")
    uncertainty_corr = maybe_read(output, "cfs_payoff_uncertainty_correlations")
    audit_status = maybe_read(output, "audit_quality_status")
    audit_summary = maybe_read(output, "audit_quality_summary")
    verification = maybe_read(output, "cfs_pdf_verification_sample")

    lines = [
        "# CFS Deep-Dive Report",
        "",
        "## Interpretation boundaries",
        "",
        "- Algebraically consistent offsetting movements are reclassification candidates; semantic verification still requires source-document checks.",
        "- A two-sided CFO adjustment distribution is evidence against a universal upward-window-dressing story, not evidence that no incentive-driven subset exists.",
        "- Common-sample anchor comparisons are the primary targeting-versus-de-noising diagnostic.",
        "- A relation between payoff and model dispersion is descriptive evidence of benchmark dependence, not a causal estimate of audit quality.",
        "- Big4 and audit-opinion results are reported only when an external metadata file joins successfully.",
        "",
    ]

    if not offset.empty:
        lines += ["## Offset destination: CFI versus CFF", "", offset.to_markdown(index=False), ""]
    if not offset_year.empty:
        lines += ["## Offset channel by year and CFO direction", "", offset_year.to_markdown(index=False), ""]
    if not chronic.empty:
        chronic_rows = chronic[chronic["chronic_reclassifier"].fillna(False)]
        direction = chronic_rows["direction_type"].value_counts(dropna=False)
        lines += [
            "## Chronic reclassifiers",
            "",
            f"- Chronic issuers under the prespecified rule: {len(chronic_rows):,}.",
            f"- Bidirectional chronic issuers: {int(direction.get('bidirectional', 0)):,}.",
            f"- Mostly audited-CFO-decrease chronic issuers: {int(direction.get('mostly_audited_decrease', 0)):,}.",
            f"- Mostly audited-CFO-increase chronic issuers: {int(direction.get('mostly_audited_increase', 0)):,}.",
            "",
            chronic.head(100).to_markdown(index=False),
            "",
        ]
    if not incentive.empty:
        lines += ["## Incentive asymmetry descriptives", "", incentive.to_markdown(index=False), ""]
    if not models.empty and "term" in models:
        core = models[
            models["term"].isin([
                "pre_loss", "pre_negative_cfo", "pre_near_zero_cfo",
                "pre_liquidity_stress", "pre_low_cash", "pre_roa",
                "pre_short_debt_scaled", "pre_current_ratio", "log_lag_assets",
            ])
        ]
        lines += ["## Firm-clustered incentive models", "", core.to_markdown(index=False), ""]
    if not anchors.empty:
        lines += ["## Same-firm-year anchor comparison", "", anchors.to_markdown(index=False), ""]
    if not uncertainty_year.empty:
        lines += [
            "## Reclassification payoff and measurement uncertainty by year",
            "",
            uncertainty_year.to_markdown(index=False),
            "",
        ]
    if not uncertainty_bin.empty:
        lines += [
            "## Candidate payoff across uncertainty bins",
            "",
            uncertainty_bin.to_markdown(index=False),
            "",
        ]
    if not uncertainty_corr.empty:
        lines += [
            "## Year-specific payoff correlations",
            "",
            uncertainty_corr.to_markdown(index=False),
            "",
        ]
    if not audit_status.empty:
        lines += ["## Audit-quality metadata status", "", audit_status.to_markdown(index=False), ""]
    if not audit_summary.empty:
        lines += ["## Big4 and audit-opinion splits", "", audit_summary.to_markdown(index=False), ""]
    if not verification.empty:
        lines += [
            "## Source-document verification sample",
            "",
            f"- Stratified cases selected: {len(verification):,}.",
            "- Verify the preliminary and audited CFO, CFI, CFF, FX, net cash change, and beginning/end cash values and their semantic labels.",
            "",
            verification.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Decision rules",
        "",
        "1. CFI dominance supports operating-versus-investing classification as the main institutional margin; CFF dominance points toward financing classifications such as interest, dividends, and borrowing-related cash flows.",
        "2. Bidirectional chronicity supports a stable reporting-process or classification-policy problem; persistent audited CFO decreases support a window-dressing subset.",
        "3. Distress and near-zero-CFO coefficients must predict audited CFO decreases, not merely candidate incidence, before claiming incentive-driven inflation.",
        "4. The targeting claim remains rejected if CFO movement is positive only on the cash-flow anchor and non-positive on the balance-sheet anchor within the identical complete-case sample.",
        "5. A falling payoff with rising cross-specification dispersion supports benchmark contamination of the measured audit effect; a flat relationship points instead to a real change in the underlying reporting transition.",
        "6. Big4/opinion associations are descriptive until client selection and pre-audit reporting quality are addressed.",
    ]

    report = output / "CFS_DEEP_DIVE_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
