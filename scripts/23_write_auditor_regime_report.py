from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve


def read_table(output: Path, name: str) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def add_table(lines: list[str], title: str, table: pd.DataFrame, empty: str) -> None:
    lines += [f"## {title}", ""]
    lines += [table.to_markdown(index=False) if not table.empty else empty, ""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the Big4/non-Big4 auditor-regime heterogeneity report"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])

    source = read_table(output, "cfs_auditor_source_status")
    window = read_table(output, "cfs_auditor_analysis_window_status")
    status = read_table(output, "cfs_auditor_regime_status")
    coverage = read_table(output, "cfs_auditor_regime_coverage")
    metrics = read_table(output, "cfs_auditor_regime_metrics")
    differences = read_table(output, "cfs_auditor_regime_metric_differences")
    bootstrap = read_table(output, "cfs_auditor_regime_bootstrap")
    interaction = read_table(output, "cfs_auditor_regime_interaction")
    balance = read_table(output, "cfs_auditor_regime_balance")
    switches = read_table(output, "cfs_auditor_switch_summary")
    mapping = read_table(output, "cfs_auditor_name_mapping")

    focal = (
        interaction[interaction["focal_term"].eq(True)].copy()
        if not interaction.empty and "focal_term" in interaction
        else pd.DataFrame()
    )
    if not balance.empty and "standardized_mean_difference" in balance:
        balance = balance.assign(
            absolute_smd=pd.to_numeric(
                balance["standardized_mean_difference"], errors="coerce"
            ).abs()
        ).sort_values("absolute_smd", ascending=False)

    lines = [
        "# Auditor-Regime Heterogeneity in CFS Proxy Validation",
        "",
        "## Interpretation boundaries",
        "",
        "- The TT200 source and target-construction period is fiscal years 2015-2025.",
        "- Fiscal year 2015 is retained as the first source/warm-up year; out-of-sample criterion-validity tests begin in 2016 or the first later year satisfying the minimum-training gate.",
        "- Auditor metadata retain 2015 for consecutive-year switch histories, while Big4/non-Big4 score metrics use common-primary test observations from 2016 onward.",
        "- Auditor identity is joined after abnormal-CFO scores are generated and is never used to fit expected CFO.",
        "- Big4/non-Big4 results describe criterion-validity heterogeneity, not a causal effect of auditor choice.",
        "- Unknown and ambiguous auditor records are reported separately and are not silently coded as non-Big4.",
        "- The interaction of the outcome-specific score with Big4 is the focal test; the Big4 main effect alone does not establish audit quality.",
        "- Cluster bootstrap and logistic standard errors use issuer as the dependence unit.",
        "",
    ]
    add_table(lines, "Auditor source and schema", source, "No auditor source was evaluated.")
    add_table(lines, "Source and out-of-sample time contract", window, "No analysis-window status was produced.")
    add_table(lines, "Auditor-regime gate", status, "No auditor-regime status was produced.")
    add_table(lines, "Coverage in the primary analysis sample", coverage, "No coverage table was produced.")
    add_table(lines, "Criterion validity by auditor group", metrics, "No stratified metrics were produced.")
    add_table(lines, "Big4 minus non-Big4 metric differences", differences, "No group differences were produced.")
    add_table(lines, "Issuer-cluster bootstrap intervals", bootstrap, "No bootstrap intervals were produced.")
    add_table(lines, "Focal interaction coefficients", focal, "No interaction model was estimable.")
    add_table(lines, "Largest selection imbalances", balance.head(50), "No balance diagnostics were produced.")
    add_table(lines, "Auditor-tier switches", switches, "No auditor-tier switches were observed.")
    add_table(lines, "Auditor-name mapping review", mapping.head(100), "No auditor-name mapping was produced.")

    lines += [
        "## Decision rules",
        "",
        "1. Report stratified AUC, average precision, and lift only when both Big4 and non-Big4 groups satisfy the minimum sample gate.",
        "2. Treat a non-zero score-by-Big4 interaction as audit-regime-dependent criterion validity, conditional on controls and fixed effects.",
        "3. Do not interpret a prevalence difference as detection quality without accounting for client selection.",
        "4. Use the balance table and switch counts to delimit selection and within-firm evidence.",
        "5. Preserve the main pooled results when auditor coverage is incomplete; heterogeneity is an extension, not a sample filter.",
        "6. Do not describe 2015 as an out-of-sample model test year; it is the first TT200 source and warm-up year.",
    ]
    report = output / "CFS_AUDITOR_REGIME_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
