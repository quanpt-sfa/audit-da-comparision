from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve


def maybe_read(output: Path, name: str) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def add_table(lines: list[str], title: str, table: pd.DataFrame, empty: str) -> None:
    lines += [f"## {title}", ""]
    lines += [table.to_markdown(index=False) if not table.empty else empty, ""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the observed CFS shifting validation report"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = config["cfs_shifting_validation"]

    inventory = maybe_read(output, "cfs_item_inventory")
    mapping_review = maybe_read(output, "cfs_item_mapping_review")
    coverage = maybe_read(output, "cfs_line_item_method_coverage")
    folds = maybe_read(output, "cfs_expected_cfo_folds")
    analysis_window = maybe_read(output, "cfs_analysis_window_status")
    time_contract = maybe_read(output, "cfs_time_contract_status")
    estimation_status = maybe_read(
        output, "cfs_expected_cfo_estimation_sample_status"
    )
    validation = maybe_read(output, "cfs_shifting_proxy_validation")
    yearly = maybe_read(output, "cfs_shifting_proxy_validation_by_year")
    incremental = maybe_read(output, "cfs_shifting_proxy_incremental_comparison")
    restrictions = maybe_read(output, "cfs_proxy_sample_restriction_status")
    common_status = maybe_read(output, "cfs_common_sample_status")
    common_comparison = maybe_read(output, "cfs_common_sample_metric_comparison")
    history_comparison = maybe_read(output, "cfs_history_incremental_comparison")
    industry_status = maybe_read(output, "cfs_industry_mapping_status")
    industry_unmatched = maybe_read(output, "cfs_industry_unmatched_tickers")
    reconciliation = maybe_read(
        output, "cfs_line_item_reconciliation_summary_common_primary_core"
    )
    top = maybe_read(
        output, "cfs_line_item_top_contributors_common_primary_core"
    )
    gate_status = maybe_read(output, "cfs_completion_gate_status")

    restriction_settings = settings.get("sample_restrictions", {})
    scale_scope_waived = not restriction_settings.get(
        "require_scale_scope_screening", True
    )
    scale_scope_reason = restriction_settings.get(
        "scale_scope_waiver_reason",
        "Scale/scope screening waived by design.",
    )

    lines = [
        "# Observed CFS Shifting Validation Report",
        "",
        "## Interpretation boundaries",
        "",
        "- The TT200 source and target-construction period is fiscal years 2015-2025.",
        "- Fiscal year 2015 is the first source/warm-up year; rolling out-of-sample tests begin in 2016 or the first later year satisfying the minimum-training gate.",
        "- Cross-model metrics use the common issuer-year intersection of all prespecified comparison models.",
        "- Observed preliminary-to-audited reclassification is a validation outcome, not direct evidence of managerial intent.",
        "- Expected-CFO models are fitted and tested only on the prespecified listed, valid-ticker, known non-financial population with positive lagged assets.",
        "- `common_primary_models` excludes firm-history requirements; `common_all_models` includes both standalone history and the nested EWC+history model.",
        "- Outcome-specific scores are mandatory: absolute residual for any revision, positive residual for CFO decreases/CFF-down, and negative residual for CFO increases/CFI-up.",
        "- Detailed line-item tables below are recomputed directly on the common-primary analysis-core firm-years.",
        "- Line-item labels and values are retained from source records verified during data construction; no separate PDF-verification gate is imposed.",
        "",
    ]

    if scale_scope_waived:
        lines += [
            "## Scale/scope design note",
            "",
            "- Status: `WAIVED_BY_DESIGN`.",
            f"- Rationale: {scale_scope_reason}",
            "- No observations are removed by an additional scale/scope filter.",
            "",
        ]

    add_table(
        lines,
        "Completion-gate status",
        gate_status,
        "No completion-gate status was produced.",
    )
    add_table(
        lines,
        "Source, training and out-of-sample window",
        analysis_window,
        "No analysis-window status was produced.",
    )
    add_table(
        lines,
        "Artifact-level TT200 time-contract check",
        time_contract,
        "No time-contract status was produced.",
    )
    add_table(
        lines,
        "Expected-CFO estimation population",
        estimation_status,
        "No estimation-sample status was produced.",
    )

    if not industry_status.empty:
        add_table(lines, "ICB industry mapping", industry_status, "")
        lines += [f"- Unmatched tickers: {len(industry_unmatched):,}.", ""]
        if not industry_unmatched.empty:
            lines += [industry_unmatched.head(100).to_markdown(index=False), ""]

    if not inventory.empty:
        status = inventory["mapping_status"].value_counts(dropna=False)
        rows = inventory.groupby("mapping_status", dropna=False)["rows"].sum()
        lines += [
            "## CFS item inventory",
            "",
            f"- Distinct source items: {len(inventory):,}.",
            f"- Mapped items: {int(status.get('mapped', 0)):,}; represented rows: {int(rows.get('mapped', 0)):,}.",
            f"- Unmapped items: {int(status.get('unmapped', 0)):,}; represented rows: {int(rows.get('unmapped', 0)):,}.",
            f"- Ambiguous items: {int(status.get('ambiguous', 0)):,}; represented rows: {int(rows.get('ambiguous', 0)):,}.",
            "",
        ]

    add_table(lines, "Selected CFS method coverage", coverage, "No method coverage table.")
    add_table(lines, "Sample-restriction status", restrictions, "No restriction status.")
    add_table(lines, "Common-sample definitions", common_status, "No common-sample status.")

    if not common_comparison.empty:
        common_core = common_comparison[
            common_comparison["sample_restriction"].eq("analysis_core")
        ].copy()
    else:
        common_core = pd.DataFrame()
    add_table(
        lines,
        "Primary versus all-model sample sensitivity",
        common_core,
        "No common-sample comparison was produced.",
    )
    add_table(
        lines,
        "Nested EWC plus firm-history incremental test",
        history_comparison,
        "No nested-history comparison was produced.",
    )

    if not folds.empty:
        fold_columns = [
            column
            for column in [
                "fiscal_year",
                "proxy_model",
                "train_rows",
                "test_rows",
                "source_start_year",
                "source_end_year",
                "training_start_year",
                "test_start_year",
                "test_end_year",
                "source_panel_minimum_year_actual",
                "source_panel_maximum_year_actual",
                "rmse",
                "winsorized_rmse",
                "rmse_ex_top_1pct",
                "mae",
                "median_absolute_error",
                "p95_absolute_error",
                "p99_absolute_error",
                "maximum_absolute_error",
                "maximum_error_issuer",
                "status",
            ]
            if column in folds.columns
        ]
        add_table(
            lines,
            "Rolling expected-CFO folds",
            folds[fold_columns],
            "No fold diagnostics were produced.",
        )

    if not validation.empty:
        primary = validation[
            validation["sample_mode"].eq("common_primary_models")
            & validation["sample_restriction"].eq("analysis_core")
        ].copy()
        all_models = validation[
            validation["sample_mode"].eq("common_all_models")
            & validation["sample_restriction"].eq("analysis_core")
        ].copy()
    else:
        primary = pd.DataFrame()
        all_models = pd.DataFrame()
    add_table(lines, "Primary common-sample validation", primary, "No primary results.")
    add_table(lines, "All-model common-sample validation", all_models, "No all-model results.")

    if not incremental.empty:
        incremental_columns = [
            column
            for column in [
                "proxy_model",
                "outcome",
                "auc",
                "reference_auc",
                "delta_auc_vs_reference",
                "average_precision",
                "reference_average_precision",
                "delta_ap_vs_reference",
                "top_decile_lift",
                "reference_top_decile_lift",
                "delta_lift_vs_reference",
            ]
            if column in incremental.columns
        ]
        for mode, title in [
            ("common_primary_models", "Incremental validity on the primary common sample"),
            ("common_all_models", "Incremental validity on the all-model common sample"),
        ]:
            table = incremental[
                incremental["sample_mode"].eq(mode)
                & incremental["sample_restriction"].eq("analysis_core")
            ].copy()
            add_table(
                lines,
                title,
                table[incremental_columns] if not table.empty else table,
                "No incremental comparison was produced.",
            )

    if not yearly.empty:
        temporal = yearly[
            yearly["sample_mode"].eq("common_primary_models")
            & yearly["sample_restriction"].eq("analysis_core")
        ].copy()
    else:
        temporal = pd.DataFrame()
    add_table(
        lines,
        "Temporal stability on the primary sample",
        temporal,
        "No annual primary results were produced.",
    )
    add_table(
        lines,
        "Common-primary/core line-item reconciliation",
        reconciliation,
        "No common-primary/core reconciliation was produced.",
    )
    add_table(
        lines,
        "Common-primary/core mapped contributors",
        top.head(100),
        "No common-primary/core contributor table was produced.",
    )

    if not mapping_review.empty:
        review = mapping_review.sort_values("rows", ascending=False).head(50)
        lines += [
            "## Residual mapping-review obligation",
            "",
            f"- Items requiring manual mapping review: {len(mapping_review):,}.",
            review.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Decision rules",
        "",
        "1. The expected-CFO coefficients are admissible only when the estimation-population and TT200 time-contract gates pass.",
        "2. Fiscal year 2015 is not described as an out-of-sample model test year.",
        "3. `any_candidate` uses the absolute residual; signed residuals are reserved for directional outcomes.",
        "4. The nested history model is retained only if it improves EWC on the identical all-model firm-year sample.",
        "5. Main mechanism claims use the common-primary/core source-record reconciliation outputs, not full-universe contributor tables.",
        "6. Scale/scope screening is waived by design and disclosed as a maintained source-consistency assumption.",
        "7. No separate PDF-verification requirement is imposed because retained source-record labels and values were verified during data construction.",
    ]

    report = output / "CFS_SHIFTING_VALIDATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
