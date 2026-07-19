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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the observed CFS shifting validation report"
    )
    parser.add_argument(
        "--config", default="config/cfs_shifting_validation.yaml"
    )
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])

    inventory = maybe_read(output, "cfs_item_inventory")
    mapping_review = maybe_read(output, "cfs_item_mapping_review")
    coverage = maybe_read(output, "cfs_line_item_method_coverage")
    folds = maybe_read(output, "cfs_expected_cfo_folds")
    validation = maybe_read(output, "cfs_shifting_proxy_validation")
    yearly = maybe_read(output, "cfs_shifting_proxy_validation_by_year")
    incremental = maybe_read(
        output, "cfs_shifting_proxy_incremental_comparison"
    )
    restrictions = maybe_read(
        output, "cfs_proxy_sample_restriction_status"
    )
    reconciliation = maybe_read(
        output, "cfs_line_item_reconciliation_summary"
    )
    top = maybe_read(output, "cfs_line_item_top_contributors")

    lines = [
        "# Observed CFS Shifting Validation Report",
        "",
        "## Interpretation boundaries",
        "",
        "- Observed preliminary-to-audited reclassification is a validation outcome, not direct evidence of managerial intent.",
        "- Outcome-specific scores are mandatory: absolute residual for any revision, positive residual for CFO decreases/CFF-down, and negative residual for CFO increases/CFI-up.",
        "- Comparisons across proxy models use the common-model firm-year sample; model-available rows are retained only as coverage diagnostics.",
        "- Raw CFO, within-year CFO percentile, firm-history deviation, and sales-only models are explicit baselines.",
        "- Detailed line-item contributor tables contain reclassification candidates only; the all-resolution table is a separate audit output.",
        "- Detailed institutional conclusions remain provisional until high-coverage unmapped items and selected source documents are checked.",
        "",
    ]

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

    if not coverage.empty:
        lines += [
            "## Selected CFS method coverage",
            "",
            coverage.to_markdown(index=False),
            "",
        ]

    if not restrictions.empty:
        lines += [
            "## Sample-restriction status",
            "",
            restrictions.to_markdown(index=False),
            "",
        ]

    if not folds.empty:
        columns = [
            column
            for column in [
                "fiscal_year",
                "proxy_model",
                "train_rows",
                "test_rows",
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
        lines += [
            "## Rolling expected-CFO folds",
            "",
            folds[columns].to_markdown(index=False),
            "",
        ]

    if not validation.empty:
        core = validation[
            validation["sample_mode"].eq("common_models")
            & validation["sample_restriction"].eq("analysis_core")
        ].copy()
        lines += [
            "## Primary common-sample validation",
            "",
            core.to_markdown(index=False)
            if not core.empty
            else "No common-model analysis-core results were produced.",
            "",
        ]

    if not incremental.empty:
        core_incremental = incremental[
            incremental["sample_mode"].eq("common_models")
            & incremental["sample_restriction"].eq("analysis_core")
        ].copy()
        columns = [
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
            if column in core_incremental.columns
        ]
        lines += [
            "## Incremental validity over raw CFO",
            "",
            core_incremental[columns].to_markdown(index=False)
            if not core_incremental.empty
            else "No incremental comparison was produced.",
            "",
        ]

    if not yearly.empty:
        temporal = yearly[
            yearly["sample_mode"].eq("common_models")
            & yearly["sample_restriction"].eq("analysis_core")
        ].copy()
        lines += [
            "## Temporal stability on the primary sample",
            "",
            temporal.to_markdown(index=False)
            if not temporal.empty
            else "No annual common-sample results were produced.",
            "",
        ]

    if not reconciliation.empty:
        candidate_label = config["cfs_shifting_validation"].get(
            "candidate_label",
            "identity_consistent_offsetting_reclassification_candidate",
        )
        candidate_reconciliation = reconciliation[
            reconciliation["cfs_resolution"].eq(candidate_label)
        ]
        lines += [
            "## Candidate-only detailed line-item reconciliation",
            "",
            candidate_reconciliation.to_markdown(index=False)
            if not candidate_reconciliation.empty
            else "No candidate reconciliation results were produced.",
            "",
        ]

    if not top.empty:
        lines += [
            "## Largest mapped contributors among candidates",
            "",
            top.head(100).to_markdown(index=False),
            "",
        ]

    if not mapping_review.empty:
        review = mapping_review.sort_values("rows", ascending=False).head(50)
        lines += [
            "## Mapping review obligation",
            "",
            f"- Items requiring manual mapping review: {len(mapping_review):,}.",
            "- Review the highest-coverage rows before naming a borrowing, lease, dividend, lending, or investment mechanism.",
            "",
            review.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Decision rules",
        "",
        "1. `any_candidate` uses the absolute residual; using the signed residual mechanically cancels positive- and negative-tail reclassifications.",
        "2. A proxy supports upward preliminary-CFO shifting only if the positive score predicts audited CFO decreases, especially CFF-dominant decreases.",
        "3. Prediction of CFI-dominant increases by the negative score indicates a bidirectional classification-reliability construct, not a one-sided manipulation construct.",
        "4. Expected-CFO models must improve on raw CFO and within-year percentile baselines on the same firm-year sample.",
        "5. Raw RMSE is not interpreted without winsorized RMSE, MAE, median error, tail diagnostics, and the maximum-error issuer.",
        "6. Main claims require stability in listed, valid-ticker, lag-asset-screened, scale/scope-clean, and non-financial samples where those fields are available.",
        "7. Line-item mechanisms are named only when mapped lines reconcile materially to aggregate CFI/CFF changes and source-document checks confirm semantic labels.",
    ]

    report = output / "CFS_SHIFTING_VALIDATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
