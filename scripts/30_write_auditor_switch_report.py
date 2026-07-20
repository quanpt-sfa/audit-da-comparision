from __future__ import annotations

import argparse

import pandas as pd

from _auditor_switch_common import load_context, read_table


def add_table(
    lines: list[str], title: str, table: pd.DataFrame, empty: str
) -> None:
    lines.extend([f"## {title}", ""])
    lines.extend(
        [table.to_markdown(index=False) if not table.empty else empty, ""]
    )


def focal_rows(table: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    result = table.copy()
    if "outcome" in result.columns:
        result = result[result["outcome"].isin(outcomes)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write auditor-switch event-study, dynamic-DiD and "
            "temporal-heterogeneity report"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    _, _, output, _, _ = load_context(args.config)

    event_status = read_table(
        output, "cfs_auditor_switch_event_study_status", False
    )
    event_summary = read_table(
        output, "cfs_auditor_switch_event_diagnostics", False
    )
    support = read_table(output, "cfs_auditor_switch_stack_support", False)
    event = read_table(output, "cfs_auditor_switch_event_study", False)
    pretrend = read_table(
        output, "cfs_auditor_switch_pretrend_tests", False
    )
    borrowing = read_table(
        output, "cfs_auditor_switch_borrowing_heterogeneity", False
    )
    did_status = read_table(
        output, "cfs_auditor_switch_dynamic_did_status", False
    )
    did = read_table(output, "cfs_auditor_switch_dynamic_did", False)
    did_pretrend = read_table(
        output, "cfs_auditor_switch_dynamic_did_pretrend", False
    )
    overlap = read_table(
        output, "cfs_auditor_switch_overlap_balance", False
    )
    auc_status = read_table(
        output, "cfs_yearly_auc_heterogeneity_status", False
    )
    yearly = read_table(output, "cfs_yearly_auc_metrics", False)
    q_test = read_table(output, "cfs_yearly_auc_generalized_q", False)
    year_wald = read_table(output, "cfs_score_by_year_joint_tests", False)

    lines = [
        "# Auditor Switch Event Study, Dynamic DiD and Temporal Heterogeneity",
        "",
        "## Interpretation boundaries",
        "",
        "- Event time zero is the first fiscal year audited by the new auditor tier.",
        "- Upgrade and downgrade events are estimated separately.",
        "- The primary stacked event study uses stable same-tier controls within each event stack and issuer-by-stack plus calendar-year-by-stack fixed effects.",
        "- Dynamic DiD compares switchers with stable stayers and treats Big4 status as reversible. Voluntary switching remains endogenous; results are dynamic associations unless parallel trends and no anticipation are credible.",
        "- Borrowing-intensity subgroups use preliminary borrowing proceeds measured before the switch, not audited post-switch values.",
        "- Generalized AUC heterogeneity uses issuer-cluster bootstrap covariance; score-by-year interactions provide a separate clustered Wald test.",
        "",
    ]
    add_table(
        lines,
        "Event-study execution status",
        event_status,
        "No event-study status was produced.",
    )
    add_table(
        lines,
        "Clean-switch event counts",
        event_summary.groupby(
            ["switch_direction", "event_status"], observed=True
        )
        .agg(
            events=("event_id", "size"),
            issuers=("issuer_ticker", "nunique"),
        )
        .reset_index()
        if not event_summary.empty
        else pd.DataFrame(),
        "No switch events were produced.",
    )
    add_table(
        lines,
        "Stack support",
        support,
        "No event-stack support table was produced.",
    )
    add_table(
        lines,
        "Primary event-study estimates",
        focal_rows(
            event,
            [
                "any_candidate",
                "cff_down_candidate",
                "cfi_up_candidate",
                "signed_cfo_correction",
            ],
        ),
        "No event-study estimates were produced.",
    )
    add_table(
        lines,
        "Event-study pretrend tests",
        pretrend,
        "No event-study pretrend tests were produced.",
    )
    add_table(
        lines,
        "Pre-event borrowing-intensity heterogeneity",
        focal_rows(
            borrowing,
            ["cff_down_candidate", "signed_cfo_correction"],
        ),
        "Borrowing-intensity heterogeneity was not estimable.",
    )
    add_table(
        lines,
        "Dynamic-DiD execution status",
        did_status,
        "No dynamic-DiD status was produced.",
    )
    add_table(
        lines,
        "Switcher-versus-stayer dynamic DiD",
        focal_rows(
            did,
            [
                "any_candidate",
                "cff_down_candidate",
                "cfi_up_candidate",
                "signed_cfo_correction",
            ],
        ),
        "No dynamic-DiD estimates were produced.",
    )
    add_table(
        lines,
        "Dynamic-DiD placebo pretrends",
        did_pretrend,
        "No dynamic-DiD pretrend tests were produced.",
    )
    add_table(
        lines,
        "Overlap-weight balance",
        overlap,
        "No overlap-balance diagnostics were produced.",
    )
    add_table(
        lines,
        "Yearly-AUC execution status",
        auc_status,
        "No yearly-AUC status was produced.",
    )
    add_table(
        lines,
        "Yearly AUC metrics",
        yearly,
        "No yearly AUC metrics were produced.",
    )
    add_table(
        lines,
        "Generalized AUC heterogeneity tests",
        q_test,
        "No generalized AUC-Q test was produced.",
    )
    add_table(
        lines,
        "Joint score-by-year Wald tests",
        year_wald,
        "No score-by-year Wald test was produced.",
    )

    lines.extend(
        [
            "## Decision rules",
            "",
            "1. Do not interpret a post-switch coefficient causally when the joint lead/placebo test rejects parallel pre-trends.",
            "2. A discipline interpretation is strongest when downgrade and upgrade estimates have opposite signs in the borrowing-related CFF channel, survive stable-control and overlap-weighted analyses, and are not mirrored in the heterogeneous CFI channel.",
            "3. If deterioration begins before a Big4 downgrade, interpret the pattern as endogenous sorting or auditor-client realignment rather than an auditor treatment effect.",
            "4. Retain the claim that temporal heterogeneity exceeds auditor heterogeneity only when either the generalized AUC-Q or the joint score-by-year Wald test rejects homogeneous ranking validity.",
        ]
    )
    report = output / "CFS_AUDITOR_SWITCH_EVENT_DID_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
