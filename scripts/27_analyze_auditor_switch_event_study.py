from __future__ import annotations

import argparse

import pandas as pd

from _auditor_switch_common import (
    load_context,
    read_table,
    restrict_years,
    update_completion_gate,
)
from audit_da.auditor_switch_event_study import (
    build_stacked_switch_sample,
    identify_clean_switch_events,
    prepare_switch_analysis_panel,
    run_switch_event_study,
)
from audit_da.diag_common import write_tables


EVENT_OUTPUTS = (
    "cfs_auditor_switch_direct_panel",
    "cfs_auditor_switch_line_item_status",
    "cfs_auditor_switch_event_diagnostics",
    "cfs_auditor_switch_stack_support",
    "cfs_auditor_switch_stacked_sample",
    "cfs_auditor_switch_event_study",
    "cfs_auditor_switch_pretrend_tests",
    "cfs_auditor_switch_borrowing_heterogeneity",
    "cfs_auditor_switch_borrowing_pretrend",
    "cfs_auditor_switch_borrowing_status",
    "cfs_auditor_switch_event_study_status",
)


def remove_stale_outputs(output) -> None:
    for name in EVENT_OUTPUTS:
        for suffix in (".csv", ".csv.gz"):
            path = output / f"{name}{suffix}"
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run direction-specific stacked event studies around auditor-tier "
            "switches"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()

    _, _, output, cfs_settings, auditor_settings = load_context(args.config)
    switch = dict(auditor_settings.get("switch_event_study", {}))
    if not switch.get("enabled", True):
        print("Auditor switch event study disabled by configuration")
        return

    direct_cases = read_table(
        output, switch.get("direct_case_table", "cfs_offset_channel_cases")
    )
    firm_year = read_table(output, "cfs_auditor_firm_year")
    line_items = read_table(
        output,
        switch.get("line_item_table", "cfs_line_item_panel"),
        required=False,
    )

    direct_cases = restrict_years(
        direct_cases, auditor_settings, use_source_window=True
    )
    firm_year = restrict_years(
        firm_year, auditor_settings, use_source_window=True
    )
    line_items = restrict_years(
        line_items, auditor_settings, use_source_window=True
    )

    panel, line_status = prepare_switch_analysis_panel(
        direct_cases,
        firm_year,
        auditor_settings,
        cfs_settings=cfs_settings,
        line_item_panel=line_items,
    )
    events = identify_clean_switch_events(firm_year, auditor_settings)
    stacked, support = (
        build_stacked_switch_sample(panel, events, auditor_settings)
        if not events.empty and not panel.empty
        else (pd.DataFrame(), pd.DataFrame())
    )
    if stacked.empty:
        remove_stale_outputs(output)
        primary_events = (
            int(events["primary_event"].fillna(False).astype(bool).sum())
            if not events.empty and "primary_event" in events.columns
            else 0
        )
        reason = (
            "No direct CFS target observations were available."
            if panel.empty
            else "No clean switch event had sufficient stable-control support."
        )
        status = pd.DataFrame(
            [
                {
                    "gate": "auditor_switch_event_study",
                    "status": "NOT_EVALUATED",
                    "candidate_switch_events": len(events),
                    "primary_clean_events": primary_events,
                    "stacked_events": 0,
                    "analysis_rows": 0,
                    "interpretation": reason,
                }
            ]
        )
        tables = {
            "cfs_auditor_switch_direct_panel": panel,
            "cfs_auditor_switch_line_item_status": line_status,
            "cfs_auditor_switch_event_diagnostics": events,
            "cfs_auditor_switch_stack_support": support,
            "cfs_auditor_switch_stacked_sample": stacked,
            "cfs_auditor_switch_event_study_status": status,
        }
    else:
        tables = run_switch_event_study(
            direct_cases,
            firm_year,
            auditor_settings,
            cfs_settings=cfs_settings,
            line_item_panel=line_items,
        )
        status = tables["cfs_auditor_switch_event_study_status"]

    tables["cfs_completion_gate_status"] = update_completion_gate(
        output, status
    )
    write_tables(tables, output)
    print(
        "Auditor switch event study complete:",
        int(status.loc[0, "stacked_events"]),
        "stacked events",
    )


if __name__ == "__main__":
    main()
