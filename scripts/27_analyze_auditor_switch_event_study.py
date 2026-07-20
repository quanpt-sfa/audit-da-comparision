from __future__ import annotations

import argparse

from _auditor_switch_common import (
    load_context,
    read_table,
    restrict_years,
    update_completion_gate,
)
from audit_da.auditor_switch_event_study import run_switch_event_study
from audit_da.diag_common import write_tables


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
