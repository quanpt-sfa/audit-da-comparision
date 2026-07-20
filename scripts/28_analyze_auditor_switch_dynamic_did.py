from __future__ import annotations

import argparse

from _auditor_switch_common import (
    load_context,
    read_table,
    update_completion_gate,
)
from audit_da.auditor_switch_dynamic_did import run_switch_dynamic_did
from audit_da.diag_common import write_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run switcher-versus-stayer dynamic DiD for reversible "
            "auditor-tier transitions"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()

    _, _, output, _, auditor_settings = load_context(args.config)
    did = dict(auditor_settings.get("dynamic_did", {}))
    if not did.get("enabled", True):
        print("Auditor switch dynamic DiD disabled by configuration")
        return

    stacked = read_table(output, "cfs_auditor_switch_stacked_sample")
    if stacked.empty:
        raise ValueError(
            "No stacked switch sample is available. Run "
            "scripts/27_analyze_auditor_switch_event_study.py first."
        )
    tables = run_switch_dynamic_did(stacked, auditor_settings)
    status = tables["cfs_auditor_switch_dynamic_did_status"]
    tables["cfs_completion_gate_status"] = update_completion_gate(
        output, status
    )
    write_tables(tables, output)
    print(
        "Auditor switch dynamic DiD complete:",
        int(status.loc[0, "estimated_cells"]),
        "estimated direction/outcome/horizon cells",
    )


if __name__ == "__main__":
    main()
