from __future__ import annotations

import argparse

import pandas as pd

from _auditor_switch_common import (
    load_context,
    read_table,
    update_completion_gate,
)
from audit_da.auditor_switch_dynamic_did import run_switch_dynamic_did
from audit_da.diag_common import write_tables


DID_OUTPUTS = (
    "cfs_auditor_switch_overlap_weights",
    "cfs_auditor_switch_overlap_balance",
    "cfs_auditor_switch_dynamic_did_event_contrasts",
    "cfs_auditor_switch_dynamic_did",
    "cfs_auditor_switch_dynamic_did_pretrend",
    "cfs_auditor_switch_dynamic_did_bootstrap",
    "cfs_auditor_switch_dynamic_did_status",
)


def remove_stale_outputs(output) -> None:
    for name in DID_OUTPUTS:
        for suffix in (".csv", ".csv.gz"):
            path = output / f"{name}{suffix}"
            if path.exists():
                path.unlink()


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

    stacked = read_table(
        output, "cfs_auditor_switch_stacked_sample", required=False
    )
    if stacked.empty:
        remove_stale_outputs(output)
        status = pd.DataFrame(
            [
                {
                    "gate": "auditor_switch_dynamic_did",
                    "status": "NOT_EVALUATED",
                    "stacked_events": 0,
                    "contrasts": 0,
                    "estimated_cells": 0,
                    "interpretation": (
                        "No supported switch-event stack was available. Run "
                        "the event-study stage and inspect its support table."
                    ),
                }
            ]
        )
        tables = {
            "cfs_auditor_switch_dynamic_did_status": status,
            "cfs_auditor_switch_dynamic_did": pd.DataFrame(),
        }
    else:
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
