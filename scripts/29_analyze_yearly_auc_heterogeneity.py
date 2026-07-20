from __future__ import annotations

import argparse

from _auditor_switch_common import (
    load_context,
    read_table,
    restrict_years,
    update_completion_gate,
)
from audit_da.diag_common import write_tables
from audit_da.yearly_auc_heterogeneity import run_yearly_auc_heterogeneity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test yearly AUC and score-slope heterogeneity with issuer-cluster "
            "dependence"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()

    _, _, output, _, auditor_settings = load_context(args.config)
    settings = dict(auditor_settings.get("yearly_auc_heterogeneity", {}))
    if not settings.get("enabled", True):
        print("Yearly AUC heterogeneity analysis disabled by configuration")
        return

    cases = read_table(
        output,
        settings.get(
            "case_table", "cfs_shifting_proxy_common_primary_core_cases"
        ),
    )
    cases = restrict_years(
        cases, auditor_settings, use_source_window=False
    )
    tables = run_yearly_auc_heterogeneity(cases, auditor_settings)
    status = tables["cfs_yearly_auc_heterogeneity_status"]
    tables["cfs_completion_gate_status"] = update_completion_gate(
        output, status
    )
    write_tables(tables, output)
    print(
        "Yearly AUC heterogeneity analysis complete:",
        int(status.loc[0, "supported_year_outcome_cells"]),
        "supported year/outcome cells",
    )


if __name__ == "__main__":
    main()
