from __future__ import annotations

import argparse

import pandas as pd

from _cfs_robustness_cli import (
    analysis_window_status,
    apply_common_overrides,
    load_context,
    not_evaluated_status,
    parse_csv_values,
    parse_csv_years,
    persist_analysis,
    remove_outputs,
)
from audit_da.cfs_robustness_runners import run_covid_robustness


COVID_OUTPUTS = (
    "cfs_covid_robustness_sample",
    "cfs_covid_sample_coverage",
    "cfs_covid_regime_metrics",
    "cfs_covid_regime_differences",
    "cfs_covid_window_sensitivity",
    "cfs_covid_cluster_bootstrap",
    "cfs_covid_interactions",
    "cfs_covid_robustness_status",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COVID-period CFS robustness analysis"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    parser.add_argument("--case-table", default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Comma-separated outcomes; defaults to the configured list.",
    )
    parser.add_argument("--pre-years", default=None)
    parser.add_argument("--shock-years", default=None)
    parser.add_argument("--recovery-years", default=None)
    args = parser.parse_args()

    context = load_context(args.config, case_table_override=args.case_table)
    settings = apply_common_overrides(
        context.settings,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        outcomes=parse_csv_values(args.outcomes),
    )
    if args.bootstrap_seed is not None:
        settings["covid_bootstrap_seed"] = int(args.bootstrap_seed)
    covid = dict(settings.get("covid", {}))
    pre_years = parse_csv_years(args.pre_years)
    shock_years = parse_csv_years(args.shock_years)
    recovery_years = parse_csv_years(args.recovery_years)
    if pre_years is not None:
        covid["pre_years"] = pre_years
    if shock_years is not None:
        covid["primary_shock_years"] = shock_years
    if recovery_years is not None:
        covid["recovery_years"] = recovery_years
    settings["covid"] = covid

    if not settings.get("enabled", True) or not covid.get("enabled", True):
        print("COVID-period robustness disabled by configuration")
        return

    if context.cases.empty:
        remove_outputs(context.output, COVID_OUTPUTS)
        status = not_evaluated_status(
            "covid_period_robustness",
            "No common-primary observations inside the configured test window.",
        )
        sample = pd.DataFrame()
        tables = {
            "cfs_covid_robustness_sample": sample,
            "cfs_covid_robustness_status": status,
        }
    else:
        tables = run_covid_robustness(context.cases, settings)
        sample = tables["cfs_covid_robustness_sample"]
        status = tables["cfs_covid_robustness_status"]

    window_status = analysis_window_status(
        "COVID",
        context.window,
        context.cases,
        sample,
        context.case_table,
    )
    persist_analysis(
        output=context.output,
        tables=tables,
        status=status,
        window_status=window_status,
        sample_key="cfs_covid_robustness_sample",
    )
    print(
        f"Wrote COVID robustness artifacts to {context.output} "
        f"({len(sample):,} analysis rows)"
    )


if __name__ == "__main__":
    main()
