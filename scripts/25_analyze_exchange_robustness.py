from __future__ import annotations

import argparse

import pandas as pd

from _cfs_robustness_cli import (
    analysis_window_status,
    apply_common_overrides,
    load_context,
    not_evaluated_status,
    parse_csv_values,
    persist_analysis,
    remove_outputs,
)
from audit_da.cfs_robustness_runners import run_exchange_robustness


EXCHANGE_OUTPUTS = (
    "cfs_exchange_robustness_sample",
    "cfs_exchange_sample_coverage",
    "cfs_exchange_robustness_metrics",
    "cfs_exchange_pairwise_differences",
    "cfs_exchange_cluster_bootstrap",
    "cfs_exchange_leave_one_out",
    "cfs_exchange_interactions",
    "cfs_exchange_robustness_status",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exchange-level CFS robustness analysis"
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
    parser.add_argument(
        "--exchanges",
        default=None,
        help="Comma-separated exchange groups; defaults to HOSE,HNX,UPCOM.",
    )
    parser.add_argument("--reference-exchange", default=None)
    args = parser.parse_args()

    context = load_context(args.config, case_table_override=args.case_table)
    settings = apply_common_overrides(
        context.settings,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        outcomes=parse_csv_values(args.outcomes),
    )
    if args.bootstrap_seed is not None:
        settings["exchange_bootstrap_seed"] = int(args.bootstrap_seed)
    exchanges = parse_csv_values(args.exchanges)
    if exchanges is not None:
        settings["exchange_groups"] = exchanges
    if args.reference_exchange is not None:
        settings["exchange_reference"] = args.reference_exchange

    if not settings.get("enabled", True) or not settings.get(
        "exchange_enabled", True
    ):
        print("Exchange robustness disabled by configuration")
        return

    if context.cases.empty:
        remove_outputs(context.output, EXCHANGE_OUTPUTS)
        status = not_evaluated_status(
            "within_exchange_robustness",
            "No common-primary observations inside the configured test window.",
        )
        sample = pd.DataFrame()
        tables = {
            "cfs_exchange_robustness_sample": sample,
            "cfs_exchange_robustness_status": status,
        }
    else:
        tables = run_exchange_robustness(context.cases, settings)
        sample = tables["cfs_exchange_robustness_sample"]
        status = tables["cfs_exchange_robustness_status"]

    window_status = analysis_window_status(
        "EXCHANGE",
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
        sample_key="cfs_exchange_robustness_sample",
    )
    print(
        f"Wrote exchange robustness artifacts to {context.output} "
        f"({len(sample):,} analysis rows)"
    )


if __name__ == "__main__":
    main()
