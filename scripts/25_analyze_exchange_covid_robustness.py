from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import window_from_section
from audit_da.cfs_regime_robustness_identified import run_regime_robustness
from audit_da.diag_common import write_tables


ROBUSTNESS_OUTPUTS = (
    "cfs_regime_robustness_sample",
    "cfs_exchange_robustness_metrics",
    "cfs_exchange_pairwise_differences",
    "cfs_exchange_cluster_bootstrap",
    "cfs_exchange_leave_one_out",
    "cfs_exchange_interactions",
    "cfs_covid_regime_metrics",
    "cfs_covid_regime_differences",
    "cfs_covid_window_sensitivity",
    "cfs_covid_cluster_bootstrap",
    "cfs_covid_interactions",
    "cfs_regime_robustness_status",
    "cfs_regime_robustness_window_status",
)


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def load_robustness_settings(config_path: Path, config: dict) -> dict:
    settings = dict(config.get("cfs_regime_robustness", {}))
    if settings:
        return settings
    auxiliary = config_path.with_name("cfs_regime_robustness.yaml")
    if not auxiliary.exists():
        raise FileNotFoundError(f"Robustness config not found: {auxiliary}")
    loaded = yaml.safe_load(auxiliary.read_text(encoding="utf-8")) or {}
    return dict(loaded.get("cfs_regime_robustness", loaded))


def remove_stale_outputs(output: Path) -> None:
    for name in ROBUSTNESS_OUTPUTS:
        for suffix in (".csv", ".csv.gz"):
            path = output / f"{name}{suffix}"
            if path.exists():
                path.unlink()


def update_completion_gates(output: Path, status: pd.DataFrame) -> pd.DataFrame:
    current = read_table(output, "cfs_completion_gate_status", required=False)
    gates = status[["gate", "status", "evidence_rows"]].copy()
    if current.empty:
        return gates
    current = current[~current["gate"].isin(gates["gate"])].copy()
    return pd.concat([current, gates], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run within-exchange and COVID-period CFS robustness tests"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    cfs_settings = dict(config["cfs_shifting_validation"])
    settings = load_robustness_settings(config_path, config)
    if not settings.get("enabled", True):
        print("Exchange/COVID robustness disabled by configuration")
        return

    window = window_from_section(cfs_settings)
    case_table = settings.get(
        "case_table", "cfs_shifting_proxy_common_primary_core_cases"
    )
    cases = read_table(output, case_table)
    year = pd.to_numeric(cases["fiscal_year"], errors="coerce")
    cases = cases.loc[window.test_mask(year)].copy()
    cases["fiscal_year"] = year.loc[cases.index].astype(int)

    if cases.empty:
        remove_stale_outputs(output)
        status = pd.DataFrame(
            [
                {
                    "gate": "within_exchange_robustness",
                    "status": "NOT_EVALUATED",
                    "evidence_rows": 0,
                    "interpretation": "No common-primary test observations.",
                },
                {
                    "gate": "covid_period_robustness",
                    "status": "NOT_EVALUATED",
                    "evidence_rows": 0,
                    "interpretation": "No common-primary test observations.",
                },
            ]
        )
        write_tables(
            {
                "cfs_regime_robustness_status": status,
                "cfs_completion_gate_status": update_completion_gates(output, status),
            },
            output,
        )
        return

    tables = run_regime_robustness(cases, settings)
    sample = tables["cfs_regime_robustness_sample"]
    tables["cfs_regime_robustness_window_status"] = pd.DataFrame(
        [
            {
                "status": "PASS",
                **window.as_dict(),
                "input_case_rows": len(cases),
                "analysis_rows": len(sample),
                "effective_minimum_year": int(sample["fiscal_year"].min()),
                "effective_maximum_year": int(sample["fiscal_year"].max()),
                "sample_rule": "earnings_working_capital on common-primary analysis-core issuer-years",
                "exchange_interpretation": "transportability robustness, not causal exchange effect",
                "covid_interpretation": "temporal-regime robustness, not causal COVID treatment effect",
            }
        ]
    )
    tables["cfs_completion_gate_status"] = update_completion_gates(
        output, tables["cfs_regime_robustness_status"]
    )
    write_tables(tables, output)


if __name__ == "__main__":
    main()
