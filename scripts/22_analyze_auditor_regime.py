from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from _next_diag_common import load_config, resolve
from audit_da.auditor_regime import run_auditor_regime_analysis
from audit_da.auditor_source import discover_auditor_sources
from audit_da.auditor_source_safe import load_auditor_firm_year_safe
from audit_da.bctc_auditor_source import (
    is_bctc_audit_annual_long,
    load_bctc_audit_annual_long,
)
from audit_da.diag_common import write_tables


AUDITOR_OUTPUTS = (
    "cfs_auditor_firm_year",
    "cfs_auditor_name_mapping",
    "cfs_auditor_analysis_sample",
    "cfs_auditor_regime_coverage",
    "cfs_auditor_regime_metrics",
    "cfs_auditor_regime_metric_differences",
    "cfs_auditor_regime_bootstrap",
    "cfs_auditor_regime_interaction",
    "cfs_auditor_regime_balance",
    "cfs_auditor_switch_events",
    "cfs_auditor_switch_summary",
)


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def configured_source_paths(
    config_path: Path, config: dict, settings: dict
) -> list[Path]:
    configured = settings.get(
        "source_preference",
        ["auditor_input", "audit_input", "panel_input", "raw_input"],
    )
    paths: list[Path] = []
    for key in configured:
        value = config.get("paths", {}).get(key)
        if value:
            paths.append(resolve(config_path, value))
    repo_root = config_path.parent.parent
    for value in settings.get("explicit_source_paths", []):
        path = Path(value)
        paths.append(path if path.is_absolute() else repo_root / path)
    return discover_auditor_sources(
        repo_root,
        paths,
        settings.get(
            "source_globs",
            [
                "data/raw/bctc_audit_annual_long.csv",
                "data/raw/bctc_audit_annual_long.csv.gz",
            ],
        ),
    )


def load_project_auditor_source(
    paths: list[Path],
    settings: dict,
    audited_label: str,
    required_scope: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prefer the verified BCTC audit metadata contract over schema guessing."""
    exact_sources = [path for path in paths if is_bctc_audit_annual_long(path)]
    if exact_sources:
        selected = exact_sources[0]
        return load_bctc_audit_annual_long(selected, settings)
    return load_auditor_firm_year_safe(
        paths,
        settings,
        audited_label=audited_label,
        required_scope=required_scope,
    )


def update_completion_gate(output: Path, auditor_status: pd.DataFrame) -> pd.DataFrame:
    current = read_table(output, "cfs_completion_gate_status", required=False)
    row = {
        "gate": "auditor_regime_heterogeneity",
        "status": (
            auditor_status.loc[0, "status"]
            if not auditor_status.empty and "status" in auditor_status
            else "NOT_EVALUATED"
        ),
        "evidence_rows": (
            int(auditor_status.loc[0, "known_auditor_rows"])
            if not auditor_status.empty and "known_auditor_rows" in auditor_status
            else 0
        ),
    }
    if current.empty:
        return pd.DataFrame([row])
    current = current[~current["gate"].eq(row["gate"])].copy()
    return pd.concat([current, pd.DataFrame([row])], ignore_index=True)


def remove_stale_outputs(output: Path) -> None:
    for name in AUDITOR_OUTPUTS:
        for suffix in (".csv", ".csv.gz"):
            path = output / f"{name}{suffix}"
            if path.exists():
                path.unlink()


def unavailable_status(cases: pd.DataFrame, source_status: pd.DataFrame) -> pd.DataFrame:
    detail = "No usable auditor source was found."
    if not source_status.empty:
        fields = [
            column
            for column in (
                "path",
                "status",
                "reason",
                "initial_error",
                "retry_error",
            )
            if column in source_status
        ]
        if fields:
            detail = "; ".join(
                " | ".join(str(row.get(column, "")) for column in fields)
                for row in source_status[fields].to_dict("records")
            )
    return pd.DataFrame(
        [
            {
                "status": "NOT_EVALUATED",
                "reason": detail,
                "analysis_rows": len(cases),
                "known_auditor_rows": 0,
                "known_auditor_share": 0.0,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate abnormal-CFO criterion validity by Big4/non-Big4 audit regime"
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    cfs_settings = dict(config["cfs_shifting_validation"])
    settings = dict(cfs_settings.get("auditor_regime", {}))
    if not settings:
        auxiliary_config = config_path.with_name("auditor_regime.yaml")
        if auxiliary_config.exists():
            loaded = yaml.safe_load(auxiliary_config.read_text(encoding="utf-8")) or {}
            settings = dict(loaded.get("auditor_regime", loaded))
    if not settings.get("enabled", True):
        print("Auditor-regime analysis disabled by configuration")
        return

    cases = read_table(
        output,
        settings.get(
            "case_table", "cfs_shifting_proxy_common_primary_core_cases"
        ),
    )
    paths = configured_source_paths(config_path, config, settings)
    firm_year, name_map, source_status = load_project_auditor_source(
        paths,
        settings,
        audited_label=cfs_settings.get("audited_label", "audited"),
        required_scope=cfs_settings.get("required_scope", "consolidated"),
    )

    if firm_year.empty:
        remove_stale_outputs(output)
        regime_status = unavailable_status(cases, source_status)
        tables = {
            "cfs_auditor_source_status": source_status,
            "cfs_auditor_regime_status": regime_status,
            "cfs_completion_gate_status": update_completion_gate(
                output, regime_status
            ),
        }
        write_tables(tables, output)
        message = regime_status.loc[0, "reason"]
        if settings.get("fail_pipeline_if_unavailable", False):
            raise ValueError(message)
        print(f"Auditor-regime analysis not evaluated: {message}")
        return

    tables = run_auditor_regime_analysis(cases, firm_year, settings)
    tables["cfs_auditor_firm_year"] = firm_year
    tables["cfs_auditor_name_mapping"] = name_map
    tables["cfs_auditor_source_status"] = source_status
    tables["cfs_completion_gate_status"] = update_completion_gate(
        output, tables["cfs_auditor_regime_status"]
    )
    write_tables(tables, output)


if __name__ == "__main__":
    main()
