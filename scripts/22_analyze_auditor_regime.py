from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from _next_diag_common import load_config, resolve
from audit_da.auditor_regime import (
    load_auditor_firm_year,
    run_auditor_regime_analysis,
)
from audit_da.diag_common import write_tables


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def source_paths(config_path: Path, config: dict, settings: dict) -> list[Path]:
    configured = settings.get(
        "source_preference", ["auditor_input", "panel_input", "raw_input"]
    )
    paths: list[Path] = []
    for key in configured:
        value = config.get("paths", {}).get(key)
        if not value:
            continue
        path = resolve(config_path, value)
        if path not in paths:
            paths.append(path)
    return paths


def update_completion_gate(output: Path, auditor_status: pd.DataFrame) -> pd.DataFrame:
    current = read_table(output, "cfs_completion_gate_status", required=False)
    row = {
        "gate": "auditor_regime_heterogeneity",
        "status": (
            auditor_status.loc[0, "status"]
            if not auditor_status.empty
            else "NOT_EVALUATED"
        ),
        "evidence_rows": (
            int(auditor_status.loc[0, "known_auditor_rows"])
            if not auditor_status.empty
            else 0
        ),
    }
    if current.empty:
        return pd.DataFrame([row])
    current = current[~current["gate"].eq(row["gate"])].copy()
    return pd.concat([current, pd.DataFrame([row])], ignore_index=True)


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
    firm_year, name_map, source_status = load_auditor_firm_year(
        source_paths(config_path, config, settings),
        settings,
        audited_label=cfs_settings.get("audited_label", "audited"),
        required_scope=cfs_settings.get("required_scope", "consolidated"),
    )
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
