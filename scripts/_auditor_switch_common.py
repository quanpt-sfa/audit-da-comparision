from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import window_from_section


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            try:
                return pd.read_csv(path, low_memory=False)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def load_auditor_settings(
    config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    cfs = dict(config.get("cfs_shifting_validation", {}))
    settings = dict(cfs.get("auditor_regime", {}))
    if settings:
        return settings
    auxiliary = config_path.with_name("auditor_regime.yaml")
    if not auxiliary.exists():
        raise FileNotFoundError(f"Auditor-regime config not found: {auxiliary}")
    loaded = yaml.safe_load(auxiliary.read_text(encoding="utf-8")) or {}
    return dict(loaded.get("auditor_regime", loaded))


def load_context(config_value: str | Path):
    config_path, config = load_config(config_value)
    output = resolve(config_path, config["paths"]["output_dir"])
    cfs_settings = dict(config["cfs_shifting_validation"])
    auditor_settings = load_auditor_settings(config_path, config)
    auditor_settings["analysis_window"] = dict(
        cfs_settings.get(
            "analysis_window", auditor_settings.get("analysis_window", {})
        )
    )
    return config_path, config, output, cfs_settings, auditor_settings


def restrict_years(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    use_source_window: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "fiscal_year" not in frame.columns:
        raise ValueError("Analysis table has no fiscal_year column")
    window = window_from_section(settings)
    year = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    mask = window.source_mask(year) if use_source_window else window.test_mask(year)
    restricted = frame.loc[mask].copy()
    restricted["fiscal_year"] = year.loc[restricted.index].astype(int)
    return restricted


def update_completion_gate(
    output: Path,
    status: pd.DataFrame,
) -> pd.DataFrame:
    current = read_table(output, "cfs_completion_gate_status", required=False)
    if status.empty or "gate" not in status.columns:
        return current
    gates = status[["gate", "status"]].copy()
    evidence_column = next(
        (
            column
            for column in [
                "analysis_rows",
                "stacked_events",
                "supported_year_outcome_cells",
            ]
            if column in status.columns
        ),
        None,
    )
    gates["evidence_rows"] = (
        pd.to_numeric(status[evidence_column], errors="coerce")
        .fillna(0)
        .astype(int)
        if evidence_column
        else 0
    )
    if current.empty:
        return gates
    current = current[~current["gate"].isin(gates["gate"])].copy()
    return pd.concat([current, gates], ignore_index=True)
