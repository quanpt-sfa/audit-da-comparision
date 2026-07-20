from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import AnalysisWindow, window_from_section
from audit_da.diag_common import write_tables


@dataclass(frozen=True)
class RobustnessContext:
    config_path: Path
    output: Path
    settings: dict[str, Any]
    window: AnalysisWindow
    cases: pd.DataFrame
    case_table: str


def read_table(output: Path, name: str, required: bool = True) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required table not found: {name}")
    return pd.DataFrame()


def load_robustness_settings(
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(config.get("cfs_regime_robustness", {}))
    if settings:
        return settings
    auxiliary = config_path.with_name("cfs_regime_robustness.yaml")
    if not auxiliary.exists():
        raise FileNotFoundError(f"Robustness config not found: {auxiliary}")
    loaded = yaml.safe_load(auxiliary.read_text(encoding="utf-8")) or {}
    return dict(loaded.get("cfs_regime_robustness", loaded))


def load_context(
    config_argument: str,
    *,
    case_table_override: str | None = None,
) -> RobustnessContext:
    config_path, config = load_config(config_argument)
    output = resolve(config_path, config["paths"]["output_dir"])
    cfs_settings = dict(config["cfs_shifting_validation"])
    settings = load_robustness_settings(config_path, config)
    window = window_from_section(cfs_settings)
    case_table = (
        case_table_override
        or settings.get("case_table")
        or "cfs_shifting_proxy_common_primary_core_cases"
    )
    cases = read_table(output, case_table)
    if "fiscal_year" not in cases:
        raise ValueError(f"{case_table} has no fiscal_year column")
    year = pd.to_numeric(cases["fiscal_year"], errors="coerce")
    cases = cases.loc[window.test_mask(year)].copy()
    cases["fiscal_year"] = year.loc[cases.index].astype(int)
    if not cases.empty:
        minimum = int(cases["fiscal_year"].min())
        maximum = int(cases["fiscal_year"].max())
        if minimum < window.test_start_year or maximum > window.test_end_year:
            raise AssertionError(
                f"Robustness input escaped test window: {minimum}-{maximum}"
            )
        if minimum < 2015:
            raise AssertionError("Pre-2015 observations are prohibited")
    return RobustnessContext(
        config_path=config_path,
        output=output,
        settings=settings,
        window=window,
        cases=cases,
        case_table=str(case_table),
    )


def parse_csv_values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def parse_csv_years(value: str | None) -> list[int] | None:
    parsed = parse_csv_values(value)
    return [int(item) for item in parsed] if parsed else None


def apply_common_overrides(
    settings: dict[str, Any],
    *,
    bootstrap_repetitions: int | None = None,
    bootstrap_seed: int | None = None,
    outcomes: list[str] | None = None,
) -> dict[str, Any]:
    updated = dict(settings)
    if bootstrap_repetitions is not None:
        if bootstrap_repetitions < 0:
            raise ValueError("bootstrap_repetitions must be nonnegative")
        updated["bootstrap_repetitions"] = int(bootstrap_repetitions)
    if bootstrap_seed is not None:
        updated["bootstrap_seed"] = int(bootstrap_seed)
    if outcomes is not None:
        updated["outcomes"] = list(outcomes)
    return updated


def remove_outputs(output: Path, names: tuple[str, ...]) -> None:
    for name in names:
        for suffix in (".csv", ".csv.gz"):
            path = output / f"{name}{suffix}"
            if path.exists():
                path.unlink()


def not_evaluated_status(gate: str, interpretation: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "status": "NOT_EVALUATED",
                "evidence_rows": 0,
                "interpretation": interpretation,
            }
        ]
    )


def analysis_window_status(
    analysis_kind: str,
    window: AnalysisWindow,
    cases: pd.DataFrame,
    sample: pd.DataFrame,
    case_table: str,
) -> pd.DataFrame:
    if sample.empty:
        status = "NOT_EVALUATED"
        minimum = pd.NA
        maximum = pd.NA
    else:
        year = pd.to_numeric(sample["fiscal_year"], errors="coerce").dropna()
        minimum = int(year.min())
        maximum = int(year.max())
        status = (
            "PASS"
            if minimum >= window.test_start_year
            and maximum <= window.test_end_year
            else "FAILED"
        )
    sample_year = pd.to_numeric(
        sample.get("fiscal_year", pd.Series(dtype=float)),
        errors="coerce",
    )
    return pd.DataFrame(
        [
            {
                "analysis_kind": analysis_kind,
                "status": status,
                **window.as_dict(),
                "case_table": case_table,
                "input_case_rows": len(cases),
                "analysis_rows": len(sample),
                "effective_minimum_year": minimum,
                "effective_maximum_year": maximum,
                "pre_2015_rows": int(sample_year.lt(2015).sum()),
                "sample_rule": (
                    "earnings_working_capital on common-primary "
                    "analysis-core issuer-years"
                ),
            }
        ]
    )


def _merge_rows(
    current: pd.DataFrame,
    new_rows: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    if new_rows.empty:
        return current
    if current.empty or any(key not in current for key in keys):
        return new_rows.copy()
    marker = new_rows[keys].astype(str).agg("|".join, axis=1)
    current_marker = current[keys].astype(str).agg("|".join, axis=1)
    current = current.loc[~current_marker.isin(set(marker))].copy()
    return pd.concat([current, new_rows], ignore_index=True, sort=False)


def persist_analysis(
    *,
    output: Path,
    tables: dict[str, pd.DataFrame],
    status: pd.DataFrame,
    window_status: pd.DataFrame,
    sample_key: str,
) -> None:
    common_status = _merge_rows(
        read_table(output, "cfs_regime_robustness_status", required=False),
        status,
        ["gate"],
    )
    common_window = _merge_rows(
        read_table(
            output,
            "cfs_regime_robustness_window_status",
            required=False,
        ),
        window_status,
        ["analysis_kind"],
    )
    completion = _merge_rows(
        read_table(output, "cfs_completion_gate_status", required=False),
        status[["gate", "status", "evidence_rows"]],
        ["gate"],
    )
    payload = dict(tables)
    payload["cfs_regime_robustness_sample"] = tables.get(
        sample_key,
        pd.DataFrame(),
    )
    payload["cfs_regime_robustness_status"] = common_status
    payload["cfs_regime_robustness_window_status"] = common_window
    payload["cfs_completion_gate_status"] = completion
    write_tables(payload, output)
