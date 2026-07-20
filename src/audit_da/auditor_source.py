from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .auditor_regime import classify_auditor_name, load_auditor_firm_year, normalize_ticker
from .diag_common import KEYS

DEFAULT_TICKER_COLUMNS = (
    "issuer_ticker", "ticker", "stock_code", "symbol", "ma_ck", "ma_chung_khoan"
)
DEFAULT_YEAR_COLUMNS = ("fiscal_year", "report_year", "year", "nam")
DEFAULT_WIDE_AUDITOR_COLUMNS = (
    "audit_firm_name", "auditor_name", "audit_firm", "auditing_firm",
    "auditing_company", "audit_company", "auditor_firm_name",
    "auditing_company_name", "company_audit_name", "don_vi_kiem_toan",
    "ten_cong_ty_kiem_toan", "ten_don_vi_kiem_toan", "cong_ty_kiem_toan",
)
DEFAULT_INDICATOR_COLUMNS = (
    "audit_indicator", "indicator", "variable", "metric", "item",
    "field", "chi_tieu", "ten_chi_tieu",
)
DEFAULT_VALUE_COLUMNS = (
    "audit_value", "value", "value_raw", "raw_value", "indicator_value",
    "text_value", "value_text", "gia_tri", "gia_tri_raw", "gia_tri_text",
)
DEFAULT_AUDITOR_INDICATOR_VALUES = (
    "audit_firm", "audit_firm_name", "auditor", "auditor_name",
    "auditing_company", "don_vi_kiem_toan", "ten_don_vi_kiem_toan",
    "cong_ty_kiem_toan", "ten_cong_ty_kiem_toan",
)


def _ascii_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _choose(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = list(columns)
    normalized = {_ascii_text(column): column for column in available}
    for candidate in candidates:
        hit = normalized.get(_ascii_text(candidate))
        if hit is not None:
            return hit
    return None


def _header(path: Path) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".parquet", ".pq")):
        return list(pd.read_parquet(path).columns)
    if suffixes.endswith((".xlsx", ".xls")):
        return list(pd.read_excel(path, nrows=0).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def _read(path: Path, columns: list[str], chunksize: int) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=columns)
    if suffixes.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, usecols=columns)
    chunks = pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False)
    frames = list(chunks)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def discover_auditor_sources(
    repo_root: Path,
    configured_paths: Iterable[Path],
    patterns: Iterable[str],
) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            ordered.append(path.resolve())

    for path in configured_paths:
        add(path)
    for pattern in patterns:
        for path in sorted(repo_root.glob(pattern)):
            if path.is_file():
                add(path)
    return ordered


def _aggregate_firm_year(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = pd.DataFrame(
        [classify_auditor_name(value) for value in raw["auditor_name_raw"]],
        index=raw.index,
    )
    work = pd.concat([raw[KEYS], classified], axis=1)
    name_map = (
        work[
            [
                "auditor_name_raw", "auditor_name_normalized", "auditor_brand",
                "auditor_group", "big4_flag", "auditor_name_status",
            ]
        ]
        .drop_duplicates()
        .sort_values(["auditor_group", "auditor_brand", "auditor_name_raw"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for (ticker, year), group in work.groupby(KEYS, observed=True, sort=False):
        valid = group[group["auditor_group"].isin(["BIG4", "NON_BIG4"])]
        brands = sorted(set(valid["auditor_brand"].dropna().astype(str)) - {""})
        groups = sorted(set(valid["auditor_group"].dropna().astype(str)))
        if not brands:
            row = classify_auditor_name("")
            row["auditor_firm_year_status"] = "MISSING_AUDITOR"
        elif len(brands) == 1 and len(groups) == 1:
            first = valid.iloc[0]
            row = {
                "auditor_name_raw": first["auditor_name_raw"],
                "auditor_name_normalized": first["auditor_name_normalized"],
                "auditor_brand": brands[0],
                "auditor_group": groups[0],
                "big4_flag": float(first["big4_flag"]),
                "auditor_name_status": first["auditor_name_status"],
                "auditor_firm_year_status": (
                    "EXACT_ONE_NAME" if len(valid) == 1 else "CONSISTENT_DUPLICATES"
                ),
            }
        else:
            row = {
                "auditor_name_raw": " | ".join(sorted(set(valid["auditor_name_raw"].astype(str)))),
                "auditor_name_normalized": " | ".join(brands),
                "auditor_brand": " | ".join(brands),
                "auditor_group": "AMBIGUOUS",
                "big4_flag": np.nan,
                "auditor_name_status": "MULTIPLE_AUDITORS",
                "auditor_firm_year_status": "AMBIGUOUS_MULTIPLE_AUDITORS",
            }
        row.update({"issuer_ticker": ticker, "fiscal_year": int(year)})
        rows.append(row)

    firm_year = pd.DataFrame(rows)
    if not firm_year.empty:
        firm_year = firm_year[
            KEYS
            + [
                "auditor_name_raw", "auditor_name_normalized", "auditor_brand",
                "auditor_group", "big4_flag", "auditor_name_status",
                "auditor_firm_year_status",
            ]
        ].sort_values(KEYS).reset_index(drop=True)
    return firm_year, name_map


def _long_schema(columns: list[str], settings: dict[str, Any]) -> dict[str, str | None]:
    return {
        "ticker": _choose(columns, settings.get("ticker_column_candidates", DEFAULT_TICKER_COLUMNS)),
        "year": _choose(columns, settings.get("year_column_candidates", DEFAULT_YEAR_COLUMNS)),
        "indicator": _choose(columns, settings.get("indicator_column_candidates", DEFAULT_INDICATOR_COLUMNS)),
        "value": _choose(columns, settings.get("value_column_candidates", DEFAULT_VALUE_COLUMNS)),
    }


def _load_long_source(
    path: Path,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    columns = _header(path)
    schema = _long_schema(columns, settings)
    if not all(schema.values()):
        return None
    usecols = [str(schema[key]) for key in ("ticker", "year", "indicator", "value")]
    raw = _read(path, usecols, int(settings.get("chunksize", 250_000)))
    indicator_values = {
        _ascii_text(value)
        for value in settings.get("auditor_indicator_values", DEFAULT_AUDITOR_INDICATOR_VALUES)
    }
    indicator = raw[str(schema["indicator"])].map(_ascii_text)
    keep = indicator.isin(indicator_values)
    if not keep.any():
        keep = indicator.str.contains(
            r"(^|_)(audit_firm|auditor|auditing_company|don_vi_kiem_toan|cong_ty_kiem_toan)($|_)",
            regex=True,
            na=False,
        )
    raw = raw.loc[keep, usecols].rename(
        columns={
            str(schema["ticker"]): "issuer_ticker",
            str(schema["year"]): "fiscal_year",
            str(schema["value"]): "auditor_name_raw",
        }
    )
    raw["issuer_ticker"] = raw["issuer_ticker"].map(normalize_ticker)
    raw["fiscal_year"] = pd.to_numeric(raw["fiscal_year"], errors="coerce")
    raw = raw[
        raw["issuer_ticker"].ne("")
        & raw["fiscal_year"].notna()
        & raw["auditor_name_raw"].notna()
    ].copy()
    if raw.empty:
        return None
    raw["fiscal_year"] = raw["fiscal_year"].astype(int)
    firm_year, name_map = _aggregate_firm_year(raw)
    status = pd.DataFrame(
        [
            {
                "status": "EVALUATED",
                "source_path": str(path),
                "source_schema": "LONG_AUDIT_INDICATOR",
                "ticker_column": schema["ticker"],
                "year_column": schema["year"],
                "indicator_column": schema["indicator"],
                "auditor_name_column": schema["value"],
                "source_rows_after_filters": len(raw),
                "firm_years": len(firm_year),
                "big4_firm_years": int(firm_year["auditor_group"].eq("BIG4").sum()),
                "non_big4_firm_years": int(firm_year["auditor_group"].eq("NON_BIG4").sum()),
                "ambiguous_firm_years": int(firm_year["auditor_group"].eq("AMBIGUOUS").sum()),
                "unknown_firm_years": int(firm_year["auditor_group"].eq("UNKNOWN").sum()),
            }
        ]
    )
    return firm_year, name_map, status


def load_auditor_firm_year_resilient(
    source_paths: list[Path],
    settings: dict[str, Any],
    audited_label: str = "audited",
    required_scope: str | None = "consolidated",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide_settings = dict(settings)
    wide_settings["required"] = False
    wide_candidates = list(settings.get("auditor_name_column_candidates", []))
    for candidate in DEFAULT_WIDE_AUDITOR_COLUMNS:
        if candidate not in wide_candidates:
            wide_candidates.append(candidate)
    wide_settings["auditor_name_column_candidates"] = wide_candidates

    firm_year, name_map, wide_status = load_auditor_firm_year(
        source_paths,
        wide_settings,
        audited_label=audited_label,
        required_scope=required_scope,
    )
    if not firm_year.empty:
        if "source_schema" not in wide_status:
            wide_status["source_schema"] = "WIDE_AUDITOR_COLUMN"
        return firm_year, name_map, wide_status

    inspected: list[dict[str, Any]] = []
    for path in source_paths:
        if not path.exists():
            inspected.append({"path": str(path), "status": "MISSING_FILE"})
            continue
        try:
            loaded = _load_long_source(path, settings)
        except Exception as exc:
            inspected.append(
                {"path": str(path), "status": "READ_ERROR", "reason": str(exc)}
            )
            continue
        if loaded is not None:
            return loaded
        try:
            columns = _header(path)
        except Exception as exc:
            inspected.append(
                {"path": str(path), "status": "HEADER_ERROR", "reason": str(exc)}
            )
            continue
        inspected.append(
            {
                "path": str(path),
                "status": "NO_USABLE_WIDE_OR_LONG_AUDITOR_SCHEMA",
                "available_columns": " | ".join(map(str, columns)),
            }
        )

    status = pd.DataFrame(inspected)
    if status.empty and not wide_status.empty:
        status = wide_status.copy()
    status.insert(0, "overall_status", "NOT_EVALUATED")
    return pd.DataFrame(), pd.DataFrame(), status
