from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .auditor_regime import classify_auditor_name, normalize_ticker
from .diag_common import KEYS

REQUIRED_COLUMNS = (
    "issuer_ticker",
    "year",
    "period_type",
    "statement_scope",
    "audit_status",
    "audit_indicator",
    "audit_value_raw",
    "audit_firm_raw",
    "source_file",
)

OPTIONAL_ENTITY_COLUMNS = (
    "source_ticker_raw",
    "firm_name_raw",
    "exchange_raw",
)

# The raw BCTC source historically reused VSM and VTS for two distinct legal
# entities. Canonicalisation must therefore use the entity name, not a global
# ticker replacement.
ENTITY_TICKER_OVERRIDES = {
    ("VSM", "chung_khoan_vsm"): "VSMS",
    ("VTS", "chung_khoan_viet_thanh"): "VTSC",
}


def _token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def canonicalize_entity_ticker(ticker: Any, firm_name: Any = None) -> str:
    """Resolve known same-symbol entity collisions using the legal entity name."""
    normalized_ticker = normalize_ticker(ticker)
    normalized_name = _token(firm_name)
    return ENTITY_TICKER_OVERRIDES.get(
        (normalized_ticker, normalized_name), normalized_ticker
    )


def is_bctc_audit_annual_long(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return set(REQUIRED_COLUMNS).issubset(columns)


def _aggregate_firm_year(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = pd.DataFrame(
        [classify_auditor_name(value) for value in raw["auditor_name_raw"]],
        index=raw.index,
    )
    work = pd.concat([raw[KEYS], classified], axis=1)

    name_map = (
        work[
            [
                "auditor_name_raw",
                "auditor_name_normalized",
                "auditor_brand",
                "auditor_group",
                "big4_flag",
                "auditor_name_status",
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
        raw_names = sorted(set(valid["auditor_name_raw"].dropna().astype(str)) - {""})

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
                "auditor_name_raw": " | ".join(raw_names),
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
                "auditor_name_raw",
                "auditor_name_normalized",
                "auditor_brand",
                "auditor_group",
                "big4_flag",
                "auditor_name_status",
                "auditor_firm_year_status",
            ]
        ].sort_values(KEYS).reset_index(drop=True)
    return firm_year, name_map


def load_bctc_audit_annual_long(
    path: Path,
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the verified project auditor metadata contract.

    Verified source schema (2026-07-20):
    - issuer-year keys: issuer_ticker, year
    - annual rows: period_type == annual
    - consolidated rows: statement_scope == Hợp nhất
    - audited rows: audit_status == audited
    - auditor rows: audit_indicator == audit_firm
    - auditor name: audit_firm_raw, with audit_value_raw as an equality check/fallback
    - known reused-symbol entities are resolved by issuer_ticker + firm_name_raw
    """
    settings = settings or {}
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(
            f"bctc_audit_annual_long schema mismatch; missing columns: {missing}"
        )

    usecols = list(
        dict.fromkeys(
            list(REQUIRED_COLUMNS)
            + [column for column in OPTIONAL_ENTITY_COLUMNS if column in columns]
        )
    )
    chunksize = int(settings.get("chunksize", 250_000))
    chunks = pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
    frames: list[pd.DataFrame] = []
    source_rows = 0
    for chunk in chunks:
        source_rows += len(chunk)
        mask = (
            chunk["period_type"].map(_token).eq("annual")
            & chunk["statement_scope"].map(_token).isin({"hop_nhat", "consolidated"})
            & chunk["audit_status"].map(_token).eq("audited")
            & chunk["audit_indicator"].map(_token).eq("audit_firm")
        )
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            frames.append(selected)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    if raw.empty:
        raise ValueError(
            "Verified bctc_audit_annual_long filters returned zero audit_firm rows"
        )

    raw["auditor_name_raw"] = raw["audit_firm_raw"].where(
        raw["audit_firm_raw"].notna(), raw["audit_value_raw"]
    )
    firm_names = (
        raw["firm_name_raw"]
        if "firm_name_raw" in raw
        else pd.Series(pd.NA, index=raw.index)
    )
    raw["issuer_ticker"] = [
        canonicalize_entity_ticker(ticker, firm_name)
        for ticker, firm_name in zip(raw["issuer_ticker"], firm_names, strict=True)
    ]
    raw["fiscal_year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw[
        raw["issuer_ticker"].ne("")
        & raw["fiscal_year"].notna()
        & raw["auditor_name_raw"].notna()
    ].copy()
    raw["fiscal_year"] = raw["fiscal_year"].astype(int)

    firm_value_equal = raw["audit_firm_raw"].fillna("").eq(
        raw["audit_value_raw"].fillna("")
    )
    firm_year, name_map = _aggregate_firm_year(raw)

    status = pd.DataFrame(
        [
            {
                "status": "EVALUATED",
                "source_contract_status": "PASS",
                "source_path": str(path),
                "source_schema": "BCTC_AUDIT_ANNUAL_LONG_V1",
                "ticker_column": "issuer_ticker",
                "year_column": "year",
                "entity_name_column": "firm_name_raw" if "firm_name_raw" in raw else "",
                "period_type_filter": "annual",
                "statement_scope_filter": "Hợp nhất",
                "audit_status_filter": "audited",
                "audit_indicator_filter": "audit_firm",
                "auditor_name_column": "audit_firm_raw",
                "auditor_value_check_column": "audit_value_raw",
                "source_rows": source_rows,
                "audit_firm_rows": len(raw),
                "firm_years": len(firm_year),
                "duplicate_firm_year_rows": int(len(raw) - len(firm_year)),
                "ambiguous_firm_years": int(
                    firm_year["auditor_group"].eq("AMBIGUOUS").sum()
                ),
                "audit_firm_value_mismatches": int((~firm_value_equal).sum()),
                "source_files": int(raw["source_file"].nunique(dropna=True)),
                "minimum_year": int(raw["fiscal_year"].min()),
                "maximum_year": int(raw["fiscal_year"].max()),
                "big4_firm_years": int(firm_year["auditor_group"].eq("BIG4").sum()),
                "non_big4_firm_years": int(
                    firm_year["auditor_group"].eq("NON_BIG4").sum()
                ),
                "unknown_firm_years": int(
                    firm_year["auditor_group"].eq("UNKNOWN").sum()
                ),
            }
        ]
    )
    return firm_year, name_map, status
