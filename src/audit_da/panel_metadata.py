from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import re
import unicodedata

import numpy as np
import pandas as pd

from .bctc_auditor_source import (
    canonicalize_entity_ticker,
    load_bctc_audit_annual_long,
)

PANEL_KEYS = ["issuer_ticker", "fiscal_year"]
INDUSTRY_COLUMNS = [
    "industry_name",
    "icb_industry_code",
    "icb_l1",
    "icb_l2",
    "icb_l3",
    "icb_l4",
    "icb_l5",
]
AUDIT_COLUMNS = [
    "auditor_name_raw",
    "auditor_name_normalized",
    "auditor_brand",
    "auditor_group",
    "big4_flag",
    "auditor_name_status",
    "auditor_firm_year_status",
    "audit_opinion_raw",
    "audit_opinion_group",
    "audit_opinion_clean_flag",
    "audit_opinion_status",
]


def _token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _choose(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = list(columns)
    normalized = {_token(column): column for column in available}
    for candidate in candidates:
        hit = normalized.get(_token(candidate))
        if hit is not None:
            return hit
    return None


def _first_known(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def _classify_financial(industry: pd.Series) -> pd.Series:
    token = industry.map(_token)
    pattern = re.compile(
        r"(^|_)(tai_chinh|ngan_hang|bao_hiem|chung_khoan|financial|bank|insurance|securit)(_|$)"
    )
    return token.map(lambda value: bool(pattern.search(value)) if value else pd.NA).astype("boolean")


def load_industry_metadata(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    raw = pd.read_csv(path, low_memory=False)
    ticker_col = _choose(raw.columns, ["issuer_ticker", "ticker", "stock_code", "ma_ck"])
    year_col = _choose(raw.columns, ["fiscal_year", "year", "report_year", "nam"])
    firm_col = _choose(raw.columns, ["firm_name_raw", "firm_name", "company_name", "ten_cong_ty"])
    if ticker_col is None:
        raise ValueError(f"Industry metadata has no ticker column: {list(raw.columns)}")

    output = pd.DataFrame(index=raw.index)
    firm_names = raw[firm_col] if firm_col else pd.Series(pd.NA, index=raw.index)
    output["issuer_ticker"] = [
        canonicalize_entity_ticker(ticker, firm_name)
        for ticker, firm_name in zip(raw[ticker_col], firm_names, strict=True)
    ]
    keys = ["issuer_ticker"]
    if year_col is not None:
        output["fiscal_year"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")
        keys.append("fiscal_year")

    source_columns: dict[str, str | None] = {
        "industry_name": _choose(raw.columns, ["industry_name", "icb_l1", "icb_industry"]),
        "icb_industry_code": _choose(raw.columns, ["icb_industry_code", "icb_code"]),
        "icb_l1": _choose(raw.columns, ["icb_l1", "icb_level_1_name"]),
        "icb_l2": _choose(raw.columns, ["icb_l2", "icb_level_2_name"]),
        "icb_l3": _choose(raw.columns, ["icb_l3", "icb_level_3_name"]),
        "icb_l4": _choose(raw.columns, ["icb_l4", "icb_level_4_name"]),
        "icb_l5": _choose(raw.columns, ["icb_l5", "icb_level_5_name"]),
    }
    for target, source in source_columns.items():
        output[target] = raw[source] if source is not None else pd.NA

    output = output[output["issuer_ticker"].ne("")].copy()
    if "fiscal_year" in output:
        output = output[output["fiscal_year"].notna()].copy()
        output["fiscal_year"] = output["fiscal_year"].astype(int)

    conflicts: list[dict[str, object]] = []
    for group_key, group in output.groupby(keys, observed=True, dropna=False):
        for column in INDUSTRY_COLUMNS:
            values = group[column].dropna().astype(str).str.strip()
            if values.nunique() > 1:
                conflicts.append({"key": group_key, "column": column, "values": " | ".join(sorted(values.unique()))})
    if conflicts:
        preview = conflicts[:10]
        raise ValueError(f"Conflicting industry metadata after entity canonicalisation: {preview}")

    aggregation = {column: (column, _first_known) for column in INDUSTRY_COLUMNS}
    output = output.groupby(keys, as_index=False, observed=True, dropna=False).agg(**aggregation)
    primary_industry = output["icb_l1"].where(output["icb_l1"].notna(), output["industry_name"])
    output["financial_flag"] = _classify_financial(primary_industry)
    status = pd.DataFrame(
        [{
            "source_path": str(path),
            "rows": len(raw),
            "mapping_rows": len(output),
            "financial_rows": int(output["financial_flag"].eq(True).sum()),
            "nonfinancial_rows": int(output["financial_flag"].eq(False).sum()),
            "unknown_rows": int(output["financial_flag"].isna().sum()),
            "ticker_column": ticker_col,
            "year_column": year_col or "",
            "firm_name_column": firm_col or "",
            "status": "LOADED",
        }]
    )
    return output, status


def _classify_opinion(value: Any) -> tuple[str, float | np.floating, str]:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    token = _token(raw)
    if not token:
        return "unknown", np.nan, "MISSING"
    if token in {"chap_nhan_toan_phan", "unmodified", "unqualified", "clean"}:
        return "unmodified", 1.0, "MAPPED"
    if "ngoai_tru" in token or token == "qualified" or token.startswith("qualified_"):
        return "qualified", 0.0, "MAPPED"
    if "tu_choi" in token or "disclaimer" in token:
        return "disclaimer", 0.0, "MAPPED"
    if "trai_nguoc" in token or "adverse" in token:
        return "adverse", 0.0, "MAPPED"
    return "other", np.nan, "UNMAPPED"


def load_audit_metadata(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    auditor, _, auditor_status = load_bctc_audit_annual_long(path)

    columns = list(pd.read_csv(path, nrows=0).columns)
    required = {
        "issuer_ticker", "year", "period_type", "statement_scope",
        "audit_status", "audit_indicator", "audit_value_raw",
    }
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Audit metadata schema mismatch; missing columns: {missing}")
    optional = [column for column in ("firm_name_raw", "audit_opinion_raw") if column in columns]
    usecols = list(required) + optional
    chunks: list[pd.DataFrame] = []
    source_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        source_rows += len(chunk)
        keep = (
            chunk["period_type"].map(_token).eq("annual")
            & chunk["statement_scope"].map(_token).isin({"hop_nhat", "consolidated"})
            & chunk["audit_status"].map(_token).eq("audited")
            & chunk["audit_indicator"].map(_token).eq("audit_opinion")
        )
        selected = chunk.loc[keep].copy()
        if not selected.empty:
            chunks.append(selected)
    opinion_raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    if opinion_raw.empty:
        opinions = pd.DataFrame(columns=PANEL_KEYS + [
            "audit_opinion_raw", "audit_opinion_group", "audit_opinion_clean_flag", "audit_opinion_status"
        ])
    else:
        firm_names = (
            opinion_raw["firm_name_raw"]
            if "firm_name_raw" in opinion_raw
            else pd.Series(pd.NA, index=opinion_raw.index)
        )
        opinion_raw["issuer_ticker"] = [
            canonicalize_entity_ticker(ticker, firm_name)
            for ticker, firm_name in zip(opinion_raw["issuer_ticker"], firm_names, strict=True)
        ]
        opinion_raw["fiscal_year"] = pd.to_numeric(opinion_raw["year"], errors="coerce")
        opinion_raw["audit_opinion_raw"] = (
            opinion_raw["audit_opinion_raw"].where(
                opinion_raw["audit_opinion_raw"].notna(), opinion_raw["audit_value_raw"]
            )
            if "audit_opinion_raw" in opinion_raw
            else opinion_raw["audit_value_raw"]
        )
        opinion_raw = opinion_raw[
            opinion_raw["issuer_ticker"].ne("")
            & opinion_raw["fiscal_year"].notna()
            & opinion_raw["audit_opinion_raw"].notna()
        ].copy()
        opinion_raw["fiscal_year"] = opinion_raw["fiscal_year"].astype(int)
        classified = opinion_raw["audit_opinion_raw"].map(_classify_opinion)
        opinion_raw[["audit_opinion_group", "audit_opinion_clean_flag", "audit_opinion_status"]] = pd.DataFrame(
            classified.tolist(), index=opinion_raw.index
        )
        rows: list[dict[str, object]] = []
        for (ticker, year), group in opinion_raw.groupby(PANEL_KEYS, observed=True, sort=False):
            groups = sorted(set(group["audit_opinion_group"].dropna().astype(str)) - {"unknown"})
            raw_values = sorted(set(group["audit_opinion_raw"].dropna().astype(str)))
            if len(groups) <= 1:
                first = group.iloc[0]
                row = {
                    "audit_opinion_raw": first["audit_opinion_raw"],
                    "audit_opinion_group": first["audit_opinion_group"],
                    "audit_opinion_clean_flag": first["audit_opinion_clean_flag"],
                    "audit_opinion_status": "EXACT_ONE_OPINION" if len(group) == 1 else "CONSISTENT_DUPLICATES",
                }
            else:
                row = {
                    "audit_opinion_raw": " | ".join(raw_values),
                    "audit_opinion_group": "ambiguous",
                    "audit_opinion_clean_flag": np.nan,
                    "audit_opinion_status": "AMBIGUOUS_MULTIPLE_OPINIONS",
                }
            row.update({"issuer_ticker": ticker, "fiscal_year": int(year)})
            rows.append(row)
        opinions = pd.DataFrame(rows)

    metadata = auditor.merge(opinions, on=PANEL_KEYS, how="outer", validate="one_to_one")
    status = auditor_status.copy()
    status["source_rows_with_opinion_scan"] = source_rows
    status["audit_opinion_firm_years"] = len(opinions)
    status["audit_metadata_firm_years"] = len(metadata)
    status["ambiguous_opinion_firm_years"] = int(
        opinions.get("audit_opinion_group", pd.Series(dtype=str)).eq("ambiguous").sum()
    )
    return metadata, status


def _derive_financial_flag(frame: pd.DataFrame) -> pd.Series:
    if "financial_flag" in frame:
        return frame["financial_flag"].astype("boolean")
    for column in ("icb_l1", "industry_name"):
        if column in frame:
            return _classify_financial(frame[column])
    return pd.Series(pd.NA, index=frame.index, dtype="boolean")


def enrich_panel_metadata(
    panel: pd.DataFrame,
    *,
    industry_path: str | Path | None = None,
    audit_metadata_path: str | Path | None = None,
    unknown_industry_policy: str = "exclude",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frame = panel.copy()
    frame["issuer_ticker"] = frame["issuer_ticker"].astype(str).str.strip().str.upper()
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")
    statuses: dict[str, pd.DataFrame] = {}

    if industry_path is not None:
        industry, status = load_industry_metadata(industry_path)
        statuses["industry"] = status
        merge_keys = ["issuer_ticker"] + (["fiscal_year"] if "fiscal_year" in industry else [])
        drop_columns = [column for column in INDUSTRY_COLUMNS + ["financial_flag"] if column in frame]
        frame = frame.drop(columns=drop_columns).merge(
            industry, on=merge_keys, how="left", validate="many_to_one"
        )

    if audit_metadata_path is not None:
        audit, status = load_audit_metadata(audit_metadata_path)
        statuses["audit"] = status
        drop_columns = [column for column in AUDIT_COLUMNS if column in frame]
        frame = frame.drop(columns=drop_columns).merge(
            audit, on=PANEL_KEYS, how="left", validate="many_to_one"
        )

    frame["financial_flag"] = _derive_financial_flag(frame)
    unknown_policy = str(unknown_industry_policy).lower()
    if unknown_policy not in {"exclude", "include", "error"}:
        raise ValueError(f"Unknown industry policy: {unknown_industry_policy}")
    if unknown_policy == "error" and frame["financial_flag"].isna().any():
        raise ValueError("Financial classification is missing for one or more panel rows")

    eligible = frame["financial_flag"].eq(False)
    if unknown_policy == "include":
        eligible |= frame["financial_flag"].isna()
    frame["analysis_eligible"] = eligible.astype(bool)
    frame["exclusion_reason"] = np.select(
        [frame["financial_flag"].eq(True), frame["financial_flag"].isna()],
        ["financial_firm", "unknown_industry"],
        default="eligible",
    )
    return frame, statuses


def select_analysis_sample(
    panel: pd.DataFrame,
    sample_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = sample_config or {}
    exclude_financial = bool(config.get("exclude_financial_firms", True))
    unknown_policy = str(config.get("unknown_industry_policy", "exclude")).lower()
    frame = panel.copy()
    frame["financial_flag"] = _derive_financial_flag(frame)
    if unknown_policy == "error" and frame["financial_flag"].isna().any():
        raise ValueError("Cannot construct analysis sample: unknown financial classifications remain")

    if exclude_financial:
        keep = frame["financial_flag"].eq(False)
        if unknown_policy == "include":
            keep |= frame["financial_flag"].isna()
    else:
        keep = pd.Series(True, index=frame.index)

    firm_years = frame[PANEL_KEYS].drop_duplicates()
    financial_firm_years = frame.loc[frame["financial_flag"].eq(True), PANEL_KEYS].drop_duplicates()
    unknown_firm_years = frame.loc[frame["financial_flag"].isna(), PANEL_KEYS].drop_duplicates()
    selected = frame.loc[keep].copy().reset_index(drop=True)
    selected_firm_years = selected[PANEL_KEYS].drop_duplicates()
    manifest = pd.DataFrame(
        [
            {"stage": "master_panel", "rows": len(frame), "issuer_years": len(firm_years)},
            {"stage": "excluded_financial_firms", "rows": int(frame["financial_flag"].eq(True).sum()), "issuer_years": len(financial_firm_years)},
            {"stage": "excluded_unknown_industry", "rows": int((frame["financial_flag"].isna() & ~keep).sum()), "issuer_years": len(unknown_firm_years) if unknown_policy != "include" else 0},
            {"stage": "nonfinancial_analysis_panel", "rows": len(selected), "issuer_years": len(selected_firm_years)},
        ]
    )
    return selected, manifest
