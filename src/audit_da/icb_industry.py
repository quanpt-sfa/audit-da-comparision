from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import numpy as np
import pandas as pd


def _normalise_label(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


def _read_csv_flexible(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding, low_memory=False)
            return frame, encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Unable to read industry file {path}: {last_error}")


def _find_column(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _truthy(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "financial", "tai chinh", "tài chính"})


def load_icb_industry(
    path: str | Path,
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or {}
    raw, encoding = _read_csv_flexible(path)
    normalised = {_normalise_label(column): column for column in raw.columns}

    ticker_column = settings.get("ticker_column") or _find_column(
        normalised,
        [
            "issuer_ticker", "ticker", "stock_code", "symbol", "code", "ma_ck",
            "ma_chung_khoan", "mack", "security_code",
        ],
    )
    if ticker_column is None:
        raise ValueError(
            "Could not identify a ticker column in industry file. "
            f"Available columns: {list(raw.columns)}"
        )

    year_column = settings.get("year_column") or _find_column(
        normalised,
        ["fiscal_year", "year", "nam", "report_year"],
    )
    explicit_flag = settings.get("financial_flag_column") or _find_column(
        normalised,
        ["financial_flag", "is_financial", "financial_sector", "tai_chinh"],
    )

    industry_candidates = settings.get(
        "industry_name_candidates",
        [
            "icb_industry_name", "icb_level_1_name", "icb1_name", "icb_industry",
            "industry_name", "industry", "sector_name", "sector", "nganh_cap_1",
            "ten_nganh_icb", "ten_nganh",
        ],
    )
    code_candidates = settings.get(
        "icb_code_candidates",
        [
            "icb_industry_code", "icb_level_1_code", "icb1_code", "icb_code",
            "industry_code", "sector_code", "ma_nganh_icb", "ma_nganh_cap_1",
        ],
    )
    industry_column = _find_column(normalised, industry_candidates)
    code_column = _find_column(normalised, code_candidates)

    output = pd.DataFrame(index=raw.index)
    output["issuer_ticker"] = (
        raw[ticker_column].fillna("").astype(str).str.strip().str.upper()
    )
    output["issuer_ticker"] = output["issuer_ticker"].str.replace(
        r"\.(HO|HN|UPCOM)$", "", regex=True
    )
    if year_column is not None:
        output["fiscal_year"] = pd.to_numeric(raw[year_column], errors="coerce").astype("Int64")
    if industry_column is not None:
        output["industry_name"] = raw[industry_column].astype("string")
    else:
        output["industry_name"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    if code_column is not None:
        output["icb_industry_code"] = raw[code_column].astype("string")
    else:
        output["icb_industry_code"] = pd.Series(pd.NA, index=raw.index, dtype="string")

    if explicit_flag is not None:
        financial = _truthy(raw[explicit_flag])
        financial_source = explicit_flag
    else:
        pattern = settings.get(
            "financial_industry_regex",
            r"financial|bank|insurance|securit|tai chinh|ngan hang|bao hiem|chung khoan",
        )
        names = output["industry_name"].fillna("").astype(str).map(_normalise_label)
        financial = names.str.contains(pattern, regex=True, na=False)
        prefixes = tuple(str(x) for x in settings.get("financial_icb_prefixes", ["8"]))
        codes = output["icb_industry_code"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        if prefixes:
            financial |= codes.str.startswith(prefixes)
        financial_source = "industry_name/icb_code"
    output["financial_flag"] = financial.astype(bool)

    output = output[output["issuer_ticker"].ne("")].copy()
    keys = ["issuer_ticker"] + (["fiscal_year"] if "fiscal_year" in output else [])
    duplicate_rows = int(output.duplicated(keys, keep=False).sum())
    output = (
        output.sort_values(keys)
        .groupby(keys, as_index=False, observed=True)
        .agg(
            industry_name=("industry_name", lambda x: x.dropna().astype(str).iloc[0] if x.notna().any() else pd.NA),
            icb_industry_code=("icb_industry_code", lambda x: x.dropna().astype(str).iloc[0] if x.notna().any() else pd.NA),
            financial_flag=("financial_flag", "max"),
        )
    )

    status = pd.DataFrame([
        {
            "industry_path": str(Path(path)),
            "encoding": encoding,
            "raw_rows": len(raw),
            "mapping_rows": len(output),
            "unique_tickers": output["issuer_ticker"].nunique(),
            "financial_rows": int(output["financial_flag"].sum()),
            "ticker_column": ticker_column,
            "year_column": year_column or "",
            "industry_column": industry_column or "",
            "icb_code_column": code_column or "",
            "financial_source": financial_source,
            "duplicate_input_rows": duplicate_rows,
            "status": "LOADED",
        }
    ])
    return output, status


def attach_icb_industry(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = panel.copy()
    frame["issuer_ticker"] = frame["issuer_ticker"].astype(str).str.strip().str.upper()
    keys = ["issuer_ticker"]
    if "fiscal_year" in mapping.columns:
        keys.append("fiscal_year")
        frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")

    for column in ["industry_name", "icb_industry_code", "financial_flag"]:
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    merged = frame.merge(mapping, on=keys, how="left", validate="many_to_one")
    matched = merged["industry_name"].notna() | merged["icb_industry_code"].notna()
    status = pd.DataFrame([
        {
            "panel_rows": len(merged),
            "panel_firm_years": merged[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0],
            "matched_rows": int(matched.sum()),
            "matched_share": float(matched.mean()),
            "matched_tickers": merged.loc[matched, "issuer_ticker"].nunique(),
            "financial_panel_rows": int(merged["financial_flag"].fillna(False).sum()),
            "status": "EVALUATED" if matched.any() else "NO_MATCHES",
        }
    ])
    unmatched = (
        merged.loc[~matched, ["issuer_ticker"]]
        .drop_duplicates()
        .sort_values("issuer_ticker")
        .reset_index(drop=True)
    )
    return merged, status, unmatched
