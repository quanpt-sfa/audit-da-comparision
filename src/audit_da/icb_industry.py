from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import pandas as pd


def _normalise_label(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")


def _read_csv_flexible(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            return frame, encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Unable to read industry file {path}: {last_error}")


def _find_column(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _truthy_nullable(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    nonmissing = text.notna() & text.ne("")
    result.loc[nonmissing] = text.loc[nonmissing].isin(
        {"1", "true", "yes", "y", "financial", "tai chinh", "tài chính"}
    )
    return result


def _first_nonmissing(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else pd.NA


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
    if ticker_column is None or ticker_column not in raw.columns:
        raise ValueError(
            "Could not identify configured ticker column in industry file. "
            f"Configured={ticker_column!r}; available={list(raw.columns)}"
        )

    year_column = settings.get("year_column")
    if year_column and year_column not in raw.columns:
        raise ValueError(
            f"Configured year column {year_column!r} not found. "
            f"Available columns: {list(raw.columns)}"
        )
    if not year_column:
        year_column = _find_column(
            normalised, ["fiscal_year", "year", "nam", "report_year"]
        )

    explicit_flag = settings.get("financial_flag_column")
    if explicit_flag and explicit_flag not in raw.columns:
        raise ValueError(
            f"Configured financial flag column {explicit_flag!r} not found. "
            f"Available columns: {list(raw.columns)}"
        )
    if not explicit_flag:
        explicit_flag = _find_column(
            normalised,
            ["financial_flag", "is_financial", "financial_sector", "tai_chinh"],
        )

    industry_column = settings.get("industry_column")
    if industry_column and industry_column not in raw.columns:
        raise ValueError(
            f"Configured industry column {industry_column!r} not found. "
            f"Available columns: {list(raw.columns)}"
        )
    if not industry_column:
        industry_column = _find_column(
            normalised,
            [
                "icb_l1", "icb_industry_name", "icb_level_1_name", "icb1_name",
                "icb_industry", "industry_name", "industry", "sector_name",
                "sector", "nganh_cap_1", "ten_nganh_icb", "ten_nganh",
            ],
        )

    code_column = settings.get("icb_code_column")
    if code_column and code_column not in raw.columns:
        raise ValueError(
            f"Configured ICB code column {code_column!r} not found. "
            f"Available columns: {list(raw.columns)}"
        )
    if not code_column:
        code_column = _find_column(
            normalised,
            [
                "icb_industry_code", "icb_level_1_code", "icb1_code", "icb_code",
                "industry_code", "sector_code", "ma_nganh_icb", "ma_nganh_cap_1",
            ],
        )

    icb_level_columns = list(settings.get("icb_level_columns", []))
    missing_levels = [column for column in icb_level_columns if column not in raw.columns]
    if missing_levels:
        raise ValueError(
            f"Configured ICB level columns not found: {missing_levels}. "
            f"Available columns: {list(raw.columns)}"
        )

    retain_columns = list(settings.get("retain_columns", []))
    missing_retained = [column for column in retain_columns if column not in raw.columns]
    if missing_retained:
        raise ValueError(
            f"Configured retained columns not found: {missing_retained}. "
            f"Available columns: {list(raw.columns)}"
        )

    output = pd.DataFrame(index=raw.index)
    output["issuer_ticker"] = raw[ticker_column].astype("string").str.strip().str.upper()
    output["issuer_ticker"] = output["issuer_ticker"].str.replace(
        r"\.(HO|HN|UPCOM)$", "", regex=True
    )

    if year_column is not None:
        output["fiscal_year"] = pd.to_numeric(
            raw[year_column], errors="coerce"
        ).astype("Int64")

    output["industry_name"] = (
        raw[industry_column].astype("string")
        if industry_column is not None
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    output["icb_industry_code"] = (
        raw[code_column].astype("string")
        if code_column is not None
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )

    for column in icb_level_columns + retain_columns:
        output[column] = raw[column].astype("string")

    if explicit_flag is not None:
        financial = _truthy_nullable(raw[explicit_flag])
        financial_source = explicit_flag
    else:
        financial = pd.Series(pd.NA, index=raw.index, dtype="boolean")
        industry_known = (
            output["industry_name"].notna()
            & output["industry_name"].str.strip().ne("")
        )
        exact_values = {
            _normalise_label(value)
            for value in settings.get("financial_industry_values", [])
        }
        names = output["industry_name"].map(_normalise_label)
        if exact_values:
            financial.loc[industry_known] = names.loc[industry_known].isin(exact_values)
            financial_source = f"exact values from {industry_column}"
        else:
            pattern = settings.get(
                "financial_industry_regex",
                r"financial|bank|insurance|securit|tai chinh|ngan hang|bao hiem|chung khoan",
            )
            financial.loc[industry_known] = names.loc[industry_known].str.contains(
                pattern, regex=True, na=False
            )
            financial_source = f"regex from {industry_column}"

        if code_column is not None:
            prefixes = tuple(
                str(value) for value in settings.get("financial_icb_prefixes", [])
            )
            codes = output["icb_industry_code"].astype("string").str.replace(
                r"\.0$", "", regex=True
            )
            code_known = codes.notna() & codes.str.strip().ne("")
            if prefixes:
                code_financial = codes.str.startswith(prefixes, na=False)
                financial.loc[code_known & code_financial] = True
                financial.loc[code_known & financial.isna()] = False
                financial_source += f"; code prefixes from {code_column}"

    output["financial_flag"] = financial
    output = output[
        output["issuer_ticker"].notna() & output["issuer_ticker"].ne("")
    ].copy()

    keys = ["issuer_ticker"] + (
        ["fiscal_year"] if "fiscal_year" in output.columns else []
    )
    duplicate_rows = int(output.duplicated(keys, keep=False).sum())

    aggregation: dict[str, tuple[str, Any]] = {
        "industry_name": ("industry_name", _first_nonmissing),
        "icb_industry_code": ("icb_industry_code", _first_nonmissing),
        "financial_flag": ("financial_flag", _first_nonmissing),
    }
    for column in icb_level_columns + retain_columns:
        aggregation[column] = (column, _first_nonmissing)

    output = (
        output.sort_values(keys)
        .groupby(keys, as_index=False, observed=True)
        .agg(**aggregation)
    )
    output["financial_flag"] = output["financial_flag"].astype("boolean")

    known_financial = output["financial_flag"].notna()
    status = pd.DataFrame(
        [
            {
                "industry_path": str(Path(path)),
                "encoding": encoding,
                "raw_rows": len(raw),
                "mapping_rows": len(output),
                "unique_tickers": output["issuer_ticker"].nunique(),
                "financial_rows": int(output["financial_flag"].fillna(False).sum()),
                "nonfinancial_rows": int(output["financial_flag"].eq(False).sum()),
                "unknown_financial_rows": int(output["financial_flag"].isna().sum()),
                "ticker_column": ticker_column,
                "year_column": year_column or "",
                "industry_column": industry_column or "",
                "icb_code_column": code_column or "",
                "financial_source": financial_source,
                "duplicate_input_rows": duplicate_rows,
                "known_financial_share": float(known_financial.mean()),
                "status": "LOADED",
            }
        ]
    )
    return output, status


def attach_icb_industry(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = panel.copy()
    frame["issuer_ticker"] = (
        frame["issuer_ticker"].astype(str).str.strip().str.upper()
    )
    keys = ["issuer_ticker"]
    if "fiscal_year" in mapping.columns:
        keys.append("fiscal_year")
        frame["fiscal_year"] = pd.to_numeric(
            frame["fiscal_year"], errors="coerce"
        ).astype("Int64")

    metadata_columns = [column for column in mapping.columns if column not in keys]
    for column in metadata_columns:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    mapping_for_join = mapping.copy()
    mapping_for_join["_industry_key_match"] = True
    merged = frame.merge(
        mapping_for_join, on=keys, how="left", validate="many_to_one"
    )
    key_matched = merged["_industry_key_match"].fillna(False).astype(bool)
    industry_known = merged["industry_name"].notna()
    if "icb_l1" in merged.columns:
        industry_known |= merged["icb_l1"].notna()

    status = pd.DataFrame(
        [
            {
                "panel_rows": len(merged),
                "panel_firm_years": merged[["issuer_ticker", "fiscal_year"]]
                .drop_duplicates()
                .shape[0],
                "key_matched_rows": int(key_matched.sum()),
                "key_matched_share": float(key_matched.mean()),
                "key_matched_tickers": merged.loc[
                    key_matched, "issuer_ticker"
                ].nunique(),
                "known_industry_rows": int(industry_known.sum()),
                "unknown_industry_rows": int((key_matched & ~industry_known).sum()),
                "known_financial_rows": int(merged["financial_flag"].notna().sum()),
                "financial_panel_rows": int(
                    merged["financial_flag"].fillna(False).sum()
                ),
                "unknown_financial_panel_rows": int(
                    merged["financial_flag"].isna().sum()
                ),
                "status": "EVALUATED" if key_matched.any() else "NO_MATCHES",
            }
        ]
    )
    unmatched = (
        merged.loc[~key_matched, ["issuer_ticker"]]
        .drop_duplicates()
        .sort_values("issuer_ticker")
        .reset_index(drop=True)
    )
    merged = merged.drop(columns=["_industry_key_match"])
    return merged, status, unmatched
