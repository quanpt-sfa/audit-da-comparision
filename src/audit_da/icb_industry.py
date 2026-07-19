from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import pandas as pd


def _normalise_label(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


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


def _truthy(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin(
        {"1", "true", "yes", "y", "financial", "tai chinh", "tài chính"}
    )


def _first_known(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


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
            "issuer_ticker",
            "ticker",
            "stock_code",
            "ticker_code",
            "stock_symbol",
            "symbol",
            "code",
            "ma_ck",
            "ma_chung_khoan",
            "mack",
            "security_code",
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
            "icb_l1",
            "icb_industry_name",
            "icb_level_1_name",
            "icb_level1_name",
            "icb_lv1_name",
            "icb1_name",
            "icb_industry",
            "industry_name",
            "industry",
            "sector_name",
            "sector",
            "nganh_cap_1",
            "nganh_icb_cap_1",
            "ten_nganh_icb",
            "ten_nganh",
        ],
    )
    code_candidates = settings.get(
        "icb_code_candidates",
        [
            "icb_industry_code",
            "icb_level_1_code",
            "icb_level1_code",
            "icb_lv1_code",
            "icb1_code",
            "icb_code",
            "industry_code",
            "sector_code",
            "ma_nganh_icb",
            "ma_nganh_cap_1",
        ],
    )
    industry_column = settings.get("industry_column") or _find_column(
        normalised, industry_candidates
    )
    code_column = settings.get("icb_code_column") or _find_column(
        normalised, code_candidates
    )

    retained_requested = list(settings.get("retain_columns", []))
    icb_levels = list(
        settings.get(
            "icb_level_columns",
            ["icb_l1", "icb_l2", "icb_l3", "icb_l4", "icb_l5"],
        )
    )
    retained_columns = [
        column
        for column in dict.fromkeys(retained_requested + icb_levels)
        if column in raw.columns
    ]

    output = pd.DataFrame(index=raw.index)
    output["issuer_ticker"] = (
        raw[ticker_column].fillna("").astype(str).str.strip().str.upper()
    )
    output["issuer_ticker"] = output["issuer_ticker"].str.replace(
        r"\.(HO|HN|UPCOM)$", "", regex=True
    )
    if year_column is not None:
        parsed_year = pd.to_numeric(raw[year_column], errors="coerce").astype(
            "Int64"
        )
        if parsed_year.notna().any():
            output["fiscal_year"] = parsed_year

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
    for column in retained_columns:
        output[column] = raw[column]

    known_financial = output["industry_name"].notna()
    if explicit_flag is not None:
        financial = _truthy(raw[explicit_flag]).astype("boolean")
        financial_source = explicit_flag
        if settings.get("missing_industry_is_unknown", False):
            financial = financial.where(known_financial, pd.NA)
    else:
        exact_values = settings.get("financial_industry_values")
        if exact_values:
            normalised_values = {
                _normalise_label(value) for value in exact_values
            }
            labels = output["industry_name"].map(_normalise_label)
            financial = labels.isin(normalised_values).astype("boolean")
            financial = financial.where(known_financial, pd.NA)
            financial_source = (
                f"exact values from {industry_column}: "
                + "|".join(str(value) for value in exact_values)
            )
        else:
            pattern = settings.get(
                "financial_industry_regex",
                r"financial|bank|insurance|securit|tai chinh|ngan hang|bao hiem|chung khoan",
            )
            names = (
                output["industry_name"]
                .fillna("")
                .astype(str)
                .map(_normalise_label)
                .str.replace("_", " ", regex=False)
            )
            financial = names.str.contains(
                pattern, regex=True, na=False
            ).astype("boolean")
            prefixes = tuple(
                str(x)
                for x in settings.get("financial_icb_prefixes", ["8"])
            )
            codes = (
                output["icb_industry_code"]
                .fillna("")
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )
            if prefixes and code_column is not None:
                financial |= codes.str.startswith(prefixes)
            if settings.get("missing_industry_is_unknown", False):
                financial = financial.where(known_financial, pd.NA)
            financial_source = "industry_name/icb_code"
    output["financial_flag"] = financial.astype("boolean")

    output = output[output["issuer_ticker"].ne("")].copy()
    keys = ["issuer_ticker"] + (
        ["fiscal_year"] if "fiscal_year" in output else []
    )
    duplicate_rows = int(output.duplicated(keys, keep=False).sum())

    aggregation: dict[str, tuple[str, Any]] = {
        "industry_name": ("industry_name", _first_known),
        "icb_industry_code": ("icb_industry_code", _first_known),
        "financial_flag": (
            "financial_flag",
            lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA,
        ),
    }
    for column in retained_columns:
        if column not in aggregation:
            aggregation[column] = (column, _first_known)

    output = (
        output.sort_values(keys)
        .groupby(keys, as_index=False, observed=True, dropna=False)
        .agg(**aggregation)
    )
    output["financial_flag"] = output["financial_flag"].astype("boolean")

    known_financial_output = output["financial_flag"].notna()
    status = pd.DataFrame(
        [
            {
                "industry_path": str(Path(path)),
                "encoding": encoding,
                "raw_rows": len(raw),
                "mapping_rows": len(output),
                "unique_tickers": output["issuer_ticker"].nunique(),
                "financial_rows": int(output["financial_flag"].eq(True).sum()),
                "nonfinancial_rows": int(
                    output["financial_flag"].eq(False).sum()
                ),
                "unknown_financial_rows": int(
                    output["financial_flag"].isna().sum()
                ),
                "ticker_column": ticker_column,
                "year_column": year_column or "",
                "industry_column": industry_column or "",
                "icb_code_column": code_column or "",
                "financial_source": financial_source,
                "duplicate_input_rows": duplicate_rows,
                "known_financial_share": float(known_financial_output.mean()),
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

    metadata_columns = [
        column for column in mapping.columns if column not in keys
    ]
    for column in metadata_columns:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    mapping_for_join = mapping.copy()
    mapping_for_join["_industry_key_match"] = True
    merged = frame.merge(
        mapping_for_join, on=keys, how="left", validate="many_to_one"
    )
    key_matched = merged["_industry_key_match"].eq(True)
    industry_known = merged["industry_name"].notna()
    if "icb_l1" in merged.columns:
        industry_known |= merged["icb_l1"].notna()

    status = pd.DataFrame(
        [
            {
                "panel_rows": len(merged),
                "panel_firm_years": merged[
                    ["issuer_ticker", "fiscal_year"]
                ]
                .drop_duplicates()
                .shape[0],
                "key_matched_rows": int(key_matched.sum()),
                "key_matched_share": float(key_matched.mean()),
                "key_matched_tickers": merged.loc[
                    key_matched, "issuer_ticker"
                ].nunique(),
                "known_industry_rows": int(industry_known.sum()),
                "unknown_industry_rows": int(
                    (key_matched & ~industry_known).sum()
                ),
                "known_financial_rows": int(
                    merged["financial_flag"].notna().sum()
                ),
                "financial_panel_rows": int(
                    merged["financial_flag"].eq(True).sum()
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
