from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis_window import AnalysisWindow
from .bctc_auditor_source import canonicalize_entity_ticker
from .io import read_long_chunks

KEYS = ["issuer_ticker", "raw_exchange", "fiscal_year", "audit_status", "scope"]


def _analysis_window(config: dict[str, Any]) -> AnalysisWindow:
    input_cfg = config.get("input", {})
    model_cfg = config.get("models", {})
    signal_cfg = config.get("signal", {})
    return AnalysisWindow.from_mapping(
        config.get("analysis_window"),
        fallback={
            "source_start_year": input_cfg.get("minimum_year", 2015),
            "source_end_year": input_cfg.get("maximum_year", 2025),
            "training_start_year": model_cfg.get("training_start_year", 2015),
            "test_start_year": signal_cfg.get("minimum_test_year", 2016),
            "test_end_year": signal_cfg.get("maximum_test_year", 2025),
        },
    )


def profile_input(
    path: str | Path,
    chunksize: int = 250_000,
    minimum_year: int = 2015,
    maximum_year: int = 2025,
) -> dict[str, Any]:
    import csv
    import gzip

    counts = {
        name: Counter()
        for name in [
            "audit_status",
            "statement_family",
            "scope",
            "unit",
            "identity_match_status",
            "retrospective_eligible",
            "prospective_flag",
        ]
    }
    item_counts: Counter[str] = Counter()
    item_names: dict[str, str] = {}
    tickers: set[str] = set()
    years: set[int] = set()
    included_rows = 0
    scanned_rows = 0
    excluded_rows = 0
    invalid_year_rows = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned_rows += 1
            try:
                fiscal_year = int(float(row.get("fiscal_year", "")))
            except (TypeError, ValueError):
                invalid_year_rows += 1
                continue
            if not minimum_year <= fiscal_year <= maximum_year:
                excluded_rows += 1
                continue

            included_rows += 1
            years.add(fiscal_year)
            tickers.add(row.get("issuer_ticker", ""))
            for name, counter in counts.items():
                counter[row.get(name, "")] += 1
            item = row.get("source_item_id", "")
            item_counts[item] += 1
            item_names.setdefault(item, row.get("item_name_raw", ""))
    return {
        "rows": included_rows,
        "rows_scanned": scanned_rows,
        "rows_excluded_outside_window": excluded_rows,
        "rows_excluded_invalid_year": invalid_year_rows,
        "source_start_year_contract": minimum_year,
        "source_end_year_contract": maximum_year,
        "ticker_count": len(tickers),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "item_count": len(item_counts),
        "counts": {name: dict(counter) for name, counter in counts.items()},
        "items": [
            {
                "source_item_id": item,
                "item_name_raw": item_names.get(item),
                "rows": count,
            }
            for item, count in item_counts.most_common()
        ],
    }


def extract_wide_panel(path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    item_map: dict[str, str] = config["items"]
    reverse = {source_id: variable for variable, source_id in item_map.items()}
    input_cfg = config["input"]
    window = _analysis_window(config)
    header = list(
        pd.read_csv(
            path,
            compression="gzip" if str(path).endswith(".gz") else "infer",
            nrows=0,
        ).columns
    )
    firm_name_column = next(
        (
            column
            for column in ("firm_name_raw", "firm_name", "company_name")
            if column in header
        ),
        None,
    )
    optional_columns = [firm_name_column] if firm_name_column else []
    usecols = KEYS + optional_columns + [
        "source_item_id",
        "value_numeric",
        "identity_match_status",
        "retrospective_eligible",
        "prospective_flag",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in read_long_chunks(path, int(input_cfg["chunksize"]), usecols):
        chunk["fiscal_year"] = pd.to_numeric(chunk["fiscal_year"], errors="coerce")
        chunk = chunk.loc[window.source_mask(chunk["fiscal_year"])].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["source_item_id"].isin(reverse)]
        if chunk.empty:
            continue
        chunk = chunk[
            chunk["identity_match_status"].isin(input_cfg["allowed_identity_status"])
            & (chunk["retrospective_eligible"].astype(str) == "1")
            & (chunk["prospective_flag"].astype(str) == "0")
            & (chunk["scope"] == input_cfg["required_scope"])
        ].copy()
        if chunk.empty:
            continue
        firm_names = (
            chunk[firm_name_column]
            if firm_name_column
            else pd.Series(pd.NA, index=chunk.index)
        )
        chunk["issuer_ticker"] = [
            canonicalize_entity_ticker(ticker, firm_name)
            for ticker, firm_name in zip(
                chunk["issuer_ticker"], firm_names, strict=True
            )
        ]
        chunk["variable"] = chunk["source_item_id"].map(reverse)
        chunk["value_numeric"] = pd.to_numeric(chunk["value_numeric"], errors="coerce")
        chunk["fiscal_year"] = chunk["fiscal_year"].astype("Int64")
        chunk = chunk.dropna(subset=["issuer_ticker", "fiscal_year", "value_numeric"])
        frames.append(chunk[KEYS + ["variable", "value_numeric"]])
    if not frames:
        raise ValueError("No configured financial-statement items were found in the input data")
    long = pd.concat(frames, ignore_index=True)
    duplicate_counts = long.groupby(KEYS + ["variable"], observed=True).size()
    if (duplicate_counts > 1).any():
        long = long.groupby(
            KEYS + ["variable"], as_index=False, observed=True
        )["value_numeric"].median()
    wide = long.pivot(
        index=KEYS, columns="variable", values="value_numeric"
    ).reset_index()
    wide.columns.name = None
    return wide


def _coalesce(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame:
            result = result.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return result


def build_accrual_features(wide: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    window = _analysis_window(config)
    frame = wide.copy()
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    frame = frame.loc[window.source_mask(frame["fiscal_year"])].copy()
    frame = frame.dropna(subset=["fiscal_year"])
    if frame.empty:
        raise ValueError("No rows remain inside the TT200 source window")
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["cfo"] = _coalesce(frame, ["cfo_indirect", "cfo_direct"])
    frame["inventory"] = _coalesce(frame, ["inventory_gross", "inventory_net"])
    frame["ppe"] = _coalesce(frame, ["ppe_gross", "ppe_net"])

    audited_label = config["input"]["audited_label"]
    audited = frame[frame["audit_status"] == audited_label].copy()
    lag_columns = [
        "assets",
        "revenue",
        "receivables",
        "current_assets",
        "cash",
        "current_liabilities",
        "short_term_debt",
        "tax_payable",
    ]
    lag = audited[
        ["issuer_ticker", "fiscal_year"]
        + [c for c in lag_columns if c in audited]
    ].copy()
    lag["fiscal_year"] += 1
    lag = lag.rename(
        columns={c: f"lag_{c}_audited" for c in lag_columns if c in lag}
    )
    frame = frame.merge(
        lag,
        on=["issuer_ticker", "fiscal_year"],
        how="left",
        validate="many_to_one",
    )

    lag_assets = pd.to_numeric(frame.get("lag_assets_audited"), errors="coerce")
    minimum_assets = float(config["panel"].get("minimum_lag_assets", 1.0))
    valid_scale = lag_assets.abs() >= minimum_assets
    frame["lag_assets"] = lag_assets.where(valid_scale)

    frame["ta_cashflow"] = pd.to_numeric(frame.get("pat"), errors="coerce") - frame["cfo"]
    dca = pd.to_numeric(frame.get("current_assets"), errors="coerce") - pd.to_numeric(
        frame.get("lag_current_assets_audited"), errors="coerce"
    )
    dcash = pd.to_numeric(frame.get("cash"), errors="coerce") - pd.to_numeric(
        frame.get("lag_cash_audited"), errors="coerce"
    )
    dcl = pd.to_numeric(frame.get("current_liabilities"), errors="coerce") - pd.to_numeric(
        frame.get("lag_current_liabilities_audited"), errors="coerce"
    )
    dstd = pd.to_numeric(frame.get("short_term_debt"), errors="coerce") - pd.to_numeric(
        frame.get("lag_short_term_debt_audited"), errors="coerce"
    )
    dtax = pd.to_numeric(frame.get("tax_payable"), errors="coerce") - pd.to_numeric(
        frame.get("lag_tax_payable_audited"), errors="coerce"
    )
    depreciation = pd.to_numeric(frame.get("depreciation"), errors="coerce")
    frame["ta_balance_sheet"] = (dca - dcash) - (dcl - dstd - dtax) - depreciation

    primary = config["panel"].get("primary_total_accruals", "cash_flow")
    cash_available = frame["ta_cashflow"].notna()
    balance_available = frame["ta_balance_sheet"].notna()
    if primary == "cash_flow":
        frame["total_accruals"] = frame["ta_cashflow"].where(
            cash_available, frame["ta_balance_sheet"]
        )
        frame["ta_source"] = np.select(
            [cash_available, (~cash_available) & balance_available],
            ["cash_flow", "balance_sheet"],
            default="missing",
        )
    else:
        frame["total_accruals"] = frame["ta_balance_sheet"].where(
            balance_available, frame["ta_cashflow"]
        )
        frame["ta_source"] = np.select(
            [balance_available, (~balance_available) & cash_available],
            ["balance_sheet", "cash_flow"],
            default="missing",
        )

    frame["drev"] = pd.to_numeric(frame.get("revenue"), errors="coerce") - pd.to_numeric(
        frame.get("lag_revenue_audited"), errors="coerce"
    )
    frame["drec"] = pd.to_numeric(frame.get("receivables"), errors="coerce") - pd.to_numeric(
        frame.get("lag_receivables_audited"), errors="coerce"
    )
    scale = frame["lag_assets"]
    frame["ta_scaled"] = frame["total_accruals"] / scale
    frame["inv_assets"] = 1.0 / scale
    frame["drev_scaled"] = frame["drev"] / scale
    frame["drec_scaled"] = frame["drec"] / scale
    frame["drev_drec_scaled"] = (frame["drev"] - frame["drec"]) / scale
    frame["ppe_scaled"] = frame["ppe"] / scale
    frame["roa"] = pd.to_numeric(frame.get("pat"), errors="coerce") / scale
    frame["cfo_scaled"] = frame["cfo"] / scale
    frame["loss"] = (pd.to_numeric(frame.get("pat"), errors="coerce") < 0).astype(float)
    frame["drev_drec_sq"] = frame["drev_drec_scaled"] ** 2
    frame["year_centered"] = frame["fiscal_year"] - frame["fiscal_year"].median()
    frame["firm_id"] = frame["issuer_ticker"].astype(str)

    exchange = pd.get_dummies(
        frame["raw_exchange"].fillna("UNKNOWN"), prefix="exchange", dtype=float
    )
    frame = pd.concat([frame, exchange], axis=1)

    pair_count = frame.groupby(
        ["issuer_ticker", "fiscal_year"], observed=True
    )["audit_status"].nunique()
    complete_pairs = pair_count[pair_count >= 2].index
    frame = frame.set_index(["issuer_ticker", "fiscal_year"])
    frame = frame.loc[frame.index.isin(complete_pairs)].reset_index()
    if frame["fiscal_year"].lt(window.source_start_year).any():
        raise AssertionError("Processed panel contains a pre-TT200 fiscal year")
    return frame.sort_values(
        ["fiscal_year", "issuer_ticker", "audit_status"]
    ).reset_index(drop=True)


def build_and_save_panel(
    input_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    wide = extract_wide_panel(input_path, config)
    panel = build_accrual_features(wide, config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(
        output_path,
        index=False,
        compression="gzip" if str(output_path).endswith(".gz") else None,
    )
    return panel
