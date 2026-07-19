from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import KEYS
from .cfs_proxy_validate import (
    OUTCOMES,
    _common_keys,
    _metric_record,
    _numeric,
    _observed_outcomes,
)


def _sample_masks(
    merged: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    restrictions = settings.get("sample_restrictions", {})
    masks: dict[str, pd.Series] = {"full": pd.Series(True, index=merged.index)}
    status_rows: list[dict[str, Any]] = []

    exchange = merged.get("raw_exchange", pd.Series("", index=merged.index))
    exchange = exchange.astype(str).str.upper()
    listed_values = {
        str(value).upper()
        for value in restrictions.get("listed_exchanges", ["HOSE", "HNX", "UPCOM"])
    }
    listed = exchange.isin(listed_values)
    masks["listed_only"] = listed

    ticker = merged["issuer_ticker"].astype(str).str.upper()
    valid_ticker = ticker.str.fullmatch(r"[A-Z][A-Z0-9]{1,7}")
    masks["valid_ticker_only"] = valid_ticker

    lag_assets = _numeric(merged, "lag_assets")
    positive_lag = lag_assets[lag_assets.gt(0)]
    quantile = float(restrictions.get("lag_assets_floor_quantile", 0.01))
    lag_floor = float(positive_lag.quantile(quantile)) if len(positive_lag) else np.nan
    lag_ok = lag_assets.ge(lag_floor) if np.isfinite(lag_floor) else pd.Series(True, index=merged.index)
    masks["lag_assets_floor"] = lag_ok

    configured_flags = restrictions.get(
        "scale_scope_flag_columns",
        [
            "combined_scale_scope_flag", "scale_scope_flag",
            "asset_pre_post_gap_flag", "asset_growth_flag", "small_lag_assets_flag",
        ],
    )
    flag_columns = [column for column in configured_flags if column in merged.columns]
    if flag_columns:
        flagged = pd.Series(False, index=merged.index)
        for column in flag_columns:
            flagged |= merged[column].fillna(False).astype(bool)
        scale_scope_ok = ~flagged
        scale_status = "EVALUATED"
        scale_reason = ",".join(flag_columns)
    else:
        scale_scope_ok = pd.Series(True, index=merged.index)
        scale_status = "NOT_EVALUATED"
        scale_reason = "No configured scale/scope flag columns found"
    masks["exclude_scale_scope_flags"] = scale_scope_ok

    if "financial_flag" in merged.columns:
        known_financial = merged["financial_flag"].notna()
        nonfinancial = known_financial & merged["financial_flag"].eq(False)
        financial_status = "EVALUATED" if known_financial.any() else "NOT_EVALUATED"
        financial_reason = (
            f"financial_flag; known_share={float(known_financial.mean()):.6f}"
            if known_financial.any()
            else "financial_flag exists but contains no known values"
        )
    else:
        nonfinancial = pd.Series(False, index=merged.index)
        financial_status = "NOT_EVALUATED"
        financial_reason = "No industry mapping or financial_flag found"
    masks["nonfinancial_only"] = nonfinancial

    analysis_core = listed & valid_ticker & lag_ok & scale_scope_ok & nonfinancial
    masks["analysis_core"] = analysis_core
    if financial_status == "EVALUATED" and scale_status == "EVALUATED":
        core_status = "EVALUATED"
        core_reason = "All configured restrictions evaluated"
    else:
        core_status = "PARTIALLY_EVALUATED"
        missing = []
        if financial_status != "EVALUATED":
            missing.append("nonfinancial")
        if scale_status != "EVALUATED":
            missing.append("scale_scope")
        core_reason = "Unavailable restrictions: " + ",".join(missing)

    for name, mask in masks.items():
        if name == "exclude_scale_scope_flags":
            evaluation_status, reason = scale_status, scale_reason
        elif name == "nonfinancial_only":
            evaluation_status, reason = financial_status, financial_reason
        elif name == "analysis_core":
            evaluation_status, reason = core_status, core_reason
        else:
            evaluation_status, reason = "EVALUATED", "OK"
        status_rows.append(
            {
                "sample_restriction": name,
                "status": evaluation_status,
                "reason": reason,
                "model_rows": int(mask.sum()),
                "firm_years": int(merged.loc[mask, KEYS].drop_duplicates().shape[0]),
                "share_model_rows": float(mask.mean()),
                "lag_assets_floor": lag_floor if name in {"lag_assets_floor", "analysis_core"} else np.nan,
            }
        )
    return masks, pd.DataFrame(status_rows)


def _mode_definitions(
    merged: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.Series], dict[str, list[str] | None], pd.DataFrame]:
    primary_models = list(
        settings.get(
            "common_primary_models",
            settings.get("common_sample_models", []),
        )
    )
    all_models = list(
        settings.get(
            "common_all_models",
            primary_models + ["firm_history_deviation"],
        )
    )
    definitions: dict[str, list[str] | None] = {
        "model_available": None,
        "common_primary_models": primary_models,
        "common_all_models": all_models,
    }
    key_index = pd.MultiIndex.from_frame(merged[KEYS])
    masks: dict[str, pd.Series] = {
        "model_available": pd.Series(True, index=merged.index),
    }
    status_rows: list[dict[str, Any]] = [
        {
            "sample_mode": "model_available",
            "required_models": "",
            "required_model_count": 0,
            "common_firm_years": merged[KEYS].drop_duplicates().shape[0],
            "coverage_vs_all_firm_years": 1.0,
        }
    ]
    total_keys = max(merged[KEYS].drop_duplicates().shape[0], 1)
    for mode in ["common_primary_models", "common_all_models"]:
        models = definitions[mode] or []
        common = _common_keys(merged, models)
        masks[mode] = pd.Series(key_index.isin(common), index=merged.index)
        status_rows.append(
            {
                "sample_mode": mode,
                "required_models": "|".join(models),
                "required_model_count": len(models),
                "common_firm_years": len(common),
                "coverage_vs_all_firm_years": len(common) / total_keys,
            }
        )
    return masks, definitions, pd.DataFrame(status_rows)


def _common_mode_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    primary = summary[summary["sample_mode"].eq("common_primary_models")].copy()
    all_models = summary[summary["sample_mode"].eq("common_all_models")].copy()
    keys = ["proxy_model", "outcome", "sample_restriction"]
    metrics = ["rows", "positives", "prevalence", "auc", "average_precision", "top_decile_lift"]
    primary = primary[keys + metrics].rename(columns={column: f"primary_{column}" for column in metrics})
    all_models = all_models[keys + metrics].rename(columns={column: f"all_models_{column}" for column in metrics})
    comparison = primary.merge(all_models, on=keys, how="outer", validate="one_to_one")
    comparison["firm_year_rows_lost"] = comparison["primary_rows"] - comparison["all_models_rows"]
    comparison["coverage_ratio_all_vs_primary"] = comparison["all_models_rows"] / comparison["primary_rows"]
    comparison["delta_auc_all_minus_primary"] = comparison["all_models_auc"] - comparison["primary_auc"]
    comparison["delta_ap_all_minus_primary"] = comparison["all_models_average_precision"] - comparison["primary_average_precision"]
    comparison["delta_lift_all_minus_primary"] = comparison["all_models_top_decile_lift"] - comparison["primary_top_decile_lift"]
    return comparison


def validate_proxy_predictions_dual_common(
    predictions: pd.DataFrame,
    observed_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    observed = _observed_outcomes(observed_cases, settings)
    observed_columns = list(KEYS) + list(OUTCOMES)
    observed_columns += [
        column
        for column in settings.get("sample_restrictions", {}).get(
            "scale_scope_flag_columns",
            [
                "combined_scale_scope_flag", "scale_scope_flag",
                "asset_pre_post_gap_flag", "asset_growth_flag", "small_lag_assets_flag",
            ],
        )
        if column in observed.columns
    ]
    merged = predictions.merge(
        observed[observed_columns],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    restriction_masks, restriction_status = _sample_masks(merged, settings)
    mode_masks, mode_models, mode_status = _mode_definitions(merged, settings)

    summary: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    primary_cases: list[pd.DataFrame] = []
    all_cases: list[pd.DataFrame] = []

    for restriction_name, restriction_mask in restriction_masks.items():
        for mode_name, mode_mask in mode_masks.items():
            sample = merged.loc[restriction_mask & mode_mask].copy()
            required_models = mode_models[mode_name]
            if required_models is not None:
                sample = sample[sample["proxy_model"].isin(required_models)]
            if sample.empty:
                continue
            for _, group in sample.groupby("proxy_model", observed=True):
                for outcome in OUTCOMES:
                    summary.append(_metric_record(group, outcome, mode_name, restriction_name))
                    for year, year_group in group.groupby("fiscal_year", observed=True):
                        record = _metric_record(year_group, outcome, mode_name, restriction_name)
                        record["fiscal_year"] = int(year)
                        yearly.append(record)
                if restriction_name == "analysis_core":
                    case = group.copy()
                    case["sample_mode"] = mode_name
                    case["sample_restriction"] = restriction_name
                    if mode_name == "common_primary_models":
                        primary_cases.append(case)
                    elif mode_name == "common_all_models":
                        all_cases.append(case)

    summary_frame = pd.DataFrame(summary)
    incremental = pd.DataFrame()
    if not summary_frame.empty:
        reference_model = settings.get("incremental_reference_model", "raw_cfo_level")
        base = summary_frame[summary_frame["proxy_model"].eq(reference_model)][
            [
                "outcome", "sample_mode", "sample_restriction",
                "auc", "average_precision", "top_decile_lift",
            ]
        ].rename(
            columns={
                "auc": "reference_auc",
                "average_precision": "reference_average_precision",
                "top_decile_lift": "reference_top_decile_lift",
            }
        )
        incremental = summary_frame.merge(
            base,
            on=["outcome", "sample_mode", "sample_restriction"],
            how="left",
            validate="many_to_one",
        )
        incremental["delta_auc_vs_reference"] = incremental["auc"] - incremental["reference_auc"]
        incremental["delta_ap_vs_reference"] = incremental["average_precision"] - incremental["reference_average_precision"]
        incremental["delta_lift_vs_reference"] = incremental["top_decile_lift"] - incremental["reference_top_decile_lift"]

    primary_frame = pd.concat(primary_cases, ignore_index=True) if primary_cases else pd.DataFrame()
    all_frame = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()
    return {
        "cfs_shifting_proxy_cases": merged,
        "cfs_shifting_proxy_validation": summary_frame,
        "cfs_shifting_proxy_validation_by_year": pd.DataFrame(yearly),
        "cfs_shifting_proxy_incremental_comparison": incremental,
        "cfs_proxy_sample_restriction_status": restriction_status,
        "cfs_common_sample_status": mode_status,
        "cfs_common_sample_metric_comparison": _common_mode_comparison(summary_frame),
        "cfs_shifting_proxy_common_core_cases": primary_frame,
        "cfs_shifting_proxy_common_primary_core_cases": primary_frame,
        "cfs_shifting_proxy_common_all_core_cases": all_frame,
    }
