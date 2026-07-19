from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import trimmed_mean
from .diag_decomposition import build_decomposition_panel


def _add_scale_scope_flags(
    rows: pd.DataFrame,
    asset_gap_threshold: float,
    asset_growth_multiple_threshold: float,
    small_lag_assets_quantile: float,
) -> pd.DataFrame:
    out = rows.copy()
    lag_cut = float(out["lag_assets_common"].abs().quantile(small_lag_assets_quantile))
    out["flag_small_lag_assets"] = out["lag_assets_common"].abs().le(lag_cut)
    out["flag_asset_pre_post_gap"] = pd.to_numeric(
        out.get("asset_pre_post_gap"), errors="coerce"
    ).gt(asset_gap_threshold)
    out["flag_asset_growth"] = pd.to_numeric(
        out.get("asset_growth_multiple"), errors="coerce"
    ).gt(asset_growth_multiple_threshold)
    out["flag_scale_or_scope"] = out[
        ["flag_small_lag_assets", "flag_asset_pre_post_gap", "flag_asset_growth"]
    ].any(axis=1)
    return out


def _state(values: pd.Series, tolerance: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric > tolerance, numeric < -tolerance],
            ["positive", "negative"],
            default="near_zero",
        ),
        index=values.index,
    )


def _summarize_group(
    group: pd.DataFrame,
    tolerance: float,
    trim_fraction: float,
) -> dict[str, Any]:
    state = _state(group["reduction"], tolerance)
    positive = group.loc[state.eq("positive"), "reduction"]
    negative = group.loc[state.eq("negative"), "reduction"]
    return {
        "rows": len(group),
        "share_positive": float(state.eq("positive").mean()),
        "share_negative": float(state.eq("negative").mean()),
        "share_near_zero": float(state.eq("near_zero").mean()),
        "positive_minus_negative_share": float(
            state.eq("positive").mean() - state.eq("negative").mean()
        ),
        "mean_reduction": float(group["reduction"].mean()),
        "trimmed_mean_reduction": trimmed_mean(
            group["reduction"].to_numpy(float), trim_fraction
        ),
        "median_reduction": float(group["reduction"].median()),
        "mean_if_positive": float(positive.mean()) if len(positive) else np.nan,
        "mean_if_negative": float(negative.mean()) if len(negative) else np.nan,
        "sum_reduction": float(group["reduction"].sum()),
        "mean_abs_delta_cfo": float(group["abs_delta_cfo"].mean()),
        "median_abs_delta_cfo": float(group["abs_delta_cfo"].median()),
        "mean_abs_delta_pat": float(group["abs_delta_pat"].mean()),
        "median_abs_delta_pat": float(group["abs_delta_pat"].median()),
        "mean_shapley_cfo": float(group["shapley_cfo"].mean()),
        "mean_shapley_pat": float(group["shapley_pat"].mean()),
    }


def cfo_tilt_tables(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    rows = build_decomposition_panel(baseline, panel)
    rows = _add_scale_scope_flags(
        rows,
        asset_gap_threshold=float(settings["asset_pre_post_gap_threshold"]),
        asset_growth_multiple_threshold=float(settings["asset_growth_multiple_threshold"]),
        small_lag_assets_quantile=float(settings["small_lag_assets_quantile"]),
    )

    ratio_grid = [float(x) for x in settings["cfo_to_pat_ratio_grid"]]
    tolerance_grid = [float(x) for x in settings["reduction_tolerance_grid"]]
    trim_fraction = float(settings["trim_fraction"])

    records: list[dict[str, Any]] = []
    yearly_records: list[dict[str, Any]] = []
    contrast_records: list[dict[str, Any]] = []

    for (model, benchmark), specification in rows.groupby(
        ["model", "benchmark"], observed=True
    ):
        samples = {
            "all_finite": specification,
            "exclude_scale_scope_flags": specification[
                ~specification["flag_scale_or_scope"]
            ],
        }
        for sample_name, sample in samples.items():
            if sample.empty:
                continue
            for ratio in ratio_grid:
                dominant = sample["abs_delta_cfo"].ge(
                    ratio * sample["abs_delta_pat"]
                )
                group_map = {
                    "all": sample,
                    "cfo_dominant": sample[dominant],
                    "not_cfo_dominant": sample[~dominant],
                }
                for tolerance in tolerance_grid:
                    summaries: dict[str, dict[str, Any]] = {}
                    for dominance_group, subset in group_map.items():
                        if subset.empty:
                            continue
                        summary = _summarize_group(
                            subset,
                            tolerance=tolerance,
                            trim_fraction=trim_fraction,
                        )
                        summaries[dominance_group] = summary
                        records.append({
                            "model": model,
                            "benchmark": benchmark,
                            "sample": sample_name,
                            "cfo_to_pat_threshold": ratio,
                            "reduction_tolerance": tolerance,
                            "dominance_group": dominance_group,
                            **summary,
                        })

                    if "cfo_dominant" in summaries and "not_cfo_dominant" in summaries:
                        dom = summaries["cfo_dominant"]
                        non = summaries["not_cfo_dominant"]
                        total_sum = summaries.get("all", {}).get("sum_reduction", np.nan)
                        contrast_records.append({
                            "model": model,
                            "benchmark": benchmark,
                            "sample": sample_name,
                            "cfo_to_pat_threshold": ratio,
                            "reduction_tolerance": tolerance,
                            "frequency_tilt_excess_cfo_vs_non": (
                                dom["positive_minus_negative_share"]
                                - non["positive_minus_negative_share"]
                            ),
                            "mean_reduction_difference_cfo_vs_non": (
                                dom["mean_reduction"] - non["mean_reduction"]
                            ),
                            "cfo_dominant_share_of_net_reduction": (
                                dom["sum_reduction"] / total_sum
                                if np.isfinite(total_sum) and abs(total_sum) > 1e-12
                                else np.nan
                            ),
                            "non_cfo_dominant_net_reduction": non["sum_reduction"],
                            "non_cfo_dominant_mean_reduction": non["mean_reduction"],
                            "non_cfo_dominant_positive_minus_negative_share": non[
                                "positive_minus_negative_share"
                            ],
                        })

                for year, year_group in sample.groupby("fiscal_year", observed=True):
                    year_dominant = year_group["abs_delta_cfo"].ge(
                        ratio * year_group["abs_delta_pat"]
                    )
                    for dominance_group, subset in {
                        "all": year_group,
                        "cfo_dominant": year_group[year_dominant],
                        "not_cfo_dominant": year_group[~year_dominant],
                    }.items():
                        if subset.empty:
                            continue
                        for tolerance in tolerance_grid:
                            yearly_records.append({
                                "model": model,
                                "benchmark": benchmark,
                                "sample": sample_name,
                                "fiscal_year": int(year),
                                "cfo_to_pat_threshold": ratio,
                                "reduction_tolerance": tolerance,
                                "dominance_group": dominance_group,
                                **_summarize_group(
                                    subset,
                                    tolerance=tolerance,
                                    trim_fraction=trim_fraction,
                                ),
                            })

    primary_model = settings.get("primary_model", "modified_jones")
    primary_benchmark = settings.get("primary_benchmark", "audited_reference")
    primary = rows[
        rows["model"].eq(primary_model)
        & rows["benchmark"].eq(primary_benchmark)
    ].copy()
    candidate_threshold = float(settings.get("candidate_abs_cfo_threshold", .05))
    persistence_ratio = float(settings.get("persistence_cfo_to_pat_threshold", 5.0))
    primary["is_cfo_candidate"] = (
        primary["abs_delta_cfo"].ge(candidate_threshold)
        & primary["abs_delta_cfo"].ge(
            persistence_ratio * primary["abs_delta_pat"]
        )
    )
    persistence = (
        primary.groupby("issuer_ticker", observed=True)
        .agg(
            observed_years=("fiscal_year", "nunique"),
            candidate_years=("is_cfo_candidate", "sum"),
            candidate_share=("is_cfo_candidate", "mean"),
            median_abs_delta_cfo=("abs_delta_cfo", "median"),
            max_abs_delta_cfo=("abs_delta_cfo", "max"),
            mean_reduction=("reduction", "mean"),
        )
        .reset_index()
    )
    persistence = persistence[persistence["candidate_years"] > 0].copy()

    return {
        "cfo_tilt_summary": pd.DataFrame(records),
        "cfo_tilt_contrasts": pd.DataFrame(contrast_records),
        "cfo_tilt_by_year": pd.DataFrame(yearly_records),
        "cfo_candidate_persistence": persistence.sort_values(
            ["candidate_years", "max_abs_delta_cfo"], ascending=False
        ),
    }
