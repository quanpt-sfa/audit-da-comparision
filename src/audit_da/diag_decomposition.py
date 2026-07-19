from __future__ import annotations

from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd

from .diag_common import KEYS, paired_panel, trimmed_mean


COMPONENTS = ("pat", "cfo", "benchmark")


def _shapley_absolute_reduction(da_pre: np.ndarray, movements: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(da_pre)
    contributions = {name: np.zeros(n, dtype=float) for name in movements}
    orders = list(permutations(movements))
    for order in orders:
        current = da_pre.copy()
        current_abs = np.abs(current)
        for name in order:
            next_state = current + movements[name]
            gain = current_abs - np.abs(next_state)
            contributions[name] += gain / len(orders)
            current = next_state
            current_abs = np.abs(current)
    return contributions


def build_decomposition_panel(baseline: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    pair = paired_panel(panel)
    needed = ["pat_pre", "pat_post", "cfo_pre", "cfo_post", "lag_assets_pre", "lag_assets_post"]
    missing = [c for c in needed if c not in pair]
    if missing:
        raise ValueError(f"Panel missing decomposition columns: {missing}")
    lag = pd.to_numeric(pair["lag_assets_pre"], errors="coerce")
    pair["lag_assets_common"] = lag
    pair["delta_pat_scaled"] = (
        pd.to_numeric(pair["pat_post"], errors="coerce") - pd.to_numeric(pair["pat_pre"], errors="coerce")
    ) / lag
    pair["delta_cfo_scaled"] = (
        pd.to_numeric(pair["cfo_post"], errors="coerce") - pd.to_numeric(pair["cfo_pre"], errors="coerce")
    ) / lag
    pair["delta_ta_from_components"] = pair["delta_pat_scaled"] - pair["delta_cfo_scaled"]
    if {"ta_scaled_pre", "ta_scaled_post"}.issubset(pair):
        pair["delta_ta_panel"] = pd.to_numeric(pair["ta_scaled_post"], errors="coerce") - pd.to_numeric(pair["ta_scaled_pre"], errors="coerce")
        pair["delta_ta_identity_error"] = pair["delta_ta_panel"] - pair["delta_ta_from_components"]
    if {"assets_pre", "assets_post"}.issubset(pair):
        a_pre = pd.to_numeric(pair["assets_pre"], errors="coerce")
        a_post = pd.to_numeric(pair["assets_post"], errors="coerce")
        pair["asset_pre_post_gap"] = (a_post - a_pre).abs() / np.maximum(a_post.abs(), 1.0)
        pair["asset_growth_multiple"] = a_post.abs() / np.maximum(lag.abs(), 1.0)
    if {"cash_pre", "cash_post"}.issubset(pair):
        pair["delta_cash_scaled"] = (
            pd.to_numeric(pair["cash_post"], errors="coerce") - pd.to_numeric(pair["cash_pre"], errors="coerce")
        ) / lag
    if {"revenue_pre", "revenue_post"}.issubset(pair):
        pair["cfo_pre_to_revenue"] = pd.to_numeric(pair["cfo_pre"], errors="coerce").abs() / np.maximum(
            pd.to_numeric(pair["revenue_pre"], errors="coerce").abs(), 1.0
        )
    merged = baseline.merge(pair, on=KEYS, how="left", validate="many_to_one")
    finite = np.isfinite(pd.to_numeric(merged["da_pre"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(merged["signed_shift"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(merged["raw_ta_shift"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(merged["delta_pat_scaled"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(merged["delta_cfo_scaled"], errors="coerce"))
    merged = merged.loc[finite].copy()
    merged["pat_movement"] = merged["delta_pat_scaled"]
    merged["cfo_movement"] = -merged["delta_cfo_scaled"]
    merged["benchmark_movement"] = merged["signed_shift"] - merged["raw_ta_shift"]
    movements = {name: merged[f"{name}_movement"].to_numpy(float) for name in COMPONENTS}
    shapley = _shapley_absolute_reduction(merged["da_pre"].to_numpy(float), movements)
    for name, values in shapley.items():
        merged[f"shapley_{name}"] = values
    merged["shapley_sum"] = sum(merged[f"shapley_{name}"] for name in COMPONENTS)
    merged["shapley_identity_error"] = merged["reduction"] - merged["shapley_sum"]
    merged["abs_delta_pat"] = merged["delta_pat_scaled"].abs()
    merged["abs_delta_cfo"] = merged["delta_cfo_scaled"].abs()
    merged["cfo_to_pat_abs_ratio"] = merged["abs_delta_cfo"] / np.maximum(merged["abs_delta_pat"], 1e-12)
    merged["component_dominance"] = np.select(
        [merged["cfo_to_pat_abs_ratio"] >= 5.0, merged["cfo_to_pat_abs_ratio"] <= 0.2],
        ["cfo_dominant_5x", "pat_dominant_5x"],
        default="mixed",
    )
    absolute_shapley = pd.DataFrame({name: merged[f"shapley_{name}"].abs() for name in COMPONENTS})
    merged["largest_shapley_component"] = absolute_shapley.idxmax(axis=1)
    return merged


def decomposition_tables(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
    cfo_dominance_grid: Iterable[float],
    materiality_grid: Iterable[float],
    trim_fraction: float,
    asset_gap_threshold: float,
    asset_growth_multiple_threshold: float,
    small_lag_assets_quantile: float,
    repeated_candidate_min_years: int,
) -> dict[str, pd.DataFrame]:
    rows = build_decomposition_panel(baseline, panel)
    lag_cut = float(rows["lag_assets_common"].abs().quantile(small_lag_assets_quantile))
    rows["flag_small_lag_assets"] = rows["lag_assets_common"].abs() <= lag_cut
    rows["flag_asset_pre_post_gap"] = rows.get("asset_pre_post_gap", pd.Series(np.nan, index=rows.index)).gt(asset_gap_threshold)
    rows["flag_asset_growth"] = rows.get("asset_growth_multiple", pd.Series(np.nan, index=rows.index)).gt(asset_growth_multiple_threshold)
    rows["flag_scale_or_scope"] = rows[["flag_small_lag_assets", "flag_asset_pre_post_gap", "flag_asset_growth"]].any(axis=1)

    summary_records: list[dict] = []
    yearly_records: list[dict] = []
    dominance_records: list[dict] = []
    for (model, benchmark), group in rows.groupby(["model", "benchmark"], observed=True):
        for sample, subset in {
            "all_finite": group,
            "exclude_scale_scope_flags": group[~group["flag_scale_or_scope"]],
            "cashflow_both": group[(group.get("ta_source_pre") == "cash_flow") & (group.get("ta_source_post") == "cash_flow")],
        }.items():
            if subset.empty:
                continue
            summary_records.append({
                "model": model,
                "benchmark": benchmark,
                "sample": sample,
                "rows": len(subset),
                "mean_reduction": float(subset["reduction"].mean()),
                "trimmed_mean_reduction": trimmed_mean(subset["reduction"].to_numpy(float), trim_fraction),
                "mean_shapley_pat": float(subset["shapley_pat"].mean()),
                "mean_shapley_cfo": float(subset["shapley_cfo"].mean()),
                "mean_shapley_benchmark": float(subset["shapley_benchmark"].mean()),
                "median_abs_delta_pat": float(subset["abs_delta_pat"].median()),
                "median_abs_delta_cfo": float(subset["abs_delta_cfo"].median()),
                "share_cfo_dominant_5x": float(subset["component_dominance"].eq("cfo_dominant_5x").mean()),
                "share_pat_dominant_5x": float(subset["component_dominance"].eq("pat_dominant_5x").mean()),
                "share_largest_shapley_pat": float(subset["largest_shapley_component"].eq("pat").mean()),
                "share_largest_shapley_cfo": float(subset["largest_shapley_component"].eq("cfo").mean()),
                "share_largest_shapley_benchmark": float(subset["largest_shapley_component"].eq("benchmark").mean()),
                "max_abs_shapley_identity_error": float(subset["shapley_identity_error"].abs().max()),
            })
        for year, subset in group.groupby("fiscal_year", observed=True):
            body = subset[~subset["flag_scale_or_scope"]]
            yearly_records.append({
                "model": model,
                "benchmark": benchmark,
                "fiscal_year": year,
                "rows": len(subset),
                "mean_reduction": float(subset["reduction"].mean()),
                "trimmed_mean_reduction": trimmed_mean(subset["reduction"].to_numpy(float), trim_fraction),
                "body_rows": len(body),
                "body_mean_reduction": float(body["reduction"].mean()) if len(body) else np.nan,
                "body_trimmed_mean_reduction": trimmed_mean(body["reduction"].to_numpy(float), trim_fraction) if len(body) else np.nan,
                "body_mean_shapley_pat": float(body["shapley_pat"].mean()) if len(body) else np.nan,
                "body_mean_shapley_cfo": float(body["shapley_cfo"].mean()) if len(body) else np.nan,
                "share_flag_scale_scope": float(subset["flag_scale_or_scope"].mean()),
            })
        for ratio in cfo_dominance_grid:
            dominance_records.append({
                "model": model,
                "benchmark": benchmark,
                "cfo_to_pat_threshold": float(ratio),
                "share_cfo_dominant": float((group["abs_delta_cfo"] > ratio * group["abs_delta_pat"]).mean()),
                "mean_reduction_cfo_dominant": float(group.loc[group["abs_delta_cfo"] > ratio * group["abs_delta_pat"], "reduction"].mean()),
                "mean_reduction_not_cfo_dominant": float(group.loc[~(group["abs_delta_cfo"] > ratio * group["abs_delta_pat"]), "reduction"].mean()),
            })
        for delta in materiality_grid:
            material = group[group["reduction"].abs() > delta]
            dominance_records.append({
                "model": model,
                "benchmark": benchmark,
                "cfo_to_pat_threshold": np.nan,
                "reduction_materiality": float(delta),
                "share_material": float(len(material) / len(group)),
                "share_cfo_dominant_5x_material": float(material["component_dominance"].eq("cfo_dominant_5x").mean()) if len(material) else np.nan,
            })

    primary = rows[(rows["model"] == "modified_jones") & (rows["benchmark"] == "audited_reference")].copy()
    issuer_repeats = primary.groupby("issuer_ticker", observed=True).agg(
        years=("fiscal_year", "nunique"),
        rows=("fiscal_year", "size"),
        max_abs_delta_cfo=("abs_delta_cfo", "max"),
        median_cfo_to_pat_ratio=("cfo_to_pat_abs_ratio", "median"),
        share_cfo_dominant_5x=("component_dominance", lambda x: float(pd.Series(x).eq("cfo_dominant_5x").mean())),
        max_abs_reduction=("reduction", lambda x: float(pd.Series(x).abs().max())),
    ).reset_index()
    candidates = primary[
        primary["component_dominance"].eq("cfo_dominant_5x")
        & (primary["abs_delta_cfo"] >= 0.05)
    ].merge(
        issuer_repeats[issuer_repeats["years"] >= repeated_candidate_min_years][["issuer_ticker", "years"]],
        on="issuer_ticker", how="inner", suffixes=("", "_candidate"),
    )
    candidate_columns = [c for c in [
        "issuer_ticker", "fiscal_year", "years", "da_pre", "da_post", "reduction", "raw_ta_shift",
        "delta_pat_scaled", "delta_cfo_scaled", "cfo_to_pat_abs_ratio", "shapley_pat", "shapley_cfo",
        "lag_assets_common", "asset_pre_post_gap", "asset_growth_multiple", "delta_cash_scaled",
        "cfo_pre_to_revenue", "ta_source_pre", "ta_source_post", "flag_scale_or_scope",
    ] if c in candidates]

    return {
        "ta_component_decomposition_cases": rows,
        "ta_component_decomposition_summary": pd.DataFrame(summary_records),
        "ta_component_decomposition_by_year": pd.DataFrame(yearly_records),
        "ta_component_dominance_grid": pd.DataFrame(dominance_records),
        "repeated_cfo_dominant_issuers": issuer_repeats.sort_values(["years", "max_abs_delta_cfo"], ascending=False),
        "cfs_manual_review_candidates": candidates[candidate_columns].sort_values(["issuer_ticker", "fiscal_year"]),
        "sample_filter_diagnostics": rows.groupby(["model", "benchmark"], observed=True).agg(
            rows=("issuer_ticker", "size"),
            share_small_lag_assets=("flag_small_lag_assets", "mean"),
            share_asset_pre_post_gap=("flag_asset_pre_post_gap", "mean"),
            share_asset_growth=("flag_asset_growth", "mean"),
            share_any_scale_scope=("flag_scale_or_scope", "mean"),
        ).reset_index(),
    }
