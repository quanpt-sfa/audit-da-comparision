from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import KEYS, paired_panel, trimmed_mean
from .diag_decomposition import build_decomposition_panel


CFS_COMPONENTS = (
    "cfo",
    "cfi",
    "cff",
    "net_cash_change",
    "cash_begin_cfs",
    "fx_effect",
    "cash_end_cfs",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _pass_with_tolerance(
    residual: pd.Series,
    lag_assets: pd.Series,
    absolute_tolerance_vnd: float,
    scaled_tolerance: float,
) -> pd.Series:
    threshold = np.maximum(
        float(absolute_tolerance_vnd),
        lag_assets.abs() * float(scaled_tolerance),
    )
    return residual.abs().le(threshold)


def build_cfs_identity_panel(
    panel: pd.DataFrame,
    absolute_tolerance_vnd: float,
    scaled_tolerance: float,
) -> pd.DataFrame:
    pair = paired_panel(panel)
    lag = _numeric(pair, "lag_assets_pre")
    pair["lag_assets_common"] = lag

    for version in ("pre", "post"):
        values = {
            name: _numeric(pair, f"{name}_{version}")
            for name in CFS_COMPONENTS
        }
        # FX effects are frequently omitted when zero. Retain a flag and use
        # zero only for the roll-forward identity.
        fx_missing = values["fx_effect"].isna()
        fx_filled = values["fx_effect"].fillna(0.0)

        required_section = (
            values["cfo"].notna()
            & values["cfi"].notna()
            & values["cff"].notna()
            & values["net_cash_change"].notna()
        )
        required_rollforward = (
            values["net_cash_change"].notna()
            & values["cash_begin_cfs"].notna()
            & values["cash_end_cfs"].notna()
        )
        required_full = (
            values["cfo"].notna()
            & values["cfi"].notna()
            & values["cff"].notna()
            & values["cash_begin_cfs"].notna()
            & values["cash_end_cfs"].notna()
        )

        section_sum = values["cfo"] + values["cfi"] + values["cff"]
        cash_change = values["cash_end_cfs"] - values["cash_begin_cfs"]
        section_residual = section_sum - values["net_cash_change"]
        rollforward_residual = values["net_cash_change"] + fx_filled - cash_change
        full_residual = section_sum + fx_filled - cash_change

        pair[f"cfs_fx_missing_assumed_zero_{version}"] = fx_missing
        pair[f"cfs_section_available_{version}"] = required_section
        pair[f"cfs_rollforward_available_{version}"] = required_rollforward
        pair[f"cfs_full_available_{version}"] = required_full
        pair[f"cfs_section_sum_{version}"] = section_sum
        pair[f"cfs_cash_change_{version}"] = cash_change
        pair[f"cfs_section_residual_{version}"] = section_residual
        pair[f"cfs_rollforward_residual_{version}"] = rollforward_residual
        pair[f"cfs_full_residual_{version}"] = full_residual
        pair[f"cfs_section_residual_scaled_{version}"] = section_residual / lag
        pair[f"cfs_rollforward_residual_scaled_{version}"] = rollforward_residual / lag
        pair[f"cfs_full_residual_scaled_{version}"] = full_residual / lag

        pair[f"cfs_section_pass_{version}"] = (
            required_section
            & _pass_with_tolerance(
                section_residual, lag, absolute_tolerance_vnd, scaled_tolerance
            )
        )
        pair[f"cfs_rollforward_pass_{version}"] = (
            required_rollforward
            & _pass_with_tolerance(
                rollforward_residual, lag, absolute_tolerance_vnd, scaled_tolerance
            )
        )
        pair[f"cfs_full_pass_{version}"] = (
            required_full
            & _pass_with_tolerance(
                full_residual, lag, absolute_tolerance_vnd, scaled_tolerance
            )
        )

        if f"cash_{version}" in pair:
            balance_cash = _numeric(pair, f"cash_{version}")
            pair[f"cfs_end_to_balance_cash_gap_{version}"] = (
                values["cash_end_cfs"] - balance_cash
            )
            pair[f"cfs_end_to_balance_cash_gap_scaled_{version}"] = (
                values["cash_end_cfs"] - balance_cash
            ) / lag

    for name in ("cfo", "cfi", "cff", "net_cash_change", "fx_effect"):
        pair[f"delta_{name}_scaled"] = (
            _numeric(pair, f"{name}_post") - _numeric(pair, f"{name}_pre")
        ) / lag

    pair["delta_cfs_cash_change_scaled"] = (
        pair["cfs_cash_change_post"] - pair["cfs_cash_change_pre"]
    ) / lag
    pair["non_cfo_offset_scaled"] = (
        pair["delta_cfi_scaled"]
        + pair["delta_cff_scaled"]
        + pair["delta_fx_effect_scaled"].fillna(0.0)
        - pair["delta_cfs_cash_change_scaled"]
    )
    pair["cfo_offset_closure_error_scaled"] = (
        pair["delta_cfo_scaled"] + pair["non_cfo_offset_scaled"]
    )
    pair["non_cfo_offset_to_cfo_ratio"] = (
        pair["non_cfo_offset_scaled"].abs()
        / np.maximum(pair["delta_cfo_scaled"].abs(), 1e-12)
    )
    pair["non_cfo_offset_opposes_cfo"] = (
        np.sign(pair["non_cfo_offset_scaled"])
        == -np.sign(pair["delta_cfo_scaled"])
    )
    pair["identity_transition"] = np.select(
        [
            pair["cfs_full_pass_pre"] & pair["cfs_full_pass_post"],
            (~pair["cfs_full_pass_pre"])
            & pair["cfs_full_pass_post"]
            & pair["cfs_full_available_pre"],
            pair["cfs_full_pass_pre"]
            & (~pair["cfs_full_pass_post"])
            & pair["cfs_full_available_post"],
            pair["cfs_full_available_pre"] & pair["cfs_full_available_post"],
        ],
        [
            "pass_to_pass",
            "fail_to_pass",
            "pass_to_fail",
            "fail_to_fail",
        ],
        default="insufficient_components",
    )
    pair["ticker_numeric_only"] = pair["issuer_ticker"].astype(str).str.fullmatch(r"\d+")
    pair["ticker_format_valid"] = pair["issuer_ticker"].astype(str).str.fullmatch(
        r"[A-Z][A-Z0-9]{1,7}"
    )
    return pair


def cfs_identity_tables(
    panel: pd.DataFrame,
    baseline: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    identity = build_cfs_identity_panel(
        panel,
        absolute_tolerance_vnd=float(settings["absolute_tolerance_vnd"]),
        scaled_tolerance=float(settings["scaled_tolerance"]),
    )

    primary_model = settings["primary_model"]
    primary_benchmark = settings["primary_benchmark"]
    decomposition = build_decomposition_panel(baseline, panel)
    primary = decomposition[
        decomposition["model"].eq(primary_model)
        & decomposition["benchmark"].eq(primary_benchmark)
    ].copy()
    columns = [
        c
        for c in [
            "issuer_ticker",
            "fiscal_year",
            "da_pre",
            "da_post",
            "reduction",
            "delta_pat_scaled",
            "delta_cfo_scaled",
            "abs_delta_pat",
            "abs_delta_cfo",
            "cfo_to_pat_abs_ratio",
            "component_dominance",
            "shapley_pat",
            "shapley_cfo",
            "shapley_benchmark",
            "lag_assets_common",
            "asset_pre_post_gap",
            "asset_growth_multiple",
        ]
        if c in primary
    ]
    cases = identity.merge(
        primary[columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_decomposition"),
    )

    large_cfo = cases["abs_delta_cfo"].ge(float(settings["material_cfo_threshold"]))
    cfo_dominant = cases["cfo_to_pat_abs_ratio"].ge(
        float(settings["cfo_to_pat_ratio_threshold"])
    )
    offset_close = cases["cfo_offset_closure_error_scaled"].abs().le(
        float(settings["offset_closure_scaled_tolerance"])
    )

    cases["cfs_resolution"] = np.select(
        [
            cases["identity_transition"].eq("insufficient_components"),
            cases["identity_transition"].eq("fail_to_pass"),
            cases["identity_transition"].eq("pass_to_fail"),
            cases["identity_transition"].eq("fail_to_fail"),
            cases["identity_transition"].eq("pass_to_pass")
            & large_cfo
            & cfo_dominant
            & offset_close,
            cases["identity_transition"].eq("pass_to_pass"),
        ],
        [
            "insufficient_cfs_components",
            "pre_internal_inconsistency_repaired",
            "audited_version_identity_failure",
            "persistent_internal_inconsistency",
            "identity_consistent_offsetting_reclassification_candidate",
            "identity_consistent_other",
        ],
        default="unclassified",
    )

    long_records: list[dict[str, Any]] = []
    for version in ("pre", "post"):
        for year, group in cases.groupby("fiscal_year", observed=True):
            available = group[f"cfs_full_available_{version}"]
            long_records.append({
                "version": version,
                "fiscal_year": int(year),
                "rows": len(group),
                "available_rows": int(available.sum()),
                "availability_share": float(available.mean()),
                "full_identity_pass_share_all": float(
                    group[f"cfs_full_pass_{version}"].mean()
                ),
                "full_identity_pass_share_available": float(
                    group.loc[available, f"cfs_full_pass_{version}"].mean()
                ) if available.any() else np.nan,
                "section_pass_share_available": float(
                    group.loc[
                        group[f"cfs_section_available_{version}"],
                        f"cfs_section_pass_{version}",
                    ].mean()
                ) if group[f"cfs_section_available_{version}"].any() else np.nan,
                "rollforward_pass_share_available": float(
                    group.loc[
                        group[f"cfs_rollforward_available_{version}"],
                        f"cfs_rollforward_pass_{version}",
                    ].mean()
                ) if group[f"cfs_rollforward_available_{version}"].any() else np.nan,
                "median_abs_full_residual_scaled": float(
                    group.loc[available, f"cfs_full_residual_scaled_{version}"].abs().median()
                ) if available.any() else np.nan,
                "p95_abs_full_residual_scaled": float(
                    group.loc[available, f"cfs_full_residual_scaled_{version}"].abs().quantile(.95)
                ) if available.any() else np.nan,
                "fx_missing_assumed_zero_share": float(
                    group[f"cfs_fx_missing_assumed_zero_{version}"].mean()
                ),
            })

    transition = (
        cases.groupby(["fiscal_year", "identity_transition"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    transition["share_within_year"] = transition["rows"] / transition.groupby(
        "fiscal_year"
    )["rows"].transform("sum")

    resolution = (
        cases.groupby(["fiscal_year", "cfs_resolution"], observed=True)
        .agg(
            rows=("issuer_ticker", "size"),
            mean_reduction=("reduction", "mean"),
            median_abs_delta_cfo=("abs_delta_cfo", "median"),
            median_abs_delta_pat=("abs_delta_pat", "median"),
            median_offset_ratio=("non_cfo_offset_to_cfo_ratio", "median"),
            share_offset_opposes_cfo=("non_cfo_offset_opposes_cfo", "mean"),
        )
        .reset_index()
    )
    resolution["share_within_year"] = resolution["rows"] / resolution.groupby(
        "fiscal_year"
    )["rows"].transform("sum")

    yearly: list[dict[str, Any]] = []
    trim_fraction = float(settings.get("trim_fraction", 0.01))
    for year, group in cases.groupby("fiscal_year", observed=True):
        finite_cfo = group["abs_delta_cfo"].replace([np.inf, -np.inf], np.nan).dropna()
        finite_pat = group["abs_delta_pat"].replace([np.inf, -np.inf], np.nan).dropna()
        yearly.append({
            "fiscal_year": int(year),
            "rows": len(group),
            "median_abs_delta_cfo": float(finite_cfo.median()),
            "mean_abs_delta_cfo": float(finite_cfo.mean()),
            "trimmed_mean_abs_delta_cfo": trimmed_mean(
                finite_cfo.to_numpy(float), trim_fraction
            ),
            "p75_abs_delta_cfo": float(finite_cfo.quantile(.75)),
            "p90_abs_delta_cfo": float(finite_cfo.quantile(.90)),
            "median_abs_delta_pat": float(finite_pat.median()),
            "share_abs_delta_cfo_gt_0_5pct": float(group["abs_delta_cfo"].gt(.005).mean()),
            "share_abs_delta_cfo_gt_1pct": float(group["abs_delta_cfo"].gt(.01).mean()),
            "share_abs_delta_cfo_gt_5pct": float(group["abs_delta_cfo"].gt(.05).mean()),
            "share_cfo_dominant_5x": float(group["cfo_to_pat_abs_ratio"].ge(5.0).mean()),
            "mean_reduction": float(group["reduction"].mean()),
            "trimmed_mean_reduction": trimmed_mean(
                group["reduction"].to_numpy(float), trim_fraction
            ),
            "share_positive_reduction": float(group["reduction"].gt(0).mean()),
            "share_negative_reduction": float(group["reduction"].lt(0).mean()),
            "positive_minus_negative_share": float(
                group["reduction"].gt(0).mean() - group["reduction"].lt(0).mean()
            ),
        })

    invalid_tickers = cases[
        cases["ticker_numeric_only"] | (~cases["ticker_format_valid"])
    ].copy()

    return {
        "cfs_identity_cases": cases,
        "cfs_identity_by_year": pd.DataFrame(long_records),
        "cfs_identity_transitions": transition,
        "cfs_candidate_resolution": resolution,
        "cfo_magnitude_by_year": pd.DataFrame(yearly),
        "invalid_ticker_cases": invalid_tickers,
    }
