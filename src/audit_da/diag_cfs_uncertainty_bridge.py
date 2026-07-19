from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import KEYS, trimmed_mean


CANDIDATE_LABEL = "identity_consistent_offsetting_reclassification_candidate"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _qcut_safe(values: pd.Series, bins: int) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").notna()
    if valid.sum() < 2:
        return output
    try:
        output.loc[valid] = pd.qcut(
            pd.to_numeric(values.loc[valid], errors="coerce"),
            q=min(bins, int(valid.sum())),
            labels=False,
            duplicates="drop",
        ).astype(float)
    except ValueError:
        output.loc[valid] = 0.0
    return output


def uncertainty_bridge_tables(
    identity_cases: pd.DataFrame,
    alignment_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    minimum_year = int(settings.get("minimum_year", 2018))
    candidate_label = settings.get("candidate_label", CANDIDATE_LABEL)
    trim = float(settings.get("trim_fraction", 0.01))

    alignment = alignment_cases.copy()
    if "component" in alignment:
        alignment = alignment[alignment["component"].eq("cfo")].copy()
    alignment = alignment[
        pd.to_numeric(alignment["fiscal_year"], errors="coerce").ge(minimum_year)
    ].copy()
    alignment["cfo_reduction_cashflow_anchor"] = _numeric(
        alignment, "cfo_reduction_cashflow_anchor"
    )
    alignment["da_pre"] = _numeric(alignment, "da_pre")

    records: list[dict[str, Any]] = []
    for keys, group in alignment.groupby(KEYS, observed=True):
        ticker, year = keys
        reduction = group["cfo_reduction_cashflow_anchor"].dropna()
        da = group["da_pre"].dropna()
        if reduction.empty:
            continue
        positive_share = float(reduction.gt(0).mean())
        negative_share = float(reduction.lt(0).mean())
        records.append({
            "issuer_ticker": ticker,
            "fiscal_year": int(year),
            "specifications": len(reduction),
            "cross_spec_mean_reduction": float(reduction.mean()),
            "cross_spec_trimmed_mean_reduction": trimmed_mean(
                reduction.to_numpy(float), trim
            ),
            "cross_spec_reduction_sd": float(reduction.std(ddof=0)),
            "cross_spec_reduction_range": float(reduction.max() - reduction.min()),
            "cross_spec_positive_share": positive_share,
            "cross_spec_negative_share": negative_share,
            "cross_spec_sign_consensus": max(positive_share, negative_share),
            "cross_spec_hard_disagreement": bool(
                reduction.gt(0).any() and reduction.lt(0).any()
            ),
            "cross_spec_da_pre_sd": float(da.std(ddof=0)) if len(da) else np.nan,
            "cross_spec_da_pre_range": float(da.max() - da.min()) if len(da) else np.nan,
        })
    firm_year = pd.DataFrame(records)

    identity_columns = [c for c in [
        "issuer_ticker", "fiscal_year", "cfs_resolution", "reduction",
        "delta_cfo_scaled", "abs_delta_cfo", "abs_delta_pat",
        "non_cfo_offset_to_cfo_ratio", "cfo_offset_closure_error_scaled",
    ] if c in identity_cases]
    identity = identity_cases[identity_columns].copy()
    identity = identity[
        pd.to_numeric(identity["fiscal_year"], errors="coerce").ge(minimum_year)
    ]
    merged = firm_year.merge(
        identity,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    merged["candidate"] = merged["cfs_resolution"].eq(candidate_label)

    merged["uncertainty_quartile_global"] = _qcut_safe(
        merged["cross_spec_reduction_sd"],
        int(settings.get("uncertainty_bins", 4)),
    )
    merged["uncertainty_quartile_within_year"] = (
        merged.groupby("fiscal_year", group_keys=False)["cross_spec_reduction_sd"]
        .apply(lambda x: _qcut_safe(x, int(settings.get("uncertainty_bins", 4))))
    )

    yearly_records: list[dict[str, Any]] = []
    for (year, candidate), group in merged.groupby(
        ["fiscal_year", "candidate"], observed=True
    ):
        yearly_records.append({
            "fiscal_year": int(year),
            "candidate": bool(candidate),
            "rows": len(group),
            "mean_primary_reduction": float(_numeric(group, "reduction").mean()),
            "mean_cross_spec_reduction": float(group["cross_spec_mean_reduction"].mean()),
            "median_cross_spec_reduction_sd": float(group["cross_spec_reduction_sd"].median()),
            "mean_cross_spec_sign_consensus": float(group["cross_spec_sign_consensus"].mean()),
            "hard_disagreement_share": float(group["cross_spec_hard_disagreement"].mean()),
            "mean_abs_delta_cfo": float(_numeric(group, "delta_cfo_scaled").abs().mean()),
        })

    bin_records: list[dict[str, Any]] = []
    candidates = merged[merged["candidate"]].copy()
    for bin_value, group in candidates.groupby(
        "uncertainty_quartile_global", observed=True
    ):
        reduction = _numeric(group, "reduction")
        bin_records.append({
            "uncertainty_quartile_global": int(bin_value),
            "rows": len(group),
            "mean_primary_reduction": float(reduction.mean()),
            "trimmed_mean_primary_reduction": trimmed_mean(
                reduction.dropna().to_numpy(float), trim
            ),
            "positive_minus_negative_share": float(
                reduction.gt(0).mean() - reduction.lt(0).mean()
            ),
            "mean_cross_spec_reduction": float(group["cross_spec_mean_reduction"].mean()),
            "median_cross_spec_reduction_sd": float(group["cross_spec_reduction_sd"].median()),
            "hard_disagreement_share": float(group["cross_spec_hard_disagreement"].mean()),
            "mean_abs_delta_cfo": float(_numeric(group, "delta_cfo_scaled").abs().mean()),
        })

    correlation_records: list[dict[str, Any]] = []
    for year, group in candidates.groupby("fiscal_year", observed=True):
        finite = group[[
            "reduction", "cross_spec_reduction_sd", "cross_spec_sign_consensus",
            "abs_delta_cfo",
        ]].replace([np.inf, -np.inf], np.nan).dropna()
        correlation_records.append({
            "fiscal_year": int(year),
            "rows": len(finite),
            "corr_payoff_with_uncertainty_sd": float(
                finite["reduction"].corr(finite["cross_spec_reduction_sd"])
            ) if len(finite) > 2 else np.nan,
            "corr_payoff_with_sign_consensus": float(
                finite["reduction"].corr(finite["cross_spec_sign_consensus"])
            ) if len(finite) > 2 else np.nan,
            "corr_payoff_with_abs_delta_cfo": float(
                finite["reduction"].corr(finite["abs_delta_cfo"])
            ) if len(finite) > 2 else np.nan,
        })

    return {
        "cfs_measurement_uncertainty_cases": merged,
        "cfs_payoff_uncertainty_by_year": pd.DataFrame(yearly_records),
        "cfs_payoff_by_uncertainty_bin": pd.DataFrame(bin_records),
        "cfs_payoff_uncertainty_correlations": pd.DataFrame(correlation_records),
    }
