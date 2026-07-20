from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import trimmed_mean


CANDIDATE_LABEL = "identity_consistent_offsetting_reclassification_candidate"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _direct_target_primitives(cases: pd.DataFrame) -> pd.DataFrame:
    """Construct target inputs from the paired source statements only.

    These fields must not depend on whether an OLS baseline/decomposition fold
    exists for the fiscal year. Baseline-derived reduction and Shapley fields
    remain optional diagnostics.
    """
    out = cases.copy()
    lag = _numeric(out, "lag_assets_common")
    delta_pat = (_numeric(out, "pat_post") - _numeric(out, "pat_pre")) / lag
    delta_cfo_from_levels = (
        _numeric(out, "cfo_post") - _numeric(out, "cfo_pre")
    ) / lag
    delta_cfo_identity = _numeric(out, "delta_cfo_scaled")

    # Prefer the identity-panel delta, but retain an independent equality check.
    out["delta_pat_scaled"] = delta_pat
    out["delta_cfo_scaled_direct"] = delta_cfo_from_levels
    out["delta_cfo_direct_gap"] = delta_cfo_identity - delta_cfo_from_levels
    out["abs_delta_pat"] = delta_pat.abs()
    out["abs_delta_cfo"] = delta_cfo_identity.abs()
    out["cfo_to_pat_abs_ratio"] = out["abs_delta_cfo"] / np.maximum(
        out["abs_delta_pat"], 1e-12
    )
    out["component_dominance"] = np.select(
        [
            out["cfo_to_pat_abs_ratio"].ge(5.0),
            out["cfo_to_pat_abs_ratio"].le(0.2),
        ],
        ["cfo_dominant_5x", "pat_dominant_5x"],
        default="mixed",
    )

    out["decomposition_available"] = _numeric(out, "reduction").notna()
    direct_complete = (
        lag.notna()
        & lag.ne(0)
        & delta_pat.notna()
        & delta_cfo_identity.notna()
        & _numeric(out, "cfo_offset_closure_error_scaled").notna()
    )
    out["target_primitives_complete"] = direct_complete
    out["target_primitives_source"] = np.where(
        direct_complete, "PAIRED_STATEMENTS_DIRECT", "INCOMPLETE_DIRECT_INPUTS"
    )
    return out


def _classify_resolution(cases: pd.DataFrame, settings: dict[str, Any]) -> pd.Series:
    large_cfo = cases["abs_delta_cfo"].ge(
        float(settings["material_cfo_threshold"])
    )
    cfo_dominant = cases["cfo_to_pat_abs_ratio"].ge(
        float(settings["cfo_to_pat_ratio_threshold"])
    )
    offset_close = _numeric(cases, "cfo_offset_closure_error_scaled").abs().le(
        float(settings["offset_closure_scaled_tolerance"])
    )
    complete = cases["target_primitives_complete"]

    return pd.Series(
        np.select(
            [
                cases["identity_transition"].eq("insufficient_components"),
                cases["identity_transition"].eq("fail_to_pass"),
                cases["identity_transition"].eq("pass_to_fail"),
                cases["identity_transition"].eq("fail_to_fail"),
                cases["identity_transition"].eq("pass_to_pass")
                & complete
                & large_cfo
                & cfo_dominant
                & offset_close,
                cases["identity_transition"].eq("pass_to_pass") & complete,
            ],
            [
                "insufficient_cfs_components",
                "pre_internal_inconsistency_repaired",
                "audited_version_identity_failure",
                "persistent_internal_inconsistency",
                CANDIDATE_LABEL,
                "identity_consistent_other",
            ],
            default="incomplete_target_primitives",
        ),
        index=cases.index,
        dtype="object",
    )


def _resolution_table(cases: pd.DataFrame) -> pd.DataFrame:
    table = (
        cases.groupby(["fiscal_year", "cfs_resolution"], observed=True)
        .agg(
            rows=("issuer_ticker", "size"),
            mean_reduction=("reduction", "mean"),
            median_abs_delta_cfo=("abs_delta_cfo", "median"),
            median_abs_delta_pat=("abs_delta_pat", "median"),
            median_offset_ratio=("non_cfo_offset_to_cfo_ratio", "median"),
            share_offset_opposes_cfo=("non_cfo_offset_opposes_cfo", "mean"),
            decomposition_coverage=("decomposition_available", "mean"),
        )
        .reset_index()
    )
    table["share_within_year"] = table["rows"] / table.groupby(
        "fiscal_year"
    )["rows"].transform("sum")
    return table


def _magnitude_table(cases: pd.DataFrame, trim_fraction: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for year, group in cases.groupby("fiscal_year", observed=True):
        finite_cfo = _numeric(group, "abs_delta_cfo").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        finite_pat = _numeric(group, "abs_delta_pat").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        reduction = _numeric(group, "reduction")
        records.append(
            {
                "fiscal_year": int(year),
                "rows": len(group),
                "target_complete_rows": int(
                    group["target_primitives_complete"].sum()
                ),
                "decomposition_available_rows": int(
                    group["decomposition_available"].sum()
                ),
                "candidate_rows": int(group["cfs_resolution"].eq(CANDIDATE_LABEL).sum()),
                "median_abs_delta_cfo": float(finite_cfo.median()),
                "mean_abs_delta_cfo": float(finite_cfo.mean()),
                "trimmed_mean_abs_delta_cfo": trimmed_mean(
                    finite_cfo.to_numpy(float), trim_fraction
                ),
                "p75_abs_delta_cfo": float(finite_cfo.quantile(0.75)),
                "p90_abs_delta_cfo": float(finite_cfo.quantile(0.90)),
                "median_abs_delta_pat": float(finite_pat.median()),
                "share_abs_delta_cfo_gt_0_5pct": float(
                    group["abs_delta_cfo"].gt(0.005).mean()
                ),
                "share_abs_delta_cfo_gt_1pct": float(
                    group["abs_delta_cfo"].gt(0.01).mean()
                ),
                "share_abs_delta_cfo_gt_5pct": float(
                    group["abs_delta_cfo"].gt(0.05).mean()
                ),
                "share_cfo_dominant_5x": float(
                    group["cfo_to_pat_abs_ratio"].ge(5.0).mean()
                ),
                "mean_reduction": float(reduction.mean()),
                "trimmed_mean_reduction": trimmed_mean(
                    reduction.dropna().to_numpy(float), trim_fraction
                ),
                "share_positive_reduction": float(reduction.gt(0).mean()),
                "share_negative_reduction": float(reduction.lt(0).mean()),
                "positive_minus_negative_share": float(
                    reduction.gt(0).mean() - reduction.lt(0).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _coverage_table(cases: pd.DataFrame) -> pd.DataFrame:
    return (
        cases.groupby("fiscal_year", observed=True)
        .agg(
            rows=("issuer_ticker", "size"),
            target_complete_rows=("target_primitives_complete", "sum"),
            decomposition_available_rows=("decomposition_available", "sum"),
            maximum_abs_direct_cfo_gap=(
                "delta_cfo_direct_gap",
                lambda values: float(pd.to_numeric(values, errors="coerce").abs().max()),
            ),
            candidate_rows=(
                "cfs_resolution",
                lambda values: int(pd.Series(values).eq(CANDIDATE_LABEL).sum()),
            ),
        )
        .reset_index()
        .assign(
            target_complete_share=lambda frame: frame["target_complete_rows"]
            / frame["rows"],
            decomposition_available_share=lambda frame: frame[
                "decomposition_available_rows"
            ]
            / frame["rows"],
        )
    )


def finalize_cfs_target_tables(
    tables: dict[str, pd.DataFrame],
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Remove baseline-fold availability from CFS target construction."""
    if "cfs_identity_cases" not in tables:
        raise KeyError("cfs_identity_cases is required")

    output = dict(tables)
    cases = _direct_target_primitives(tables["cfs_identity_cases"])
    cases["cfs_resolution"] = _classify_resolution(cases, settings)

    output["cfs_identity_cases"] = cases
    output["cfs_candidate_resolution"] = _resolution_table(cases)
    output["cfo_magnitude_by_year"] = _magnitude_table(
        cases, float(settings.get("trim_fraction", 0.01))
    )
    output["cfs_target_input_coverage"] = _coverage_table(cases)
    return output
