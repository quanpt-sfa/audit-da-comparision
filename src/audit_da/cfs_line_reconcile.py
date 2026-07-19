from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cfs_item_map import pair_line_items
from .diag_common import KEYS


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_median(values: pd.Series, absolute: bool = False) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return np.nan
    if absolute:
        numeric = np.abs(numeric)
    return float(np.median(numeric))


def _aggregate_contributors(
    contribution: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if contribution.empty:
        return pd.DataFrame()
    return (
        contribution.groupby(
            group_columns + ["concept"],
            dropna=False,
            observed=True,
        )
        .agg(
            rows=("issuer_ticker", "size"),
            issuers=("issuer_ticker", "nunique"),
            total_absolute_change=("absolute_line_item_change_scaled", "sum"),
            median_absolute_change=("absolute_line_item_change_scaled", "median"),
            mean_signed_change=("line_item_change_scaled", "mean"),
        )
        .reset_index()
        .sort_values("total_absolute_change", ascending=False)
    )


def line_item_reconciliation(
    line_item_panel: pd.DataFrame,
    observed_cases: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    paired = pair_line_items(line_item_panel, settings)
    empty = {
        "cfs_line_item_reconciliation_cases": pd.DataFrame(),
        "cfs_line_item_reconciliation_summary": pd.DataFrame(),
        "cfs_line_item_top_contributors": pd.DataFrame(),
        "cfs_line_item_top_contributors_all": pd.DataFrame(),
    }
    if paired.empty:
        return empty

    audited = (
        panel[
            panel["audit_status"].eq(settings.get("audited_label", "audited"))
        ][KEYS + ["lag_assets"]]
        .drop_duplicates(KEYS)
    )
    cases = (
        observed_cases.drop_duplicates(KEYS)
        .merge(paired, on=KEYS, how="left", validate="one_to_one")
        .merge(audited, on=KEYS, how="left", validate="one_to_one")
    )
    sections = settings.get("concept_sections", {})
    scale = _numeric(cases, "lag_assets")
    for concept in sections:
        cases[f"delta_{concept}_scaled"] = (
            _numeric(cases, f"delta_{concept}") / scale
        )

    rows: list[dict[str, Any]] = []
    contributors: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        for section, aggregate_column in [
            ("investing", "delta_cfi_scaled"),
            ("financing", "delta_cff_scaled"),
        ]:
            concepts = [
                concept
                for concept, mapped_section in sections.items()
                if mapped_section == section
            ]
            values = {
                concept: row.get(f"delta_{concept}_scaled", np.nan)
                for concept in concepts
            }
            values = {
                concept: float(value)
                for concept, value in values.items()
                if np.isfinite(value)
            }
            mapped = float(sum(values.values())) if values else np.nan
            aggregate = pd.to_numeric(
                pd.Series([row.get(aggregate_column)]), errors="coerce"
            ).iloc[0]
            dominant = (
                max(values, key=lambda concept: abs(values[concept]))
                if values
                else None
            )
            residual = (
                mapped - aggregate
                if np.isfinite(mapped) and np.isfinite(aggregate)
                else np.nan
            )
            mapped_share = (
                mapped / aggregate
                if np.isfinite(mapped)
                and np.isfinite(aggregate)
                and abs(aggregate) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "issuer_ticker": row["issuer_ticker"],
                    "fiscal_year": row["fiscal_year"],
                    "section": section,
                    "cfs_resolution": row.get("cfs_resolution"),
                    "offset_channel_pattern": row.get("offset_channel_pattern"),
                    "cfo_adjustment_direction": row.get("cfo_adjustment_direction"),
                    "aggregate_section_change_scaled": aggregate,
                    "mapped_line_change_sum_scaled": mapped,
                    "reconciliation_residual_scaled": residual,
                    "mapped_share_of_aggregate": mapped_share,
                    "mapped_share_within_80_120pct": bool(
                        np.isfinite(mapped_share)
                        and 0.80 <= mapped_share <= 1.20
                    ),
                    "mapped_concepts_available": len(values),
                    "dominant_line_item": dominant,
                    "dominant_line_item_change_scaled": (
                        values.get(dominant, np.nan) if dominant else np.nan
                    ),
                }
            )
            contributors.extend(
                {
                    "issuer_ticker": row["issuer_ticker"],
                    "fiscal_year": row["fiscal_year"],
                    "section": section,
                    "cfs_resolution": row.get("cfs_resolution"),
                    "concept": concept,
                    "line_item_change_scaled": value,
                    "absolute_line_item_change_scaled": abs(value),
                    "offset_channel_pattern": row.get("offset_channel_pattern"),
                    "cfo_adjustment_direction": row.get("cfo_adjustment_direction"),
                }
                for concept, value in values.items()
            )

    reconciliation = pd.DataFrame(rows)
    contribution = pd.DataFrame(contributors)
    summary = (
        reconciliation.groupby(
            ["section", "cfs_resolution", "offset_channel_pattern"],
            dropna=False,
            observed=True,
        )
        .agg(
            rows=("issuer_ticker", "size"),
            issuers=("issuer_ticker", "nunique"),
            median_mapped_concepts=("mapped_concepts_available", "median"),
            median_abs_reconciliation_residual=(
                "reconciliation_residual_scaled",
                lambda x: _finite_median(x, absolute=True),
            ),
            median_mapped_share=(
                "mapped_share_of_aggregate",
                lambda x: _finite_median(x),
            ),
            share_mapped_within_80_120pct=(
                "mapped_share_within_80_120pct",
                "mean",
            ),
        )
        .reset_index()
    )

    group_columns = [
        "cfs_resolution",
        "section",
        "offset_channel_pattern",
        "cfo_adjustment_direction",
    ]
    top_all = _aggregate_contributors(contribution, group_columns)
    candidate_label = settings.get(
        "candidate_label",
        "identity_consistent_offsetting_reclassification_candidate",
    )
    candidate_contribution = contribution[
        contribution["cfs_resolution"].eq(candidate_label)
    ].copy()
    top_candidate = _aggregate_contributors(
        candidate_contribution,
        ["section", "offset_channel_pattern", "cfo_adjustment_direction"],
    )

    return {
        "cfs_line_item_reconciliation_cases": reconciliation,
        "cfs_line_item_reconciliation_summary": summary,
        "cfs_line_item_top_contributors": top_candidate,
        "cfs_line_item_top_contributors_all": top_all,
    }
