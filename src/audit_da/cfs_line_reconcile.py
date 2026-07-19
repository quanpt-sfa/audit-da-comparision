from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .diag_common import KEYS
from .cfs_item_map import pair_line_items


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def line_item_reconciliation(line_item_panel: pd.DataFrame, observed_cases: pd.DataFrame, panel: pd.DataFrame, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paired = pair_line_items(line_item_panel, settings)
    if paired.empty:
        return {"cfs_line_item_reconciliation_cases": pd.DataFrame(), "cfs_line_item_reconciliation_summary": pd.DataFrame(), "cfs_line_item_top_contributors": pd.DataFrame()}
    audited = panel[panel["audit_status"].eq(settings.get("audited_label", "audited"))][KEYS + ["lag_assets"]].drop_duplicates(KEYS)
    cases = observed_cases.drop_duplicates(KEYS).merge(paired, on=KEYS, how="left", validate="one_to_one").merge(audited, on=KEYS, how="left", validate="one_to_one")
    sections = settings.get("concept_sections", {})
    scale = _numeric(cases, "lag_assets")
    for concept in sections:
        cases[f"delta_{concept}_scaled"] = _numeric(cases, f"delta_{concept}") / scale
    rows: list[dict[str, Any]] = []
    contributors: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        for section, aggregate_column in [("investing", "delta_cfi_scaled"), ("financing", "delta_cff_scaled")]:
            concepts = [c for c, s in sections.items() if s == section]
            values = {c: row.get(f"delta_{c}_scaled", np.nan) for c in concepts}
            values = {c: float(v) for c, v in values.items() if np.isfinite(v)}
            mapped = float(sum(values.values())) if values else np.nan
            aggregate = pd.to_numeric(pd.Series([row.get(aggregate_column)]), errors="coerce").iloc[0]
            dominant = max(values, key=lambda c: abs(values[c])) if values else None
            rows.append({
                "issuer_ticker": row["issuer_ticker"], "fiscal_year": row["fiscal_year"], "section": section,
                "cfs_resolution": row.get("cfs_resolution"), "offset_channel_pattern": row.get("offset_channel_pattern"),
                "cfo_adjustment_direction": row.get("cfo_adjustment_direction"), "aggregate_section_change_scaled": aggregate,
                "mapped_line_change_sum_scaled": mapped,
                "reconciliation_residual_scaled": mapped - aggregate if np.isfinite(mapped) and np.isfinite(aggregate) else np.nan,
                "mapped_share_of_aggregate": mapped / aggregate if np.isfinite(mapped) and np.isfinite(aggregate) and abs(aggregate) > 1e-12 else np.nan,
                "mapped_concepts_available": len(values), "dominant_line_item": dominant,
                "dominant_line_item_change_scaled": values.get(dominant, np.nan) if dominant else np.nan,
            })
            contributors.extend({
                "issuer_ticker": row["issuer_ticker"], "fiscal_year": row["fiscal_year"], "section": section,
                "concept": concept, "line_item_change_scaled": value, "absolute_line_item_change_scaled": abs(value),
                "offset_channel_pattern": row.get("offset_channel_pattern"), "cfo_adjustment_direction": row.get("cfo_adjustment_direction"),
            } for concept, value in values.items())
    reconciliation = pd.DataFrame(rows)
    contribution = pd.DataFrame(contributors)
    summary = reconciliation.groupby(["section", "cfs_resolution", "offset_channel_pattern"], dropna=False, observed=True).agg(
        rows=("issuer_ticker", "size"), issuers=("issuer_ticker", "nunique"),
        median_mapped_concepts=("mapped_concepts_available", "median"),
        median_abs_reconciliation_residual=("reconciliation_residual_scaled", lambda x: float(pd.Series(x).abs().median())),
        median_mapped_share=("mapped_share_of_aggregate", "median"),
    ).reset_index()
    top = pd.DataFrame()
    if not contribution.empty:
        top = contribution.groupby(["section", "offset_channel_pattern", "cfo_adjustment_direction", "concept"], dropna=False, observed=True).agg(
            rows=("issuer_ticker", "size"), issuers=("issuer_ticker", "nunique"),
            total_absolute_change=("absolute_line_item_change_scaled", "sum"),
            median_absolute_change=("absolute_line_item_change_scaled", "median"),
            mean_signed_change=("line_item_change_scaled", "mean"),
        ).reset_index().sort_values("total_absolute_change", ascending=False)
    return {"cfs_line_item_reconciliation_cases": reconciliation, "cfs_line_item_reconciliation_summary": summary, "cfs_line_item_top_contributors": top}
