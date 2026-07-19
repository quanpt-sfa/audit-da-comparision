from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .diag_common import KEYS, paired_panel


def family_discordance(baseline: pd.DataFrame, panel: pd.DataFrame, families: dict[str, list[str]], tolerances: Iterable[float]):
    pivot = baseline.pivot_table(index=KEYS + ["benchmark"], columns="model", values="reduction", aggfunc="first").reset_index()
    move = baseline.groupby(KEYS + ["benchmark"], observed=True).agg(
        raw_ta_shift=("raw_ta_shift", "first"), signed_shift=("signed_shift", "first")
    ).reset_index()
    pivot = pivot.merge(move, on=KEYS + ["benchmark"], validate="one_to_one")
    for family, members in families.items():
        available = [m for m in members if m in pivot]
        if not available:
            raise ValueError(f"No models for family {family}")
        pivot[f"{family}_complete"] = pivot[available].notna().all(axis=1)
        pivot[f"{family}_reduction"] = pivot[available].mean(axis=1, skipna=False)
        pivot[f"{family}_model_sd"] = pivot[available].std(axis=1, ddof=0, skipna=False)
    if len(families) != 2:
        raise ValueError("Exactly two model families required")
    left, right = list(families)
    pivot["family_complete_case"] = pivot[f"{left}_complete"] & pivot[f"{right}_complete"]

    pair = paired_panel(panel)
    for name in ["roa", "revenue", "receivables", "ta_scaled", "total_accruals"]:
        a, b = f"{name}_pre", f"{name}_post"
        if a in pair and b in pair:
            pair[f"delta_{name}"] = pd.to_numeric(pair[b], errors="coerce") - pd.to_numeric(pair[a], errors="coerce")
    if {"ta_source_pre", "ta_source_post"}.issubset(pair):
        pair["ta_source_mismatch"] = pair.ta_source_pre.ne(pair.ta_source_post)
    pivot = pivot.merge(pair, on=KEYS, how="left", validate="many_to_one")

    summaries, years, cases, classifications = [], [], [], []
    for tol in tolerances:
        w = pivot.copy()
        w["tolerance"] = float(tol)
        valid = w["family_complete_case"]
        lc = np.full(len(w), np.nan)
        rc = np.full(len(w), np.nan)
        lc[valid] = np.select(
            [w.loc[valid, f"{left}_reduction"].gt(tol), w.loc[valid, f"{left}_reduction"].lt(-tol)],
            [1, -1], default=0,
        )
        rc[valid] = np.select(
            [w.loc[valid, f"{right}_reduction"].gt(tol), w.loc[valid, f"{right}_reduction"].lt(-tol)],
            [1, -1], default=0,
        )
        w[f"{left}_class"] = lc
        w[f"{right}_class"] = rc
        w["hard_opposite_sign"] = valid & ((lc * rc) == -1)
        w["any_family_discordance"] = valid & (lc != rc)
        w["family_gap"] = w[f"{left}_reduction"] - w[f"{right}_reduction"]
        w["discordance_category"] = np.select(
            [
                ~valid,
                (lc == 1) & (rc == -1),
                (lc == -1) & (rc == 1),
                (lc == 0) & (rc != 0),
                (lc != 0) & (rc == 0),
            ],
            [
                "incomplete_family_models",
                f"{left}_improve__{right}_deteriorate",
                f"{left}_deteriorate__{right}_improve",
                f"{left}_near_zero",
                f"{right}_near_zero",
            ],
            default="agreement",
        )
        classifications.append(w)
        complete = w[valid]
        cases.append(complete[complete.any_family_discordance])
        for benchmark, group in complete.groupby("benchmark", observed=True):
            summaries.append({
                "benchmark": benchmark,
                "tolerance": float(tol),
                "rows_all": int((w["benchmark"] == benchmark).sum()),
                "rows_complete": len(group),
                "complete_case_share": len(group) / max(int((w["benchmark"] == benchmark).sum()), 1),
                "any_discordance_share": float(group.any_family_discordance.mean()),
                "hard_opposite_sign_share": float(group.hard_opposite_sign.mean()),
                "mean_abs_family_gap": float(group.family_gap.abs().mean()),
                "median_abs_family_gap": float(group.family_gap.abs().median()),
                "mean_abs_raw_ta_shift_discordant": float(group.loc[group.any_family_discordance, "raw_ta_shift"].abs().mean()),
            })
        for (benchmark, year), group in complete.groupby(["benchmark", "fiscal_year"], observed=True):
            years.append({
                "benchmark": benchmark, "fiscal_year": year, "tolerance": float(tol),
                "rows_complete": len(group),
                "any_discordance_share": float(group.any_family_discordance.mean()),
                "hard_opposite_sign_share": float(group.hard_opposite_sign.mean()),
            })
    all_classifications = pd.concat(classifications, ignore_index=True) if classifications else pd.DataFrame()
    case = pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()
    cov = []
    if not all_classifications.empty:
        primary = all_classifications[(all_classifications.tolerance.eq(min(tolerances))) & all_classifications.family_complete_case]
        for variable in ["raw_ta_shift", "delta_roa", "delta_revenue", "delta_receivables", "delta_ta_scaled", "delta_total_accruals"]:
            if variable not in primary:
                continue
            for status, group in primary.groupby("any_family_discordance", observed=True):
                values = pd.to_numeric(group[variable], errors="coerce")
                cov.append({
                    "variable": variable, "discordant": bool(status), "rows": int(values.notna().sum()),
                    "mean": float(values.mean()), "median": float(values.median()),
                    "mean_absolute": float(values.abs().mean()),
                })
    return {
        "family_discordance_summary": pd.DataFrame(summaries),
        "family_discordance_by_year": pd.DataFrame(years),
        "family_discordance_cases": case,
        "family_discordance_classifications": all_classifications,
        "family_discordance_covariates": pd.DataFrame(cov),
    }
