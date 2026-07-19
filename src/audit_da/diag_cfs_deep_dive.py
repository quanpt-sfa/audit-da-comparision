from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from .diag_common import KEYS, trimmed_mean


CANDIDATE_LABEL = "identity_consistent_offsetting_reclassification_candidate"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _safe_share(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def _direction(values: pd.Series, tolerance: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric > tolerance, numeric < -tolerance],
            ["audited_cfo_increase", "audited_cfo_decrease"],
            default="near_zero",
        ),
        index=values.index,
    )


def build_offset_channel_panel(cases: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    frame = cases.copy()
    minimum_year = int(settings.get("minimum_year", 2018))
    frame = frame[pd.to_numeric(frame["fiscal_year"], errors="coerce").ge(minimum_year)].copy()

    for column in [
        "delta_cfo_scaled", "delta_cfi_scaled", "delta_cff_scaled",
        "delta_fx_effect_scaled", "delta_cfs_cash_change_scaled",
        "reduction", "abs_delta_cfo", "abs_delta_pat", "lag_assets_common",
    ]:
        frame[column] = _numeric(frame, column)

    frame["candidate"] = frame["cfs_resolution"].eq(
        settings.get("candidate_label", CANDIDATE_LABEL)
    )
    frame["cfo_adjustment_direction"] = _direction(
        frame["delta_cfo_scaled"],
        float(settings.get("direction_tolerance", 0.0)),
    )
    frame["required_offset_scaled"] = -frame["delta_cfo_scaled"]
    frame["offset_cfi_term"] = frame["delta_cfi_scaled"]
    frame["offset_cff_term"] = frame["delta_cff_scaled"]
    frame["offset_fx_term"] = frame["delta_fx_effect_scaled"].fillna(0.0)
    frame["offset_cash_change_term"] = -frame["delta_cfs_cash_change_scaled"]
    term_columns = [
        "offset_cfi_term", "offset_cff_term", "offset_fx_term",
        "offset_cash_change_term",
    ]
    frame["offset_term_sum"] = frame[term_columns].sum(axis=1, min_count=1)
    frame["offset_reconstruction_error"] = (
        frame["offset_term_sum"] - frame["required_offset_scaled"]
    )

    denominator = frame["required_offset_scaled"].replace(0.0, np.nan)
    absolute_total = frame[term_columns].abs().sum(axis=1).replace(0.0, np.nan)
    channel_names = {
        "offset_cfi_term": "cfi",
        "offset_cff_term": "cff",
        "offset_fx_term": "fx",
        "offset_cash_change_term": "cash_change",
    }
    for column, name in channel_names.items():
        frame[f"{name}_signed_offset_share"] = frame[column] / denominator
        frame[f"{name}_absolute_offset_share"] = frame[column].abs() / absolute_total
        frame[f"{name}_opposes_cfo"] = (
            np.sign(frame[column]) == np.sign(frame["required_offset_scaled"])
        )

    absolute_share_columns = [f"{name}_absolute_offset_share" for name in channel_names.values()]
    dominant_column = frame[absolute_share_columns].idxmax(axis=1)
    dominant_share = frame[absolute_share_columns].max(axis=1)
    frame["dominant_offset_share"] = dominant_share
    frame["dominant_offset_channel"] = dominant_column.str.replace(
        "_absolute_offset_share", "", regex=False
    )
    threshold = float(settings.get("dominant_offset_share_threshold", 0.60))
    frame.loc[dominant_share.lt(threshold) | dominant_share.isna(), "dominant_offset_channel"] = "mixed"

    frame["offset_channel_pattern"] = np.select(
        [
            frame["dominant_offset_channel"].eq("cfi"),
            frame["dominant_offset_channel"].eq("cff"),
            frame["dominant_offset_channel"].isin(["fx", "cash_change"]),
        ],
        ["cfi_dominant", "cff_dominant", "other_dominant"],
        default="mixed",
    )
    frame["reduction_state"] = np.select(
        [frame["reduction"].gt(0), frame["reduction"].lt(0)],
        ["toward_zero", "away_from_zero"],
        default="no_absolute_change",
    )
    frame["cfo_pre_scaled"] = _numeric(frame, "cfo_pre") / _numeric(frame, "lag_assets_common")
    frame["cfo_post_scaled"] = _numeric(frame, "cfo_post") / _numeric(frame, "lag_assets_common")
    return frame


def offset_channel_tables(frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    candidates = frame[frame["candidate"]].copy()
    trim = float(settings.get("trim_fraction", 0.01))

    summary_records: list[dict[str, Any]] = []
    group_columns = ["fiscal_year", "offset_channel_pattern", "cfo_adjustment_direction"]
    for keys, group in candidates.groupby(group_columns, observed=True):
        year, channel, direction = keys
        summary_records.append({
            "fiscal_year": int(year),
            "offset_channel_pattern": channel,
            "cfo_adjustment_direction": direction,
            "rows": len(group),
            "issuers": int(group["issuer_ticker"].nunique()),
            "mean_abs_delta_cfo": float(group["delta_cfo_scaled"].abs().mean()),
            "median_abs_delta_cfo": float(group["delta_cfo_scaled"].abs().median()),
            "mean_reduction": float(group["reduction"].mean()),
            "trimmed_mean_reduction": trimmed_mean(group["reduction"].to_numpy(float), trim),
            "positive_minus_negative_share": _safe_share(group["reduction"].gt(0)) - _safe_share(group["reduction"].lt(0)),
            "median_dominant_offset_share": float(group["dominant_offset_share"].median()),
            "median_abs_reconstruction_error": float(group["offset_reconstruction_error"].abs().median()),
        })

    overall = (
        candidates.groupby(["offset_channel_pattern", "cfo_adjustment_direction"], observed=True)
        .agg(
            rows=("issuer_ticker", "size"),
            issuers=("issuer_ticker", "nunique"),
            mean_reduction=("reduction", "mean"),
            median_reduction=("reduction", "median"),
            mean_abs_delta_cfo=("delta_cfo_scaled", lambda x: float(pd.Series(x).abs().mean())),
            median_abs_delta_cfo=("delta_cfo_scaled", lambda x: float(pd.Series(x).abs().median())),
            median_dominant_offset_share=("dominant_offset_share", "median"),
        )
        .reset_index()
    )
    overall["share_candidates"] = overall["rows"] / max(len(candidates), 1)

    return {
        "cfs_offset_channel_cases": frame,
        "cfs_offset_channel_by_year": pd.DataFrame(summary_records),
        "cfs_offset_channel_summary": overall,
    }


def chronic_reclassifier_tables(frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    minimum_years = int(settings.get("chronic_min_candidate_years", 4))
    share_threshold = float(settings.get("chronic_candidate_share", 0.75))
    direction_threshold = float(settings.get("direction_consistency_threshold", 0.80))

    records: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []
    for issuer, group in frame.sort_values("fiscal_year").groupby("issuer_ticker", observed=True):
        candidate = group[group["candidate"]].copy()
        observed_years = int(group["fiscal_year"].nunique())
        candidate_years = int(candidate["fiscal_year"].nunique())
        if candidate_years:
            increases = int(candidate["cfo_adjustment_direction"].eq("audited_cfo_increase").sum())
            decreases = int(candidate["cfo_adjustment_direction"].eq("audited_cfo_decrease").sum())
            increase_share = increases / candidate_years
            decrease_share = decreases / candidate_years
            channel_counts = candidate["offset_channel_pattern"].value_counts()
            modal_channel = str(channel_counts.index[0]) if len(channel_counts) else "none"
            modal_channel_share = float(channel_counts.iloc[0] / candidate_years) if len(channel_counts) else np.nan
            sorted_candidate = candidate.sort_values("fiscal_year")
            directions = sorted_candidate["cfo_adjustment_direction"].tolist()
            switches = sum(a != b for a, b in zip(directions[:-1], directions[1:]))
        else:
            increases = decreases = 0
            increase_share = decrease_share = np.nan
            modal_channel = "none"
            modal_channel_share = np.nan
            switches = 0

        chronic = candidate_years >= minimum_years and candidate_years / max(observed_years, 1) >= share_threshold
        if not candidate_years:
            direction_type = "never_candidate"
        elif increase_share >= direction_threshold:
            direction_type = "mostly_audited_increase"
        elif decrease_share >= direction_threshold:
            direction_type = "mostly_audited_decrease"
        elif increases > 0 and decreases > 0:
            direction_type = "bidirectional"
        else:
            direction_type = "single_direction_sparse"

        records.append({
            "issuer_ticker": issuer,
            "observed_years": observed_years,
            "candidate_years": candidate_years,
            "candidate_share": candidate_years / max(observed_years, 1),
            "chronic_reclassifier": chronic,
            "audited_cfo_increase_years": increases,
            "audited_cfo_decrease_years": decreases,
            "audited_cfo_increase_share": increase_share,
            "audited_cfo_decrease_share": decrease_share,
            "direction_type": direction_type,
            "direction_switches": switches,
            "modal_offset_channel": modal_channel,
            "modal_offset_channel_share": modal_channel_share,
            "mean_reduction_candidates": float(candidate["reduction"].mean()) if candidate_years else np.nan,
            "median_abs_delta_cfo_candidates": float(candidate["delta_cfo_scaled"].abs().median()) if candidate_years else np.nan,
        })

        if candidate_years:
            for _, row in candidate.iterrows():
                transition_records.append({
                    "issuer_ticker": issuer,
                    "fiscal_year": int(row["fiscal_year"]),
                    "cfo_adjustment_direction": row["cfo_adjustment_direction"],
                    "offset_channel_pattern": row["offset_channel_pattern"],
                    "reduction_state": row["reduction_state"],
                    "reduction": row["reduction"],
                    "delta_cfo_scaled": row["delta_cfo_scaled"],
                })

    profiles = pd.DataFrame(records)
    return {
        "chronic_reclassifier_profiles": profiles.sort_values(
            ["chronic_reclassifier", "candidate_share", "candidate_years"],
            ascending=[False, False, False],
        ),
        "chronic_reclassifier_years": pd.DataFrame(transition_records),
    }


def _cluster_ols(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    categorical: list[str],
    cluster: str,
    model_name: str,
) -> pd.DataFrame:
    columns = [outcome, cluster] + predictors + categorical
    sample = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(sample) < 100 or sample[cluster].nunique() < 20:
        return pd.DataFrame([{
            "model": model_name, "outcome": outcome, "term": "__STATUS__",
            "coef": np.nan, "se_cluster": np.nan, "z": np.nan, "p_value": np.nan,
            "n": len(sample), "clusters": sample[cluster].nunique(),
            "status": "INSUFFICIENT_SAMPLE",
        }])

    numeric = sample[predictors].astype(float)
    for column in predictors:
        sd = numeric[column].std(ddof=0)
        if np.isfinite(sd) and sd > 0:
            numeric[column] = (numeric[column] - numeric[column].mean()) / sd
        else:
            numeric[column] = 0.0
    dummies = pd.get_dummies(sample[categorical].astype(str), prefix=categorical, drop_first=True, dtype=float)
    x_frame = pd.concat([
        pd.Series(1.0, index=sample.index, name="intercept"),
        numeric,
        dummies,
    ], axis=1)
    X = x_frame.to_numpy(float)
    y = sample[outcome].to_numpy(float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    cluster_values = sample[cluster].astype(str).to_numpy()
    unique_clusters = np.unique(cluster_values)
    for value in unique_clusters:
        idx = np.flatnonzero(cluster_values == value)
        score = X[idx].T @ residual[idx]
        meat += np.outer(score, score)
    vcov = xtx_inv @ meat @ xtx_inv
    n, k, g = len(sample), X.shape[1], len(unique_clusters)
    if g > 1 and n > k:
        vcov *= (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * norm.sf(np.abs(z))
    fitted = X @ beta
    denominator = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - fitted) ** 2) / denominator if denominator > 0 else np.nan

    return pd.DataFrame({
        "model": model_name,
        "outcome": outcome,
        "term": x_frame.columns,
        "coef": beta,
        "se_cluster": se,
        "z": z,
        "p_value": p,
        "n": n,
        "clusters": g,
        "r2": r2,
        "status": "OK",
    })


def build_incentive_panel(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    unaudited_label = settings.get("unaudited_label", "unaudited")
    pre = panel[panel["audit_status"].eq(unaudited_label)].drop_duplicates(KEYS).copy()
    keep = [c for c in [
        "issuer_ticker", "fiscal_year", "raw_exchange", "lag_assets", "assets",
        "pat", "cfo", "current_assets", "current_liabilities", "cash",
        "short_term_debt", "revenue", "roa", "loss",
    ] if c in pre.columns]
    pre = pre[keep].copy()
    merged = frame.merge(pre, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_pre_panel"))

    lag = _numeric(merged, "lag_assets")
    merged["pre_roa"] = _numeric(merged, "pat") / lag
    merged["pre_cfo_scaled"] = _numeric(merged, "cfo") / lag
    merged["pre_cash_scaled"] = _numeric(merged, "cash") / lag
    merged["pre_short_debt_scaled"] = _numeric(merged, "short_term_debt") / lag
    merged["pre_current_ratio"] = _numeric(merged, "current_assets") / np.maximum(
        _numeric(merged, "current_liabilities").abs(), 1.0
    )
    merged["log_lag_assets"] = np.log(np.maximum(lag.abs(), 1.0))
    merged["pre_loss"] = _numeric(merged, "pat").lt(0).astype(float)
    merged["pre_negative_cfo"] = merged["pre_cfo_scaled"].lt(0).astype(float)
    merged["pre_near_zero_cfo"] = merged["pre_cfo_scaled"].abs().le(
        float(settings.get("near_zero_cfo_threshold", 0.01))
    ).astype(float)
    merged["pre_liquidity_stress"] = merged["pre_current_ratio"].lt(
        float(settings.get("current_ratio_threshold", 1.0))
    ).astype(float)
    merged["pre_low_cash"] = merged["pre_cash_scaled"].le(
        float(settings.get("low_cash_threshold", 0.02))
    ).astype(float)
    merged["distress_score"] = merged[[
        "pre_loss", "pre_negative_cfo", "pre_liquidity_stress", "pre_low_cash"
    ]].sum(axis=1)
    material = float(settings.get("material_cfo_threshold", 0.05))
    merged["material_cfo_adjustment"] = merged["delta_cfo_scaled"].abs().ge(material).astype(float)
    merged["audited_cfo_decrease"] = merged["delta_cfo_scaled"].le(-material).astype(float)
    merged["audited_cfo_increase"] = merged["delta_cfo_scaled"].ge(material).astype(float)
    merged["abs_delta_cfo"] = merged["delta_cfo_scaled"].abs()
    return merged


def incentive_tables(data: pd.DataFrame, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    indicators = [
        "pre_loss", "pre_negative_cfo", "pre_near_zero_cfo",
        "pre_liquidity_stress", "pre_low_cash",
    ]
    descriptive_records: list[dict[str, Any]] = []
    for indicator in indicators:
        for value, group in data.groupby(indicator, observed=True):
            candidates = group[group["candidate"]]
            descriptive_records.append({
                "indicator": indicator,
                "indicator_value": int(value),
                "rows": len(group),
                "candidate_incidence": _safe_share(group["candidate"]),
                "material_cfo_adjustment_share": _safe_share(group["material_cfo_adjustment"].eq(1)),
                "downward_share_all": _safe_share(group["audited_cfo_decrease"].eq(1)),
                "downward_share_candidates": _safe_share(candidates["audited_cfo_decrease"].eq(1)),
                "upward_share_candidates": _safe_share(candidates["audited_cfo_increase"].eq(1)),
                "mean_abs_delta_cfo_candidates": float(candidates["abs_delta_cfo"].mean()) if len(candidates) else np.nan,
                "mean_reduction_candidates": float(candidates["reduction"].mean()) if len(candidates) else np.nan,
            })

    predictors = indicators + [
        "pre_roa", "pre_short_debt_scaled", "pre_current_ratio", "log_lag_assets"
    ]
    categorical = [c for c in ["fiscal_year", "raw_exchange"] if c in data.columns]
    model_frames = [
        _cluster_ols(
            data.assign(candidate_numeric=data["candidate"].astype(float)),
            "candidate_numeric", predictors, categorical, "issuer_ticker",
            "candidate_incidence_lpm",
        ),
        _cluster_ols(
            data[data["candidate"]].copy(),
            "audited_cfo_decrease", predictors, categorical, "issuer_ticker",
            "downward_direction_among_candidates_lpm",
        ),
        _cluster_ols(
            data[data["candidate"]].copy(),
            "abs_delta_cfo", predictors, categorical, "issuer_ticker",
            "candidate_magnitude_ols",
        ),
    ]
    return {
        "cfs_incentive_cases": data,
        "cfs_incentive_descriptives": pd.DataFrame(descriptive_records),
        "cfs_incentive_models": pd.concat(model_frames, ignore_index=True),
    }


def common_sample_anchor_table(alignment_cases: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    frame = alignment_cases.copy()
    component_column = "component" if "component" in frame.columns else None
    if component_column:
        frame = frame[frame[component_column].eq("cfo")]
    records: list[dict[str, Any]] = []
    for (model, benchmark), group in frame.groupby(["model", "benchmark"], observed=True):
        required = [
            "cfo_component_movement", "da_pre", "da_pre_balance_sheet_anchor",
            "cfo_reduction_cashflow_anchor", "cfo_reduction_balance_sheet_anchor",
        ]
        finite = np.ones(len(group), dtype=bool)
        for column in required:
            finite &= np.isfinite(_numeric(group, column).to_numpy(float))
        sample = group.loc[finite].copy()
        if sample.empty:
            continue
        movement = sample["cfo_component_movement"].to_numpy(float)
        cf_anchor = sample["da_pre"].to_numpy(float)
        bs_anchor = sample["da_pre_balance_sheet_anchor"].to_numpy(float)
        cf_reduction = sample["cfo_reduction_cashflow_anchor"].to_numpy(float)
        bs_reduction = sample["cfo_reduction_balance_sheet_anchor"].to_numpy(float)
        trim = float(settings.get("trim_fraction", 0.01))
        records.append({
            "model": model,
            "benchmark": benchmark,
            "common_rows": len(sample),
            "mean_reduction_cashflow_anchor": float(cf_reduction.mean()),
            "mean_reduction_balance_sheet_anchor": float(bs_reduction.mean()),
            "paired_mean_difference_cf_minus_bs": float((cf_reduction - bs_reduction).mean()),
            "trimmed_mean_reduction_cashflow_anchor": trimmed_mean(cf_reduction, trim),
            "trimmed_mean_reduction_balance_sheet_anchor": trimmed_mean(bs_reduction, trim),
            "positive_minus_negative_cashflow_anchor": float((cf_reduction > 0).mean() - (cf_reduction < 0).mean()),
            "positive_minus_negative_balance_sheet_anchor": float((bs_reduction > 0).mean() - (bs_reduction < 0).mean()),
            "anchor_direction_agreement": float(np.sign(cf_reduction).eq(np.sign(bs_reduction)).mean()) if isinstance(cf_reduction, pd.Series) else float((np.sign(cf_reduction) == np.sign(bs_reduction)).mean()),
            "corr_movement_negative_cf_anchor": float(np.corrcoef(movement, -cf_anchor)[0, 1]) if len(sample) > 1 else np.nan,
            "corr_movement_negative_bs_anchor": float(np.corrcoef(movement, -bs_anchor)[0, 1]) if len(sample) > 1 else np.nan,
        })
    return pd.DataFrame(records)


def audit_quality_tables(
    incentive_data: pd.DataFrame,
    metadata_path: Path,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    status = {
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.exists(),
        "status": "NOT_EVALUATED",
        "reason": "Audit metadata file not found",
    }
    metadata = _read_optional_csv(metadata_path)
    if metadata.empty:
        return {
            "audit_quality_status": pd.DataFrame([status]),
            "audit_quality_summary": pd.DataFrame(),
            "audit_quality_models": pd.DataFrame(),
        }

    required = set(KEYS)
    if not required.issubset(metadata.columns):
        status["reason"] = f"Metadata missing keys: {sorted(required - set(metadata.columns))}"
        return {
            "audit_quality_status": pd.DataFrame([status]),
            "audit_quality_summary": pd.DataFrame(),
            "audit_quality_models": pd.DataFrame(),
        }

    auditor_column = settings.get("auditor_column", "auditor_name")
    opinion_column = settings.get("opinion_column", "audit_opinion")
    merged = incentive_data.merge(metadata, on=KEYS, how="left", validate="one_to_one")
    big4_patterns = settings.get("big4_patterns", [
        "deloitte", "kpmg", "ernst", "ey", "pricewaterhouse", "pwc",
    ])
    nonclean_patterns = settings.get("nonclean_patterns", [
        "qualified", "adverse", "disclaimer", "except for", "going concern",
        "ngoại trừ", "không chấp nhận", "từ chối", "hoạt động liên tục",
    ])
    if auditor_column in merged:
        auditor_text = merged[auditor_column].fillna("").astype(str).str.lower()
        regex = "|".join(re.escape(x.lower()) for x in big4_patterns)
        merged["big4"] = auditor_text.str.contains(regex, regex=True).astype(float)
        merged.loc[auditor_text.eq(""), "big4"] = np.nan
    else:
        merged["big4"] = np.nan
    if opinion_column in merged:
        opinion_text = merged[opinion_column].fillna("").astype(str).str.lower()
        regex = "|".join(re.escape(x.lower()) for x in nonclean_patterns)
        merged["nonclean_opinion"] = opinion_text.str.contains(regex, regex=True).astype(float)
        merged.loc[opinion_text.eq(""), "nonclean_opinion"] = np.nan
    else:
        merged["nonclean_opinion"] = np.nan

    available = merged[["big4", "nonclean_opinion"]].notna().any(axis=1)
    if not available.any():
        status["reason"] = "Metadata joined but auditor/opinion fields are unavailable"
        return {
            "audit_quality_status": pd.DataFrame([status]),
            "audit_quality_summary": pd.DataFrame(),
            "audit_quality_models": pd.DataFrame(),
        }

    status.update({
        "status": "EVALUATED",
        "reason": "OK",
        "joined_rows": len(merged),
        "rows_with_any_audit_metadata": int(available.sum()),
    })
    summary_records: list[dict[str, Any]] = []
    for variable in ["big4", "nonclean_opinion"]:
        if merged[variable].notna().any():
            for value, group in merged.dropna(subset=[variable]).groupby(variable, observed=True):
                candidates = group[group["candidate"]]
                summary_records.append({
                    "audit_quality_variable": variable,
                    "value": int(value),
                    "rows": len(group),
                    "candidate_incidence": _safe_share(group["candidate"]),
                    "downward_share_candidates": _safe_share(candidates["audited_cfo_decrease"].eq(1)),
                    "mean_abs_delta_cfo_candidates": float(candidates["abs_delta_cfo"].mean()) if len(candidates) else np.nan,
                    "mean_reduction_candidates": float(candidates["reduction"].mean()) if len(candidates) else np.nan,
                })

    predictors = [c for c in [
        "big4", "nonclean_opinion", "pre_loss", "pre_negative_cfo",
        "pre_liquidity_stress", "pre_roa", "log_lag_assets",
    ] if c in merged.columns and merged[c].notna().any()]
    categorical = [c for c in ["fiscal_year", "raw_exchange"] if c in merged.columns]
    models = _cluster_ols(
        merged.assign(candidate_numeric=merged["candidate"].astype(float)),
        "candidate_numeric", predictors, categorical, "issuer_ticker",
        "audit_quality_candidate_incidence_lpm",
    ) if predictors else pd.DataFrame()
    return {
        "audit_quality_status": pd.DataFrame([status]),
        "audit_quality_summary": pd.DataFrame(summary_records),
        "audit_quality_models": models,
    }


def verification_sample(
    data: pd.DataFrame,
    chronic_profiles: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    merged = data.merge(
        chronic_profiles[[
            "issuer_ticker", "candidate_years", "candidate_share",
            "chronic_reclassifier", "direction_type", "modal_offset_channel",
        ]],
        on="issuer_ticker",
        how="left",
        validate="many_to_one",
    )
    random_seed = int(settings.get("random_seed", 20260720))
    rng = np.random.default_rng(random_seed)
    total = int(settings.get("pdf_sample_size", 40))

    selected: list[pd.DataFrame] = []
    quotas = {
        "pre_internal_inconsistency_repaired": int(settings.get("pdf_fail_to_pass_quota", 5)),
        "insufficient_cfs_components": int(settings.get("pdf_insufficient_quota", 5)),
    }
    for label, quota in quotas.items():
        pool = merged[merged["cfs_resolution"].eq(label)]
        if len(pool):
            selected.append(pool.sample(min(quota, len(pool)), random_state=random_seed))

    candidate_pool = merged[merged["candidate"]].copy()
    candidate_pool["verification_stratum"] = (
        candidate_pool["offset_channel_pattern"].astype(str)
        + "|" + candidate_pool["cfo_adjustment_direction"].astype(str)
        + "|chronic=" + candidate_pool["chronic_reclassifier"].fillna(False).astype(str)
    )
    remaining = max(total - sum(len(x) for x in selected), 0)
    if remaining and len(candidate_pool):
        strata = list(candidate_pool["verification_stratum"].dropna().unique())
        picks: list[pd.DataFrame] = []
        while remaining > 0 and strata:
            next_strata: list[str] = []
            for stratum in strata:
                pool = candidate_pool[
                    candidate_pool["verification_stratum"].eq(stratum)
                    & ~candidate_pool.index.isin(pd.concat(picks).index if picks else [])
                ]
                if len(pool):
                    index = int(rng.integers(0, len(pool)))
                    picks.append(pool.iloc[[index]])
                    remaining -= 1
                    if len(pool) > 1:
                        next_strata.append(stratum)
                if remaining <= 0:
                    break
            strata = next_strata
        if picks:
            selected.append(pd.concat(picks))

    if not selected:
        return pd.DataFrame()
    sample = pd.concat(selected).drop_duplicates(KEYS).copy()
    sample["verification_priority"] = np.select(
        [
            sample["cfs_resolution"].eq("pre_internal_inconsistency_repaired"),
            sample["cfs_resolution"].eq("insufficient_cfs_components"),
            sample["chronic_reclassifier"].fillna(False),
            sample["abs_delta_cfo"].ge(sample["abs_delta_cfo"].quantile(.90)),
        ],
        ["identity_repair", "missing_components", "chronic_candidate", "extreme_candidate"],
        default="stratified_candidate",
    )
    columns = [c for c in [
        "issuer_ticker", "fiscal_year", "verification_priority", "cfs_resolution",
        "offset_channel_pattern", "cfo_adjustment_direction", "reduction_state",
        "delta_cfo_scaled", "delta_cfi_scaled", "delta_cff_scaled",
        "delta_fx_effect_scaled", "delta_cfs_cash_change_scaled",
        "offset_reconstruction_error", "reduction", "abs_delta_pat",
        "candidate_years", "candidate_share", "chronic_reclassifier",
        "direction_type", "pre_loss", "pre_negative_cfo", "pre_near_zero_cfo",
        "pre_liquidity_stress", "raw_exchange",
    ] if c in sample.columns]
    return sample[columns].sort_values(["verification_priority", "issuer_ticker", "fiscal_year"])


def deep_dive_tables(
    identity_cases: pd.DataFrame,
    alignment_cases: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict[str, Any],
    audit_metadata_path: Path,
) -> dict[str, pd.DataFrame]:
    offset_panel = build_offset_channel_panel(identity_cases, settings)
    output = offset_channel_tables(offset_panel, settings)
    chronic = chronic_reclassifier_tables(offset_panel, settings)
    output.update(chronic)

    incentive = build_incentive_panel(offset_panel, panel, settings)
    incentive_output = incentive_tables(incentive, settings)
    output.update(incentive_output)

    output["component_anchor_common_sample"] = common_sample_anchor_table(
        alignment_cases, settings
    )
    output.update(audit_quality_tables(
        incentive,
        audit_metadata_path,
        settings.get("audit_metadata", {}),
    ))
    output["cfs_pdf_verification_sample"] = verification_sample(
        incentive,
        chronic["chronic_reclassifier_profiles"],
        settings,
    )
    return output
