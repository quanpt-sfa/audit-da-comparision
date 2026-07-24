from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    KEYS,
    CompletionSettings,
    _adjust_pvalues,
    _cluster_ols,
    _design_matrix,
    _find_column,
    paired_panel,
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.abs() > 0))


def _winsorise(
    frame: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        lo, hi = valid.quantile([lower, upper])
        frame[column] = values.clip(lo, hi)


def _fully_interacted_stacked(
    y_pre: np.ndarray,
    y_aud: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use [X beta_pre, state*X delta], interacting every slope and FE."""
    zeros = np.zeros_like(x)
    stacked_x = np.vstack([
        np.hstack([x, zeros]),
        np.hstack([x, x]),
    ])
    stacked_y = np.concatenate([y_pre, y_aud])
    stacked_clusters = np.concatenate([clusters, clusters])
    return _cluster_ols(stacked_y, stacked_x, stacked_clusters)


def _prepare_pair(
    panel: pd.DataFrame,
    settings: CompletionSettings,
) -> tuple[pd.DataFrame, str | None]:
    pair = paired_panel(panel, settings)
    industry_col = _find_column(pair, [
        "icb_l1_pre", "industry_name_pre", "icb_industry_pre", "industry_pre",
    ])
    lag_assets = _find_column(pair, ["lag_assets_pre", "lag_assets_audited_pre"])
    if lag_assets:
        pair["__log_assets"] = np.log(np.maximum(
            pd.to_numeric(pair[lag_assets], errors="coerce").abs(), 1.0
        ))
    elif "assets_pre" in pair:
        pair["__log_assets"] = np.log(np.maximum(
            pd.to_numeric(pair.assets_pre, errors="coerce").abs(), 1.0
        ))
    else:
        pair["__log_assets"] = np.nan
    if "short_term_debt_pre" in pair and lag_assets:
        pair["__short_debt_intensity"] = _safe_ratio(
            pair["short_term_debt_pre"], pair[lag_assets]
        )
    if "current_assets_pre" in pair and "current_liabilities_pre" in pair:
        pair["__current_ratio"] = _safe_ratio(
            pair["current_assets_pre"], pair["current_liabilities_pre"]
        )
    return pair, industry_col


def _unique_test_id(model: str, focal: str, outcome: str) -> str:
    if outcome == "signed_da":
        return f"signed_da::{focal}"
    return f"high_da::{model}::{focal}"


def applied_consequence_tables(
    accrual_rows: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return descriptive model rows, unique change tests, and a manifest."""
    pair, industry_col = _prepare_pair(panel, settings)
    aliases = {
        "big4": ["big4_flag_pre", "big4_pre", "is_big4_pre", "big_four_pre"],
        "short_debt": [
            "__short_debt_intensity", "short_term_debt_intensity_pre", "std_intensity_pre",
        ],
        "loss": ["loss_pre"],
        "roa": ["roa_pre"],
        "current_ratio": ["__current_ratio", "current_ratio_pre"],
    }
    resolved = {key: _find_column(pair, values) for key, values in aliases.items()}
    rows: list[dict] = []
    manifest: list[dict] = []
    primary = accrual_rows[
        accrual_rows.architecture.eq("pooled")
        & accrual_rows.benchmark.eq("audited_reference")
    ].copy()

    for model, group in primary.groupby("model", observed=True):
        data = group.merge(pair, on=KEYS, how="left", validate="many_to_one")
        cutoff = data.da_post.abs().quantile(settings.tail_quantile)
        data["high_da_pre"] = data.da_pre.abs().ge(cutoff).astype(float)
        data["high_da_post"] = data.da_post.abs().ge(cutoff).astype(float)
        for focal_name in ("big4", "short_debt", "loss"):
            focal = resolved[focal_name]
            if focal is None:
                manifest.append({
                    "model": model, "design": focal_name,
                    "status": "missing_focal_column",
                })
                continue
            controls = ["__log_assets"] + [
                column for name, column in resolved.items()
                if name != focal_name and column is not None
            ]
            for outcome_name, pre_col, aud_col in (
                ("signed_da", "da_pre", "da_post"),
                ("high_da", "high_da_pre", "high_da_post"),
            ):
                needed = [
                    focal, *controls, pre_col, aud_col,
                    "issuer_ticker", "fiscal_year",
                ] + ([industry_col] if industry_col else [])
                sample = (
                    data.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=needed).copy()
                )
                if len(sample) < 50:
                    manifest.append({
                        "model": model,
                        "design": f"{focal_name}_{outcome_name}",
                        "status": "insufficient_rows", "rows": len(sample),
                    })
                    continue
                continuous = [
                    column for column in {
                        "__log_assets", resolved.get("short_debt"),
                        resolved.get("roa"), resolved.get("current_ratio"),
                    } if column is not None
                ]
                _winsorise(sample, continuous, settings.winsor_lower, settings.winsor_upper)
                x, names = _design_matrix(
                    sample, [focal, *controls], industry_col=industry_col
                )
                focal_idx = names.index(focal)
                y_pre = sample[pre_col].to_numpy(float)
                y_aud = sample[aud_col].to_numpy(float)
                clusters = sample.issuer_ticker.to_numpy()
                b_pre, se_pre, p_pre = _cluster_ols(y_pre, x, clusters)
                b_aud, se_aud, p_aud = _cluster_ols(y_aud, x, clusters)
                b_stack, se_stack, p_stack = _fully_interacted_stacked(
                    y_pre, y_aud, x, clusters
                )
                stacked_delta_idx = x.shape[1] + focal_idx
                difference = y_aud - y_pre
                b_diff, se_diff, p_diff = _cluster_ols(difference, x, clusters)
                pre_sd = float(sample[pre_col].std(ddof=1))
                coefficient_difference = b_aud[focal_idx] - b_pre[focal_idx]
                alignment_error = max(
                    abs(coefficient_difference - b_stack[stacked_delta_idx]),
                    abs(coefficient_difference - b_diff[focal_idx]),
                )
                rows.append({
                    "model": model,
                    "focal": focal_name,
                    "outcome": outcome_name,
                    "difference_test_id": _unique_test_id(model, focal_name, outcome_name),
                    "n": len(sample),
                    "issuers": sample.issuer_ticker.nunique(),
                    "pre_beta": b_pre[focal_idx],
                    "pre_se": se_pre[focal_idx],
                    "pre_p": p_pre[focal_idx],
                    "aud_beta": b_aud[focal_idx],
                    "aud_se": se_aud[focal_idx],
                    "aud_p": p_aud[focal_idx],
                    "post_beta": b_aud[focal_idx],
                    "post_se": se_aud[focal_idx],
                    "post_p": p_aud[focal_idx],
                    "beta_difference": coefficient_difference,
                    "standardised_beta_difference": (
                        coefficient_difference / pre_sd if pre_sd > 0 else np.nan
                    ),
                    "interaction_beta": b_stack[stacked_delta_idx],
                    "interaction_se": se_stack[stacked_delta_idx],
                    "interaction_p": p_stack[stacked_delta_idx],
                    "paired_difference_beta": b_diff[focal_idx],
                    "paired_difference_se": se_diff[focal_idx],
                    "paired_difference_p": p_diff[focal_idx],
                    "estimand_alignment_error": alignment_error,
                    "stacked_state_slopes": "fully_interacted",
                    "primary_change_test": "paired_difference",
                    "algebraically_model_invariant": outcome_name == "signed_da",
                    "significance_status_switch": (
                        (p_pre[focal_idx] < 0.05) != (p_aud[focal_idx] < 0.05)
                    ),
                })
                manifest.append({
                    "model": model,
                    "design": f"{focal_name}_{outcome_name}",
                    "status": "estimated", "rows": len(sample),
                    "stacked_state_slopes": "fully_interacted",
                })

    full = pd.DataFrame(rows)
    if full.empty:
        return full, pd.DataFrame(), pd.DataFrame(manifest)
    tolerance = 1.0e-10
    if float(full.estimand_alignment_error.max()) > tolerance:
        raise AssertionError(
            "Fully interacted stacked and paired estimands diverged: "
            f"max_error={float(full.estimand_alignment_error.max())}"
        )
    unique_rows: list[pd.Series] = []
    for test_id, group in full.groupby("difference_test_id", observed=True, sort=False):
        ordered = group.sort_values("model", kind="mergesort")
        if ordered.outcome.iloc[0] == "signed_da":
            spread = float(
                ordered.paired_difference_beta.max()
                - ordered.paired_difference_beta.min()
            )
            if abs(spread) > tolerance:
                raise AssertionError(
                    "Signed-DA fixed-reference difference is not model invariant: "
                    f"test={test_id} spread={spread}"
                )
            representative = (
                ordered.loc[ordered.model.eq("modified_jones")].iloc[0]
                if ordered.model.eq("modified_jones").any()
                else ordered.iloc[0]
            ).copy()
            representative["source_models"] = ",".join(
                sorted(ordered.model.astype(str).unique())
            )
            representative["replication_count"] = len(ordered)
        else:
            representative = ordered.iloc[0].copy()
            representative["source_models"] = str(representative["model"])
            representative["replication_count"] = 1
        unique_rows.append(representative)
    unique = pd.DataFrame(unique_rows).reset_index(drop=True)
    unique["interaction_q_bh"] = _adjust_pvalues(
        unique.interaction_p.fillna(1.0), "bh"
    )
    unique["paired_difference_q_bh"] = _adjust_pvalues(
        unique.paired_difference_p.fillna(1.0), "bh"
    )
    q_map = unique.set_index("difference_test_id")[[
        "interaction_q_bh", "paired_difference_q_bh"
    ]]
    full = full.drop(
        columns=["interaction_q_bh", "paired_difference_q_bh"], errors="ignore"
    ).merge(
        q_map, left_on="difference_test_id", right_index=True,
        how="left", validate="many_to_one",
    )
    return full, unique, pd.DataFrame(manifest)
