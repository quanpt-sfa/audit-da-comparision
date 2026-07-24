from __future__ import annotations

import numpy as np
import pandas as pd

from .core import KEYS, CompletionSettings, _numeric, paired_panel
from .switching import _common_categories, _midrank_against_reference


def _finite(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def _profit_gate_complete(pair: pd.DataFrame, threshold: float) -> pd.Series:
    """Return a nullable gate; missing state values are never coded outside-gate."""
    output = pd.Series(pd.NA, index=pair.index, dtype="boolean")
    required = ["pat_pre", "pat_post", "lag_assets_pre"]
    valid = _finite(pair, required)
    assets = pd.to_numeric(pair["lag_assets_pre"], errors="coerce").abs()
    valid &= assets.gt(0)
    if not valid.any():
        return output

    pat_pre = pd.to_numeric(pair.loc[valid, "pat_pre"], errors="coerce")
    pat_post = pd.to_numeric(pair.loc[valid, "pat_post"], errors="coerce")
    valid_assets = assets.loc[valid]
    sign_change = np.signbit(pat_pre) != np.signbit(pat_post)
    denominator = np.maximum(pat_pre.abs(), 0.001 * valid_assets)
    ratio = (pat_post - pat_pre).abs() / denominator
    output.loc[valid] = (sign_change | ratio.ge(threshold)).to_numpy(bool)
    return output


def switching_cases(
    accrual_rows: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct switching cases on explicit complete-case populations.

    The direct layer uses one common complete sample for PAT, CFO, total accruals,
    and beginning assets. Model rows may exist outside this direct sample, but
    their profit gate remains missing and is excluded from coverage inference.
    """
    pair = paired_panel(panel, settings)
    numeric = [
        "pat_pre",
        "pat_post",
        "cfo_pre",
        "cfo_post",
        "lag_assets_pre",
        "ta_scaled_pre",
        "ta_scaled_post",
    ]
    pair = _numeric(pair, numeric)
    direct_valid = _finite(pair, numeric)
    direct_valid &= pd.to_numeric(pair["lag_assets_pre"], errors="coerce").abs().gt(0)
    base = pair.loc[direct_valid].copy()

    base["gate_0_05"] = _profit_gate_complete(base, 0.05).astype(bool)
    base["cfo_sign_switch"] = np.signbit(base.cfo_pre) != np.signbit(base.cfo_post)
    base["cfo_sign_magnitude"] = (
        (base.cfo_post - base.cfo_pre).abs() / base.lag_assets_pre.abs()
    )

    category_frames: list[pd.DataFrame] = []
    for _, group in base.groupby("fiscal_year", observed=True):
        tmp = group.copy()
        pre_cat, post_cat, _ = _common_categories(
            tmp.cfo_pre / tmp.lag_assets_pre,
            tmp.cfo_post / tmp.lag_assets_pre,
            5,
        )
        tmp["cfo_category_pre"] = pre_cat
        tmp["cfo_category_post"] = post_cat
        tmp["cfo_category_switch"] = pre_cat.ne(post_cat)
        tmp["cfo_category_distance"] = (post_cat - pre_cat).abs()

        ta_cut = float(tmp.ta_scaled_post.abs().quantile(settings.tail_quantile))
        tmp["high_ta_pre"] = tmp.ta_scaled_pre.abs().ge(ta_cut)
        tmp["high_ta_post"] = tmp.ta_scaled_post.abs().ge(ta_cut)
        tmp["high_ta_switch"] = tmp.high_ta_pre.ne(tmp.high_ta_post)
        tmp["high_ta_magnitude"] = (
            tmp.ta_scaled_post.abs() - tmp.ta_scaled_pre.abs()
        ).abs()
        category_frames.append(tmp)

    direct = pd.concat(category_frames, ignore_index=True)
    gate_map = direct[KEYS + ["gate_0_05"]]

    model_frames: list[pd.DataFrame] = []
    for (model, architecture, benchmark, _), group in accrual_rows.groupby(
        ["model", "architecture", "benchmark", "fiscal_year"], observed=True
    ):
        if architecture != "pooled":
            continue
        tmp = group.merge(gate_map, on=KEYS, how="left", validate="many_to_one")
        finite_da = _finite(tmp, ["da_pre", "da_post", "signed_shift"])
        tmp = tmp.loc[finite_da].copy()

        reference = tmp.da_post.abs()
        cutoff = float(reference.quantile(settings.tail_quantile))
        tmp["da_sign_switch"] = np.signbit(tmp.da_pre) != np.signbit(tmp.da_post)
        tmp["da_sign_magnitude"] = tmp.signed_shift.abs()
        tmp["high_da_pre"] = tmp.da_pre.abs().ge(cutoff)
        tmp["high_da_post"] = tmp.da_post.abs().ge(cutoff)
        tmp["high_da_switch"] = tmp.high_da_pre.ne(tmp.high_da_post)
        tmp["high_da_magnitude"] = (tmp.da_post.abs() - tmp.da_pre.abs()).abs()
        tmp["rank_pre"] = _midrank_against_reference(tmp.da_pre.abs(), reference)
        tmp["rank_post"] = _midrank_against_reference(tmp.da_post.abs(), reference)
        tmp["rank_displacement"] = (tmp.rank_post - tmp.rank_pre).abs()
        model_frames.append(tmp)

    model_cases = (
        pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    )
    direct["outcome_scope"] = "direct"
    if not model_cases.empty:
        model_cases["outcome_scope"] = "model"
    return direct, model_cases


def profit_gate_sensitivity(
    direct: pd.DataFrame,
    model_cases: pd.DataFrame,
    settings: CompletionSettings,
) -> pd.DataFrame:
    """Re-estimate gate coverage without converting missing gates to False."""
    rows: list[dict] = []
    for threshold in settings.profit_thresholds:
        current = direct.copy()
        current["gate"] = _profit_gate_complete(current, threshold)
        for outcome, switch in (
            ("cfo_sign", "cfo_sign_switch"),
            ("cfo_category", "cfo_category_switch"),
            ("high_ta", "high_ta_switch"),
        ):
            valid = current[switch].notna() & current["gate"].notna()
            switched = current.loc[valid & current[switch].astype(bool)]
            rows.append(
                {
                    "threshold": threshold,
                    "outcome": outcome,
                    "model": "direct",
                    "switch_n": len(switched),
                    "outside_gate_share": (
                        float((~switched.gate.astype(bool)).mean())
                        if len(switched)
                        else np.nan
                    ),
                }
            )

        if model_cases.empty:
            continue
        gate_map = current[KEYS + ["gate"]]
        for (model, benchmark), group in model_cases.groupby(
            ["model", "benchmark"], observed=True
        ):
            merged = group.drop(columns=["gate_0_05"], errors="ignore").merge(
                gate_map, on=KEYS, how="left", validate="many_to_one"
            )
            for outcome, switch in (
                ("da_sign", "da_sign_switch"),
                ("high_da", "high_da_switch"),
            ):
                valid = merged[switch].notna() & merged["gate"].notna()
                switched = merged.loc[valid & merged[switch].astype(bool)]
                rows.append(
                    {
                        "threshold": threshold,
                        "outcome": outcome,
                        "model": model,
                        "benchmark": benchmark,
                        "switch_n": len(switched),
                        "outside_gate_share": (
                            float((~switched.gate.astype(bool)).mean())
                            if len(switched)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)
