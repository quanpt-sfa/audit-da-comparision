from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .diag_common import KEYS, paired_panel


def _finite_abs_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(np.abs(numeric))) if len(numeric) else np.nan


def _finite_abs_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.max(np.abs(numeric))) if len(numeric) else np.nan


def ta_source_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = panel.copy()
    for column in ["total_accruals", "ta_cashflow", "ta_balance_sheet", "pat", "cfo"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    expected = np.select(
        [frame.ta_source.eq("cash_flow"), frame.ta_source.eq("balance_sheet")],
        [frame.ta_cashflow, frame.ta_balance_sheet],
        default=np.nan,
    )
    frame["ta_identity_error"] = frame.total_accruals - expected
    frame["ta_cashflow_formula_error"] = frame.ta_cashflow - (frame.pat - frame.cfo)

    summary = frame.groupby(["audit_status", "ta_source"], observed=True).agg(
        rows=("issuer_ticker", "size"),
        firms=("issuer_ticker", "nunique"),
        years=("fiscal_year", "nunique"),
        median_abs_identity_error=("ta_identity_error", _finite_abs_median),
        max_abs_identity_error=("ta_identity_error", _finite_abs_max),
        median_abs_cashflow_formula_error=("ta_cashflow_formula_error", _finite_abs_median),
    ).reset_index()
    summary["share_within_status"] = (
        summary.rows / summary.groupby("audit_status").rows.transform("sum")
    )

    pair = paired_panel(frame)
    if {"ta_source_pre", "ta_source_post"}.issubset(pair.columns):
        pair["ta_source_pair"] = (
            pair.ta_source_pre.astype(str) + "__to__" + pair.ta_source_post.astype(str)
        )
        pair["ta_source_mismatch"] = pair.ta_source_pre != pair.ta_source_post
    return summary, pair


def tail_case_tables(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
    primary_model: str,
    primary_benchmark: str,
    tail_fraction: float = .01,
    manual_cases_per_side: int = 30,
    special_year: int = 2024,
) -> dict[str, pd.DataFrame]:
    chosen = baseline[
        baseline.model.eq(primary_model)
        & baseline.benchmark.eq(primary_benchmark)
    ].copy()
    chosen = chosen.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["da_pre", "da_post", "reduction", "raw_ta_shift"]
    )
    if chosen.empty:
        raise ValueError(f"No finite rows for {primary_model}/{primary_benchmark}")

    _, pair = ta_source_audit(panel)
    x = chosen.merge(pair, on=KEYS, how="left", validate="one_to_one")
    lo, hi = x.reduction.quantile([tail_fraction, 1 - tail_fraction])
    x["tail_side"] = np.select(
        [x.reduction <= lo, x.reduction >= hi],
        ["negative_tail", "positive_tail"],
        default="body",
    )
    x["abs_reduction"] = x.reduction.abs()
    x["sign_flip"] = np.signbit(x.da_pre) != np.signbit(x.da_post)
    x["panel_raw_ta_shift"] = (
        x.get("ta_scaled_post", np.nan) - x.get("ta_scaled_pre", np.nan)
    )
    x["raw_ta_shift_identity_error"] = x.raw_ta_shift - x.panel_raw_ta_shift

    for name in [
        "pat",
        "cfo",
        "total_accruals",
        "ta_scaled",
        "revenue",
        "receivables",
        "ppe",
        "roa",
    ]:
        pre_column, post_column = f"{name}_pre", f"{name}_post"
        if pre_column in x and post_column in x:
            x[f"delta_{name}"] = (
                pd.to_numeric(x[post_column], errors="coerce")
                - pd.to_numeric(x[pre_column], errors="coerce")
            )

    manual = pd.concat(
        [
            x.nsmallest(manual_cases_per_side, "reduction"),
            x.nlargest(manual_cases_per_side, "reduction"),
            x[x.fiscal_year.eq(special_year)].nsmallest(
                manual_cases_per_side, "reduction"
            ),
            x[x.fiscal_year.eq(special_year)].nlargest(
                manual_cases_per_side, "reduction"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(KEYS)

    return {
        "tail_cases_all": x[x.tail_side.ne("body")].sort_values("reduction"),
        "tail_cases_manual_review": manual.sort_values(["fiscal_year", "reduction"]),
        "tail_source_summary": x.groupby(
            ["tail_side", "ta_source_pair"], observed=True
        ).agg(
            rows=("issuer_ticker", "size"),
            mean_reduction=("reduction", "mean"),
            median_reduction=("reduction", "median"),
        ).reset_index(),
        "tail_year_summary": x.groupby(
            ["fiscal_year", "tail_side"], observed=True
        ).agg(
            rows=("issuer_ticker", "size"),
            mean_reduction=("reduction", "mean"),
            median_reduction=("reduction", "median"),
        ).reset_index(),
    }


def sign_state(values, epsilon):
    numeric = np.asarray(values, float)
    return np.select(
        [numeric < -epsilon, numeric > epsilon],
        ["negative", "positive"],
        default="near_zero",
    )


def sign_transition_tables(
    baseline: pd.DataFrame,
    sign_epsilons: Iterable[float],
    reduction_deltas: Iterable[float],
):
    matrices, flips, cases = [], [], []
    clean = baseline.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["da_pre", "da_post", "reduction", "raw_ta_shift"]
    )

    for (model, benchmark), group in clean.groupby(
        ["model", "benchmark"], observed=True
    ):
        for epsilon in sign_epsilons:
            working = group.copy()
            working["pre_sign"] = sign_state(working.da_pre, float(epsilon))
            working["post_sign"] = sign_state(working.da_post, float(epsilon))
            counts = working.groupby(
                ["pre_sign", "post_sign"], observed=True
            ).size()
            n = len(working)

            for pre_sign in ["negative", "near_zero", "positive"]:
                for post_sign in ["negative", "near_zero", "positive"]:
                    count = int(counts.get((pre_sign, post_sign), 0))
                    matrices.append({
                        "model": model,
                        "benchmark": benchmark,
                        "sign_epsilon": float(epsilon),
                        "pre_sign": pre_sign,
                        "post_sign": post_sign,
                        "count": count,
                        "share": count / n,
                    })

            strict = (
                working.pre_sign.isin(["negative", "positive"])
                & working.post_sign.isin(["negative", "positive"])
                & working.pre_sign.ne(working.post_sign)
            )
            working["abs_ratio_post_pre"] = (
                working.da_post.abs()
                / np.maximum(working.da_pre.abs(), 1e-12)
            )

            for delta in reduction_deltas:
                category = np.select(
                    [
                        strict & working.reduction.gt(delta),
                        strict & working.reduction.lt(-delta),
                        strict,
                    ],
                    [
                        "crossed_closer",
                        "crossed_farther",
                        "symmetric_or_near_equal_crossing",
                    ],
                    default="not_strict_flip",
                )
                for name in [
                    "crossed_closer",
                    "crossed_farther",
                    "symmetric_or_near_equal_crossing",
                ]:
                    mask = category == name
                    flips.append({
                        "model": model,
                        "benchmark": benchmark,
                        "sign_epsilon": float(epsilon),
                        "reduction_delta": float(delta),
                        "flip_category": name,
                        "count": int(mask.sum()),
                        "share_all": float(mask.mean()),
                        "share_among_strict_flips": float(
                            mask.sum() / max(strict.sum(), 1)
                        ),
                        "median_abs_ratio_post_pre": float(
                            working.loc[mask, "abs_ratio_post_pre"].median()
                        ) if mask.any() else np.nan,
                    })

                hidden = strict & working.reduction.abs().le(delta)
                flips.append({
                    "model": model,
                    "benchmark": benchmark,
                    "sign_epsilon": float(epsilon),
                    "reduction_delta": float(delta),
                    "flip_category": "strict_flips_hidden_inside_R_near_zero",
                    "count": int(hidden.sum()),
                    "share_all": float(hidden.mean()),
                    "share_among_strict_flips": float(
                        hidden.sum() / max(strict.sum(), 1)
                    ),
                    "median_abs_ratio_post_pre": float(
                        working.loc[hidden, "abs_ratio_post_pre"].median()
                    ) if hidden.any() else np.nan,
                })

            case = working.loc[
                strict,
                KEYS
                + [
                    "model",
                    "benchmark",
                    "da_pre",
                    "da_post",
                    "reduction",
                    "raw_ta_shift",
                    "pre_sign",
                    "post_sign",
                    "abs_ratio_post_pre",
                ],
            ].copy()
            case["sign_epsilon"] = float(epsilon)
            cases.append(case)

    return {
        "sign_transition_matrix": pd.DataFrame(matrices),
        "sign_flip_summary": pd.DataFrame(flips),
        "sign_flip_cases": (
            pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()
        ),
    }
