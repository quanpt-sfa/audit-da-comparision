from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import KEYS, paired_panel, trimmed_mean
from .diag_decomposition import build_decomposition_panel


def _quantile_bins(
    values: pd.Series,
    groups: pd.Series,
    bins: int,
) -> pd.Series:
    if bins <= 0:
        return pd.Series(0, index=values.index, dtype="Int64")

    def one_group(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna()
        output = pd.Series(-1, index=series.index, dtype="Int64")
        if valid.sum() < 2:
            return output
        try:
            cut = pd.qcut(
                numeric.loc[valid],
                q=min(bins, int(valid.sum())),
                labels=False,
                duplicates="drop",
            )
            output.loc[valid] = cut.astype("Int64")
        except ValueError:
            output.loc[valid] = 0
        return output

    return values.groupby(groups, group_keys=False).apply(one_group)


def _build_strata(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    component_movement: pd.Series,
) -> pd.Series:
    within_columns = [
        c for c in settings.get("within", ["fiscal_year"])
        if c in frame.columns
    ]
    if within_columns:
        within_key = (
            frame[within_columns]
            .fillna("__MISSING__")
            .astype(str)
            .agg("|".join, axis=1)
        )
    else:
        within_key = pd.Series("__ALL__", index=frame.index)

    size_bins = _quantile_bins(
        frame["lag_assets_common"].abs(),
        within_key,
        int(settings.get("size_bins", 0)),
    )
    da_bins = _quantile_bins(
        frame["da_pre"].abs(),
        within_key,
        int(settings.get("abs_da_pre_bins", 0)),
    )
    component_bins = _quantile_bins(
        component_movement.abs(),
        within_key,
        int(settings.get("abs_component_bins", 0)),
    )
    key = (
        within_key.astype(str)
        + "|s" + size_bins.astype(str)
        + "|d" + da_bins.astype(str)
        + "|m" + component_bins.astype(str)
    )
    minimum = int(settings.get("minimum_stratum_size", 20))
    counts = key.value_counts()
    return key.where(key.map(counts).ge(minimum), within_key + "|POOLED")


def _permute_within(
    values: np.ndarray,
    codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty_like(values)
    for code in np.unique(codes):
        index = np.flatnonzero(codes == code)
        output[index] = values[index][rng.permutation(len(index))]
    return output


def _reduction(anchor: np.ndarray, movement: np.ndarray) -> np.ndarray:
    return np.abs(anchor) - np.abs(anchor + movement)


def build_component_alignment_panel(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = build_decomposition_panel(baseline, panel)
    pair = paired_panel(panel)
    anchor_columns = [
        c for c in [
            "issuer_ticker",
            "fiscal_year",
            "ta_scaled_pre",
            "ta_balance_sheet_pre",
            "lag_assets_pre",
            "raw_exchange_pre",
        ]
        if c in pair.columns
    ]
    anchor = pair[anchor_columns].copy().rename(
        columns={"raw_exchange_pre": "raw_exchange_anchor"}
    )
    rows = rows.merge(
        anchor,
        on=KEYS,
        how="left",
        validate="many_to_one",
        suffixes=("", "_anchor"),
    )
    if "raw_exchange" not in rows and "raw_exchange_anchor" in rows:
        rows["raw_exchange"] = rows["raw_exchange_anchor"]

    rows["implied_nda_pre"] = (
        pd.to_numeric(rows["ta_scaled_pre"], errors="coerce")
        - pd.to_numeric(rows["da_pre"], errors="coerce")
    )
    lag = pd.to_numeric(rows["lag_assets_common"], errors="coerce")
    rows["ta_balance_sheet_scaled_pre"] = (
        pd.to_numeric(rows.get("ta_balance_sheet_pre"), errors="coerce") / lag
    )
    rows["da_pre_balance_sheet_anchor"] = (
        rows["ta_balance_sheet_scaled_pre"] - rows["implied_nda_pre"]
    )
    rows["pat_component_movement"] = rows["delta_pat_scaled"]
    rows["cfo_component_movement"] = -rows["delta_cfo_scaled"]

    for component in ("pat", "cfo"):
        movement = pd.to_numeric(
            rows[f"{component}_component_movement"], errors="coerce"
        )
        rows[f"{component}_reduction_cashflow_anchor"] = (
            rows["da_pre"].abs() - (rows["da_pre"] + movement).abs()
        )
        rows[f"{component}_reduction_balance_sheet_anchor"] = (
            rows["da_pre_balance_sheet_anchor"].abs()
            - (rows["da_pre_balance_sheet_anchor"] + movement).abs()
        )
    return rows


def component_placebo_tables(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    rows = build_component_alignment_panel(baseline, panel)
    models = settings.get("models")
    benchmarks = settings.get("benchmarks")
    if models:
        rows = rows[rows["model"].isin(models)]
    if benchmarks:
        rows = rows[rows["benchmark"].isin(benchmarks)]

    components = list(settings.get("components", ["pat", "cfo"]))
    permutations = int(settings["permutations"])
    trim_fraction = float(settings["trim_fraction"])
    rng = np.random.default_rng(int(settings["random_seed"]))
    strata_settings = dict(settings.get("conditioning_bins", {}))
    strata_settings["minimum_stratum_size"] = int(
        settings.get("minimum_stratum_size", 20)
    )

    summary_records: list[dict[str, Any]] = []
    draw_records: list[dict[str, Any]] = []
    anchor_records: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []

    for (model, benchmark), specification in rows.groupby(
        ["model", "benchmark"], observed=True
    ):
        for component in components:
            movement_column = f"{component}_component_movement"
            real_column = f"{component}_reduction_cashflow_anchor"
            required = [
                "da_pre",
                movement_column,
                real_column,
                "lag_assets_common",
                "fiscal_year",
            ]
            finite = np.ones(len(specification), dtype=bool)
            for column in required:
                finite &= np.isfinite(
                    pd.to_numeric(
                        specification[column], errors="coerce"
                    ).to_numpy(float)
                )
            sample = specification.loc[finite].copy()
            if sample.empty:
                continue

            movement = pd.to_numeric(sample[movement_column], errors="coerce")
            anchor_cf = pd.to_numeric(
                sample["da_pre"], errors="coerce"
            ).to_numpy(float)
            raw = movement.to_numpy(float)
            real = _reduction(anchor_cf, raw)
            sample["component"] = component
            sample["component_reduction_real"] = real
            sample["component_alignment_state"] = np.select(
                [real > 0, real < 0],
                ["toward_zero", "away_from_zero"],
                default="no_absolute_change",
            )
            case_frames.append(sample)

            strata_key = _build_strata(
                sample,
                settings=strata_settings,
                component_movement=movement,
            )
            codes, _ = pd.factorize(strata_key, sort=True)
            centered = (
                movement - movement.groupby(strata_key).transform("mean")
            ).to_numpy(float)

            real_mean = float(real.mean())
            real_trimmed = trimmed_mean(real, trim_fraction)
            real_positive = float((real > 0).mean())
            real_negative = float((real < 0).mean())

            anchor_bs = pd.to_numeric(
                sample["da_pre_balance_sheet_anchor"], errors="coerce"
            ).to_numpy(float)
            bs_finite = np.isfinite(anchor_bs)
            bs_reduction = _reduction(anchor_bs[bs_finite], raw[bs_finite])
            anchor_records.append({
                "model": model,
                "benchmark": benchmark,
                "component": component,
                "cashflow_anchor_rows": len(sample),
                "balance_sheet_anchor_rows": int(bs_finite.sum()),
                "mean_reduction_cashflow_anchor": real_mean,
                "trimmed_mean_reduction_cashflow_anchor": real_trimmed,
                "positive_minus_negative_share_cashflow_anchor": (
                    real_positive - real_negative
                ),
                "mean_reduction_balance_sheet_anchor": (
                    float(bs_reduction.mean()) if len(bs_reduction) else np.nan
                ),
                "trimmed_mean_reduction_balance_sheet_anchor": (
                    trimmed_mean(bs_reduction, trim_fraction)
                    if len(bs_reduction) else np.nan
                ),
                "positive_minus_negative_share_balance_sheet_anchor": (
                    float((bs_reduction > 0).mean() - (bs_reduction < 0).mean())
                    if len(bs_reduction) else np.nan
                ),
                "correlation_movement_with_negative_cashflow_anchor": (
                    float(np.corrcoef(raw, -anchor_cf)[0, 1])
                    if len(raw) > 1 else np.nan
                ),
                "correlation_movement_with_negative_balance_sheet_anchor": (
                    float(np.corrcoef(raw[bs_finite], -anchor_bs[bs_finite])[0, 1])
                    if bs_finite.sum() > 1 else np.nan
                ),
            })

            for placebo_type in (
                "raw_permutation",
                "centered_permutation",
                "symmetric_sign",
            ):
                mean_draws = np.empty(permutations)
                trimmed_draws = np.empty(permutations)
                positive_draws = np.empty(permutations)
                negative_draws = np.empty(permutations)

                for draw in range(permutations):
                    if placebo_type == "raw_permutation":
                        placebo_movement = _permute_within(raw, codes, rng)
                    elif placebo_type == "centered_permutation":
                        placebo_movement = _permute_within(centered, codes, rng)
                    else:
                        magnitude = _permute_within(
                            np.abs(centered), codes, rng
                        )
                        placebo_movement = magnitude * rng.choice(
                            [-1.0, 1.0], size=len(magnitude)
                        )

                    placebo_reduction = _reduction(anchor_cf, placebo_movement)
                    mean_draws[draw] = placebo_reduction.mean()
                    trimmed_draws[draw] = trimmed_mean(
                        placebo_reduction, trim_fraction
                    )
                    positive_draws[draw] = (placebo_reduction > 0).mean()
                    negative_draws[draw] = (placebo_reduction < 0).mean()
                    draw_records.append({
                        "model": model,
                        "benchmark": benchmark,
                        "component": component,
                        "placebo_type": placebo_type,
                        "permutation": draw,
                        "mean_component_reduction": mean_draws[draw],
                        "trimmed_mean_component_reduction": trimmed_draws[draw],
                        "share_positive": positive_draws[draw],
                        "share_negative": negative_draws[draw],
                    })

                summary_records.append({
                    "model": model,
                    "benchmark": benchmark,
                    "component": component,
                    "placebo_type": placebo_type,
                    "rows": len(sample),
                    "strata": int(pd.Series(codes).nunique()),
                    "real_mean_component_reduction": real_mean,
                    "real_trimmed_mean_component_reduction": real_trimmed,
                    "real_positive_minus_negative_share": (
                        real_positive - real_negative
                    ),
                    "placebo_mean": float(mean_draws.mean()),
                    "placebo_q025": float(np.quantile(mean_draws, .025)),
                    "placebo_q975": float(np.quantile(mean_draws, .975)),
                    "placebo_trimmed_mean": float(trimmed_draws.mean()),
                    "placebo_trimmed_q025": float(
                        np.quantile(trimmed_draws, .025)
                    ),
                    "placebo_trimmed_q975": float(
                        np.quantile(trimmed_draws, .975)
                    ),
                    "placebo_positive_minus_negative_share": float(
                        (positive_draws - negative_draws).mean()
                    ),
                    "component_corrective_excess_mean": (
                        real_mean - float(mean_draws.mean())
                    ),
                    "component_corrective_excess_trimmed": (
                        real_trimmed - float(trimmed_draws.mean())
                    ),
                    "randomization_p_ge_real": float(
                        (1 + (mean_draws >= real_mean).sum())
                        / (permutations + 1)
                    ),
                    "randomization_p_trimmed_ge_real": float(
                        (1 + (trimmed_draws >= real_trimmed).sum())
                        / (permutations + 1)
                    ),
                    "share_placebo_mean_negative": float(
                        (mean_draws < 0).mean()
                    ),
                    "all_outputs_finite": bool(
                        np.isfinite(mean_draws).all()
                        and np.isfinite(trimmed_draws).all()
                    ),
                })

    cases = (
        pd.concat(case_frames, ignore_index=True)
        if case_frames else pd.DataFrame()
    )
    return {
        "component_placebo_summary": pd.DataFrame(summary_records),
        "component_placebo_draws": pd.DataFrame(draw_records),
        "component_anchor_diagnostics": pd.DataFrame(anchor_records),
        "component_alignment_cases": cases,
    }
