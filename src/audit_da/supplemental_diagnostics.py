from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .results_completion.core import (
    CompletionSettings,
    cluster_bootstrap_1d,
    paired_panel,
    stable_task_seed,
)


KEYS = ["issuer_ticker", "fiscal_year"]


@dataclass(frozen=True)
class SupplementalSettings:
    audited_label: str = "audited"
    unaudited_label: str = "unaudited"
    minimum_year: int = 2016
    maximum_year: int = 2025
    concentration_draws: int = 1000
    concentration_min_active_concepts: int = 2
    concentration_absolute_tolerance: float = 0.0
    concentration_pool_minimum: int = 25
    near_zero_threshold: float = 0.02
    near_zero_distance_bins: int = 10
    near_zero_draws: int = 5000
    bootstrap_draws: int = 2000
    seed: int = 20260723


def normalized_hhi(weights: Sequence[float] | np.ndarray) -> float:
    """Return HHI normalized to [0, 1] conditional on active item count."""
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return np.nan
    shares = values / values.sum()
    if len(shares) == 1:
        return 1.0
    raw = float(np.square(shares).sum())
    floor = 1.0 / len(shares)
    return float(np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0))


def _pair_line_item_long(
    line_items: pd.DataFrame,
    settings: SupplementalSettings,
) -> pd.DataFrame:
    required = set(KEYS + ["audit_status", "concept", "value_numeric"])
    missing = sorted(required - set(line_items.columns))
    if missing:
        raise ValueError(f"Line-item input missing columns: {missing}")

    frame = line_items.copy()
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    frame["value_numeric"] = pd.to_numeric(frame["value_numeric"], errors="coerce")
    frame = frame.loc[
        frame["fiscal_year"].between(settings.minimum_year, settings.maximum_year)
        & frame["audit_status"].isin(
            [settings.audited_label, settings.unaudited_label]
        )
        & frame["concept"].notna()
        & frame["value_numeric"].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby(KEYS + ["audit_status", "concept"], observed=True)[
            "value_numeric"
        ]
        .sum(min_count=1)
        .reset_index()
    )
    pre = (
        grouped.loc[grouped.audit_status.eq(settings.unaudited_label)]
        .drop(columns="audit_status")
        .rename(columns={"value_numeric": "value_pre"})
    )
    post = (
        grouped.loc[grouped.audit_status.eq(settings.audited_label)]
        .drop(columns="audit_status")
        .rename(columns={"value_numeric": "value_post"})
    )
    paired = pre.merge(
        post,
        on=KEYS + ["concept"],
        how="outer",
        validate="one_to_one",
    )
    paired["pre_record_present"] = paired["value_pre"].notna()
    paired["post_record_present"] = paired["value_post"].notna()
    paired["value_pre"] = paired["value_pre"].fillna(0.0)
    paired["value_post"] = paired["value_post"].fillna(0.0)
    paired["delta"] = paired["value_post"] - paired["value_pre"]
    paired["absolute_delta"] = paired["delta"].abs()
    return paired


def concentration_cases(
    line_items: pd.DataFrame,
    settings: SupplementalSettings,
) -> pd.DataFrame:
    """Compare observed line-item concentration with an empirical-share null.

    The null preserves total absolute revision magnitude and active-concept count.
    Positive revision magnitudes are sampled with replacement from the same-year
    empirical pool and normalized to shares before NHHI is recomputed.
    """
    paired = _pair_line_item_long(line_items, settings)
    if paired.empty:
        return pd.DataFrame()

    tolerance = float(settings.concentration_absolute_tolerance)
    active = paired.loc[paired.absolute_delta.gt(tolerance)].copy()
    if active.empty:
        return pd.DataFrame()

    year_pools = {
        int(year): group.absolute_delta.to_numpy(float)
        for year, group in active.groupby("fiscal_year", observed=True)
        if len(group) >= settings.concentration_pool_minimum
    }
    global_pool = active.absolute_delta.to_numpy(float)
    global_pool = global_pool[np.isfinite(global_pool) & (global_pool > 0)]
    if len(global_pool) < settings.concentration_pool_minimum:
        raise ValueError(
            "Insufficient positive line-item revisions for concentration null"
        )

    rows: list[dict[str, object]] = []
    for (issuer, year), group in active.groupby(KEYS, observed=True, sort=True):
        magnitudes = group.absolute_delta.to_numpy(float)
        active_count = int(len(magnitudes))
        if active_count < int(settings.concentration_min_active_concepts):
            continue
        total = float(magnitudes.sum())
        if not np.isfinite(total) or total <= 0:
            continue

        observed = normalized_hhi(magnitudes)
        pool = year_pools.get(int(year), global_pool)
        rng = np.random.default_rng(
            stable_task_seed(settings.seed, "concentration", issuer, int(year))
        )
        sampled = rng.choice(
            pool,
            size=(int(settings.concentration_draws), active_count),
            replace=True,
        )
        denominators = sampled.sum(axis=1, keepdims=True)
        shares = np.divide(
            sampled,
            denominators,
            out=np.full_like(sampled, np.nan),
            where=denominators > 0,
        )
        raw_hhi = np.square(shares).sum(axis=1)
        floor = 1.0 / active_count
        simulated = np.clip((raw_hhi - floor) / (1.0 - floor), 0.0, 1.0)
        expected = float(np.nanmean(simulated))

        rows.append(
            {
                "issuer_ticker": issuer,
                "fiscal_year": int(year),
                "active_concepts": active_count,
                "total_absolute_revision": total,
                "observed_nhhi": observed,
                "expected_nhhi": expected,
                "expected_nhhi_p025": float(np.nanquantile(simulated, 0.025)),
                "expected_nhhi_p975": float(np.nanquantile(simulated, 0.975)),
                "excess_nhhi": float(observed - expected),
                "null_draws": int(settings.concentration_draws),
                "null_pool": (
                    "same_fiscal_year"
                    if int(year) in year_pools
                    else "all_years_fallback"
                ),
                "pre_only_concepts": int((~group.pre_record_present).sum()),
                "post_only_concepts": int((~group.post_record_present).sum()),
            }
        )
    return pd.DataFrame(rows)


def _distance_bins(
    values: pd.Series,
    years: pd.Series,
    bins: int,
) -> pd.Series:
    output = pd.Series(pd.NA, index=values.index, dtype="Int64")
    for _, indices in years.groupby(years, sort=False).groups.items():
        current = pd.to_numeric(values.loc[indices], errors="coerce")
        finite = current[np.isfinite(current)]
        if finite.empty:
            continue
        quantiles = min(int(bins), int(finite.nunique()))
        if quantiles < 1:
            continue
        if quantiles == 1:
            output.loc[finite.index] = 0
            continue
        ranked = finite.rank(method="first")
        output.loc[finite.index] = pd.qcut(
            ranked,
            q=quantiles,
            labels=False,
            duplicates="drop",
        ).astype("Int64")
    return output


def near_zero_cfo_cases(
    panel: pd.DataFrame,
    settings: SupplementalSettings,
) -> pd.DataFrame:
    """Build paired near-zero CFO cases with matched absolute-distance bins."""
    completion = CompletionSettings(
        audited_label=settings.audited_label,
        unaudited_label=settings.unaudited_label,
    )
    pair = paired_panel(panel, completion)
    required = ["cfo_pre", "cfo_post", "lag_assets_pre", "fiscal_year"]
    missing = sorted(set(required) - set(pair.columns))
    if missing:
        raise ValueError(f"Panel missing near-zero CFO columns: {missing}")

    pair = pair.copy()
    for column in required:
        pair[column] = pd.to_numeric(pair[column], errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    pair = pair.loc[
        pair.fiscal_year.between(settings.minimum_year, settings.maximum_year)
        & pair.lag_assets_pre.abs().gt(0)
    ].copy()
    if pair.empty:
        return pd.DataFrame()

    pair["cfo_pre_scaled"] = pair.cfo_pre / pair.lag_assets_pre.abs()
    pair["cfo_post_scaled"] = pair.cfo_post / pair.lag_assets_pre.abs()
    pair["abs_cfo_pre"] = pair.cfo_pre_scaled.abs()
    pair["abs_cfo_post"] = pair.cfo_post_scaled.abs()
    pair = pair.loc[
        pair[["abs_cfo_pre", "abs_cfo_post"]]
        .max(axis=1)
        .le(float(settings.near_zero_threshold))
    ].copy()
    if pair.empty:
        return pd.DataFrame()

    pooled_values = pd.concat(
        [
            pair[["fiscal_year", "abs_cfo_pre"]]
            .rename(columns={"abs_cfo_pre": "absolute_distance"})
            .assign(state="pre", pair_index=pair.index),
            pair[["fiscal_year", "abs_cfo_post"]]
            .rename(columns={"abs_cfo_post": "absolute_distance"})
            .assign(state="audited", pair_index=pair.index),
        ],
        ignore_index=True,
    )
    pooled_values["distance_bin"] = _distance_bins(
        pooled_values.absolute_distance,
        pooled_values.fiscal_year,
        settings.near_zero_distance_bins,
    )
    bins = pooled_values.pivot(
        index="pair_index",
        columns="state",
        values="distance_bin",
    )
    pair["distance_bin_pre"] = bins.get("pre")
    pair["distance_bin_post"] = bins.get("audited")
    pair["matched_distance_bin"] = (
        pair.distance_bin_pre.notna()
        & pair.distance_bin_post.notna()
        & pair.distance_bin_pre.eq(pair.distance_bin_post)
    )
    pair = pair.loc[pair.matched_distance_bin].copy()
    if pair.empty:
        return pd.DataFrame()

    pair["positive_pre"] = pair.cfo_pre_scaled.gt(0).astype(int)
    pair["positive_post"] = pair.cfo_post_scaled.gt(0).astype(int)
    pair["sign_shift"] = pair.positive_post - pair.positive_pre
    pair["crossed_zero"] = pair.sign_shift.ne(0)
    pair["crossing_direction"] = np.select(
        [pair.sign_shift.gt(0), pair.sign_shift.lt(0)],
        ["negative_to_positive", "positive_to_negative"],
        default="no_crossing",
    )
    pair["near_zero_threshold"] = float(settings.near_zero_threshold)
    pair["distance_bins"] = int(settings.near_zero_distance_bins)
    return pair[
        KEYS
        + [
            "cfo_pre_scaled",
            "cfo_post_scaled",
            "abs_cfo_pre",
            "abs_cfo_post",
            "distance_bin_pre",
            "distance_bin_post",
            "positive_pre",
            "positive_post",
            "sign_shift",
            "crossed_zero",
            "crossing_direction",
            "near_zero_threshold",
            "distance_bins",
        ]
    ].reset_index(drop=True)


def near_zero_randomisation(
    cases: pd.DataFrame,
    settings: SupplementalSettings,
) -> pd.DataFrame:
    """Within-pair state-swap randomisation for mean positive-sign shift."""
    if cases.empty:
        return pd.DataFrame(columns=["draw", "statistic"])
    shifts = pd.to_numeric(cases["sign_shift"], errors="coerce").dropna().to_numpy(float)
    if len(shifts) == 0:
        return pd.DataFrame(columns=["draw", "statistic"])

    observed = float(np.mean(shifts))
    rng = np.random.default_rng(stable_task_seed(settings.seed, "near_zero"))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(int(settings.near_zero_draws), len(shifts)),
        replace=True,
    )
    simulated = (signs * shifts).mean(axis=1)
    return pd.DataFrame(
        {
            "draw": ["observed", *range(1, int(settings.near_zero_draws) + 1)],
            "statistic": np.concatenate([[observed], simulated]),
        }
    )


def supplemental_summary(
    concentration: pd.DataFrame,
    near_zero_cases_frame: pd.DataFrame,
    near_zero_draws: pd.DataFrame,
    settings: SupplementalSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not concentration.empty:
        boot = cluster_bootstrap_1d(
            concentration.excess_nhhi,
            concentration.issuer_ticker,
            statistic="mean",
            draws=int(settings.bootstrap_draws),
            seed=stable_task_seed(settings.seed, "concentration_bootstrap"),
            null=0.0,
        )
        rows.append(
            {
                "diagnostic": "cfs_revision_excess_concentration",
                **boot,
                "n": int(len(concentration)),
                "issuers": int(concentration.issuer_ticker.nunique()),
                "share_positive": float(concentration.excess_nhhi.gt(0).mean()),
                "design": "empirical_share_reallocation",
            }
        )

    if not near_zero_draws.empty:
        observed = float(
            near_zero_draws.loc[
                near_zero_draws.draw.astype(str).eq("observed"), "statistic"
            ].iloc[0]
        )
        simulated = pd.to_numeric(
            near_zero_draws.loc[
                ~near_zero_draws.draw.astype(str).eq("observed"), "statistic"
            ],
            errors="coerce",
        ).dropna().to_numpy(float)
        p_two_sided = float(
            (1 + np.sum(np.abs(simulated) >= abs(observed)))
            / (len(simulated) + 1)
        )
        rows.append(
            {
                "diagnostic": "near_zero_cfo_positive_sign_shift",
                "estimate": observed,
                "ci_low": float(np.quantile(simulated, 0.025)),
                "ci_high": float(np.quantile(simulated, 0.975)),
                "p_directional": p_two_sided,
                "n": int(len(near_zero_cases_frame)),
                "issuers": int(
                    near_zero_cases_frame.issuer_ticker.nunique()
                    if not near_zero_cases_frame.empty
                    else 0
                ),
                "share_positive": float(
                    near_zero_cases_frame.sign_shift.gt(0).mean()
                    if not near_zero_cases_frame.empty
                    else np.nan
                ),
                "design": "within_pair_state_swap_same_distance_bin",
            }
        )
    return pd.DataFrame(rows)
