from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _baseline_rows(stacked: pd.DataFrame) -> pd.DataFrame:
    baseline = stacked[stacked["event_time"].eq(-1)].copy()
    return baseline.drop_duplicates(["event_id", "issuer_ticker"])


def estimate_overlap_weights(
    stacked: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("dynamic_did", settings)
    baseline = _baseline_rows(stacked)
    if baseline.empty:
        return pd.DataFrame(), pd.DataFrame()
    numeric_covariates = [
        column
        for column in cfg.get(
            "overlap_numeric_covariates",
            [
                "lag_assets_common",
                "lag_assets",
                "pre_cfo_scaled",
                "pre_event_borrowing_intensity",
                "signed_cfo_correction",
            ],
        )
        if column in baseline.columns
    ]
    categorical_covariates = [
        column
        for column in cfg.get(
            "overlap_categorical_covariates",
            ["cohort_year", "raw_exchange", "industry_name"],
        )
        if column in baseline.columns
    ]
    x = pd.DataFrame(index=baseline.index)
    for column in numeric_covariates:
        values = _numeric(baseline, column)
        if "assets" in column:
            values = np.log(values.clip(lower=1.0))
        median = values.median()
        values = values.fillna(median if np.isfinite(median) else 0.0)
        sd = values.std(ddof=0)
        x[column] = (
            (values - values.mean()) / sd
            if np.isfinite(sd) and sd > 0
            else 0.0
        )
    if categorical_covariates:
        dummies = pd.get_dummies(
            baseline[categorical_covariates].fillna("UNKNOWN").astype(str),
            prefix=categorical_covariates,
            drop_first=True,
            dtype=float,
        )
        x = pd.concat([x, dummies], axis=1)
    y = baseline["treated"].astype(int)
    minimum_treated = int(cfg.get("minimum_overlap_treated", 20))
    minimum_controls = int(cfg.get("minimum_overlap_controls", 50))
    if (
        y.sum() < minimum_treated
        or y.eq(0).sum() < minimum_controls
        or x.empty
    ):
        baseline["propensity"] = y.mean()
        baseline["overlap_weight"] = 1.0
        status = "FALLBACK_UNWEIGHTED_INSUFFICIENT_SAMPLE"
    else:
        model = LogisticRegression(
            C=float(cfg.get("propensity_c", 1.0)),
            max_iter=int(cfg.get("propensity_max_iter", 1000)),
            solver="lbfgs",
        )
        model.fit(
            x,
            y,
            sample_weight=pd.to_numeric(
                baseline.get(
                    "stack_weight", pd.Series(1.0, index=baseline.index)
                ),
                errors="coerce",
            ).fillna(1.0),
        )
        propensity = np.clip(
            model.predict_proba(x)[:, 1],
            float(cfg.get("propensity_clip_lower", 0.01)),
            float(cfg.get("propensity_clip_upper", 0.99)),
        )
        baseline["propensity"] = propensity
        baseline["overlap_weight"] = np.where(
            y.eq(1), 1.0 - propensity, propensity
        )
        status = "ESTIMATED"
    baseline["did_weight"] = baseline["overlap_weight"] * pd.to_numeric(
        baseline.get("stack_weight", pd.Series(1.0, index=baseline.index)),
        errors="coerce",
    ).fillna(1.0)

    diagnostics: list[dict[str, Any]] = []
    for column in numeric_covariates:
        values = _numeric(baseline, column)
        for weighted in (False, True):
            weights = (
                baseline["did_weight"]
                if weighted
                else pd.Series(1.0, index=baseline.index)
            )
            means = {}
            for group in (0, 1):
                mask = baseline["treated"].eq(group) & values.notna()
                means[group] = (
                    float(
                        np.average(
                            values.loc[mask], weights=weights.loc[mask]
                        )
                    )
                    if mask.any()
                    else np.nan
                )
            pooled_sd = values.std(ddof=0)
            smd = (
                (means[1] - means[0]) / pooled_sd
                if np.isfinite(pooled_sd) and pooled_sd > 0
                else np.nan
            )
            diagnostics.append(
                {
                    "variable": column,
                    "weighted": weighted,
                    "treated_mean": means[1],
                    "control_mean": means[0],
                    "standardized_mean_difference": smd,
                }
            )
    baseline["overlap_status"] = status
    return baseline[
        [
            "event_id",
            "issuer_ticker",
            "switch_direction",
            "treated",
            "propensity",
            "overlap_weight",
            "did_weight",
            "overlap_status",
        ]
    ], pd.DataFrame(diagnostics)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    return (
        float(np.average(values.loc[valid], weights=weights.loc[valid]))
        if valid.any()
        else np.nan
    )


def build_dynamic_did_contrasts(
    stacked: pd.DataFrame,
    weights: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    cfg = settings.get("dynamic_did", settings)
    outcomes = list(
        cfg.get(
            "outcomes",
            [
                "any_candidate",
                "cff_down_candidate",
                "cfi_up_candidate",
                "signed_cfo_correction",
                "absolute_cfo_correction",
            ],
        )
    )
    horizons = [int(value) for value in cfg.get("horizons", [-2, 0, 1, 2])]
    frame = stacked.merge(
        weights,
        on=["event_id", "issuer_ticker", "switch_direction", "treated"],
        how="left",
        validate="many_to_one",
    )
    frame["did_weight"] = frame["did_weight"].fillna(
        pd.to_numeric(
            frame.get("stack_weight", pd.Series(1.0, index=frame.index)),
            errors="coerce",
        ).fillna(1.0)
    )
    rows: list[dict[str, Any]] = []
    for (event_id, direction), event_frame in frame.groupby(
        ["event_id", "switch_direction"], observed=True
    ):
        treated_rows = event_frame.loc[
            event_frame["treated"].eq(1), "issuer_ticker"
        ]
        if treated_rows.empty:
            continue
        treated_issuer = str(treated_rows.iloc[0])
        cohort_year = int(event_frame["cohort_year"].iloc[0])
        for outcome in outcomes:
            if outcome not in event_frame.columns:
                continue
            wide = event_frame.pivot_table(
                index=["issuer_ticker", "treated", "did_weight"],
                columns="event_time",
                values=outcome,
                aggfunc="first",
            ).reset_index()
            if -1 not in wide.columns:
                continue
            for horizon in horizons:
                if horizon not in wide.columns:
                    continue
                delta = pd.to_numeric(
                    wide[horizon], errors="coerce"
                ) - pd.to_numeric(wide[-1], errors="coerce")
                treated_mask = wide["treated"].eq(1)
                control_mask = wide["treated"].eq(0)
                treated_change = _weighted_mean(
                    delta.loc[treated_mask],
                    wide.loc[treated_mask, "did_weight"],
                )
                control_change = _weighted_mean(
                    delta.loc[control_mask],
                    wide.loc[control_mask, "did_weight"],
                )
                rows.append(
                    {
                        "event_id": event_id,
                        "treated_issuer": treated_issuer,
                        "switch_direction": direction,
                        "cohort_year": cohort_year,
                        "outcome": outcome,
                        "horizon": horizon,
                        "treated_change": treated_change,
                        "control_change": control_change,
                        "event_att": treated_change - control_change,
                        "controls": int(control_mask.sum()),
                        "status": "OK"
                        if np.isfinite(treated_change)
                        and np.isfinite(control_change)
                        else "INCOMPLETE_OUTCOME_SUPPORT",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_dynamic_did(
    contrasts: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("dynamic_did", settings)
    repetitions = int(cfg.get("bootstrap_repetitions", 500))
    seed = int(cfg.get("bootstrap_seed", 240721))
    minimum_events = int(cfg.get("minimum_events", 20))
    if contrasts.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    valid = contrasts[
        contrasts["status"].eq("OK")
        & pd.to_numeric(contrasts["event_att"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    estimate_rows: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    for keys, group in valid.groupby(
        ["switch_direction", "outcome", "horizon"], observed=True
    ):
        direction, outcome, horizon = keys
        issuers = group["treated_issuer"].astype(str).unique()
        point = float(group["event_att"].mean())
        draws: list[float] = []
        if len(issuers) >= 2:
            by_issuer = {
                str(issuer): part
                for issuer, part in group.groupby(
                    "treated_issuer", observed=True
                )
            }
            for _ in range(repetitions):
                selected = rng.choice(
                    issuers, size=len(issuers), replace=True
                )
                draw = pd.concat(
                    [by_issuer[str(issuer)] for issuer in selected],
                    ignore_index=True,
                )
                draws.append(float(draw["event_att"].mean()))
        values = pd.Series(draws, dtype=float)
        estimate_rows.append(
            {
                "switch_direction": direction,
                "outcome": outcome,
                "horizon": int(horizon),
                "estimate": point,
                "bootstrap_se": float(values.std(ddof=1))
                if len(values) > 1
                else np.nan,
                "ci_lower_95": float(values.quantile(0.025))
                if len(values)
                else np.nan,
                "ci_upper_95": float(values.quantile(0.975))
                if len(values)
                else np.nan,
                "events": int(group["event_id"].nunique()),
                "treated_issuers": len(issuers),
                "mean_controls_per_event": float(group["controls"].mean()),
                "status": "OK"
                if len(issuers) >= minimum_events
                else "LOW_EVENT_SUPPORT",
            }
        )
        for draw_index, value in enumerate(draws):
            bootstrap_records.append(
                {
                    "switch_direction": direction,
                    "outcome": outcome,
                    "horizon": int(horizon),
                    "draw": draw_index,
                    "estimate": value,
                }
            )

    estimates = pd.DataFrame(estimate_rows)
    bootstrap = pd.DataFrame(bootstrap_records)
    tests: list[dict[str, Any]] = []
    if not bootstrap.empty:
        for (direction, outcome), group in estimates.groupby(
            ["switch_direction", "outcome"], observed=True
        ):
            lead_horizons = sorted(
                group.loc[group["horizon"].lt(-1), "horizon"].unique()
            )
            if not lead_horizons:
                continue
            point = (
                group.set_index("horizon")
                .loc[lead_horizons, "estimate"]
                .to_numpy(float)
            )
            wide = bootstrap[
                bootstrap["switch_direction"].eq(direction)
                & bootstrap["outcome"].eq(outcome)
                & bootstrap["horizon"].isin(lead_horizons)
            ].pivot(index="draw", columns="horizon", values="estimate")
            wide = wide.reindex(columns=lead_horizons).dropna()
            if wide.empty:
                continue
            covariance = np.cov(wide.to_numpy(float), rowvar=False)
            covariance = np.atleast_2d(covariance)
            statistic = float(
                point.T @ np.linalg.pinv(covariance) @ point
            )
            tests.append(
                {
                    "switch_direction": direction,
                    "outcome": outcome,
                    "test": "JOINT_PLACEBO_PRETREND",
                    "horizons": ",".join(map(str, lead_horizons)),
                    "chi_square": statistic,
                    "df": len(lead_horizons),
                    "p_value": float(
                        chi2.sf(statistic, len(lead_horizons))
                    ),
                    "bootstrap_draws": len(wide),
                }
            )
    return estimates, pd.DataFrame(tests), bootstrap


def run_switch_dynamic_did(
    stacked: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    weights, balance = estimate_overlap_weights(stacked, settings)
    contrasts = build_dynamic_did_contrasts(stacked, weights, settings)
    estimates, pretrend, bootstrap = aggregate_dynamic_did(
        contrasts, settings
    )
    successful = (
        not estimates.empty
        and "status" in estimates.columns
        and estimates["status"].eq("OK").any()
    )
    status = pd.DataFrame(
        [
            {
                "gate": "auditor_switch_dynamic_did",
                "status": "PASS" if successful else "PARTIALLY_EVALUATED",
                "stacked_events": int(stacked["event_id"].nunique())
                if not stacked.empty
                else 0,
                "contrasts": len(contrasts),
                "estimated_cells": len(estimates),
                "interpretation": (
                    "Switcher-versus-stayer dynamic DiD around reversible "
                    "auditor-tier transitions; causal interpretation requires "
                    "parallel trends and no anticipation."
                ),
            }
        ]
    )
    return {
        "cfs_auditor_switch_overlap_weights": weights,
        "cfs_auditor_switch_overlap_balance": balance,
        "cfs_auditor_switch_dynamic_did_event_contrasts": contrasts,
        "cfs_auditor_switch_dynamic_did": estimates,
        "cfs_auditor_switch_dynamic_did_pretrend": pretrend,
        "cfs_auditor_switch_dynamic_did_bootstrap": bootstrap,
        "cfs_auditor_switch_dynamic_did_status": status,
    }
