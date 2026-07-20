from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .auditor_regime import _standardize
from .cfs_regime_robustness import (
    SCORE_RULES,
    _fit_interactions,
    _score,
    run_regime_robustness as _run_base_robustness,
)


def _covid_slope_design(
    frame: pd.DataFrame,
    outcome: str,
    shock_years: set[int],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build an identified score-by-COVID design under full year fixed effects.

    The COVID-period main dummy is a deterministic linear combination of the
    fiscal-year dummies and is therefore omitted. The identified focal parameter
    is the change in the standardized score slope during the configured period.
    """
    work = frame.copy()
    work["score_z"] = _standardize(
        _score(work["abnormal_cfo_proxy"], SCORE_RULES[outcome])
    )
    shock = pd.to_numeric(work["fiscal_year"], errors="coerce").isin(
        shock_years
    ).astype(float)

    x = pd.DataFrame(index=work.index)
    x["intercept"] = 1.0
    x["score_z"] = work["score_z"]
    x["score_x_covid_shock"] = work["score_z"] * shock

    if "lag_assets" in work:
        work["log_lag_assets"] = np.log(
            pd.to_numeric(work["lag_assets"], errors="coerce").clip(lower=1.0)
        )
    for column in settings.get(
        "continuous_controls", ["log_lag_assets", "pre_cfo_scaled"]
    ):
        if column in work:
            x[column] = _standardize(work[column])

    for column in settings.get(
        "covid_fixed_effects", ["fiscal_year", "raw_exchange", "industry_name"]
    ):
        if column not in work:
            continue
        dummies = pd.get_dummies(
            work[column].fillna("UNKNOWN").astype(str),
            prefix=column,
            drop_first=True,
            dtype=float,
        )
        x = pd.concat([x, dummies], axis=1)

    y = pd.to_numeric(work[outcome], errors="coerce")
    valid = y.notna() & x.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    design = pd.concat(
        [work.loc[valid, ["issuer_ticker"]], x.loc[valid]], axis=1
    )
    return design, y.loc[valid].astype(int), ["score_x_covid_shock"]


def identified_covid_interactions(
    sample: pd.DataFrame,
    settings: dict[str, Any],
    outcomes: list[str],
) -> pd.DataFrame:
    covid = settings.get("covid", {})
    windows = {
        "PRIMARY_2020_2021": covid.get("primary_shock_years", [2020, 2021]),
        **covid.get(
            "alternative_shock_windows",
            {
                "COVID_2020_ONLY": [2020],
                "COVID_2020_2021": [2020, 2021],
                "COVID_2020_2022": [2020, 2021, 2022],
            },
        ),
    }
    frames: list[pd.DataFrame] = []
    for label, years in windows.items():
        year_set = {int(value) for value in years}

        def builder(frame: pd.DataFrame, outcome: str):
            return _covid_slope_design(frame, outcome, year_set, settings)

        result = _fit_interactions(
            sample,
            outcomes,
            builder,
            settings,
            "COVID_PERIOD",
            f"{label}:{','.join(map(str, sorted(year_set)))}",
        )
        if not result.empty:
            result["covid_main_dummy_included"] = False
            result["identification_note"] = (
                "COVID main dummy omitted because full fiscal-year fixed effects "
                "absorb period-level intercept differences."
            )
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_regime_robustness(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    tables = _run_base_robustness(cases, settings)
    sample = tables["cfs_regime_robustness_sample"]
    outcomes = [
        outcome
        for outcome in settings.get("outcomes", list(SCORE_RULES))
        if outcome in sample
    ]
    tables["cfs_covid_interactions"] = identified_covid_interactions(
        sample, settings, outcomes
    )
    return tables
