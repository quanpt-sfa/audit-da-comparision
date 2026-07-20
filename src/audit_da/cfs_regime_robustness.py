from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from .auditor_regime import _ap, _auc, _fit_logit_clustered, _standardize
from .diag_common import KEYS


DEFAULT_OUTCOMES = (
    "any_candidate",
    "audited_cfo_decrease",
    "audited_cfo_increase",
    "cff_down_candidate",
    "cfi_up_candidate",
)

SCORE_RULES = {
    "any_candidate": "absolute",
    "audited_cfo_decrease": "positive",
    "audited_cfo_increase": "negative",
    "cff_down_candidate": "positive",
    "cfi_up_candidate": "negative",
}


def _score(values: pd.Series, rule: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if rule == "absolute":
        return numeric.abs()
    if rule == "negative":
        return -numeric
    return numeric


def normalize_exchange(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    compact = text.replace(" ", "").replace("-", "")
    if compact in {"HOSE", "HSX", "HOCHIMINH", "HOCHIMINHSTOCKEXCHANGE"}:
        return "HOSE"
    if compact in {"HNX", "HANOI", "HANOISTOCKEXCHANGE"}:
        return "HNX"
    if compact in {"UPCOM", "UPCOMMARKET"}:
        return "UPCOM"
    return "UNKNOWN"


def prepare_regime_sample(cases: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    model = settings.get("proxy_model", "earnings_working_capital")
    sample_mode = settings.get("sample_mode", "common_primary_models")
    restriction = settings.get("sample_restriction", "analysis_core")
    frame = cases.copy()
    if "proxy_model" in frame:
        frame = frame[frame["proxy_model"].eq(model)]
    if "sample_mode" in frame:
        frame = frame[frame["sample_mode"].eq(sample_mode)]
    if "sample_restriction" in frame:
        frame = frame[frame["sample_restriction"].eq(restriction)]
    frame = frame.drop_duplicates(KEYS).copy()
    if "raw_exchange" not in frame:
        raise ValueError("Common-primary case table has no raw_exchange column")
    frame["exchange_group"] = frame["raw_exchange"].map(normalize_exchange)
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")
    return frame


def _metrics(frame: pd.DataFrame, outcome: str) -> dict[str, Any]:
    if frame.empty or outcome not in frame:
        return {
            "rows": 0,
            "issuers": 0,
            "positives": 0,
            "prevalence": np.nan,
            "auc": np.nan,
            "average_precision": np.nan,
            "top_decile_rate": np.nan,
            "top_decile_lift": np.nan,
        }
    y = pd.to_numeric(frame[outcome], errors="coerce")
    score = _score(frame["abnormal_cfo_proxy"], SCORE_RULES[outcome])
    valid = y.notna() & score.notna()
    work = frame.loc[valid].copy()
    y = y.loc[valid].astype(int)
    score = score.loc[valid]
    prevalence = float(y.mean()) if len(y) else np.nan
    within_year_rank = score.groupby(work["fiscal_year"]).rank(
        pct=True, method="average"
    )
    top = within_year_rank.ge(0.90)
    top_rate = float(y.loc[top].mean()) if top.any() else np.nan
    return {
        "rows": len(work),
        "issuers": int(work["issuer_ticker"].nunique()),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auc": _auc(y.to_numpy(), score.to_numpy(float)),
        "average_precision": _ap(y.to_numpy(), score.to_numpy(float)),
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / prevalence
        if prevalence > 0 and np.isfinite(top_rate)
        else np.nan,
    }


def grouped_metrics(
    sample: pd.DataFrame,
    group_column: str,
    groups: list[str],
    outcomes: list[str],
    group_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for group in groups:
            subset = sample[sample[group_column].eq(group)]
            rows.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "outcome": outcome,
                    "score_rule": SCORE_RULES[outcome],
                    **_metrics(subset, outcome),
                }
            )
    return pd.DataFrame(rows)


def pairwise_differences(
    metrics: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    indexed = metrics.set_index(["outcome", "group"])
    for outcome in metrics["outcome"].drop_duplicates():
        for left, right in pairs:
            if (outcome, left) not in indexed.index or (outcome, right) not in indexed.index:
                continue
            a = indexed.loc[(outcome, left)]
            b = indexed.loc[(outcome, right)]
            rows.append(
                {
                    "group_type": a["group_type"],
                    "outcome": outcome,
                    "left_group": left,
                    "right_group": right,
                    "left_rows": int(a["rows"]),
                    "right_rows": int(b["rows"]),
                    "delta_prevalence_left_minus_right": a["prevalence"] - b["prevalence"],
                    "delta_auc_left_minus_right": a["auc"] - b["auc"],
                    "delta_ap_left_minus_right": a["average_precision"] - b["average_precision"],
                    "delta_lift_left_minus_right": a["top_decile_lift"] - b["top_decile_lift"],
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_pairwise(
    sample: pd.DataFrame,
    group_column: str,
    pairs: list[tuple[str, str]],
    outcomes: list[str],
    repetitions: int,
    seed: int,
    group_type: str,
) -> pd.DataFrame:
    issuers = sample["issuer_ticker"].dropna().astype(str).unique()
    if len(issuers) < 2:
        return pd.DataFrame()
    by_issuer = {
        str(key): value.copy()
        for key, value in sample.groupby("issuer_ticker", observed=True, sort=False)
    }
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for _ in range(repetitions):
        selected = rng.choice(issuers, size=len(issuers), replace=True)
        boot = pd.concat([by_issuer[str(key)] for key in selected], ignore_index=True)
        for outcome in outcomes:
            for left, right in pairs:
                left_metrics = _metrics(boot[boot[group_column].eq(left)], outcome)
                right_metrics = _metrics(boot[boot[group_column].eq(right)], outcome)
                if left_metrics["rows"] == 0 or right_metrics["rows"] == 0:
                    continue
                draws.setdefault((outcome, left, right), []).append(
                    {
                        "delta_prevalence": left_metrics["prevalence"] - right_metrics["prevalence"],
                        "delta_auc": left_metrics["auc"] - right_metrics["auc"],
                        "delta_ap": left_metrics["average_precision"] - right_metrics["average_precision"],
                        "delta_lift": left_metrics["top_decile_lift"] - right_metrics["top_decile_lift"],
                    }
                )
    output: list[dict[str, Any]] = []
    for (outcome, left, right), records in draws.items():
        table = pd.DataFrame(records)
        for metric in ("delta_prevalence", "delta_auc", "delta_ap", "delta_lift"):
            values = pd.to_numeric(table[metric], errors="coerce").dropna()
            if values.empty:
                continue
            output.append(
                {
                    "group_type": group_type,
                    "outcome": outcome,
                    "left_group": left,
                    "right_group": right,
                    "metric": metric,
                    "bootstrap_repetitions_requested": repetitions,
                    "bootstrap_repetitions_valid": len(values),
                    "bootstrap_mean": float(values.mean()),
                    "ci_lower_2_5pct": float(values.quantile(0.025)),
                    "ci_upper_97_5pct": float(values.quantile(0.975)),
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(output)


def leave_one_exchange_out(
    sample: pd.DataFrame,
    exchanges: list[str],
    outcomes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        pooled = _metrics(sample[sample["exchange_group"].isin(exchanges)], outcome)
        for excluded in exchanges:
            subset = sample[
                sample["exchange_group"].isin(exchanges)
                & sample["exchange_group"].ne(excluded)
            ]
            metric = _metrics(subset, outcome)
            rows.append(
                {
                    "outcome": outcome,
                    "excluded_exchange": excluded,
                    "rows": metric["rows"],
                    "positives": metric["positives"],
                    "auc": metric["auc"],
                    "average_precision": metric["average_precision"],
                    "top_decile_lift": metric["top_decile_lift"],
                    "delta_auc_vs_pooled": metric["auc"] - pooled["auc"],
                    "delta_ap_vs_pooled": metric["average_precision"]
                    - pooled["average_precision"],
                    "delta_lift_vs_pooled": metric["top_decile_lift"]
                    - pooled["top_decile_lift"],
                }
            )
    return pd.DataFrame(rows)


def assign_covid_regime(
    years: pd.Series,
    pre_years: list[int],
    shock_years: list[int],
    recovery_years: list[int],
) -> pd.Series:
    numeric = pd.to_numeric(years, errors="coerce")
    return pd.Series(
        np.select(
            [
                numeric.isin(pre_years),
                numeric.isin(shock_years),
                numeric.isin(recovery_years),
            ],
            ["PRE_COVID", "COVID_SHOCK", "RECOVERY"],
            default="OUTSIDE_CONFIGURED_REGIMES",
        ),
        index=years.index,
        dtype="object",
    )


def covid_regime_tables(
    sample: pd.DataFrame,
    settings: dict[str, Any],
    outcomes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    covid = settings.get("covid", {})
    pre = [int(value) for value in covid.get("pre_years", [2016, 2017, 2018, 2019])]
    shock = [int(value) for value in covid.get("primary_shock_years", [2020, 2021])]
    recovery = [int(value) for value in covid.get("recovery_years", [2022, 2023, 2024, 2025])]
    work = sample.copy()
    work["covid_regime"] = assign_covid_regime(
        work["fiscal_year"], pre, shock, recovery
    )
    regimes = ["PRE_COVID", "COVID_SHOCK", "RECOVERY"]
    metrics = grouped_metrics(work, "covid_regime", regimes, outcomes, "COVID_REGIME")
    differences = pairwise_differences(
        metrics,
        [("COVID_SHOCK", "PRE_COVID"), ("RECOVERY", "PRE_COVID")],
    )

    sensitivity_rows: list[dict[str, Any]] = []
    windows = covid.get(
        "alternative_shock_windows",
        {
            "COVID_2020_ONLY": [2020],
            "COVID_2020_2021": [2020, 2021],
            "COVID_2020_2022": [2020, 2021, 2022],
        },
    )
    for label, years in windows.items():
        year_set = {int(value) for value in years}
        work["_shock"] = np.where(
            pd.to_numeric(work["fiscal_year"], errors="coerce").isin(year_set),
            "SHOCK",
            "NON_SHOCK",
        )
        table = grouped_metrics(
            work, "_shock", ["SHOCK", "NON_SHOCK"], outcomes, str(label)
        )
        diff = pairwise_differences(table, [("SHOCK", "NON_SHOCK")])
        if not diff.empty:
            diff.insert(0, "shock_window", str(label))
            diff.insert(1, "shock_years", ",".join(map(str, sorted(year_set))))
            sensitivity_rows.append(diff)
    sensitivity = (
        pd.concat(sensitivity_rows, ignore_index=True)
        if sensitivity_rows
        else pd.DataFrame()
    )
    return metrics, differences, sensitivity


def _interaction_design(
    frame: pd.DataFrame,
    outcome: str,
    focal_dummies: dict[str, pd.Series],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    work = frame.copy()
    work["score_z"] = _standardize(
        _score(work["abnormal_cfo_proxy"], SCORE_RULES[outcome])
    )
    x = pd.DataFrame(index=work.index)
    x["intercept"] = 1.0
    x["score_z"] = work["score_z"]
    focal_terms: list[str] = []
    for name, values in focal_dummies.items():
        dummy = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
        x[name] = dummy
        interaction = f"score_x_{name}"
        x[interaction] = work["score_z"] * dummy
        focal_terms.append(interaction)

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
        "fixed_effects", ["fiscal_year", "industry_name"]
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
    design = pd.concat([work.loc[valid, ["issuer_ticker"]], x.loc[valid]], axis=1)
    return design, y.loc[valid].astype(int), focal_terms


def _fit_interactions(
    sample: pd.DataFrame,
    outcomes: list[str],
    design_builder,
    settings: dict[str, Any],
    analysis_type: str,
    specification: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        design, y, focal_terms = design_builder(sample, outcome)
        positives = int(y.sum()) if len(y) else 0
        clusters = int(design["issuer_ticker"].nunique()) if not design.empty else 0
        minimum_rows = int(settings.get("minimum_interaction_rows", 300))
        minimum_positives = int(settings.get("minimum_interaction_positives", 20))
        if len(y) < minimum_rows or positives < minimum_positives:
            for focal in focal_terms:
                rows.append(
                    {
                        "analysis_type": analysis_type,
                        "specification": specification,
                        "outcome": outcome,
                        "term": focal,
                        "status": "INSUFFICIENT_SAMPLE",
                        "rows": len(y),
                        "positives": positives,
                        "clusters": clusters,
                        "focal_term": True,
                    }
                )
            continue
        terms = [column for column in design.columns if column != "issuer_ticker"]
        beta, se, status = _fit_logit_clustered(
            design.copy(),
            y,
            float(settings.get("interaction_ridge", 1e-6)),
            int(settings.get("interaction_max_iter", 100)),
            float(settings.get("interaction_tolerance", 1e-8)),
        )
        for term, estimate, standard_error in zip(terms, beta, se):
            z = estimate / standard_error if standard_error > 0 else np.nan
            p_value = math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "specification": specification,
                    "outcome": outcome,
                    "term": term,
                    "estimate": float(estimate),
                    "cluster_se": float(standard_error),
                    "z_value": float(z) if np.isfinite(z) else np.nan,
                    "p_value_two_sided": float(p_value) if np.isfinite(p_value) else np.nan,
                    "odds_ratio": float(np.exp(np.clip(estimate, -20, 20))),
                    "rows": len(y),
                    "positives": positives,
                    "clusters": clusters,
                    "status": status,
                    "focal_term": term in focal_terms,
                }
            )
    return pd.DataFrame(rows)


def exchange_interactions(
    sample: pd.DataFrame,
    settings: dict[str, Any],
    outcomes: list[str],
    exchanges: list[str],
) -> pd.DataFrame:
    reference = settings.get("exchange_reference", "HOSE")
    work = sample[sample["exchange_group"].isin(exchanges)].copy()
    non_reference = [value for value in exchanges if value != reference]

    def builder(frame: pd.DataFrame, outcome: str):
        dummies = {
            f"exchange_{value}": frame["exchange_group"].eq(value).astype(float)
            for value in non_reference
        }
        local = dict(settings)
        local["fixed_effects"] = [
            value
            for value in settings.get("exchange_fixed_effects", ["fiscal_year", "industry_name"])
            if value != "raw_exchange"
        ]
        return _interaction_design(frame, outcome, dummies, local)

    return _fit_interactions(
        work,
        outcomes,
        builder,
        settings,
        "EXCHANGE",
        f"reference={reference}",
    )


def covid_interactions(
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
            shock = pd.to_numeric(frame["fiscal_year"], errors="coerce").isin(year_set)
            local = dict(settings)
            # Year fixed effects absorb the COVID-period main effect. The focal
            # parameter is the score-slope change during the configured window.
            local["fixed_effects"] = settings.get(
                "covid_fixed_effects", ["fiscal_year", "raw_exchange", "industry_name"]
            )
            return _interaction_design(
                frame,
                outcome,
                {"covid_shock": shock.astype(float)},
                local,
            )

        result = _fit_interactions(
            sample,
            outcomes,
            builder,
            settings,
            "COVID_PERIOD",
            f"{label}:{','.join(map(str, sorted(year_set)))}",
        )
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def robustness_status(
    sample: pd.DataFrame,
    settings: dict[str, Any],
    exchanges: list[str],
) -> pd.DataFrame:
    minimum_rows = int(settings.get("minimum_group_rows", 100))
    minimum_positives = int(settings.get("minimum_group_positives", 10))
    exchange_rows = sample["exchange_group"].value_counts()
    exchange_positive = (
        sample.groupby("exchange_group", observed=True)["any_candidate"].sum()
        if "any_candidate" in sample
        else pd.Series(dtype=float)
    )
    exchange_pass = all(
        int(exchange_rows.get(group, 0)) >= minimum_rows
        and int(exchange_positive.get(group, 0)) >= minimum_positives
        for group in exchanges
    )

    covid = settings.get("covid", {})
    regime = assign_covid_regime(
        sample["fiscal_year"],
        [int(value) for value in covid.get("pre_years", [2016, 2017, 2018, 2019])],
        [int(value) for value in covid.get("primary_shock_years", [2020, 2021])],
        [int(value) for value in covid.get("recovery_years", [2022, 2023, 2024, 2025])],
    )
    covid_rows = regime.value_counts()
    covid_positive = (
        sample.assign(_regime=regime)
        .groupby("_regime", observed=True)["any_candidate"]
        .sum()
        if "any_candidate" in sample
        else pd.Series(dtype=float)
    )
    covid_groups = ["PRE_COVID", "COVID_SHOCK", "RECOVERY"]
    covid_pass = all(
        int(covid_rows.get(group, 0)) >= minimum_rows
        and int(covid_positive.get(group, 0)) >= minimum_positives
        for group in covid_groups
    )
    return pd.DataFrame(
        [
            {
                "gate": "within_exchange_robustness",
                "status": "PASS" if exchange_pass else "PARTIALLY_EVALUATED",
                "evidence_rows": len(sample),
                "interpretation": "Transportability across HOSE, HNX and UPCOM; not a causal exchange effect.",
            },
            {
                "gate": "covid_period_robustness",
                "status": "PASS" if covid_pass else "PARTIALLY_EVALUATED",
                "evidence_rows": len(sample),
                "interpretation": "Temporal regime robustness; not a causal COVID treatment effect.",
            },
        ]
    )


def run_regime_robustness(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sample = prepare_regime_sample(cases, settings)
    outcomes = [
        value for value in settings.get("outcomes", list(DEFAULT_OUTCOMES)) if value in sample
    ]
    exchanges = [
        normalize_exchange(value)
        for value in settings.get("exchange_groups", ["HOSE", "HNX", "UPCOM"])
    ]
    exchange_metrics = grouped_metrics(
        sample, "exchange_group", exchanges, outcomes, "EXCHANGE"
    )
    exchange_pairs = list(itertools.combinations(exchanges, 2))
    exchange_diff = pairwise_differences(exchange_metrics, exchange_pairs)
    exchange_bootstrap = cluster_bootstrap_pairwise(
        sample[sample["exchange_group"].isin(exchanges)],
        "exchange_group",
        exchange_pairs,
        outcomes,
        int(settings.get("bootstrap_repetitions", 500)),
        int(settings.get("bootstrap_seed", 240720)),
        "EXCHANGE",
    )
    exchange_leave_out = leave_one_exchange_out(sample, exchanges, outcomes)
    exchange_model = exchange_interactions(sample, settings, outcomes, exchanges)

    covid_metrics, covid_diff, covid_sensitivity = covid_regime_tables(
        sample, settings, outcomes
    )
    covid_bootstrap = cluster_bootstrap_pairwise(
        sample.assign(
            covid_regime=assign_covid_regime(
                sample["fiscal_year"],
                settings.get("covid", {}).get("pre_years", [2016, 2017, 2018, 2019]),
                settings.get("covid", {}).get("primary_shock_years", [2020, 2021]),
                settings.get("covid", {}).get("recovery_years", [2022, 2023, 2024, 2025]),
            )
        ),
        "covid_regime",
        [("COVID_SHOCK", "PRE_COVID"), ("RECOVERY", "PRE_COVID")],
        outcomes,
        int(settings.get("bootstrap_repetitions", 500)),
        int(settings.get("bootstrap_seed", 240720)) + 1,
        "COVID_REGIME",
    )
    covid_model = covid_interactions(sample, settings, outcomes)
    status = robustness_status(sample, settings, exchanges)

    return {
        "cfs_regime_robustness_sample": sample,
        "cfs_exchange_robustness_metrics": exchange_metrics,
        "cfs_exchange_pairwise_differences": exchange_diff,
        "cfs_exchange_cluster_bootstrap": exchange_bootstrap,
        "cfs_exchange_leave_one_out": exchange_leave_out,
        "cfs_exchange_interactions": exchange_model,
        "cfs_covid_regime_metrics": covid_metrics,
        "cfs_covid_regime_differences": covid_diff,
        "cfs_covid_window_sensitivity": covid_sensitivity,
        "cfs_covid_cluster_bootstrap": covid_bootstrap,
        "cfs_covid_interactions": covid_model,
        "cfs_regime_robustness_status": status,
    }
