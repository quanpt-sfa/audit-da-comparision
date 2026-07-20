from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from .cfs_regime_robustness import (
    DEFAULT_OUTCOMES,
    assign_covid_regime,
    cluster_bootstrap_pairwise,
    covid_regime_tables,
    exchange_interactions,
    grouped_metrics,
    leave_one_exchange_out,
    normalize_exchange,
    pairwise_differences,
    prepare_regime_sample,
    robustness_status,
)
from .cfs_regime_robustness_identified import identified_covid_interactions


def _configured_outcomes(
    sample: pd.DataFrame,
    settings: dict[str, Any],
) -> list[str]:
    return [
        value
        for value in settings.get("outcomes", list(DEFAULT_OUTCOMES))
        if value in sample.columns
    ]


def _configured_exchanges(settings: dict[str, Any]) -> list[str]:
    exchanges = [
        normalize_exchange(value)
        for value in settings.get("exchange_groups", ["HOSE", "HNX", "UPCOM"])
    ]
    return list(dict.fromkeys(value for value in exchanges if value != "UNKNOWN"))


def _sample_coverage(
    sample: pd.DataFrame,
    group_column: str,
    configured_groups: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_groups = (
        sample[group_column].fillna("UNKNOWN").astype(str).drop_duplicates().tolist()
        if group_column in sample
        else []
    )
    for group in list(dict.fromkeys(configured_groups + observed_groups)):
        subset = sample[sample[group_column].fillna("UNKNOWN").astype(str).eq(group)]
        row: dict[str, Any] = {
            "group": group,
            "configured_group": group in configured_groups,
            "rows": len(subset),
            "issuers": int(subset["issuer_ticker"].nunique()) if not subset.empty else 0,
            "minimum_year": (
                int(pd.to_numeric(subset["fiscal_year"], errors="coerce").min())
                if not subset.empty
                else pd.NA
            ),
            "maximum_year": (
                int(pd.to_numeric(subset["fiscal_year"], errors="coerce").max())
                if not subset.empty
                else pd.NA
            ),
        }
        if "any_candidate" in subset:
            outcome = pd.to_numeric(subset["any_candidate"], errors="coerce")
            row["any_candidate_positives"] = int(outcome.fillna(0).sum())
            row["any_candidate_prevalence"] = (
                float(outcome.mean()) if outcome.notna().any() else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_exchange_robustness(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sample = prepare_regime_sample(cases, settings)
    outcomes = _configured_outcomes(sample, settings)
    exchanges = _configured_exchanges(settings)
    if not exchanges:
        raise ValueError("At least one configured exchange is required")

    analysis_sample = sample[sample["exchange_group"].isin(exchanges)].copy()
    exchange_metrics = grouped_metrics(
        analysis_sample,
        "exchange_group",
        exchanges,
        outcomes,
        "EXCHANGE",
    )
    exchange_pairs = list(itertools.combinations(exchanges, 2))
    exchange_differences = pairwise_differences(exchange_metrics, exchange_pairs)
    exchange_bootstrap = cluster_bootstrap_pairwise(
        analysis_sample,
        "exchange_group",
        exchange_pairs,
        outcomes,
        int(settings.get("bootstrap_repetitions", 500)),
        int(
            settings.get(
                "exchange_bootstrap_seed",
                settings.get("bootstrap_seed", 240720),
            )
        ),
        "EXCHANGE",
    )
    exchange_leave_out = leave_one_exchange_out(
        analysis_sample,
        exchanges,
        outcomes,
    )
    exchange_model = exchange_interactions(
        analysis_sample,
        settings,
        outcomes,
        exchanges,
    )
    status = robustness_status(sample, settings, exchanges)
    status = status[status["gate"].eq("within_exchange_robustness")].copy()
    if not status.empty:
        status["evidence_rows"] = len(analysis_sample)
        status["configured_exchanges"] = "|".join(exchanges)
        status["excluded_unknown_exchange_rows"] = int(
            sample["exchange_group"].eq("UNKNOWN").sum()
        )

    analysis_sample["analysis_kind"] = "EXCHANGE"
    analysis_sample["configured_exchange"] = True
    coverage = _sample_coverage(sample, "exchange_group", exchanges)
    coverage.insert(0, "analysis_kind", "EXCHANGE")

    return {
        "cfs_exchange_robustness_sample": analysis_sample,
        "cfs_exchange_sample_coverage": coverage,
        "cfs_exchange_robustness_metrics": exchange_metrics,
        "cfs_exchange_pairwise_differences": exchange_differences,
        "cfs_exchange_cluster_bootstrap": exchange_bootstrap,
        "cfs_exchange_leave_one_out": exchange_leave_out,
        "cfs_exchange_interactions": exchange_model,
        "cfs_exchange_robustness_status": status,
    }


def run_covid_robustness(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sample = prepare_regime_sample(cases, settings)
    outcomes = _configured_outcomes(sample, settings)
    exchanges = _configured_exchanges(settings)
    covid = settings.get("covid", {})
    pre_years = [
        int(value)
        for value in covid.get("pre_years", [2016, 2017, 2018, 2019])
    ]
    shock_years = [
        int(value)
        for value in covid.get("primary_shock_years", [2020, 2021])
    ]
    recovery_years = [
        int(value)
        for value in covid.get("recovery_years", [2022, 2023, 2024, 2025])
    ]
    configured_regimes = ["PRE_COVID", "COVID_SHOCK", "RECOVERY"]

    sample["covid_regime"] = assign_covid_regime(
        sample["fiscal_year"],
        pre_years,
        shock_years,
        recovery_years,
    )
    analysis_sample = sample[sample["covid_regime"].isin(configured_regimes)].copy()

    covid_metrics, covid_differences, covid_sensitivity = covid_regime_tables(
        analysis_sample,
        settings,
        outcomes,
    )
    covid_bootstrap = cluster_bootstrap_pairwise(
        analysis_sample,
        "covid_regime",
        [("COVID_SHOCK", "PRE_COVID"), ("RECOVERY", "PRE_COVID")],
        outcomes,
        int(settings.get("bootstrap_repetitions", 500)),
        int(
            settings.get(
                "covid_bootstrap_seed",
                settings.get("bootstrap_seed", 240720) + 1,
            )
        ),
        "COVID_REGIME",
    )
    covid_model = identified_covid_interactions(
        analysis_sample,
        settings,
        outcomes,
    )
    status = robustness_status(sample, settings, exchanges)
    status = status[status["gate"].eq("covid_period_robustness")].copy()
    if not status.empty:
        status["evidence_rows"] = len(analysis_sample)
        status["pre_covid_years"] = ",".join(map(str, pre_years))
        status["primary_covid_years"] = ",".join(map(str, shock_years))
        status["recovery_years"] = ",".join(map(str, recovery_years))
        status["outside_configured_regime_rows"] = int(
            sample["covid_regime"].eq("OUTSIDE_CONFIGURED_REGIMES").sum()
        )

    analysis_sample["analysis_kind"] = "COVID"
    coverage = _sample_coverage(sample, "covid_regime", configured_regimes)
    coverage.insert(0, "analysis_kind", "COVID")

    return {
        "cfs_covid_robustness_sample": analysis_sample,
        "cfs_covid_sample_coverage": coverage,
        "cfs_covid_regime_metrics": covid_metrics,
        "cfs_covid_regime_differences": covid_differences,
        "cfs_covid_window_sensitivity": covid_sensitivity,
        "cfs_covid_cluster_bootstrap": covid_bootstrap,
        "cfs_covid_interactions": covid_model,
        "cfs_covid_robustness_status": status,
    }
