from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd

from .predictive_validity import (
    PREDICTIVE_TESTS,
    PredictiveValiditySettings,
    _test_complete_cases,
    accrual_quality_coefficients,
    accrual_quality_crossfit,
    build_accrual_quality_cases,
    build_predictive_cases,
    expanding_oos_predictions,
    pooled_state_comparisons,
)


def _bootstrap_p(values: np.ndarray) -> float:
    return float(
        min(
            1.0,
            2.0
            * min(
                (1 + np.sum(values <= 0)) / (len(values) + 1),
                (1 + np.sum(values >= 0)) / (len(values) + 1),
            ),
        )
    )


def _bootstrap_summary(estimate: float, values: np.ndarray) -> dict[str, float]:
    low, high = np.nanquantile(values, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": _bootstrap_p(values),
    }


def _sample_cluster_totals(
    aggregates: np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int,
    reducer,
) -> np.ndarray:
    cluster_count = int(len(aggregates))
    if cluster_count < 2:
        return np.full(int(draws), np.nan, dtype=float)
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(draws), dtype=float)
    for start in range(0, int(draws), int(batch_size)):
        stop = min(start + int(batch_size), int(draws))
        sampled = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        totals = aggregates[sampled].sum(axis=1)
        output[start:stop] = reducer(totals)
    return output


def _oos_task(task: dict[str, object]) -> list[dict[str, object]]:
    matrix = np.asarray(task["aggregates"], dtype=float)
    total = matrix.sum(axis=0)
    n = total[0]
    rmse_estimate = np.sqrt(total[2] / n) - np.sqrt(total[1] / n)
    mae_estimate = total[4] / n - total[3] / n

    def rmse_reducer(totals: np.ndarray) -> np.ndarray:
        n_draw = totals[:, 0]
        return np.sqrt(totals[:, 2] / n_draw) - np.sqrt(totals[:, 1] / n_draw)

    def mae_reducer(totals: np.ndarray) -> np.ndarray:
        n_draw = totals[:, 0]
        return totals[:, 4] / n_draw - totals[:, 3] / n_draw

    rmse_values = _sample_cluster_totals(
        matrix,
        draws=int(task["draws"]),
        seed=int(task["seed"]),
        batch_size=int(task["batch_size"]),
        reducer=rmse_reducer,
    )
    mae_values = _sample_cluster_totals(
        matrix,
        draws=int(task["draws"]),
        seed=int(task["seed"]) + 1,
        batch_size=int(task["batch_size"]),
        reducer=mae_reducer,
    )
    base = {
        "test": task["test"],
        "construct": task["construct"],
        "contrast": "audited_minus_pre",
        "negative_favours_audited": True,
        "n": int(task["n"]),
        "issuers": int(task["issuers"]),
    }
    return [
        {**base, "metric": "rmse", **_bootstrap_summary(rmse_estimate, rmse_values)},
        {**base, "metric": "mae", **_bootstrap_summary(mae_estimate, mae_values)},
    ]


def _aq_task(task: dict[str, object]) -> list[dict[str, object]]:
    matrix = np.asarray(task["aggregates"], dtype=float)
    total = matrix.sum(axis=0)
    n = total[0]
    estimates = {
        "rmse": np.sqrt(total[2] / n) - np.sqrt(total[1] / n),
        "mae": total[4] / n - total[3] / n,
        "residual_sd": (
            np.sqrt(max((total[8] - total[7] ** 2 / n) / max(n - 1, 1), 0.0))
            - np.sqrt(max((total[6] - total[5] ** 2 / n) / max(n - 1, 1), 0.0))
        ),
    }

    def rmse_reducer(totals: np.ndarray) -> np.ndarray:
        n_draw = totals[:, 0]
        return np.sqrt(totals[:, 2] / n_draw) - np.sqrt(totals[:, 1] / n_draw)

    def mae_reducer(totals: np.ndarray) -> np.ndarray:
        n_draw = totals[:, 0]
        return totals[:, 4] / n_draw - totals[:, 3] / n_draw

    def sd_reducer(totals: np.ndarray) -> np.ndarray:
        n_draw = totals[:, 0]
        denominator = np.maximum(n_draw - 1.0, 1.0)
        pre_var = np.maximum((totals[:, 6] - totals[:, 5] ** 2 / n_draw) / denominator, 0.0)
        audited_var = np.maximum(
            (totals[:, 8] - totals[:, 7] ** 2 / n_draw) / denominator,
            0.0,
        )
        return np.sqrt(audited_var) - np.sqrt(pre_var)

    reducers = {
        "rmse": rmse_reducer,
        "mae": mae_reducer,
        "residual_sd": sd_reducer,
    }
    rows: list[dict[str, object]] = []
    for offset, metric in enumerate(("rmse", "mae", "residual_sd")):
        values = _sample_cluster_totals(
            matrix,
            draws=int(task["draws"]),
            seed=int(task["seed"]) + offset,
            batch_size=int(task["batch_size"]),
            reducer=reducers[metric],
        )
        rows.append(
            {
                "metric": metric,
                "contrast": "audited_minus_pre",
                **_bootstrap_summary(float(estimates[metric]), values),
                "negative_favours_audited": True,
                "n": int(task["n"]),
                "issuers": int(task["issuers"]),
            }
        )
    return rows


def _execute_tasks(
    tasks: list[dict[str, object]],
    workers: int,
    progress: Callable[[str], None] | None,
) -> list[list[dict[str, object]]]:
    if not tasks:
        return []
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            fn = _oos_task if task["kind"] == "oos" else _aq_task
            results.append(fn(task))
            if progress is not None:
                progress(f"bootstrap task {index}/{len(tasks)} complete")
        return results

    results_by_index: dict[int, list[dict[str, object]]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for index, task in enumerate(tasks):
            fn = _oos_task if task["kind"] == "oos" else _aq_task
            futures[executor.submit(fn, task)] = index
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results_by_index[index] = future.result()
            completed += 1
            if progress is not None:
                progress(
                    f"bootstrap task {completed}/{len(tasks)} complete "
                    f"({worker_count} workers)"
                )
    return [results_by_index[index] for index in range(len(tasks))]


def _oos_aggregates(group: pd.DataFrame) -> np.ndarray:
    work = group.assign(
        __n=1.0,
        __sq_pre=pd.to_numeric(group.squared_error_pre, errors="coerce"),
        __sq_audited=pd.to_numeric(group.squared_error_audited, errors="coerce"),
        __abs_pre=pd.to_numeric(group.absolute_error_pre, errors="coerce"),
        __abs_audited=pd.to_numeric(group.absolute_error_audited, errors="coerce"),
    )
    return (
        work.groupby("issuer_ticker", observed=True)[
            ["__n", "__sq_pre", "__sq_audited", "__abs_pre", "__abs_audited"]
        ]
        .sum()
        .to_numpy(float)
    )


def summarize_oos_parallel(
    predictions: pd.DataFrame,
    settings: PredictiveValiditySettings,
    *,
    workers: int,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    state_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    r2_rows: list[dict[str, object]] = []

    for test_name, group in predictions.groupby("test", observed=True, sort=True):
        actual = pd.to_numeric(group.actual, errors="coerce")
        benchmark_error = actual - pd.to_numeric(group.prediction_benchmark, errors="coerce")
        benchmark_sse = float(np.square(benchmark_error).sum())
        metrics: dict[str, dict[str, float]] = {}
        for state in ("pre", "audited"):
            error = pd.to_numeric(group[f"error_{state}"], errors="coerce")
            sse = float(np.square(error).sum())
            metrics[state] = {
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "r2_oos": float(1.0 - sse / benchmark_sse)
                if benchmark_sse > 0
                else np.nan,
            }
            state_rows.append(
                {
                    "test": test_name,
                    "construct": group.construct.iloc[0],
                    "state": state,
                    "n": int(len(group)),
                    "issuers": int(group.issuer_ticker.nunique()),
                    "test_year_min": int(group.fiscal_year.min()),
                    "test_year_max": int(group.fiscal_year.max()),
                    **metrics[state],
                }
            )
        tasks.append(
            {
                "kind": "oos",
                "test": test_name,
                "construct": group.construct.iloc[0],
                "aggregates": _oos_aggregates(group),
                "draws": int(settings.bootstrap_draws),
                "seed": int(settings.seed)
                + sum(ord(char) for char in f"{test_name}:bootstrap"),
                "batch_size": int(batch_size),
                "n": int(len(group)),
                "issuers": int(group.issuer_ticker.nunique()),
            }
        )
        r2_rows.append(
            {
                "test": test_name,
                "construct": group.construct.iloc[0],
                "metric": "r2_oos",
                "contrast": "audited_minus_pre",
                "estimate": metrics["audited"]["r2_oos"] - metrics["pre"]["r2_oos"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_two_sided": np.nan,
                "negative_favours_audited": False,
                "n": int(len(group)),
                "issuers": int(group.issuer_ticker.nunique()),
            }
        )

    for rows in _execute_tasks(tasks, workers, progress):
        difference_rows.extend(rows)
    difference_rows.extend(r2_rows)
    return (
        pd.DataFrame(state_rows).sort_values(["test", "state"]).reset_index(drop=True),
        pd.DataFrame(difference_rows)
        .sort_values(["test", "metric"])
        .reset_index(drop=True),
    )


def _aq_aggregates(crossfit: pd.DataFrame) -> np.ndarray:
    work = crossfit.assign(
        __n=1.0,
        __sq_pre=pd.to_numeric(crossfit.squared_residual_pre, errors="coerce"),
        __sq_audited=pd.to_numeric(crossfit.squared_residual_audited, errors="coerce"),
        __abs_pre=pd.to_numeric(crossfit.absolute_residual_pre, errors="coerce"),
        __abs_audited=pd.to_numeric(crossfit.absolute_residual_audited, errors="coerce"),
        __res_pre=pd.to_numeric(crossfit.residual_pre, errors="coerce"),
        __res_audited=pd.to_numeric(crossfit.residual_audited, errors="coerce"),
    )
    work["__res_pre_sq"] = np.square(work["__res_pre"])
    work["__res_audited_sq"] = np.square(work["__res_audited"])
    return (
        work.groupby("issuer_ticker", observed=True)[
            [
                "__n",
                "__sq_pre",
                "__sq_audited",
                "__abs_pre",
                "__abs_audited",
                "__res_pre",
                "__res_pre_sq",
                "__res_audited",
                "__res_audited_sq",
            ]
        ]
        .sum()
        .to_numpy(float)
    )


def summarize_accrual_quality_parallel(
    crossfit: pd.DataFrame,
    settings: PredictiveValiditySettings,
    *,
    workers: int,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if crossfit.empty:
        return pd.DataFrame(), pd.DataFrame()
    state_rows: list[dict[str, object]] = []
    for state in ("pre", "audited"):
        residual = pd.to_numeric(crossfit[f"residual_{state}"], errors="coerce")
        state_rows.append(
            {
                "state": state,
                "n": int(len(crossfit)),
                "issuers": int(crossfit.issuer_ticker.nunique()),
                "year_min": int(crossfit.fiscal_year.min()),
                "year_max": int(crossfit.fiscal_year.max()),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "residual_sd": float(residual.std(ddof=1)),
            }
        )
    tasks = [
        {
            "kind": "aq",
            "aggregates": _aq_aggregates(crossfit),
            "draws": int(settings.bootstrap_draws),
            "seed": int(settings.seed) + sum(ord(char) for char in "aq:bootstrap"),
            "batch_size": int(batch_size),
            "n": int(len(crossfit)),
            "issuers": int(crossfit.issuer_ticker.nunique()),
        }
    ]
    rows = _execute_tasks(tasks, workers, progress)[0]
    return pd.DataFrame(state_rows), pd.DataFrame(rows)


def run_predictive_validity_parallel(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
    *,
    workers: int = 1,
    bootstrap_batch_size: int = 128,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    predictive_cases = build_predictive_cases(panel, settings)
    coefficients, pooled_fit = pooled_state_comparisons(predictive_cases, settings)
    oos_predictions, oos_folds = expanding_oos_predictions(predictive_cases, settings)
    oos_summary, oos_difference = summarize_oos_parallel(
        oos_predictions,
        settings,
        workers=workers,
        batch_size=bootstrap_batch_size,
        progress=progress,
    )

    aq_cases = build_accrual_quality_cases(panel, settings)
    aq_coefficients = accrual_quality_coefficients(aq_cases, settings)
    aq_crossfit = accrual_quality_crossfit(aq_cases, settings)
    aq_summary, aq_difference = summarize_accrual_quality_parallel(
        aq_crossfit,
        settings,
        workers=workers,
        batch_size=bootstrap_batch_size,
        progress=progress,
    )

    sample_rows: list[dict[str, object]] = []
    for test_name in PREDICTIVE_TESTS:
        clean, outcome, predictors = _test_complete_cases(predictive_cases, test_name)
        sample_rows.append(
            {
                "analysis": test_name,
                "outcome": outcome,
                "predictors": "|".join(predictors),
                "rows": int(len(clean)),
                "issuers": int(clean.issuer_ticker.nunique()),
                "year_min": int(clean.fiscal_year.min()) if len(clean) else np.nan,
                "year_max": int(clean.fiscal_year.max()) if len(clean) else np.nan,
            }
        )
    sample_rows.append(
        {
            "analysis": "accrual_quality",
            "outcome": "wca_scaled",
            "predictors": "cfo_lag_audited|cfo_current_state|cfo_lead_audited",
            "rows": int(len(aq_cases)),
            "issuers": int(aq_cases.issuer_ticker.nunique()),
            "year_min": int(aq_cases.fiscal_year.min()) if len(aq_cases) else np.nan,
            "year_max": int(aq_cases.fiscal_year.max()) if len(aq_cases) else np.nan,
        }
    )
    return {
        "predictive_validity_cases": predictive_cases,
        "predictive_validity_sample_manifest": pd.DataFrame(sample_rows),
        "predictive_validity_coefficients": coefficients,
        "predictive_validity_pooled_fit": pooled_fit,
        "predictive_validity_oos_folds": oos_folds,
        "predictive_validity_oos_predictions": oos_predictions,
        "predictive_validity_oos_summary": oos_summary,
        "predictive_validity_oos_state_differences": oos_difference,
        "accrual_quality_cases": aq_cases,
        "accrual_quality_coefficients": aq_coefficients,
        "accrual_quality_crossfit_cases": aq_crossfit,
        "accrual_quality_summary": aq_summary,
        "accrual_quality_state_differences": aq_difference,
    }
