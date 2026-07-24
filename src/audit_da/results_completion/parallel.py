from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    CompletionSettings,
    cluster_bootstrap_1d,
    configure_worker_environment,
    resolve_parallel_workers,
    stable_task_seed,
)


def _run_tasks(
    tasks: list[dict],
    worker,
    settings: CompletionSettings,
    stage_name: str,
    progress: Callable[[str], None] | None,
) -> list:
    if not tasks:
        return []
    worker_count = resolve_parallel_workers(settings.parallel_workers, len(tasks))
    configure_worker_environment(settings.blas_threads_per_worker)
    if progress:
        progress(f"{stage_name}: {len(tasks)} tasks on {worker_count} process workers")
    results: list = []
    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if progress:
                progress(f"{stage_name} task {index}/{len(tasks)} complete")
        return results

    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if progress:
                progress(f"{stage_name} task {index}/{len(tasks)} complete")
    return results


def _attribution_worker(task: dict) -> tuple[dict, list[dict]]:
    configure_worker_environment(task["blas_threads"])
    contrast = task["component_contrast"]
    cfo_larger = task["cfo_larger"]
    clusters = task["clusters"]
    median_result = cluster_bootstrap_1d(
        contrast,
        clusters,
        statistic="median",
        draws=task["draws"],
        seed=task["median_seed"],
        null=0.0,
        batch_size=task["batch_size"],
    )
    share_result = cluster_bootstrap_1d(
        cfo_larger,
        clusters,
        statistic="mean",
        draws=task["draws"],
        seed=task["share_seed"],
        null=0.5,
        batch_size=task["batch_size"],
    )
    finite_reduction = task["reduction"][np.isfinite(task["reduction"])]
    row = {
        "model": task["model"],
        "architecture": task["architecture"],
        "benchmark": task["benchmark"],
        "n": len(contrast),
        **{f"median_contrast_{key}": value for key, value in median_result.items()},
        **{f"cfo_larger_{key}": value for key, value in share_result.items()},
        "median_normalised_contrast": float(np.nanmedian(task["normalised_contrast"])),
        "mean_reduction": float(np.nanmean(task["reduction"])),
        "trimmed_mean_reduction": float(stats.trim_mean(finite_reduction, 0.01)),
    }
    quadrants: list[dict] = []
    values, counts = np.unique(task["signed_quadrant"], return_counts=True)
    for value, count in zip(values, counts, strict=True):
        quadrants.append(
            {
                "model": task["model"],
                "architecture": task["architecture"],
                "benchmark": task["benchmark"],
                "signed_quadrant": value,
                "count": int(count),
                "share": float(count / len(contrast)),
            }
        )
    return row, quadrants


def attribution_tables(
    cases: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    tasks: list[dict] = []
    for (model, architecture, benchmark), group in cases.groupby(
        ["model", "architecture", "benchmark"], observed=True
    ):
        tasks.append(
            {
                "model": model,
                "architecture": architecture,
                "benchmark": benchmark,
                "clusters": group.issuer_ticker.to_numpy(object),
                "component_contrast": pd.to_numeric(
                    group.component_contrast, errors="coerce"
                ).to_numpy(float),
                "cfo_larger": group.cfo_larger.astype(float).to_numpy(),
                "normalised_contrast": pd.to_numeric(
                    group.normalised_component_contrast, errors="coerce"
                ).to_numpy(float),
                "reduction": pd.to_numeric(group.reduction, errors="coerce").to_numpy(float),
                "signed_quadrant": group.signed_quadrant.astype(str).to_numpy(),
                "draws": settings.bootstrap_draws,
                "batch_size": settings.simulation_batch_size,
                "blas_threads": settings.blas_threads_per_worker,
                "median_seed": stable_task_seed(
                    settings.seed, "attribution", model, architecture, benchmark, "median"
                ),
                "share_seed": stable_task_seed(
                    settings.seed, "attribution", model, architecture, benchmark, "share"
                ),
            }
        )
    outputs = _run_tasks(tasks, _attribution_worker, settings, "attribution-bootstrap", progress)
    summary = [item[0] for item in outputs]
    quadrants = [row for item in outputs for row in item[1]]
    return {
        "rq1_attribution_matrix": pd.DataFrame(summary).sort_values(
            ["model", "architecture", "benchmark"], kind="mergesort"
        ).reset_index(drop=True),
        "rq1_signed_quadrants": pd.DataFrame(quadrants).sort_values(
            ["model", "architecture", "benchmark", "signed_quadrant"],
            kind="mergesort",
        ).reset_index(drop=True),
    }


def _switch_result(
    switch_values: np.ndarray,
    gate_values: np.ndarray,
    magnitude_values: np.ndarray,
    clusters: np.ndarray,
    task: dict,
    outcome: str,
    seed_suffix: str,
) -> tuple[dict, dict | None]:
    valid = np.isfinite(switch_values) & pd.notna(clusters)
    switch = switch_values[valid].astype(bool)
    gate = gate_values[valid].astype(bool)
    magnitude = magnitude_values[valid]
    cluster = clusters[valid]
    rate = cluster_bootstrap_1d(
        switch.astype(float),
        cluster,
        statistic="mean",
        draws=task["draws"],
        seed=stable_task_seed(task["seed"], seed_suffix, outcome, "rate"),
        batch_size=task["batch_size"],
    )
    switched = switch
    if switched.any():
        outside = cluster_bootstrap_1d(
            (~gate[switched]).astype(float),
            cluster[switched],
            statistic="mean",
            draws=task["draws"],
            seed=stable_task_seed(task["seed"], seed_suffix, outcome, "outside"),
            null=0.5,
            batch_size=task["batch_size"],
        )
        values = magnitude[switched]
        values = values[np.isfinite(values)]
        magnitude_row = {
            "outcome": outcome,
            "model": task["model"],
            "benchmark": task.get("benchmark"),
            "n": len(values),
            "median": float(np.median(values)) if len(values) else np.nan,
            "p75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            "p90": float(np.quantile(values, 0.90)) if len(values) else np.nan,
        }
    else:
        outside = {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_directional": np.nan,
        }
        magnitude_row = None
    summary = {
        "outcome": outcome,
        "model": task["model"],
        "benchmark": task.get("benchmark"),
        **{f"switch_{key}": value for key, value in rate.items()},
        **{f"outside_gate_{key}": value for key, value in outside.items()},
        "switch_n": int(switched.sum()),
        "denominator": int(len(switch)),
    }
    return summary, magnitude_row


def _jaccard_array(pre: np.ndarray, post: np.ndarray) -> float:
    pre = pre.astype(bool)
    post = post.astype(bool)
    union = np.sum(pre | post)
    return float(np.sum(pre & post) / union) if union else np.nan


def _switching_worker(task: dict) -> tuple[list[dict], list[dict], list[dict]]:
    configure_worker_environment(task["blas_threads"])
    summaries: list[dict] = []
    magnitudes: list[dict] = []
    jaccards: list[dict] = []
    if task["scope"] == "direct":
        summary, magnitude = _switch_result(
            task["switch"],
            task["gate"],
            task["magnitude"],
            task["clusters"],
            task,
            task["outcome"],
            "direct",
        )
        summaries.append(summary)
        if magnitude is not None:
            magnitudes.append(magnitude)
        return summaries, magnitudes, jaccards

    for outcome, switch_key, magnitude_key in (
        ("da_sign", "da_sign_switch", "da_sign_magnitude"),
        ("high_da", "high_da_switch", "high_da_magnitude"),
    ):
        summary, magnitude = _switch_result(
            task[switch_key],
            task["gate"],
            task[magnitude_key],
            task["clusters"],
            task,
            outcome,
            "model",
        )
        summaries.append(summary)
        if magnitude is not None:
            magnitudes.append(magnitude)

    rank = task["rank_displacement"]
    rank = rank[np.isfinite(rank)]
    magnitudes.append(
        {
            "outcome": "common_cdf_rank_displacement",
            "model": task["model"],
            "benchmark": task["benchmark"],
            "n": len(rank),
            "median": float(np.median(rank)) if len(rank) else np.nan,
            "p75": float(np.quantile(rank, 0.75)) if len(rank) else np.nan,
            "p90": float(np.quantile(rank, 0.90)) if len(rank) else np.nan,
        }
    )
    jaccards.append(
        {
            "model": task["model"],
            "benchmark": task["benchmark"],
            "fiscal_year": "pooled",
            "jaccard_high_da": _jaccard_array(task["high_da_pre"], task["high_da_post"]),
        }
    )
    years = task["fiscal_year"]
    for year in np.unique(years):
        index = years == year
        jaccards.append(
            {
                "model": task["model"],
                "benchmark": task["benchmark"],
                "fiscal_year": int(year),
                "jaccard_high_da": _jaccard_array(
                    task["high_da_pre"][index], task["high_da_post"][index]
                ),
            }
        )
    return summaries, magnitudes, jaccards


def switching_tables(
    direct: pd.DataFrame,
    model_cases: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    tasks: list[dict] = []
    direct_specs = (
        ("cfo_sign", "cfo_sign_switch", "cfo_sign_magnitude"),
        ("cfo_category", "cfo_category_switch", "cfo_category_distance"),
        ("high_ta", "high_ta_switch", "high_ta_magnitude"),
    )
    for outcome, switch_column, magnitude_column in direct_specs:
        tasks.append(
            {
                "scope": "direct",
                "outcome": outcome,
                "model": "direct",
                "benchmark": None,
                "switch": direct[switch_column].astype(float).to_numpy(),
                "gate": direct.gate_0_05.astype(bool).to_numpy(),
                "magnitude": pd.to_numeric(direct[magnitude_column], errors="coerce").to_numpy(float),
                "clusters": direct.issuer_ticker.to_numpy(object),
                "draws": settings.bootstrap_draws,
                "batch_size": settings.simulation_batch_size,
                "blas_threads": settings.blas_threads_per_worker,
                "seed": settings.seed,
            }
        )
    if not model_cases.empty:
        for (model, benchmark), group in model_cases.groupby(
            ["model", "benchmark"], observed=True
        ):
            tasks.append(
                {
                    "scope": "model",
                    "model": model,
                    "benchmark": benchmark,
                    "clusters": group.issuer_ticker.to_numpy(object),
                    "gate": group.gate_0_05.astype(bool).to_numpy(),
                    "da_sign_switch": group.da_sign_switch.astype(float).to_numpy(),
                    "da_sign_magnitude": pd.to_numeric(
                        group.da_sign_magnitude, errors="coerce"
                    ).to_numpy(float),
                    "high_da_switch": group.high_da_switch.astype(float).to_numpy(),
                    "high_da_magnitude": pd.to_numeric(
                        group.high_da_magnitude, errors="coerce"
                    ).to_numpy(float),
                    "rank_displacement": pd.to_numeric(
                        group.rank_displacement, errors="coerce"
                    ).to_numpy(float),
                    "high_da_pre": group.high_da_pre.astype(bool).to_numpy(),
                    "high_da_post": group.high_da_post.astype(bool).to_numpy(),
                    "fiscal_year": pd.to_numeric(group.fiscal_year, errors="coerce").to_numpy(int),
                    "draws": settings.bootstrap_draws,
                    "batch_size": settings.simulation_batch_size,
                    "blas_threads": settings.blas_threads_per_worker,
                    "seed": settings.seed,
                }
            )
    outputs = _run_tasks(tasks, _switching_worker, settings, "switching-bootstrap", progress)
    summaries = [row for output in outputs for row in output[0]]
    magnitudes = [row for output in outputs for row in output[1]]
    jaccards = [row for output in outputs for row in output[2]]
    return {
        "rq2_switch_summary": pd.DataFrame(summaries).sort_values(
            ["model", "benchmark", "outcome"], kind="mergesort", na_position="first"
        ).reset_index(drop=True),
        "rq2_switch_magnitudes": pd.DataFrame(magnitudes).sort_values(
            ["model", "benchmark", "outcome"], kind="mergesort", na_position="first"
        ).reset_index(drop=True),
        "rq2_jaccard": pd.DataFrame(jaccards).sort_values(
            ["model", "benchmark", "fiscal_year"],
            key=lambda column: column.astype(str),
            kind="mergesort",
        ).reset_index(drop=True),
    }


def _mc_p(observed: float, simulated: np.ndarray, greater: bool = True) -> float:
    count = np.sum(simulated >= observed) if greater else np.sum(simulated <= observed)
    return float((1 + count) / (len(simulated) + 1))


def _randomisation_worker(task: dict) -> list[dict]:
    configure_worker_environment(task["blas_threads"])
    rng = np.random.default_rng(task["seed"])
    pre = task["values_pre"]
    shifts = task["shifts"]
    draws = int(task["draws"])
    batch_size = max(1, int(task["batch_size"]))
    symmetric = np.empty(draws, dtype=float)
    reassigned = np.empty(draws, dtype=float)
    pre_sign = np.signbit(pre)[None, :]
    absolute_shift = np.abs(shifts)
    nonzero = shifts != 0

    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        current_batch = stop - start
        signs = rng.integers(0, 2, size=(current_batch, len(shifts)), dtype=np.int8)
        signs = signs.astype(float) * 2.0 - 1.0
        simulated_shift = np.where(nonzero[None, :], absolute_shift[None, :] * signs, 0.0)
        symmetric[start:stop] = np.mean(
            pre_sign != np.signbit(pre[None, :] + simulated_shift), axis=1
        )

        random_keys = rng.random((current_batch, len(shifts)), dtype=np.float32)
        order = np.argsort(random_keys, axis=1)
        permuted_shift = shifts[order]
        reassigned[start:stop] = np.mean(
            pre_sign != np.signbit(pre[None, :] + permuted_shift), axis=1
        )

    rows: list[dict] = []
    for benchmark, values in (
        ("symmetric_sign", symmetric),
        ("signed_shift_reassignment", reassigned),
    ):
        rows.append(
            {
                "outcome": task["outcome"],
                "model": task["model"],
                "benchmark_model": task["benchmark_model"],
                "benchmark": benchmark,
                "observed_switch_rate": task["observed_switch"],
                "sim_mean": float(values.mean()),
                "sim_p025": float(np.quantile(values, 0.025)),
                "sim_p975": float(np.quantile(values, 0.975)),
                "mc_p": _mc_p(task["observed_switch"], values),
                "draws": draws,
                "seed": int(task["seed"]),
            }
        )
    return rows


def randomisation_benchmarks(
    direct: pd.DataFrame,
    model_cases: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    tasks: list[dict] = []
    direct_clean = direct.dropna(subset=["cfo_pre", "cfo_post"]).copy()
    tasks.append(
        {
            "outcome": "cfo_sign",
            "model": "direct",
            "benchmark_model": "direct",
            "values_pre": direct_clean.cfo_pre.to_numpy(float),
            "shifts": (direct_clean.cfo_post - direct_clean.cfo_pre).to_numpy(float),
            "observed_switch": float(direct_clean.cfo_sign_switch.mean()),
            "draws": settings.simulation_draws,
            "batch_size": settings.simulation_batch_size,
            "blas_threads": settings.blas_threads_per_worker,
            "seed": stable_task_seed(settings.seed, "randomisation", "cfo_sign", "direct"),
        }
    )
    if not model_cases.empty:
        for (model, benchmark), group in model_cases.groupby(
            ["model", "benchmark"], observed=True
        ):
            tasks.append(
                {
                    "outcome": "da_sign",
                    "model": model,
                    "benchmark_model": benchmark,
                    "values_pre": group.da_pre.to_numpy(float),
                    "shifts": group.signed_shift.to_numpy(float),
                    "observed_switch": float(group.da_sign_switch.mean()),
                    "draws": settings.simulation_draws,
                    "batch_size": settings.simulation_batch_size,
                    "blas_threads": settings.blas_threads_per_worker,
                    "seed": stable_task_seed(
                        settings.seed, "randomisation", "da_sign", model, benchmark
                    ),
                }
            )
    outputs = _run_tasks(
        tasks, _randomisation_worker, settings, "randomisation", progress
    )
    rows = [row for output in outputs for row in output]
    return pd.DataFrame(rows).sort_values(
        ["outcome", "model", "benchmark_model", "benchmark"], kind="mergesort"
    ).reset_index(drop=True)
