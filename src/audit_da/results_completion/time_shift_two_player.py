from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Callable

import numpy as np
import pandas as pd

from .core import (
    KEYS,
    CompletionSettings,
    configure_worker_environment,
    paired_panel,
    resolve_parallel_workers,
    stable_task_seed,
)
from .switching import _mc_p
from .time_shift import _DONOR_DESIGNS, _donor_indices, _group_indices


def _vectorized_two_player_contrast(
    da_pre: np.ndarray,
    pat_moves: np.ndarray,
    cfo_moves: np.ndarray,
) -> np.ndarray:
    if pat_moves.ndim != 2 or cfo_moves.ndim != 2:
        raise ValueError("PAT and CFO movement matrices must be draw-by-row")
    if pat_moves.shape != cfo_moves.shape:
        raise ValueError("PAT and CFO movement matrices must share shape")
    draw_count, row_count = pat_moves.shape
    base = np.broadcast_to(np.asarray(da_pre, dtype=float), (draw_count, row_count))
    pat_first = np.abs(base) - np.abs(base + pat_moves)
    pat_second = np.abs(base + cfo_moves) - np.abs(base + cfo_moves + pat_moves)
    cfo_first = np.abs(base) - np.abs(base + cfo_moves)
    cfo_second = np.abs(base + pat_moves) - np.abs(base + pat_moves + cfo_moves)
    phi_pat = 0.5 * (pat_first + pat_second)
    phi_cfo = 0.5 * (cfo_first + cfo_second)
    return np.median(np.abs(phi_cfo) - np.abs(phi_pat), axis=1)


def _simulate_time_shift_task(task: dict) -> dict:
    configure_worker_environment(task["blas_threads"])
    rng = np.random.default_rng(task["seed"])
    draws = int(task["draws"])
    batch_size = max(1, int(task["batch_size"]))
    simulated = np.empty(draws, dtype=float)
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        donors = _donor_indices(
            task["groups"], len(task["da_pre"]), stop - start, rng, task["donor_design"]
        )
        simulated[start:stop] = _vectorized_two_player_contrast(
            task["da_pre"], task["pat_move"][donors], task["cfo_move"][donors]
        )
    observed = float(task["observed"])
    return {
        "model": task["model"],
        "architecture": task["architecture"],
        "benchmark": task["benchmark"],
        "donor_design": task["donor_design"],
        "attribution_player_count": 2,
        "n": len(task["da_pre"]),
        "observed_median_contrast": observed,
        "sim_mean": float(simulated.mean()),
        "sim_median": float(np.median(simulated)),
        "sim_p025": float(np.quantile(simulated, 0.025)),
        "sim_p975": float(np.quantile(simulated, 0.975)),
        "observed_minus_sim_median": float(observed - np.median(simulated)),
        "mc_p": _mc_p(observed, simulated),
        "draws": draws,
        "seed": int(task["seed"]),
    }


def _time_shift_tasks(
    cases: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
) -> list[dict]:
    if not cases["attribution_player_count"].eq(2).all():
        raise ValueError("Final time-shift requires two-player attribution cases")
    pair = paired_panel(panel, settings)
    industry_candidates = [
        column
        for column in (
            "icb_l1_pre", "industry_name_pre", "icb_industry_pre",
            "industry_pre", "raw_exchange_pre",
        )
        if column in pair
    ]
    industry_col = industry_candidates[0] if industry_candidates else None
    extra = KEYS + ([industry_col] if industry_col else [])
    base = cases.merge(pair[extra], on=KEYS, how="left", validate="many_to_one")
    tasks: list[dict] = []
    for (model, architecture, benchmark), group0 in base.groupby(
        ["model", "architecture", "benchmark"], observed=True
    ):
        if architecture != "pooled":
            continue
        group = group0.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
        eligible_counts = group.groupby("issuer_ticker").fiscal_year.nunique()
        eligible = eligible_counts.index[eligible_counts >= 2]
        group = group[group.issuer_ticker.isin(eligible)].reset_index(drop=True)
        if group.empty:
            continue
        issuer_codes, _ = pd.factorize(group.issuer_ticker, sort=True)
        issuer_groups = _group_indices(issuer_codes)
        if industry_col:
            peer_key = pd.MultiIndex.from_frame(group[["fiscal_year", industry_col]])
            peer_codes, _ = pd.factorize(peer_key, sort=True)
        else:
            peer_codes, _ = pd.factorize(group.fiscal_year, sort=True)
        peer_groups = _group_indices(peer_codes)
        common = {
            "model": model,
            "architecture": architecture,
            "benchmark": benchmark,
            "da_pre": group.da_pre.to_numpy(float),
            "pat_move": group.pat_move.to_numpy(float),
            "cfo_move": group.cfo_move.to_numpy(float),
            "observed": float(group.component_contrast.median()),
            "draws": settings.simulation_draws,
            "batch_size": settings.simulation_batch_size,
            "blas_threads": settings.blas_threads_per_worker,
        }
        for design_index, donor_design in enumerate(_DONOR_DESIGNS):
            tasks.append({
                **common,
                "donor_design": donor_design,
                "groups": issuer_groups if donor_design != "same_year_peer" else peer_groups,
                "seed": stable_task_seed(
                    settings.seed + 101, "two_player", model, architecture,
                    benchmark, donor_design, design_index,
                ),
            })
    return tasks


def time_shift_benchmarks(
    cases: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    tasks = _time_shift_tasks(cases, panel, settings)
    if not tasks:
        return pd.DataFrame()
    worker_count = resolve_parallel_workers(settings.parallel_workers, len(tasks))
    configure_worker_environment(settings.blas_threads_per_worker)
    if progress:
        progress(
            f"two-player time-shift: {len(tasks)} tasks, "
            f"{settings.simulation_draws:,} draws each, {worker_count} workers"
        )
    rows: list[dict] = []
    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_simulate_time_shift_task(task))
            if progress:
                progress(f"two-player time-shift task {index}/{len(tasks)} complete")
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {executor.submit(_simulate_time_shift_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if progress:
                    task = futures[future]
                    progress(
                        f"two-player time-shift task {index}/{len(tasks)}: "
                        f"{task['model']}/{task['benchmark']}/{task['donor_design']}"
                    )
    return pd.DataFrame(rows).sort_values(
        ["model", "architecture", "benchmark", "donor_design"],
        kind="mergesort",
    ).reset_index(drop=True)
