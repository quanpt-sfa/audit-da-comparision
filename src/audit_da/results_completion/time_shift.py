from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
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


_DONOR_DESIGNS = (
    "cyclic_within_issuer",
    "independent_within_issuer",
    "same_year_peer",
)


def _group_indices(codes: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(codes == code) for code in np.unique(codes)]


def _donor_indices(
    groups: list[np.ndarray],
    n_rows: int,
    batch_size: int,
    rng: np.random.Generator,
    design: str,
) -> np.ndarray:
    donors = np.broadcast_to(np.arange(n_rows, dtype=np.int64), (batch_size, n_rows)).copy()
    for indices in groups:
        size = len(indices)
        if size < 2:
            continue
        positions = np.arange(size, dtype=np.int64)[None, :]
        if design == "cyclic_within_issuer":
            offsets = rng.integers(1, size, size=(batch_size, 1), dtype=np.int64)
            local = (positions - offsets) % size
        else:
            offsets = rng.integers(1, size, size=(batch_size, size), dtype=np.int64)
            local = (positions + offsets) % size
        donors[:, indices] = indices[local]
    return donors


def _vectorized_shapley_contrast(
    da_pre: np.ndarray,
    pat_moves: np.ndarray,
    cfo_moves: np.ndarray,
    benchmark_move: np.ndarray,
) -> np.ndarray:
    """Return one median |phi_CFO|-|phi_PAT| contrast per simulation draw."""
    if pat_moves.ndim != 2 or cfo_moves.ndim != 2:
        raise ValueError("pat_moves and cfo_moves must be draw-by-row matrices")
    draw_count, row_count = pat_moves.shape
    if cfo_moves.shape != (draw_count, row_count):
        raise ValueError("PAT and CFO movement matrices must share shape")

    base = np.broadcast_to(np.asarray(da_pre, dtype=float), (draw_count, row_count))
    benchmark = np.broadcast_to(
        np.asarray(benchmark_move, dtype=float), (draw_count, row_count)
    )
    moves = (pat_moves, cfo_moves, benchmark)
    contributions = [np.zeros_like(pat_moves), np.zeros_like(pat_moves), np.zeros_like(pat_moves)]

    for order in permutations(range(3)):
        current = base
        current_abs = np.abs(current)
        for component in order:
            next_state = current + moves[component]
            contributions[component] += (current_abs - np.abs(next_state)) / 6.0
            current = next_state
            current_abs = np.abs(current)

    return np.median(np.abs(contributions[1]) - np.abs(contributions[0]), axis=1)


def _simulate_time_shift_task(task: dict) -> dict:
    configure_worker_environment(task["blas_threads"])
    rng = np.random.default_rng(task["seed"])
    da_pre = task["da_pre"]
    pat_move = task["pat_move"]
    cfo_move = task["cfo_move"]
    benchmark_move = task["benchmark_move"]
    groups = task["groups"]
    design = task["donor_design"]
    draws = int(task["draws"])
    batch_size = max(1, int(task["batch_size"]))
    simulated = np.empty(draws, dtype=float)

    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        current_batch = stop - start
        donors = _donor_indices(groups, len(da_pre), current_batch, rng, design)
        simulated[start:stop] = _vectorized_shapley_contrast(
            da_pre,
            pat_move[donors],
            cfo_move[donors],
            benchmark_move,
        )

    observed = float(task["observed"])
    return {
        "model": task["model"],
        "architecture": task["architecture"],
        "benchmark": task["benchmark"],
        "donor_design": design,
        "n": len(da_pre),
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
    pair = paired_panel(panel, settings)
    industry_candidates = [
        column
        for column in ("icb_industry_pre", "industry_pre", "raw_exchange_pre")
        if column in pair
    ]
    industry_col = industry_candidates[0] if industry_candidates else None
    extra = KEYS + ([industry_col] if industry_col else [])
    base = cases.merge(pair[extra], on=KEYS, how="left", validate="many_to_one")
    tasks: list[dict] = []

    grouped = base.groupby(["model", "architecture", "benchmark"], observed=True)
    for (model, architecture, benchmark), group0 in grouped:
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
            "benchmark_move": group.benchmark_move.to_numpy(float),
            "observed": float(group.component_contrast.median()),
            "draws": settings.simulation_draws,
            "batch_size": settings.simulation_batch_size,
            "blas_threads": settings.blas_threads_per_worker,
        }
        for design_index, donor_design in enumerate(_DONOR_DESIGNS):
            tasks.append(
                {
                    **common,
                    "donor_design": donor_design,
                    "groups": issuer_groups if donor_design != "same_year_peer" else peer_groups,
                    "seed": stable_task_seed(
                        settings.seed + 101,
                        model,
                        architecture,
                        benchmark,
                        donor_design,
                        design_index,
                    ),
                }
            )
    return tasks


def time_shift_benchmarks(
    cases: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run vectorized donor simulations, parallelized by group and donor design."""
    tasks = _time_shift_tasks(cases, panel, settings)
    if not tasks:
        return pd.DataFrame()

    worker_count = resolve_parallel_workers(settings.parallel_workers, len(tasks))
    configure_worker_environment(settings.blas_threads_per_worker)
    if progress:
        progress(
            f"time-shift: {len(tasks)} tasks, {settings.simulation_draws:,} draws each, "
            f"{worker_count} process workers"
        )

    rows: list[dict] = []
    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_simulate_time_shift_task(task))
            if progress:
                progress(f"time-shift task {index}/{len(tasks)} complete")
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {executor.submit(_simulate_time_shift_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if progress:
                    task = futures[future]
                    progress(
                        f"time-shift task {index}/{len(tasks)} complete: "
                        f"{task['model']}/{task['benchmark']}/{task['donor_design']}"
                    )

    return pd.DataFrame(rows).sort_values(
        ["model", "architecture", "benchmark", "donor_design"], kind="mergesort"
    ).reset_index(drop=True)
