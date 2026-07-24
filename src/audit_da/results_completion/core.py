from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["issuer_ticker", "fiscal_year"]
DEFAULT_MODELS: dict[str, list[str]] = {
    "jones": ["inv_assets", "drev_scaled", "ppe_scaled"],
    "modified_jones": ["inv_assets", "drev_drec_scaled", "ppe_scaled"],
    "kothari": ["inv_assets", "drev_drec_scaled", "ppe_scaled", "roa"],
    "nonlinear_modified_jones": [
        "inv_assets",
        "drev_drec_scaled",
        "ppe_scaled",
        "roa",
        "loss",
        "drev_drec_sq",
    ],
}
BENCHMARKS = ("audited_reference", "pre_reference", "version_specific")


@dataclass(frozen=True)
class CompletionSettings:
    audited_label: str = "audited"
    unaudited_label: str = "unaudited"
    # The first source year supplies lagged inputs only. With source data beginning
    # in 2015, 2016 is the first model-complete estimation year and 2017 is the
    # first model-based test year. Direct reporting-state comparisons can still
    # use 2016 because their beginning-assets lag is available from 2015.
    source_start_year: int = 2015
    training_start_year: int = 2016
    test_start_year: int = 2017
    test_end_year: int = 2025
    min_train_rows: int = 100
    min_industry_rows: int = 40
    trailing_years: int = 5
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    bootstrap_draws: int = 2000
    simulation_draws: int = 2000
    seed: int = 20260723
    profit_thresholds: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1)
    direct_thresholds: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    tail_quantile: float = 0.9
    negligible_sd: float = 1e-12
    parallel_workers: int = 0
    simulation_batch_size: int = 32
    blas_threads_per_worker: int = 1


def resolve_parallel_workers(requested: int, task_count: int) -> int:
    """Resolve a process count without oversubscribing logical SMT threads."""
    if task_count <= 1:
        return 1
    if requested > 0:
        return max(1, min(int(requested), int(task_count)))
    logical = os.cpu_count() or 1
    physical_guess = max(1, logical // 2) if logical > 2 else logical
    return max(1, min(int(task_count), physical_guess))


def configure_worker_environment(blas_threads: int = 1) -> None:
    """Limit BLAS/OpenMP threads inherited by spawned worker processes."""
    value = str(max(1, int(blas_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def stable_task_seed(base_seed: int, *parts: object) -> int:
    """Create a scheduling-independent uint64 seed from a task label."""
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"audit-da").digest()
    return int.from_bytes(digest, "little", signed=False)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def paired_panel(panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    needed = set(KEYS + ["audit_status"])
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")
    pre = panel.loc[panel.audit_status.eq(settings.unaudited_label)].drop_duplicates(KEYS).copy()
    post = panel.loc[panel.audit_status.eq(settings.audited_label)].drop_duplicates(KEYS).copy()
    shared = sorted((set(pre.columns) & set(post.columns)) - set(KEYS + ["audit_status"]))
    pre = pre[KEYS + shared].rename(columns={column: f"{column}_pre" for column in shared})
    post = post[KEYS + shared].rename(columns={column: f"{column}_post" for column in shared})
    return pre.merge(post, on=KEYS, how="inner", validate="one_to_one")


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    cluster: str = "issuer_ticker",
    draws: int = 2000,
    seed: int = 20260723,
    null: float | None = None,
) -> dict[str, float]:
    """Compatibility bootstrap for arbitrary DataFrame statistics."""
    clean = frame.dropna(subset=[cluster]).copy()
    clusters = clean[cluster].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return {
            "estimate": float(statistic(clean)),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    estimate = float(statistic(clean))
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    by_cluster = {key: value for key, value in clean.groupby(cluster, sort=False)}
    for draw in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [by_cluster[key].assign(**{cluster: f"{key}__{index}"}) for index, key in enumerate(sampled)]
        values[draw] = float(statistic(pd.concat(pieces, ignore_index=True)))
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    ci_low, ci_high = np.quantile(finite, [0.025, 0.975])
    p_value = np.nan
    if null is not None:
        centred = finite - estimate
        distance = abs(estimate - null)
        p_value = float((1 + np.sum(np.abs(centred) >= distance)) / (len(finite) + 1))
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }
