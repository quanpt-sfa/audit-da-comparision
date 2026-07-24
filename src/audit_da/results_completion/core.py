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
            "p_directional": np.nan,
        }
    rng = np.random.default_rng(seed)
    estimate = float(statistic(clean))
    values = np.empty(draws)
    grouped_indices = [indices.to_numpy() for indices in clean.groupby(cluster, sort=False).groups.values()]
    for draw in range(draws):
        sampled = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        row_index = np.concatenate([grouped_indices[index] for index in sampled])
        values[draw] = statistic(clean.loc[row_index])
    return _bootstrap_summary(estimate, values, null)


def cluster_bootstrap_1d(
    values: Sequence[float] | np.ndarray | pd.Series,
    clusters: Sequence[object] | np.ndarray | pd.Series,
    *,
    statistic: str,
    draws: int = 2000,
    seed: int = 20260723,
    null: float | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """Vectorized issuer-cluster bootstrap for one-dimensional mean or median."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    cluster_array = np.asarray(clusters, dtype=object)
    keep = np.isfinite(x) & pd.notna(cluster_array)
    x = x[keep]
    cluster_array = cluster_array[keep]
    if not len(x):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_directional": np.nan}

    codes, uniques = pd.factorize(cluster_array, sort=False)
    cluster_count = len(uniques)
    if cluster_count < 2:
        estimate = float(np.mean(x) if statistic == "mean" else np.median(x))
        return {"estimate": estimate, "ci_low": np.nan, "ci_high": np.nan, "p_directional": np.nan}
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    estimate = float(np.mean(x) if statistic == "mean" else np.median(x))
    rng = np.random.default_rng(seed)
    simulated = np.empty(int(draws), dtype=float)
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    cluster_sizes = np.bincount(codes, minlength=cluster_count).astype(np.int64)

    if statistic == "mean":
        cluster_sums = np.bincount(codes, weights=x, minlength=cluster_count)
    else:
        order = np.argsort(x, kind="mergesort")
        sorted_values = x[order]
        sorted_codes = codes[order]

    batch_size = max(1, int(batch_size))
    for start in range(0, int(draws), batch_size):
        stop = min(int(draws), start + batch_size)
        multiplicities = rng.multinomial(cluster_count, probabilities, size=stop - start)
        if statistic == "mean":
            numerator = multiplicities @ cluster_sums
            denominator = multiplicities @ cluster_sizes
            simulated[start:stop] = np.divide(
                numerator,
                denominator,
                out=np.full(stop - start, np.nan),
                where=denominator > 0,
            )
        else:
            row_weights = multiplicities[:, sorted_codes]
            cumulative = np.cumsum(row_weights, axis=1)
            totals = multiplicities @ cluster_sizes
            left_rank = (totals - 1) // 2
            right_rank = totals // 2
            left_index = (cumulative > left_rank[:, None]).argmax(axis=1)
            right_index = (cumulative > right_rank[:, None]).argmax(axis=1)
            simulated[start:stop] = (sorted_values[left_index] + sorted_values[right_index]) / 2.0

    return _bootstrap_summary(estimate, simulated, null)


def _bootstrap_summary(estimate: float, values: np.ndarray, null: float | None) -> dict[str, float]:
    lo, hi = np.nanquantile(values, [0.025, 0.975])
    p = np.nan
    if null is not None:
        centred = values - estimate
        p = (1 + np.sum(centred <= -(estimate - null))) / (len(values) + 1)
    return {
        "estimate": float(estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_directional": float(p),
    }


def _adjust_pvalues(pvalues: Sequence[float], method: str) -> np.ndarray:
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (n - rank) * p[index])
            adjusted[index] = min(running, 1.0)
    elif method == "bh":
        running = 1.0
        for reverse_rank, index in enumerate(order[::-1], start=1):
            rank = n - reverse_rank + 1
            running = min(running, p[index] * n / rank)
            adjusted[index] = min(running, 1.0)
    else:
        raise ValueError(method)
    return adjusted


def output_hash(frame: pd.DataFrame) -> str:
    """Return a row-order-invariant SHA-256 hash for a result table."""
    if frame.empty and len(frame.columns) == 0:
        ordered = frame.reset_index(drop=True)
    else:
        canonical = frame.reindex(columns=sorted(frame.columns)).astype("string").fillna("<NA>")
        order = canonical.sort_values(
            canonical.columns.tolist(),
            kind="mergesort",
            na_position="last",
        ).index
        ordered = frame.loc[order].reset_index(drop=True)
    payload = ordered.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_outputs(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    metadata: Mapping[str, object],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"metadata": dict(metadata), "outputs": {}}
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["outputs"][name] = {
            "path": str(path),
            "rows": len(frame),
            "sha256": output_hash(frame),
        }
    (out / "results_completion_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def sample_exclusion_manifest(
    panel: pd.DataFrame,
    accrual_rows: pd.DataFrame,
    settings: CompletionSettings,
) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append(
        {
            "stage": "raw_panel_rows",
            "rows": len(panel),
            "issuer_years": panel[KEYS].drop_duplicates().shape[0],
        }
    )
    duplicate_keys = panel.duplicated(KEYS + ["audit_status"], keep=False)
    rows.append(
        {
            "stage": "duplicate_state_keys",
            "rows": int(duplicate_keys.sum()),
            "issuer_years": panel.loc[duplicate_keys, KEYS].drop_duplicates().shape[0],
        }
    )
    pair = paired_panel(panel, settings)
    rows.append({"stage": "paired_state_population", "rows": len(pair), "issuer_years": len(pair)})
    direct_needed = ["pat_pre", "pat_post", "cfo_pre", "cfo_post", "lag_assets_pre"]
    complete_direct = pair.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[column for column in direct_needed if column in pair]
    )
    rows.append(
        {
            "stage": "complete_direct_measure_population",
            "rows": len(complete_direct),
            "issuer_years": len(complete_direct),
        }
    )
    for keys, group in accrual_rows.groupby(["model", "architecture", "benchmark"], observed=True):
        model, architecture, benchmark = keys
        rows.append(
            {
                "stage": "accrual_model_population",
                "model": model,
                "architecture": architecture,
                "benchmark": benchmark,
                "rows": len(group),
                "issuer_years": group[KEYS].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    for column in aliases:
        if column in frame:
            return column
    return None


def _design_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    year_col: str = "fiscal_year",
    industry_col: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    parts = [pd.Series(1.0, index=frame.index, name="intercept")]
    names = ["intercept"]
    for column in columns:
        parts.append(pd.to_numeric(frame[column], errors="coerce").rename(column))
        names.append(column)
    if year_col in frame:
        dummies = pd.get_dummies(
            frame[year_col].astype(str), prefix="year", drop_first=True, dtype=float
        )
        parts.extend([dummies[column] for column in dummies])
        names.extend(dummies.columns.tolist())
    if industry_col and industry_col in frame:
        dummies = pd.get_dummies(
            frame[industry_col].astype(str), prefix="ind", drop_first=True, dtype=float
        )
        parts.extend([dummies[column] for column in dummies])
        names.extend(dummies.columns.tolist())
    matrix = pd.concat(parts, axis=1)
    return matrix.to_numpy(float), names


def _cluster_ols(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1) & pd.notna(clusters)
    y, x, clusters = y[keep], x[keep], clusters[keep]
    n, k = x.shape
    if n <= k or len(np.unique(clusters)) < 2:
        return np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((k, k))
    unique = np.unique(clusters)
    for cluster_value in unique:
        index = clusters == cluster_value
        score = x[index].T @ residual[index]
        meat += np.outer(score, score)
    cluster_count = len(unique)
    correction = cluster_count / (cluster_count - 1) * ((n - 1) / max(n - k, 1))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    p_value = 2 * stats.t.sf(np.abs(t_value), df=max(cluster_count - 1, 1))
    return beta, standard_error, p_value
