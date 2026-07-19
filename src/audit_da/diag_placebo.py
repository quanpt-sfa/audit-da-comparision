from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .diag_common import KEYS, trimmed_mean


def _safe_qcut(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        codes = pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.Series(np.nan, index=values.index)
    return codes.map(lambda x: f"{prefix}{int(x)}" if pd.notna(x) else f"{prefix}MISSING")


def _prepare_matching_columns(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = frame.copy()
    bins = settings.get("conditioning_bins", {})
    by = list(bins.get("within", ["fiscal_year"]))
    if bins.get("size_bins", 0) and "lag_assets" in out:
        out["size_bin"] = out.groupby(by, observed=True, group_keys=False)["lag_assets"].apply(
            lambda x: _safe_qcut(np.log1p(pd.to_numeric(x, errors="coerce").abs()), int(bins["size_bins"]), "S")
        )
    if bins.get("abs_da_pre_bins", 0):
        out["abs_da_pre_bin"] = out.groupby(by, observed=True, group_keys=False)["da_pre"].apply(
            lambda x: _safe_qcut(pd.to_numeric(x, errors="coerce").abs(), int(bins["abs_da_pre_bins"]), "D")
        )
    if bins.get("abs_adjustment_bins", 0):
        out["abs_adjustment_bin"] = out.groupby(by, observed=True, group_keys=False)["raw_ta_shift"].apply(
            lambda x: _safe_qcut(pd.to_numeric(x, errors="coerce").abs(), int(bins["abs_adjustment_bins"]), "A")
        )
    return out


def _codes(meta: pd.DataFrame, columns: list[str], minimum_size: int) -> tuple[np.ndarray, pd.Series]:
    cols = [c for c in columns if c in meta.columns]
    if not cols:
        keys = pd.Series("__POOLED__", index=meta.index)
    else:
        keys = meta[cols].fillna("__MISSING__").astype(str).agg("|".join, axis=1)
    counts = keys.value_counts()
    small = keys.map(counts).to_numpy() < minimum_size
    if small.any():
        fallback = meta["fiscal_year"].astype(str) + "|__POOLED__" if "fiscal_year" in meta else "__POOLED__"
        keys = pd.Series(np.where(small, fallback, keys), index=meta.index)
    return pd.factorize(keys, sort=True)[0], keys


def _permute(values: np.ndarray, codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(values)
    for code in np.unique(codes):
        idx = np.flatnonzero(codes == code)
        out[idx] = values[idx][rng.permutation(len(idx))]
    return out


def _finite_placebo_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = ["da_pre", "raw_ta_shift", "reduction", "signed_shift"]
    mask = np.ones(len(frame), dtype=bool)
    for column in required:
        mask &= np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float))
    return frame.loc[mask].reset_index(drop=True), int((~mask).sum())


def directional_placebo(
    baseline: pd.DataFrame,
    panel: pd.DataFrame,
    models: Iterable[str],
    benchmarks: Iterable[str],
    strata_columns: list[str],
    minimum_stratum_size: int,
    permutations: int,
    trim_fraction: float,
    random_seed: int,
    identity_tolerance: float = 1e-8,
    conditioning_bins: dict | None = None,
):
    conditioning_bins = conditioning_bins or {}
    pre_panel = panel[panel.audit_status.eq("unaudited")].drop_duplicates(KEYS).copy()
    meta_columns = KEYS + [c for c in ["raw_exchange", "lag_assets", "assets"] if c in pre_panel]
    meta = pre_panel[meta_columns]
    rng = np.random.default_rng(random_seed)
    summaries: list[dict] = []
    draws: list[dict] = []
    for model in models:
        for benchmark in benchmarks:
            raw = baseline[baseline.model.eq(model) & baseline.benchmark.eq(benchmark)].merge(
                meta, on=KEYS, how="left", validate="many_to_one"
            )
            if raw.empty:
                continue
            g, dropped = _finite_placebo_rows(raw)
            if g.empty:
                continue
            identity = float(np.max(np.abs(g.signed_shift.to_numpy(float) - g.raw_ta_shift.to_numpy(float))))
            if identity > identity_tolerance:
                raise ValueError(f"Placebo requires common benchmark: {model}/{benchmark}, error={identity:.3g}")
            settings = {"conditioning_bins": conditioning_bins}
            g = _prepare_matching_columns(g, settings)
            expanded_strata = list(strata_columns)
            for generated in ["size_bin", "abs_da_pre_bin", "abs_adjustment_bin"]:
                if generated in g and generated not in expanded_strata:
                    expanded_strata.append(generated)
            codes, stratum_keys = _codes(g, expanded_strata, minimum_stratum_size)
            pre = g.da_pre.to_numpy(float)
            adj = g.raw_ta_shift.to_numpy(float)
            means = pd.Series(adj).groupby(codes).transform("mean").to_numpy()
            centered = adj - means
            real = g.reduction.to_numpy(float)
            real_mean = float(np.mean(real))
            real_trim = trimmed_mean(real, trim_fraction)
            for kind in ["raw_permutation", "centered_permutation", "symmetric_sign"]:
                pm = np.empty(permutations)
                pt = np.empty(permutations)
                pp = np.empty(permutations)
                pn = np.empty(permutations)
                for b in range(permutations):
                    if kind == "raw_permutation":
                        eta = _permute(adj, codes, rng)
                    elif kind == "centered_permutation":
                        eta = _permute(centered, codes, rng)
                    else:
                        eta = _permute(np.abs(centered), codes, rng) * rng.choice([-1.0, 1.0], size=len(adj))
                    r = np.abs(pre) - np.abs(pre + eta)
                    finite_r = r[np.isfinite(r)]
                    pm[b] = float(np.mean(finite_r))
                    pt[b] = trimmed_mean(finite_r, trim_fraction)
                    pp[b] = float((finite_r > 0).mean())
                    pn[b] = float((finite_r < 0).mean())
                    draws.append({
                        "model": model, "benchmark": benchmark, "placebo_type": kind,
                        "permutation": b, "mean_reduction": pm[b], "trimmed_mean_reduction": pt[b],
                        "share_positive": pp[b], "share_negative": pn[b],
                    })
                summaries.append({
                    "model": model,
                    "benchmark": benchmark,
                    "placebo_type": kind,
                    "rows_raw": len(raw),
                    "rows_finite": len(g),
                    "rows_dropped_nonfinite": dropped,
                    "strata_count": int(pd.Series(codes).nunique()),
                    "median_stratum_size": float(pd.Series(codes).value_counts().median()),
                    "max_common_benchmark_identity_error": identity,
                    "real_mean_reduction": real_mean,
                    "real_trimmed_mean_reduction": real_trim,
                    "placebo_mean": float(pm.mean()),
                    "placebo_q025": float(np.quantile(pm, .025)),
                    "placebo_q975": float(np.quantile(pm, .975)),
                    "placebo_trimmed_mean": float(pt.mean()),
                    "placebo_trimmed_q025": float(np.quantile(pt, .025)),
                    "placebo_trimmed_q975": float(np.quantile(pt, .975)),
                    "corrective_excess_mean": real_mean - float(pm.mean()),
                    "corrective_excess_trimmed": real_trim - float(pt.mean()),
                    "randomization_p_mean_ge_real": float((1 + (pm >= real_mean).sum()) / (permutations + 1)),
                    "randomization_p_trimmed_ge_real": float((1 + (pt >= real_trim).sum()) / (permutations + 1)),
                    "share_placebo_mean_negative": float((pm < 0).mean()),
                    "share_placebo_trimmed_negative": float((pt < 0).mean()),
                    "placebo_positive_minus_negative_share": float((pp - pn).mean()),
                    "all_outputs_finite": bool(np.isfinite(pm).all() and np.isfinite(pt).all()),
                })
    return pd.DataFrame(summaries), pd.DataFrame(draws)
