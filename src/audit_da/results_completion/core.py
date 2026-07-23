from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
KEYS = ['issuer_ticker', 'fiscal_year']
DEFAULT_MODELS: dict[str, list[str]] = {'jones': ['inv_assets', 'drev_scaled', 'ppe_scaled'], 'modified_jones': ['inv_assets', 'drev_drec_scaled', 'ppe_scaled'], 'kothari': ['inv_assets', 'drev_drec_scaled', 'ppe_scaled', 'roa'], 'nonlinear_modified_jones': ['inv_assets', 'drev_drec_scaled', 'ppe_scaled', 'roa', 'loss', 'drev_drec_sq']}
BENCHMARKS = ('audited_reference', 'pre_reference', 'version_specific')

@dataclass(frozen=True)
class CompletionSettings:
    audited_label: str = 'audited'
    unaudited_label: str = 'unaudited'
    training_start_year: int = 2015
    test_start_year: int = 2016
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

def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for c in columns:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def paired_panel(panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    needed = set(KEYS + ['audit_status'])
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f'Panel missing required columns: {sorted(missing)}')
    pre = panel.loc[panel.audit_status.eq(settings.unaudited_label)].drop_duplicates(KEYS).copy()
    post = panel.loc[panel.audit_status.eq(settings.audited_label)].drop_duplicates(KEYS).copy()
    shared = sorted((set(pre.columns) & set(post.columns)) - set(KEYS + ['audit_status']))
    pre = pre[KEYS + shared].rename(columns={c: f'{c}_pre' for c in shared})
    post = post[KEYS + shared].rename(columns={c: f'{c}_post' for c in shared})
    return pre.merge(post, on=KEYS, how='inner', validate='one_to_one')

def cluster_bootstrap(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], cluster: str='issuer_ticker', draws: int=2000, seed: int=20260723, null: float | None=None) -> dict[str, float]:
    clean = frame.dropna(subset=[cluster]).copy()
    clusters = clean[cluster].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return {'estimate': float(statistic(clean)), 'ci_low': np.nan, 'ci_high': np.nan, 'p_directional': np.nan}
    rng = np.random.default_rng(seed)
    estimate = float(statistic(clean))
    values = np.empty(draws)
    grouped = {key: g for key, g in clean.groupby(cluster, sort=False)}
    for b in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        values[b] = statistic(boot)
    lo, hi = np.nanquantile(values, [0.025, 0.975])
    p = np.nan
    if null is not None:
        centred = values - estimate
        p = (1 + np.sum(centred <= -(estimate - null))) / (draws + 1)
    return {'estimate': estimate, 'ci_low': float(lo), 'ci_high': float(hi), 'p_directional': float(p)}

def _adjust_pvalues(pvalues: Sequence[float], method: str) -> np.ndarray:
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    if method == 'holm':
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (n - rank) * p[idx])
            adjusted[idx] = min(running, 1.0)
    elif method == 'bh':
        running = 1.0
        for rev_rank, idx in enumerate(order[::-1], start=1):
            rank = n - rev_rank + 1
            running = min(running, p[idx] * n / rank)
            adjusted[idx] = min(running, 1.0)
    else:
        raise ValueError(method)
    return adjusted

def output_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values([c for c in KEYS if c in frame], kind='mergesort').reset_index(drop=True)
    payload = ordered.to_csv(index=False, float_format='%.17g').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def write_outputs(tables: Mapping[str, pd.DataFrame], output_dir: str | Path, metadata: Mapping[str, object]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {'metadata': dict(metadata), 'outputs': {}}
    for name, frame in tables.items():
        path = out / f'{name}.csv'
        frame.to_csv(path, index=False)
        manifest['outputs'][name] = {'path': str(path), 'rows': len(frame), 'sha256': output_hash(frame)}
    (out / 'results_completion_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')

def sample_exclusion_manifest(panel: pd.DataFrame, accrual_rows: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append({'stage': 'raw_panel_rows', 'rows': len(panel), 'issuer_years': panel[KEYS].drop_duplicates().shape[0]})
    duplicate_keys = panel.duplicated(KEYS + ['audit_status'], keep=False)
    rows.append({'stage': 'duplicate_state_keys', 'rows': int(duplicate_keys.sum()), 'issuer_years': panel.loc[duplicate_keys, KEYS].drop_duplicates().shape[0]})
    pair = paired_panel(panel, settings)
    rows.append({'stage': 'paired_state_population', 'rows': len(pair), 'issuer_years': len(pair)})
    direct_needed = ['pat_pre', 'pat_post', 'cfo_pre', 'cfo_post', 'lag_assets_pre']
    complete_direct = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in direct_needed if c in pair])
    rows.append({'stage': 'complete_direct_measure_population', 'rows': len(complete_direct), 'issuer_years': len(complete_direct)})
    for keys, g in accrual_rows.groupby(['model', 'architecture', 'benchmark'], observed=True):
        model, architecture, benchmark = keys
        rows.append({'stage': 'accrual_model_population', 'model': model, 'architecture': architecture, 'benchmark': benchmark, 'rows': len(g), 'issuer_years': g[KEYS].drop_duplicates().shape[0]})
    return pd.DataFrame(rows)

def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    for c in aliases:
        if c in frame:
            return c
    return None

def _design_matrix(frame: pd.DataFrame, columns: Sequence[str], year_col: str='fiscal_year', industry_col: str | None=None) -> tuple[np.ndarray, list[str]]:
    parts = [pd.Series(1.0, index=frame.index, name='intercept')]
    names = ['intercept']
    for c in columns:
        parts.append(pd.to_numeric(frame[c], errors='coerce').rename(c))
        names.append(c)
    if year_col in frame:
        d = pd.get_dummies(frame[year_col].astype(str), prefix='year', drop_first=True, dtype=float)
        parts.extend([d[c] for c in d])
        names.extend(d.columns.tolist())
    if industry_col and industry_col in frame:
        d = pd.get_dummies(frame[industry_col].astype(str), prefix='ind', drop_first=True, dtype=float)
        parts.extend([d[c] for c in d])
        names.extend(d.columns.tolist())
    x = pd.concat(parts, axis=1)
    return (x.to_numpy(float), names)

def _cluster_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1) & pd.notna(clusters)
    y, x, clusters = (y[keep], x[keep], clusters[keep])
    n, k = x.shape
    if n <= k or len(np.unique(clusters)) < 2:
        return (np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan))
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((k, k))
    unique = np.unique(clusters)
    for c in unique:
        idx = clusters == c
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    g = len(unique)
    correction = g / (g - 1) * ((n - 1) / max(n - k, 1))
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=max(g - 1, 1))
    return (beta, se, p)
