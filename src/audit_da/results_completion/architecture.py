from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .core import KEYS, BENCHMARKS, DEFAULT_MODELS, CompletionSettings, _numeric, paired_panel, cluster_bootstrap

def _winsor_bounds(frame: pd.DataFrame, columns: Sequence[str], lo: float, hi: float) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for c in columns:
        x = pd.to_numeric(frame[c], errors='coerce')
        bounds[c] = (float(x.quantile(lo)), float(x.quantile(hi)))
    return bounds

def _apply_bounds(frame: pd.DataFrame, bounds: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    out = frame.copy()
    for c, (lo, hi) in bounds.items():
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce').clip(lo, hi)
    return out

def _fit_model(training: pd.DataFrame, features: Sequence[str]) -> tuple[StandardScaler, LinearRegression, float]:
    scaler = StandardScaler().fit(training[list(features)])
    x = scaler.transform(training[list(features)])
    model = LinearRegression().fit(x, training['ta_scaled'])
    residual = training['ta_scaled'].to_numpy(float) - model.predict(x)
    residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan
    return (scaler, model, residual_sd)

def _design_pairs(panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    pair = paired_panel(panel, settings)
    required = ['ta_scaled_pre', 'ta_scaled_post', 'pat_pre', 'pat_post', 'cfo_pre', 'cfo_post']
    missing = [c for c in required if c not in pair]
    if missing:
        raise ValueError(f'Panel missing paired analysis columns: {missing}')
    return pair

def estimate_accrual_architectures(panel: pd.DataFrame, settings: CompletionSettings, models: Mapping[str, Sequence[str]]=DEFAULT_MODELS, industry_column: str='icb_industry') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate pooled and sensitivity accrual architectures.

    Returns issuer-year/state residual rows and a model-estimation manifest. The
    three reference constructions are produced for every estimable architecture.
    """
    panel = panel.copy()
    pair = _design_pairs(panel, settings)
    source = panel.loc[panel.fiscal_year.between(settings.training_start_year, settings.test_end_year)].copy()
    result_rows: list[pd.DataFrame] = []
    manifest: list[dict] = []
    architectures = ['pooled', 'industry_fe', 'industry_slopes', 'trailing_pooled']
    for test_year in range(settings.test_start_year, settings.test_end_year + 1):
        pair_y = pair.loc[pair.fiscal_year.eq(test_year)].copy()
        if pair_y.empty:
            continue
        historical_all = source.loc[source.audit_status.eq(settings.audited_label) & source.fiscal_year.between(settings.training_start_year, test_year - 1)].copy()
        for model_name, features0 in models.items():
            features = list(features0)
            needed = ['ta_scaled', *features]
            if any((c not in source for c in needed)):
                manifest.append({'test_year': test_year, 'model': model_name, 'architecture': 'all', 'status': 'missing_columns', 'detail': ','.join((c for c in needed if c not in source))})
                continue
            for architecture in architectures:
                historical = historical_all.copy()
                if architecture == 'trailing_pooled':
                    historical = historical.loc[historical.fiscal_year.ge(max(settings.training_start_year, test_year - settings.trailing_years))]
                if architecture == 'industry_slopes':
                    if industry_column not in source or f'{industry_column}_pre' not in pair_y:
                        manifest.append({'test_year': test_year, 'model': model_name, 'architecture': architecture, 'status': 'missing_industry', 'detail': industry_column})
                        continue
                    groups = sorted(pair_y[f'{industry_column}_pre'].dropna().unique())
                else:
                    groups = [None]
                for group_value in groups:
                    train = historical.copy()
                    current = pair_y.copy()
                    group_label = 'all'
                    if architecture == 'industry_slopes':
                        group_label = str(group_value)
                        train = train.loc[train[industry_column].eq(group_value)]
                        current = current.loc[current[f'{industry_column}_pre'].eq(group_value)]
                    complete_train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
                    min_rows = settings.min_industry_rows if architecture == 'industry_slopes' else settings.min_train_rows
                    if len(complete_train) < min_rows or current.empty:
                        manifest.append({'test_year': test_year, 'model': model_name, 'architecture': architecture, 'group': group_label, 'status': 'insufficient_rows', 'train_rows': len(complete_train), 'current_rows': len(current)})
                        continue
                    fit_features = features.copy()
                    if architecture == 'industry_fe':
                        if industry_column not in complete_train or f'{industry_column}_pre' not in current:
                            manifest.append({'test_year': test_year, 'model': model_name, 'architecture': architecture, 'status': 'missing_industry', 'detail': industry_column})
                            continue
                        categories = sorted(complete_train[industry_column].dropna().astype(str).unique())
                        for cat in categories[1:]:
                            col = f'__ind_{cat}'
                            complete_train[col] = complete_train[industry_column].astype(str).eq(cat).astype(float)
                            current[f'{col}_pre'] = current[f'{industry_column}_pre'].astype(str).eq(cat).astype(float)
                            current[f'{col}_post'] = current.get(f'{industry_column}_post', current[f'{industry_column}_pre']).astype(str).eq(cat).astype(float)
                            fit_features.append(col)
                    bounds = _winsor_bounds(complete_train, ['ta_scaled', *fit_features], settings.winsor_lower, settings.winsor_upper)
                    fit_train = _apply_bounds(complete_train, bounds)
                    scaler, model, residual_sd = _fit_model(fit_train, fit_features)
                    for benchmark in BENCHMARKS:
                        current_b = current.copy()
                        state_data: dict[str, pd.DataFrame] = {}
                        for state in ('pre', 'post'):
                            x = pd.DataFrame(index=current_b.index)
                            for feat in fit_features:
                                if feat.startswith('__ind_'):
                                    x[feat] = current_b[f'{feat}_{state}']
                                    continue
                                if benchmark == 'version_specific':
                                    suffix = state
                                elif benchmark == 'pre_reference':
                                    suffix = 'pre'
                                else:
                                    suffix = 'post'
                                x[feat] = current_b[f'{feat}_{suffix}']
                            x['ta_scaled'] = current_b[f'ta_scaled_{state}']
                            x = _apply_bounds(x, bounds)
                            state_data[state] = x
                        valid = state_data['pre'].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
                        valid &= state_data['post'].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
                        if not valid.any():
                            continue
                        xpre, xpost = (state_data['pre'].loc[valid], state_data['post'].loc[valid])
                        nda_pre = model.predict(scaler.transform(xpre[fit_features]))
                        nda_post = model.predict(scaler.transform(xpost[fit_features]))
                        da_pre = xpre.ta_scaled.to_numpy(float) - nda_pre
                        da_post = xpost.ta_scaled.to_numpy(float) - nda_post
                        keys = current_b.loc[valid, KEYS].reset_index(drop=True)
                        out = keys.assign(model=model_name, architecture=architecture, architecture_group=group_label, benchmark=benchmark, da_pre=da_pre, da_post=da_post, nda_pre=nda_pre, nda_post=nda_post, signed_shift=da_post - da_pre, reduction=np.abs(da_pre) - np.abs(da_post), historical_residual_sd=residual_sd, train_rows=len(fit_train), train_min_year=int(fit_train.fiscal_year.min()), train_max_year=int(fit_train.fiscal_year.max()))
                        result_rows.append(out)
                    manifest.append({'test_year': test_year, 'model': model_name, 'architecture': architecture, 'group': group_label, 'status': 'estimated', 'train_rows': len(fit_train), 'residual_sd': residual_sd})
    if not result_rows:
        raise ValueError('No accrual architecture rows were estimated')
    return (pd.concat(result_rows, ignore_index=True), pd.DataFrame(manifest))

def _shapley_three(da_pre: np.ndarray, pat: np.ndarray, cfo: np.ndarray, benchmark: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moves = [pat, cfo, benchmark]
    n = len(da_pre)
    contrib = [np.zeros(n), np.zeros(n), np.zeros(n)]
    from itertools import permutations
    for order in permutations(range(3)):
        current = da_pre.copy()
        current_abs = np.abs(current)
        for idx in order:
            nxt = current + moves[idx]
            contrib[idx] += (current_abs - np.abs(nxt)) / 6.0
            current = nxt
            current_abs = np.abs(current)
    return (contrib[0], contrib[1], contrib[2])

def build_attribution_cases(accrual_rows: pd.DataFrame, panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    pair = paired_panel(panel, settings)
    needed = ['pat_pre', 'pat_post', 'cfo_pre', 'cfo_post', 'lag_assets_pre']
    missing = [c for c in needed if c not in pair]
    if missing:
        raise ValueError(f'Panel missing attribution columns: {missing}')
    pair = _numeric(pair, needed)
    pair['pat_move'] = (pair.pat_post - pair.pat_pre) / pair.lag_assets_pre
    pair['cfo_move'] = -(pair.cfo_post - pair.cfo_pre) / pair.lag_assets_pre
    x = accrual_rows.merge(pair[KEYS + ['pat_move', 'cfo_move']], on=KEYS, how='left', validate='many_to_one')
    x['benchmark_move'] = x.signed_shift - x.pat_move - x.cfo_move
    finite = np.isfinite(x[['da_pre', 'pat_move', 'cfo_move', 'benchmark_move']]).all(axis=1)
    x = x.loc[finite].copy()
    phi_pat, phi_cfo, phi_benchmark = _shapley_three(x.da_pre.to_numpy(float), x.pat_move.to_numpy(float), x.cfo_move.to_numpy(float), x.benchmark_move.to_numpy(float))
    x['phi_pat'] = phi_pat
    x['phi_cfo'] = phi_cfo
    x['phi_benchmark'] = phi_benchmark
    x['component_contrast'] = np.abs(phi_cfo) - np.abs(phi_pat)
    x['cfo_larger'] = np.abs(phi_cfo) > np.abs(phi_pat)
    sd = pd.to_numeric(x.historical_residual_sd, errors='coerce')
    x['normalised_component_contrast'] = np.where(sd > settings.negligible_sd, x.component_contrast / sd, np.nan)
    x['signed_quadrant'] = np.select([(x.phi_pat >= 0) & (x.phi_cfo >= 0), (x.phi_pat < 0) & (x.phi_cfo >= 0), (x.phi_pat >= 0) & (x.phi_cfo < 0)], ['both_reduce_abs_da', 'cfo_reduces_pat_increases', 'pat_reduces_cfo_increases'], default='both_increase_abs_da')
    x['signed_residual_direction'] = np.sign(x.signed_shift).astype(int)
    return x

def attribution_tables(cases: pd.DataFrame, settings: CompletionSettings) -> dict[str, pd.DataFrame]:
    summary: list[dict] = []
    quadrants: list[dict] = []
    for keys, group in cases.groupby(['model', 'architecture', 'benchmark'], observed=True):
        model, architecture, benchmark = keys
        med = cluster_bootstrap(group, lambda z: float(z.component_contrast.median()), draws=settings.bootstrap_draws, seed=settings.seed, null=0.0)
        share = cluster_bootstrap(group, lambda z: float(z.cfo_larger.mean()), draws=settings.bootstrap_draws, seed=settings.seed + 1, null=0.5)
        row = {'model': model, 'architecture': architecture, 'benchmark': benchmark, 'n': len(group), **{f'median_contrast_{k}': v for k, v in med.items()}, **{f'cfo_larger_{k}': v for k, v in share.items()}, 'median_normalised_contrast': float(group.normalised_component_contrast.median()), 'mean_reduction': float(group.reduction.mean()), 'trimmed_mean_reduction': float(stats.trim_mean(group.reduction.dropna(), 0.01))}
        summary.append(row)
        counts = group.signed_quadrant.value_counts(dropna=False)
        for q, count in counts.items():
            quadrants.append({'model': model, 'architecture': architecture, 'benchmark': benchmark, 'signed_quadrant': q, 'count': int(count), 'share': float(count / len(group))})
    return {'rq1_attribution_matrix': pd.DataFrame(summary), 'rq1_signed_quadrants': pd.DataFrame(quadrants)}
