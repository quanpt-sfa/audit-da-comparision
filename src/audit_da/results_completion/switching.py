from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from .core import KEYS, CompletionSettings, paired_panel, cluster_bootstrap, _adjust_pvalues, _numeric

def direct_revision_tables(panel: pd.DataFrame, settings: CompletionSettings) -> dict[str, pd.DataFrame]:
    pair = paired_panel(panel, settings)
    pair = _numeric(pair, ['pat_pre', 'pat_post', 'cfo_pre', 'cfo_post', 'lag_assets_pre'])
    pair['delta_pat'] = (pair.pat_post - pair.pat_pre) / pair.lag_assets_pre
    pair['delta_cfo'] = (pair.cfo_post - pair.cfo_pre) / pair.lag_assets_pre
    finite = np.isfinite(pair[['delta_pat', 'delta_cfo']]).all(axis=1)
    pair = pair.loc[finite].copy()
    symmetric = []
    for cutoff in settings.direct_thresholds:
        pat = pair.delta_pat.abs() >= cutoff
        cfo = pair.delta_cfo.abs() >= cutoff
        symmetric.extend([{'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'pat_material', 'share': float(pat.mean()), 'n': len(pair)}, {'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'cfo_material', 'share': float(cfo.mean()), 'n': len(pair)}, {'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'cfo_only', 'share': float((cfo & ~pat).mean()), 'n': len(pair)}, {'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'pat_only', 'share': float((pat & ~cfo).mean()), 'n': len(pair)}, {'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'both', 'share': float((pat & cfo).mean()), 'n': len(pair)}, {'panel': 'symmetric', 'cutoff': cutoff, 'classification': 'neither', 'share': float((~pat & ~cfo).mean()), 'n': len(pair)}])
    pat_stable = pair.delta_pat.abs() <= 0.005
    cfo_material = pair.delta_cfo.abs() >= 0.01
    asymmetric = pd.DataFrame([{'panel': 'asymmetric', 'classification': 'cfo_material_pat_stable', 'share': float((cfo_material & pat_stable).mean()), 'n': len(pair)}, {'panel': 'asymmetric', 'classification': 'pat_not_stable_cfo_not_material', 'share': float((~pat_stable & ~cfo_material).mean()), 'n': len(pair)}])
    return {'direct_revision_cases': pair, 'direct_revision_symmetric': pd.DataFrame(symmetric), 'direct_revision_asymmetric': asymmetric}

def _midrank_against_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = np.sort(pd.to_numeric(reference, errors='coerce').dropna().to_numpy(float))
    x = pd.to_numeric(values, errors='coerce').to_numpy(float)
    if not len(ref):
        return pd.Series(np.nan, index=values.index)
    left = np.searchsorted(ref, x, side='left')
    right = np.searchsorted(ref, x, side='right')
    rank = (left + right) / (2.0 * len(ref))
    rank[~np.isfinite(x)] = np.nan
    return pd.Series(rank, index=values.index)

def _common_categories(pre: pd.Series, post: pd.Series, q: int=5) -> tuple[pd.Series, pd.Series, np.ndarray]:
    ref = pd.to_numeric(post, errors='coerce').dropna()
    cuts = np.unique(ref.quantile(np.linspace(0, 1, q + 1)).to_numpy(float))
    if len(cuts) < 3:
        return (pd.Series(np.nan, index=pre.index), pd.Series(np.nan, index=post.index), cuts)
    cuts[0], cuts[-1] = (-np.inf, np.inf)
    return (pd.cut(pre, cuts, labels=False, include_lowest=True), pd.cut(post, cuts, labels=False, include_lowest=True), cuts)

def _profit_gate(pair: pd.DataFrame, threshold: float) -> pd.Series:
    pat_pre = pd.to_numeric(pair.pat_pre, errors='coerce')
    pat_post = pd.to_numeric(pair.pat_post, errors='coerce')
    assets = pd.to_numeric(pair.lag_assets_pre, errors='coerce').abs()
    sign_change = np.signbit(pat_pre) != np.signbit(pat_post)
    ratio = (pat_post - pat_pre).abs() / np.maximum(pat_pre.abs(), 0.001 * assets)
    return pd.Series(sign_change | ratio.ge(threshold), index=pair.index)

def switching_cases(accrual_rows: pd.DataFrame, panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    pair = paired_panel(panel, settings)
    pair = _numeric(pair, ['pat_pre', 'pat_post', 'cfo_pre', 'cfo_post', 'lag_assets_pre', 'ta_scaled_pre', 'ta_scaled_post'])
    base = pair[KEYS + [c for c in pair.columns if c not in KEYS]].copy()
    base['gate_0_05'] = _profit_gate(base, 0.05)
    base['cfo_sign_switch'] = np.signbit(base.cfo_pre) != np.signbit(base.cfo_post)
    base['cfo_sign_magnitude'] = (base.cfo_post - base.cfo_pre).abs() / base.lag_assets_pre.abs()
    category_frames = []
    for year, g in base.groupby('fiscal_year', observed=True):
        pre_cat, post_cat, _ = _common_categories(g.cfo_pre / g.lag_assets_pre, g.cfo_post / g.lag_assets_pre, 5)
        tmp = g.copy()
        tmp['cfo_category_pre'] = pre_cat
        tmp['cfo_category_post'] = post_cat
        tmp['cfo_category_switch'] = pre_cat.ne(post_cat) & pre_cat.notna() & post_cat.notna()
        tmp['cfo_category_distance'] = (post_cat - pre_cat).abs()
        abs_ta_post = tmp.ta_scaled_post.abs()
        ta_cut = float(abs_ta_post.quantile(settings.tail_quantile))
        tmp['high_ta_pre'] = tmp.ta_scaled_pre.abs().ge(ta_cut)
        tmp['high_ta_post'] = tmp.ta_scaled_post.abs().ge(ta_cut)
        tmp['high_ta_switch'] = tmp.high_ta_pre.ne(tmp.high_ta_post)
        tmp['high_ta_magnitude'] = (tmp.ta_scaled_post.abs() - tmp.ta_scaled_pre.abs()).abs()
        category_frames.append(tmp)
    direct = pd.concat(category_frames, ignore_index=True)
    rows: list[pd.DataFrame] = []
    for (model, architecture, benchmark, year), g in accrual_rows.groupby(['model', 'architecture', 'benchmark', 'fiscal_year'], observed=True):
        if architecture != 'pooled':
            continue
        ref = g.da_post.abs()
        cut = float(ref.quantile(settings.tail_quantile))
        tmp = g.merge(direct[KEYS + ['gate_0_05']], on=KEYS, how='left', validate='many_to_one')
        tmp['da_sign_switch'] = np.signbit(tmp.da_pre) != np.signbit(tmp.da_post)
        tmp['da_sign_magnitude'] = tmp.signed_shift.abs()
        tmp['high_da_pre'] = tmp.da_pre.abs().ge(cut)
        tmp['high_da_post'] = tmp.da_post.abs().ge(cut)
        tmp['high_da_switch'] = tmp.high_da_pre.ne(tmp.high_da_post)
        tmp['high_da_magnitude'] = (tmp.da_post.abs() - tmp.da_pre.abs()).abs()
        tmp['rank_pre'] = _midrank_against_reference(tmp.da_pre.abs(), ref)
        tmp['rank_post'] = _midrank_against_reference(tmp.da_post.abs(), ref)
        tmp['rank_displacement'] = (tmp.rank_post - tmp.rank_pre).abs()
        rows.append(tmp)
    model_cases = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    direct['outcome_scope'] = 'direct'
    model_cases['outcome_scope'] = 'model'
    return (direct, model_cases)

def _jaccard(a: pd.Series, b: pd.Series) -> float:
    a = a.fillna(False).astype(bool)
    b = b.fillna(False).astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan

def switching_tables(direct: pd.DataFrame, model_cases: pd.DataFrame, settings: CompletionSettings) -> dict[str, pd.DataFrame]:
    summary: list[dict] = []
    magnitudes: list[dict] = []
    jaccard: list[dict] = []
    direct_specs = [('cfo_sign', 'cfo_sign_switch', 'cfo_sign_magnitude'), ('cfo_category', 'cfo_category_switch', 'cfo_category_distance'), ('high_ta', 'high_ta_switch', 'high_ta_magnitude')]
    for name, switch_col, mag_col in direct_specs:
        valid = direct.dropna(subset=[switch_col, 'gate_0_05']).copy()
        rate = cluster_bootstrap(valid, lambda z, c=switch_col: float(z[c].mean()), draws=settings.bootstrap_draws, seed=settings.seed)
        switched = valid.loc[valid[switch_col]].copy()
        outside = cluster_bootstrap(switched, lambda z: float((~z.gate_0_05).mean()), draws=settings.bootstrap_draws, seed=settings.seed + 7, null=0.5) if len(switched) else {'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_directional': np.nan}
        summary.append({'outcome': name, 'model': 'direct', **{f'switch_{k}': v for k, v in rate.items()}, **{f'outside_gate_{k}': v for k, v in outside.items()}, 'switch_n': len(switched), 'denominator': len(valid)})
        if len(switched):
            x = pd.to_numeric(switched[mag_col], errors='coerce')
            magnitudes.append({'outcome': name, 'model': 'direct', 'n': int(x.notna().sum()), 'median': float(x.median()), 'p75': float(x.quantile(0.75)), 'p90': float(x.quantile(0.9))})
    if not model_cases.empty:
        for (model, benchmark), g in model_cases.groupby(['model', 'benchmark'], observed=True):
            for name, switch_col, mag_col in [('da_sign', 'da_sign_switch', 'da_sign_magnitude'), ('high_da', 'high_da_switch', 'high_da_magnitude')]:
                valid = g.dropna(subset=[switch_col, 'gate_0_05']).copy()
                rate = cluster_bootstrap(valid, lambda z, c=switch_col: float(z[c].mean()), draws=settings.bootstrap_draws, seed=settings.seed)
                switched = valid.loc[valid[switch_col]].copy()
                outside = cluster_bootstrap(switched, lambda z: float((~z.gate_0_05).mean()), draws=settings.bootstrap_draws, seed=settings.seed + 7, null=0.5) if len(switched) else {'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_directional': np.nan}
                summary.append({'outcome': name, 'model': model, 'benchmark': benchmark, **{f'switch_{k}': v for k, v in rate.items()}, **{f'outside_gate_{k}': v for k, v in outside.items()}, 'switch_n': len(switched), 'denominator': len(valid)})
                if len(switched):
                    x = pd.to_numeric(switched[mag_col], errors='coerce')
                    magnitudes.append({'outcome': name, 'model': model, 'benchmark': benchmark, 'n': int(x.notna().sum()), 'median': float(x.median()), 'p75': float(x.quantile(0.75)), 'p90': float(x.quantile(0.9))})
            rank = pd.to_numeric(g.rank_displacement, errors='coerce')
            magnitudes.append({'outcome': 'common_cdf_rank_displacement', 'model': model, 'benchmark': benchmark, 'n': int(rank.notna().sum()), 'median': float(rank.median()), 'p75': float(rank.quantile(0.75)), 'p90': float(rank.quantile(0.9))})
            jaccard.append({'model': model, 'benchmark': benchmark, 'fiscal_year': 'pooled', 'jaccard_high_da': _jaccard(g.high_da_pre, g.high_da_post)})
            for year, gy in g.groupby('fiscal_year', observed=True):
                jaccard.append({'model': model, 'benchmark': benchmark, 'fiscal_year': int(year), 'jaccard_high_da': _jaccard(gy.high_da_pre, gy.high_da_post)})
    return {'rq2_switch_summary': pd.DataFrame(summary), 'rq2_switch_magnitudes': pd.DataFrame(magnitudes), 'rq2_jaccard': pd.DataFrame(jaccard)}

def profit_gate_sensitivity(direct: pd.DataFrame, model_cases: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold in settings.profit_thresholds:
        d = direct.copy()
        d['gate'] = _profit_gate(d, threshold)
        for outcome, switch in [('cfo_sign', 'cfo_sign_switch'), ('cfo_category', 'cfo_category_switch'), ('high_ta', 'high_ta_switch')]:
            sw = d.loc[d[switch]]
            rows.append({'threshold': threshold, 'outcome': outcome, 'model': 'direct', 'switch_n': len(sw), 'outside_gate_share': float((~sw.gate).mean()) if len(sw) else np.nan})
        if not model_cases.empty:
            for (model, benchmark), g in model_cases.groupby(['model', 'benchmark'], observed=True):
                merged = g.drop(columns=['gate_0_05'], errors='ignore').merge(d[KEYS + ['gate']], on=KEYS, how='left', validate='many_to_one')
                for outcome, switch in [('da_sign', 'da_sign_switch'), ('high_da', 'high_da_switch')]:
                    sw = merged.loc[merged[switch]]
                    rows.append({'threshold': threshold, 'outcome': outcome, 'model': model, 'benchmark': benchmark, 'switch_n': len(sw), 'outside_gate_share': float((~sw.gate).mean()) if len(sw) else np.nan})
    return pd.DataFrame(rows)

def _mc_p(observed: float, simulated: np.ndarray, greater: bool=True) -> float:
    count = np.sum(simulated >= observed) if greater else np.sum(simulated <= observed)
    return float((1 + count) / (len(simulated) + 1))

def randomisation_benchmarks(direct: pd.DataFrame, model_cases: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed)
    rows: list[dict] = []
    def simulate(values_pre: np.ndarray, shifts: np.ndarray, observed_switch: float, label: dict) -> None:
        nonzero = shifts != 0
        symmetric = np.empty(settings.simulation_draws)
        reassigned = np.empty(settings.simulation_draws)
        for b in range(settings.simulation_draws):
            signs = rng.choice([-1.0, 1.0], size=len(shifts))
            sim_shift = np.where(nonzero, np.abs(shifts) * signs, 0.0)
            symmetric[b] = np.mean(np.signbit(values_pre) != np.signbit(values_pre + sim_shift))
            perm = rng.permutation(shifts)
            reassigned[b] = np.mean(np.signbit(values_pre) != np.signbit(values_pre + perm))
        for benchmark, sims in [('symmetric_sign', symmetric), ('signed_shift_reassignment', reassigned)]:
            rows.append({**label, 'benchmark': benchmark, 'observed_switch_rate': observed_switch, 'sim_mean': float(sims.mean()), 'sim_p025': float(np.quantile(sims, 0.025)), 'sim_p975': float(np.quantile(sims, 0.975)), 'mc_p': _mc_p(observed_switch, sims), 'draws': settings.simulation_draws, 'seed': settings.seed})
    d = direct.dropna(subset=['cfo_pre', 'cfo_post']).copy()
    simulate(d.cfo_pre.to_numpy(float), (d.cfo_post - d.cfo_pre).to_numpy(float), float(d.cfo_sign_switch.mean()), {'outcome': 'cfo_sign', 'model': 'direct', 'benchmark_model': 'direct'})
    if not model_cases.empty:
        for (model, benchmark), g in model_cases.groupby(['model', 'benchmark'], observed=True):
            simulate(g.da_pre.to_numpy(float), g.signed_shift.to_numpy(float), float(g.da_sign_switch.mean()), {'outcome': 'da_sign', 'model': model, 'benchmark_model': benchmark})
    return pd.DataFrame(rows)
