from __future__ import annotations
import numpy as np
import pandas as pd
from .core import KEYS, CompletionSettings, paired_panel
from .architecture import _shapley_three
from .switching import _mc_p

def time_shift_benchmarks(cases: pd.DataFrame, panel: pd.DataFrame, settings: CompletionSettings) -> pd.DataFrame:
    pair = paired_panel(panel, settings)
    industry_candidates = [c for c in ['icb_industry_pre', 'industry_pre', 'raw_exchange_pre'] if c in pair]
    industry_col = industry_candidates[0] if industry_candidates else None
    extra = KEYS + ([industry_col] if industry_col else [])
    base = cases.merge(pair[extra], on=KEYS, how='left', validate='many_to_one')
    rng = np.random.default_rng(settings.seed + 101)
    rows: list[dict] = []
    for (model, architecture, benchmark), g0 in base.groupby(['model', 'architecture', 'benchmark'], observed=True):
        if architecture != 'pooled':
            continue
        g = g0.sort_values(KEYS).copy().reset_index(drop=True)
        eligible_issuers = g.groupby('issuer_ticker').fiscal_year.nunique()
        eligible = set(eligible_issuers[eligible_issuers >= 2].index)
        g = g[g.issuer_ticker.isin(eligible)].copy().reset_index(drop=True)
        if g.empty:
            continue
        observed = float(g.component_contrast.median())
        sims = {'cyclic_within_issuer': [], 'independent_within_issuer': [], 'same_year_peer': []}
        issuer_indices = {k: idx.to_numpy() for k, idx in g.groupby('issuer_ticker').groups.items()}
        year_groups = {k: idx.to_numpy() for k, idx in g.groupby('fiscal_year').groups.items()}
        peer_groups = {k: idx.to_numpy() for k, idx in g.groupby(['fiscal_year', industry_col], dropna=False).groups.items()} if industry_col else {}
        for _ in range(settings.simulation_draws):
            donors_cyclic = np.arange(len(g))
            donors_independent = np.arange(len(g))
            for _, idx in issuer_indices.items():
                n = len(idx)
                lag = int(rng.integers(1, n))
                donors_cyclic[idx] = np.roll(idx, lag)
                for pos in idx:
                    choices = idx[idx != pos]
                    donors_independent[pos] = int(rng.choice(choices))
            donors_peer = np.arange(len(g))
            for pos, row in g.iterrows():
                candidates = peer_groups.get((row.fiscal_year, row[industry_col]), np.array([], dtype=int)) if industry_col else year_groups.get(row.fiscal_year, np.array([], dtype=int))
                candidates = candidates[candidates != pos]
                if len(candidates):
                    donors_peer[pos] = int(rng.choice(candidates))
            for name, donor in [('cyclic_within_issuer', donors_cyclic), ('independent_within_issuer', donors_independent), ('same_year_peer', donors_peer)]:
                phi_pat, phi_cfo, _ = _shapley_three(g.da_pre.to_numpy(float), g.pat_move.to_numpy(float)[donor], g.cfo_move.to_numpy(float)[donor], g.benchmark_move.to_numpy(float))
                sims[name].append(float(np.median(np.abs(phi_cfo) - np.abs(phi_pat))))
        for name, values0 in sims.items():
            values = np.asarray(values0, float)
            rows.append({'model': model, 'architecture': architecture, 'benchmark': benchmark, 'donor_design': name, 'n': len(g), 'observed_median_contrast': observed, 'sim_mean': float(values.mean()), 'sim_median': float(np.median(values)), 'sim_p025': float(np.quantile(values, 0.025)), 'sim_p975': float(np.quantile(values, 0.975)), 'observed_minus_sim_median': float(observed - np.median(values)), 'mc_p': _mc_p(observed, values), 'draws': settings.simulation_draws, 'seed': settings.seed + 101})
    return pd.DataFrame(rows)
