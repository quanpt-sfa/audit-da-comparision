from __future__ import annotations
import numpy as np
import pandas as pd
from .core import KEYS, CompletionSettings, paired_panel, cluster_bootstrap, _find_column, _design_matrix, _cluster_ols, _adjust_pvalues
from .switching import _mc_p

def applied_consequence_tables(accrual_rows: pd.DataFrame, panel: pd.DataFrame, settings: CompletionSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = paired_panel(panel, settings)
    industry_col = _find_column(pair, ['icb_industry_pre', 'industry_pre'])
    aliases = {'big4': ['big4_pre', 'is_big4_pre', 'big_four_pre'], 'short_debt': ['short_term_debt_intensity_pre', 'short_term_debt_pre', 'std_intensity_pre'], 'loss': ['loss_pre'], 'roa': ['roa_pre'], 'current_ratio': ['current_ratio_pre']}
    resolved = {k: _find_column(pair, v) for k, v in aliases.items()}
    if 'lag_assets_pre' in pair:
        pair['__log_assets'] = np.log(np.maximum(pd.to_numeric(pair.lag_assets_pre, errors='coerce').abs(), 1.0))
    elif 'assets_pre' in pair:
        pair['__log_assets'] = np.log(np.maximum(pd.to_numeric(pair.assets_pre, errors='coerce').abs(), 1.0))
    else:
        pair['__log_assets'] = np.nan
    rows: list[dict] = []
    manifest: list[dict] = []
    primary = accrual_rows[(accrual_rows.architecture == 'pooled') & (accrual_rows.benchmark == 'audited_reference')].copy()
    for model, g in primary.groupby('model', observed=True):
        data = g.merge(pair, on=KEYS, how='left', validate='many_to_one')
        cut = data.da_post.abs().quantile(settings.tail_quantile)
        data['high_da_pre'] = data.da_pre.abs().ge(cut).astype(float)
        data['high_da_post'] = data.da_post.abs().ge(cut).astype(float)
        for focal_name in ['big4', 'short_debt', 'loss']:
            focal = resolved[focal_name]
            if focal is None:
                manifest.append({'model': model, 'design': focal_name, 'status': 'missing_focal_column'})
                continue
            controls = ['__log_assets'] + [c for name, c in resolved.items() if name not in {focal_name} and c is not None]
            for outcome_name, pre_col, post_col in [('signed_da', 'da_pre', 'da_post'), ('high_da', 'high_da_pre', 'high_da_post')]:
                needed = [focal, *controls, pre_col, post_col, 'issuer_ticker', 'fiscal_year'] + ([industry_col] if industry_col else [])
                d = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
                if len(d) < 50:
                    manifest.append({'model': model, 'design': f'{focal_name}_{outcome_name}', 'status': 'insufficient_rows', 'rows': len(d)})
                    continue
                x, names = _design_matrix(d, [focal, *controls], industry_col=industry_col)
                focal_idx = names.index(focal)
                b_pre, se_pre, p_pre = _cluster_ols(d[pre_col].to_numpy(float), x, d.issuer_ticker.to_numpy())
                b_post, se_post, p_post = _cluster_ols(d[post_col].to_numpy(float), x, d.issuer_ticker.to_numpy())
                stacked = pd.concat([d.assign(__y=d[pre_col], __post=0.0, __interaction=0.0), d.assign(__y=d[post_col], __post=1.0, __interaction=pd.to_numeric(d[focal], errors='coerce'))], ignore_index=True)
                stacked['__focal'] = pd.to_numeric(stacked[focal], errors='coerce')
                xs, snames = _design_matrix(stacked, ['__focal', '__post', '__interaction', *controls], industry_col=industry_col)
                inter_idx = snames.index('__interaction')
                b_st, se_st, p_st = _cluster_ols(stacked.__y.to_numpy(float), xs, stacked.issuer_ticker.to_numpy())
                d['__diff'] = d[post_col] - d[pre_col]
                xd, dnames = _design_matrix(d, [focal, *controls], industry_col=industry_col)
                diff_idx = dnames.index(focal)
                b_diff, se_diff, p_diff = _cluster_ols(d.__diff.to_numpy(float), xd, d.issuer_ticker.to_numpy())
                pre_sd = float(d[pre_col].std(ddof=1))
                rows.append({'model': model, 'focal': focal_name, 'outcome': outcome_name, 'n': len(d), 'issuers': d.issuer_ticker.nunique(), 'pre_beta': b_pre[focal_idx], 'pre_se': se_pre[focal_idx], 'pre_p': p_pre[focal_idx], 'post_beta': b_post[focal_idx], 'post_se': se_post[focal_idx], 'post_p': p_post[focal_idx], 'beta_difference': b_post[focal_idx] - b_pre[focal_idx], 'standardised_beta_difference': (b_post[focal_idx] - b_pre[focal_idx]) / pre_sd if pre_sd > 0 else np.nan, 'interaction_beta': b_st[inter_idx], 'interaction_se': se_st[inter_idx], 'interaction_p': p_st[inter_idx], 'paired_difference_beta': b_diff[diff_idx], 'paired_difference_se': se_diff[diff_idx], 'paired_difference_p': p_diff[diff_idx], 'significance_status_switch': (p_pre[focal_idx] < 0.05) != (p_post[focal_idx] < 0.05)})
                manifest.append({'model': model, 'design': f'{focal_name}_{outcome_name}', 'status': 'estimated', 'rows': len(d)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['interaction_q_bh'] = _adjust_pvalues(out.interaction_p.fillna(1.0), 'bh')
        out['paired_difference_q_bh'] = _adjust_pvalues(out.paired_difference_p.fillna(1.0), 'bh')
    return (out, pd.DataFrame(manifest))

def supplemental_inference(concentration: pd.DataFrame | None, near_zero: pd.DataFrame | None, settings: CompletionSettings) -> pd.DataFrame:
    rows: list[dict] = []
    if concentration is not None and (not concentration.empty) and ('excess_nhhi' in concentration):
        x = concentration.replace([np.inf, -np.inf], np.nan).dropna(subset=['excess_nhhi'])
        boot = cluster_bootstrap(x, lambda z: float(z.excess_nhhi.mean()), draws=settings.bootstrap_draws, seed=settings.seed, null=0.0)
        rows.append({'diagnostic': 'excess_nhhi', **boot, 'share_positive': float(x.excess_nhhi.gt(0).mean()), 'n': len(x)})
    if near_zero is not None and (not near_zero.empty) and {'draw', 'statistic'}.issubset(near_zero):
        observed_rows = near_zero.loc[near_zero.draw.astype(str).eq('observed'), 'statistic']
        simulated = pd.to_numeric(near_zero.loc[~near_zero.draw.astype(str).eq('observed'), 'statistic'], errors='coerce').dropna().to_numpy(float)
        if len(observed_rows) and len(simulated):
            observed = float(observed_rows.iloc[0])
            rows.append({'diagnostic': 'near_zero_random_sign', 'estimate': observed, 'ci_low': float(np.quantile(simulated, 0.025)), 'ci_high': float(np.quantile(simulated, 0.975)), 'p_directional': _mc_p(abs(observed), np.abs(simulated)), 'n': len(simulated)})
    return pd.DataFrame(rows)
