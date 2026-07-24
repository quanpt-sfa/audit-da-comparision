from __future__ import annotations
import numpy as np
import pandas as pd
from .core import DEFAULT_MODELS, _adjust_pvalues

def confirmatory_summary(attribution: pd.DataFrame, switch: pd.DataFrame, randomisation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    primary = attribution[(attribution.architecture == 'pooled') & (attribution.benchmark == 'audited_reference') & attribution.model.isin(['jones', 'modified_jones'])]
    rq1_p = float(primary.median_contrast_p_directional.max()) if len(primary) == 2 else np.nan
    rows.append({'family': 'rq1_attribution', 'component_count': len(primary), 'family_p': rq1_p, 'decision_rule': 'Jones and modified-Jones median component contrasts > 0'})
    coverage_names = {'cfo_sign', 'cfo_category', 'high_ta'}
    cov = switch[(switch.model == 'direct') & switch.outcome.isin(coverage_names)]
    rq2_cov_p = float(cov.outside_gate_p_directional.max()) if len(cov) == 3 else np.nan
    rows.append({'family': 'rq2_coverage', 'component_count': len(cov), 'family_p': rq2_cov_p, 'decision_rule': 'All three outside-gate shares > 0.5'})
    sym = randomisation[randomisation.benchmark.eq('symmetric_sign')]
    needed = sym[sym.outcome.eq('cfo_sign') | (sym.outcome.eq('da_sign') & sym.model.isin(DEFAULT_MODELS) & sym.benchmark_model.eq('audited_reference'))]
    rq2_sw_p = float(needed.mc_p.max()) if len(needed) == 5 else np.nan
    rows.append({'family': 'rq2_excess_switching', 'component_count': len(needed), 'family_p': rq2_sw_p, 'decision_rule': 'CFO and all four audited-reference DA sign-switch rates exceed symmetric benchmark'})
    out = pd.DataFrame(rows)
    valid = out.family_p.notna()
    out.loc[valid, 'holm_p'] = _adjust_pvalues(out.loc[valid, 'family_p'], 'holm')
    out['confirmatory_reject'] = out.holm_p.lt(0.05)
    return out
