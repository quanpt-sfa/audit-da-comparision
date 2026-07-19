from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from .diag_common import KEYS, trimmed_mean


def _codes(meta, columns, minimum_size):
    cols=[c for c in columns if c in meta.columns]
    if not cols: return np.zeros(len(meta),int)
    keys=meta[cols].fillna("__MISSING__").astype(str).agg("|".join,axis=1); counts=keys.value_counts()
    if "fiscal_year" in cols: keys=np.where(keys.map(counts).to_numpy()<minimum_size,meta.fiscal_year.astype(str)+"|__POOLED__",keys)
    else: keys=np.where(keys.map(counts).to_numpy()<minimum_size,"__POOLED__",keys)
    return pd.factorize(keys,sort=True)[0]


def _permute(values,codes,rng):
    out=np.empty_like(values)
    for code in np.unique(codes):
        idx=np.flatnonzero(codes==code); out[idx]=values[idx][rng.permutation(len(idx))]
    return out


def directional_placebo(baseline: pd.DataFrame,panel: pd.DataFrame,models: Iterable[str],benchmarks: Iterable[str],
                        strata_columns:list[str],minimum_stratum_size:int,permutations:int,trim_fraction:float,
                        random_seed:int,identity_tolerance=1e-8):
    extra=[c for c in strata_columns if c not in KEYS and c in panel.columns]
    meta=panel[panel.audit_status.eq("unaudited")].drop_duplicates(KEYS)[KEYS+extra]
    rng=np.random.default_rng(random_seed); summaries=[]; draws=[]
    for model in models:
        for benchmark in benchmarks:
            g=baseline[baseline.model.eq(model)&baseline.benchmark.eq(benchmark)].merge(meta,on=KEYS,how="left",validate="many_to_one")
            if g.empty: continue
            identity=float(np.max(np.abs(g.signed_shift.to_numpy(float)-g.raw_ta_shift.to_numpy(float))))
            if identity>identity_tolerance: raise ValueError(f"Placebo requires common benchmark: {model}/{benchmark}, error={identity:.3g}")
            pre=g.da_pre.to_numpy(float); adj=g.raw_ta_shift.to_numpy(float); codes=_codes(g,strata_columns,minimum_stratum_size)
            means=pd.Series(adj).groupby(codes).transform("mean").to_numpy(); centered=adj-means; real=g.reduction.to_numpy(float)
            real_mean=float(real.mean()); real_trim=trimmed_mean(real,trim_fraction)
            for kind in ["raw_permutation","centered_permutation","symmetric_sign"]:
                pm=np.empty(permutations); pt=np.empty(permutations); pp=np.empty(permutations); pn=np.empty(permutations)
                for b in range(permutations):
                    if kind=="raw_permutation": eta=_permute(adj,codes,rng)
                    elif kind=="centered_permutation": eta=_permute(centered,codes,rng)
                    else: eta=_permute(np.abs(centered),codes,rng)*rng.choice([-1.0,1.0],size=len(adj))
                    r=np.abs(pre)-np.abs(pre+eta); pm[b]=r.mean(); pt[b]=trimmed_mean(r,trim_fraction); pp[b]=(r>0).mean(); pn[b]=(r<0).mean()
                    draws.append(dict(model=model,benchmark=benchmark,placebo_type=kind,permutation=b,mean_reduction=pm[b],trimmed_mean_reduction=pt[b],share_positive=pp[b],share_negative=pn[b]))
                summaries.append(dict(model=model,benchmark=benchmark,placebo_type=kind,rows=len(g),max_common_benchmark_identity_error=identity,
                    real_mean_reduction=real_mean,real_trimmed_mean_reduction=real_trim,placebo_mean=float(pm.mean()),placebo_q025=float(np.quantile(pm,.025)),placebo_q975=float(np.quantile(pm,.975)),placebo_trimmed_mean=float(pt.mean()),corrective_excess_mean=real_mean-float(pm.mean()),corrective_excess_trimmed=real_trim-float(pt.mean()),randomization_p_ge_real=float((1+(pm>=real_mean).sum())/(permutations+1)),share_placebo_mean_negative=float((pm<0).mean()),placebo_positive_minus_negative_share=float((pp-pn).mean())))
    return pd.DataFrame(summaries),pd.DataFrame(draws)
