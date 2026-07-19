from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from .diag_common import KEYS, paired_panel


def family_discordance(baseline: pd.DataFrame,panel: pd.DataFrame,families:dict[str,list[str]],tolerances:Iterable[float]):
    pivot=baseline.pivot_table(index=KEYS+["benchmark"],columns="model",values="reduction",aggfunc="first").reset_index()
    move=baseline.groupby(KEYS+["benchmark"],observed=True).agg(raw_ta_shift=("raw_ta_shift","first"),signed_shift=("signed_shift","first")).reset_index(); pivot=pivot.merge(move,on=KEYS+["benchmark"],validate="one_to_one")
    for fam,members in families.items():
        have=[m for m in members if m in pivot]
        if not have: raise ValueError(f"No models for family {fam}")
        pivot[f"{fam}_reduction"]=pivot[have].mean(1); pivot[f"{fam}_model_sd"]=pivot[have].std(1,ddof=0)
    if len(families)!=2: raise ValueError("Exactly two model families required")
    left,right=list(families); pair=paired_panel(panel)
    for name in ["roa","revenue","receivables","ta_scaled","total_accruals"]:
        a,b=f"{name}_pre",f"{name}_post"
        if a in pair and b in pair: pair[f"delta_{name}"]=pd.to_numeric(pair[b],errors="coerce")-pd.to_numeric(pair[a],errors="coerce")
    if {"ta_source_pre","ta_source_post"}.issubset(pair): pair["ta_source_mismatch"]=pair.ta_source_pre.ne(pair.ta_source_post)
    pivot=pivot.merge(pair,on=KEYS,how="left",validate="many_to_one"); summaries=[]; years=[]; cases=[]
    for tol in tolerances:
        lc=np.select([pivot[f"{left}_reduction"].gt(tol),pivot[f"{left}_reduction"].lt(-tol)],[1,-1],default=0); rc=np.select([pivot[f"{right}_reduction"].gt(tol),pivot[f"{right}_reduction"].lt(-tol)],[1,-1],default=0)
        w=pivot.copy(); w["tolerance"]=float(tol); w[f"{left}_class"]=lc; w[f"{right}_class"]=rc; w["hard_opposite_sign"]=(lc*rc)==-1; w["any_family_discordance"]=lc!=rc; w["family_gap"]=w[f"{left}_reduction"]-w[f"{right}_reduction"]
        w["discordance_category"]=np.select([(lc==1)&(rc==-1),(lc==-1)&(rc==1),(lc==0)&(rc!=0),(lc!=0)&(rc==0)],[f"{left}_improve__{right}_deteriorate",f"{left}_deteriorate__{right}_improve",f"{left}_near_zero",f"{right}_near_zero"],default="agreement")
        cases.append(w[w.any_family_discordance])
        for bench,g in w.groupby("benchmark",observed=True): summaries.append(dict(benchmark=bench,tolerance=float(tol),rows=len(g),any_discordance_share=float(g.any_family_discordance.mean()),hard_opposite_sign_share=float(g.hard_opposite_sign.mean()),mean_abs_family_gap=float(g.family_gap.abs().mean()),median_abs_family_gap=float(g.family_gap.abs().median()),mean_abs_raw_ta_shift_discordant=float(g.loc[g.any_family_discordance,"raw_ta_shift"].abs().mean())))
        for (bench,year),g in w.groupby(["benchmark","fiscal_year"],observed=True): years.append(dict(benchmark=bench,fiscal_year=year,tolerance=float(tol),rows=len(g),any_discordance_share=float(g.any_family_discordance.mean()),hard_opposite_sign_share=float(g.hard_opposite_sign.mean())))
    case=pd.concat(cases,ignore_index=True) if cases else pd.DataFrame(); cov=[]
    if not case.empty:
        primary=case[case.tolerance.eq(min(tolerances))]
        for var in ["raw_ta_shift","delta_roa","delta_revenue","delta_receivables","delta_ta_scaled","delta_total_accruals"]:
            if var not in primary: continue
            for status,g in primary.groupby("any_family_discordance",observed=True):
                v=pd.to_numeric(g[var],errors="coerce"); cov.append(dict(variable=var,discordant=bool(status),rows=int(v.notna().sum()),mean=float(v.mean()),median=float(v.median()),mean_absolute=float(v.abs().mean())))
    return {"family_discordance_summary":pd.DataFrame(summaries),"family_discordance_by_year":pd.DataFrame(years),"family_discordance_cases":case,"family_discordance_covariates":pd.DataFrame(cov)}
