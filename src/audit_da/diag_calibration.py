from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .bayes import ApproxHierarchicalBayes
from .stacking import stacking_weights


def _metrics(y,mu,sd):
    ok=np.isfinite(y)&np.isfinite(mu)&np.isfinite(sd)&(sd>0); y,mu,sd=y[ok],mu[ok],sd[ok]
    if not len(y): return {k:np.nan for k in ["n","rmse","mae","r2","mean_log_score","coverage80","coverage95","mean_predictive_sd"]}
    e=y-mu; den=((y-y.mean())**2).sum()
    return dict(n=len(y),rmse=float(np.sqrt((e**2).mean())),mae=float(np.abs(e).mean()),r2=float(1-(e**2).sum()/den) if den>0 else np.nan,mean_log_score=float(norm.logpdf(y,mu,sd).mean()),coverage80=float((np.abs(e)<=norm.ppf(.9)*sd).mean()),coverage95=float((np.abs(e)<=norm.ppf(.975)*sd).mean()),mean_predictive_sd=float(sd.mean()))


def _bounds(train,cols,lo,hi):
    return {c:(float(pd.to_numeric(train[c],errors="coerce").quantile(lo)),float(pd.to_numeric(train[c],errors="coerce").quantile(hi))) for c in cols}


def _clip(frame,bounds):
    x=frame.copy()
    for c,(lo,hi) in bounds.items(): x[c]=pd.to_numeric(x[c],errors="coerce").clip(lo,hi)
    return x


def rolling_calibration(panel,model_specs,minimum_train_rows,minimum_validation_rows,minimum_test_year,maximum_test_year,winsor_lower,winsor_upper,random_seed):
    audited=panel[panel.audit_status.eq("audited")].copy(); features=sorted({f for fs in model_specs.values() for f in fs}); rows=[]; wr=[]
    for year in range(minimum_test_year,maximum_test_year+1):
        train0=audited[audited.fiscal_year.le(year-1)].copy(); fit0=audited[audited.fiscal_year.le(year-2)].copy(); val0=audited[audited.fiscal_year.eq(year-1)].copy(); test0=audited[audited.fiscal_year.eq(year)].copy()
        if len(train0)<minimum_train_rows or test0.empty: continue
        b=_bounds(train0,["ta_scaled"]+features,winsor_lower,winsor_upper); train,fit,val,test=map(lambda z:_clip(z,b),[train0,fit0,val0,test0])
        vy=val.ta_scaled.to_numpy(float); common=np.isfinite(vy); vp={}
        for i,(name,fs) in enumerate(model_specs.items()):
            s=fit.replace([np.inf,-np.inf],np.nan).dropna(subset=["ta_scaled","firm_id"]+fs)
            if len(s)<minimum_train_rows: continue
            m=ApproxHierarchicalBayes(random_state=random_seed+year*100+i).fit(s[fs].to_numpy(float),s.ta_scaled.to_numpy(float),s.firm_id.astype(str).to_numpy(),fs)
            valid=val[fs].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy(); common &= valid
            p=m.posterior_mean_sd(val[fs].fillna(0).to_numpy(float),val.firm_id.astype(str).to_numpy(),True); vp[name]=(p.mean,p.sd)
        names=[n for n in model_specs if n in vp]
        if not names: continue
        weights=stacking_weights(vy[common],[vp[n][0][common] for n in names],[vp[n][1][common] for n in names]) if common.sum()>=minimum_validation_rows else np.repeat(1/len(names),len(names))
        wr += [dict(fiscal_year=year,model=n,weight=float(w),validation_rows=int(common.sum())) for n,w in zip(names,weights)]
        fitted={}; ok=np.isfinite(test.ta_scaled.to_numpy(float))
        for i,n in enumerate(names):
            fs=model_specs[n]; s=train.replace([np.inf,-np.inf],np.nan).dropna(subset=["ta_scaled","firm_id"]+fs)
            fitted[n]=ApproxHierarchicalBayes(random_state=random_seed+year*1000+i).fit(s[fs].to_numpy(float),s.ta_scaled.to_numpy(float),s.firm_id.astype(str).to_numpy(),fs)
            ok &= test[fs].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        idx=np.flatnonzero(ok)
        if not len(idx): continue
        yt=test.iloc[idx].ta_scaled.to_numpy(float); yr=test0.loc[test.index[idx],"ta_scaled"].to_numpy(float)
        for mode in ["conditional_existing_firm","marginal_new_firm"]:
            mus=[]; sds=[]
            for n in names:
                fs=model_specs[n]; ids=test.iloc[idx].firm_id.astype(str).to_numpy() if mode.startswith("conditional") else np.repeat("__NEW_FIRM__",len(idx))
                p=fitted[n].posterior_mean_sd(test.iloc[idx][fs].to_numpy(float),ids,True); mus.append(p.mean); sds.append(p.sd)
                for tv,y in [("winsorized_target",yt),("raw_target",yr)]: rows.append(dict(fiscal_year=year,model=n,prediction_mode=mode,target_variant=tv)|_metrics(y,p.mean,p.sd))
            M=np.vstack(mus).T; S=np.vstack(sds).T; mu=(M*weights).sum(1); sd=np.sqrt(np.maximum(((S**2+M**2)*weights).sum(1)-mu**2,1e-12))
            for tv,y in [("winsorized_target",yt),("raw_target",yr)]: rows.append(dict(fiscal_year=year,model="stacked_ensemble",prediction_mode=mode,target_variant=tv)|_metrics(y,mu,sd))
    out=pd.DataFrame(rows)
    if not out.empty:
        for c in ["rmse","mae","mean_log_score"]: out[f"{c}_year_z"]=out.groupby(["model","prediction_mode","target_variant"],observed=True)[c].transform(lambda x:(x-x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else 0.)
    return out,pd.DataFrame(wr)
