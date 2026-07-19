from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from .bayes import ApproxHierarchicalBayes
from .stacking import solve_stacking


def _metrics(y, mu, sd, student_t_dfs=(3, 5, 10, 30)):
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    y, mu, sd = y[ok], mu[ok], sd[ok]
    keys = ["n", "rmse", "mae", "r2", "mean_log_score", "coverage80", "coverage95", "mean_predictive_sd"]
    if not len(y):
        out = {k: np.nan for k in keys}
        for df in student_t_dfs:
            out[f"mean_student_t_log_score_df{df}"] = np.nan
        return out
    e = y - mu
    den = ((y - y.mean()) ** 2).sum()
    out = dict(
        n=len(y),
        rmse=float(np.sqrt((e ** 2).mean())),
        mae=float(np.abs(e).mean()),
        r2=float(1 - (e ** 2).sum() / den) if den > 0 else np.nan,
        mean_log_score=float(norm.logpdf(y, mu, sd).mean()),
        coverage80=float((np.abs(e) <= norm.ppf(.9) * sd).mean()),
        coverage95=float((np.abs(e) <= norm.ppf(.975) * sd).mean()),
        mean_predictive_sd=float(sd.mean()),
    )
    z = e / sd
    for df in student_t_dfs:
        # Scale the Student-t so its variance matches sd^2 when df > 2.
        scale = sd * np.sqrt((df - 2) / df)
        out[f"mean_student_t_log_score_df{df}"] = float((t.logpdf(e / scale, df=df) - np.log(scale)).mean())
    out["best_robust_log_score"] = max(out[f"mean_student_t_log_score_df{df}"] for df in student_t_dfs)
    out["robust_minus_gaussian_log_score"] = out["best_robust_log_score"] - out["mean_log_score"]
    return out


def _bounds(train, cols, lo, hi):
    return {c: (float(pd.to_numeric(train[c], errors="coerce").quantile(lo)), float(pd.to_numeric(train[c], errors="coerce").quantile(hi))) for c in cols}


def _clip(frame, bounds):
    x = frame.copy()
    for c, (lo, hi) in bounds.items():
        x[c] = pd.to_numeric(x[c], errors="coerce").clip(lo, hi)
    return x


def rolling_calibration(panel, model_specs, minimum_train_rows, minimum_validation_rows, minimum_test_year,
                        maximum_test_year, winsor_lower, winsor_upper, random_seed, student_t_dfs=(3, 5, 10, 30)):
    audited = panel[panel.audit_status.eq("audited")].copy()
    features = sorted({f for fs in model_specs.values() for f in fs})
    rows, wr, residual_rows = [], [], []
    for year in range(minimum_test_year, maximum_test_year + 1):
        train0 = audited[audited.fiscal_year.le(year - 1)].copy()
        fit0 = audited[audited.fiscal_year.le(year - 2)].copy()
        val0 = audited[audited.fiscal_year.eq(year - 1)].copy()
        test0 = audited[audited.fiscal_year.eq(year)].copy()
        if len(train0) < minimum_train_rows or test0.empty:
            continue
        b = _bounds(train0, ["ta_scaled"] + features, winsor_lower, winsor_upper)
        train, fit, val, test = map(lambda z: _clip(z, b), [train0, fit0, val0, test0])
        vy = val.ta_scaled.to_numpy(float)
        common = np.isfinite(vy)
        vp = {}
        for i, (name, fs) in enumerate(model_specs.items()):
            s = fit.replace([np.inf, -np.inf], np.nan).dropna(subset=["ta_scaled", "firm_id"] + fs)
            if len(s) < minimum_train_rows:
                continue
            m = ApproxHierarchicalBayes(random_state=random_seed + year * 100 + i).fit(
                s[fs].to_numpy(float), s.ta_scaled.to_numpy(float), s.firm_id.astype(str).to_numpy(), fs
            )
            valid = val[fs].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
            common &= valid
            p = m.posterior_mean_sd(val[fs].fillna(0).to_numpy(float), val.firm_id.astype(str).to_numpy(), True)
            vp[name] = (p.mean, p.sd)
        names = [n for n in model_specs if n in vp]
        if not names:
            continue
        if common.sum() >= minimum_validation_rows:
            stack = solve_stacking(vy[common], [vp[n][0][common] for n in names], [vp[n][1][common] for n in names])
        else:
            weights = np.repeat(1 / len(names), len(names))
            from .stacking import StackingResult
            stack = StackingResult(weights, False, "Insufficient validation rows; equal-weight fallback", np.nan, np.nan, np.nan,
                                   int(common.sum()), float(np.log(len(names))), float(len(names)))
        wr += [dict(
            fiscal_year=year, model=n, weight=float(w), validation_rows=int(common.sum()),
            optimizer_success=stack.success, optimizer_message=stack.message,
            stacking_objective=stack.objective, equal_weight_objective=stack.equal_weight_objective,
            best_single_objective=stack.best_single_objective, weight_entropy=stack.weight_entropy,
            effective_model_count=stack.effective_model_count,
        ) for n, w in zip(names, stack.weights)]
        fitted = {}
        ok = np.isfinite(test.ta_scaled.to_numpy(float))
        for i, n in enumerate(names):
            fs = model_specs[n]
            s = train.replace([np.inf, -np.inf], np.nan).dropna(subset=["ta_scaled", "firm_id"] + fs)
            fitted[n] = ApproxHierarchicalBayes(random_state=random_seed + year * 1000 + i).fit(
                s[fs].to_numpy(float), s.ta_scaled.to_numpy(float), s.firm_id.astype(str).to_numpy(), fs
            )
            ok &= test[fs].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        yt = test.iloc[idx].ta_scaled.to_numpy(float)
        yr = test0.loc[test.index[idx], "ta_scaled"].to_numpy(float)
        identifiers = test.iloc[idx][[c for c in ["issuer_ticker", "fiscal_year", "raw_exchange", "ta_source"] if c in test.columns]].reset_index(drop=True)
        for mode in ["conditional_existing_firm", "marginal_new_firm"]:
            mus, sds = [], []
            for n in names:
                fs = model_specs[n]
                ids = test.iloc[idx].firm_id.astype(str).to_numpy() if mode.startswith("conditional") else np.repeat("__NEW_FIRM__", len(idx))
                p = fitted[n].posterior_mean_sd(test.iloc[idx][fs].to_numpy(float), ids, True)
                mus.append(p.mean); sds.append(p.sd)
                for tv, y in [("winsorized_target", yt), ("raw_target", yr)]:
                    rows.append(dict(fiscal_year=year, model=n, prediction_mode=mode, target_variant=tv) | _metrics(y, p.mean, p.sd, student_t_dfs))
            M = np.vstack(mus).T; S = np.vstack(sds).T
            mu = (M * stack.weights).sum(1)
            sd = np.sqrt(np.maximum(((S ** 2 + M ** 2) * stack.weights).sum(1) - mu ** 2, 1e-12))
            for tv, y in [("winsorized_target", yt), ("raw_target", yr)]:
                rows.append(dict(fiscal_year=year, model="stacked_ensemble", prediction_mode=mode, target_variant=tv) | _metrics(y, mu, sd, student_t_dfs))
                residual = identifiers.copy()
                residual["model"] = "stacked_ensemble"
                residual["prediction_mode"] = mode
                residual["target_variant"] = tv
                residual["target"] = y
                residual["prediction_mean"] = mu
                residual["prediction_sd"] = sd
                residual["residual"] = y - mu
                residual["standardized_residual"] = (y - mu) / np.maximum(sd, 1e-12)
                residual_rows.append(residual)
    out = pd.DataFrame(rows)
    if not out.empty:
        for c in ["rmse", "mae", "mean_log_score", "best_robust_log_score"]:
            out[f"{c}_year_z"] = out.groupby(["model", "prediction_mode", "target_variant"], observed=True)[c].transform(
                lambda x: (x - x.mean()) / x.std(dof=0) if x.std(ddof=0) > 0 else 0.
            )
    residuals = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
    return out, pd.DataFrame(wr), residuals
