from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .diag_common import KEYS


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_preliminary_proxy_panel(panel: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    pre = panel[panel["audit_status"].eq(settings.get("unaudited_label", "unaudited"))].drop_duplicates(KEYS).copy()
    for column in ["lag_assets", "cfo", "pat", "revenue", "drev", "drec", "inv_assets", "loss"]:
        pre[column] = _numeric(pre, column)
    scale = pre["lag_assets"]
    pre["pre_cfo_scaled"] = pre["cfo"] / scale
    pre["pre_pat_scaled"] = pre["pat"] / scale
    pre["pre_revenue_scaled"] = pre["revenue"] / scale
    pre["pre_drev_scaled"] = pre["drev"] / scale
    pre["pre_drec_scaled"] = pre["drec"] / scale
    pre["pre_loss"] = pre["pat"].lt(0).astype(float)
    return pre


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], ridge: float) -> np.ndarray:
    train = train[predictors + ["pre_cfo_scaled"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    test_x = test[predictors].replace([np.inf, -np.inf], np.nan).copy()
    bounds: dict[str, tuple[float, float]] = {}
    for column in predictors + ["pre_cfo_scaled"]:
        bounds[column] = (float(train[column].quantile(.01)), float(train[column].quantile(.99)))
        train[column] = train[column].clip(*bounds[column])
    for column in predictors:
        test_x[column] = test_x[column].clip(*bounds[column])
    x = np.column_stack([np.ones(len(train)), train[predictors].to_numpy(float)])
    y = train["pre_cfo_scaled"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return np.column_stack([np.ones(len(test_x)), test_x.to_numpy(float)]) @ beta


def rolling_expected_cfo_proxies(panel: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = build_preliminary_proxy_panel(panel, settings)
    models = settings.get("proxy_models", {})
    records: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for year in range(int(settings.get("minimum_test_year", 2018)), int(settings.get("maximum_test_year", 2025)) + 1):
        train, test = data[data["fiscal_year"].lt(year)], data[data["fiscal_year"].eq(year)]
        for name, predictors in models.items():
            train_valid = train[list(predictors) + ["pre_cfo_scaled"]].replace([np.inf, -np.inf], np.nan).dropna()
            test_valid = test[KEYS + ["raw_exchange", "pre_cfo_scaled"] + list(predictors)].replace([np.inf, -np.inf], np.nan).dropna()
            if len(train_valid) < int(settings.get("minimum_train_rows", 500)) or test_valid.empty:
                folds.append({"fiscal_year": year, "proxy_model": name, "train_rows": len(train_valid), "test_rows": len(test_valid), "status": "INSUFFICIENT_SAMPLE"})
                continue
            predicted = _fit_predict(train_valid, test_valid, list(predictors), float(settings.get("ridge", 1e-8)))
            out = test_valid[KEYS + ["raw_exchange", "pre_cfo_scaled"]].copy()
            out["proxy_model"] = name
            out["expected_cfo_scaled"] = predicted
            out["abnormal_cfo_proxy"] = out["pre_cfo_scaled"] - predicted
            out["proxy_rank_within_year"] = out["abnormal_cfo_proxy"].rank(pct=True, method="average")
            records.append(out)
            folds.append({"fiscal_year": year, "proxy_model": name, "train_rows": len(train_valid), "test_rows": len(test_valid), "rmse": float(np.sqrt(np.mean((out["pre_cfo_scaled"] - predicted) ** 2))), "status": "OK"})
    return (pd.concat(records, ignore_index=True) if records else pd.DataFrame(), pd.DataFrame(folds))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = np.asarray(y, int), np.asarray(score, float)
    p, n = int(y.sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    if y.sum() == 0:
        return np.nan
    ranked = y[np.argsort(-np.asarray(score, float), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].mean())


def validate_proxy_predictions(predictions: pd.DataFrame, observed_cases: pd.DataFrame, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    material = float(settings.get("material_cfo_threshold", 0.05))
    label = settings.get("candidate_label", "identity_consistent_offsetting_reclassification_candidate")
    observed = observed_cases.drop_duplicates(KEYS).copy()
    observed["any_candidate"] = observed["cfs_resolution"].eq(label).astype(int)
    delta = _numeric(observed, "delta_cfo_scaled")
    observed["audited_cfo_decrease"] = (observed["any_candidate"].eq(1) & delta.le(-material)).astype(int)
    observed["audited_cfo_increase"] = (observed["any_candidate"].eq(1) & delta.ge(material)).astype(int)
    offset = observed.get("offset_channel_pattern", pd.Series("", index=observed.index))
    observed["cff_down_candidate"] = (observed["audited_cfo_decrease"].eq(1) & offset.eq("cff_dominant")).astype(int)
    observed["cfi_up_candidate"] = (observed["audited_cfo_increase"].eq(1) & offset.eq("cfi_dominant")).astype(int)
    outcomes = ["any_candidate", "audited_cfo_decrease", "audited_cfo_increase", "cff_down_candidate", "cfi_up_candidate"]
    merged = predictions.merge(observed[KEYS + outcomes], on=KEYS, how="inner", validate="many_to_one")
    summary: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    for model, group in merged.groupby("proxy_model", observed=True):
        for outcome in outcomes:
            y, score = group[outcome].to_numpy(int), group["abnormal_cfo_proxy"].to_numpy(float)
            prevalence = float(y.mean())
            top = group["proxy_rank_within_year"].ge(.90)
            top_rate = float(group.loc[top, outcome].mean()) if top.any() else np.nan
            summary.append({
                "proxy_model": model, "outcome": outcome, "rows": len(group), "positives": int(y.sum()),
                "prevalence": prevalence, "auc": _auc(y, score), "average_precision": _ap(y, score),
                "top_decile_rate": top_rate, "top_decile_lift": top_rate / prevalence if prevalence > 0 else np.nan,
                "mean_proxy_positive": float(group.loc[group[outcome].eq(1), "abnormal_cfo_proxy"].mean()),
                "mean_proxy_negative": float(group.loc[group[outcome].eq(0), "abnormal_cfo_proxy"].mean()),
            })
            for year, g in group.groupby("fiscal_year", observed=True):
                yy, ss = g[outcome].to_numpy(int), g["abnormal_cfo_proxy"].to_numpy(float)
                yearly.append({"proxy_model": model, "outcome": outcome, "fiscal_year": int(year), "rows": len(g), "positives": int(yy.sum()), "prevalence": float(yy.mean()), "auc": _auc(yy, ss), "average_precision": _ap(yy, ss)})
    return {"cfs_shifting_proxy_cases": merged, "cfs_shifting_proxy_validation": pd.DataFrame(summary), "cfs_shifting_proxy_validation_by_year": pd.DataFrame(yearly)}
