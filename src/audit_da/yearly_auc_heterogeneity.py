from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .auditor_regime import SCORE_RULES, _auc, _score, _standardize


def prepare_yearly_score_sample(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    cfg = settings.get("yearly_auc_heterogeneity", settings)
    frame = cases.copy()
    model = cfg.get("proxy_model", "earnings_working_capital")
    mode = cfg.get("sample_mode", "common_primary_models")
    restriction = cfg.get("sample_restriction", "analysis_core")
    if "proxy_model" in frame.columns:
        frame = frame[frame["proxy_model"].eq(model)]
    if "sample_mode" in frame.columns:
        frame = frame[frame["sample_mode"].eq(mode)]
    if "sample_restriction" in frame.columns:
        frame = frame[frame["sample_restriction"].eq(restriction)]
    frame = frame.drop_duplicates(["issuer_ticker", "fiscal_year"]).copy()
    frame["fiscal_year"] = pd.to_numeric(
        frame["fiscal_year"], errors="coerce"
    ).astype("Int64")
    return frame[frame["fiscal_year"].notna()].copy()


def yearly_auc_table(
    sample: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    cfg = settings.get("yearly_auc_heterogeneity", settings)
    outcomes = [
        outcome
        for outcome in cfg.get("outcomes", list(SCORE_RULES))
        if outcome in sample.columns
    ]
    minimum_positives = int(cfg.get("minimum_year_positives", 10))
    minimum_negatives = int(cfg.get("minimum_year_negatives", 10))
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        score = _score(sample["abnormal_cfo_proxy"], SCORE_RULES[outcome])
        for year, group in sample.assign(_score=score).groupby(
            "fiscal_year", observed=True
        ):
            y = pd.to_numeric(group[outcome], errors="coerce")
            s = pd.to_numeric(group["_score"], errors="coerce")
            valid = y.notna() & s.notna()
            y = y.loc[valid].astype(int)
            s = s.loc[valid]
            positives = int(y.sum())
            negatives = int(y.eq(0).sum())
            supported = (
                positives >= minimum_positives
                and negatives >= minimum_negatives
            )
            rows.append(
                {
                    "outcome": outcome,
                    "fiscal_year": int(year),
                    "rows": len(y),
                    "positives": positives,
                    "negatives": negatives,
                    "prevalence": float(y.mean()) if len(y) else np.nan,
                    "auc": _auc(y.to_numpy(), s.to_numpy(float))
                    if supported
                    else np.nan,
                    "status": "OK"
                    if supported
                    else "INSUFFICIENT_CLASS_SUPPORT",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_yearly_auc_covariance(
    sample: pd.DataFrame,
    yearly: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("yearly_auc_heterogeneity", settings)
    repetitions = int(cfg.get("bootstrap_repetitions", 500))
    seed = int(cfg.get("bootstrap_seed", 240722))
    rng = np.random.default_rng(seed)
    issuers = sample["issuer_ticker"].dropna().astype(str).unique()
    by_issuer = {
        str(issuer): group.copy()
        for issuer, group in sample.groupby("issuer_ticker", observed=True)
    }
    draws: list[dict[str, Any]] = []
    outcomes = (
        yearly["outcome"].drop_duplicates().tolist()
        if not yearly.empty
        else []
    )
    supported_years = {
        outcome: sorted(
            yearly[
                yearly["outcome"].eq(outcome)
                & yearly["status"].eq("OK")
            ]["fiscal_year"]
            .astype(int)
            .tolist()
        )
        for outcome in outcomes
    }
    if len(issuers) >= 2:
        for draw in range(repetitions):
            selected = rng.choice(
                issuers, size=len(issuers), replace=True
            )
            boot = pd.concat(
                [by_issuer[str(issuer)] for issuer in selected],
                ignore_index=True,
            )
            for outcome in outcomes:
                score = _score(
                    boot["abnormal_cfo_proxy"], SCORE_RULES[outcome]
                )
                local = boot.assign(_score=score)
                for year in supported_years[outcome]:
                    group = local[local["fiscal_year"].eq(year)]
                    y = pd.to_numeric(group[outcome], errors="coerce")
                    s = pd.to_numeric(group["_score"], errors="coerce")
                    valid = y.notna() & s.notna()
                    value = _auc(
                        y.loc[valid].astype(int).to_numpy(),
                        s.loc[valid].to_numpy(float),
                    )
                    draws.append(
                        {
                            "draw": draw,
                            "outcome": outcome,
                            "fiscal_year": year,
                            "auc": value,
                        }
                    )
    draw_frame = pd.DataFrame(draws)
    covariance_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        years = supported_years[outcome]
        point = (
            yearly[yearly["outcome"].eq(outcome)]
            .set_index("fiscal_year")
            .reindex(years)["auc"]
            .to_numpy(float)
        )
        wide = (
            draw_frame[draw_frame["outcome"].eq(outcome)]
            .pivot(index="draw", columns="fiscal_year", values="auc")
            .reindex(columns=years)
            .dropna()
            if not draw_frame.empty
            else pd.DataFrame()
        )
        if len(years) < 2 or wide.empty:
            continue
        covariance = np.cov(wide.to_numpy(float), rowvar=False)
        covariance = np.atleast_2d(covariance)
        for row_index, left in enumerate(years):
            for column_index, right in enumerate(years):
                covariance_rows.append(
                    {
                        "outcome": outcome,
                        "left_year": left,
                        "right_year": right,
                        "covariance": float(
                            covariance[row_index, column_index]
                        ),
                        "bootstrap_draws": len(wide),
                    }
                )
        precision = np.linalg.pinv(covariance)
        ones = np.ones(len(years))
        denominator = float(ones.T @ precision @ ones)
        common_auc = (
            float((ones.T @ precision @ point) / denominator)
            if denominator > 0
            else float(np.mean(point))
        )
        centered = point - common_auc
        statistic = float(centered.T @ precision @ centered)
        q_rows.append(
            {
                "outcome": outcome,
                "test": "GENERALIZED_AUC_HETEROGENEITY_Q",
                "years": ",".join(map(str, years)),
                "common_auc_gls": common_auc,
                "chi_square": statistic,
                "df": len(years) - 1,
                "p_value": float(chi2.sf(statistic, len(years) - 1)),
                "bootstrap_draws": len(wide),
            }
        )
    return (
        draw_frame,
        pd.DataFrame(covariance_rows),
        pd.DataFrame(q_rows),
    )


def _fit_clustered_logit_covariance(
    design: pd.DataFrame,
    y: pd.Series,
    ridge: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    clusters = design.pop("issuer_ticker").astype(str).to_numpy()
    x = design.to_numpy(float)
    target = y.to_numpy(float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    status = "MAX_ITER"
    for _ in range(max_iter):
        eta = np.clip(x @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        hessian = x.T @ (x * weight[:, None]) + penalty
        gradient = x.T @ (target - probability) - penalty @ beta
        step = np.linalg.pinv(hessian) @ gradient
        beta += step
        if np.max(np.abs(step)) < tolerance:
            status = "CONVERGED"
            break
    eta = np.clip(x @ beta, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = np.clip(probability * (1.0 - probability), 1e-8, None)
    bread = np.linalg.pinv(x.T @ (x * weight[:, None]) + penalty)
    scores = x * (target - probability)[:, None]
    score_frame = pd.DataFrame(scores)
    score_frame["cluster"] = clusters
    cluster_scores = (
        score_frame.groupby("cluster", observed=True)
        .sum(numeric_only=True)
        .to_numpy(float)
    )
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread
    n, k, g = len(target), x.shape[1], len(cluster_scores)
    if g > 1 and n > k:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return beta, se, covariance, status


def score_by_year_logit(
    sample: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("yearly_auc_heterogeneity", settings)
    outcomes = [
        outcome
        for outcome in cfg.get("outcomes", list(SCORE_RULES))
        if outcome in sample.columns
    ]
    estimates: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for outcome in outcomes:
        work = sample.copy()
        work["score_z"] = _standardize(
            _score(work["abnormal_cfo_proxy"], SCORE_RULES[outcome])
        )
        years = sorted(
            pd.to_numeric(work["fiscal_year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(years) < 2:
            continue
        reference = int(cfg.get("reference_year", years[0]))
        if reference not in years:
            reference = years[0]
        x = pd.DataFrame(index=work.index)
        x["intercept"] = 1.0
        x["score_z"] = work["score_z"]
        focal_terms: list[str] = []
        for year in years:
            if year == reference:
                continue
            dummy = work["fiscal_year"].eq(year).astype(float)
            x[f"year_{year}"] = dummy
            term = f"score_x_year_{year}"
            x[term] = work["score_z"] * dummy
            focal_terms.append(term)
        for column in cfg.get(
            "continuous_controls", ["lag_assets", "pre_cfo_scaled"]
        ):
            if column not in work.columns:
                continue
            values = pd.to_numeric(work[column], errors="coerce")
            if "assets" in column:
                values = np.log(values.clip(lower=1.0))
            x[column] = _standardize(values)
        for column in cfg.get(
            "fixed_effects", ["raw_exchange", "industry_name"]
        ):
            if column not in work.columns:
                continue
            dummies = pd.get_dummies(
                work[column].fillna("UNKNOWN").astype(str),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            x = pd.concat([x, dummies], axis=1)
        y = pd.to_numeric(work[outcome], errors="coerce")
        valid = (
            y.notna()
            & x.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        )
        design = pd.concat(
            [work.loc[valid, ["issuer_ticker"]], x.loc[valid]], axis=1
        )
        y_valid = y.loc[valid].astype(int)
        positives = int(y_valid.sum())
        if (
            len(y_valid) < int(cfg.get("minimum_interaction_rows", 300))
            or positives
            < int(cfg.get("minimum_interaction_positives", 20))
        ):
            continue
        terms = [
            column for column in design.columns if column != "issuer_ticker"
        ]
        beta, se, covariance, status = _fit_clustered_logit_covariance(
            design.copy(),
            y_valid,
            float(cfg.get("interaction_ridge", 1e-6)),
            int(cfg.get("interaction_max_iter", 100)),
            float(cfg.get("interaction_tolerance", 1e-8)),
        )
        term_index = {term: index for index, term in enumerate(terms)}
        for term in focal_terms:
            index = term_index[term]
            z = beta[index] / se[index] if se[index] > 0 else np.nan
            estimates.append(
                {
                    "outcome": outcome,
                    "reference_year": reference,
                    "term": term,
                    "year": int(term.rsplit("_", 1)[1]),
                    "estimate": float(beta[index]),
                    "cluster_se": float(se[index]),
                    "odds_ratio": float(
                        np.exp(np.clip(beta[index], -20, 20))
                    ),
                    "z_value": float(z) if np.isfinite(z) else np.nan,
                    "p_value_two_sided": float(2 * norm.sf(abs(z)))
                    if np.isfinite(z)
                    else np.nan,
                    "rows": len(y_valid),
                    "positives": positives,
                    "clusters": int(design["issuer_ticker"].nunique()),
                    "status": status,
                }
            )
        indices = [term_index[term] for term in focal_terms]
        focal_beta = beta[indices]
        focal_cov = covariance[np.ix_(indices, indices)]
        statistic = float(
            focal_beta.T @ np.linalg.pinv(focal_cov) @ focal_beta
        )
        tests.append(
            {
                "outcome": outcome,
                "test": "JOINT_SCORE_BY_YEAR_INTERACTIONS",
                "reference_year": reference,
                "terms": "|".join(focal_terms),
                "chi_square": statistic,
                "df": len(indices),
                "p_value": float(chi2.sf(statistic, len(indices))),
                "rows": len(y_valid),
                "positives": positives,
                "status": status,
            }
        )
    return pd.DataFrame(estimates), pd.DataFrame(tests)


def run_yearly_auc_heterogeneity(
    cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sample = prepare_yearly_score_sample(cases, settings)
    yearly = yearly_auc_table(sample, settings)
    draws, covariance, q_test = bootstrap_yearly_auc_covariance(
        sample, yearly, settings
    )
    interaction, interaction_test = score_by_year_logit(sample, settings)
    status = pd.DataFrame(
        [
            {
                "gate": "yearly_auc_heterogeneity",
                "status": "PASS"
                if not q_test.empty and not interaction_test.empty
                else "PARTIALLY_EVALUATED",
                "analysis_rows": len(sample),
                "issuers": int(sample["issuer_ticker"].nunique())
                if not sample.empty
                else 0,
                "supported_year_outcome_cells": int(
                    yearly["status"].eq("OK").sum()
                )
                if not yearly.empty
                else 0,
                "interpretation": (
                    "Formal temporal-heterogeneity test accounting for repeated "
                    "issuer observations through issuer-cluster bootstrap and "
                    "clustered score-by-year interactions."
                ),
            }
        ]
    )
    return {
        "cfs_yearly_auc_heterogeneity_sample": sample,
        "cfs_yearly_auc_metrics": yearly,
        "cfs_yearly_auc_bootstrap_draws": draws,
        "cfs_yearly_auc_covariance": covariance,
        "cfs_yearly_auc_generalized_q": q_test,
        "cfs_score_by_year_interactions": interaction,
        "cfs_score_by_year_joint_tests": interaction_test,
        "cfs_yearly_auc_heterogeneity_status": status,
    }
