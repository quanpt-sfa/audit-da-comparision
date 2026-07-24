from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["issuer_ticker", "fiscal_year"]

PREDICTIVE_TESTS: dict[str, dict[str, object]] = {
    "earnings_persistence": {
        "outcome": "future_roa_audited",
        "predictors": ("roa",),
        "construct": "earnings_persistence",
    },
    "earnings_to_future_cfo": {
        "outcome": "future_cfo_audited",
        "predictors": ("roa",),
        "construct": "future_cfo_informativeness",
    },
    "cfo_persistence": {
        "outcome": "future_cfo_audited",
        "predictors": ("cfo_scaled",),
        "construct": "cash_flow_persistence",
    },
    "earnings_cfo_horse_race": {
        "outcome": "future_cfo_audited",
        "predictors": ("roa", "cfo_scaled"),
        "construct": "incremental_future_cfo_information",
    },
}


@dataclass(frozen=True)
class PredictiveValiditySettings:
    audited_label: str = "audited"
    unaudited_label: str = "unaudited"
    predictor_start_year: int = 2016
    predictor_end_year: int = 2024
    oos_test_start_year: int = 2018
    oos_test_end_year: int = 2024
    minimum_train_rows: int = 250
    minimum_train_years: int = 2
    pooled_specifications: tuple[str, ...] = ("canonical", "year_industry_fe")
    industry_column: str = "icb_l1"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    bootstrap_draws: int = 2000
    seed: int = 20260724
    aq_start_year: int = 2017
    aq_end_year: int = 2024
    aq_minimum_train_rows: int = 250


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _coalesce_text(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in names:
        if name in frame:
            output = output.fillna(frame[name].astype("string"))
    return output


def _scaled_state_variables(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    lag_assets = pd.to_numeric(output.get("lag_assets"), errors="coerce").abs()
    derived_roa = pd.to_numeric(output.get("pat"), errors="coerce").div(
        lag_assets.where(lag_assets.gt(0))
    )
    derived_cfo = pd.to_numeric(output.get("cfo"), errors="coerce").div(
        lag_assets.where(lag_assets.gt(0))
    )
    if "roa" in output:
        output["roa"] = pd.to_numeric(output["roa"], errors="coerce").fillna(
            derived_roa
        )
    else:
        output["roa"] = derived_roa
    if "cfo_scaled" in output:
        output["cfo_scaled"] = pd.to_numeric(
            output["cfo_scaled"], errors="coerce"
        ).fillna(derived_cfo)
    else:
        output["cfo_scaled"] = derived_cfo
    return output


def pair_reporting_states(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    required = set(KEYS + ["audit_status"])
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Panel missing reporting-state keys: {missing}")

    frame = _scaled_state_variables(panel)
    pre = (
        frame.loc[frame.audit_status.eq(settings.unaudited_label)]
        .drop_duplicates(KEYS)
        .copy()
    )
    audited = (
        frame.loc[frame.audit_status.eq(settings.audited_label)]
        .drop_duplicates(KEYS)
        .copy()
    )
    shared = sorted(
        (set(pre.columns) & set(audited.columns)) - set(KEYS + ["audit_status"])
    )
    pre = pre[KEYS + shared].rename(
        columns={column: f"{column}_pre" for column in shared}
    )
    audited = audited[KEYS + shared].rename(
        columns={column: f"{column}_audited" for column in shared}
    )
    return pre.merge(audited, on=KEYS, how="inner", validate="one_to_one")


def _audited_leads(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    audited = _scaled_state_variables(
        panel.loc[panel.audit_status.eq(settings.audited_label)].copy()
    )
    audited["fiscal_year"] = pd.to_numeric(audited["fiscal_year"], errors="coerce")
    audited = audited.dropna(subset=["issuer_ticker", "fiscal_year"]).copy()
    audited["fiscal_year"] = audited["fiscal_year"].astype(int) - 1
    return audited[["issuer_ticker", "fiscal_year", "roa", "cfo_scaled"]].rename(
        columns={
            "roa": "future_roa_audited",
            "cfo_scaled": "future_cfo_audited",
        }
    )


def build_predictive_cases(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    pair = pair_reporting_states(panel, settings)
    future = _audited_leads(panel, settings)
    cases = pair.merge(future, on=KEYS, how="inner", validate="one_to_one")
    cases["fiscal_year"] = pd.to_numeric(cases["fiscal_year"], errors="coerce")
    cases = cases.loc[
        cases.fiscal_year.between(
            settings.predictor_start_year, settings.predictor_end_year
        )
    ].copy()
    cases["outcome_fiscal_year"] = cases["fiscal_year"].astype(int) + 1
    cases["industry"] = _coalesce_text(
        cases,
        [
            f"{settings.industry_column}_audited",
            f"{settings.industry_column}_pre",
        ],
    )
    cases = _numeric(
        cases,
        [
            "roa_pre",
            "roa_audited",
            "cfo_scaled_pre",
            "cfo_scaled_audited",
            "future_roa_audited",
            "future_cfo_audited",
        ],
    )
    return cases.sort_values(KEYS).reset_index(drop=True)


def _test_complete_cases(
    cases: pd.DataFrame,
    test_name: str,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    definition = PREDICTIVE_TESTS[test_name]
    outcome = str(definition["outcome"])
    predictors = tuple(str(value) for value in definition["predictors"])
    required = [outcome, "issuer_ticker", "fiscal_year"]
    for predictor in predictors:
        required.extend([f"{predictor}_pre", f"{predictor}_audited"])
    missing = sorted(set(required) - set(cases.columns))
    if missing:
        raise ValueError(f"Predictive cases missing columns for {test_name}: {missing}")
    clean = cases.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return clean, outcome, predictors


def _dummy_frame(
    frame: pd.DataFrame,
    specification: str,
) -> tuple[pd.DataFrame, list[str]]:
    if specification == "canonical":
        return pd.DataFrame(index=frame.index), []
    if specification != "year_industry_fe":
        raise ValueError(f"Unknown pooled specification: {specification}")
    year = pd.get_dummies(
        frame["fiscal_year"].astype(int).astype(str),
        prefix="year",
        drop_first=True,
        dtype=float,
    )
    industry = pd.get_dummies(
        frame["industry"].fillna("UNKNOWN").astype(str),
        prefix="industry",
        drop_first=True,
        dtype=float,
    )
    controls = pd.concat([year, industry], axis=1)
    return controls, controls.columns.tolist()


def _state_design(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    state: str,
    controls: pd.DataFrame,
    control_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(frame), 1), dtype=float)]
    names = ["intercept"]
    for predictor in predictors:
        parts.append(
            pd.to_numeric(frame[f"{predictor}_{state}"], errors="coerce")
            .to_numpy(float)[:, None]
        )
        names.append(predictor)
    if control_names:
        parts.append(controls[list(control_names)].to_numpy(float))
        names.extend(control_names)
    return np.column_stack(parts), names


def _cluster_ols_cov(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1) & pd.notna(clusters)
    y, x, clusters = y[keep], x[keep], clusters[keep]
    n, k = x.shape
    unique = pd.unique(clusters)
    if n <= k or len(unique) < 2:
        nan_beta = np.full(k, np.nan)
        nan_cov = np.full((k, k), np.nan)
        return nan_beta, nan_cov, nan_beta.copy(), nan_beta.copy(), len(unique)

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster_value in unique:
        index = clusters == cluster_value
        score = x[index].T @ residual[index]
        meat += np.outer(score, score)
    g = len(unique)
    correction = g / (g - 1) * ((n - 1) / max(n - k, 1))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    p_value = 2 * stats.t.sf(np.abs(t_value), df=max(g - 1, 1))
    return beta, covariance, standard_error, p_value, g


def _contrast(
    beta: np.ndarray,
    covariance: np.ndarray,
    left: int,
    right: int,
    cluster_count: int,
) -> tuple[float, float, float]:
    vector = np.zeros(len(beta), dtype=float)
    vector[left], vector[right] = 1.0, -1.0
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    t_value = estimate / standard_error if standard_error > 0 else np.nan
    p_value = (
        float(2 * stats.t.sf(abs(t_value), df=max(cluster_count - 1, 1)))
        if np.isfinite(t_value)
        else np.nan
    )
    return estimate, standard_error, p_value


def pooled_state_comparisons(
    cases: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for test_name, definition in PREDICTIVE_TESTS.items():
        clean, outcome, predictors = _test_complete_cases(cases, test_name)
        if clean.empty:
            continue
        y = pd.to_numeric(clean[outcome], errors="coerce").to_numpy(float)
        clusters = clean["issuer_ticker"].to_numpy(object)
        for specification in settings.pooled_specifications:
            controls, control_names = _dummy_frame(clean, specification)
            x_pre, names = _state_design(
                clean, predictors, "pre", controls, control_names
            )
            x_audited, audited_names = _state_design(
                clean, predictors, "audited", controls, control_names
            )
            if names != audited_names:
                raise AssertionError("State-specific design columns diverged")
            zeros = np.zeros_like(x_pre)
            stacked_x = np.block([[x_pre, zeros], [zeros, x_audited]])
            stacked_y = np.concatenate([y, y])
            stacked_clusters = np.concatenate([clusters, clusters])
            beta, covariance, se, p, cluster_count = _cluster_ols_cov(
                stacked_y, stacked_x, stacked_clusters
            )
            width = x_pre.shape[1]
            for state, offset, x_state in (
                ("pre", 0, x_pre),
                ("audited", width, x_audited),
            ):
                predictions = x_state @ beta[offset : offset + width]
                residual = y - predictions
                sst = float(np.square(y - np.mean(y)).sum())
                fit_rows.append(
                    {
                        "test": test_name,
                        "construct": definition["construct"],
                        "specification": specification,
                        "state": state,
                        "n": int(len(clean)),
                        "issuers": int(clean.issuer_ticker.nunique()),
                        "predictor_year_min": int(clean.fiscal_year.min()),
                        "predictor_year_max": int(clean.fiscal_year.max()),
                        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                        "mae": float(np.mean(np.abs(residual))),
                        "r_squared": float(1.0 - np.square(residual).sum() / sst)
                        if sst > 0
                        else np.nan,
                    }
                )
                for term in ["intercept", *predictors]:
                    index = names.index(term) + offset
                    coefficient_rows.append(
                        {
                            "test": test_name,
                            "construct": definition["construct"],
                            "specification": specification,
                            "term": term,
                            "contrast": state,
                            "estimate": float(beta[index]),
                            "standard_error": float(se[index]),
                            "p_value": float(p[index]),
                            "n": int(len(clean)),
                            "issuers": int(clean.issuer_ticker.nunique()),
                        }
                    )
            for term in predictors:
                estimate, standard_error, p_value = _contrast(
                    beta,
                    covariance,
                    names.index(term) + width,
                    names.index(term),
                    cluster_count,
                )
                coefficient_rows.append(
                    {
                        "test": test_name,
                        "construct": definition["construct"],
                        "specification": specification,
                        "term": term,
                        "contrast": "audited_minus_pre",
                        "estimate": estimate,
                        "standard_error": standard_error,
                        "p_value": p_value,
                        "n": int(len(clean)),
                        "issuers": int(clean.issuer_ticker.nunique()),
                    }
                )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(fit_rows)


def _winsor_bounds(
    frame: pd.DataFrame,
    columns: Sequence[str],
    lower: float,
    upper: float,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No finite historical values for {column}")
        lo, hi = values.quantile([lower, upper])
        bounds[column] = (float(lo), float(hi))
    return bounds


def _fit_oos_state(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    state: str,
    settings: PredictiveValiditySettings,
) -> tuple[np.ndarray, dict[str, tuple[float, float]], float]:
    predictor_columns = [f"{name}_{state}" for name in predictors]
    bounds = _winsor_bounds(
        train,
        [outcome, *predictor_columns],
        settings.winsor_lower,
        settings.winsor_upper,
    )
    y_train = pd.to_numeric(train[outcome], errors="coerce").clip(
        *bounds[outcome]
    ).to_numpy(float)
    x_train_parts = [np.ones((len(train), 1), dtype=float)]
    x_test_parts = [np.ones((len(test), 1), dtype=float)]
    for column in predictor_columns:
        lo, hi = bounds[column]
        x_train_parts.append(
            pd.to_numeric(train[column], errors="coerce")
            .clip(lo, hi)
            .to_numpy(float)[:, None]
        )
        x_test_parts.append(
            pd.to_numeric(test[column], errors="coerce")
            .clip(lo, hi)
            .to_numpy(float)[:, None]
        )
    x_train, x_test = np.column_stack(x_train_parts), np.column_stack(x_test_parts)
    beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
    return x_test @ beta, bounds, float(np.mean(y_train))


def expanding_oos_predictions(
    cases: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for test_name, definition in PREDICTIVE_TESTS.items():
        clean, outcome, predictors = _test_complete_cases(cases, test_name)
        for test_year in range(
            settings.oos_test_start_year, settings.oos_test_end_year + 1
        ):
            train = clean.loc[clean.fiscal_year.lt(test_year)].copy()
            test = clean.loc[clean.fiscal_year.eq(test_year)].copy()
            train_years = int(train.fiscal_year.nunique())
            status, reason = "estimated", ""
            if len(train) < settings.minimum_train_rows:
                status, reason = (
                    "insufficient_train_rows",
                    f"{len(train)}<{settings.minimum_train_rows}",
                )
            elif train_years < settings.minimum_train_years:
                status, reason = (
                    "insufficient_train_years",
                    f"{train_years}<{settings.minimum_train_years}",
                )
            elif test.empty:
                status, reason = "empty_test_year", "no complete common sample"
            fold: dict[str, object] = {
                "test": test_name,
                "construct": definition["construct"],
                "test_year": int(test_year),
                "train_rows": int(len(train)),
                "train_issuers": int(train.issuer_ticker.nunique()),
                "train_years": train_years,
                "test_rows": int(len(test)),
                "status": status,
                "reason": reason,
            }
            if status != "estimated":
                fold_rows.append(fold)
                continue
            pred_pre, pre_bounds, benchmark = _fit_oos_state(
                train, test, outcome, predictors, "pre", settings
            )
            pred_audited, audited_bounds, _ = _fit_oos_state(
                train, test, outcome, predictors, "audited", settings
            )
            actual = pd.to_numeric(test[outcome], errors="coerce").to_numpy(float)
            current = test[KEYS].copy()
            current["outcome_fiscal_year"] = test["outcome_fiscal_year"].to_numpy()
            current["test"] = test_name
            current["construct"] = definition["construct"]
            current["actual"] = actual
            current["prediction_pre"] = pred_pre
            current["prediction_audited"] = pred_audited
            current["prediction_benchmark"] = benchmark
            for state in ("pre", "audited"):
                current[f"error_{state}"] = actual - current[f"prediction_{state}"]
                current[f"squared_error_{state}"] = np.square(
                    current[f"error_{state}"]
                )
                current[f"absolute_error_{state}"] = current[f"error_{state}"].abs()
            prediction_rows.append(current)
            fold["pre_predictor_bounds"] = repr(
                {key: value for key, value in pre_bounds.items() if key != outcome}
            )
            fold["audited_predictor_bounds"] = repr(
                {
                    key: value
                    for key, value in audited_bounds.items()
                    if key != outcome
                }
            )
            fold["training_outcome_bounds"] = repr(pre_bounds[outcome])
            fold_rows.append(fold)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return predictions, pd.DataFrame(fold_rows)


def _cluster_bootstrap(
    frame: pd.DataFrame,
    statistic,
    draws: int,
    seed: int,
) -> dict[str, float]:
    groups = [
        group.copy()
        for _, group in frame.groupby("issuer_ticker", sort=False, observed=True)
    ]
    estimate = float(statistic(frame))
    if len(groups) < 2:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided": np.nan,
        }
    rng = np.random.default_rng(seed)
    values = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sampled = rng.integers(0, len(groups), size=len(groups))
        current = pd.concat([groups[index] for index in sampled], ignore_index=True)
        values[draw] = float(statistic(current))
    ci_low, ci_high = np.nanquantile(values, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2.0
        * min(
            (1 + np.sum(values <= 0)) / (len(values) + 1),
            (1 + np.sum(values >= 0)) / (len(values) + 1),
        ),
    )
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": float(p_two_sided),
    }


def summarize_oos(
    predictions: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    for test_name, group in predictions.groupby("test", observed=True):
        actual = pd.to_numeric(group["actual"], errors="coerce")
        benchmark_error = actual - pd.to_numeric(
            group["prediction_benchmark"], errors="coerce"
        )
        benchmark_sse = float(np.square(benchmark_error).sum())
        metrics: dict[str, dict[str, float]] = {}
        for state in ("pre", "audited"):
            error = pd.to_numeric(group[f"error_{state}"], errors="coerce")
            sse = float(np.square(error).sum())
            metrics[state] = {
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "r2_oos": float(1.0 - sse / benchmark_sse)
                if benchmark_sse > 0
                else np.nan,
            }
            state_rows.append(
                {
                    "test": test_name,
                    "construct": group["construct"].iloc[0],
                    "state": state,
                    "n": int(len(group)),
                    "issuers": int(group.issuer_ticker.nunique()),
                    "test_year_min": int(group.fiscal_year.min()),
                    "test_year_max": int(group.fiscal_year.max()),
                    **metrics[state],
                }
            )
        for metric, statistic in (
            (
                "rmse",
                lambda z: np.sqrt(z.squared_error_audited.mean())
                - np.sqrt(z.squared_error_pre.mean()),
            ),
            (
                "mae",
                lambda z: z.absolute_error_audited.mean()
                - z.absolute_error_pre.mean(),
            ),
        ):
            boot = _cluster_bootstrap(
                group,
                statistic,
                settings.bootstrap_draws,
                settings.seed + sum(ord(char) for char in f"{test_name}:{metric}"),
            )
            difference_rows.append(
                {
                    "test": test_name,
                    "construct": group["construct"].iloc[0],
                    "metric": metric,
                    "contrast": "audited_minus_pre",
                    **boot,
                    "negative_favours_audited": True,
                    "n": int(len(group)),
                    "issuers": int(group.issuer_ticker.nunique()),
                }
            )
        difference_rows.append(
            {
                "test": test_name,
                "construct": group["construct"].iloc[0],
                "metric": "r2_oos",
                "contrast": "audited_minus_pre",
                "estimate": metrics["audited"]["r2_oos"]
                - metrics["pre"]["r2_oos"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_two_sided": np.nan,
                "negative_favours_audited": False,
                "n": int(len(group)),
                "issuers": int(group.issuer_ticker.nunique()),
            }
        )
    return pd.DataFrame(state_rows), pd.DataFrame(difference_rows)


def _derive_wca_scaled(paired: pd.DataFrame, state: str) -> pd.Series:
    lag_assets = pd.to_numeric(
        paired.get(f"lag_assets_{state}"), errors="coerce"
    ).abs()
    ta_balance_sheet = pd.to_numeric(
        paired.get(f"ta_balance_sheet_{state}"), errors="coerce"
    )
    depreciation = pd.to_numeric(
        paired.get(f"depreciation_{state}"), errors="coerce"
    )
    result = (ta_balance_sheet + depreciation).div(
        lag_assets.where(lag_assets.gt(0))
    )
    names = {
        "current_assets": f"current_assets_{state}",
        "lag_current_assets": f"lag_current_assets_audited_{state}",
        "cash": f"cash_{state}",
        "lag_cash": f"lag_cash_audited_{state}",
        "current_liabilities": f"current_liabilities_{state}",
        "lag_current_liabilities": f"lag_current_liabilities_audited_{state}",
        "short_term_debt": f"short_term_debt_{state}",
        "lag_short_term_debt": f"lag_short_term_debt_audited_{state}",
        "tax_payable": f"tax_payable_{state}",
        "lag_tax_payable": f"lag_tax_payable_audited_{state}",
    }
    if all(column in paired for column in names.values()):
        values = {
            name: pd.to_numeric(paired[column], errors="coerce")
            for name, column in names.items()
        }
        dca = values["current_assets"] - values["lag_current_assets"]
        dcash = values["cash"] - values["lag_cash"]
        dcl = values["current_liabilities"] - values["lag_current_liabilities"]
        dstd = values["short_term_debt"] - values["lag_short_term_debt"]
        dtax = values["tax_payable"] - values["lag_tax_payable"]
        fallback = ((dca - dcash) - (dcl - dstd - dtax)).div(
            lag_assets.where(lag_assets.gt(0))
        )
        result = result.fillna(fallback)
    return result


def build_accrual_quality_cases(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    pair = pair_reporting_states(panel, settings)
    pair["wca_scaled_pre"] = _derive_wca_scaled(pair, "pre")
    pair["wca_scaled_audited"] = _derive_wca_scaled(pair, "audited")
    audited = _scaled_state_variables(
        panel.loc[panel.audit_status.eq(settings.audited_label)].copy()
    )
    audited["fiscal_year"] = pd.to_numeric(audited["fiscal_year"], errors="coerce")
    audited = audited.dropna(subset=["issuer_ticker", "fiscal_year"]).copy()
    audited["fiscal_year"] = audited["fiscal_year"].astype(int)
    lag = audited[["issuer_ticker", "fiscal_year", "cfo_scaled"]].copy()
    lag["fiscal_year"] += 1
    lag = lag.rename(columns={"cfo_scaled": "cfo_lag_audited"})
    lead = audited[["issuer_ticker", "fiscal_year", "cfo_scaled"]].copy()
    lead["fiscal_year"] -= 1
    lead = lead.rename(columns={"cfo_scaled": "cfo_lead_audited"})
    cases = pair.merge(lag, on=KEYS, how="inner", validate="one_to_one").merge(
        lead, on=KEYS, how="inner", validate="one_to_one"
    )
    cases["industry"] = _coalesce_text(
        cases,
        [
            f"{settings.industry_column}_audited",
            f"{settings.industry_column}_pre",
        ],
    )
    cases = _numeric(
        cases,
        [
            "wca_scaled_pre",
            "wca_scaled_audited",
            "cfo_lag_audited",
            "cfo_scaled_pre",
            "cfo_scaled_audited",
            "cfo_lead_audited",
        ],
    )
    required = [
        "wca_scaled_pre",
        "wca_scaled_audited",
        "cfo_lag_audited",
        "cfo_scaled_pre",
        "cfo_scaled_audited",
        "cfo_lead_audited",
        "issuer_ticker",
        "fiscal_year",
    ]
    cases = cases.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    cases = cases.loc[
        cases.fiscal_year.between(settings.aq_start_year, settings.aq_end_year)
    ].copy()
    return cases[
        KEYS
        + [
            "industry",
            "wca_scaled_pre",
            "wca_scaled_audited",
            "cfo_lag_audited",
            "cfo_scaled_pre",
            "cfo_scaled_audited",
            "cfo_lead_audited",
        ]
    ].sort_values(KEYS).reset_index(drop=True)


def _aq_design(
    frame: pd.DataFrame,
    state: str,
    specification: str,
) -> tuple[np.ndarray, list[str]]:
    controls, control_names = _dummy_frame(frame, specification)
    columns = ["cfo_lag_audited", f"cfo_scaled_{state}", "cfo_lead_audited"]
    parts = [np.ones((len(frame), 1), dtype=float)]
    names = ["intercept", "cfo_lag", "cfo_current", "cfo_lead"]
    for column in columns:
        parts.append(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(float)[:, None]
        )
    if control_names:
        parts.append(controls[control_names].to_numpy(float))
        names.extend(control_names)
    return np.column_stack(parts), names


def accrual_quality_coefficients(
    cases: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cases.empty:
        return pd.DataFrame()
    clusters = cases.issuer_ticker.to_numpy(object)
    for specification in settings.pooled_specifications:
        x_pre, names = _aq_design(cases, "pre", specification)
        x_audited, names_audited = _aq_design(cases, "audited", specification)
        if names != names_audited:
            raise AssertionError("Accrual-quality designs diverged")
        zeros = np.zeros_like(x_pre)
        x = np.block([[x_pre, zeros], [zeros, x_audited]])
        y = np.concatenate(
            [
                cases.wca_scaled_pre.to_numpy(float),
                cases.wca_scaled_audited.to_numpy(float),
            ]
        )
        beta, covariance, se, p, cluster_count = _cluster_ols_cov(
            y, x, np.concatenate([clusters, clusters])
        )
        width = x_pre.shape[1]
        for state, offset in (("pre", 0), ("audited", width)):
            for term in ["intercept", "cfo_lag", "cfo_current", "cfo_lead"]:
                index = names.index(term) + offset
                rows.append(
                    {
                        "specification": specification,
                        "term": term,
                        "contrast": state,
                        "estimate": float(beta[index]),
                        "standard_error": float(se[index]),
                        "p_value": float(p[index]),
                        "n": int(len(cases)),
                        "issuers": int(cases.issuer_ticker.nunique()),
                    }
                )
        for term in ("cfo_lag", "cfo_current", "cfo_lead"):
            estimate, standard_error, p_value = _contrast(
                beta,
                covariance,
                names.index(term) + width,
                names.index(term),
                cluster_count,
            )
            rows.append(
                {
                    "specification": specification,
                    "term": term,
                    "contrast": "audited_minus_pre",
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "p_value": p_value,
                    "n": int(len(cases)),
                    "issuers": int(cases.issuer_ticker.nunique()),
                }
            )
    return pd.DataFrame(rows)


def accrual_quality_crossfit(
    cases: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    if cases.empty:
        return pd.DataFrame()
    for holdout_year in sorted(cases.fiscal_year.unique()):
        train = cases.loc[cases.fiscal_year.ne(holdout_year)].copy()
        test = cases.loc[cases.fiscal_year.eq(holdout_year)].copy()
        if len(train) < settings.aq_minimum_train_rows or test.empty:
            continue
        current = test[KEYS].copy()
        for state in ("pre", "audited"):
            x_train, _ = _aq_design(train, state, "canonical")
            x_test, _ = _aq_design(test, state, "canonical")
            y_train = train[f"wca_scaled_{state}"].to_numpy(float)
            y_test = test[f"wca_scaled_{state}"].to_numpy(float)
            beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
            prediction = x_test @ beta
            residual = y_test - prediction
            current[f"actual_{state}"] = y_test
            current[f"prediction_{state}"] = prediction
            current[f"residual_{state}"] = residual
            current[f"absolute_residual_{state}"] = np.abs(residual)
            current[f"squared_residual_{state}"] = np.square(residual)
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def summarize_accrual_quality(
    crossfit: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if crossfit.empty:
        return pd.DataFrame(), pd.DataFrame()
    state_rows: list[dict[str, object]] = []
    for state in ("pre", "audited"):
        residual = pd.to_numeric(crossfit[f"residual_{state}"], errors="coerce")
        state_rows.append(
            {
                "state": state,
                "n": int(len(crossfit)),
                "issuers": int(crossfit.issuer_ticker.nunique()),
                "year_min": int(crossfit.fiscal_year.min()),
                "year_max": int(crossfit.fiscal_year.max()),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "residual_sd": float(residual.std(ddof=1)),
            }
        )
    statistics = {
        "rmse": lambda z: np.sqrt(z.squared_residual_audited.mean())
        - np.sqrt(z.squared_residual_pre.mean()),
        "mae": lambda z: z.absolute_residual_audited.mean()
        - z.absolute_residual_pre.mean(),
        "residual_sd": lambda z: z.residual_audited.std(ddof=1)
        - z.residual_pre.std(ddof=1),
    }
    difference_rows = []
    for metric, statistic in statistics.items():
        boot = _cluster_bootstrap(
            crossfit,
            statistic,
            settings.bootstrap_draws,
            settings.seed + sum(ord(char) for char in f"aq:{metric}"),
        )
        difference_rows.append(
            {
                "metric": metric,
                "contrast": "audited_minus_pre",
                **boot,
                "negative_favours_audited": True,
                "n": int(len(crossfit)),
                "issuers": int(crossfit.issuer_ticker.nunique()),
            }
        )
    return pd.DataFrame(state_rows), pd.DataFrame(difference_rows)


def run_predictive_validity(
    panel: pd.DataFrame,
    settings: PredictiveValiditySettings,
) -> dict[str, pd.DataFrame]:
    predictive_cases = build_predictive_cases(panel, settings)
    coefficients, pooled_fit = pooled_state_comparisons(predictive_cases, settings)
    oos_predictions, oos_folds = expanding_oos_predictions(
        predictive_cases, settings
    )
    oos_summary, oos_difference = summarize_oos(oos_predictions, settings)

    aq_cases = build_accrual_quality_cases(panel, settings)
    aq_coefficients = accrual_quality_coefficients(aq_cases, settings)
    aq_crossfit = accrual_quality_crossfit(aq_cases, settings)
    aq_summary, aq_difference = summarize_accrual_quality(aq_crossfit, settings)

    sample_rows: list[dict[str, object]] = []
    for test_name in PREDICTIVE_TESTS:
        clean, outcome, predictors = _test_complete_cases(
            predictive_cases, test_name
        )
        sample_rows.append(
            {
                "analysis": test_name,
                "outcome": outcome,
                "predictors": "|".join(predictors),
                "rows": int(len(clean)),
                "issuers": int(clean.issuer_ticker.nunique()),
                "year_min": int(clean.fiscal_year.min()) if len(clean) else np.nan,
                "year_max": int(clean.fiscal_year.max()) if len(clean) else np.nan,
            }
        )
    sample_rows.append(
        {
            "analysis": "accrual_quality",
            "outcome": "wca_scaled",
            "predictors": "cfo_lag_audited|cfo_current_state|cfo_lead_audited",
            "rows": int(len(aq_cases)),
            "issuers": int(aq_cases.issuer_ticker.nunique()),
            "year_min": int(aq_cases.fiscal_year.min()) if len(aq_cases) else np.nan,
            "year_max": int(aq_cases.fiscal_year.max()) if len(aq_cases) else np.nan,
        }
    )
    return {
        "predictive_validity_cases": predictive_cases,
        "predictive_validity_sample_manifest": pd.DataFrame(sample_rows),
        "predictive_validity_coefficients": coefficients,
        "predictive_validity_pooled_fit": pooled_fit,
        "predictive_validity_oos_folds": oos_folds,
        "predictive_validity_oos_predictions": oos_predictions,
        "predictive_validity_oos_summary": oos_summary,
        "predictive_validity_oos_state_differences": oos_difference,
        "accrual_quality_cases": aq_cases,
        "accrual_quality_coefficients": aq_coefficients,
        "accrual_quality_crossfit_cases": aq_crossfit,
        "accrual_quality_summary": aq_summary,
        "accrual_quality_state_differences": aq_difference,
    }
