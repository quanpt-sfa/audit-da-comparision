from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diag_common import KEYS


OUTCOMES = (
    "any_candidate",
    "audited_cfo_decrease",
    "audited_cfo_increase",
    "cff_down_candidate",
    "cfi_up_candidate",
)

OUTCOME_SCORE_RULES = {
    "any_candidate": "absolute",
    "audited_cfo_decrease": "positive",
    "audited_cfo_increase": "negative",
    "cff_down_candidate": "positive",
    "cfi_up_candidate": "negative",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _validation_score(values: pd.Series, rule: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if rule == "absolute":
        return numeric.abs()
    if rule == "negative":
        return -numeric
    return numeric


def _winsorized(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    lo, hi = np.quantile(finite, [lower, upper])
    return np.clip(values, lo, hi)


def build_preliminary_proxy_panel(
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    pre = (
        panel[
            panel["audit_status"].eq(
                settings.get("unaudited_label", "unaudited")
            )
        ]
        .drop_duplicates(KEYS)
        .copy()
    )
    numeric_columns = [
        "lag_assets",
        "cfo",
        "pat",
        "revenue",
        "drev",
        "drec",
        "inv_assets",
        "loss",
    ]
    for column in numeric_columns:
        pre[column] = _numeric(pre, column)

    scale = pre["lag_assets"]
    pre["pre_cfo_scaled"] = pre["cfo"] / scale
    pre["pre_pat_scaled"] = pre["pat"] / scale
    pre["pre_revenue_scaled"] = pre["revenue"] / scale
    pre["pre_drev_scaled"] = pre["drev"] / scale
    pre["pre_drec_scaled"] = pre["drec"] / scale
    pre["pre_loss"] = pre["pat"].lt(0).astype(float)

    pre = pre.sort_values(["issuer_ticker", "fiscal_year"]).copy()
    pre["firm_prior_cfo_median"] = (
        pre.groupby("issuer_ticker", observed=True)["pre_cfo_scaled"]
        .transform(
            lambda series: series.shift(1).expanding(min_periods=1).median()
        )
    )
    pre["firm_history_cfo_deviation"] = (
        pre["pre_cfo_scaled"] - pre["firm_prior_cfo_median"]
    )
    return pre


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
    ridge: float,
) -> np.ndarray:
    train = (
        train[predictors + ["pre_cfo_scaled"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    test_x = (
        test[predictors]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    bounds: dict[str, tuple[float, float]] = {}
    for column in predictors + ["pre_cfo_scaled"]:
        bounds[column] = (
            float(train[column].quantile(0.01)),
            float(train[column].quantile(0.99)),
        )
        train[column] = train[column].clip(*bounds[column])
    for column in predictors:
        test_x[column] = test_x[column].clip(*bounds[column])

    x = np.column_stack(
        [np.ones(len(train)), train[predictors].to_numpy(float)]
    )
    y = train["pre_cfo_scaled"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return (
        np.column_stack(
            [np.ones(len(test_x)), test_x.to_numpy(float)]
        )
        @ beta
    )


def _fold_metrics(
    out: pd.DataFrame,
    train_rows: int,
    model: str,
    year: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    residual = (
        pd.to_numeric(out["pre_cfo_scaled"], errors="coerce")
        - pd.to_numeric(out["expected_cfo_scaled"], errors="coerce")
    ).to_numpy(float)
    finite = np.isfinite(residual)
    residual = residual[finite]
    identifiers = out.loc[finite, "issuer_ticker"].astype(str).to_numpy()
    if residual.size == 0:
        return {
            "fiscal_year": year,
            "proxy_model": model,
            "train_rows": train_rows,
            "test_rows": 0,
            "status": "NO_FINITE_TEST_ROWS",
        }

    absolute = np.abs(residual)
    lower = float(settings.get("fold_winsor_lower", 0.01))
    upper = float(settings.get("fold_winsor_upper", 0.99))
    winsorized = _winsorized(residual, lower, upper)
    cutoff = np.quantile(absolute, 0.99)
    leave_tail = residual[absolute <= cutoff]
    maximum = int(np.argmax(absolute))

    return {
        "fiscal_year": year,
        "proxy_model": model,
        "train_rows": train_rows,
        "test_rows": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "winsorized_rmse": float(np.sqrt(np.mean(winsorized**2))),
        "rmse_ex_top_1pct": float(np.sqrt(np.mean(leave_tail**2)))
        if leave_tail.size
        else np.nan,
        "mae": float(absolute.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error": float(absolute[maximum]),
        "maximum_error_issuer": identifiers[maximum],
        "status": "OK",
    }


def _baseline_predictions(test: pd.DataFrame, year: int) -> list[pd.DataFrame]:
    columns = [
        c
        for c in (
            KEYS
            + [
                "raw_exchange",
                "lag_assets",
                "pre_cfo_scaled",
                "industry",
                "industry_name",
                "sector",
                "sector_name",
                "financial_flag",
            ]
        )
        if c in test.columns
    ]
    base = test[columns].copy()
    output: list[pd.DataFrame] = []

    raw = base.copy()
    raw["proxy_model"] = "raw_cfo_level"
    raw["expected_cfo_scaled"] = 0.0
    raw["abnormal_cfo_proxy"] = raw["pre_cfo_scaled"]
    raw["proxy_family"] = "simple_baseline"
    output.append(raw)

    percentile = base.copy()
    percentile["proxy_model"] = "within_year_cfo_percentile"
    percentile["expected_cfo_scaled"] = np.nan
    percentile["abnormal_cfo_proxy"] = (
        percentile["pre_cfo_scaled"].rank(pct=True, method="average") - 0.5
    )
    percentile["proxy_family"] = "simple_baseline"
    output.append(percentile)

    if "firm_history_cfo_deviation" in test:
        history_columns = columns + ["firm_history_cfo_deviation"]
        history = test[history_columns].dropna(
            subset=["firm_history_cfo_deviation"]
        ).copy()
        if not history.empty:
            history["proxy_model"] = "firm_history_deviation"
            history["expected_cfo_scaled"] = (
                history["pre_cfo_scaled"]
                - history["firm_history_cfo_deviation"]
            )
            history["abnormal_cfo_proxy"] = history[
                "firm_history_cfo_deviation"
            ]
            history["proxy_family"] = "simple_baseline"
            history = history.drop(columns=["firm_history_cfo_deviation"])
            output.append(history)

    for frame in output:
        frame["fiscal_year"] = year
    return output


def rolling_expected_cfo_proxies(
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = build_preliminary_proxy_panel(panel, settings)
    models = settings.get("proxy_models", {})
    records: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    minimum_year = int(settings.get("minimum_test_year", 2018))
    maximum_year = int(settings.get("maximum_test_year", 2025))
    minimum_train = int(settings.get("minimum_train_rows", 500))
    ridge = float(settings.get("ridge", 1e-8))

    for year in range(minimum_year, maximum_year + 1):
        train = data[data["fiscal_year"].lt(year)]
        test = data[data["fiscal_year"].eq(year)].copy()
        if test.empty:
            continue

        records.extend(_baseline_predictions(test, year))

        optional = [
            c
            for c in [
                "industry",
                "industry_name",
                "sector",
                "sector_name",
                "financial_flag",
            ]
            if c in test.columns
        ]
        for name, predictors in models.items():
            predictors = list(predictors)
            train_valid = (
                train[predictors + ["pre_cfo_scaled"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            test_columns = (
                KEYS
                + ["raw_exchange", "lag_assets", "pre_cfo_scaled"]
                + optional
                + predictors
            )
            test_valid = (
                test[test_columns]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=predictors + ["pre_cfo_scaled"])
            )
            if len(train_valid) < minimum_train or test_valid.empty:
                folds.append(
                    {
                        "fiscal_year": year,
                        "proxy_model": name,
                        "train_rows": len(train_valid),
                        "test_rows": len(test_valid),
                        "status": "INSUFFICIENT_SAMPLE",
                    }
                )
                continue

            predicted = _fit_predict(train_valid, test_valid, predictors, ridge)
            keep = (
                KEYS
                + ["raw_exchange", "lag_assets", "pre_cfo_scaled"]
                + optional
            )
            out = test_valid[keep].copy()
            out["proxy_model"] = name
            out["expected_cfo_scaled"] = predicted
            out["abnormal_cfo_proxy"] = out["pre_cfo_scaled"] - predicted
            out["proxy_family"] = "expected_cfo_model"
            records.append(out)
            folds.append(_fold_metrics(out, len(train_valid), name, year, settings))

    predictions = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not predictions.empty:
        predictions["proxy_rank_within_year_signed"] = (
            predictions.groupby(
                ["proxy_model", "fiscal_year"], observed=True
            )["abnormal_cfo_proxy"].rank(pct=True, method="average")
        )
    return predictions, pd.DataFrame(folds)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    if y.sum() == 0:
        return np.nan
    ranked = y[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].mean())


def _observed_outcomes(
    observed_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    material = float(settings.get("material_cfo_threshold", 0.05))
    label = settings.get(
        "candidate_label",
        "identity_consistent_offsetting_reclassification_candidate",
    )
    observed = observed_cases.drop_duplicates(KEYS).copy()
    observed["any_candidate"] = observed["cfs_resolution"].eq(label).astype(int)
    delta = _numeric(observed, "delta_cfo_scaled")
    observed["audited_cfo_decrease"] = (
        observed["any_candidate"].eq(1) & delta.le(-material)
    ).astype(int)
    observed["audited_cfo_increase"] = (
        observed["any_candidate"].eq(1) & delta.ge(material)
    ).astype(int)
    offset = observed.get(
        "offset_channel_pattern", pd.Series("", index=observed.index)
    )
    observed["cff_down_candidate"] = (
        observed["audited_cfo_decrease"].eq(1) & offset.eq("cff_dominant")
    ).astype(int)
    observed["cfi_up_candidate"] = (
        observed["audited_cfo_increase"].eq(1) & offset.eq("cfi_dominant")
    ).astype(int)
    return observed


def _sample_masks(
    merged: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    restrictions = settings.get("sample_restrictions", {})
    masks: dict[str, pd.Series] = {"full": pd.Series(True, index=merged.index)}
    status: list[dict[str, Any]] = []

    exchange = merged.get("raw_exchange", pd.Series("", index=merged.index))
    exchange = exchange.astype(str).str.upper()
    listed_values = {
        str(value).upper()
        for value in restrictions.get(
            "listed_exchanges", ["HOSE", "HNX", "UPCOM"]
        )
    }
    listed = exchange.isin(listed_values)
    masks["listed_only"] = listed

    ticker = merged["issuer_ticker"].astype(str)
    valid_ticker = ticker.str.fullmatch(r"[A-Z][A-Z0-9]{1,7}")
    masks["valid_ticker_only"] = valid_ticker

    lag_assets = _numeric(merged, "lag_assets")
    positive_lag = lag_assets[lag_assets.gt(0)]
    quantile = float(restrictions.get("lag_assets_floor_quantile", 0.01))
    lag_floor = (
        float(positive_lag.quantile(quantile)) if len(positive_lag) else np.nan
    )
    lag_ok = (
        lag_assets.ge(lag_floor)
        if np.isfinite(lag_floor)
        else pd.Series(True, index=merged.index)
    )
    masks["lag_assets_floor"] = lag_ok

    flag_columns = [
        column
        for column in restrictions.get(
            "scale_scope_flag_columns",
            [
                "combined_scale_scope_flag",
                "scale_scope_flag",
                "asset_pre_post_gap_flag",
                "asset_growth_flag",
                "small_lag_assets_flag",
            ],
        )
        if column in merged.columns
    ]
    if flag_columns:
        flagged = pd.Series(False, index=merged.index)
        for column in flag_columns:
            flagged |= merged[column].fillna(False).astype(bool)
        scale_scope_ok = ~flagged
        scale_status = "EVALUATED"
        scale_reason = ",".join(flag_columns)
    else:
        scale_scope_ok = pd.Series(True, index=merged.index)
        scale_status = "NOT_EVALUATED"
        scale_reason = "No configured scale/scope flag columns found"
    masks["exclude_scale_scope_flags"] = scale_scope_ok

    financial_status = "NOT_EVALUATED"
    financial_reason = "No industry or financial flag found"
    nonfinancial = pd.Series(True, index=merged.index)
    if "financial_flag" in merged.columns:
        financial = merged["financial_flag"].fillna(False).astype(bool)
        nonfinancial = ~financial
        financial_status = "EVALUATED"
        financial_reason = "financial_flag"
    else:
        industry_column = next(
            (
                column
                for column in [
                    "industry",
                    "industry_name",
                    "sector",
                    "sector_name",
                ]
                if column in merged.columns
            ),
            None,
        )
        if industry_column:
            pattern = restrictions.get(
                "financial_industry_regex",
                r"bank|financial|insurance|securit|ngan hang|bao hiem|chung khoan",
            )
            text = merged[industry_column].fillna("").astype(str).str.lower()
            nonfinancial = ~text.str.contains(pattern, regex=True, na=False)
            financial_status = "EVALUATED"
            financial_reason = industry_column
    masks["nonfinancial_only"] = nonfinancial

    masks["analysis_core"] = (
        listed & valid_ticker & lag_ok & scale_scope_ok & nonfinancial
    )

    for name, mask in masks.items():
        if name == "exclude_scale_scope_flags":
            evaluation_status = scale_status
            reason = scale_reason
        elif name == "nonfinancial_only":
            evaluation_status = financial_status
            reason = financial_reason
        else:
            evaluation_status = "EVALUATED"
            reason = "OK"
        status.append(
            {
                "sample_restriction": name,
                "status": evaluation_status,
                "reason": reason,
                "model_rows": int(mask.sum()),
                "firm_years": int(merged.loc[mask, KEYS].drop_duplicates().shape[0]),
                "share_model_rows": float(mask.mean()),
                "lag_assets_floor": lag_floor
                if name in {"lag_assets_floor", "analysis_core"}
                else np.nan,
            }
        )
    return masks, pd.DataFrame(status)


def _common_keys(predictions: pd.DataFrame, models: list[str]) -> pd.MultiIndex:
    selected = predictions[predictions["proxy_model"].isin(models)]
    if selected.empty:
        return pd.MultiIndex.from_arrays([[], []], names=KEYS)
    counts = selected.groupby(KEYS, observed=True)["proxy_model"].nunique()
    return counts[counts.eq(len(models))].index


def _metric_record(
    group: pd.DataFrame,
    outcome: str,
    sample_mode: str,
    sample_restriction: str,
) -> dict[str, Any]:
    rule = OUTCOME_SCORE_RULES[outcome]
    score = _validation_score(group["abnormal_cfo_proxy"], rule)
    y = group[outcome].to_numpy(int)
    rank = score.groupby(group["fiscal_year"]).rank(pct=True, method="average")
    top = rank.ge(0.90)
    prevalence = float(np.mean(y))
    top_rate = float(group.loc[top, outcome].mean()) if top.any() else np.nan
    return {
        "proxy_model": str(group["proxy_model"].iloc[0]),
        "proxy_family": str(group["proxy_family"].iloc[0]),
        "outcome": outcome,
        "score_rule": rule,
        "sample_mode": sample_mode,
        "sample_restriction": sample_restriction,
        "rows": len(group),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auc": _auc(y, score.to_numpy(float)),
        "average_precision": _ap(y, score.to_numpy(float)),
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / prevalence if prevalence > 0 else np.nan,
        "mean_raw_proxy_positive": float(
            group.loc[group[outcome].eq(1), "abnormal_cfo_proxy"].mean()
        ),
        "mean_raw_proxy_negative": float(
            group.loc[group[outcome].eq(0), "abnormal_cfo_proxy"].mean()
        ),
        "mean_validation_score_positive": float(
            score[group[outcome].eq(1)].mean()
        ),
        "mean_validation_score_negative": float(
            score[group[outcome].eq(0)].mean()
        ),
    }


def validate_proxy_predictions(
    predictions: pd.DataFrame,
    observed_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    observed = _observed_outcomes(observed_cases, settings)
    observed_columns = list(KEYS) + list(OUTCOMES)
    observed_columns += [
        column
        for column in settings.get("sample_restrictions", {}).get(
            "scale_scope_flag_columns",
            [
                "combined_scale_scope_flag",
                "scale_scope_flag",
                "asset_pre_post_gap_flag",
                "asset_growth_flag",
                "small_lag_assets_flag",
            ],
        )
        if column in observed.columns
    ]
    merged = predictions.merge(
        observed[observed_columns],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    masks, restriction_status = _sample_masks(merged, settings)
    comparison_models = list(
        settings.get(
            "common_sample_models",
            list(settings.get("proxy_models", {}).keys())
            + ["raw_cfo_level", "within_year_cfo_percentile"],
        )
    )
    common = _common_keys(merged, comparison_models)
    key_index = pd.MultiIndex.from_frame(merged[KEYS])
    common_mask = key_index.isin(common)

    summary: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []

    for restriction_name, restriction_mask in masks.items():
        restricted = merged.loc[restriction_mask].copy()
        restricted_common = pd.Series(
            common_mask[restriction_mask.to_numpy()], index=restricted.index
        )
        for sample_mode, mode_mask in (
            ("model_available", pd.Series(True, index=restricted.index)),
            ("common_models", restricted_common),
        ):
            sample = restricted.loc[mode_mask].copy()
            if sample.empty:
                continue
            for _, group in sample.groupby("proxy_model", observed=True):
                group = group.copy()
                for outcome in OUTCOMES:
                    summary.append(
                        _metric_record(
                            group, outcome, sample_mode, restriction_name
                        )
                    )
                    for year, year_group in group.groupby(
                        "fiscal_year", observed=True
                    ):
                        record = _metric_record(
                            year_group, outcome, sample_mode, restriction_name
                        )
                        record["fiscal_year"] = int(year)
                        yearly.append(record)

                if (
                    sample_mode == "common_models"
                    and restriction_name == "analysis_core"
                ):
                    case = group.copy()
                    case["sample_mode"] = sample_mode
                    case["sample_restriction"] = restriction_name
                    case_frames.append(case)

    summary_frame = pd.DataFrame(summary)
    incremental = pd.DataFrame()
    if not summary_frame.empty:
        reference_model = settings.get(
            "incremental_reference_model", "raw_cfo_level"
        )
        base = summary_frame[
            summary_frame["proxy_model"].eq(reference_model)
        ][
            [
                "outcome",
                "sample_mode",
                "sample_restriction",
                "auc",
                "average_precision",
                "top_decile_lift",
            ]
        ].rename(
            columns={
                "auc": "reference_auc",
                "average_precision": "reference_average_precision",
                "top_decile_lift": "reference_top_decile_lift",
            }
        )
        incremental = summary_frame.merge(
            base,
            on=["outcome", "sample_mode", "sample_restriction"],
            how="left",
            validate="many_to_one",
        )
        incremental["delta_auc_vs_reference"] = (
            incremental["auc"] - incremental["reference_auc"]
        )
        incremental["delta_ap_vs_reference"] = (
            incremental["average_precision"]
            - incremental["reference_average_precision"]
        )
        incremental["delta_lift_vs_reference"] = (
            incremental["top_decile_lift"]
            - incremental["reference_top_decile_lift"]
        )

    return {
        "cfs_shifting_proxy_cases": merged,
        "cfs_shifting_proxy_validation": summary_frame,
        "cfs_shifting_proxy_validation_by_year": pd.DataFrame(yearly),
        "cfs_shifting_proxy_incremental_comparison": incremental,
        "cfs_proxy_sample_restriction_status": restriction_status,
        "cfs_shifting_proxy_common_core_cases": (
            pd.concat(case_frames, ignore_index=True)
            if case_frames
            else pd.DataFrame()
        ),
    }
