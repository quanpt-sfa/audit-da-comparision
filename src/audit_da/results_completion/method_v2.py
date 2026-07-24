from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .architecture import _apply_bounds, _design_pairs, _winsor_bounds
from .core import BENCHMARKS, DEFAULT_MODELS, KEYS, CompletionSettings, _numeric, paired_panel
from .final_contract import LOCKED_FINAL_CONTRACT


_FIXED_REFERENCE_BENCHMARKS = tuple(
    LOCKED_FINAL_CONTRACT["attribution_benchmarks"]
)


def _fit_model_no_intercept(
    training: pd.DataFrame,
    features: Sequence[str],
) -> tuple[StandardScaler, LinearRegression, float]:
    columns = list(features)
    scaler = StandardScaler(with_mean=False).fit(training[columns])
    design = scaler.transform(training[columns])
    model = LinearRegression(fit_intercept=False).fit(
        design, training["ta_scaled"]
    )
    fitted = model.predict(design)
    residual = training["ta_scaled"].to_numpy(float) - fitted
    residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan
    if bool(model.fit_intercept) or bool(scaler.with_mean):
        raise AssertionError("Final Jones estimator reintroduced an ordinary intercept")
    return scaler, model, residual_sd


def _predictor_bounds(
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {key: value for key, value in bounds.items() if key != "ta_scaled"}


def _validate_training_history(
    training_panel: pd.DataFrame,
    settings: CompletionSettings,
) -> None:
    required = {"fiscal_year", "audit_status"}
    missing = sorted(required - set(training_panel.columns))
    if missing:
        raise ValueError(f"Training panel missing columns: {missing}")
    start = training_panel.loc[
        training_panel["fiscal_year"].eq(settings.training_start_year)
        & training_panel["audit_status"].eq(settings.audited_label)
    ]
    if start.empty:
        raise ValueError(
            "The unrestricted training panel has no audited observations in "
            f"training_start_year={settings.training_start_year}. The Methods promise "
            "cannot be satisfied by silently starting one year later."
        )


def estimate_accrual_architectures(
    analysis_panel: pd.DataFrame,
    training_panel: pd.DataFrame,
    settings: CompletionSettings,
    models: Mapping[str, Sequence[str]] = DEFAULT_MODELS,
    industry_column: str = "icb_l1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate locked test cases from unrestricted prior audited history.

    Historical outcomes and predictors are winsorised for fitting. Current-state
    predictors may be clipped to historical support, but current total-accrual
    outcomes are never clipped. Fixed-reference PAT/CFO movements therefore retain
    their accounting identity.
    """
    analysis_panel = analysis_panel.copy()
    training_panel = training_panel.copy()
    _validate_training_history(training_panel, settings)
    pair = _design_pairs(analysis_panel, settings)
    source = training_panel.loc[
        training_panel.fiscal_year.between(
            settings.training_start_year, settings.test_end_year
        )
    ].copy()
    result_rows: list[pd.DataFrame] = []
    manifest: list[dict] = []
    architectures = ("pooled", "industry_fe", "industry_slopes", "trailing_pooled")

    for test_year in range(settings.test_start_year, settings.test_end_year + 1):
        pair_y = pair.loc[pair.fiscal_year.eq(test_year)].copy()
        if pair_y.empty:
            continue
        historical_all = source.loc[
            source.audit_status.eq(settings.audited_label)
            & source.fiscal_year.between(settings.training_start_year, test_year - 1)
        ].copy()
        for model_name, features0 in models.items():
            features = list(features0)
            if LOCKED_FINAL_CONTRACT["jones_scale_regressor"] not in features:
                raise ValueError(f"Model {model_name} omits inv_assets")
            needed = ["ta_scaled", *features]
            missing = [column for column in needed if column not in source]
            if missing:
                manifest.append({
                    "test_year": test_year, "model": model_name,
                    "architecture": "all", "status": "missing_columns",
                    "detail": ",".join(missing),
                })
                continue
            for architecture in architectures:
                historical = historical_all.copy()
                if architecture == "trailing_pooled":
                    historical = historical.loc[
                        historical.fiscal_year.ge(max(
                            settings.training_start_year,
                            test_year - settings.trailing_years,
                        ))
                    ]
                if architecture == "industry_slopes":
                    if industry_column not in source or f"{industry_column}_pre" not in pair_y:
                        manifest.append({
                            "test_year": test_year, "model": model_name,
                            "architecture": architecture, "status": "missing_industry",
                            "detail": industry_column,
                        })
                        continue
                    groups = sorted(pair_y[f"{industry_column}_pre"].dropna().unique())
                else:
                    groups = [None]
                for group_value in groups:
                    train = historical.copy()
                    current = pair_y.copy()
                    group_label = "all"
                    if architecture == "industry_slopes":
                        group_label = str(group_value)
                        train = train.loc[train[industry_column].eq(group_value)]
                        current = current.loc[current[f"{industry_column}_pre"].eq(group_value)]
                    complete_train = (
                        train.replace([np.inf, -np.inf], np.nan)
                        .dropna(subset=needed).copy()
                    )
                    min_rows = (
                        settings.min_industry_rows
                        if architecture == "industry_slopes"
                        else settings.min_train_rows
                    )
                    if len(complete_train) < min_rows or current.empty:
                        manifest.append({
                            "test_year": test_year, "model": model_name,
                            "architecture": architecture, "group": group_label,
                            "status": "insufficient_rows",
                            "train_rows": len(complete_train),
                            "current_rows": len(current),
                        })
                        continue
                    fit_features = features.copy()
                    if architecture == "industry_fe":
                        if industry_column not in complete_train or f"{industry_column}_pre" not in current:
                            manifest.append({
                                "test_year": test_year, "model": model_name,
                                "architecture": architecture, "status": "missing_industry",
                                "detail": industry_column,
                            })
                            continue
                        categories = sorted(
                            complete_train[industry_column].dropna().astype(str).unique()
                        )
                        for category in categories:
                            column = f"__ind_{category}"
                            complete_train[column] = (
                                complete_train[industry_column].astype(str).eq(category).astype(float)
                            )
                            current[f"{column}_pre"] = (
                                current[f"{industry_column}_pre"].astype(str).eq(category).astype(float)
                            )
                            post_industry = current.get(
                                f"{industry_column}_post", current[f"{industry_column}_pre"]
                            )
                            current[f"{column}_post"] = (
                                post_industry.astype(str).eq(category).astype(float)
                            )
                            fit_features.append(column)
                    bounds = _winsor_bounds(
                        complete_train, ["ta_scaled", *fit_features],
                        settings.winsor_lower, settings.winsor_upper,
                    )
                    fit_train = _apply_bounds(complete_train, bounds)
                    scaler, model, residual_sd = _fit_model_no_intercept(
                        fit_train, fit_features
                    )
                    current_feature_bounds = _predictor_bounds(bounds)
                    for benchmark in BENCHMARKS:
                        state_data: dict[str, pd.DataFrame] = {}
                        for state in ("pre", "post"):
                            state_frame = pd.DataFrame(index=current.index)
                            for feature in fit_features:
                                if feature.startswith("__ind_"):
                                    state_frame[feature] = current[f"{feature}_{state}"]
                                    continue
                                if benchmark == "version_specific":
                                    suffix = state
                                elif benchmark == "pre_reference":
                                    suffix = "pre"
                                else:
                                    suffix = "post"
                                state_frame[feature] = current[f"{feature}_{suffix}"]
                            raw_outcome = pd.to_numeric(
                                current[f"ta_scaled_{state}"], errors="coerce"
                            )
                            bounded_features = _apply_bounds(
                                state_frame[fit_features], current_feature_bounds
                            )
                            bounded_features["ta_scaled"] = raw_outcome
                            state_data[state] = bounded_features
                        valid = (
                            state_data["pre"].replace([np.inf, -np.inf], np.nan)
                            .notna().all(axis=1)
                        )
                        valid &= (
                            state_data["post"].replace([np.inf, -np.inf], np.nan)
                            .notna().all(axis=1)
                        )
                        if not valid.any():
                            continue
                        xpre = state_data["pre"].loc[valid]
                        xpost = state_data["post"].loc[valid]
                        nda_pre = model.predict(scaler.transform(xpre[fit_features]))
                        nda_post = model.predict(scaler.transform(xpost[fit_features]))
                        da_pre = xpre.ta_scaled.to_numpy(float) - nda_pre
                        da_post = xpost.ta_scaled.to_numpy(float) - nda_post
                        keys = current.loc[valid, KEYS].reset_index(drop=True)
                        result_rows.append(keys.assign(
                            model=model_name, architecture=architecture,
                            architecture_group=group_label, benchmark=benchmark,
                            da_pre=da_pre, da_post=da_post,
                            nda_pre=nda_pre, nda_post=nda_post,
                            signed_shift=da_post - da_pre,
                            reduction=np.abs(da_pre) - np.abs(da_post),
                            historical_residual_sd=residual_sd,
                            train_rows=len(fit_train),
                            train_min_year=int(fit_train.fiscal_year.min()),
                            train_max_year=int(fit_train.fiscal_year.max()),
                            ordinary_intercept=False,
                            feature_centering=False,
                            scale_regressor="inv_assets",
                            current_outcome_clipped=False,
                            training_population=LOCKED_FINAL_CONTRACT["training_population"],
                        ))
                    manifest.append({
                        "test_year": test_year, "model": model_name,
                        "architecture": architecture, "group": group_label,
                        "status": "estimated", "train_rows": len(fit_train),
                        "train_min_year": int(fit_train.fiscal_year.min()),
                        "train_max_year": int(fit_train.fiscal_year.max()),
                        "residual_sd": residual_sd,
                        "ordinary_intercept": False,
                        "feature_centering": False,
                        "scale_regressor": "inv_assets",
                        "current_outcome_clipped": False,
                        "training_population": LOCKED_FINAL_CONTRACT["training_population"],
                    })
    if not result_rows:
        raise ValueError("No final accrual architecture rows were estimated")
    return pd.concat(result_rows, ignore_index=True), pd.DataFrame(manifest)


def _shapley_two(
    da_pre: np.ndarray, pat_move: np.ndarray, cfo_move: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    da_pre = np.asarray(da_pre, dtype=float)
    pat_move = np.asarray(pat_move, dtype=float)
    cfo_move = np.asarray(cfo_move, dtype=float)
    pat_first = np.abs(da_pre) - np.abs(da_pre + pat_move)
    pat_second = np.abs(da_pre + cfo_move) - np.abs(da_pre + cfo_move + pat_move)
    cfo_first = np.abs(da_pre) - np.abs(da_pre + cfo_move)
    cfo_second = np.abs(da_pre + pat_move) - np.abs(da_pre + pat_move + cfo_move)
    return 0.5 * (pat_first + pat_second), 0.5 * (cfo_first + cfo_second)


def build_attribution_cases(
    accrual_rows: pd.DataFrame,
    analysis_panel: pd.DataFrame,
    settings: CompletionSettings,
) -> pd.DataFrame:
    """Build exact PAT/CFO Shapley cases for fixed-reference benchmarks only."""
    pair = paired_panel(analysis_panel, settings)
    needed = ["pat_pre", "pat_post", "cfo_pre", "cfo_post", "lag_assets_pre"]
    missing = [column for column in needed if column not in pair]
    if missing:
        raise ValueError(f"Panel missing attribution columns: {missing}")
    pair = _numeric(pair, needed)
    pair["pat_move"] = (pair.pat_post - pair.pat_pre) / pair.lag_assets_pre
    pair["cfo_move"] = -(pair.cfo_post - pair.cfo_pre) / pair.lag_assets_pre
    fixed = accrual_rows.loc[
        accrual_rows.benchmark.isin(_FIXED_REFERENCE_BENCHMARKS)
    ].copy()
    cases = fixed.merge(
        pair[KEYS + ["pat_move", "cfo_move"]],
        on=KEYS, how="left", validate="many_to_one",
    )
    cases["benchmark_move"] = cases.signed_shift - cases.pat_move - cases.cfo_move
    finite = np.isfinite(cases[[
        "da_pre", "da_post", "pat_move", "cfo_move", "benchmark_move"
    ]]).all(axis=1)
    cases = cases.loc[finite].copy()
    tolerance = max(float(settings.negligible_sd) * 100.0, 1.0e-10)
    max_benchmark_move = float(cases.benchmark_move.abs().max()) if len(cases) else 0.0
    if max_benchmark_move > tolerance:
        raise AssertionError(
            "Fixed-reference attribution is not exhausted by PAT and CFO. "
            f"max_benchmark_move={max_benchmark_move} tolerance={tolerance}. "
            "Check current-outcome clipping or benchmark construction."
        )
    phi_pat, phi_cfo = _shapley_two(
        cases.da_pre.to_numpy(float),
        cases.pat_move.to_numpy(float),
        cases.cfo_move.to_numpy(float),
    )
    cases["phi_pat"] = phi_pat
    cases["phi_cfo"] = phi_cfo
    cases["phi_benchmark"] = 0.0
    cases["component_contrast"] = cases.phi_cfo.abs() - cases.phi_pat.abs()
    cases["cfo_larger"] = cases.phi_cfo.abs() > cases.phi_pat.abs()
    sd = pd.to_numeric(cases.historical_residual_sd, errors="coerce")
    cases["normalised_component_contrast"] = np.where(
        sd > settings.negligible_sd, cases.component_contrast / sd, np.nan
    )
    cases["signed_quadrant"] = np.select(
        [
            (cases.phi_pat >= 0) & (cases.phi_cfo >= 0),
            (cases.phi_pat < 0) & (cases.phi_cfo >= 0),
            (cases.phi_pat >= 0) & (cases.phi_cfo < 0),
        ],
        ["both_reduce_abs_da", "cfo_reduces_pat_increases", "pat_reduces_cfo_increases"],
        default="both_increase_abs_da",
    )
    cases["signed_residual_direction"] = np.sign(cases.signed_shift).astype(int)
    error = (cases.phi_pat + cases.phi_cfo - cases.reduction).abs()
    if len(error) and float(error.max()) > tolerance:
        raise AssertionError(
            "Two-player Shapley efficiency failed: "
            f"max_error={float(error.max())} tolerance={tolerance}"
        )
    cases["shapley_efficiency_error"] = error
    cases["attribution_estimand"] = LOCKED_FINAL_CONTRACT["attribution_estimand"]
    cases["attribution_player_count"] = 2
    cases["current_outcome_clipped"] = False
    return cases


def benchmark_movement_diagnostic(
    accrual_rows: pd.DataFrame,
    analysis_panel: pd.DataFrame,
    settings: CompletionSettings,
) -> pd.DataFrame:
    """Report model-input movement outside the fixed-reference estimand."""
    pair = paired_panel(analysis_panel, settings)
    needed = ["pat_pre", "pat_post", "cfo_pre", "cfo_post", "lag_assets_pre"]
    pair = _numeric(pair, needed)
    pair["pat_move"] = (pair.pat_post - pair.pat_pre) / pair.lag_assets_pre
    pair["cfo_move"] = -(pair.cfo_post - pair.cfo_pre) / pair.lag_assets_pre
    diagnostic = accrual_rows.loc[
        accrual_rows.benchmark.eq("version_specific")
    ].merge(
        pair[KEYS + ["pat_move", "cfo_move"]],
        on=KEYS, how="left", validate="many_to_one",
    )
    diagnostic["benchmark_move"] = (
        diagnostic.signed_shift - diagnostic.pat_move - diagnostic.cfo_move
    )
    return diagnostic[KEYS + [
        "model", "architecture", "benchmark", "pat_move", "cfo_move",
        "benchmark_move", "signed_shift", "reduction",
    ]].copy()
