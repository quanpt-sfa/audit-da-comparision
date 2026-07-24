from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .architecture import (
    _apply_bounds,
    _design_pairs,
    _winsor_bounds,
    build_attribution_cases as _build_attribution_cases_legacy,
)
from .core import (
    BENCHMARKS,
    DEFAULT_MODELS,
    KEYS,
    CompletionSettings,
    configure_worker_environment,
    stable_task_seed,
)
from .method_contract import LOCKED_METHOD_CONTRACT
from .parallel import _run_tasks


def _fit_model_no_intercept(
    training: pd.DataFrame,
    features: Sequence[str],
) -> tuple[StandardScaler, LinearRegression, float]:
    """Fit the locked Jones-family regression without an ordinary intercept.

    Predictors may be rescaled for numerical stability but are not centred. Centring
    combined with ``fit_intercept=False`` would reintroduce an effective intercept
    after transforming predictions back to the original feature space.
    """
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
        raise AssertionError("Locked Jones estimator reintroduced an ordinary intercept")
    return scaler, model, residual_sd


def estimate_accrual_architectures(
    panel: pd.DataFrame,
    settings: CompletionSettings,
    models: Mapping[str, Sequence[str]] = DEFAULT_MODELS,
    industry_column: str = "icb_l1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the locked no-ordinary-intercept accrual architectures."""
    panel = panel.copy()
    pair = _design_pairs(panel, settings)
    source = panel.loc[
        panel.fiscal_year.between(
            settings.training_start_year, settings.test_end_year
        )
    ].copy()
    result_rows: list[pd.DataFrame] = []
    manifest: list[dict] = []
    architectures = (
        "pooled",
        "industry_fe",
        "industry_slopes",
        "trailing_pooled",
    )

    for test_year in range(settings.test_start_year, settings.test_end_year + 1):
        pair_y = pair.loc[pair.fiscal_year.eq(test_year)].copy()
        if pair_y.empty:
            continue
        historical_all = source.loc[
            source.audit_status.eq(settings.audited_label)
            & source.fiscal_year.between(
                settings.training_start_year, test_year - 1
            )
        ].copy()

        for model_name, features0 in models.items():
            features = list(features0)
            if LOCKED_METHOD_CONTRACT["jones_scale_regressor"] not in features:
                raise ValueError(
                    f"Model {model_name} omits the locked scale regressor inv_assets"
                )
            needed = ["ta_scaled", *features]
            missing = [column for column in needed if column not in source]
            if missing:
                manifest.append(
                    {
                        "test_year": test_year,
                        "model": model_name,
                        "architecture": "all",
                        "status": "missing_columns",
                        "detail": ",".join(missing),
                    }
                )
                continue

            for architecture in architectures:
                historical = historical_all.copy()
                if architecture == "trailing_pooled":
                    historical = historical.loc[
                        historical.fiscal_year.ge(
                            max(
                                settings.training_start_year,
                                test_year - settings.trailing_years,
                            )
                        )
                    ]
                if architecture == "industry_slopes":
                    if (
                        industry_column not in source
                        or f"{industry_column}_pre" not in pair_y
                    ):
                        manifest.append(
                            {
                                "test_year": test_year,
                                "model": model_name,
                                "architecture": architecture,
                                "status": "missing_industry",
                                "detail": industry_column,
                            }
                        )
                        continue
                    groups = sorted(
                        pair_y[f"{industry_column}_pre"].dropna().unique()
                    )
                else:
                    groups = [None]

                for group_value in groups:
                    train = historical.copy()
                    current = pair_y.copy()
                    group_label = "all"
                    if architecture == "industry_slopes":
                        group_label = str(group_value)
                        train = train.loc[train[industry_column].eq(group_value)]
                        current = current.loc[
                            current[f"{industry_column}_pre"].eq(group_value)
                        ]

                    complete_train = (
                        train.replace([np.inf, -np.inf], np.nan)
                        .dropna(subset=needed)
                        .copy()
                    )
                    min_rows = (
                        settings.min_industry_rows
                        if architecture == "industry_slopes"
                        else settings.min_train_rows
                    )
                    if len(complete_train) < min_rows or current.empty:
                        manifest.append(
                            {
                                "test_year": test_year,
                                "model": model_name,
                                "architecture": architecture,
                                "group": group_label,
                                "status": "insufficient_rows",
                                "train_rows": len(complete_train),
                                "current_rows": len(current),
                            }
                        )
                        continue

                    fit_features = features.copy()
                    if architecture == "industry_fe":
                        if (
                            industry_column not in complete_train
                            or f"{industry_column}_pre" not in current
                        ):
                            manifest.append(
                                {
                                    "test_year": test_year,
                                    "model": model_name,
                                    "architecture": architecture,
                                    "status": "missing_industry",
                                    "detail": industry_column,
                                }
                            )
                            continue
                        categories = sorted(
                            complete_train[industry_column]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                        # With no global intercept, retain every industry indicator.
                        for category in categories:
                            column = f"__ind_{category}"
                            complete_train[column] = (
                                complete_train[industry_column]
                                .astype(str)
                                .eq(category)
                                .astype(float)
                            )
                            current[f"{column}_pre"] = (
                                current[f"{industry_column}_pre"]
                                .astype(str)
                                .eq(category)
                                .astype(float)
                            )
                            post_industry = current.get(
                                f"{industry_column}_post",
                                current[f"{industry_column}_pre"],
                            )
                            current[f"{column}_post"] = (
                                post_industry.astype(str)
                                .eq(category)
                                .astype(float)
                            )
                            fit_features.append(column)

                    bounds = _winsor_bounds(
                        complete_train,
                        ["ta_scaled", *fit_features],
                        settings.winsor_lower,
                        settings.winsor_upper,
                    )
                    fit_train = _apply_bounds(complete_train, bounds)
                    scaler, model, residual_sd = _fit_model_no_intercept(
                        fit_train, fit_features
                    )

                    for benchmark in BENCHMARKS:
                        current_b = current.copy()
                        state_data: dict[str, pd.DataFrame] = {}
                        for state in ("pre", "post"):
                            state_frame = pd.DataFrame(index=current_b.index)
                            for feature in fit_features:
                                if feature.startswith("__ind_"):
                                    state_frame[feature] = current_b[
                                        f"{feature}_{state}"
                                    ]
                                    continue
                                if benchmark == "version_specific":
                                    suffix = state
                                elif benchmark == "pre_reference":
                                    suffix = "pre"
                                else:
                                    suffix = "post"
                                state_frame[feature] = current_b[
                                    f"{feature}_{suffix}"
                                ]
                            state_frame["ta_scaled"] = current_b[
                                f"ta_scaled_{state}"
                            ]
                            state_data[state] = _apply_bounds(
                                state_frame, bounds
                            )

                        valid = (
                            state_data["pre"]
                            .replace([np.inf, -np.inf], np.nan)
                            .notna()
                            .all(axis=1)
                        )
                        valid &= (
                            state_data["post"]
                            .replace([np.inf, -np.inf], np.nan)
                            .notna()
                            .all(axis=1)
                        )
                        if not valid.any():
                            continue
                        xpre = state_data["pre"].loc[valid]
                        xpost = state_data["post"].loc[valid]
                        nda_pre = model.predict(
                            scaler.transform(xpre[fit_features])
                        )
                        nda_post = model.predict(
                            scaler.transform(xpost[fit_features])
                        )
                        da_pre = xpre.ta_scaled.to_numpy(float) - nda_pre
                        da_post = xpost.ta_scaled.to_numpy(float) - nda_post
                        keys = current_b.loc[valid, KEYS].reset_index(drop=True)
                        result_rows.append(
                            keys.assign(
                                model=model_name,
                                architecture=architecture,
                                architecture_group=group_label,
                                benchmark=benchmark,
                                da_pre=da_pre,
                                da_post=da_post,
                                nda_pre=nda_pre,
                                nda_post=nda_post,
                                signed_shift=da_post - da_pre,
                                reduction=np.abs(da_pre) - np.abs(da_post),
                                historical_residual_sd=residual_sd,
                                train_rows=len(fit_train),
                                train_min_year=int(fit_train.fiscal_year.min()),
                                train_max_year=int(fit_train.fiscal_year.max()),
                                ordinary_intercept=False,
                                feature_centering=False,
                                scale_regressor="inv_assets",
                            )
                        )

                    manifest.append(
                        {
                            "test_year": test_year,
                            "model": model_name,
                            "architecture": architecture,
                            "group": group_label,
                            "status": "estimated",
                            "train_rows": len(fit_train),
                            "residual_sd": residual_sd,
                            "ordinary_intercept": False,
                            "feature_centering": False,
                            "scale_regressor": "inv_assets",
                        }
                    )

    if not result_rows:
        raise ValueError("No accrual architecture rows were estimated")
    return pd.concat(result_rows, ignore_index=True), pd.DataFrame(manifest)


def build_attribution_cases(
    accrual_rows: pd.DataFrame,
    panel: pd.DataFrame,
    settings: CompletionSettings,
) -> pd.DataFrame:
    """Build and audit the locked three-player Shapley estimand."""
    cases = _build_attribution_cases_legacy(accrual_rows, panel, settings)
    reconstructed = cases[["phi_pat", "phi_cfo", "phi_benchmark"]].sum(axis=1)
    error = (reconstructed - cases["reduction"]).abs()
    tolerance = max(float(settings.negligible_sd) * 100.0, 1.0e-10)
    if len(error) and float(error.max()) > tolerance:
        raise AssertionError(
            "Three-player Shapley efficiency identity failed: "
            f"max_error={float(error.max())} tolerance={tolerance}"
        )
    cases["shapley_efficiency_error"] = error
    cases["attribution_estimand"] = LOCKED_METHOD_CONTRACT[
        "attribution_estimand"
    ]
    cases["attribution_player_count"] = 3
    return cases


def _within_cell_permutations(
    values: np.ndarray,
    cells: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently permute values inside each fiscal-year cell."""
    values = np.asarray(values, dtype=float)
    cells = np.asarray(cells)
    if len(values) != len(cells):
        raise ValueError("values and cells must have identical lengths")
    output = np.broadcast_to(values, (batch_size, len(values))).copy()
    for cell in pd.unique(cells):
        indices = np.flatnonzero(cells == cell)
        if len(indices) <= 1:
            continue
        random_keys = rng.random((batch_size, len(indices)), dtype=np.float32)
        order = np.argsort(random_keys, axis=1)
        output[:, indices] = values[indices][order]
    return output


def _mc_p(observed: float, simulated: np.ndarray) -> float:
    return float((1 + np.sum(simulated >= observed)) / (len(simulated) + 1))


def _randomisation_worker_method_locked(task: dict) -> list[dict]:
    configure_worker_environment(task["blas_threads"])
    rng = np.random.default_rng(task["seed"])
    pre = np.asarray(task["values_pre"], dtype=float)
    shifts = np.asarray(task["shifts"], dtype=float)
    years = np.asarray(task["fiscal_year"])
    draws = int(task["draws"])
    batch_size = max(1, int(task["batch_size"]))
    symmetric = np.empty(draws, dtype=float)
    reassigned = np.empty(draws, dtype=float)
    pre_sign = np.signbit(pre)[None, :]
    absolute_shift = np.abs(shifts)
    nonzero = shifts != 0

    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        current_batch = stop - start
        signs = rng.integers(
            0, 2, size=(current_batch, len(shifts)), dtype=np.int8
        )
        signs = signs.astype(float) * 2.0 - 1.0
        simulated_shift = np.where(
            nonzero[None, :], absolute_shift[None, :] * signs, 0.0
        )
        symmetric[start:stop] = np.mean(
            pre_sign != np.signbit(pre[None, :] + simulated_shift), axis=1
        )

        permuted_shift = _within_cell_permutations(
            shifts, years, current_batch, rng
        )
        reassigned[start:stop] = np.mean(
            pre_sign != np.signbit(pre[None, :] + permuted_shift), axis=1
        )

    rows: list[dict] = []
    for benchmark, values in (
        ("symmetric_sign", symmetric),
        ("signed_shift_reassignment", reassigned),
    ):
        rows.append(
            {
                "outcome": task["outcome"],
                "model": task["model"],
                "benchmark_model": task["benchmark_model"],
                "benchmark": benchmark,
                "observed_switch_rate": task["observed_switch"],
                "sim_mean": float(values.mean()),
                "sim_p025": float(np.quantile(values, 0.025)),
                "sim_p975": float(np.quantile(values, 0.975)),
                "mc_p": _mc_p(task["observed_switch"], values),
                "draws": draws,
                "seed": int(task["seed"]),
                "reassignment_cell": (
                    "fiscal_year"
                    if benchmark == "signed_shift_reassignment"
                    else "not_applicable"
                ),
                "cell_count": int(pd.Series(years).nunique()),
            }
        )
    return rows


def randomisation_benchmarks(
    direct: pd.DataFrame,
    model_cases: pd.DataFrame,
    settings: CompletionSettings,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run sign benchmarks with signed shifts reassigned within fiscal year."""
    tasks: list[dict] = []
    direct_clean = direct.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["cfo_pre", "cfo_post", "fiscal_year"]
    )
    tasks.append(
        {
            "outcome": "cfo_sign",
            "model": "direct",
            "benchmark_model": "direct",
            "values_pre": direct_clean.cfo_pre.to_numpy(float),
            "shifts": (
                direct_clean.cfo_post - direct_clean.cfo_pre
            ).to_numpy(float),
            "fiscal_year": direct_clean.fiscal_year.to_numpy(),
            "observed_switch": float(direct_clean.cfo_sign_switch.mean()),
            "draws": settings.simulation_draws,
            "batch_size": settings.simulation_batch_size,
            "blas_threads": settings.blas_threads_per_worker,
            "seed": stable_task_seed(
                settings.seed, "randomisation", "cfo_sign", "direct"
            ),
        }
    )

    if not model_cases.empty:
        for (model, benchmark), group0 in model_cases.groupby(
            ["model", "benchmark"], observed=True
        ):
            group = group0.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["da_pre", "signed_shift", "fiscal_year"]
            )
            tasks.append(
                {
                    "outcome": "da_sign",
                    "model": model,
                    "benchmark_model": benchmark,
                    "values_pre": group.da_pre.to_numpy(float),
                    "shifts": group.signed_shift.to_numpy(float),
                    "fiscal_year": group.fiscal_year.to_numpy(),
                    "observed_switch": float(group.da_sign_switch.mean()),
                    "draws": settings.simulation_draws,
                    "batch_size": settings.simulation_batch_size,
                    "blas_threads": settings.blas_threads_per_worker,
                    "seed": stable_task_seed(
                        settings.seed,
                        "randomisation",
                        "da_sign",
                        model,
                        benchmark,
                    ),
                }
            )

    outputs = _run_tasks(
        tasks,
        _randomisation_worker_method_locked,
        settings,
        "randomisation-within-year",
        progress,
    )
    rows = [row for output in outputs for row in output]
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["outcome", "model", "benchmark_model", "benchmark"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
