from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bayes import ApproxHierarchicalBayes
from .stacking import stacking_weights


@dataclass
class FittedCandidate:
    name: str
    features: list[str]
    model: ApproxHierarchicalBayes


def _finite_rows(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    cols = ["ta_scaled", "firm_id"] + features
    return frame[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def _clip_from_training(train: pd.DataFrame, frames: list[pd.DataFrame], columns: list[str], lower: float, upper: float) -> list[pd.DataFrame]:
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        series = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        bounds[column] = (float(series.quantile(lower)), float(series.quantile(upper)))
    output: list[pd.DataFrame] = []
    for frame in frames:
        copied = frame.copy()
        for column, (lo, hi) in bounds.items():
            copied[column] = pd.to_numeric(copied[column], errors="coerce").clip(lo, hi)
        output.append(copied)
    return output


def _fit_candidates(train: pd.DataFrame, model_specs: dict[str, list[str]], seed: int) -> list[FittedCandidate]:
    fitted: list[FittedCandidate] = []
    for offset, (name, features) in enumerate(model_specs.items()):
        mask = _finite_rows(train, features)
        subset = train.loc[mask]
        model = ApproxHierarchicalBayes(random_state=seed + offset)
        model.fit(
            subset[features].to_numpy(float),
            subset["ta_scaled"].to_numpy(float),
            subset["firm_id"].to_numpy(str),
            features,
        )
        fitted.append(FittedCandidate(name, features, model))
    return fitted


def _validation_weights(
    fit_train: pd.DataFrame,
    validation: pd.DataFrame,
    model_specs: dict[str, list[str]],
    seed: int,
    minimum_validation_rows: int,
) -> tuple[np.ndarray, list[str]]:
    if len(validation) < minimum_validation_rows:
        names = list(model_specs)
        return np.repeat(1.0 / len(names), len(names)), names
    candidates = _fit_candidates(fit_train, model_specs, seed)
    y = validation["ta_scaled"].to_numpy(float)
    common = np.isfinite(y)
    means: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    for candidate in candidates:
        valid = _finite_rows(validation, candidate.features).to_numpy()
        common &= valid
    if common.sum() < minimum_validation_rows:
        return np.repeat(1.0 / len(candidates), len(candidates)), [c.name for c in candidates]
    for candidate in candidates:
        pred = candidate.model.posterior_mean_sd(
            validation.loc[common, candidate.features].to_numpy(float),
            validation.loc[common, "firm_id"].to_numpy(str),
            include_residual=True,
        )
        means.append(pred.mean)
        sds.append(pred.sd)
    weights = stacking_weights(y[common], means, sds)
    return weights, [candidate.name for candidate in candidates]


def _paired_rows(panel: pd.DataFrame, year: int, audited_label: str, unaudited_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = panel[panel["fiscal_year"] == year].copy()
    pre = current[current["audit_status"] == unaudited_label].copy()
    post = current[current["audit_status"] == audited_label].copy()
    common = sorted(set(pre["issuer_ticker"]) & set(post["issuer_ticker"]))
    pre = pre[pre["issuer_ticker"].isin(common)].set_index("issuer_ticker").loc[common].reset_index()
    post = post[post["issuer_ticker"].isin(common)].set_index("issuer_ticker").loc[common].reset_index()
    return pre, post


def _correlated_error_draws(
    rows: int,
    selected_sigma: np.ndarray,
    rho: float,
    post_sd_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    draws = len(selected_sigma)
    first = rng.normal(size=(rows, draws))
    second_independent = rng.normal(size=(rows, draws))
    second = rho * first + np.sqrt(max(1.0 - rho * rho, 0.0)) * second_independent
    scale = selected_sigma[None, :]
    return scale * first, post_sd_ratio * scale * second


def classify_draws(pre_da: np.ndarray, post_da: np.ndarray, delta: float) -> dict[str, np.ndarray]:
    pre_abs = np.abs(pre_da)
    post_abs = np.abs(post_da)
    improvement = pre_abs - post_abs
    sign_change = np.signbit(pre_da) != np.signbit(post_da)
    overshoot = sign_change & (post_abs > delta)
    normalization = (~overshoot) & (improvement > delta)
    deterioration = (~overshoot) & (improvement < -delta)
    partial = (~overshoot) & (~normalization) & (~deterioration) & (improvement > 0)
    no_movement = ~(overshoot | normalization | deterioration | partial)
    return {
        "normalization": normalization,
        "partial_correction": partial,
        "overshoot": overshoot,
        "deterioration": deterioration,
        "no_material_movement": no_movement,
    }


def run_signal_gate(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = config["models"]
    signal_cfg = config["signal"]
    panel_cfg = config["panel"]
    audited = config["input"]["audited_label"]
    unaudited = config["input"]["unaudited_label"]
    seed = int(model_cfg["random_seed"])
    draws = int(model_cfg["posterior_draws"])
    model_specs = {name: list(features) for name, features in model_cfg["candidate_models"].items()}
    all_features = sorted({feature for features in model_specs.values() for feature in features})
    lower = float(panel_cfg["winsor_lower"])
    upper = float(panel_cfg["winsor_upper"])
    rng = np.random.default_rng(seed)
    results: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for year in range(int(signal_cfg["minimum_test_year"]), int(signal_cfg["maximum_test_year"]) + 1):
        pre, post = _paired_rows(panel, year, audited, unaudited)
        if pre.empty:
            continue
        train = panel[(panel["audit_status"] == audited) & (panel["fiscal_year"] <= year - 1)].copy()
        fit_train = train[train["fiscal_year"] <= year - 2].copy()
        validation = train[train["fiscal_year"] == year - 1].copy()
        if len(train) < int(model_cfg["minimum_train_rows"]):
            continue
        train, fit_train, validation, pre, post = _clip_from_training(
            train, [train, fit_train, validation, pre, post],
            ["ta_scaled"] + all_features, lower, upper,
        )
        weights, weight_names = _validation_weights(
            fit_train, validation, model_specs, seed + year * 100,
            int(model_cfg["minimum_validation_rows"]),
        )
        candidates = _fit_candidates(train, model_specs, seed + year * 1000)
        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        ordered = [candidate_by_name[name] for name in weight_names]
        model_index = rng.choice(len(ordered), size=draws, p=weights)

        for benchmark in signal_cfg["benchmarks"]:
            model_pre_draws: list[np.ndarray] = []
            model_post_draws: list[np.ndarray] = []
            model_sigmas: list[float] = []
            valid_all = np.ones(len(pre), dtype=bool)
            for candidate in ordered:
                valid_all &= _finite_rows(pre, candidate.features).to_numpy()
                valid_all &= _finite_rows(post, candidate.features).to_numpy()
            pre_valid = pre.loc[valid_all].reset_index(drop=True)
            post_valid = post.loc[valid_all].reset_index(drop=True)
            if pre_valid.empty:
                continue
            for candidate in ordered:
                coef_draws, firm_draws = candidate.model.draw_components(draws, rng)
                if benchmark == "version_specific":
                    x_pre = pre_valid[candidate.features].to_numpy(float)
                    x_post = post_valid[candidate.features].to_numpy(float)
                elif benchmark == "pre_reference":
                    x_pre = pre_valid[candidate.features].to_numpy(float)
                    x_post = x_pre
                elif benchmark == "audited_reference":
                    x_post = post_valid[candidate.features].to_numpy(float)
                    x_pre = x_post
                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")
                firms = pre_valid["firm_id"].to_numpy(str)
                model_pre_draws.append(candidate.model.latent_draws(x_pre, firms, coef_draws, firm_draws))
                model_post_draws.append(candidate.model.latent_draws(x_post, firms, coef_draws, firm_draws))
                model_sigmas.append(np.sqrt(candidate.model.residual_var))

            rows = len(pre_valid)
            latent_pre = np.empty((rows, draws))
            latent_post = np.empty((rows, draws))
            selected_sigma = np.empty(draws)
            for draw_index, candidate_index in enumerate(model_index):
                latent_pre[:, draw_index] = model_pre_draws[candidate_index][:, draw_index]
                latent_post[:, draw_index] = model_post_draws[candidate_index][:, draw_index]
                selected_sigma[draw_index] = model_sigmas[candidate_index]

            y_pre = pre_valid["ta_scaled"].to_numpy(float)[:, None]
            y_post = post_valid["ta_scaled"].to_numpy(float)[:, None]
            for rho in signal_cfg["rho_grid"]:
                for error_sd_ratio in signal_cfg.get("error_sd_ratio_grid", [1.0]):
                    e_pre, e_post = _correlated_error_draws(
                        rows, selected_sigma, float(rho), float(error_sd_ratio), rng
                    )
                    da_pre = y_pre - (latent_pre + e_pre)
                    da_post = y_post - (latent_post + e_post)
                    signed_shift = da_post - da_pre
                    reduction = np.abs(da_pre) - np.abs(da_post)
                    for delta in signal_cfg["delta_grid"]:
                        states = classify_draws(da_pre, da_post, float(delta))
                        output = pd.DataFrame({
                            "issuer_ticker": pre_valid["issuer_ticker"],
                            "raw_exchange": pre_valid["raw_exchange"],
                            "fiscal_year": year,
                            "benchmark": benchmark,
                            "rho": float(rho),
                            "error_sd_ratio": float(error_sd_ratio),
                            "delta": float(delta),
                            "ta_pre": pre_valid["ta_scaled"],
                            "ta_post": post_valid["ta_scaled"],
                            "raw_ta_shift": post_valid["ta_scaled"].to_numpy() - pre_valid["ta_scaled"].to_numpy(),
                            "da_pre_mean": da_pre.mean(axis=1),
                            "da_post_mean": da_post.mean(axis=1),
                            "signed_shift_mean": signed_shift.mean(axis=1),
                            "signed_shift_sd": signed_shift.std(axis=1, ddof=1),
                            "reduction_mean": reduction.mean(axis=1),
                            "reduction_sd": reduction.std(axis=1, ddof=1),
                            "prob_improve": (reduction > float(delta)).mean(axis=1),
                            "prob_deteriorate": (reduction < -float(delta)).mean(axis=1),
                            "snr_reduction": np.abs(reduction.mean(axis=1)) / np.maximum(reduction.std(axis=1, ddof=1), 1e-12),
                        })
                        for state_name, state_draws in states.items():
                            output[f"prob_{state_name}"] = state_draws.mean(axis=1)
                        results.append(output)

            fold_row = {
                "fiscal_year": year,
                "benchmark": benchmark,
                "test_pairs": rows,
            }
            fold_row.update({f"weight_{name}": float(weight) for name, weight in zip(weight_names, weights)})
            fold_rows.append(fold_row)

    if not results:
        raise ValueError("No rolling test folds produced posterior results")
    return pd.concat(results, ignore_index=True), pd.DataFrame(fold_rows)
