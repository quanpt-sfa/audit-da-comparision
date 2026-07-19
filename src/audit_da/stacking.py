from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class StackingResult:
    weights: np.ndarray
    success: bool
    message: str
    objective: float
    equal_weight_objective: float
    best_single_objective: float
    finite_rows: int
    weight_entropy: float
    effective_model_count: float


def _objective_from_weights(weights: np.ndarray, log_density: np.ndarray) -> float:
    safe = np.clip(np.asarray(weights, float), 1e-12, 1.0)
    safe = safe / safe.sum()
    return float(-np.sum(logsumexp(log_density + np.log(safe), axis=1)))


def solve_stacking(y: np.ndarray, means: list[np.ndarray], sds: list[np.ndarray]) -> StackingResult:
    model_count = len(means)
    if model_count == 0:
        raise ValueError("At least one model is required")
    arrays = [np.asarray(y, float)] + [np.asarray(x, float) for x in means] + [np.asarray(x, float) for x in sds]
    length = len(arrays[0])
    if any(len(x) != length for x in arrays):
        raise ValueError("Stacking inputs must have the same length")
    finite = np.isfinite(arrays[0])
    for mean, sd in zip(means, sds):
        finite &= np.isfinite(mean) & np.isfinite(sd) & (np.asarray(sd) > 0)
    if not finite.any():
        weights = np.repeat(1.0 / model_count, model_count)
        return StackingResult(weights, False, "No finite validation rows", np.nan, np.nan, np.nan, 0,
                              float(-np.sum(weights * np.log(weights))), float(model_count))
    y_f = np.asarray(y, float)[finite]
    log_density = np.column_stack([
        norm.logpdf(y_f, loc=np.asarray(mean, float)[finite], scale=np.maximum(np.asarray(sd, float)[finite], 1e-8))
        for mean, sd in zip(means, sds)
    ])
    equal = np.repeat(1.0 / model_count, model_count)
    equal_obj = _objective_from_weights(equal, log_density)
    single_objs = np.array([_objective_from_weights(np.eye(model_count)[j], log_density) for j in range(model_count)])
    best_single = float(single_objs.min())
    if model_count == 1:
        return StackingResult(np.array([1.0]), True, "Single model", best_single, best_single, best_single,
                              int(finite.sum()), 0.0, 1.0)

    result = minimize(
        lambda w: _objective_from_weights(w, log_density),
        x0=equal,
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * model_count,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if result.success and np.all(np.isfinite(result.x)):
        weights = np.clip(result.x, 0.0, 1.0)
        weights = weights / weights.sum()
        objective = _objective_from_weights(weights, log_density)
        success = True
        message = str(result.message)
    else:
        # A failed optimizer should not silently masquerade as estimated equal weights.
        # Fall back to the best single predictive model and expose the failure.
        best = int(np.argmin(single_objs))
        weights = np.eye(model_count)[best]
        objective = float(single_objs[best])
        success = False
        message = f"Optimizer failed; best-single fallback: {result.message}"
    positive = weights[weights > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    effective = float(np.exp(entropy))
    return StackingResult(weights, success, message, objective, equal_obj, best_single,
                          int(finite.sum()), entropy, effective)


def stacking_weights(y: np.ndarray, means: list[np.ndarray], sds: list[np.ndarray]) -> np.ndarray:
    """Backward-compatible weights-only interface."""
    return solve_stacking(y, means, sds).weights
