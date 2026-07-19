from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


def stacking_weights(y: np.ndarray, means: list[np.ndarray], sds: list[np.ndarray]) -> np.ndarray:
    model_count = len(means)
    if model_count == 1:
        return np.array([1.0])
    log_density = np.column_stack([
        norm.logpdf(y, loc=mean, scale=np.maximum(sd, 1e-8))
        for mean, sd in zip(means, sds)
    ])

    def objective(raw: np.ndarray) -> float:
        weights = np.exp(raw - logsumexp(raw))
        return float(-np.sum(logsumexp(log_density + np.log(weights), axis=1)))

    result = minimize(objective, np.zeros(model_count), method="BFGS")
    raw = result.x if result.success else np.zeros(model_count)
    return np.exp(raw - logsumexp(raw))
