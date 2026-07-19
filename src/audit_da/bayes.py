from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


@dataclass
class PosteriorPrediction:
    mean: np.ndarray
    sd: np.ndarray
    draws: np.ndarray | None = None


class ApproxHierarchicalBayes:
    """Fast approximate Bayesian regression with a Normal-Normal firm intercept.

    BayesianRidge supplies the fixed-effect posterior. A second Normal-Normal layer
    partially pools firm residual means. The same fixed-effect and firm draws can be
    reused for paired pre/post predictions.
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = BayesianRidge(fit_intercept=False, compute_score=True)
        self.feature_names: list[str] = []
        self.firm_posterior: dict[str, tuple[float, float]] = {}
        self.random_intercept_var = 0.0
        self.residual_var = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, firm_ids: np.ndarray, feature_names: list[str]) -> "ApproxHierarchicalBayes":
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = X[mask]
        y = y[mask]
        firm_ids = np.asarray(firm_ids, dtype=str)[mask]
        if len(y) < max(30, X.shape[1] * 5):
            raise ValueError(f"Insufficient rows for Bayesian fit: {len(y)}")
        Xs = self.scaler.fit_transform(X)
        design = np.column_stack([np.ones(len(Xs)), Xs])
        self.feature_names = ["intercept"] + list(feature_names)
        self.model.fit(design, y)
        fitted = design @ self.model.coef_
        residual = y - fitted
        self.residual_var = max(float(1.0 / self.model.alpha_), 1e-10)

        unique, inverse = np.unique(firm_ids, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=residual)
        means = sums / np.maximum(counts, 1)
        sampling_var = self.residual_var / np.maximum(counts, 1)
        between = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
        self.random_intercept_var = max(between - float(np.mean(sampling_var)), 1e-10)
        for idx, firm in enumerate(unique):
            posterior_var = 1.0 / (1.0 / self.random_intercept_var + counts[idx] / self.residual_var)
            posterior_mean = posterior_var * sums[idx] / self.residual_var
            self.firm_posterior[str(firm)] = (float(posterior_mean), float(posterior_var))
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.column_stack([np.ones(len(Xs)), Xs])

    def posterior_mean_sd(self, X: np.ndarray, firm_ids: np.ndarray, include_residual: bool = True) -> PosteriorPrediction:
        design = self._design(X)
        fixed_mean = design @ self.model.coef_
        fixed_var = np.einsum("ij,jk,ik->i", design, self.model.sigma_, design)
        firm_mean = np.zeros(len(X))
        firm_var = np.full(len(X), self.random_intercept_var)
        for idx, firm in enumerate(np.asarray(firm_ids, dtype=str)):
            if firm in self.firm_posterior:
                firm_mean[idx], firm_var[idx] = self.firm_posterior[firm]
        variance = fixed_var + firm_var + (self.residual_var if include_residual else 0.0)
        return PosteriorPrediction(fixed_mean + firm_mean, np.sqrt(np.maximum(variance, 1e-12)))

    def draw_components(self, n_draws: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        coef = rng.multivariate_normal(self.model.coef_, self.model.sigma_, size=n_draws)
        firm_draws: dict[str, np.ndarray] = {}
        for firm, (mean, var) in self.firm_posterior.items():
            firm_draws[firm] = rng.normal(mean, np.sqrt(var), size=n_draws)
        return coef, firm_draws

    def latent_draws(
        self,
        X: np.ndarray,
        firm_ids: np.ndarray,
        coef_draws: np.ndarray,
        firm_draws: dict[str, np.ndarray],
    ) -> np.ndarray:
        design = self._design(X)
        values = design @ coef_draws.T
        for idx, firm in enumerate(np.asarray(firm_ids, dtype=str)):
            draws = firm_draws.get(firm)
            if draws is not None:
                values[idx] += draws
        return values
