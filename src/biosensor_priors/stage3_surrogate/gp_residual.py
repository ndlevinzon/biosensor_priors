"""GP residual learner over sequence/chemistry/physics features."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class GPResidualModel:
    """Zero-mean GP fit to residuals y − μ₀(x)."""

    random_state: int = 42
    n_restarts_optimizer: int = 2
    noise_level: float = 0.03
    model_: GaussianProcessRegressor | None = None

    def _make_kernel(self, n_features: int):
        length_scale = np.ones(n_features, dtype=float)
        return (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-5, 0.5))
        )

    def fit(self, X: np.ndarray, residual: np.ndarray) -> GPResidualModel:
        X = np.asarray(X, dtype=float)
        residual = np.asarray(residual, dtype=float)
        kernel = self._make_kernel(X.shape[1])
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X, residual)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("GPResidualModel must be fit before predict.")
        mean, std = self.model_.predict(np.asarray(X, dtype=float), return_std=True)
        return np.asarray(mean, dtype=float), np.maximum(np.asarray(std, dtype=float), 1e-12)
