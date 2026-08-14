"""GP residual learner over sequence/chemistry features (physics in μ₀ only)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from biosensor_priors.stage3_surrogate.kernels import HammingPlusMaternKernel

KernelKind = Literal["hamming", "matern52"]


@dataclass
class GPResidualModel:
    """Zero-mean GP fit to residuals y − α μ₀(x) − intercept(version)."""

    random_state: int = 42
    n_restarts_optimizer: int = 1
    noise_level: float = 0.03
    kernel: KernelKind = "hamming"
    n_hamming: int = 0
    model_: GaussianProcessRegressor | None = None

    def _make_kernel(self, n_features: int):
        """Build residual kernel (Hamming mutation-set or ARD Matérn).

        Parameters
        ----------
        n_features : int
            Number of GP input dimensions (physics columns already stripped).

        Returns
        -------
        sklearn.gaussian_process.kernels.Kernel
            Composite kernel for :class:`GaussianProcessRegressor`.
        """
        white = WhiteKernel(
            noise_level=self.noise_level, noise_level_bounds=(1e-5, 0.5)
        )
        if self.kernel == "hamming":
            n_h = min(int(self.n_hamming), n_features)
            return HammingPlusMaternKernel(n_hamming=n_h) + white
        length_scale = np.ones(max(n_features, 1), dtype=float)
        return (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(
                length_scale=length_scale,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + white
        )

    def fit(self, X: np.ndarray, residual: np.ndarray) -> GPResidualModel:
        """Fit GP to residuals y − μ₀(x).

        Parameters
        ----------
        X : numpy.ndarray
            Non-physics feature matrix used for residual learning.
        residual : numpy.ndarray
            Target residuals after subtracting physics mean and intercepts.

        Returns
        -------
        GPResidualModel
            Fitted GP model (``self``).
        """
        X = np.asarray(X, dtype=float)
        residual = np.asarray(residual, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            X = np.zeros((len(residual), 1), dtype=float)
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
        """Predict GP residual mean and standard deviation.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix for prediction.

        Returns
        -------
        mean : numpy.ndarray
            Predicted residual mean per row.
        std : numpy.ndarray
            Predicted residual standard deviation per row.

        Raises
        ------
        RuntimeError
            When called before :meth:`fit`.
        """
        if self.model_ is None:
            raise RuntimeError("GPResidualModel must be fit before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            X = np.zeros((len(X), 1), dtype=float)
        mean, std = self.model_.predict(X, return_std=True)
        return np.asarray(mean, dtype=float), np.maximum(
            np.asarray(std, dtype=float), 1e-12
        )

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one posterior sample of the residual at ``X``."""
        mean, std = self.predict(X)
        return rng.normal(mean, std)
