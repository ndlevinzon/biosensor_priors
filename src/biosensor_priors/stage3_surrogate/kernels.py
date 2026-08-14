"""Custom sklearn GP kernels: Hamming mutation-set + small physchem Matérn."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel


def _hamming_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise mutation-set Hamming: ``|S Δ T|`` for 0/1 indicator rows."""
    # Broadcasting: (n, 1, d) vs (1, m, d)
    return np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)


def _matern52(dist: np.ndarray) -> np.ndarray:
    """Isotropic Matérn-5/2 on pre-scaled Euclidean distances."""
    s5 = np.sqrt(5.0)
    return (1.0 + s5 * dist + 5.0 / 3.0 * dist**2) * np.exp(-s5 * dist)


class HammingPlusMaternKernel(Kernel):
    """Additive ``σ_h² Hamming + σ_m² Matérn-5/2`` on column blocks.

    The first ``n_hamming`` columns are 0/1 mutation indicators. Hamming
    distance is the symmetric-difference size of mutation sets. Remaining
    columns are a small physicochemical kernel (physics stays in μ₀).
    """

    def __init__(
        self,
        n_hamming: int = 0,
        hamming_length_scale: float = 2.0,
        hamming_length_scale_bounds: tuple[float, float] = (0.2, 50.0),
        hamming_variance: float = 1.0,
        hamming_variance_bounds: tuple[float, float] = (1e-2, 1e2),
        matern_length_scale: float = 1.0,
        matern_length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
        matern_variance: float = 0.3,
        matern_variance_bounds: tuple[float, float] = (1e-3, 10.0),
    ) -> None:
        self.n_hamming = int(n_hamming)
        self.hamming_length_scale = hamming_length_scale
        self.hamming_length_scale_bounds = hamming_length_scale_bounds
        self.hamming_variance = hamming_variance
        self.hamming_variance_bounds = hamming_variance_bounds
        self.matern_length_scale = matern_length_scale
        self.matern_length_scale_bounds = matern_length_scale_bounds
        self.matern_variance = matern_variance
        self.matern_variance_bounds = matern_variance_bounds

    @property
    def hyperparameter_hamming_length_scale(self) -> Hyperparameter:
        return Hyperparameter(
            "hamming_length_scale", "numeric", self.hamming_length_scale_bounds
        )

    @property
    def hyperparameter_hamming_variance(self) -> Hyperparameter:
        return Hyperparameter(
            "hamming_variance", "numeric", self.hamming_variance_bounds
        )

    @property
    def hyperparameter_matern_length_scale(self) -> Hyperparameter:
        return Hyperparameter(
            "matern_length_scale", "numeric", self.matern_length_scale_bounds
        )

    @property
    def hyperparameter_matern_variance(self) -> Hyperparameter:
        return Hyperparameter(
            "matern_variance", "numeric", self.matern_variance_bounds
        )

    def is_stationary(self) -> bool:
        return True

    def diag(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        var = 0.0
        if self.n_hamming > 0:
            var += float(self.hamming_variance)
        if np.asarray(X).shape[1] > self.n_hamming:
            var += float(self.matern_variance)
        if var <= 0:
            var = 1.0
        return np.full(n, var, dtype=float)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
    ):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y_is_X = Y is None
        Y = X if Y_is_X else np.atleast_2d(np.asarray(Y, dtype=float))
        n_h = min(int(self.n_hamming), X.shape[1])
        k_h = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
        k_m = np.zeros_like(k_h)
        d_h = None
        d_m = None
        ell_h = max(float(np.squeeze(self.hamming_length_scale)), 1e-8)
        ell_m = max(float(np.squeeze(self.matern_length_scale)), 1e-8)
        v_h = float(np.squeeze(self.hamming_variance))
        v_m = float(np.squeeze(self.matern_variance))

        if n_h > 0:
            d_h = _hamming_counts(X[:, :n_h], Y[:, :n_h])
            k_h = v_h * np.exp(-d_h / ell_h)
        if X.shape[1] > n_h:
            x_m = X[:, n_h:]
            y_m = Y[:, n_h:]
            delta = x_m[:, None, :] - y_m[None, :, :]
            d_m = np.sqrt(np.sum(delta * delta, axis=2)) / ell_m
            k_m = v_m * _matern52(d_m)

        K = k_h + k_m
        if not eval_gradient:
            return K

        n_hp = 4
        grad = np.zeros((X.shape[0], Y.shape[0], n_hp), dtype=float)
        if n_h > 0 and d_h is not None:
            # dK/d ℓ_h = v_h * exp(-d/ℓ) * d / ℓ²
            grad[:, :, 0] = k_h * d_h / (ell_h**2)
            # dK/d v_h = exp(-d/ℓ)
            grad[:, :, 1] = k_h / max(v_h, 1e-12)
        if d_m is not None:
            # Matern52 wrt length_scale: K * (5/3 r² (1+√5 r)) / ℓ  ... 
            # r = d_eucl / ℓ; ∂k/∂ℓ = v * ∂matern52(r)/∂ℓ
            s5 = np.sqrt(5.0)
            r = d_m
            matern = _matern52(r)
            # dm/dr = - (5/3) r (1 + √5 r) exp(-√5 r)
            dm_dr = -(5.0 / 3.0) * r * (1.0 + s5 * r) * np.exp(-s5 * r)
            dr_dell = -r / ell_m
            grad[:, :, 2] = v_m * dm_dr * dr_dell
            grad[:, :, 3] = matern
        return K, grad
