"""Physics mean model μ₀ fitted on train; residuals go to the GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, RidgeCV

ShrinkageKind = Literal["ridge_cv", "ridge", "horseshoe", "ols", "bayesian_ridge"]


def _horseshoe_map(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_iter: int = 80,
) -> tuple[np.ndarray, float]:
    """EM-style horseshoe MAP for a linear mean (Carvalho–Polson–Scott).

    Local scales λ_j and global τ shrink physics weights that the data do
    not support, so μ₀ can down-weight itself.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    intercept = float(np.mean(y))
    yc = y - intercept
    xc = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(xc, axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    xs = xc / scale
    tau = max((p / max(n, 1)) ** 0.5 / max(n, 1) ** 0.5, 1e-3)
    lam = np.ones(p, dtype=float)
    beta_std = np.zeros(p, dtype=float)
    sigma2 = float(np.var(yc) or 1.0)
    for _ in range(n_iter):
        d_inv = 1.0 / np.maximum((tau * lam) ** 2, 1e-12)
        gram = xs.T @ xs + sigma2 * np.diag(d_inv)
        try:
            beta_std = np.linalg.solve(gram, xs.T @ yc)
        except np.linalg.LinAlgError:
            beta_std = np.linalg.lstsq(gram, xs.T @ yc, rcond=None)[0]
        resid = yc - xs @ beta_std
        sigma2 = float(np.dot(resid, resid) / max(n, 1) + 1e-8)
        # Half-Cauchy mixing: λ_j² ≈ |β_j| / τ  (MAP-style update)
        lam = np.sqrt(np.abs(beta_std) / max(tau, 1e-8) + 1e-6)
        lam = np.clip(lam, 1e-3, 1e3)
        tau = float(np.sqrt(np.mean(beta_std**2 / np.maximum(lam**2, 1e-12)) + 1e-8))
        tau = float(np.clip(tau, 1e-4, 10.0))
    coef = beta_std / scale
    return coef, intercept


@dataclass
class PhysicsMeanModel:
    """
    Fit μ₀(x) on training data.

    If physics features are present, fit a shrunk linear combination of them
    (RidgeCV by default, optional horseshoe). Otherwise fall back to an
    intercept-only mean (constant μ₀).
    """

    ridge_alpha: float = 1.0
    use_ridge: bool = True
    shrinkage: ShrinkageKind = "ridge_cv"
    coefficients_: np.ndarray | None = None
    intercept_: float = 0.0
    n_features_: int = 0
    mode_: str = "intercept"

    def fit(self, X_physics: np.ndarray, y: np.ndarray) -> PhysicsMeanModel:
        """Fit μ₀(x) on physics features or intercept-only when absent.

        Parameters
        ----------
        X_physics : numpy.ndarray
            Physics feature matrix of shape ``(n_samples, n_physics_features)``.
        y : numpy.ndarray
            Target fitness values.

        Returns
        -------
        PhysicsMeanModel
            Fitted mean model (``self``).
        """
        y = np.asarray(y, dtype=float)
        X_physics = np.asarray(X_physics, dtype=float)
        self.n_features_ = X_physics.shape[1] if X_physics.ndim == 2 else 0

        if self.n_features_ == 0:
            self.mode_ = "intercept"
            self.intercept_ = float(np.mean(y))
            self.coefficients_ = np.zeros(0, dtype=float)
            return self

        finite = np.isfinite(y)
        if not np.any(finite):
            self.mode_ = "intercept"
            self.intercept_ = 0.0
            self.coefficients_ = np.zeros(self.n_features_, dtype=float)
            return self
        X_fit = X_physics[finite]
        y_fit = y[finite]
        kind = self.shrinkage
        if kind == "ols" or (kind == "ridge" and not self.use_ridge):
            model = LinearRegression(fit_intercept=True)
            model.fit(X_fit, y_fit)
            self.coefficients_ = np.asarray(model.coef_, dtype=float)
            self.intercept_ = float(model.intercept_)
            self.mode_ = "physics_ols"
        elif kind == "horseshoe":
            coef, intercept = _horseshoe_map(X_fit, y_fit)
            self.coefficients_ = coef
            self.intercept_ = intercept
            self.mode_ = "physics_horseshoe"
        elif kind == "bayesian_ridge":
            model = BayesianRidge(fit_intercept=True)
            model.fit(X_fit, y_fit)
            self.coefficients_ = np.asarray(model.coef_, dtype=float)
            self.intercept_ = float(model.intercept_)
            self.mode_ = "physics_bayesian_ridge"
        elif kind == "ridge":
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            model.fit(X_fit, y_fit)
            self.coefficients_ = np.asarray(model.coef_, dtype=float)
            self.intercept_ = float(model.intercept_)
            self.mode_ = "physics_ridge"
        else:
            alphas = np.logspace(-3, 3, 16)
            model = RidgeCV(alphas=alphas, fit_intercept=True)
            model.fit(X_fit, y_fit)
            self.coefficients_ = np.asarray(model.coef_, dtype=float)
            self.intercept_ = float(model.intercept_)
            self.mode_ = "physics_ridge_cv"
        return self

    def predict(self, X_physics: np.ndarray) -> np.ndarray:
        """Predict physics mean μ₀ for each row.

        Parameters
        ----------
        X_physics : numpy.ndarray
            Physics feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted mean values, one per row.
        """
        X_physics = np.asarray(X_physics, dtype=float)
        if self.mode_ == "intercept" or self.n_features_ == 0:
            return np.full(len(X_physics), self.intercept_, dtype=float)
        return self.intercept_ + X_physics @ self.coefficients_

    def as_weight_dict(self, names: list[str] | None = None) -> dict[str, float | str]:
        """Serialize fitted coefficients as a name-to-weight mapping.

        Parameters
        ----------
        names : list of str, optional
            Feature names for coefficients; defaults to ``w0``, ``w1``, ...

        Returns
        -------
        dict
            Intercept, mode, and optional per-feature weights.
        """
        out: dict[str, float | str] = {"intercept": self.intercept_, "mode": self.mode_}
        if self.coefficients_ is None or len(self.coefficients_) == 0:
            return out
        names = names or [f"w{i}" for i in range(len(self.coefficients_))]
        for name, coef in zip(names, self.coefficients_, strict=False):
            out[str(name)] = float(coef)
        return out
