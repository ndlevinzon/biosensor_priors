"""Physics mean model μ₀ fitted on train; residuals go to the GP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class PhysicsMeanModel:
    """
    Fit μ₀(x) on training data.

    If physics features are present, fit a linear combination of them.
    Otherwise fall back to an intercept-only mean (constant μ₀).
    """

    ridge_alpha: float = 1.0
    use_ridge: bool = True
    coefficients_: np.ndarray | None = None
    intercept_: float = 0.0
    n_features_: int = 0
    mode_: str = "intercept"

    def fit(self, X_physics: np.ndarray, y: np.ndarray) -> PhysicsMeanModel:
        y = np.asarray(y, dtype=float)
        X_physics = np.asarray(X_physics, dtype=float)
        self.n_features_ = X_physics.shape[1] if X_physics.ndim == 2 else 0

        if self.n_features_ == 0:
            self.mode_ = "intercept"
            self.intercept_ = float(np.mean(y))
            self.coefficients_ = np.zeros(0, dtype=float)
            return self

        model: LinearRegression | Ridge
        if self.use_ridge:
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        else:
            model = LinearRegression(fit_intercept=True)
        model.fit(X_physics, y)
        self.mode_ = "physics_linear"
        self.coefficients_ = np.asarray(model.coef_, dtype=float)
        self.intercept_ = float(model.intercept_)
        return self

    def predict(self, X_physics: np.ndarray) -> np.ndarray:
        X_physics = np.asarray(X_physics, dtype=float)
        if self.mode_ == "intercept" or self.n_features_ == 0:
            return np.full(len(X_physics), self.intercept_, dtype=float)
        return self.intercept_ + X_physics @ self.coefficients_

    def as_weight_dict(self, names: list[str] | None = None) -> dict[str, float | str]:
        out: dict[str, float | str] = {"intercept": self.intercept_, "mode": self.mode_}
        if self.coefficients_ is None or len(self.coefficients_) == 0:
            return out
        names = names or [f"w{i}" for i in range(len(self.coefficients_))]
        for name, coef in zip(names, self.coefficients_, strict=False):
            out[str(name)] = float(coef)
        return out
