"""Version / scaffold intercept so the GP residual is a mutation effect."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GroupIntercept:
    """Additive group mean on residuals after μ₀.

    Fit on training groups only. Unseen groups (new versions, LOCO of a
    singleton family) receive 0 so the intercept does not leak held-out
    construct identity. Design-space candidates inherit the parent
    version intercept.
    """

    column: str = "version"
    intercepts_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, residual: np.ndarray) -> GroupIntercept:
        """Estimate per-group mean residual.

        Parameters
        ----------
        df : pandas.DataFrame
            Training rows containing ``column``.
        residual : numpy.ndarray
            Residuals after physics mean (and optional α shrinkage).

        Returns
        -------
        GroupIntercept
            Fitted intercept model (``self``).
        """
        residual = np.asarray(residual, dtype=float)
        self.intercepts_ = {}
        if self.column not in df.columns or len(df) == 0:
            return self
        groups = df[self.column].astype(str).to_numpy()
        for g in pd.unique(groups):
            mask = groups == str(g)
            vals = residual[mask]
            if np.isfinite(vals).any():
                self.intercepts_[str(g)] = float(np.nanmean(vals))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return intercepts aligned with ``df`` rows (0 if group unseen)."""
        if not self.intercepts_ or self.column not in df.columns:
            return np.zeros(len(df), dtype=float)
        return np.asarray(
            [self.intercepts_.get(str(g), 0.0) for g in df[self.column]],
            dtype=float,
        )
