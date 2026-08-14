"""Calibrate GP + structure + physics uncertainties (λ and CV+ / split conformal)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _sigma_effective(
    sigma_gp: np.ndarray,
    sigma_structure: np.ndarray | float = 0.0,
    sigma_physics: np.ndarray | float = 0.0,
    *,
    lambda_structure: float = 1.0,
    lambda_physics: float = 1.0,
) -> np.ndarray:
    """Local copy of Stage-4 σ_eff to avoid a Stage-3 ↔ Stage-4 import cycle."""
    sg = np.asarray(sigma_gp, dtype=float)
    ss = np.asarray(sigma_structure, dtype=float)
    sp = np.asarray(sigma_physics, dtype=float)
    var = sg**2 + lambda_structure * ss**2 + lambda_physics * sp**2
    return np.sqrt(np.maximum(var, 1e-24))

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass
class UncertaintyCalibrator:
    """LOCO-fitted λ's and a conformal residual quantile.

    Acquisition should use ``sigma_calibrated`` rather than raw GP σ.
    """

    lambda_structure: float = 1.0
    lambda_physics: float = 1.0
    conformal_quantile: float = 1.0
    target_coverage: float = 0.90
    empirical_coverage: float = float("nan")
    n: int = 0
    z_nominal: float = 1.6448536269514722  # Φ⁻¹(0.95) for 90% two-sided

    def sigma_calibrated(
        self,
        sigma_gp: np.ndarray,
        sigma_structure: np.ndarray | float = 0.0,
        sigma_physics: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Return ``q · σ_eff(λ)``."""
        sig = _sigma_effective(
            sigma_gp,
            sigma_structure,
            sigma_physics,
            lambda_structure=self.lambda_structure,
            lambda_physics=self.lambda_physics,
        )
        return np.asarray(self.conformal_quantile, dtype=float) * sig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def structural_and_physics_sigma(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract σ_structure and σ_physics columns from a candidate table."""
    n = len(df)
    if "structural_confidence" in df.columns:
        conf = pd.to_numeric(df["structural_confidence"], errors="coerce").fillna(1.0)
        ss = (1.0 - conf).to_numpy(dtype=float)
    else:
        ss = np.zeros(n, dtype=float)
    if "physics_score_std" in df.columns:
        sp = (
            pd.to_numeric(df["physics_score_std"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        sp = np.zeros(n, dtype=float)
    return ss, sp


def conformal_quantile(scores: np.ndarray, *, alpha: float = 0.10) -> float:
    """Split-conformal / CV+ residual quantile (exchangeable finite-sample).

    Uses ``ceil((n+1)(1-α))/n`` as in Romano, Patterson, Candès.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _coverage(
    abs_err: np.ndarray,
    sigma: np.ndarray,
    z: float,
) -> float:
    ok = np.isfinite(abs_err) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(abs_err[ok] <= z * sigma[ok]))


def fit_uncertainty_calibration(
    cv_predictions: pd.DataFrame,
    *,
    target_coverage: float = 0.90,
    lambda_grid: Iterable[float] = LAMBDA_GRID,
) -> UncertaintyCalibrator:
    """Fit λ_s, λ_p for nominal coverage, then a CV+ conformal quantile.

    Parameters
    ----------
    cv_predictions : pandas.DataFrame
        Rows with ``y_true``, ``fitness_mean``, ``fitness_std``, and optional
        ``sigma_structure``, ``sigma_physics``.
    target_coverage : float, optional
        Desired two-sided coverage (default 0.90).
    lambda_grid : iterable of float, optional
        Candidate λ values shared by structure and physics.

    Returns
    -------
    UncertaintyCalibrator
        Fitted calibrator.
    """
    y = pd.to_numeric(cv_predictions["y_true"], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(cv_predictions["fitness_mean"], errors="coerce").to_numpy(
        dtype=float
    )
    sg = pd.to_numeric(cv_predictions["fitness_std"], errors="coerce").to_numpy(
        dtype=float
    )
    if "sigma_structure" in cv_predictions.columns:
        ss = pd.to_numeric(cv_predictions["sigma_structure"], errors="coerce").fillna(
            0.0
        ).to_numpy(dtype=float)
    else:
        ss = np.zeros(len(cv_predictions), dtype=float)
    if "sigma_physics" in cv_predictions.columns:
        sp = pd.to_numeric(cv_predictions["sigma_physics"], errors="coerce").fillna(
            0.0
        ).to_numpy(dtype=float)
    else:
        sp = np.zeros(len(cv_predictions), dtype=float)

    abs_err = np.abs(y - mu)
    finite = np.isfinite(abs_err) & np.isfinite(sg)
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - target_coverage) / 2.0))

    best_ls, best_lp = 0.0, 0.0
    best_gap = np.inf
    grid = list(lambda_grid)
    for ls in grid:
        for lp in grid:
            sig = _sigma_effective(
                sg, ss, sp, lambda_structure=ls, lambda_physics=lp
            )
            cov = _coverage(abs_err[finite], sig[finite], z)
            if not np.isfinite(cov):
                continue
            gap = abs(cov - target_coverage)
            if gap < best_gap:
                best_gap = gap
                best_ls, best_lp = float(ls), float(lp)

    sig = _sigma_effective(
        sg, ss, sp, lambda_structure=best_ls, lambda_physics=best_lp
    )
    scores = abs_err / np.maximum(sig, 1e-8)
    q = conformal_quantile(scores[finite], alpha=1.0 - target_coverage)
    cal_sig = q * sig
    emp = _coverage(abs_err[finite], cal_sig[finite], 1.0)
    return UncertaintyCalibrator(
        lambda_structure=best_ls,
        lambda_physics=best_lp,
        conformal_quantile=q,
        target_coverage=target_coverage,
        empirical_coverage=emp,
        n=int(finite.sum()),
        z_nominal=z,
    )
