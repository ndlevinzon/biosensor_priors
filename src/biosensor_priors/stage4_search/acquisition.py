"""Uncertainty-aware acquisition using effective σ (GP + structure + physics)."""

from __future__ import annotations

import numpy as np


def sigma_effective(
    sigma_gp: np.ndarray,
    sigma_structure: np.ndarray | float = 0.0,
    sigma_physics: np.ndarray | float = 0.0,
    *,
    lambda_structure: float = 1.0,
    lambda_physics: float = 1.0,
) -> np.ndarray:
    """σ_eff² = σ_GP² + λ_s σ_structure² + λ_p σ_physics²."""
    sg = np.asarray(sigma_gp, dtype=float)
    ss = np.asarray(sigma_structure, dtype=float)
    sp = np.asarray(sigma_physics, dtype=float)
    var = sg**2 + lambda_structure * ss**2 + lambda_physics * sp**2
    return np.sqrt(np.maximum(var, 1e-24))


def ucb(
    mu: np.ndarray,
    sigma_eff: np.ndarray,
    *,
    kappa: float = 1.0,
) -> np.ndarray:
    """UCB(x) = μ(x) + κ σ_eff(x)."""
    return np.asarray(mu, dtype=float) + float(kappa) * np.asarray(sigma_eff, dtype=float)
