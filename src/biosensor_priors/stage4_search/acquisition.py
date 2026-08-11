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
    """Combine GP, structural, and physics uncertainties into effective sigma.

    Computes ``σ_eff² = σ_GP² + λ_s σ_structure² + λ_p σ_physics²`` and returns
    the square root, floored at a small positive value for numerical stability.

    Parameters
    ----------
    sigma_gp : np.ndarray
        GP posterior standard deviation per candidate.
    sigma_structure : np.ndarray or float, optional
        Structural uncertainty per candidate (default 0.0).
    sigma_physics : np.ndarray or float, optional
        Physics-score uncertainty per candidate (default 0.0).
    lambda_structure : float, optional
        Weight on structural variance (default 1.0).
    lambda_physics : float, optional
        Weight on physics variance (default 1.0).

    Returns
    -------
    np.ndarray
        Effective standard deviation ``σ_eff`` with the same broadcast shape as
        the input arrays.
    """
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
    """Upper confidence bound acquisition score.

    Parameters
    ----------
    mu : np.ndarray
        Predictive mean fitness per candidate.
    sigma_eff : np.ndarray
        Effective uncertainty per candidate (see :func:`sigma_effective`).
    kappa : float, optional
        Exploration weight (default 1.0).

    Returns
    -------
    np.ndarray
        Acquisition scores ``μ + κ σ_eff``.
    """
    return np.asarray(mu, dtype=float) + float(kappa) * np.asarray(sigma_eff, dtype=float)
