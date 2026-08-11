"""Discount physics features by structural confidence; keep raw and weighted."""

from __future__ import annotations

import numpy as np

from biosensor_priors.stage3_surrogate.features import PHYSICS_FEATURE_COLUMNS


def apply_confidence_weighting(
    X: np.ndarray,
    feature_names: list[str],
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and confidence-weighted feature matrices.

    Physics columns in the weighted copy are multiplied by per-row
    structural confidence.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    feature_names : list of str
        Column names aligned with ``X``.
    confidence : numpy.ndarray
        Per-row structural confidence in ``[0, 1]``.

    Returns
    -------
    X_raw : numpy.ndarray
        Unmodified copy of ``X``.
    X_weighted : numpy.ndarray
        Copy with physics columns scaled by ``confidence``.
    """
    X_raw = np.asarray(X, dtype=float).copy()
    X_w = X_raw.copy()
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    physics_idx = [i for i, n in enumerate(feature_names) if n in PHYSICS_FEATURE_COLUMNS]
    for i in physics_idx:
        X_w[:, i] = X_w[:, i] * conf
    return X_raw, X_w
