"""Preregistered multi-output phenotypes and weighted combination."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.fitness import DEFAULT_WEIGHTS, PHENOTYPES

AUX_PHENOTYPES: tuple[str, ...] = ()

SCORE_COLUMNS: dict[str, str] = {
    "selectivity": "_fitness_selectivity_score",
    "affinity": "_fitness_affinity_score",
    "fc": "_fitness_fc_score",
    "brightness": "_fitness_brightness_score",
    "fc_prop": "_fitness_fc_prop_score",
}


def phenotype_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return preregistered phenotype weights, validating the key set."""
    weight_map = dict(weights or DEFAULT_WEIGHTS)
    if set(weight_map) != set(PHENOTYPES):
        raise ValueError(
            "Fitness weights must define selectivity, affinity, fc, "
            f"brightness, and fc_prop (got {sorted(weight_map)})."
        )
    return {name: float(weight_map[name]) for name in PHENOTYPES}


def phenotype_score_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract per-phenotype [0, 1] score columns (NaN where unmeasured)."""
    out: dict[str, np.ndarray] = {}
    for name, col in SCORE_COLUMNS.items():
        if col in df.columns:
            out[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        else:
            out[name] = np.full(len(df), np.nan, dtype=float)
    return out


def combine_phenotype_means(
    means: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
    available: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Weighted combination with missing-weight redistribution.

    Parameters
    ----------
    means : mapping
        Per-phenotype predictive means, each shape ``(n,)``.
    weights : mapping, optional
        Preregistered weights (default Stage-0 fitness weights).
    available : mapping, optional
        Boolean masks; default uses finite means.

    Returns
    -------
    numpy.ndarray
        Combined score per row. NaN when no component is available.
    """
    w = phenotype_weights(weights)
    names = list(PHENOTYPES)
    n = len(next(iter(means.values()))) if means else 0
    numer = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float)
    for name in names:
        mu = np.asarray(means.get(name, np.full(n, np.nan)), dtype=float)
        if available is not None and name in available:
            mask = np.asarray(available[name], dtype=bool) & np.isfinite(mu)
        else:
            mask = np.isfinite(mu)
        numer[mask] += w[name] * mu[mask]
        denom[mask] += w[name]
    out = np.full(n, np.nan, dtype=float)
    ok = denom > 0
    out[ok] = numer[ok] / denom[ok]
    return out


def combine_phenotype_std(
    stds: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
    available: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Independent-Gaussian std of the redistributed weighted combination."""
    w = phenotype_weights(weights)
    names = list(PHENOTYPES)
    n = len(next(iter(stds.values()))) if stds else 0
    var = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float)
    for name in names:
        sig = np.asarray(stds.get(name, np.full(n, np.nan)), dtype=float)
        if available is not None and name in available:
            mask = np.asarray(available[name], dtype=bool) & np.isfinite(sig)
        else:
            mask = np.isfinite(sig)
        var[mask] += (w[name] ** 2) * (sig[mask] ** 2)
        denom[mask] += w[name]
    out = np.full(n, np.nan, dtype=float)
    ok = denom > 0
    out[ok] = np.sqrt(np.maximum(var[ok], 0.0)) / denom[ok]
    return np.nan_to_num(out, nan=1e-6)


def labeled_mask(scores: np.ndarray, *, min_n: int = 3) -> np.ndarray | None:
    """Return a boolean train mask, or None when too few labels exist."""
    mask = np.isfinite(np.asarray(scores, dtype=float))
    if int(mask.sum()) < min_n:
        return None
    return mask


def minmax_from_train(
    values: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Map combined raw scores onto the train fitness range."""
    values = np.asarray(values, dtype=float)
    span = hi - lo
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(span) < 1e-12:
        return values
    return (values - lo) / span


def constraint_probability(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    minimum: float,
) -> np.ndarray:
    """P(y >= minimum) under an independent Gaussian posterior."""
    from scipy.stats import norm

    mu = np.asarray(mean, dtype=float)
    sig = np.maximum(np.asarray(std, dtype=float), 1e-8)
    return 1.0 - norm.cdf(minimum, loc=mu, scale=sig)

