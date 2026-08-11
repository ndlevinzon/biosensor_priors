"""Compare frozen predictions to new wet-lab observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Compute Pearson or Spearman correlation with safe fallbacks.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.
    y_pred : np.ndarray
        Predicted values.
    method : str
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    float
        Correlation coefficient, or NaN when undefined (fewer than 3 finite pairs
        or zero variance).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    if np.nanstd(yt) == 0 or np.nanstd(yp) == 0:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(yt, yp).statistic)
    return float(stats.spearmanr(yt, yp).statistic)


def ranking_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """Fraction of true top-k constructs recovered by predicted top-k ranking.

    Parameters
    ----------
    y_true : np.ndarray
        Observed fitness values.
    y_pred : np.ndarray
        Predicted fitness values.
    k : int, optional
        Rank cutoff (default 3).

    Returns
    -------
    float
        Intersection size divided by ``k``, or NaN when fewer than ``k`` pairs.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < k:
        return float("nan")
    true_top = set(np.argsort(yt)[-k:])
    pred_top = set(np.argsort(yp)[-k:])
    return len(true_top & pred_top) / float(k)


def interval_coverage(
    y_true: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> float:
    """Empirical coverage of prediction intervals against observations.

    Parameters
    ----------
    y_true : np.ndarray
        Observed fitness values.
    ci_low : np.ndarray
        Lower interval bounds.
    ci_high : np.ndarray
        Upper interval bounds.

    Returns
    -------
    float
        Fraction of finite observations falling inside ``[ci_low, ci_high]``,
        or NaN when no valid pairs exist.
    """
    mask = np.isfinite(y_true) & np.isfinite(ci_low) & np.isfinite(ci_high)
    if mask.sum() == 0:
        return float("nan")
    inside = (y_true[mask] >= ci_low[mask]) & (y_true[mask] <= ci_high[mask])
    return float(np.mean(inside))


def join_predictions_and_observations(
    frozen: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    id_col_frozen: str = "candidate_id",
    id_col_obs: str = "construct_id",
    fitness_col: str = "fitness",
) -> pd.DataFrame:
    """Inner-join frozen predictions with measured fitness on construct ID.

    Parameters
    ----------
    frozen : pd.DataFrame
        Immutable freeze table from Stage 5A.
    observations : pd.DataFrame
        Cleaned wet-lab results with measured fitness.
    id_col_frozen : str, optional
        Identifier column in ``frozen`` (default ``"candidate_id"``).
    id_col_obs : str, optional
        Identifier column in ``observations`` (default ``"construct_id"``).
    fitness_col : str, optional
        Measured fitness column in ``observations`` (default ``"fitness"``).

    Returns
    -------
    pd.DataFrame
        Inner join with ``observed_fitness`` appended from observations.
    """
    left = frozen.copy()
    right = observations.copy()
    left["_join_id"] = left[id_col_frozen].astype(str)
    right["_join_id"] = right[id_col_obs].astype(str)
    merged = left.merge(
        right[["_join_id", fitness_col]].rename(columns={fitness_col: "observed_fitness"}),
        on="_join_id",
        how="inner",
    )
    return merged.drop(columns=["_join_id"])


def prospective_validation(
    frozen: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    prior_best_fitness: float | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Compute prospective validation metrics and per-algorithm breakdowns.

    Metrics include Pearson, Spearman, RMSE, MAE, ranking precision@k,
    95% interval coverage, fitness improvement rate, and best fitness found.

    Parameters
    ----------
    frozen : pd.DataFrame
        Immutable prediction freeze from before synthesis.
    observations : pd.DataFrame
        Cleaned measured results for the same round.
    prior_best_fitness : float or None, optional
        Best fitness in the master table before this round; enables improvement stats.
    top_k : int, optional
        Rank cutoff for precision@k (default 3).

    Returns
    -------
    dict
        Validation report with ``passed``, ``overall``, ``by_algorithm``,
        ``physics_revalidation``, and ``joined`` tables when matches exist.
    """
    joined = join_predictions_and_observations(frozen, observations)
    if joined.empty:
        return {
            "passed": False,
            "reason": "no overlapping candidate_ids between freeze and observations",
            "n_matched": 0,
        }

    y_true = joined["observed_fitness"].to_numpy(dtype=float)
    y_pred = joined["predicted_fitness"].to_numpy(dtype=float)
    err = y_true - y_pred

    best_obs = float(np.nanmax(y_true))
    mean_obs = float(np.nanmean(y_true))
    if prior_best_fitness is None:
        improvement_rate = float("nan")
        improved = float("nan")
    else:
        improved = float(np.mean(y_true > prior_best_fitness))
        improvement_rate = best_obs - float(prior_best_fitness)

    overall = {
        "n_matched": int(len(joined)),
        "pearson": _safe_corr(y_true, y_pred, "pearson"),
        "spearman": _safe_corr(y_true, y_pred, "spearman"),
        "rmse": float(np.sqrt(np.nanmean(err**2))),
        "mae": float(np.nanmean(np.abs(err))),
        "ranking_precision_at_k": ranking_precision_at_k(y_true, y_pred, k=top_k),
        "interval_coverage_95": interval_coverage(
            y_true,
            joined["ci95_low"].to_numpy(dtype=float),
            joined["ci95_high"].to_numpy(dtype=float),
        ),
        "best_fitness_found": best_obs,
        "mean_fitness_found": mean_obs,
        "fitness_improvement_vs_prior_best": improvement_rate,
        "fraction_improved_vs_prior_best": improved,
        "prior_best_fitness": prior_best_fitness,
    }

    by_algo = []
    if "selection_algorithm" in joined.columns:
        for algo, group in joined.groupby("selection_algorithm"):
            yt = group["observed_fitness"].to_numpy(dtype=float)
            yp = group["predicted_fitness"].to_numpy(dtype=float)
            e = yt - yp
            by_algo.append(
                {
                    "selection_algorithm": str(algo),
                    "n": int(len(group)),
                    "pearson": _safe_corr(yt, yp, "pearson"),
                    "spearman": _safe_corr(yt, yp, "spearman"),
                    "rmse": float(np.sqrt(np.nanmean(e**2))),
                    "mae": float(np.nanmean(np.abs(e))),
                    "best_fitness_found": float(np.nanmax(yt)),
                    "ranking_precision_at_k": ranking_precision_at_k(yt, yp, k=min(top_k, len(yt))),
                }
            )

    # Lightweight physics re-validation proxy: correlation of physics component vs obs
    physics_revalidation = {}
    if "physics_component" in joined.columns:
        phys = joined["physics_component"].to_numpy(dtype=float)
        physics_revalidation = {
            "physics_vs_obs_pearson": _safe_corr(y_true, phys, "pearson"),
            "physics_vs_obs_spearman": _safe_corr(y_true, phys, "spearman"),
        }

    # Soft pass: at least matched rows and finite RMSE
    passed = bool(overall["n_matched"] > 0 and np.isfinite(overall["rmse"]))
    return {
        "passed": passed,
        "overall": overall,
        "by_algorithm": by_algo,
        "physics_revalidation": physics_revalidation,
        "joined": joined,
    }
