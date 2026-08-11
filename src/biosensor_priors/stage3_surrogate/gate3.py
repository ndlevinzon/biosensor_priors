"""Gate 3: fused model must improve over physics-only and GP-only baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Compute Pearson or Spearman correlation with safe fallbacks.

    Parameters
    ----------
    y_true : numpy.ndarray
        Observed target values.
    y_pred : numpy.ndarray
        Predicted values.
    method : str
        ``pearson`` or ``spearman``.

    Returns
    -------
    float
        Correlation coefficient, or NaN when undefined.
    """
    if len(y_true) < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(y_true, y_pred).statistic)
    return float(stats.spearmanr(y_true, y_pred).statistic)


def _topk_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """Fraction of true top-k constructs recovered by predicted ranking.

    Parameters
    ----------
    y_true : numpy.ndarray
        Observed fitness values.
    y_pred : numpy.ndarray
        Predicted fitness values.
    k : int, optional
        Top-k set size (default 3).

    Returns
    -------
    float
        Overlap fraction in ``[0, 1]``, or NaN when ``len(y_true) < k``.
    """
    if len(y_true) < k:
        return float("nan")
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / float(k)


def metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for paired true/predicted values.

    Parameters
    ----------
    y_true : numpy.ndarray
        Observed fitness values.
    y_pred : numpy.ndarray
        Predicted fitness values.

    Returns
    -------
    dict
        Keys ``rmse``, ``mae``, ``pearson``, ``spearman``, and ``topk3``.
    """
    err = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": _safe_corr(y_true, y_pred, "pearson"),
        "spearman": _safe_corr(y_true, y_pred, "spearman"),
        "topk3": _topk_accuracy(y_true, y_pred, k=3),
    }


def _holm_adjust(pvalues: list[float]) -> list[float]:
    """Apply Holm step-down correction to a list of p-values.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values to adjust.

    Returns
    -------
    list of float
        Holm-adjusted p-values in original order.
    """
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.zeros(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        raw = pvalues[idx] * (m - rank)
        val = min(1.0, max(prev, raw))
        adj[idx] = val
        prev = val
    return adj.tolist()


def _bootstrap_rmse_delta(
    err_a: np.ndarray,
    err_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for RMSE(a) − RMSE(b).

    Negative delta favors model ``a``.

    Parameters
    ----------
    err_a : numpy.ndarray
        Signed errors for model ``a``.
    err_b : numpy.ndarray
        Signed errors for model ``b``.
    n_boot : int, optional
        Number of bootstrap resamples (default 1000).
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    dict
        Bootstrap mean and 95% CI for the RMSE difference.
    """
    rng = np.random.default_rng(seed)
    n = len(err_a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_a = float(np.sqrt(np.mean(err_a[idx] ** 2)))
        rmse_b = float(np.sqrt(np.mean(err_b[idx] ** 2)))
        deltas.append(rmse_a - rmse_b)
    deltas_arr = np.asarray(deltas, dtype=float)
    return {
        "delta_rmse_mean": float(np.mean(deltas_arr)),
        "delta_rmse_ci_low": float(np.quantile(deltas_arr, 0.025)),
        "delta_rmse_ci_high": float(np.quantile(deltas_arr, 0.975)),
    }


def summarize_by_model(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CV metrics grouped by ``model_kind``.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Cross-validation prediction rows with ``model_kind``, ``y_true``,
        and ``fitness_mean``.

    Returns
    -------
    pandas.DataFrame
        One row per model kind with regression metrics.
    """
    rows = []
    for kind, group in predictions.groupby("model_kind"):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["fitness_mean"].to_numpy(dtype=float)
        rows.append({"model_kind": kind, **metrics_for_predictions(y_true, y_pred)})
    return pd.DataFrame(rows)


def evaluate_gate3(
    predictions: pd.DataFrame,
    *,
    fused_kind: str = "physics_gp",
    baselines: tuple[str, ...] = ("physics_only", "gp_zero_mean"),
    alpha: float = 0.05,
    n_boot: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate Gate 3: fused model must improve over baseline models.

    Criteria (all required):
      1. Point-estimate RMSE(fused) <= RMSE(each baseline)
      2. For each baseline, either Holm-corrected Wilcoxon on |error| favors
         fused, or bootstrap CI for RMSE delta is entirely <= 0.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Cross-validation predictions for all model kinds.
    fused_kind : str, optional
        Fused model identifier (default ``physics_gp``).
    baselines : tuple of str, optional
        Baseline model kinds to compare against.
    alpha : float, optional
        Significance level for Holm correction (default 0.05).
    n_boot : int, optional
        Bootstrap resamples for RMSE delta CI (default 1000).
    random_seed : int, optional
        Random seed for bootstrap (default 42).

    Returns
    -------
    dict
        Gate verdict with ``passed``, ``comparisons``, and ``summary``.
    """
    summary = summarize_by_model(predictions)
    summary_map = summary.set_index("model_kind").to_dict(orient="index")
    if fused_kind not in summary_map:
        return {"passed": False, "reason": f"missing fused kind {fused_kind}", "summary": summary}

    # Align paired rows by split_id + construct_id
    fused = predictions[predictions["model_kind"] == fused_kind].copy()
    comparisons = []
    pvals = []

    for baseline in baselines:
        base = predictions[predictions["model_kind"] == baseline].copy()
        if base.empty:
            return {"passed": False, "reason": f"missing baseline {baseline}", "summary": summary}
        merged = fused.merge(
            base,
            on=["split_id", "construct_id"],
            suffixes=("_fused", "_base"),
        )
        if merged.empty:
            return {"passed": False, "reason": f"no paired rows vs {baseline}", "summary": summary}

        err_f = merged["abs_error_fused"].to_numpy(dtype=float)
        err_b = merged["abs_error_base"].to_numpy(dtype=float)
        # Wilcoxon: lower abs error is better → compare base - fused (positive => fused better)
        diff = err_b - err_f
        if np.allclose(diff, 0):
            stat, p_two = 0.0, 1.0
        else:
            try:
                stat, p_two = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            except ValueError:
                stat, p_two = 0.0, 1.0
        pvals.append(float(p_two))

        sq_f = merged["sq_error_fused"].to_numpy(dtype=float)
        sq_b = merged["sq_error_base"].to_numpy(dtype=float)
        # bootstrap on signed errors via sqrt mean squares of abs? use residual signs from y
        # Reconstruct signed errors from stored abs is lossy; use fitness means.
        signed_f = merged["y_true_fused"].to_numpy(dtype=float) - merged["fitness_mean_fused"].to_numpy(
            dtype=float
        )
        signed_b = merged["y_true_base"].to_numpy(dtype=float) - merged["fitness_mean_base"].to_numpy(
            dtype=float
        )
        boot = _bootstrap_rmse_delta(signed_f, signed_b, n_boot=n_boot, seed=random_seed)

        comparisons.append(
            {
                "baseline": baseline,
                "wilcoxon_stat": float(stat),
                "wilcoxon_p": float(p_two),
                "rmse_fused": float(np.sqrt(np.mean(sq_f))),
                "rmse_baseline": float(np.sqrt(np.mean(sq_b))),
                **boot,
            }
        )

    holm = _holm_adjust(pvals)
    for comp, p_adj in zip(comparisons, holm, strict=True):
        comp["wilcoxon_p_holm"] = float(p_adj)
        comp["rmse_point_improved"] = comp["rmse_fused"] <= comp["rmse_baseline"] + 1e-12
        comp["wilcoxon_significant"] = comp["wilcoxon_p_holm"] < alpha
        comp["bootstrap_nonpositive"] = comp["delta_rmse_ci_high"] <= 0.0
        comp["evidence"] = bool(
            comp["rmse_point_improved"]
            and (comp["wilcoxon_significant"] or comp["bootstrap_nonpositive"])
        )

    passed = all(c["evidence"] for c in comparisons)
    # Soft pass for tiny GP-only interim: require point RMSE improvement vs both.
    soft_passed = all(c["rmse_point_improved"] for c in comparisons)

    return {
        "passed": passed,
        "soft_passed_point_rmse": soft_passed,
        "fused_kind": fused_kind,
        "baselines": list(baselines),
        "alpha": alpha,
        "comparisons": comparisons,
        "summary": summary.to_dict(orient="records"),
        "n_prediction_rows": int(len(predictions)),
    }
