"""Paired bootstrap, Wilcoxon, Holm, effect sizes, and CIs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    """Apply Holm step-down multiple-comparison adjustment.

    Parameters
    ----------
    pvalues : Iterable[float]
        Raw p-values in comparison order.

    Returns
    -------
    list[float]
        Holm-adjusted p-values in the same order as the input.
    """
    p = list(pvalues)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.zeros(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        raw = p[idx] * (m - rank)
        val = min(1.0, max(prev, raw))
        adj[idx] = val
        prev = val
    return adj.tolist()


def paired_bootstrap_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    statistic: str = "mean",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap confidence interval for ``stat(a) - stat(b)``.

    For ``statistic="rmse"``, ``values_*`` should be signed residuals
    (``y - pred``). For ``"mae"`` or ``"mean"``, values are the paired
    observations (e.g. absolute errors).

    Parameters
    ----------
    values_a : np.ndarray
        First paired sample.
    values_b : np.ndarray
        Second paired sample (same length as ``values_a``).
    statistic : {"mean", "mae", "rmse"}, default ``"mean"``
        Statistic applied to each resampled pair before differencing.
    n_boot : int, default 1000
        Number of bootstrap replicates.
    seed : int, default 42
        Random seed for resampling.
    alpha : float, default 0.05
        Two-sided CI tail probability.

    Returns
    -------
    dict[str, float]
        Observed delta, bootstrap mean, CI bounds, sample size, and metadata
        keys (``n_boot``, ``alpha``, ``statistic``). NaN deltas when ``n=0``.

    Raises
    ------
    ValueError
        If ``values_a`` and ``values_b`` differ in length.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired bootstrap requires equal-length arrays")
    n = len(a)
    if n == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }

    def _stat(x: np.ndarray) -> float:
        """Compute the configured statistic on a 1-D array."""
        if statistic == "rmse":
            return float(np.sqrt(np.mean(x**2)))
        if statistic == "mae":
            return float(np.mean(np.abs(x)))
        return float(np.mean(x))

    observed = _stat(a) - _stat(b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _stat(a[idx]) - _stat(b[idx])
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {
        "delta": float(observed),
        "delta_boot_mean": float(np.mean(deltas)),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "statistic": statistic,
    }


def wilcoxon_paired(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired differences ``a - b``.

    Parameters
    ----------
    values_a : np.ndarray
        First paired sample.
    values_b : np.ndarray
        Second paired sample.
    alternative : {"two-sided", "less", "greater"}, default ``"two-sided"``
        Alternative hypothesis passed to :func:`scipy.stats.wilcoxon`.

    Returns
    -------
    dict[str, float]
        Test statistic, p-value, and pair count. Returns p-value 1.0 when
        fewer than three pairs or all differences are zero.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    if len(diff) < 3 or np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0, "n": int(len(diff))}
    try:
        res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox")
        return {
            "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
            "n": int(len(diff)),
        }
    except ValueError:
        return {"statistic": 0.0, "pvalue": 1.0, "n": int(len(diff))}


def cohens_d_paired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute paired Cohen's d from mean and SD of differences.

    Parameters
    ----------
    values_a : np.ndarray
        First paired sample.
    values_b : np.ndarray
        Second paired sample.

    Returns
    -------
    float
        ``mean(a - b) / sd(a - b)``. NaN when fewer than two pairs; 0.0 or
        ``inf`` when the difference SD is negligible.
    """
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0 if abs(float(np.mean(diff))) < 1e-12 else float("inf")
    return float(np.mean(diff) / sd)


def cliffs_delta(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Cliff's delta nonparametric effect size.

    Uses the unpaired dominance formula on the two paired vectors, which
    remains informative at Stage-0 LOCO scale.

    Parameters
    ----------
    values_a : np.ndarray
        First sample vector.
    values_b : np.ndarray
        Second sample vector.

    Returns
    -------
    float
        ``(P(a > b) - P(a < b))`` over all cross-pairs. NaN when either
        vector is empty.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Efficient-ish for small n (Stage-0 LOCO scale)
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (len(a) * len(b)))


def align_paired_predictions(
    predictions: pd.DataFrame,
    config_a: str,
    config_b: str,
    *,
    id_cols: tuple[str, ...] = ("split_id", "construct_id"),
) -> pd.DataFrame:
    """Inner-join predictions from two ablation configs for paired tests.

    Parameters
    ----------
    predictions : pd.DataFrame
        Long-form ablation predictions with ``ablation_id``.
    config_a : str
        First ablation configuration ID.
    config_b : str
        Second ablation configuration ID.
    id_cols : tuple[str, ...], default (``"split_id"``, ``"construct_id"``)
        Columns used to align held-out rows.

    Returns
    -------
    pd.DataFrame
        Merged rows with ``_a`` / ``_b`` suffixes on overlapping columns.
        Empty when either config has no rows or no overlap exists.
    """
    left = predictions[predictions["ablation_id"] == config_a].copy()
    right = predictions[predictions["ablation_id"] == config_b].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(
        right,
        on=list(id_cols),
        suffixes=("_a", "_b"),
    )
    return merged


def compare_ablation_pair(
    predictions: pd.DataFrame,
    config_a: str,
    config_b: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two ablation configs on paired held-out predictions.

    Negative RMSE or MAE delta indicates ``config_a`` has lower error.

    Parameters
    ----------
    predictions : pd.DataFrame
        Long-form ablation prediction table.
    config_a : str
        Configuration treated as the focal (potentially better) model.
    config_b : str
        Reference or baseline configuration.
    n_boot : int, default 1000
        Bootstrap replicates for delta CIs.
    seed : int, default 42
        Random seed for bootstrap resampling.
    alpha : float, default 0.05
        CI tail probability.

    Returns
    -------
    dict[str, Any]
        RMSE/MAE summaries, bootstrap deltas with CIs, Wilcoxon p-values,
        effect sizes, and nested ``bootstrap`` dict. When no pairs overlap,
        returns ``ok=False`` with ``reason``.
    """
    paired = align_paired_predictions(predictions, config_a, config_b)
    if paired.empty:
        return {
            "config_a": config_a,
            "config_b": config_b,
            "n_paired": 0,
            "ok": False,
            "reason": "no overlapping split/construct pairs",
        }

    err_a = paired["y_true_a"].to_numpy(dtype=float) - paired["fitness_mean_a"].to_numpy(dtype=float)
    err_b = paired["y_true_b"].to_numpy(dtype=float) - paired["fitness_mean_b"].to_numpy(dtype=float)
    abs_a = np.abs(err_a)
    abs_b = np.abs(err_b)

    boot_rmse = paired_bootstrap_delta(
        err_a, err_b, statistic="rmse", n_boot=n_boot, seed=seed, alpha=alpha
    )
    boot_mae = paired_bootstrap_delta(
        abs_a, abs_b, statistic="mean", n_boot=n_boot, seed=seed, alpha=alpha
    )
    # Wilcoxon on abs errors: positive (b-a) favors a when alternative='greater'
    wil = wilcoxon_paired(abs_b, abs_a, alternative="greater")
    wil_two = wilcoxon_paired(abs_a, abs_b, alternative="two-sided")

    return {
        "config_a": config_a,
        "config_b": config_b,
        "n_paired": int(len(paired)),
        "ok": True,
        "rmse_a": float(np.sqrt(np.mean(err_a**2))),
        "rmse_b": float(np.sqrt(np.mean(err_b**2))),
        "mae_a": float(np.mean(abs_a)),
        "mae_b": float(np.mean(abs_b)),
        "delta_rmse": boot_rmse["delta"],
        "delta_rmse_ci_low": boot_rmse["ci_low"],
        "delta_rmse_ci_high": boot_rmse["ci_high"],
        "delta_mae": boot_mae["delta"],
        "delta_mae_ci_low": boot_mae["ci_low"],
        "delta_mae_ci_high": boot_mae["ci_high"],
        "wilcoxon_stat": wil["statistic"],
        "wilcoxon_p_greater_a_better": wil["pvalue"],
        "wilcoxon_p_two_sided": wil_two["pvalue"],
        "cohens_d_abs_error": cohens_d_paired(abs_a, abs_b),
        "cliffs_delta_abs_error": cliffs_delta(abs_a, abs_b),
        "bootstrap": {"rmse": boot_rmse, "mae": boot_mae},
    }


def run_ablation_statistics(
    predictions: pd.DataFrame,
    *,
    reference_config_id: str | None = None,
    pairwise: bool = False,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run paired statistical comparisons across the ablation matrix.

    By default each configuration is compared to ``reference_config_id``.
    Set ``pairwise=True`` for the full upper triangle with Holm correction
    over all pairs.

    Parameters
    ----------
    predictions : pd.DataFrame
        Long-form ablation predictions with ``ablation_id``.
    reference_config_id : str, optional
        Baseline config for one-vs-reference mode. Defaults to the first
        sorted ablation ID.
    pairwise : bool, default False
        When ``True``, compare every unordered pair of configs.
    n_boot : int, default 1000
        Bootstrap replicates per comparison.
    seed : int, default 42
        Random seed for bootstrap resampling.
    alpha : float, default 0.05
        Significance level for Holm-adjusted Wilcoxon tests.

    Returns
    -------
    dict[str, Any]
        Report with ``comparisons`` list (Holm-adjusted p-values and evidence
        flags), reference ID, and summary counts. Returns ``ok=False`` when
        predictions are empty.

    Raises
    ------
    ValueError
        If ``reference_config_id`` is not among the prediction ablation IDs.
    """
    if predictions.empty or "ablation_id" not in predictions.columns:
        return {"ok": False, "reason": "empty predictions", "comparisons": []}

    ids = sorted(predictions["ablation_id"].astype(str).unique().tolist())
    if reference_config_id is None:
        reference_config_id = ids[0]
    if reference_config_id not in ids:
        raise ValueError(f"Reference {reference_config_id} not in {ids}")

    pairs: list[tuple[str, str]] = []
    if pairwise:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pairs.append((a, b))
    else:
        for a in ids:
            if a == reference_config_id:
                continue
            pairs.append((a, reference_config_id))

    comparisons = [
        compare_ablation_pair(
            predictions, a, b, n_boot=n_boot, seed=seed, alpha=alpha
        )
        for a, b in pairs
    ]

    pvals = [
        c.get("wilcoxon_p_two_sided", 1.0)
        for c in comparisons
        if c.get("ok")
    ]
    adj = holm_adjust(pvals)
    j = 0
    for c in comparisons:
        if not c.get("ok"):
            c["wilcoxon_p_holm"] = None
            c["significant_holm"] = False
            continue
        c["wilcoxon_p_holm"] = float(adj[j])
        c["significant_holm"] = bool(adj[j] < alpha)
        # Evidence: Holm-significant OR bootstrap CI for ΔRMSE entirely < 0 (a better)
        ci_hi = c.get("delta_rmse_ci_high")
        c["bootstrap_favors_a"] = ci_hi is not None and ci_hi < 0
        c["evidence_a_better"] = bool(
            c["significant_holm"] or c.get("bootstrap_favors_a")
        )
        j += 1

    return {
        "ok": True,
        "reference_config_id": reference_config_id,
        "pairwise": pairwise,
        "alpha": alpha,
        "n_boot": n_boot,
        "random_seed": seed,
        "ablation_ids": ids,
        "comparisons": comparisons,
        "n_comparisons": len(comparisons),
        "n_significant_holm": sum(1 for c in comparisons if c.get("significant_holm")),
    }


def comparisons_to_frame(stats_report: dict[str, Any]) -> pd.DataFrame:
    """Flatten comparison dicts into a tabular DataFrame.

    Parameters
    ----------
    stats_report : dict[str, Any]
        Statistics report containing a ``comparisons`` list of dicts.

    Returns
    -------
    pd.DataFrame
        One row per comparison; nested ``bootstrap`` blobs are omitted.
    """
    rows = []
    for c in stats_report.get("comparisons", []):
        row = {k: v for k, v in c.items() if k != "bootstrap"}
        rows.append(row)
    return pd.DataFrame(rows)
