"""Automatic figure generation for ablations and prospective rounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _try_matplotlib():
    """Import matplotlib with a non-interactive backend when available.

    Parameters
    ----------
    None

    Returns
    -------
    module or None
        ``matplotlib.pyplot`` on success, or ``None`` if matplotlib is not
        installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def write_metrics_table(metrics: pd.DataFrame, path: Path) -> Path:
    """Write ablation metrics to CSV.

    Parameters
    ----------
    metrics : pd.DataFrame
        Per-config metrics table.
    path : Path
        Destination CSV path (parent directories are created).

    Returns
    -------
    Path
        ``path`` after writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False, encoding="utf-8")
    return path


def write_comparisons_table(comparisons: pd.DataFrame, path: Path) -> Path:
    """Write paired comparison statistics to CSV.

    Parameters
    ----------
    comparisons : pd.DataFrame
        Flattened pairwise comparison table.
    path : Path
        Destination CSV path (parent directories are created).

    Returns
    -------
    Path
        ``path`` after writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(path, index=False, encoding="utf-8")
    return path


def figure_rmse_by_config(
    metrics: pd.DataFrame,
    path: Path,
    *,
    title: str = "Ablation RMSE by configuration",
) -> Path | None:
    """Bar chart of RMSE per ablation configuration.

    Parameters
    ----------
    metrics : pd.DataFrame
        Table with an ``rmse`` column and optional ``ablation_label`` or
        ``ablation_id``.
    path : Path
        Output PNG path.
    title : str
        Plot title.

    Returns
    -------
    Path or None
        ``path`` when the figure is written; ``None`` if matplotlib is missing,
        data is empty, or ``rmse`` is absent.
    """
    plt = _try_matplotlib()
    if plt is None or metrics.empty or "rmse" not in metrics.columns:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics.get("ablation_label", metrics.get("ablation_id", metrics.index.astype(str)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(metrics))
    ax.bar(x, metrics["rmse"].to_numpy(dtype=float), color="#2F4F4F")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in labels], rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_delta_forest(
    comparisons: pd.DataFrame,
    path: Path,
    *,
    title: str = "ΔRMSE vs reference (paired bootstrap 95% CI)",
) -> Path | None:
    """Forest plot of paired RMSE deltas with bootstrap confidence intervals.

    Parameters
    ----------
    comparisons : pd.DataFrame
        Table with ``config_a``, ``delta_rmse``, ``delta_rmse_ci_low``, and
        ``delta_rmse_ci_high`` columns.
    path : Path
        Output PNG path.
    title : str
        Plot title.

    Returns
    -------
    Path or None
        ``path`` when the figure is written; ``None`` if matplotlib is missing,
        required columns are absent, or no valid rows remain.
    """
    plt = _try_matplotlib()
    needed = {"config_a", "delta_rmse", "delta_rmse_ci_low", "delta_rmse_ci_high"}
    if plt is None or comparisons.empty or not needed.issubset(comparisons.columns):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df = comparisons.dropna(subset=["delta_rmse"]).copy()
    if df.empty:
        return None
    df = df.iloc[::-1]  # top = first comparison
    y = range(len(df))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(df) + 1)))
    ax.axvline(0.0, color="#888888", lw=1, linestyle="--")
    ax.errorbar(
        df["delta_rmse"],
        list(y),
        xerr=[
            df["delta_rmse"] - df["delta_rmse_ci_low"],
            df["delta_rmse_ci_high"] - df["delta_rmse"],
        ],
        fmt="o",
        color="#1F4E79",
        ecolor="#1F4E79",
        capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(df["config_a"], df["config_b"], strict=True)])
    ax.set_xlabel("ΔRMSE (config_a − config_b); negative favors a")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_effect_sizes(
    comparisons: pd.DataFrame,
    path: Path,
    *,
    title: str = "Effect sizes (paired Cohen's d on |error|)",
) -> Path | None:
    """Horizontal bar chart of paired Cohen's d effect sizes on absolute error.

    Parameters
    ----------
    comparisons : pd.DataFrame
        Table with ``config_a``, ``config_b``, and ``cohens_d_abs_error``.
    path : Path
        Output PNG path.
    title : str
        Plot title.

    Returns
    -------
    Path or None
        ``path`` when the figure is written; ``None`` if matplotlib is missing,
        data is empty, or effect-size column is absent.
    """
    plt = _try_matplotlib()
    if plt is None or comparisons.empty or "cohens_d_abs_error" not in comparisons.columns:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df = comparisons.dropna(subset=["cohens_d_abs_error"]).copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(df) + 1)))
    labels = [f"{a} vs {b}" for a, b in zip(df["config_a"], df["config_b"], strict=True)]
    ax.barh(labels, df["cohens_d_abs_error"].to_numpy(dtype=float), color="#4A6FA5")
    ax.axvline(0.0, color="#888888", lw=1, linestyle="--")
    ax.set_xlabel("Cohen's d (paired |error|)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_ablation_figures(
    *,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    """Write ablation tables and optional matplotlib figures.

    CSV tables are always written; PNG figures are produced when matplotlib is
    installed.

    Parameters
    ----------
    metrics : pd.DataFrame
        Per-config aggregate metrics.
    comparisons : pd.DataFrame
        Flattened pairwise comparison statistics.
    out_dir : Path
        Output directory for CSV and PNG artifacts.

    Returns
    -------
    dict[str, Any]
        Artifact paths keyed by name (``metrics_csv``, ``comparisons_csv``,
        figure keys, ``matplotlib_available``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "metrics_csv": str(write_metrics_table(metrics, out_dir / "ablation_metrics.csv")),
        "comparisons_csv": str(
            write_comparisons_table(comparisons, out_dir / "ablation_comparisons.csv")
        ),
        "matplotlib_available": _try_matplotlib() is not None,
    }
    for name, fn in (
        ("rmse_by_config", lambda: figure_rmse_by_config(metrics, out_dir / "fig_rmse_by_config.png")),
        ("delta_forest", lambda: figure_delta_forest(comparisons, out_dir / "fig_delta_rmse_forest.png")),
        ("effect_sizes", lambda: figure_effect_sizes(comparisons, out_dir / "fig_effect_sizes.png")),
    ):
        path = fn()
        artifacts[name] = str(path) if path is not None else None
    return artifacts
