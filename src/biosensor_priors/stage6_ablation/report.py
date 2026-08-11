"""Assemble tables/figures into a reproducible scientific report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.stage6_ablation.figures import generate_ablation_figures
from biosensor_priors.stage6_ablation.statistics import comparisons_to_frame


def _fmt(x: Any, digits: int = 4) -> str:
    """Format a numeric value for Markdown tables.

    Parameters
    ----------
    x : Any
        Value to format (typically float).
    digits : int, default 4
        Decimal places for finite numbers.

    Returns
    -------
    str
        Fixed-point string, or ``"—"`` for missing or non-numeric values.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.{digits}f}"


def build_markdown_report(
    *,
    metrics: pd.DataFrame,
    stats_report: dict[str, Any],
    figure_artifacts: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    """Render a concise Markdown scientific report for Stage 6.

    Parameters
    ----------
    metrics : pd.DataFrame
        Per-config aggregate metrics table.
    stats_report : dict[str, Any]
        Output of :func:`run_ablation_statistics` with comparison dicts.
    figure_artifacts : dict[str, Any]
        Paths to CSV/PNG artifacts from figure generation.
    meta : dict[str, Any]
        Run metadata (seed, encoding, split count, etc.).

    Returns
    -------
    str
        Full Markdown document body.
    """
    lines = [
        "# Stage 6 — Ablation report",
        "",
        f"- Random seed: `{meta.get('random_seed')}`",
        f"- Encoding: `{meta.get('encoding')}`",
        f"- Splits: `{meta.get('n_splits')}`",
        f"- Reference config: `{stats_report.get('reference_config_id')}`",
        f"- Bootstrap samples: `{stats_report.get('n_boot')}`",
        f"- Alpha: `{stats_report.get('alpha')}`",
        "",
        "## Ablation matrix metrics",
        "",
    ]

    if metrics.empty:
        lines.append("_No metrics available._")
    else:
        cols = [
            c
            for c in (
                "ablation_id",
                "label",
                "physics",
                "gp",
                "confidence_weighting",
                "structure_source",
                "prefilter",
                "rmse",
                "mae",
                "pearson",
                "spearman",
                "topk3",
                "n",
            )
            if c in metrics.columns
        ]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines.extend([header, sep])
        for _, row in metrics.iterrows():
            cells = []
            for c in cols:
                val = row[c]
                if c in {"rmse", "mae", "pearson", "spearman", "topk3"}:
                    cells.append(_fmt(val))
                else:
                    cells.append(str(val) if val is not None and val == val else "—")
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Paired comparisons", ""])
    comps = comparisons_to_frame(stats_report)
    if comps.empty:
        lines.append("_No paired comparisons._")
    else:
        lines.append(
            "| config_a | config_b | ΔRMSE | 95% CI | Wilcoxon p (Holm) | Cohen's d | evidence a better |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in comps.iterrows():
            ci = f"[{_fmt(row.get('delta_rmse_ci_low'))}, {_fmt(row.get('delta_rmse_ci_high'))}]"
            lines.append(
                "| {a} | {b} | {d} | {ci} | {p} | {es} | {ev} |".format(
                    a=row.get("config_a"),
                    b=row.get("config_b"),
                    d=_fmt(row.get("delta_rmse")),
                    ci=ci,
                    p=_fmt(row.get("wilcoxon_p_holm"), 4),
                    es=_fmt(row.get("cohens_d_abs_error")),
                    ev=bool(row.get("evidence_a_better")),
                )
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Metrics CSV: `{figure_artifacts.get('metrics_csv')}`",
            f"- Comparisons CSV: `{figure_artifacts.get('comparisons_csv')}`",
            f"- RMSE figure: `{figure_artifacts.get('rmse_by_config')}`",
            f"- ΔRMSE forest: `{figure_artifacts.get('delta_forest')}`",
            f"- Effect sizes: `{figure_artifacts.get('effect_sizes')}`",
            "",
            "## Notes",
            "",
            "Physics weights / structure-source slots that lack Stage-1/2 artifacts "
            "still run under shared splits; ``structure_available=false`` means a "
            "deterministic confidence proxy was used until real AF2/AF3 tables land.",
            "",
            "Stage 6 does not replace Gates 0–5; it supplies the scientific evidence "
            "matrix those gates summarize.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    out_dir: Path,
    metrics: pd.DataFrame,
    stats_report: dict[str, Any],
    predictions: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write tables, figures, JSON stats, predictions, and Markdown report.

    Parameters
    ----------
    out_dir : Path
        Output directory for all Stage-6 artifacts.
    metrics : pd.DataFrame
        Per-config aggregate metrics.
    stats_report : dict[str, Any]
        Full statistics report (bootstrap blobs stripped in saved JSON).
    predictions : pd.DataFrame, optional
        Per-construct prediction rows to persist as parquet.
    meta : dict[str, Any], optional
        Run metadata embedded in the Markdown report.

    Returns
    -------
    dict[str, Any]
        Artifact paths: CSVs, figures, ``report_md``, ``statistics_json``,
        and optional ``predictions`` parquet path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    comps = comparisons_to_frame(stats_report)
    figures = generate_ablation_figures(metrics=metrics, comparisons=comps, out_dir=out_dir)

    stats_path = out_dir / "ablation_statistics.json"
    # Drop nested bootstrap arrays for lean JSON? keep summaries only
    lean = dict(stats_report)
    lean_comps = []
    for c in stats_report.get("comparisons", []):
        lean_comps.append({k: v for k, v in c.items() if k != "bootstrap"})
    lean["comparisons"] = lean_comps
    stats_path.write_text(json.dumps(lean, indent=2, default=str), encoding="utf-8")

    if predictions is not None and not predictions.empty:
        pred_path = out_dir / "ablation_predictions.parquet"
        store = predictions.copy()
        for col in store.columns:
            if store[col].dtype == object:
                store[col] = store[col].map(lambda x: None if x is None else str(x))
        store.to_parquet(pred_path, index=False)
        figures["predictions"] = str(pred_path)

    md = build_markdown_report(
        metrics=metrics,
        stats_report=stats_report,
        figure_artifacts=figures,
        meta=meta,
    )
    report_path = out_dir / "ablation_report.md"
    report_path.write_text(md, encoding="utf-8")
    figures["report_md"] = str(report_path)
    figures["statistics_json"] = str(stats_path)
    return figures
