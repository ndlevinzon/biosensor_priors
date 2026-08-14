"""Stage-gate figure reports: observations, metrics, statistics, confidence.

Each stage writes a self-contained folder under ``outputs/gate_reports/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path

INK = "#1B1B18"
MUTED = "#5E5850"
PAPER = "#F7F6F3"
PANEL = "#FFFFFF"
RULE = "#D4CFC6"
PASS = "#2C6E49"
FAIL = "#9B2C2C"
TEAL = "#1F4E5F"
OCEAN = "#3D7A6A"
SAND = "#C4A574"
SLATE = "#5B7C8D"
WARN = "#B56A2B"

STAGE_TITLES = {
    "stage0": "Stage 0  Ground truth and fitness",
    "stage1": "Stage 1  Structure ensemble and confidence",
    "stage2": "Stage 2  Physics landscape (RF3 docking)",
    "stage3": "Stage 3  Physics-informed surrogate",
    "stage4": "Stage 4  Search and design proposals",
    "stage5": "Stage 5  Prospective wet-lab loop",
    "stage6": "Stage 6  Ablation and validation",
}


def _pyplot():
    """Import matplotlib with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "savefig.facecolor": PAPER,
            "axes.grid": False,
        }
    )
    return plt, GridSpec


def gate_reports_root(repo_root: Path | None = None) -> Path:
    """Return ``outputs/gate_reports``, creating it if needed."""
    root = Path(repo_root or REPO_ROOT)
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    configured = (pipeline.get("paths") or {}).get("gate_reports")
    if configured:
        out = resolve_path(configured, root)
    else:
        out = resolve_path(pipeline["paths"]["outputs"], root) / "gate_reports"
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# Gate reports",
                    "",
                    "One folder per pipeline stage, written when that stage's",
                    "gate runs. Each folder holds an overview figure, focused",
                    "plots, ``index.md``, ``gate.json``, and ``stats.json``.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return out


def stage_report_dir(stage: str, *, repo_root: Path | None = None) -> Path:
    """Return and create ``outputs/gate_reports/<stage>/``."""
    path = gate_reports_root(repo_root) / str(stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _style_ax(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, pad=8, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=3, labelsize=8)
    ax.set_facecolor(PANEL)


def _empty(ax, message: str = "No observations yet") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(PANEL)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=10)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        if pd.isna(value):
            return "--"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _passed(gate: dict[str, Any] | None) -> bool:
    if not gate:
        return False
    if "operational_passed" in gate:
        return bool(gate.get("operational_passed") or gate.get("passed"))
    return bool(gate.get("passed"))


def _checks(gate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not gate:
        return []
    checks = list(gate.get("checks") or [])
    if checks:
        return checks
    tests = list(gate.get("tests") or [])
    if tests:
        return [
            {
                "name": str(t.get("mutation") or t.get("name") or "test"),
                "passed": bool(t.get("passed")),
            }
            for t in tests
        ]
    comparisons = list(gate.get("comparisons") or [])
    if comparisons:
        return [
            {
                "name": str(c.get("baseline") or c.get("name") or "cmp"),
                "passed": bool(c.get("evidence", c.get("passed", False))),
            }
            for c in comparisons
        ]
    return []


def _draw_header(ax, title: str, passed: bool, kpis: Sequence[str]) -> None:
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    color = PASS if passed else FAIL
    ax.add_patch(
        Rectangle(
            (0.0, 0.08),
            0.01,
            0.84,
            transform=ax.transAxes,
            color=color,
            linewidth=0,
        )
    )
    ax.text(
        0.03,
        0.68,
        title,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.03,
        0.28,
        "   |   ".join(kpis),
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="center",
    )
    ax.text(
        0.98,
        0.5,
        "PASS" if passed else "FAIL",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=color,
        ha="right",
        va="center",
    )


def _draw_checks(ax, checks: Sequence[dict[str, Any]]) -> None:
    _style_ax(ax, "Gate checks")
    if not checks:
        _empty(ax, "No named checks")
        return
    names = [str(c.get("name", "check")) for c in checks]
    ok = [bool(c.get("passed")) for c in checks]
    y = np.arange(len(names))
    ax.barh(
        y,
        np.ones(len(names)),
        color=[PASS if v else FAIL for v in ok],
        height=0.62,
        linewidth=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    for i, flag in enumerate(ok):
        ax.text(
            0.02,
            i,
            "pass" if flag else "fail",
            va="center",
            ha="left",
            color="white",
            fontsize=8,
            fontweight="bold",
        )


def render_dashboard(
    *,
    title: str,
    passed: bool,
    kpis: Sequence[str],
    panels: Sequence[tuple[str, Callable]],
    checks: Sequence[dict[str, Any]],
    path: Path,
) -> Path:
    """Write a three-row overview figure (header, panels, gate checks)."""
    plt, grid_spec = _pyplot()
    n = max(len(panels), 1)
    fig = plt.figure(figsize=(13.6, 8.6))
    gs = grid_spec(
        3,
        n,
        height_ratios=[0.18, 1.15, 0.38],
        hspace=0.42,
        wspace=0.32,
        left=0.06,
        right=0.97,
        top=0.96,
        bottom=0.07,
    )
    ax_h = fig.add_subplot(gs[0, :])
    _draw_header(ax_h, title, passed, kpis)
    if panels:
        for i, (panel_title, draw) in enumerate(panels):
            ax = fig.add_subplot(gs[1, i])
            _style_ax(ax, panel_title)
            draw(ax)
    else:
        ax = fig.add_subplot(gs[1, :])
        _empty(ax)
    ax_c = fig.add_subplot(gs[2, :])
    _draw_checks(ax_c, checks)
    _save(fig, path)
    plt.close(fig)
    return path


def _try_overview(**kwargs) -> dict[str, Path]:
    """Render a dashboard, or skip figures when matplotlib is missing."""
    try:
        return {"overview": render_dashboard(**kwargs)}
    except ImportError:
        return {}
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Gate overview figure skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        path = kwargs.get("path")
        if isinstance(path, Path):
            (path.parent / "overview_error.txt").write_text(
                repr(exc), encoding="utf-8"
            )
        return {}


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        number = float(obj)
        return number if np.isfinite(number) else None
    return obj


def _write_sidecar(
    out_dir: Path,
    *,
    stage: str,
    gate: dict[str, Any],
    stats: dict[str, Any],
    observations: Sequence[str],
    figures: dict[str, str],
) -> Path:
    gate_ready = _json_ready(gate)
    stats_ready = _json_ready(stats)
    (out_dir / "gate.json").write_text(
        json.dumps(gate_ready, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(
        json.dumps(stats_ready, indent=2, default=str), encoding="utf-8"
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {STAGE_TITLES.get(stage, stage)}",
        "",
        f"- Gate: **{'PASS' if _passed(gate) else 'FAIL'}**",
        f"- Written (UTC): `{stamp}`",
        "",
        "## Observations",
        "",
    ]
    for item in observations:
        lines.append(f"- {item}")
    if not observations:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Statistics",
            "",
            "```json",
            json.dumps(stats_ready, indent=2, default=str),
            "```",
            "",
        ]
    )
    if figures:
        lines.extend(["## Figures", ""])
        for name, rel in figures.items():
            lines.append(f"- `{name}`: `{rel}`")
        lines.append("")
    path = out_dir / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _finish(
    *,
    stage: str,
    repo_root: Path,
    out_dir: Path,
    gate: dict[str, Any],
    stats: dict[str, Any],
    observations: Sequence[str],
    figures: dict[str, Path],
) -> dict[str, Any]:
    fig_rel = {k: _rel(v, repo_root) for k, v in figures.items()}
    index = _write_sidecar(
        out_dir,
        stage=stage,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=fig_rel,
    )
    return {
        "directory": _rel(out_dir, repo_root),
        "index": _rel(index, repo_root),
        "figures": fig_rel,
        "stats": stats,
    }


def _series(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def write_stage0_report(
    master: pd.DataFrame,
    gate: dict[str, Any],
    *,
    splits: Sequence[dict[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fitness coverage, catalog scores, and Stage-0 gate checks."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage0", repo_root=root)
    n = int(len(master))
    n_fit = int(master["fitness"].notna().sum()) if "fitness" in master.columns else 0
    n_mis = (
        int((master["mutation_audit"].astype(str) == "MISMATCH").sum())
        if "mutation_audit" in master.columns
        else 0
    )
    n_ver = (
        int(master["version"].nunique()) if "version" in master.columns else 0
    )
    fit = (
        pd.to_numeric(master["fitness"], errors="coerce")
        if "fitness" in master.columns
        else pd.Series(dtype=float)
    )

    phenos = [
        ("selectivity", "_fitness_selectivity_raw"),
        ("affinity", "_fitness_affinity_raw"),
        ("FC Ac", "_fitness_fc_raw"),
        ("brightness", "_fitness_brightness_raw"),
        ("FC Prop", "_fitness_fc_prop_raw"),
    ]
    coverage = []
    for label, col in phenos:
        if col in master.columns:
            labeled = pd.to_numeric(master[col], errors="coerce")
            coverage.append((label, float(labeled.notna().mean())))
        else:
            coverage.append((label, 0.0))

    def panel_coverage(ax) -> None:
        labels, vals = zip(*coverage, strict=True)
        ax.bar(labels, vals, color=OCEAN, width=0.65)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction labeled")
        ax.tick_params(axis="x", rotation=20)

    def panel_fitness(ax) -> None:
        vals = fit.dropna().to_numpy(dtype=float)
        if vals.size == 0:
            _empty(ax, "No fitness labels")
            return
        ax.hist(vals, bins=min(12, max(5, vals.size // 2)), color=TEAL, edgecolor=PAPER)
        ax.set_xlabel("catalog fitness")
        ax.set_ylabel("constructs")

    def panel_versions(ax) -> None:
        if "version" not in master.columns:
            _empty(ax, "No version column")
            return
        labeled = (
            master["fitness"].notna()
            if "fitness" in master.columns
            else pd.Series(False, index=master.index)
        )
        counts = (
            master.assign(_lab=labeled)
            .groupby(master["version"].astype(str))["_lab"]
            .agg(["sum", "count"])
        )
        if counts.empty:
            _empty(ax)
            return
        x = np.arange(len(counts))
        ax.bar(x, counts["count"], color=SLATE, label="all", width=0.7)
        ax.bar(x, counts["sum"], color=OCEAN, label="labeled", width=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(list(counts.index), rotation=30, ha="right")
        ax.set_ylabel("constructs")
        ax.legend(frameon=False, fontsize=8)

    kpis = [
        f"{n} constructs",
        f"{n_fit} labeled",
        f"{n_ver} versions",
        f"{n_mis} MISMATCH",
        f"mean F {_fmt(fit.mean())}",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage0"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("Phenotype coverage", panel_coverage),
            ("Fitness distribution", panel_fitness),
            ("Constructs by version", panel_versions),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "n_constructs": n,
        "n_labeled": n_fit,
        "n_mismatch": n_mis,
        "n_versions": n_ver,
        "n_splits": int(len(splits or [])),
        "fitness_mean": float(fit.mean()) if fit.notna().any() else None,
        "fitness_std": float(fit.std()) if fit.notna().sum() > 1 else None,
        "phenotype_coverage": {k: v for k, v in coverage},
    }
    observations = [
        f"{n_fit} of {n} constructs carry catalog fitness (min_components rule).",
        f"{n_mis} identities are MISMATCH and unlabeled.",
        "FC PropCoA is in F (higher = less off-target response).",
        f"{len(splits or [])} frozen splits written for paired Stage 3/6 work.",
    ]
    return _finish(
        stage="stage0",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage1_report(
    gate: dict[str, Any],
    *,
    registry: pd.DataFrame | None = None,
    models: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Jobs, ingested models, and per-residue structural confidence."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage1", repo_root=root)
    registry = registry if registry is not None else pd.DataFrame()
    models = models if models is not None else pd.DataFrame()
    confidence = confidence if confidence is not None else pd.DataFrame()
    conf = _series(confidence, "Confidence", "structural_confidence", "confidence")
    plddt = _series(confidence, "pLDDT", "plddt")
    reliable_frac = gate.get("reliable_fraction")
    if reliable_frac is None and "Reliable" in confidence.columns:
        rel = confidence["Reliable"]
        if rel.dtype == object:
            flags = rel.astype(str).str.lower().isin(["yes", "true", "1"])
            reliable_frac = float(flags.mean())
        elif len(rel):
            reliable_frac = float(pd.Series(rel).astype(bool).mean())

    def panel_jobs(ax) -> None:
        col = next(
            (c for c in ("predictor", "method", "model") if c in registry.columns),
            None,
        )
        if registry.empty or col is None:
            _empty(ax, "No jobs in registry")
            return
        counts = registry[col].astype(str).value_counts()
        ax.barh(counts.index.astype(str)[::-1], counts.to_numpy()[::-1], color=SLATE)
        ax.set_xlabel("jobs")

    def panel_conf(ax) -> None:
        if conf is None or not conf.notna().any():
            _empty(ax, "No confidence table")
            return
        ax.hist(
            conf.dropna().to_numpy(dtype=float),
            bins=18,
            color=TEAL,
            edgecolor=PAPER,
        )
        ax.set_xlabel("structural confidence")
        ax.set_ylabel("positions")

    def panel_ipsae(ax) -> None:
        ipsae = _series(models, "ipsae", "ipSAE")
        if ipsae is None or not ipsae.notna().any():
            _empty(ax, "No ipSAE on models")
            return
        ax.hist(
            ipsae.dropna().to_numpy(dtype=float),
            bins=12,
            color=OCEAN,
            edgecolor=PAPER,
        )
        ax.set_xlabel("ipSAE")
        ax.set_ylabel("models")

    kpis = [
        f"{int(len(registry))} jobs",
        f"{int(len(models))} models",
        f"{int(len(confidence))} residue rows",
        f"reliable {_fmt(reliable_frac)}",
        f"mean conf {_fmt(conf.mean() if conf is not None else None)}",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage1"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("Jobs by predictor", panel_jobs),
            ("Residue confidence", panel_conf),
            ("Interface ipSAE", panel_ipsae),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "n_jobs": int(len(registry)),
        "n_models": int(len(models)),
        "n_confidence_rows": int(len(confidence)),
        "reliable_fraction": reliable_frac,
        "mean_confidence": (
            float(conf.mean())
            if conf is not None and conf.notna().any()
            else None
        ),
        "mean_plddt": (
            float(plddt.mean())
            if plddt is not None and plddt.notna().any()
            else None
        ),
    }
    observations = [
        "Confidence is a composite of pLDDT, cross-model RMSD, and PAE.",
        "Missing models before HPC is expected when only jobs were scripted.",
        f"Reliable-residue fraction: {_fmt(reliable_frac)}.",
    ]
    return _finish(
        stage="stage1",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage2_report(
    gate: dict[str, Any],
    *,
    summary: pd.DataFrame | None = None,
    long_table: pd.DataFrame | None = None,
    conformers: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Control direction tests, $\\Delta$RIF distribution, physics uncertainty."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage2", repo_root=root)
    summary = summary if summary is not None else pd.DataFrame()
    long_table = long_table if long_table is not None else pd.DataFrame()
    conformers = conformers if conformers is not None else pd.DataFrame()
    delta = _series(summary, "delta_rif_sel_mean", "delta_rif_sel")
    std = _series(summary, "delta_rif_sel_std", "delta_rif_sel_sd")
    conf = _series(summary, "structural_confidence", "confidence")
    tests = list(gate.get("tests") or [])

    def panel_controls(ax) -> None:
        if not tests:
            _empty(ax, "No control tests")
            return
        names = [str(t.get("mutation", "?")) for t in tests]
        vals = [
            float(t["delta_rif_sel"]) if t.get("delta_rif_sel") is not None else 0.0
            for t in tests
        ]
        colors = [PASS if t.get("passed") else FAIL for t in tests]
        ax.axhline(0.0, color=RULE, lw=1)
        ax.bar(names, vals, color=colors, width=0.55)
        ax.set_ylabel(r"$\Delta$RIF$_{\mathrm{sel}}$")

    def panel_delta(ax) -> None:
        if delta is None or not delta.notna().any():
            _empty(ax, "No $\\Delta$RIF scores")
            return
        ax.hist(
            delta.dropna().to_numpy(dtype=float),
            bins=16,
            color=TEAL,
            edgecolor=PAPER,
        )
        ax.axvline(0.0, color=WARN, lw=1, ls="--")
        ax.set_xlabel(r"$\Delta$RIF$_{\mathrm{sel}}$")
        ax.set_ylabel("mutations")

    def panel_unc(ax) -> None:
        if std is None or not std.notna().any():
            _empty(ax, "No physics uncertainty")
            return
        x = delta.fillna(0.0) if delta is not None else pd.Series(np.zeros(len(std)))
        c = conf.fillna(0.4) if conf is not None else pd.Series(np.full(len(std), 0.4))
        ax.scatter(x, std, c=c, cmap="YlGnBu", s=18, alpha=0.85, edgecolors="none")
        ax.set_xlabel(r"$\Delta$RIF$_{\mathrm{sel}}$")
        ax.set_ylabel("score std")

    kpis = [
        f"{int(len(summary))} mutations",
        f"{int(len(long_table))} scan rows",
        f"{int(len(conformers))} conformers",
        f"physics_gate {gate.get('physics_gate', '--')}",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage2"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("Control directional tests", panel_controls),
            (r"Landscape $\Delta$RIF$_{\mathrm{sel}}$", panel_delta),
            ("Uncertainty vs score", panel_unc),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "n_mutations": int(len(summary)),
        "n_scan_rows": int(len(long_table)),
        "n_conformers": int(len(conformers)),
        "mean_delta_rif_sel": (
            float(delta.mean())
            if delta is not None and delta.notna().any()
            else None
        ),
        "mean_delta_std": (
            float(std.mean())
            if std is not None and std.notna().any()
            else None
        ),
        "allow_full_physics_weight": bool(gate.get("allow_full_physics_weight")),
        "failed_controls": list(gate.get("failed") or []),
    }
    observations = [
        "More negative physics score is better (frozen convention).",
        r"$\Delta$RIF_sel = RIF_Ac - RIF_Prop (negated RF3 docking confidence).",
        (
            "Controls passed; Stage 3 may use physics at full weight."
            if gate.get("allow_full_physics_weight")
            else "Gate 2 FAIL: Stage 3 must not use physics at full weight."
        ),
    ]
    return _finish(
        stage="stage2",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage3_report(
    gate: dict[str, Any],
    *,
    predictions: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    calibrator: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """CV accuracy, fused-vs-baseline evidence, and calibrated uncertainty."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage3", repo_root=root)
    predictions = predictions if predictions is not None else pd.DataFrame()
    if summary is None:
        summary = pd.DataFrame(gate.get("summary") or [])
    fused = str(gate.get("fused_kind") or "physics_gp")
    cal = calibrator or {}

    def panel_rmse(ax) -> None:
        if summary.empty or "rmse" not in summary.columns:
            _empty(ax, "No model metrics")
            return
        labels = summary.get("model_kind", summary.index.astype(str))
        ax.bar(list(map(str, labels)), summary["rmse"], color=TEAL, width=0.6)
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=15)

    def panel_scatter(ax) -> None:
        sub = predictions
        if "model_kind" in predictions.columns:
            sub = predictions[predictions["model_kind"].astype(str) == fused]
        if sub.empty or "y_true" not in sub.columns:
            _empty(ax, "No fused CV predictions")
            return
        y = pd.to_numeric(sub["y_true"], errors="coerce")
        p = pd.to_numeric(sub["fitness_mean"], errors="coerce")
        ax.scatter(y, p, s=22, color=OCEAN, alpha=0.8, edgecolors="none")
        lo = float(np.nanmin([y.min(), p.min()]))
        hi = float(np.nanmax([y.max(), p.max()]))
        ax.plot([lo, hi], [lo, hi], color=RULE, lw=1, ls="--")
        ax.set_xlabel("observed fitness")
        ax.set_ylabel("predicted")

    def panel_sigma(ax) -> None:
        sub = predictions
        if "model_kind" in predictions.columns:
            sub = predictions[predictions["model_kind"].astype(str) == fused]
        std = _series(sub, "fitness_std", "sigma_cal", "sigma_eff")
        if std is None or not std.notna().any():
            q = cal.get("conformal_quantile")
            _empty(ax, f"conformal q={_fmt(q)}" if q is not None else "No sigma")
            return
        ax.hist(
            std.dropna().to_numpy(dtype=float),
            bins=14,
            color=SLATE,
            edgecolor=PAPER,
        )
        ax.set_xlabel(r"predictive $\sigma$")
        ax.set_ylabel("CV rows")

    kpis = [
        f"fused {fused}",
        f"{int(gate.get('n_prediction_rows', len(predictions)))} CV rows",
        f"hard {'pass' if gate.get('passed') else 'fail'}",
        f"soft {'pass' if gate.get('soft_passed_point_rmse') else 'fail'}",
        f"q {_fmt(cal.get('conformal_quantile'))}",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage3"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("RMSE by model", panel_rmse),
            ("Fused predicted vs observed", panel_scatter),
            (r"Calibrated $\sigma$", panel_sigma),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "fused_kind": fused,
        "operational_passed": bool(gate.get("operational_passed", gate.get("passed"))),
        "hard_passed": bool(gate.get("passed")),
        "soft_passed_point_rmse": bool(gate.get("soft_passed_point_rmse")),
        "n_prediction_rows": int(len(predictions)),
        "metrics": summary.to_dict(orient="records") if not summary.empty else [],
        "conformal_quantile": cal.get("conformal_quantile"),
        "physics_weight_allowed": gate.get("physics_weight_allowed"),
    }
    observations = [
        "Fused model must beat physics-only and GP-only on RMSE (Gate 3).",
        "Labels are train-fold percentiles; catalog fitness is not the CV scale.",
        f"Conformal quantile q = {_fmt(cal.get('conformal_quantile'))}.",
    ]
    return _finish(
        stage="stage3",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage4_report(
    gate: dict[str, Any],
    *,
    observed: pd.DataFrame | None = None,
    design: pd.DataFrame | None = None,
    exploit: pd.DataFrame | None = None,
    explore: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Design-space size, exploit/explore batches, cost vs predicted fitness."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage4", repo_root=root)
    observed = observed if observed is not None else pd.DataFrame()
    design = design if design is not None else pd.DataFrame()
    exploit = exploit if exploit is not None else pd.DataFrame()
    explore = explore if explore is not None else pd.DataFrame()

    def panel_parents(ax) -> None:
        col = "parent_version" if "parent_version" in design.columns else "version"
        if design.empty or col not in design.columns:
            _empty(ax, "Empty design library")
            return
        counts = design[col].astype(str).value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy(), color=SLATE, width=0.6)
        ax.set_ylabel("candidates")

    def panel_cost(ax) -> None:
        parts = []
        if not exploit.empty:
            parts.append(exploit.assign(_role="exploit"))
        if not explore.empty:
            parts.append(explore.assign(_role="explore"))
        pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if pool.empty or "mutation_cost" not in pool.columns:
            _empty(ax, "No scored proposals")
            return
        mu = _series(pool, "pred_fitness_mean", "net_fitness")
        cost = pd.to_numeric(pool["mutation_cost"], errors="coerce")
        colors = np.where(pool["_role"].eq("exploit"), OCEAN, SAND)
        ax.scatter(cost, mu, c=colors, s=36, alpha=0.9, edgecolors="none")
        ax.set_xlabel("mutation cost")
        ax.set_ylabel(r"predicted $\mu$")

    def panel_unc_explore(ax) -> None:
        if explore.empty:
            _empty(ax, "No explore batch")
            return
        std = _series(explore, "pred_fitness_std")
        if std is None or not std.notna().any():
            _empty(ax, "No predictive std")
            return
        order = np.argsort(-std.to_numpy(dtype=float))
        ax.bar(np.arange(len(order)), std.to_numpy()[order], color=SAND, width=0.7)
        ax.set_xlabel("explore rank")
        ax.set_ylabel(r"$\sigma$")

    kpis = [
        f"{int(len(observed))} observed",
        f"{int(len(design))} design",
        f"{int(len(exploit))} exploit",
        f"{int(len(explore))} explore",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage4"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("Library by parent", panel_parents),
            ("Cost vs predicted fitness", panel_cost),
            ("Explore uncertainty", panel_unc_explore),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "n_observed": int(len(observed)),
        "n_design": int(len(design)),
        "n_exploit": int(len(exploit)),
        "n_explore": int(len(explore)),
        "mean_exploit_mu": (
            float(pd.to_numeric(exploit["pred_fitness_mean"], errors="coerce").mean())
            if "pred_fitness_mean" in exploit.columns and not exploit.empty
            else None
        ),
        "mean_explore_std": (
            float(pd.to_numeric(explore["pred_fitness_std"], errors="coerce").mean())
            if "pred_fitness_std" in explore.columns and not explore.empty
            else None
        ),
    }
    observations = [
        "Exploit: brightness / FC Prop floors and cost-compensating net fitness.",
        "Explore: leftover candidates ranked by predictive standard deviation.",
        "Primary artifacts: proposals_exploit.csv and proposals_explore.csv.",
    ]
    return _finish(
        stage="stage4",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage5_report(
    gate: dict[str, Any],
    *,
    validation: dict[str, Any] | None = None,
    joined: pd.DataFrame | None = None,
    round_id: int | str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Frozen predictions vs new wet-lab observations (Gate 4)."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage5", repo_root=root)
    validation = validation or {}
    overall = dict(validation.get("overall") or gate.get("overall") or {})
    joined = joined if joined is not None else pd.DataFrame()
    by_alg = list(validation.get("by_algorithm") or gate.get("by_algorithm") or [])

    def panel_scatter(ax) -> None:
        y = _series(joined, "fitness", "y_true", "observed_fitness")
        p = _series(joined, "predicted_fitness", "pred_fitness_mean")
        if y is None or p is None or not y.notna().any():
            _empty(ax, "No matched freeze rows")
            return
        ax.scatter(y, p, s=28, color=OCEAN, alpha=0.85, edgecolors="none")
        lo = float(np.nanmin([y.min(), p.min()]))
        hi = float(np.nanmax([y.max(), p.max()]))
        ax.plot([lo, hi], [lo, hi], color=RULE, lw=1, ls="--")
        ax.set_xlabel("measured fitness")
        ax.set_ylabel("frozen prediction")

    def panel_alg(ax) -> None:
        if not by_alg:
            _empty(ax, "No per-algorithm split")
            return
        names = [
            str(r.get("selection_algorithm") or r.get("algorithm") or "?")
            for r in by_alg
        ]
        rmse = [
            r.get("rmse") if r.get("rmse") is not None else np.nan for r in by_alg
        ]
        ax.bar(names, rmse, color=TEAL, width=0.55)
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=15)

    def panel_cover(ax) -> None:
        cov = overall.get("coverage") or overall.get("interval_coverage")
        n = overall.get("n_matched", 0)
        ax.bar(
            ["matched n", "coverage"],
            [float(n or 0), float(cov or 0)],
            color=[SLATE, OCEAN],
        )
        ax.set_ylabel("value")

    kpis = [
        f"round {round_id}",
        f"matched {_fmt(overall.get('n_matched'))}",
        f"RMSE {_fmt(overall.get('rmse'))}",
        f"coverage {_fmt(overall.get('coverage') or overall.get('interval_coverage'))}",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage5"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("Frozen vs measured", panel_scatter),
            ("RMSE by algorithm", panel_alg),
            ("Match count / interval coverage", panel_cover),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "round_id": round_id,
        "n_matched": overall.get("n_matched"),
        "rmse": overall.get("rmse"),
        "mae": overall.get("mae"),
        "coverage": overall.get("coverage") or overall.get("interval_coverage"),
        "failed": list(gate.get("failed") or []),
    }
    observations = [
        "Predictions were frozen before wet-lab return (anti-leakage).",
        "Gate 4 must pass before the master table is appended / the model refit.",
        f"Matched observations: {_fmt(overall.get('n_matched'))}.",
    ]
    return _finish(
        stage="stage5",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )


def write_stage6_report(
    gate: dict[str, Any],
    *,
    metrics: pd.DataFrame | None = None,
    comparisons: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Ablation RMSE, paired deltas, and evidentiary summary."""
    root = Path(repo_root or REPO_ROOT)
    out = stage_report_dir("stage6", repo_root=root)
    metrics = metrics if metrics is not None else pd.DataFrame()
    comparisons = comparisons if comparisons is not None else pd.DataFrame()

    def panel_rmse(ax) -> None:
        if metrics.empty or "rmse" not in metrics.columns:
            _empty(ax, "No ablation metrics")
            return
        labels = metrics.get(
            "ablation_label",
            metrics.get("ablation_id", metrics.index.astype(str)),
        )
        ax.bar(np.arange(len(metrics)), metrics["rmse"], color=TEAL, width=0.65)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(
            [str(v) for v in labels], rotation=30, ha="right", fontsize=7
        )
        ax.set_ylabel("RMSE")

    def panel_delta(ax) -> None:
        needed = {"delta_rmse", "delta_rmse_ci_low", "delta_rmse_ci_high"}
        if comparisons.empty or not needed.issubset(comparisons.columns):
            _empty(ax, "No paired deltas")
            return
        df = comparisons.dropna(subset=["delta_rmse"]).iloc[::-1]
        y = np.arange(len(df))
        ax.axvline(0.0, color=RULE, lw=1, ls="--")
        ax.errorbar(
            df["delta_rmse"],
            y,
            xerr=[
                df["delta_rmse"] - df["delta_rmse_ci_low"],
                df["delta_rmse_ci_high"] - df["delta_rmse"],
            ],
            fmt="o",
            color=TEAL,
            ecolor=TEAL,
            capsize=3,
        )
        ax.set_yticks(y)
        labels = [
            f"{a} vs {b}"
            for a, b in zip(
                df.get("config_a", df.index.astype(str)),
                df.get("config_b", [""] * len(df)),
                strict=True,
            )
        ]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(r"$\Delta$RMSE (negative favors a)")

    def panel_effect(ax) -> None:
        if comparisons.empty or "cohens_d_abs_error" not in comparisons.columns:
            _empty(ax, "No effect sizes")
            return
        df = comparisons.dropna(subset=["cohens_d_abs_error"])
        labels = [
            f"{a} vs {b}"
            for a, b in zip(df["config_a"], df["config_b"], strict=True)
        ]
        ax.barh(labels, df["cohens_d_abs_error"], color=OCEAN)
        ax.axvline(0.0, color=RULE, lw=1, ls="--")
        ax.set_xlabel("paired Cohen's d")

    kpis = [
        f"{int(len(metrics))} configs",
        f"{int(len(comparisons))} comparisons",
        "evidentiary (does not replace Gates 0-5)",
    ]
    figures = _try_overview(
        title=STAGE_TITLES["stage6"],
        passed=_passed(gate),
        kpis=kpis,
        panels=[
            ("RMSE by configuration", panel_rmse),
            ("Paired RMSE delta (95% CI)", panel_delta),
            ("Effect sizes", panel_effect),
        ],
        checks=_checks(gate),
        path=out / "overview.png",
    )
    stats = {
        "n_configs": int(len(metrics)),
        "n_comparisons": int(len(comparisons)),
        "best_rmse": (
            float(metrics["rmse"].min())
            if not metrics.empty and "rmse" in metrics.columns
            else None
        ),
    }
    observations = [
        "Stage 6 is evidentiary and does not replace operational Gates 0-5.",
        "Paired tests use the same Stage-0 splits and random seed.",
        "Negative $\\Delta$RMSE favors the listed config a.",
    ]
    return _finish(
        stage="stage6",
        repo_root=root,
        out_dir=out,
        gate=gate,
        stats=stats,
        observations=observations,
        figures=figures,
    )
