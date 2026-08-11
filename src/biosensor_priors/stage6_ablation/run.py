"""Stage 6 orchestration: ablation matrix → statistics → report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import write_manifest
from biosensor_priors.stage3_surrogate.cross_validate import ensure_splits_for_fitness
from biosensor_priors.stage6_ablation.experiments import (
    load_ablation_matrix,
    run_ablation_matrix,
)
from biosensor_priors.stage6_ablation.report import write_report
from biosensor_priors.stage6_ablation.statistics import run_ablation_statistics


def _load_master(root: Path) -> pd.DataFrame:
    """Load the Stage-0 experiment master table from processed data.

    Parameters
    ----------
    root : Path
        Repository root containing ``data/processed/``.

    Returns
    -------
    pd.DataFrame
        Experiment master table (pickle preferred, parquet fallback).

    Raises
    ------
    FileNotFoundError
        If neither ``experiment_master.pkl`` nor ``experiment_master.parquet``
        exists.
    """
    pkl = root / "data" / "processed" / "experiment_master.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    parquet = root / "data" / "processed" / "experiment_master.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(
        "experiment_master not found. Run Stage 0 first "
        "(python -m biosensor_priors.stage0_ground_truth.load_experiments)."
    )


def run_stage6(
    *,
    repo_root: Path | None = None,
    pairwise: bool = False,
    n_bootstrap: int | None = None,
) -> dict[str, Any]:
    """Run the full Stage-6 ablation pipeline end to end.

    Executes the ablation matrix on shared Stage-0 splits and seeds, then
    paired statistics and automatic report generation.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root. Defaults to :data:`REPO_ROOT`.
    pairwise : bool, default False
        When ``True``, run all pairwise config comparisons instead of each
        config versus the reference.
    n_bootstrap : int, optional
        Bootstrap replicates for paired tests. Taken from ablation YAML when
        ``None``.

    Returns
    -------
    dict[str, Any]
        Keys include ``metrics_table``, ``predictions``, ``statistics``,
        ``artifacts``, ``output_dir``, ``manifest_path``, and ``config_meta``.

    Raises
    ------
    RuntimeError
        If no fitness-labeled constructs are available.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    configs, ablation_cfg = load_ablation_matrix(root / "configs" / "ablation.yaml")

    seed = ablation_cfg.get("random_seed")
    seed = int(pipeline.get("random_seed", 42) if seed is None else seed)
    encoding = ablation_cfg.get("encoding") or thresholds.get("gp", {}).get("encoding", "hybrid")
    encoding = str(encoding)
    n_boot = int(n_bootstrap if n_bootstrap is not None else ablation_cfg.get("n_bootstrap", 1000))
    alpha = float(ablation_cfg.get("alpha", 0.05))
    reference = str(
        ablation_cfg.get("reference_config_id") or "physics_gp_conf_consensus"
    )

    master = _load_master(root)
    fit_df = master[master["fitness"].notna()].copy()
    if fit_df.empty:
        raise RuntimeError("No fitness-labeled constructs for Stage 6.")

    splits_dir = resolve_path(pipeline["paths"]["splits"], root)
    splits = ensure_splits_for_fitness(
        fit_df,
        splits_dir,
        prefer_loco=bool(thresholds.get("gp", {}).get("leave_one_construct_out", True)),
        random_seed=seed,
    )

    score_direction = str(
        thresholds.get("physics", {}).get("score_direction", "more_negative_is_better")
    )
    matrix = run_ablation_matrix(
        fit_df,
        configs=configs,
        splits=splits,
        encoding=encoding,
        random_seed=seed,
        score_direction=score_direction,
        repo_root=root,
        ablation_settings=ablation_cfg,
    )

    stats = run_ablation_statistics(
        matrix["predictions"],
        reference_config_id=reference
        if reference in {c.id for c in configs}
        else configs[-1].id if len(configs) > 1 else configs[0].id,
        pairwise=pairwise,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )

    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage6"
    artifacts = write_report(
        out_dir=out_dir,
        metrics=matrix["metrics_table"],
        stats_report=stats,
        predictions=matrix["predictions"],
        meta={
            "random_seed": seed,
            "encoding": encoding,
            "n_splits": matrix["n_splits"],
            "n_configs": len(configs),
        },
    )

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage6_manifest.json",
        stage="stage6_ablation",
        inputs={
            "n_fitness_constructs": int(len(fit_df)),
            "n_splits": matrix["n_splits"],
            "ablation_config": "configs/ablation.yaml",
        },
        parameters={
            "random_seed": seed,
            "encoding": encoding,
            "n_bootstrap": n_boot,
            "alpha": alpha,
            "reference_config_id": stats.get("reference_config_id"),
            "pairwise": pairwise,
            "config_ids": [c.id for c in configs],
        },
        outputs={
            "output_dir": str(out_dir.relative_to(root)),
            "report": artifacts.get("report_md"),
            "metrics": artifacts.get("metrics_csv"),
            "comparisons": artifacts.get("comparisons_csv"),
        },
        random_seed=seed,
        gate={
            "passed": True,
            "notes": "Stage 6 is evidentiary; it does not replace Gates 0–5.",
            "n_significant_holm": stats.get("n_significant_holm"),
        },
        notes="Ablation matrix run under identical splits and seeds.",
    )

    # Persist config meta for provenance
    meta_path = out_dir / "ablation_config_meta.json"
    meta_path.write_text(
        json.dumps(matrix["config_meta"], indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "metrics_table": matrix["metrics_table"],
        "predictions": matrix["predictions"],
        "statistics": stats,
        "artifacts": artifacts,
        "output_dir": out_dir,
        "manifest_path": manifest,
        "config_meta": matrix["config_meta"],
    }


def main() -> None:
    """CLI entry point for Stage-6 ablation and scientific reporting."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6 ablation + scientific reporting")
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Full pairwise comparisons (default: each config vs reference)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=None)
    args = parser.parse_args()
    result = run_stage6(pairwise=args.pairwise, n_bootstrap=args.n_bootstrap)
    print(f"Configs: {len(result['config_meta'])}")
    print(f"Prediction rows: {len(result['predictions'])}")
    print(f"Comparisons: {result['statistics'].get('n_comparisons')}")
    print(f"Wrote: {result['output_dir']}")
    print(f"Report: {result['artifacts'].get('report_md')}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
