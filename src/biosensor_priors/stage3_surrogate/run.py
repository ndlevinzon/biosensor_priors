"""Stage 3 orchestration: CV over frozen splits + Gate 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.gate_reports import write_stage3_report
from biosensor_priors.common.provenance import sha256_file, write_manifest
from biosensor_priors.stage0_ground_truth.fitness import FoldFitnessScaler
from biosensor_priors.stage3_surrogate.attach_priors import (
    attach_physics_and_confidence,
    resolve_multi_mutant,
)
from biosensor_priors.stage3_surrogate.calibration import fit_uncertainty_calibration
from biosensor_priors.stage3_surrogate.cross_validate import (
    ensure_splits_for_fitness,
    run_split_evaluation,
    save_cv_predictions,
)
from biosensor_priors.stage3_surrogate.gate3 import evaluate_gate3, summarize_by_model
from biosensor_priors.stage3_surrogate.surrogate import (
    FusedSurrogate,
    surrogate_kwargs_from_cfg,
)


def _load_master(root: Path) -> pd.DataFrame:
    """Load experiment master table from processed data directory.

    Parameters
    ----------
    root : pathlib.Path
        Repository root path.

    Returns
    -------
    pandas.DataFrame
        Experiment master table.

    Raises
    ------
    FileNotFoundError
        When neither pickle nor parquet master file exists.
    """
    processed = root / "data" / "processed"
    pkl = processed / "experiment_master.pkl"
    parquet = processed / "experiment_master.parquet"
    if pkl.exists():
        return pd.read_pickle(pkl)
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(
        "experiment_master not found. Run Stage 0 first "
        "(python -m biosensor_priors.stage0_ground_truth.load_experiments)."
    )


def run_stage3(
    *,
    repo_root: Path | None = None,
    use_confidence_weighting: bool = True,
    prefer_loco: bool = True,
    require_hard_gate: bool = False,
) -> dict[str, Any]:
    """Run Stage 3 end-to-end: CV evaluation, Gate 3, and final fused model fit.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root for config and output paths.
    use_confidence_weighting : bool, optional
        Apply structural confidence weighting to physics features.
    prefer_loco : bool, optional
        Prefer leave-one-construct-out splits when loading/generating.
    require_hard_gate : bool, optional
        When True, only hard Gate 3 pass counts as operational pass.

    Returns
    -------
    dict
        Keys ``predictions``, ``summary``, ``gate``, ``fused``,
        ``manifest_path``, and ``output_dir``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    fitness_cfg = load_yaml(root / "configs" / "fitness.yaml")
    seed = int(pipeline.get("random_seed", 42))
    gp_cfg = thresholds.get("gp", {})
    encoding = str(gp_cfg.get("encoding", "mutation_bag"))
    skw = surrogate_kwargs_from_cfg(gp_cfg, fitness_cfg)
    priors_cfg = thresholds.get("priors", {})
    multi_mutant = resolve_multi_mutant(priors_cfg.get("multi_mutant"))

    master = attach_physics_and_confidence(
        _load_master(root),
        repo_root=root,
        multi_mutant=multi_mutant,
    )
    splits_dir = resolve_path(pipeline["paths"]["splits"], root)
    splits = ensure_splits_for_fitness(
        master,
        splits_dir,
        prefer_loco=prefer_loco or bool(thresholds.get("gp", {}).get("leave_one_construct_out", True)),
        random_seed=seed,
    )

    # Gate 2: refuse silent full physics weight when physics_gate=FAIL
    gate2_path = resolve_path(pipeline["paths"]["physics"], root) / "gate2.json"
    physics_weight_allowed = True
    fused_kind = "physics_gp"
    if gate2_path.exists():
        gate2 = json.loads(gate2_path.read_text(encoding="utf-8"))
        physics_weight_allowed = bool(gate2.get("allow_full_physics_weight", gate2.get("passed")))
        if (
            str(pipeline.get("gates", {}).get("stage2", "")) == "required_for_physics_weight"
            and not physics_weight_allowed
        ):
            # Fall back to GP-only rather than quietly trusting physics.
            use_confidence_weighting = False
            fused_kind = "gp_zero_mean"

    predictions = run_split_evaluation(
        master,
        splits,
        use_confidence_weighting=use_confidence_weighting,
        random_seed=seed,
        encoding=encoding,
        surrogate_kwargs=skw,
    )
    if predictions.empty:
        raise RuntimeError("Stage 3 produced no CV predictions.")

    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_cv_predictions(predictions, out_dir / "cv_predictions.parquet")
    summary = summarize_by_model(predictions)
    summary_path = out_dir / "model_metrics.json"
    summary_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    gate = evaluate_gate3(predictions, random_seed=seed)
    # Interim GP-only path: hard statistical gate may be underpowered; record both.
    gate_passed = bool(gate["passed"] or (not require_hard_gate and gate["soft_passed_point_rmse"]))
    gate["operational_passed"] = gate_passed
    gate["physics_weight_allowed"] = physics_weight_allowed
    gate["fused_kind"] = fused_kind

    # Fit final fused model on all fitness rows for Stage 4 (train-only label scale).
    fit_df = master[master["fitness"].notna()].copy()
    fit_df = FoldFitnessScaler(
        weights=fitness_cfg.get("weights"),
        min_components=int(fitness_cfg.get("min_components", 2)),
    ).fit_transform(fit_df)
    fit_df = fit_df[fit_df["fitness"].notna()].copy()
    fused = FusedSurrogate(
        kind=fused_kind,  # type: ignore[arg-type]
        use_confidence_weighting=use_confidence_weighting,
        random_state=seed,
        **skw,
    )
    fused.fit(fit_df, fit_df["fitness"].to_numpy(dtype=float))
    fused_rows = predictions[predictions["model_kind"] == fused_kind]
    if not fused_rows.empty:
        fused.calibrator_ = fit_uncertainty_calibration(fused_rows)
        cal_path = out_dir / "uncertainty_calibration.json"
        cal_path.write_text(
            json.dumps(fused.calibrator_.as_dict(), indent=2), encoding="utf-8"
        )
    model_meta = fused.metadata()
    model_meta["physics_weight_allowed"] = physics_weight_allowed
    meta_path = out_dir / "fused_model_metadata.json"
    meta_path.write_text(json.dumps(model_meta, indent=2, default=str), encoding="utf-8")

    # Persist a lightweight joblib-free snapshot of training IDs for Stage 4 reload.
    train_ids_path = out_dir / "fused_train_construct_ids.json"
    train_ids_path.write_text(
        json.dumps(fit_df["construct_id"].astype(str).tolist(), indent=2),
        encoding="utf-8",
    )

    cal_payload = None
    if fused.calibrator_ is not None:
        cal_payload = fused.calibrator_.as_dict()
    gate_report = write_stage3_report(
        gate,
        predictions=predictions,
        summary=summary,
        calibrator=cal_payload,
        repo_root=root,
    )

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage3_manifest.json",
        stage="stage3_surrogate",
        inputs={
            "experiment_master": {
                "path": "data/processed/experiment_master.pkl",
                "sha256": sha256_file(root / "data/processed/experiment_master.pkl")
                if (root / "data/processed/experiment_master.pkl").exists()
                else None,
            },
            "n_splits": len(splits),
            "gate2_path": str(gate2_path.relative_to(root)) if gate2_path.exists() else None,
        },
        parameters={
            "use_confidence_weighting": use_confidence_weighting,
            "prefer_loco": prefer_loco,
            "random_seed": seed,
            "gp": thresholds.get("gp", {}),
            "fused_kind": fused_kind,
            "physics_weight_allowed": physics_weight_allowed,
        },
        outputs={
            "cv_predictions": {"path": str(pred_path.relative_to(root)), "sha256": sha256_file(pred_path)},
            "metrics": {"path": str(summary_path.relative_to(root))},
            "fused_metadata": {"path": str(meta_path.relative_to(root))},
            "gate_report": gate_report,
        },
        random_seed=seed,
        gate=gate,
        notes=(
            "Physics joined when Stage 2 tables exist; missing confidence is 0 not 1. "
            "CV labels use train-fold percentiles. Gate 2 FAIL forces gp_zero_mean."
        ),
    )

    return {
        "predictions": predictions,
        "summary": summary,
        "gate": gate,
        "fused": fused,
        "manifest_path": manifest,
        "output_dir": out_dir,
    }


def main() -> None:
    """CLI entry point for Stage 3 surrogate training and evaluation.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints metrics and gate status to stdout.
    """
    result = run_stage3()
    print("Stage 3 metrics:")
    print(result["summary"].to_string(index=False))
    print(f"Gate operational_passed={result['gate'].get('operational_passed')}")
    print(f"Hard gate passed={result['gate'].get('passed')}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
