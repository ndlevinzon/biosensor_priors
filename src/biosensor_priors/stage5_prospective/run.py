"""Stage 5 orchestration: freeze → import → validate → update → next batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.gates import require_gate
from biosensor_priors.common.provenance import sha256_file, write_manifest
from biosensor_priors.stage5_prospective.freeze_predictions import (
    freeze_predictions,
    load_frozen_predictions,
)
from biosensor_priors.stage5_prospective.gate4 import evaluate_gate4
from biosensor_priors.stage5_prospective.import_results import (
    append_to_experiment_master,
    load_and_clean_results_file,
)
from biosensor_priors.stage5_prospective.prospective_validation import prospective_validation
from biosensor_priors.stage5_prospective.update_model import (
    append_physics_weights_row,
    generate_next_batch,
    refit_surrogate,
    rerun_calibration_gates,
    save_model_update_artifacts,
)


def _load_master(root: Path) -> pd.DataFrame:
    pkl = root / "data" / "processed" / "experiment_master.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    return pd.read_parquet(root / "data" / "processed" / "experiment_master.parquet")


def freeze_round_batch(
    batch: pd.DataFrame,
    *,
    round_id: int | str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """5A — freeze predictions for a synthesis round."""
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    rounds_dir = resolve_path(pipeline["paths"]["rounds"], root)
    meta = freeze_predictions(batch, rounds_dir=rounds_dir, round_id=round_id)
    write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / f"stage5_round_{round_id}_freeze.json",
        stage="stage5_freeze",
        inputs={"n_candidates": meta["n_candidates"]},
        parameters={"round_id": round_id},
        outputs={"freeze": meta},
        random_seed=int(pipeline.get("random_seed", 42)),
        gate={"passed": True, "immutable": True},
        notes="Predictions frozen before wet-lab synthesis; file must not be rewritten.",
    )
    return meta


def ingest_and_validate_round(
    results_path: Path,
    *,
    round_id: int | str,
    repo_root: Path | None = None,
    update_model: bool = True,
    next_strategy: str = "bo",
    candidate_pool: pd.DataFrame | None = None,
    master_path: Path | None = None,
    rounds_dir: Path | None = None,
    out_dir: Path | None = None,
    rerun_gates: bool = True,
) -> dict[str, Any]:
    """
    5B–5D: import results via Stage-0 cleaning, validate vs freeze, optionally update model.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    search_cfg = load_yaml(root / "configs" / "search.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    seed = int(pipeline.get("random_seed", 42))
    encoding = str(thresholds.get("gp", {}).get("encoding", "hybrid"))

    rounds_dir = Path(rounds_dir) if rounds_dir is not None else resolve_path(
        pipeline["paths"]["rounds"], root
    )
    out_dir = Path(out_dir) if out_dir is not None else (
        resolve_path(pipeline["paths"]["outputs"], root) / "stage5"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    default_master = resolve_path(pipeline["paths"]["processed"], root) / "experiment_master.parquet"
    master_parquet = Path(master_path) if master_path is not None else default_master
    master_pkl = master_parquet.with_suffix(".pkl")

    if master_pkl.exists():
        master_before = pd.read_pickle(master_pkl)
    elif master_parquet.exists():
        master_before = pd.read_parquet(master_parquet)
    else:
        master_before = _load_master(root)

    prior_best = (
        float(master_before["fitness"].max())
        if "fitness" in master_before.columns and master_before["fitness"].notna().any()
        else None
    )

    # 5B — same cleaning pathway
    new_rows = load_and_clean_results_file(
        results_path,
        repo_root=root,
        experimental_round=round_id,
    )
    new_path = out_dir / f"round_{round_id}_cleaned_results.parquet"
    store = new_rows.copy()
    for col in store.columns:
        if store[col].dtype == object:
            store[col] = store[col].map(lambda x: None if x is None else str(x))
    store.to_parquet(new_path, index=False)

    # 5C — prospective validation against immutable freeze
    frozen = load_frozen_predictions(rounds_dir, round_id)
    validation = prospective_validation(
        frozen,
        new_rows,
        prior_best_fitness=prior_best,
    )
    report = {k: v for k, v in validation.items() if k != "joined"}
    report_path = out_dir / f"round_{round_id}_prospective_validation.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if "joined" in validation:
        validation["joined"].to_csv(out_dir / f"round_{round_id}_joined.csv", index=False)

    gate = evaluate_gate4(validation, rounds_dir=rounds_dir, round_id=round_id)
    gate_policy = str(pipeline.get("gates", {}).get("stage5", "required_before_model_update"))

    result: dict[str, Any] = {
        "round_id": round_id,
        "n_new_rows": int(len(new_rows)),
        "validation": report,
        "gate": gate,
        "freeze_path": str(rounds_dir),
    }

    if update_model:
        if gate_policy == "required_before_model_update":
            require_gate(gate, stage="stage5")

        # Append to authoritative master only after gate
        master = append_to_experiment_master(
            new_rows,
            master_path=master_parquet,
            master_pickle_path=master_pkl,
        )

        surrogate, metadata = refit_surrogate(
            master,
            encoding=encoding,
            random_seed=seed,
        )
        weights = metadata.get("physics_weights", {})
        hist_path = out_dir / "physics_weights_by_round.csv"
        hist = append_physics_weights_row(hist_path, round_id=round_id, weights=weights)

        calibration = None
        if rerun_gates:
            splits_dir = resolve_path(pipeline["paths"]["splits"], root)
            calibration = rerun_calibration_gates(
                master,
                splits_dir=splits_dir if splits_dir.exists() else None,
                encoding=encoding,
                random_seed=seed,
            )

        arts = save_model_update_artifacts(
            out_dir=out_dir,
            round_id=round_id,
            metadata=metadata,
            weights_history=hist,
            calibration_gate=calibration,
        )

        next_batch = None
        if candidate_pool is not None and not candidate_pool.empty:
            next_batch = generate_next_batch(
                master=master,
                candidate_pool=candidate_pool,
                surrogate=surrogate,
                search_cfg=search_cfg,
                strategy=next_strategy,
                random_seed=seed,
            )
            next_batch.to_csv(out_dir / f"round_{round_id}_next_batch_{next_strategy}.csv", index=False)

        result.update(
            {
                "master_n_rows": int(len(master)),
                "model_metadata": metadata,
                "calibration_gate": calibration,
                "artifacts": {k: str(v) for k, v in arts.items()},
                "next_batch": next_batch,
                "surrogate": surrogate,
            }
        )

    manifest_dir = resolve_path(pipeline["paths"]["manifests"], root)
    try:
        validation_rel = str(report_path.relative_to(root))
        cleaned_rel = str(new_path.relative_to(root))
    except ValueError:
        validation_rel = str(report_path)
        cleaned_rel = str(new_path)

    write_manifest(
        manifest_dir / f"stage5_round_{round_id}_manifest.json",
        stage="stage5_prospective",
        inputs={
            "results_path": str(results_path),
            "results_sha256": sha256_file(Path(results_path)) if Path(results_path).exists() else None,
        },
        parameters={"round_id": round_id, "update_model": update_model, "encoding": encoding},
        outputs={
            "validation_report": validation_rel,
            "cleaned_results": cleaned_rel,
        },
        random_seed=seed,
        gate=gate,
        notes="Prospective wet-lab loop: freeze integrity enforced; model update only after Gate 4.",
    )
    return result


def main() -> None:
    """
    CLI helper.

    Examples
    --------
    Freeze a Stage-4 batch CSV::

        python -m biosensor_priors.stage5_prospective.run freeze --round 3 --batch outputs/stage4/batch_design_bo.csv

    Ingest results and update::

        python -m biosensor_priors.stage5_prospective.run ingest --round 3 --results path/to/plate.xlsx
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 5 prospective wet-lab loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze", help="Freeze predictions before synthesis")
    p_freeze.add_argument("--round", required=True)
    p_freeze.add_argument("--batch", required=True, help="CSV/parquet batch with predictions")

    p_ingest = sub.add_parser("ingest", help="Import results, validate, update model")
    p_ingest.add_argument("--round", required=True)
    p_ingest.add_argument("--results", required=True, help="New plate Excel/CSV")
    p_ingest.add_argument("--strategy", default="bo")
    p_ingest.add_argument("--no-update", action="store_true")

    args = parser.parse_args()
    if args.cmd == "freeze":
        path = Path(args.batch)
        if path.suffix.lower() == ".csv":
            batch = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            batch = pd.read_parquet(path)
        else:
            batch = pd.read_pickle(path)
        meta = freeze_round_batch(batch, round_id=args.round)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "ingest":
        result = ingest_and_validate_round(
            Path(args.results),
            round_id=args.round,
            update_model=not args.no_update,
            next_strategy=args.strategy,
        )
        printable = {k: v for k, v in result.items() if k not in {"surrogate", "next_batch"}}
        print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
