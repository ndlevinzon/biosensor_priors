"""Stage 5 prospective wet-lab loop tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biosensor_priors.stage0_ground_truth.clean import load_raw_experimental_workbook
from biosensor_priors.stage5_prospective.freeze_predictions import (
    REQUIRED_FREEZE_COLUMNS,
    freeze_predictions,
    load_frozen_predictions,
    verify_freeze_integrity,
)
from biosensor_priors.stage5_prospective.gate4 import evaluate_gate4
from biosensor_priors.stage5_prospective.import_results import (
    append_to_experiment_master,
    clean_new_results,
)
from biosensor_priors.stage5_prospective.prospective_validation import prospective_validation
from biosensor_priors.stage5_prospective.update_model import (
    append_physics_weights_row,
    generate_next_batch,
    refit_surrogate,
)
from biosensor_priors.common.config import load_fitness_config, load_pipeline_config, resolve_path


def _toy_batch(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mu = rng.normal(0.5, 0.1, size=n)
    sigma = rng.uniform(0.05, 0.15, size=n)
    return pd.DataFrame(
        {
            "construct_id": [f"cand_{i}" for i in range(n)],
            "pred_fitness_mean": mu,
            "pred_fitness_std": sigma,
            "pred_physics_mean": mu * 0.3,
            "pred_gp_residual_mean": mu * 0.7,
            "structural_confidence": rng.uniform(0.5, 1.0, size=n),
            "search_strategy": ["bo"] * (n // 2) + ["adalead"] * (n - n // 2),
        }
    )


def test_freeze_predictions_immutable(tmp_path: Path) -> None:
    batch = _toy_batch()
    meta = freeze_predictions(batch, rounds_dir=tmp_path, round_id=3)
    assert meta["immutable"] is True
    assert (tmp_path / "round_03_predictions.parquet").exists()
    assert (tmp_path / "round_03_predictions.sha256").exists()

    frozen = load_frozen_predictions(tmp_path, 3)
    for col in REQUIRED_FREEZE_COLUMNS:
        assert col in frozen.columns

    integrity = verify_freeze_integrity(tmp_path, 3)
    assert integrity["ok"] is True

    with pytest.raises(FileExistsError):
        freeze_predictions(batch, rounds_dir=tmp_path, round_id=3)


def test_prospective_validation_metrics() -> None:
    frozen = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "predicted_fitness": [0.1, 0.2, 0.3, 0.4, 0.5],
            "ci95_low": [0.0, 0.1, 0.2, 0.3, 0.4],
            "ci95_high": [0.2, 0.3, 0.4, 0.5, 0.6],
            "physics_component": [0.05, 0.1, 0.15, 0.2, 0.25],
            "selection_algorithm": ["bo", "bo", "adalead", "adalead", "bo"],
        }
    )
    obs = pd.DataFrame(
        {
            "construct_id": ["a", "b", "c", "d", "e"],
            "fitness": [0.12, 0.18, 0.35, 0.38, 0.55],
        }
    )
    report = prospective_validation(frozen, obs, prior_best_fitness=0.3, top_k=2)
    assert report["passed"]
    overall = report["overall"]
    assert overall["n_matched"] == 5
    assert np.isfinite(overall["pearson"])
    assert np.isfinite(overall["spearman"])
    assert np.isfinite(overall["rmse"])
    assert np.isfinite(overall["mae"])
    assert 0.0 <= overall["interval_coverage_95"] <= 1.0
    assert overall["best_fitness_found"] == pytest.approx(0.55)
    assert report["by_algorithm"]
    assert "physics_vs_obs_pearson" in report["physics_revalidation"]


def test_import_uses_stage0_cleaning(stage0_result, tmp_path: Path) -> None:
    master, _ = stage0_result
    from biosensor_priors.common.config import REPO_ROOT

    root = REPO_ROOT
    pipeline = load_pipeline_config()
    fitness = load_fitness_config()
    workbook = resolve_path(pipeline["paths"]["experimental"], root) / pipeline["experimental"][
        "workbook"
    ]
    versions = pd.read_pickle(
        resolve_path(pipeline["paths"]["constructs"], root)
        / pipeline["constructs"]["versions_pickle"]
    )
    raw = load_raw_experimental_workbook(workbook)
    # Re-ingest a small subset through the same cleaning path as Stage 0
    subset = raw.head(5).copy()
    cleaned = clean_new_results(
        subset,
        versions=versions,
        fitness_cfg=fitness,
        pipeline_cfg=pipeline,
        experimental_round=3,
    )
    assert "fitness" in cleaned.columns
    assert "construct_id" in cleaned.columns
    assert (cleaned["experimental_round"] == 3).all()

    master_path = tmp_path / "experiment_master.parquet"
    # Seed with a tiny master slice
    seed = master.head(3).copy()
    seed.to_pickle(master_path.with_suffix(".pkl"))
    combined = append_to_experiment_master(
        cleaned,
        master_path=master_path,
        master_pickle_path=master_path.with_suffix(".pkl"),
    )
    assert len(combined) >= len(seed)


def test_gate4_and_weight_history(tmp_path: Path, stage0_result) -> None:
    master, _ = stage0_result
    batch = _toy_batch(4)
    freeze_predictions(batch, rounds_dir=tmp_path, round_id=1)
    frozen = load_frozen_predictions(tmp_path, 1)
    obs = pd.DataFrame(
        {
            "construct_id": frozen["candidate_id"].tolist(),
            "fitness": frozen["predicted_fitness"].to_numpy(dtype=float)
            + np.array([0.01, -0.02, 0.03, 0.0]),
        }
    )
    validation = prospective_validation(frozen, obs, prior_best_fitness=0.0)
    gate = evaluate_gate4(validation, rounds_dir=tmp_path, round_id=1)
    assert gate["passed"] is True

    hist = append_physics_weights_row(
        tmp_path / "physics_weights_by_round.csv",
        round_id=1,
        weights={"rif_ac": 0.1, "rpx": -0.2, "delta_rif_sel": 0.05, "intercept": 0.0, "mode": "physics_linear"},
    )
    assert list(hist.columns)[:4] == ["Round", "w_RIF_Ac", "w_RPX", "w_ΔRIF"]

    fit = master[master["fitness"].notna()].copy()
    surrogate, meta = refit_surrogate(fit, encoding="hybrid", random_seed=0)
    assert "physics_weights" in meta
    pool = fit.sample(n=min(6, len(fit)), random_state=0)
    nxt = generate_next_batch(
        master=fit,
        candidate_pool=pool,
        surrogate=surrogate,
        search_cfg={"batch_size": 3, "candidate_m": 32, "mcmc": {"n_steps": 10, "n_chains": 2}},
        strategy="bo",
        random_seed=0,
    )
    assert len(nxt) <= 3
    assert "selection_algorithm" in nxt.columns
    assert "selection_rank" in nxt.columns
