"""Stage 3 surrogate tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.cross_validate import run_split_evaluation
from biosensor_priors.stage3_surrogate.features import FeatureBuilder
from biosensor_priors.stage3_surrogate.gate3 import evaluate_gate3, summarize_by_model
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage0_ground_truth.splits import generate_leave_one_out_splits


def test_feature_builder_fit_inside_split(stage0_result) -> None:
    master, _ = stage0_result
    df = master[master["fitness"].notna()].copy()
    assert len(df) >= 5
    train = df.iloc[: max(3, len(df) // 2)]
    test = df.iloc[max(3, len(df) // 2) :]
    fb = FeatureBuilder(encoding="hybrid")
    X_train = fb.fit_transform(train)
    X_test = fb.transform(test)
    assert X_train.shape[1] == X_test.shape[1]
    assert fb.means_ is not None
    # Standardization uses train stats only.
    assert np.allclose(np.nanmean(X_train, axis=0), 0.0, atol=1e-5)


def test_fused_surrogate_decomposition(stage0_result) -> None:
    master, _ = stage0_result
    df = master[master["fitness"].notna()].copy()
    model = FusedSurrogate(kind="physics_gp", random_state=0, encoding="hybrid")
    model.fit(df, df["fitness"].to_numpy(dtype=float))
    pred = model.predict(df.head(5))
    assert np.allclose(pred.fitness_mean, pred.physics_mean + pred.gp_residual_mean)
    assert len(pred.fitness_std) == 5


def test_cv_and_gate3_smoke(stage0_result) -> None:
    master, _ = stage0_result
    df = master[master["fitness"].notna()].copy()
    ids = df["construct_id"].astype(str).tolist()
    # Keep runtime small: a few LOCO splits
    splits = generate_leave_one_out_splits(ids, random_seed=0)[:5]
    preds = run_split_evaluation(df, splits, random_seed=0, encoding="hybrid")
    assert set(preds["model_kind"]) == {"physics_only", "gp_zero_mean", "physics_gp"}
    summary = summarize_by_model(preds)
    assert len(summary) == 3
    gate = evaluate_gate3(preds, random_seed=0, n_boot=200)
    assert "operational_passed" not in gate  # raw gate API
    assert "comparisons" in gate
    assert gate["soft_passed_point_rmse"] in (True, False)
