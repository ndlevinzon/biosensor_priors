"""Shared ``propose(observed, candidate_pool, surrogate, batch_size)`` interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate, SurrogatePrediction


class SearchPolicy(Protocol):
    name: str

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Return a batch of candidates (rows from ``candidate_pool``)."""


def predict_pool(surrogate: FusedSurrogate, pool: pd.DataFrame) -> SurrogatePrediction:
    return surrogate.predict(pool)


def scores_from_prediction(
    pred: SurrogatePrediction,
    *,
    kappa: float,
    lambda_structure: float,
    lambda_physics: float,
    structural_uncertainty: np.ndarray | float = 0.0,
    physics_uncertainty: np.ndarray | float = 0.0,
) -> np.ndarray:
    from biosensor_priors.stage4_search.acquisition import sigma_effective, ucb

    sig = sigma_effective(
        pred.fitness_std,
        structural_uncertainty,
        physics_uncertainty,
        lambda_structure=lambda_structure,
        lambda_physics=lambda_physics,
    )
    return ucb(pred.fitness_mean, sig, kappa=kappa)


def attach_predictions(pool: pd.DataFrame, pred: SurrogatePrediction) -> pd.DataFrame:
    out = pool.copy()
    out["pred_fitness_mean"] = pred.fitness_mean
    out["pred_fitness_std"] = pred.fitness_std
    out["pred_physics_mean"] = pred.physics_mean
    out["pred_gp_residual_mean"] = pred.gp_residual_mean
    return out
