"""Shared ``propose(observed, candidate_pool, surrogate, batch_size)`` interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import (
    FusedSurrogate,
    SurrogatePrediction,
)


class SearchPolicy(Protocol):
    name: str

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Propose the next experimental batch from the candidate pool.

        Parameters
        ----------
        observed : pd.DataFrame
            Constructs with measured fitness used to condition the surrogate.
        candidate_pool : pd.DataFrame
            Unmeasured candidates eligible for selection.
        surrogate : FusedSurrogate
            Fitted surrogate model providing predictions and uncertainties.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Subset of ``candidate_pool`` rows chosen for the next round,
            typically with an ``acquisition`` score column attached.
        """


def predict_pool(surrogate: FusedSurrogate, pool: pd.DataFrame) -> SurrogatePrediction:
    """Run surrogate predictions on every row in a candidate pool.

    Parameters
    ----------
    surrogate : FusedSurrogate
        Fitted surrogate model.
    pool : pd.DataFrame
        Candidate constructs to score.

    Returns
    -------
    SurrogatePrediction
        Bundle of predictive means, standard deviations, and component scores.
    """
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
    """Compute UCB acquisition scores from a surrogate prediction.

    Parameters
    ----------
    pred : SurrogatePrediction
        Surrogate output containing fitness mean and GP standard deviation.
    kappa : float
        UCB exploration coefficient.
    lambda_structure : float
        Weight on structural uncertainty in effective sigma.
    lambda_physics : float
        Weight on physics uncertainty in effective sigma.
    structural_uncertainty : np.ndarray or float, optional
        Per-candidate structural uncertainty (default 0.0).
    physics_uncertainty : np.ndarray or float, optional
        Per-candidate physics uncertainty (default 0.0).

    Returns
    -------
    np.ndarray
        UCB scores ``μ + κ σ_eff`` for each candidate.
    """
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
    """Copy a candidate pool and append surrogate prediction columns.

    Parameters
    ----------
    pool : pd.DataFrame
        Candidate constructs.
    pred : SurrogatePrediction
        Surrogate output from :func:`predict_pool`.

    Returns
    -------
    pd.DataFrame
        Copy of ``pool`` with ``pred_fitness_mean``, ``pred_fitness_std``,
        ``pred_physics_mean``, and ``pred_gp_residual_mean`` columns added.
    """
    out = pool.copy()
    out["pred_fitness_mean"] = pred.fitness_mean
    out["pred_fitness_std"] = pred.fitness_std
    out["pred_physics_mean"] = pred.physics_mean
    out["pred_gp_residual_mean"] = pred.gp_residual_mean
    for name, vals in (pred.phenotype_mean or {}).items():
        out[f"pred_{name}_mean"] = vals
        if name in pred.phenotype_std:
            out[f"pred_{name}_std"] = pred.phenotype_std[name]
    return out


def build_search_policies(
    search_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Instantiate Stage-4 search policies from ``search.yaml``.

    Parameters
    ----------
    search_cfg : dict
        Parsed search configuration.
    seed : int
        Random seed for stochastic policies.

    Returns
    -------
    dict of str to SearchPolicy
        Mapping from strategy name to policy instance.
    """
    from biosensor_priors.stage4_search.adalead import AdaLeadPolicy
    from biosensor_priors.stage4_search.bo import BOPolicy
    from biosensor_priors.stage4_search.mcmc import MCMCPolicy
    from biosensor_priors.stage4_search.random_search import RandomSearchPolicy
    from biosensor_priors.stage4_search.thompson import ThompsonPolicy

    adalead_cfg = search_cfg.get("adalead", {})
    unc = search_cfg.get("uncertainty", {})
    th = search_cfg.get("thompson", {})
    return {
        "random": RandomSearchPolicy(
            candidate_m=int(search_cfg.get("candidate_m", 256)),
            random_seed=seed,
        ),
        "adalead": AdaLeadPolicy(
            kappa=float(adalead_cfg.get("kappa", 0.05)),
            epsilon=adalead_cfg.get("epsilon"),
            parent_mode=str(adalead_cfg.get("parent_mode", "relative_kappa")),
        ),
        "mcmc": MCMCPolicy(
            temperature=float(search_cfg.get("mcmc", {}).get("temperature", 0.10)),
            n_steps=int(search_cfg.get("mcmc", {}).get("n_steps", 300)),
            n_chains=int(search_cfg.get("mcmc", {}).get("n_chains", 8)),
            candidate_m=int(search_cfg.get("candidate_m", 256)),
            random_seed=seed,
        ),
        "bo": BOPolicy(
            kappa=float(search_cfg.get("ucb", {}).get("kappa", 1.5)),
            use_effective_uncertainty=bool(unc.get("use_effective", True)),
            lambda_structure=float(unc.get("lambda_structure", 1.0)),
            lambda_physics=float(unc.get("lambda_physics", 1.0)),
        ),
        "thompson": ThompsonPolicy(
            random_seed=seed,
            primary=str(th.get("primary", "fitness")),
            constraints=th.get("constraints") or {},
            min_feasibility=float(th.get("min_feasibility", 0.0)),
        ),
    }
