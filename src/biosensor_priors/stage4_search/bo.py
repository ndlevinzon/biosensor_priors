"""Batched enumerative BO (BO-EVO SI): exhaustive UCB → top B."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import top_b_by_score
from biosensor_priors.stage4_search.policy import (
    attach_predictions,
    predict_pool,
    scores_from_prediction,
)


class BOPolicy:
    """
    Enumerative GP-UCB over the entire candidate pool.

    Paper default: UCB = μ + κσ with GPR surrogate; propose top B.
    Optional σ_eff terms (structure/physics) remain available as extensions.
    """

    name = "bo"

    def __init__(
        self,
        *,
        kappa: float = 1.5,
        lambda_structure: float = 0.0,
        lambda_physics: float = 0.0,
        use_effective_uncertainty: bool = False,
    ) -> None:
        """Configure GP-UCB batch selection over a finite candidate pool.

        Parameters
        ----------
        kappa : float, optional
            UCB exploration coefficient (default 1.5).
        lambda_structure : float, optional
            Weight on structural uncertainty in effective sigma (default 0.0).
        lambda_physics : float, optional
            Weight on physics uncertainty in effective sigma (default 0.0).
        use_effective_uncertainty : bool, optional
            When True, use :func:`~biosensor_priors.stage4_search.acquisition.sigma_effective`
            instead of GP sigma alone (default False).

        Returns
        -------
        None
        """
        self.kappa = kappa
        self.lambda_structure = lambda_structure
        self.lambda_physics = lambda_physics
        self.use_effective_uncertainty = use_effective_uncertainty

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Propose top-B candidates by GP-UCB over the entire pool.

        Parameters
        ----------
        observed : pd.DataFrame
            Measured constructs (unused for pure UCB ranking but kept for API parity).
        candidate_pool : pd.DataFrame
            Unmeasured candidates to rank.
        surrogate : FusedSurrogate
            Fitted surrogate providing predictive mean and uncertainty.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Top ``batch_size`` pool rows sorted by acquisition score.
        """
        _ = observed
        if candidate_pool.empty:
            return candidate_pool.copy()

        pred = predict_pool(surrogate, candidate_pool)
        scored = attach_predictions(candidate_pool, pred)

        cal = getattr(surrogate, "calibrator_", None)
        if cal is not None:
            from biosensor_priors.stage3_surrogate.calibration import (
                structural_and_physics_sigma,
            )

            ss, sp = structural_and_physics_sigma(scored)
            sig = cal.sigma_calibrated(pred.fitness_std, ss, sp)
            scored["acquisition"] = pred.fitness_mean + float(self.kappa) * sig
        elif self.use_effective_uncertainty:
            if "structural_confidence" in scored.columns:
                conf = pd.to_numeric(scored["structural_confidence"], errors="coerce").fillna(1.0)
            else:
                conf = pd.Series(1.0, index=scored.index)
            struct_unc = (1.0 - conf).to_numpy(dtype=float)
            phys_unc = np.zeros(len(scored), dtype=float)
            if "physics_score_std" in scored.columns:
                phys_unc = (
                    pd.to_numeric(scored["physics_score_std"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            scored["acquisition"] = scores_from_prediction(
                pred,
                kappa=self.kappa,
                lambda_structure=self.lambda_structure,
                lambda_physics=self.lambda_physics,
                structural_uncertainty=struct_unc,
                physics_uncertainty=phys_unc,
            )
        else:
            # Paper UCB: μ + κσ_GP
            scored["acquisition"] = pred.fitness_mean + float(self.kappa) * pred.fitness_std

        return top_b_by_score(scored, "acquisition", batch_size)
