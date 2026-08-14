"""Thompson sampling over the fused posterior (batch: independent draws, top B)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.phenotypes import constraint_probability
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import top_b_by_score
from biosensor_priors.stage4_search.policy import attach_predictions, predict_pool


class ThompsonPolicy:
    """Batch Thompson sampling: one posterior draw per candidate, take top B.

    Optional phenotype constraints (affinity / brightness) reject draws that
    fall below preregistered minima. If every candidate is infeasible, rank
    by constraint probability × sampled primary score.
    """

    name = "thompson"

    def __init__(
        self,
        *,
        random_seed: int = 42,
        primary: str = "fitness",
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
        min_feasibility: float = 0.0,
    ) -> None:
        """Configure Thompson sampling.

        Parameters
        ----------
        random_seed : int, optional
            RNG seed (default 42).
        primary : str, optional
            ``fitness`` or a phenotype name used to rank feasible draws
            (default ``fitness``; ``selectivity`` uses that head when present).
        constraints : mapping, optional
            ``{phenotype: {min: float, min_prob: float}}``.
        min_feasibility : float, optional
            Soft floor on joint constraint probability when ranking
            infeasible pools.
        """
        self.random_seed = int(random_seed)
        self.primary = str(primary)
        self.constraints = {k: dict(v) for k, v in (constraints or {}).items()}
        self.min_feasibility = float(min_feasibility)

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Propose top-B candidates by a single posterior draw.

        Parameters
        ----------
        observed : pd.DataFrame
            Measured constructs (unused except API parity).
        candidate_pool : pd.DataFrame
            Unmeasured candidates to rank.
        surrogate : FusedSurrogate
            Fitted surrogate.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Top ``batch_size`` pool rows sorted by Thompson score.
        """
        _ = observed
        if candidate_pool.empty:
            return candidate_pool.copy()
        rng = np.random.default_rng(self.random_seed)
        pred = predict_pool(surrogate, candidate_pool)
        scored = attach_predictions(candidate_pool, pred)
        fitness_s, pheno_s = surrogate.sample_fitness(candidate_pool, rng)

        primary_s = fitness_s
        if self.primary != "fitness" and self.primary in pheno_s:
            primary_s = pheno_s[self.primary]

        feasible = np.ones(len(scored), dtype=bool)
        joint_prob = np.ones(len(scored), dtype=float)
        for name, spec in self.constraints.items():
            minimum = float(spec.get("min", 0.0))
            min_prob = float(spec.get("min_prob", 0.0))
            if name in pheno_s:
                sample = pheno_s[name]
            elif name in pred.phenotype_mean:
                sample = pred.phenotype_mean[name]
            else:
                continue
            feasible &= sample >= minimum
            if name in pred.phenotype_mean:
                p = constraint_probability(
                    pred.phenotype_mean[name],
                    pred.phenotype_std.get(name, np.full(len(sample), 1e-6)),
                    minimum=minimum,
                )
                if min_prob > 0:
                    feasible &= p >= min_prob
                joint_prob *= p
                scored[f"prob_{name}"] = p

        score = np.asarray(primary_s, dtype=float)
        if not np.any(feasible):
            score = score * np.maximum(joint_prob, self.min_feasibility)
        else:
            score = np.where(feasible, score, -np.inf)
        scored["acquisition"] = score
        scored["thompson_feasible"] = feasible
        scored["constraint_prob"] = joint_prob
        return top_b_by_score(scored, "acquisition", batch_size)
