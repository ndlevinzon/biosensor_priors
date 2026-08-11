"""MCMC solver (BO-EVO SI): parallel MH, collect M, rank by μ, propose top B."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import build_landscape_view, top_b_by_score
from biosensor_priors.stage4_search.policy import attach_predictions, predict_pool


class MCMCPolicy:
    """
    Parallel Metropolis-Hastings over the candidate+observed graph.

    Target (maximization form used in BO-EVO comparisons / our benchmark):
        π(x) ∝ exp(μ(x) / T)

    Paper SI writes exp(-ŷ/kT) treating ŷ as an energy; we maximize fitness,
    so the sign follows exp(+μ/T) as in the project reference implementation.

    After sampling, collect up to M visited unseen candidates, rank by μ, take B.
    """

    name = "mcmc"

    def __init__(
        self,
        *,
        temperature: float = 0.10,
        n_steps: int = 300,
        n_chains: int = 8,
        candidate_m: int = 256,
        random_seed: int = 42,
    ) -> None:
        """Configure parallel Metropolis-Hastings search over the landscape graph.

        Parameters
        ----------
        temperature : float, optional
            Sampling temperature ``T`` in ``exp(μ / T)`` (default 0.10).
        n_steps : int, optional
            MH steps per chain (default 300).
        n_chains : int, optional
            Number of parallel chains (default 8).
        candidate_m : int, optional
            Maximum unique pool candidates to collect before ranking (default 256).
        random_seed : int, optional
            RNG seed for reproducibility (default 42).

        Returns
        -------
        None
        """
        self.temperature = temperature
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.candidate_m = candidate_m
        self.random_seed = random_seed

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Sample candidates via MH, rank by predicted mean, return top B.

        Parameters
        ----------
        observed : pd.DataFrame
            Measured constructs defining the landscape graph and start states.
        candidate_pool : pd.DataFrame
            Unmeasured candidates eligible for selection.
        surrogate : FusedSurrogate
            Fitted surrogate providing GP means on pool rows.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Top ``batch_size`` visited pool candidates ranked by ``pred_fitness_mean``.
        """
        if candidate_pool.empty:
            return candidate_pool.copy()

        pred = predict_pool(surrogate, candidate_pool)
        scored_pool = attach_predictions(candidate_pool, pred).reset_index(drop=True)
        obs = observed.reset_index(drop=True)

        combined = pd.concat([obs, scored_pool], ignore_index=True)
        view = build_landscape_view(combined)
        n_obs = len(obs)
        n_pool = len(scored_pool)
        D = view.distance

        # Predictive mean on the full combined index: observed use measured fitness
        # as a stand-in start; pool uses GP mean.
        mu = np.zeros(n_obs + n_pool, dtype=float)
        if n_obs and "fitness" in obs.columns:
            mu[:n_obs] = pd.to_numeric(obs["fitness"], errors="coerce").fillna(0.0).to_numpy()
        mu[n_obs:] = scored_pool["pred_fitness_mean"].to_numpy(dtype=float)

        all_idx = list(range(n_obs + n_pool))
        neighbors: list[list[int]] = []
        for i in all_idx:
            d1 = [j for j in all_idx if j != i and D[i, j] == 1]
            if d1:
                neighbors.append(d1)
            else:
                dists = sorted(((j, int(D[i, j])) for j in all_idx if j != i), key=lambda t: t[1])
                neighbors.append([j for j, _ in dists[: min(10, len(dists))]])

        rng = np.random.default_rng(self.random_seed)
        starts = list(np.argsort(mu[: max(n_obs, 1)])[::-1][: self.n_chains]) if n_obs else [0]
        if not starts:
            starts = [0]
        while len(starts) < self.n_chains:
            starts.append(int(rng.integers(0, max(1, n_obs))))

        collected: list[int] = []
        T = max(float(self.temperature), 1e-6)
        for start in starts[: self.n_chains]:
            current = int(start)
            for _ in range(self.n_steps):
                nbrs = neighbors[current]
                if not nbrs:
                    break
                prop = int(rng.choice(nbrs))
                log_alpha = (mu[prop] - mu[current]) / T
                log_alpha += math.log(max(len(neighbors[current]), 1)) - math.log(
                    max(len(neighbors[prop]), 1)
                )
                if math.log(max(rng.random(), 1e-16)) < min(0.0, log_alpha):
                    current = prop
                if current >= n_obs:
                    local = current - n_obs
                    if local not in collected:
                        collected.append(local)
                    if len(collected) >= self.candidate_m:
                        break
            if len(collected) >= self.candidate_m:
                break

        if not collected:
            collected = list(range(n_pool))

        cand = scored_pool.iloc[collected].copy()
        cand["acquisition"] = cand["pred_fitness_mean"]
        return top_b_by_score(cand, "acquisition", batch_size)
