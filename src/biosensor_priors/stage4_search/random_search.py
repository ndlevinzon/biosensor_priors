"""Random solver (BO-EVO SI): parent mutation at rate 1/N → M candidates → sample B."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import build_landscape_view
from biosensor_priors.stage4_search.policy import attach_predictions, predict_pool


class RandomSearchPolicy:
    """
    Paper Random solver adapted to a finite measured / design pool.

    1. Sample a measured parent.
    2. Score unseen pool members by the probability implied by independent
       per-site mutation at rate 1/N.
    3. Draw until M unique candidates are collected.
    4. Randomly select B of those candidates.
    """

    name = "random"

    def __init__(
        self,
        *,
        candidate_m: int = 256,
        mutation_rate: float | None = None,
        random_seed: int = 42,
    ) -> None:
        """Configure random parent-mutation search over a finite pool.

        Parameters
        ----------
        candidate_m : int, optional
            Number of unique candidates to collect before subsampling (default 256).
        mutation_rate : float or None, optional
            Per-site mutation probability; defaults to ``1 / n_sites`` when None.
        random_seed : int, optional
            RNG seed for reproducibility (default 42).
        """
        self.candidate_m = candidate_m
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Draw candidates via parent mutation model, randomly select batch B.

        Parameters
        ----------
        observed : pd.DataFrame
            Measured constructs used as mutation parents.
        candidate_pool : pd.DataFrame
            Unmeasured candidates to sample from.
        surrogate : FusedSurrogate
            Fitted surrogate; predictions are attached for logging parity.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Randomly selected batch of up to ``batch_size`` pool rows.
        """
        if candidate_pool.empty:
            return candidate_pool.copy()

        # Surrogate predictions attached for downstream logging / parity with other policies.
        pred = predict_pool(surrogate, candidate_pool)
        scored_pool = attach_predictions(candidate_pool, pred).reset_index(drop=True)

        obs = observed.reset_index(drop=True)
        if obs.empty:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(len(scored_pool), size=min(batch_size, len(scored_pool)), replace=False)
            out = scored_pool.iloc[list(idx)].copy()
            out["acquisition"] = 0.0
            return out

        combined = pd.concat([obs, scored_pool], ignore_index=True)
        view = build_landscape_view(combined)
        n_obs = len(obs)
        n_pool = len(scored_pool)
        parent_indices = list(range(n_obs))
        pool_indices = list(range(n_obs, n_obs + n_pool))
        sequences = view.sequences
        alphabets = view.site_alphabets
        n_sites = max(1, view.n_sites)
        rate = self.mutation_rate if self.mutation_rate is not None else 1.0 / n_sites

        rng = np.random.default_rng(self.random_seed)
        candidates: list[int] = []
        attempts = 0
        target_m = min(self.candidate_m, n_pool)

        while len(candidates) < target_m and attempts < target_m * 20:
            attempts += 1
            parent = int(rng.choice(parent_indices))
            parent_seq = sequences[parent]
            log_weights = []
            for idx in pool_indices:
                cand_seq = sequences[idx]
                log_p = 0.0
                for site, (aa_p, aa_c) in enumerate(zip(parent_seq, cand_seq, strict=True)):
                    if aa_p == aa_c:
                        log_p += math.log(max(1e-12, 1.0 - rate))
                    else:
                        alts = max(1, len(alphabets[site]) - 1)
                        log_p += math.log(max(1e-12, rate / alts))
                log_weights.append(log_p)

            log_weights_arr = np.asarray(log_weights, dtype=float)
            log_weights_arr -= np.max(log_weights_arr)
            weights = np.exp(log_weights_arr)
            total = weights.sum()
            if not np.isfinite(total) or total <= 0:
                chosen_pool_pos = int(rng.integers(0, n_pool))
            else:
                chosen_pool_pos = int(rng.choice(n_pool, p=weights / total))
            chosen = pool_indices[chosen_pool_pos]
            local_idx = chosen - n_obs
            if local_idx not in candidates:
                candidates.append(local_idx)

        if not candidates:
            candidates = list(range(min(batch_size, n_pool)))

        rng.shuffle(candidates)
        take = candidates[: min(batch_size, len(candidates))]
        out = scored_pool.iloc[take].copy()
        out["acquisition"] = 0.0
        return out
