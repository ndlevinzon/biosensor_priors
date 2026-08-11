"""AdaLead (Sinai et al. / BO-EVO SI) over a finite measured or design pool."""

from __future__ import annotations

import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import build_landscape_view
from biosensor_priors.stage4_search.policy import attach_predictions, predict_pool


class AdaLeadPolicy:
    """
    Paper AdaLead:

    1. Parents with fitness within (1-κ) of the max measured fitness
       (F ≥ (1-κ)·F_max). Absolute ε mode also supported.
    2. Explore local measured neighbors (Hamming-1, else nearest) and
       recombinants of parent alleles.
    3. Keep children whose GP-predicted fitness exceeds the corresponding parent.
    4. Rank candidates by predicted fitness; propose top B.
    """

    name = "adalead"

    def __init__(
        self,
        *,
        kappa: float = 0.05,
        epsilon: float | None = None,
        parent_mode: str = "relative_kappa",
    ) -> None:
        """Configure AdaLead parent selection and local search.

        Parameters
        ----------
        kappa : float, optional
            Relative fitness band ``(1-κ)·F_max`` for parent selection (default 0.05).
        epsilon : float or None, optional
            Absolute fitness margin for ``absolute_epsilon`` parent mode.
        parent_mode : str, optional
            ``"relative_kappa"`` or ``"absolute_epsilon"`` (default ``"relative_kappa"``).
        """
        self.kappa = kappa
        self.epsilon = epsilon
        self.parent_mode = parent_mode

    def _parent_mask(self, fitness: pd.Series) -> pd.Series:
        """Return boolean mask of constructs eligible as AdaLead parents.

        Parameters
        ----------
        fitness : pd.Series
            Measured fitness values for observed constructs.

        Returns
        -------
        pd.Series
            True for rows meeting the configured parent threshold.
        """
        best = float(fitness.max())
        if self.parent_mode == "absolute_epsilon":
            eps = 0.05 if self.epsilon is None else float(self.epsilon)
            return fitness >= best - eps
        return fitness >= (1.0 - float(self.kappa)) * best

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        """Propose AdaLead children exceeding parent GP predictions.

        Parameters
        ----------
        observed : pd.DataFrame
            Measured constructs defining parent set and fitness landscape.
        candidate_pool : pd.DataFrame
            Unmeasured candidates for local mutation and recombination.
        surrogate : FusedSurrogate
            Fitted surrogate providing GP means on pool rows.
        batch_size : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Top ``batch_size`` AdaLead children ranked by predicted fitness.
        """
        if candidate_pool.empty:
            return candidate_pool.copy()

        pred = predict_pool(surrogate, candidate_pool)
        scored_pool = attach_predictions(candidate_pool, pred).reset_index(drop=True)

        if "fitness" not in observed.columns or observed["fitness"].notna().sum() == 0:
            scored_pool["acquisition"] = scored_pool["pred_fitness_mean"]
            return scored_pool.sort_values("acquisition", ascending=False).head(batch_size)

        obs = observed[observed["fitness"].notna()].reset_index(drop=True)
        parent_rows = obs.loc[self._parent_mask(obs["fitness"])]
        if parent_rows.empty:
            parent_rows = obs.nlargest(1, "fitness")

        combined = pd.concat([obs, scored_pool], ignore_index=True)
        view = build_landscape_view(combined)
        n_obs = len(obs)
        D = view.distance
        sequences = view.sequences

        parent_pos = list(parent_rows.index.astype(int))
        # parent_rows.index refers to obs positions after reset_index
        parent_pos = [int(i) for i in parent_rows.index.tolist()]
        y = obs["fitness"].to_numpy(dtype=float)
        gp_mean = scored_pool["pred_fitness_mean"].to_numpy(dtype=float)
        pool_pos = list(range(n_obs, n_obs + len(scored_pool)))

        candidate_parent: dict[int, int] = {}
        for parent in parent_pos:
            neighbors = [j for j in pool_pos if int(D[parent, j]) == 1]
            if not neighbors:
                dists = sorted(((j, int(D[parent, j])) for j in pool_pos), key=lambda t: t[1])
                neighbors = [j for j, _ in dists[: min(10, len(dists))]]
            for child in neighbors:
                local = child - n_obs
                if gp_mean[local] > y[parent]:
                    prev = candidate_parent.get(local)
                    if prev is None or y[parent] > y[prev]:
                        candidate_parent[local] = parent

        if len(candidate_parent) < batch_size and len(parent_pos) >= 2:
            parent_seqs = [sequences[p] for p in parent_pos]
            allowed = [{seq[j] for seq in parent_seqs} for j in range(len(parent_seqs[0]))]
            best_parent = max(parent_pos, key=lambda p: y[p])
            for child in pool_pos:
                seq = sequences[child]
                if all(seq[j] in allowed[j] for j in range(len(seq))):
                    local = child - n_obs
                    if gp_mean[local] > y[best_parent]:
                        candidate_parent.setdefault(local, best_parent)

        if candidate_parent:
            cand_locals = sorted(candidate_parent, key=lambda i: float(gp_mean[i]), reverse=True)
        else:
            cand_locals = sorted(range(len(scored_pool)), key=lambda i: float(gp_mean[i]), reverse=True)

        for i in sorted(range(len(scored_pool)), key=lambda k: float(gp_mean[k]), reverse=True):
            if len(cand_locals) >= batch_size:
                break
            if i not in cand_locals:
                cand_locals.append(i)

        out = scored_pool.iloc[cand_locals[:batch_size]].copy()
        out["acquisition"] = out["pred_fitness_mean"]
        return out
