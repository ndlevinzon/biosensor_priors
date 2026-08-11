"""BO-EVO hybrid search strategy (stub for future extension)."""

from __future__ import annotations

import pandas as pd

from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.bo import BOPolicy


class BOEvoPolicy(BOPolicy):
    """Placeholder that currently delegates to GP-UCB (BO)."""

    name = "bo_evo"

    def propose(
        self,
        observed: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        surrogate: FusedSurrogate,
        batch_size: int,
    ) -> pd.DataFrame:
        return super().propose(observed, candidate_pool, surrogate, batch_size)
