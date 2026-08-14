"""Physics prefilter returning PASS / SOFT_FAIL / HARD_FAIL / EXPLORATION_RESERVED."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class PrefilterCategory(str, Enum):
    PASS = "PASS"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"
    EXPLORATION_RESERVED = "EXPLORATION_RESERVED"


def physics_prefilter(
    candidates: pd.DataFrame,
    *,
    score_direction: str = "more_negative_is_better",
    hard_threshold: float | None = None,
    soft_threshold: float | None = None,
    confidence_col: str = "structural_confidence",
    physics_score_col: str = "delta_rif_sel",
    high_confidence_min: float = 0.7,
) -> pd.DataFrame:
    """Categorize candidates by physics score without discarding exploration options.

    Heuristic until Stage-2 thresholds are calibrated:

    * missing physics -> PASS (GP-only path)
    * bad physics + high confidence -> HARD_FAIL
    * bad physics + unknown/low confidence -> EXPLORATION_RESERVED
    * marginal physics -> SOFT_FAIL
    * otherwise PASS

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate table with optional physics and confidence columns.
    score_direction : str, optional
        ``"more_negative_is_better"`` or opposite (default
        ``"more_negative_is_better"``).
    hard_threshold : float or None, optional
        Goodness cutoff for HARD_FAIL vs SOFT_FAIL; inferred from quantiles when None.
    soft_threshold : float or None, optional
        Goodness cutoff for PASS vs SOFT_FAIL; inferred from quantiles when None.
    confidence_col : str, optional
        Column name for structural confidence (default ``"structural_confidence"``).
    physics_score_col : str, optional
        Column name for physics score (default ``"delta_rif_sel"``).
    high_confidence_min : float, optional
        Confidence above which bad physics becomes HARD_FAIL (default 0.7).

    Returns
    -------
    pd.DataFrame
        Copy of ``candidates`` with ``prefilter`` and ``physics_goodness`` columns.
    """
    out = candidates.copy()
    if physics_score_col not in out.columns:
        out["prefilter"] = PrefilterCategory.PASS.value
        return out

    score = pd.to_numeric(out[physics_score_col], errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(score)
    if bool(missing.all()):
        out["prefilter"] = PrefilterCategory.PASS.value
        return out

    if confidence_col in out.columns:
        conf = pd.to_numeric(out[confidence_col], errors="coerce").to_numpy(dtype=float)
        conf = np.where(np.isfinite(conf), conf, 0.0)
    else:
        conf = np.zeros(len(out), dtype=float)

    # Convert to "higher is better" internal score.
    if score_direction == "more_negative_is_better":
        goodness = -score
    else:
        goodness = score

    finite_score = score[~missing]
    if len(finite_score) < 2 or np.allclose(finite_score, finite_score[0]):
        out["prefilter"] = PrefilterCategory.PASS.value
        return out

    finite_good = goodness[np.isfinite(goodness)]
    soft_t = soft_threshold
    hard_t = hard_threshold
    if soft_t is None:
        soft_t = float(np.quantile(finite_good, 0.25))
    if hard_t is None:
        hard_t = float(np.quantile(finite_good, 0.10))

    cats = []
    for g, c, miss in zip(goodness, conf, missing, strict=True):
        if miss:
            cats.append(PrefilterCategory.PASS.value)
        elif g >= soft_t:
            cats.append(PrefilterCategory.PASS.value)
        elif g >= hard_t:
            cats.append(PrefilterCategory.SOFT_FAIL.value)
        elif c >= high_confidence_min:
            cats.append(PrefilterCategory.HARD_FAIL.value)
        else:
            cats.append(PrefilterCategory.EXPLORATION_RESERVED.value)
    out["prefilter"] = cats
    out["physics_goodness"] = goodness
    return out


def select_search_pools(
    candidates: pd.DataFrame,
    *,
    hard_fail_exclude: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split prefiltered candidates into main, exploration, and excluded pools.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate table, optionally already containing a ``prefilter`` column.
    hard_fail_exclude : bool, optional
        When True, HARD_FAIL rows are excluded from the main pool (default True).

    Returns
    -------
    dict of str to pd.DataFrame
        Keys ``"main"``, ``"exploration"``, and ``"excluded"`` mapping to filtered
        subsets.
    """
    df = candidates if "prefilter" in candidates.columns else physics_prefilter(candidates)
    main_mask = df["prefilter"].isin(
        [PrefilterCategory.PASS.value, PrefilterCategory.SOFT_FAIL.value]
    )
    if not hard_fail_exclude:
        main_mask = main_mask | (df["prefilter"] == PrefilterCategory.HARD_FAIL.value)
    exploration = df["prefilter"] == PrefilterCategory.EXPLORATION_RESERVED.value
    return {
        "main": df.loc[main_mask].copy(),
        "exploration": df.loc[exploration].copy(),
        "excluded": df.loc[df["prefilter"] == PrefilterCategory.HARD_FAIL.value].copy(),
    }
