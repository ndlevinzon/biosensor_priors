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
    """
    Categorize candidates without silently deleting exploration options.

    Heuristic until Stage-2 thresholds are calibrated:
      * missing/zero physics → PASS (GP-only path)
      * bad physics + high confidence → HARD_FAIL
      * bad physics + low confidence → EXPLORATION_RESERVED
      * marginal physics → SOFT_FAIL
      * otherwise PASS
    """
    out = candidates.copy()
    if physics_score_col not in out.columns:
        out["prefilter"] = PrefilterCategory.PASS.value
        return out

    score = pd.to_numeric(out[physics_score_col], errors="coerce").fillna(0.0).to_numpy()
    conf = (
        pd.to_numeric(out.get(confidence_col, 1.0), errors="coerce").fillna(1.0).to_numpy()
        if confidence_col in out.columns
        else np.ones(len(out))
    )

    # Convert to "higher is better" internal score.
    if score_direction == "more_negative_is_better":
        goodness = -score
    else:
        goodness = score

    # Default thresholds from empirical quantiles if unset and physics varies.
    if np.allclose(score, score[0]):
        out["prefilter"] = PrefilterCategory.PASS.value
        return out

    soft_t = soft_threshold
    hard_t = hard_threshold
    if soft_t is None:
        soft_t = float(np.quantile(goodness, 0.25))
    if hard_t is None:
        hard_t = float(np.quantile(goodness, 0.10))

    cats = []
    for g, c in zip(goodness, conf, strict=True):
        if g >= soft_t:
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
    """Split into main pool vs exploration pool."""
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
