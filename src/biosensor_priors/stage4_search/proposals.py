"""Exploit / explore proposal lists for a Stage-4 design-space run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.edits import DEFAULT_COSTS
from biosensor_priors.stage3_surrogate.phenotypes import constraint_probability
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.policy import attach_predictions, predict_pool


def _constraint_mask(
    scored: pd.DataFrame,
    pred,
    constraints: dict[str, dict[str, Any]],
) -> np.ndarray:
    feasible = np.ones(len(scored), dtype=bool)
    for name, spec in (constraints or {}).items():
        minimum = float(spec.get("min", 0.0))
        min_prob = float(spec.get("min_prob", 0.0))
        mean_col = f"pred_{name}_mean"
        std_col = f"pred_{name}_std"
        if mean_col not in scored.columns and name not in pred.phenotype_mean:
            continue
        mean = (
            pred.phenotype_mean[name]
            if name in pred.phenotype_mean
            else scored[mean_col].to_numpy(dtype=float)
        )
        std = pred.phenotype_std.get(
            name,
            scored[std_col].to_numpy(dtype=float)
            if std_col in scored.columns
            else np.full(len(scored), 1e-6),
        )
        feasible &= np.asarray(mean, dtype=float) >= minimum
        if min_prob > 0:
            p = constraint_probability(mean, std, minimum=minimum)
            scored[f"prob_{name}"] = p
            feasible &= p >= min_prob
    return feasible


def split_exploit_explore(
    observed: pd.DataFrame,
    pool: pd.DataFrame,
    surrogate: FusedSurrogate,
    *,
    search_cfg: dict[str, Any] | None = None,
    fitness_cfg: dict[str, Any] | None = None,
    exploit_size: int | None = None,
    explore_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank an exploit batch (improve F) and an explore batch (reduce uncertainty).

    Exploit applies Thompson phenotype floors and a mutation-cost filter:
    ``net = mu - lambda * cost`` must meet or beat the best observed fitness
    when any such candidate exists. Explore ranks leftover candidates by
    predictive standard deviation and does not apply the cost filter.

    Parameters
    ----------
    observed : pd.DataFrame
        Labeled training constructs with a ``fitness`` column.
    pool : pd.DataFrame
        Prefiltered design-space candidates.
    surrogate : FusedSurrogate
        Fitted Stage-3/4 surrogate.
    search_cfg : dict or None, optional
        ``search.yaml`` (``thompson.constraints``, ``proposals`` sizes).
    fitness_cfg : dict or None, optional
        ``fitness.yaml`` (``mutation_cost.lambda``).
    exploit_size : int or None, optional
        Override exploit batch length.
    explore_size : int or None, optional
        Override explore batch length.

    Returns
    -------
    exploit : pd.DataFrame
        Constructs proposed to improve function (``proposal_role=exploit``).
    explore : pd.DataFrame
        Constructs proposed to reduce design-space uncertainty
        (``proposal_role=explore``).
    """
    search_cfg = search_cfg or {}
    fitness_cfg = fitness_cfg or {}
    prop_cfg = search_cfg.get("proposals") or {}
    n_exploit = int(
        exploit_size
        or prop_cfg.get("exploit_size")
        or search_cfg.get("batch_size", 8)
    )
    n_explore = int(
        explore_size
        or prop_cfg.get("explore_size")
        or search_cfg.get("batch_size", 8)
    )
    if pool.empty:
        empty = pool.iloc[0:0].copy()
        return empty, empty

    pred = predict_pool(surrogate, pool)
    scored = attach_predictions(pool, pred)
    cost = (
        pd.to_numeric(scored["mutation_cost"], errors="coerce")
        if "mutation_cost" in scored.columns
        else pd.Series(0.0, index=scored.index)
    )
    cost = cost.fillna(0.0).to_numpy(dtype=float)
    lam = float(
        (fitness_cfg.get("mutation_cost") or {}).get("lambda", DEFAULT_COSTS["lambda"])
    )
    mu = np.asarray(scored["pred_fitness_mean"], dtype=float)
    net = mu - lam * cost
    scored["mutation_cost"] = cost
    scored["net_fitness"] = net
    scored["cost_lambda"] = lam

    best_obs = float(
        pd.to_numeric(observed["fitness"], errors="coerce").max()
        if "fitness" in observed.columns and not observed.empty
        else 0.0
    )
    if not np.isfinite(best_obs):
        best_obs = 0.0
    scored["best_observed_fitness"] = best_obs
    compensates = np.isfinite(net) & (net >= best_obs)
    scored["compensates_cost"] = compensates

    constraints = (search_cfg.get("thompson") or {}).get("constraints") or {}
    feasible = _constraint_mask(scored, pred, constraints)
    scored["proposal_feasible"] = feasible

    exploit_ok = feasible & compensates
    if not np.any(exploit_ok):
        exploit_ok = feasible
    exploit = scored.loc[exploit_ok].copy()
    if not exploit.empty:
        exploit = exploit.sort_values("net_fitness", ascending=False).head(n_exploit)
    exploit = exploit.copy()
    exploit["proposal_role"] = "exploit"
    exploit["selection_algorithm"] = "exploit"
    exploit["selection_rank"] = range(1, len(exploit) + 1)

    taken = set(exploit["construct_id"].astype(str)) if not exploit.empty else set()
    explore = scored.copy()
    if taken:
        explore = explore.loc[~explore["construct_id"].astype(str).isin(taken)]
    std = pd.to_numeric(explore.get("pred_fitness_std"), errors="coerce")
    explore = explore.assign(_explore_score=std)
    if not explore.empty:
        explore = explore.sort_values("_explore_score", ascending=False).head(n_explore)
        explore = explore.drop(columns=["_explore_score"])
    explore = explore.copy()
    explore["proposal_role"] = "explore"
    explore["selection_algorithm"] = "explore"
    explore["selection_rank"] = range(1, len(explore) + 1)
    return exploit, explore


def write_stage4_proposals(
    exploit: pd.DataFrame,
    explore: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    """Write the primary Stage-4 proposal CSVs.

    Parameters
    ----------
    exploit : pd.DataFrame
        Exploit batch.
    explore : pd.DataFrame
        Explore batch.
    out_dir : Path
        Stage-4 output directory.

    Returns
    -------
    dict of str to str
        Relative-looking filenames written under ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exploit_path = out_dir / "proposals_exploit.csv"
    explore_path = out_dir / "proposals_explore.csv"
    exploit.to_csv(exploit_path, index=False)
    explore.to_csv(explore_path, index=False)
    return {
        "proposals_exploit": str(exploit_path),
        "proposals_explore": str(explore_path),
    }
