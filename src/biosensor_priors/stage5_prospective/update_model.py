"""Append data, refit physics/GP, recalibrate gates, propose next batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT
from biosensor_priors.stage0_ground_truth.fitness import FoldFitnessScaler
from biosensor_priors.stage3_surrogate.attach_priors import (
    attach_physics_and_confidence,
)
from biosensor_priors.stage3_surrogate.cross_validate import (
    ensure_splits_for_fitness,
    run_split_evaluation,
)
from biosensor_priors.stage3_surrogate.features import PHYSICS_FEATURE_COLUMNS
from biosensor_priors.stage3_surrogate.gate3 import evaluate_gate3
from biosensor_priors.stage3_surrogate.surrogate import (
    FusedSurrogate,
    surrogate_kwargs_from_cfg,
)
from biosensor_priors.stage4_search.policy import build_search_policies

# Round history columns (w_ΔRIF is the selectivity ΔRIF weight).
WEIGHT_HISTORY_COLUMNS = ("Round", "w_RIF_Ac", "w_RIF_Prop", "w_ΔRIF", "intercept", "mode")


def append_physics_weights_row(
    history_path: Path,
    *,
    round_id: int | str,
    weights: dict[str, Any],
) -> pd.DataFrame:
    """Append one round of fitted physics coefficients to the history CSV.

    Columns: Round, w_RIF_Ac, w_RIF_Prop, w_ΔRIF (+ intercept/mode extras).
    Weights trending toward zero as labeled data accumulates is a legitimate
    outcome and is recorded, not suppressed.

    Parameters
    ----------
    history_path : Path
        CSV path for cumulative physics-weight history.
    round_id : int or str
        Prospective round identifier.
    weights : dict
        Physics coefficient dict from surrogate metadata.

    Returns
    -------
    pd.DataFrame
        Updated history table including the new row.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Round": round_id,
        "w_RIF_Ac": weights.get("rif_ac", weights.get("w_RIF_Ac")),
        "w_RIF_Prop": weights.get("rif_prop", weights.get("w_RIF_Prop")),
        "w_ΔRIF": weights.get(
            "delta_rif_sel",
            weights.get("w_ΔRIF", weights.get("w_delta_RIF")),
        ),
        "intercept": weights.get("intercept"),
        "mode": weights.get("mode"),
    }
    if history_path.exists():
        hist = pd.read_csv(history_path)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(history_path, index=False, encoding="utf-8")
    return hist


def refit_surrogate(
    master: pd.DataFrame,
    *,
    encoding: str = "hybrid",
    use_confidence_weighting: bool = True,
    random_seed: int = 42,
    kind: str = "physics_gp",
) -> tuple[FusedSurrogate, dict[str, Any]]:
    """Refit fused surrogate on all constructs with usable fitness.

    Parameters
    ----------
    master : pd.DataFrame
        Authoritative experiment master table.
    encoding : str, optional
        Feature encoding for the surrogate (default ``"hybrid"``).
    use_confidence_weighting : bool, optional
        Whether to discount physics features by structural confidence (default True).
    random_seed : int, optional
        Random seed for GP fitting (default 42).
    kind : str, optional
        Surrogate kind passed to :class:`FusedSurrogate` (default ``"physics_gp"``).

    Returns
    -------
    tuple of (FusedSurrogate, dict)
        Fitted model and its metadata dict.

    Raises
    ------
    RuntimeError
        If no fitness-labeled constructs are available.
    """
    fit_df = master[master["fitness"].notna()].copy()
    fit_df = attach_physics_and_confidence(fit_df, repo_root=REPO_ROOT)
    fit_df = FoldFitnessScaler().fit_transform(fit_df)
    fit_df = fit_df[fit_df["fitness"].notna()].copy()
    if fit_df.empty:
        raise RuntimeError("No fitness-labeled constructs available for model update.")
    model = FusedSurrogate(
        kind=kind,  # type: ignore[arg-type]
        use_confidence_weighting=use_confidence_weighting,
        random_state=random_seed,
        encoding=encoding,
        **{
            k: v
            for k, v in surrogate_kwargs_from_cfg({"encoding": encoding}).items()
            if k != "encoding"
        },
    )
    model.fit(fit_df, fit_df["fitness"].to_numpy(dtype=float))
    return model, model.metadata()


def rerun_calibration_gates(
    master: pd.DataFrame,
    *,
    splits_dir: Path | None = None,
    encoding: str = "hybrid",
    use_confidence_weighting: bool = True,
    random_seed: int = 42,
    require_hard_gate: bool = False,
) -> dict[str, Any]:
    """Re-run Stage-3 cross-validation and Gate 3 after appending prospective data.

    Parameters
    ----------
    master : pd.DataFrame
        Updated experiment master with new fitness labels.
    splits_dir : Path or None, optional
        Directory for cached CV splits; created on demand when provided.
    encoding : str, optional
        Feature encoding for CV evaluation (default ``"hybrid"``).
    use_confidence_weighting : bool, optional
        Whether to apply confidence weighting during CV (default True).
    random_seed : int, optional
        Random seed for split generation and model fitting (default 42).
    require_hard_gate : bool, optional
        When False, allow soft RMSE pass for operational gating (default False).

    Returns
    -------
    dict
        Gate 3 report with ``operational_passed``, ``n_cv_rows``, and ``n_splits``.
    """
    fit_df = master[master["fitness"].notna()].copy()
    fit_df = attach_physics_and_confidence(fit_df, repo_root=REPO_ROOT)
    splits = ensure_splits_for_fitness(
        fit_df,
        splits_dir,
        prefer_loco=True,
        random_seed=random_seed,
    )
    predictions = run_split_evaluation(
        fit_df,
        splits,
        use_confidence_weighting=use_confidence_weighting,
        random_seed=random_seed,
        encoding=encoding,
    )
    if predictions.empty:
        return {
            "passed": False,
            "operational_passed": False,
            "reason": "no CV predictions after model update",
        }
    gate = evaluate_gate3(predictions, random_seed=random_seed)
    gate["operational_passed"] = bool(
        gate["passed"] or (not require_hard_gate and gate.get("soft_passed_point_rmse"))
    )
    gate["n_cv_rows"] = int(len(predictions))
    gate["n_splits"] = int(predictions["split_id"].nunique()) if "split_id" in predictions else 0
    return gate


def _build_policies(search_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Instantiate search policies for next-batch generation after model update."""
    return build_search_policies(search_cfg, seed)


def generate_next_batch(
    *,
    master: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    surrogate: FusedSurrogate,
    search_cfg: dict[str, Any],
    strategy: str = "bo",
    batch_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Propose the next experimental batch with the updated surrogate.

    Parameters
    ----------
    master : pd.DataFrame
        Authoritative experiment master including all observed fitness rows.
    candidate_pool : pd.DataFrame
        Candidates eligible for the next round.
    surrogate : FusedSurrogate
        Refitted surrogate model.
    search_cfg : dict
        Parsed ``search.yaml`` configuration.
    strategy : str, optional
        Search strategy name (default ``"bo"``).
    batch_size : int or None, optional
        Batch size; defaults to ``search_cfg["batch_size"]``.
    random_seed : int, optional
        Random seed for stochastic policies (default 42).

    Returns
    -------
    pd.DataFrame
        Proposed batch with ``selection_algorithm`` and ``selection_rank`` columns.

    Raises
    ------
    ValueError
        If ``strategy`` is not among configured policies.
    """
    observed = master[master["fitness"].notna()].copy()
    policies = _build_policies(search_cfg, random_seed)
    if strategy not in policies:
        raise ValueError(f"Unknown strategy {strategy}; choose from {sorted(policies)}")
    b = int(batch_size if batch_size is not None else search_cfg.get("batch_size", 8))
    batch = policies[strategy].propose(observed, candidate_pool, surrogate, b)
    batch["selection_algorithm"] = strategy
    batch["selection_rank"] = range(1, len(batch) + 1)
    return batch


def save_model_update_artifacts(
    *,
    out_dir: Path,
    round_id: int | str,
    metadata: dict[str, Any],
    weights_history: pd.DataFrame,
    calibration_gate: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write JSON/CSV artifacts documenting a post-round model update.

    Parameters
    ----------
    out_dir : Path
        Output directory for round artifacts.
    round_id : int or str
        Prospective round identifier.
    metadata : dict
        Surrogate metadata from :func:`refit_surrogate`.
    weights_history : pd.DataFrame
        Cumulative physics-weight history table.
    calibration_gate : dict or None, optional
        Optional Gate 3 recalibration report to persist.

    Returns
    -------
    dict of str to Path
        Paths to written metadata, weights history, named weights, and optional gate JSON.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"round_{round_id}_model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    hist_path = out_dir / "physics_weights_by_round.csv"
    weights_history.to_csv(hist_path, index=False, encoding="utf-8")
    named = {
        "round_id": round_id,
        "physics_feature_columns": list(PHYSICS_FEATURE_COLUMNS),
        "weights": metadata.get("physics_weights", {}),
        "weight_history_columns": list(WEIGHT_HISTORY_COLUMNS),
    }
    named_path = out_dir / f"round_{round_id}_physics_weights.json"
    named_path.write_text(json.dumps(named, indent=2, default=str), encoding="utf-8")
    out: dict[str, Path] = {
        "metadata": meta_path,
        "weights_history": hist_path,
        "named_weights": named_path,
    }
    if calibration_gate is not None:
        gate_path = out_dir / f"round_{round_id}_calibration_gate.json"
        gate_path.write_text(
            json.dumps(calibration_gate, indent=2, default=str),
            encoding="utf-8",
        )
        out["calibration_gate"] = gate_path
    return out
