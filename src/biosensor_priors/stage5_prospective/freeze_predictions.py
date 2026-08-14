"""Freeze immutable hashed predictions before synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from biosensor_priors.common.provenance import sha256_file


REQUIRED_FREEZE_COLUMNS = [
    "candidate_id",
    "predicted_fitness",
    "ci95_low",
    "ci95_high",
    "physics_component",
    "gp_component",
    "structural_confidence",
    "selection_algorithm",
    "selection_rank",
]


def _z_for_coverage(coverage: float = 0.95) -> float:
    """Return the normal z-score for symmetric interval coverage.

    Parameters
    ----------
    coverage : float, optional
        Target interval coverage probability (default 0.95).

    Returns
    -------
    float
        Two-sided normal critical value for ``coverage``.
    """
    return float(stats.norm.ppf(0.5 + coverage / 2.0))


def build_freeze_table(
    batch: pd.DataFrame,
    *,
    round_id: int | str,
    coverage: float = 0.95,
) -> pd.DataFrame:
    """Normalize a Stage-4 batch into the immutable freeze schema.

    Accepts either already-named freeze columns or Stage-4 prediction columns
    (``pred_fitness_mean``, ``pred_fitness_std``, etc.).

    Parameters
    ----------
    batch : pd.DataFrame
        Proposal batch from Stage 4.
    round_id : int or str
        Prospective round identifier.
    coverage : float, optional
        Confidence interval coverage for ``ci95_low`` / ``ci95_high`` (default 0.95).

    Returns
    -------
    pd.DataFrame
        Normalized freeze table with required prediction and selection columns.
    """
    df = batch.copy()
    z = _z_for_coverage(coverage)

    def col(*names: str, default: Any = np.nan) -> pd.Series:
        """Return the first present column from ``names``, else a default series.

        Parameters
        ----------
        *names : str
            Candidate column names in priority order.
        default : Any, optional
            Fill value when no column is found (default NaN).

        Returns
        -------
        pd.Series
            Selected column or constant default series aligned to ``df``.
        """
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([default] * len(df), index=df.index)

    candidate = col("candidate_id", "construct_id", "Construct", default="")
    mu = pd.to_numeric(col("predicted_fitness", "pred_fitness_mean", "fitness_mean"), errors="coerce")
    sigma = pd.to_numeric(col("predicted_std", "pred_fitness_std", "fitness_std"), errors="coerce").fillna(0.0)
    physics = pd.to_numeric(col("physics_component", "pred_physics_mean", "physics_mean"), errors="coerce")
    gp = pd.to_numeric(col("gp_component", "pred_gp_residual_mean", "gp_residual_mean"), errors="coerce")
    conf = pd.to_numeric(col("structural_confidence"), errors="coerce").fillna(0.0)
    algo = col("selection_algorithm", "search_strategy", default="unknown").astype(str)

    if "selection_rank" in df.columns:
        rank = pd.to_numeric(df["selection_rank"], errors="coerce")
    else:
        rank = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)

    out = pd.DataFrame(
        {
            "round_id": round_id,
            "candidate_id": candidate.astype(str),
            "predicted_fitness": mu,
            "predicted_std": sigma,
            "ci95_low": mu - z * sigma,
            "ci95_high": mu + z * sigma,
            "physics_component": physics,
            "gp_component": gp,
            "structural_confidence": conf,
            "selection_algorithm": algo,
            "selection_rank": rank.astype(int),
        }
    )
    # Preserve useful extras when present
    for extra in ("mutation_codes", "mutations", "version", "parent_version", "n_mutations"):
        if extra in df.columns:
            out[extra] = df[extra].values
    return out


def freeze_predictions(
    batch: pd.DataFrame,
    *,
    rounds_dir: Path,
    round_id: int | str,
    coverage: float = 0.95,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write immutable round predictions parquet once and record its SHA-256 hash.

    Also writes a sidecar ``round_{id}_predictions.sha256`` containing the digest.

    Parameters
    ----------
    batch : pd.DataFrame
        Stage-4 proposal batch to freeze.
    rounds_dir : Path
        Directory for round artifacts under ``data/rounds/``.
    round_id : int or str
        Prospective round identifier.
    coverage : float, optional
        Confidence interval coverage (default 0.95).
    overwrite : bool, optional
        Allow rewriting an existing freeze for administrative recovery (default False).

    Returns
    -------
    dict
        Metadata dict with path, sha256, candidate count, and algorithm list.

    Raises
    ------
    FileExistsError
        If the freeze file already exists and ``overwrite`` is False.
    ValueError
        If required freeze columns are missing after normalization.
    """
    rounds_dir = Path(rounds_dir)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    round_tag = f"{int(round_id):02d}" if str(round_id).isdigit() else str(round_id)
    path = rounds_dir / f"round_{round_tag}_predictions.parquet"
    hash_path = rounds_dir / f"round_{round_tag}_predictions.sha256"
    meta_path = rounds_dir / f"round_{round_tag}_predictions.meta.json"

    if path.exists() and not overwrite:
        digest = sha256_file(path)
        raise FileExistsError(
            f"Frozen predictions already exist and are immutable: {path} (sha256={digest}). "
            "Pass overwrite=True only for explicit administrative recovery."
        )

    freeze = build_freeze_table(batch, round_id=round_id, coverage=coverage)
    missing = [c for c in REQUIRED_FREEZE_COLUMNS if c not in freeze.columns]
    if missing:
        raise ValueError(f"Freeze table missing required columns: {missing}")

    # Write parquet with object columns stringified for stability
    store = freeze.copy()
    for col_name in store.columns:
        if store[col_name].dtype == object:
            store[col_name] = store[col_name].map(lambda x: None if x is None else str(x))
    store.to_parquet(path, index=False)

    # Also keep a pickle with native Python objects for round-trip validation
    pkl_path = rounds_dir / f"round_{round_tag}_predictions.pkl"
    freeze.to_pickle(pkl_path)

    digest = sha256_file(path)
    hash_path.write_text(digest + "\n", encoding="utf-8")
    meta = {
        "round_id": round_id,
        "path": str(path),
        "sha256": digest,
        "n_candidates": int(len(freeze)),
        "coverage": coverage,
        "immutable": True,
        "algorithms": sorted(freeze["selection_algorithm"].astype(str).unique().tolist()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def verify_freeze_integrity(rounds_dir: Path, round_id: int | str) -> dict[str, Any]:
    """Recompute parquet SHA-256 and compare to the sidecar digest.

    Parameters
    ----------
    rounds_dir : Path
        Directory containing frozen round artifacts.
    round_id : int or str
        Prospective round identifier.

    Returns
    -------
    dict
        Integrity report with ``expected_sha256``, ``actual_sha256``, and ``ok``.

    Raises
    ------
    FileNotFoundError
        If the parquet or sidecar hash file is missing.
    """
    rounds_dir = Path(rounds_dir)
    round_tag = f"{int(round_id):02d}" if str(round_id).isdigit() else str(round_id)
    path = rounds_dir / f"round_{round_tag}_predictions.parquet"
    hash_path = rounds_dir / f"round_{round_tag}_predictions.sha256"
    if not path.exists():
        raise FileNotFoundError(path)
    if not hash_path.exists():
        raise FileNotFoundError(hash_path)
    expected = hash_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    return {
        "path": str(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "ok": expected == actual,
    }


def load_frozen_predictions(rounds_dir: Path, round_id: int | str) -> pd.DataFrame:
    """Load immutable frozen predictions for a prospective round.

    Parameters
    ----------
    rounds_dir : Path
        Directory containing frozen round artifacts.
    round_id : int or str
        Prospective round identifier.

    Returns
    -------
    pd.DataFrame
        Frozen prediction table preferring pickle over parquet when both exist.

    Raises
    ------
    FileNotFoundError
        If no freeze file exists for the requested round.
    """
    rounds_dir = Path(rounds_dir)
    round_tag = f"{int(round_id):02d}" if str(round_id).isdigit() else str(round_id)
    pkl = rounds_dir / f"round_{round_tag}_predictions.pkl"
    parquet = rounds_dir / f"round_{round_tag}_predictions.parquet"
    if pkl.exists():
        return pd.read_pickle(pkl)
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(f"No frozen predictions for round {round_id} in {rounds_dir}")
