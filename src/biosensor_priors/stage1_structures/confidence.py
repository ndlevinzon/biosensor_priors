"""Combine pLDDT and cross-model RMSD into ``structural_confidence.parquet``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage1_structures.structural_compare import (
    per_position_rmsd_across_models,
)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_structural_confidence(
    models: pd.DataFrame,
    residues: pd.DataFrame,
    *,
    conf_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """
    Build per-position structural confidence and reliability flags.

    Composite score (clipped to [0, 1]):

    ``w_plddt * pLDDT/100 + w_rmsd * max(0, 1 - RMSD/rmsd_max)
    + w_pae * max(0, 1 - PAE/pae_max)``

    Missing RMSD/PAE terms redistribute remaining weight onto pLDDT.

    Parameters
    ----------
    models : pandas.DataFrame
        Model registry from adapters.
    residues : pandas.DataFrame
        Per-residue adapter table.
    conf_cfg : dict, optional
        Confidence thresholds/weights from ``structures.yaml`` /
        ``thresholds.yaml``.
    repo_root : pathlib.Path, optional
        Used to load defaults when ``conf_cfg`` is omitted.

    Returns
    -------
    pandas.DataFrame
        Columns: Version, Canonical key, pLDDT, RMSD, PAE pocket,
        Confidence, Reliable (plus snake_case mirrors).
    """
    root = repo_root or REPO_ROOT
    if conf_cfg is None:
        structures = load_yaml(root / "configs" / "structures.yaml")
        thresholds = load_yaml(root / "configs" / "thresholds.yaml")
        conf_cfg = {
            **(thresholds.get("structure", {}).get("confidence") or {}),
            **(structures.get("confidence") or {}),
        }

    agg = per_position_rmsd_across_models(models, residues)
    if agg.empty:
        return pd.DataFrame(
            columns=[
                "Version",
                "Canonical key",
                "pLDDT",
                "RMSD",
                "PAE pocket",
                "Confidence",
                "Reliable",
                "version",
                "canonical_position",
                "plddt",
                "rmsd",
                "pae_pocket",
                "confidence",
                "reliable",
                "n_models",
            ]
        )

    plddt_min = float(conf_cfg.get("plDDT_min_reliable", 70.0))
    rmsd_max = float(conf_cfg.get("rmsd_max_reliable", 2.0))
    pae_max = float(conf_cfg.get("pae_pocket_max_reliable", 10.0))
    conf_min = float(conf_cfg.get("confidence_min_reliable", 0.5))
    w_plddt = float(conf_cfg.get("w_plddt", 0.5))
    w_rmsd = float(conf_cfg.get("w_rmsd", 0.3))
    w_pae = float(conf_cfg.get("w_pae", 0.2))

    plddt = agg["plddt"].to_numpy(dtype=float)
    rmsd = agg["rmsd"].to_numpy(dtype=float)
    pae = agg["pae_pocket"].to_numpy(dtype=float)

    term_p = _clip01(plddt / 100.0)
    term_r = np.where(np.isnan(rmsd), np.nan, _clip01(1.0 - rmsd / max(rmsd_max, 1e-6)))
    term_a = np.where(np.isnan(pae), np.nan, _clip01(1.0 - pae / max(pae_max, 1e-6)))

    scores = np.zeros(len(agg), dtype=float)
    for i in range(len(agg)):
        weights = []
        vals = []
        weights.append(w_plddt)
        vals.append(term_p[i])
        if not np.isnan(term_r[i]):
            weights.append(w_rmsd)
            vals.append(term_r[i])
        if not np.isnan(term_a[i]):
            weights.append(w_pae)
            vals.append(term_a[i])
        wsum = sum(weights) or 1.0
        scores[i] = sum(w * v for w, v in zip(weights, vals)) / wsum

    reliable = (
        (plddt >= plddt_min)
        & (scores >= conf_min)
        & (np.isnan(rmsd) | (rmsd <= rmsd_max))
        & (np.isnan(pae) | (pae <= pae_max))
    )

    out = agg.copy()
    out["confidence"] = scores
    out["reliable"] = reliable
    out["Version"] = out["version"]
    out["Canonical key"] = out["canonical_position"]
    out["pLDDT"] = out["plddt"]
    out["RMSD"] = out["rmsd"]
    out["PAE pocket"] = out["pae_pocket"]
    out["Confidence"] = out["confidence"]
    out["Reliable"] = np.where(out["reliable"], "yes", "no")
    return out


def write_structural_confidence(
    table: pd.DataFrame,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Write ``structural_confidence.parquet`` under ``data/structures``.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`compute_structural_confidence`.
    repo_root : pathlib.Path, optional
        Repository root.

    Returns
    -------
    pathlib.Path
        Written parquet path.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    out = resolve_path(pipeline["paths"]["structures"], root) / "structural_confidence.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return out
