"""Cross-model structural comparison (RMSD, contacts, pocket PAE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ca_coords_from_pdb(path: Path) -> dict[int, np.ndarray]:
    """Map residue sequence number → Cα coordinates from a PDB file."""
    coords: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[12:16].strip() not in {"CA", "Cα"}:
            continue
        try:
            resseq = int(line[22:26])
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except ValueError:
            continue
        coords[resseq] = xyz
    return coords


def pairwise_ca_rmsd(path_a: Path | str, path_b: Path | str) -> float:
    """Compute Cα RMSD over shared residue indices (no Kabsch; after identity map).

    For Gate-1 bookkeeping this is a cheap consistency metric. Pocket-focused
    superposition can replace it later.

    Parameters
    ----------
    path_a, path_b : path-like
        PDB paths.

    Returns
    -------
    float
        RMSD in Å, or NaN if fewer than 3 shared Cα atoms.
    """
    a = _ca_coords_from_pdb(Path(path_a))
    b = _ca_coords_from_pdb(Path(path_b))
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float("nan")
    xa = np.stack([a[k] for k in keys])
    xb = np.stack([b[k] for k in keys])
    # Center both
    xa = xa - xa.mean(axis=0)
    xb = xb - xb.mean(axis=0)
    # Kabsch
    h = xa.T @ xb
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ u.T
    xa_aligned = xa @ r
    diff = xa_aligned - xb
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def per_position_rmsd_across_models(
    models: pd.DataFrame,
    residues: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-position pLDDT and cross-model RMSD proxies.

    When multiple models share ``canonical_position``, RMSD is estimated from
    pairwise structure paths when PDBs exist; otherwise RMSD is NaN and
    confidence relies on pLDDT / PAE.

    Parameters
    ----------
    models : pandas.DataFrame
        Model-level table with ``structure_path``.
    residues : pandas.DataFrame
        Per-residue table with ``plddt`` and ``canonical_position``.

    Returns
    -------
    pandas.DataFrame
        Columns: version, canonical_position, plddt, rmsd, pae_pocket, n_models.
    """
    if residues is None or residues.empty:
        return pd.DataFrame(
            columns=[
                "version",
                "canonical_position",
                "plddt",
                "rmsd",
                "pae_pocket",
                "n_models",
            ]
        )

    # Global pairwise mean RMSD per version (broadcast to positions when PDBs exist)
    version_rmsd: dict[str, float] = {}
    if models is not None and not models.empty and "structure_path" in models.columns:
        for version, group in models.groupby("version"):
            paths = [Path(p) for p in group["structure_path"].dropna().tolist() if Path(p).exists()]
            paths = [p for p in paths if p.suffix.lower() == ".pdb"]
            if len(paths) < 2:
                version_rmsd[str(version)] = float("nan")
                continue
            vals = []
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    vals.append(pairwise_ca_rmsd(paths[i], paths[j]))
            version_rmsd[str(version)] = float(np.nanmean(vals)) if vals else float("nan")

    rows = []
    for (version, pos), group in residues.groupby(["version", "canonical_position"]):
        pae = (
            float(group["pae_pocket"].mean())
            if "pae_pocket" in group.columns and group["pae_pocket"].notna().any()
            else float("nan")
        )
        rows.append(
            {
                "version": version,
                "canonical_position": int(pos),
                "plddt": float(group["plddt"].mean()),
                "rmsd": version_rmsd.get(str(version), float("nan")),
                "pae_pocket": pae,
                "n_models": int(group["structure_model_id"].nunique())
                if "structure_model_id" in group.columns
                else len(group),
            }
        )
    return pd.DataFrame(rows)
