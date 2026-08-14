"""Dunbrack ipSAE (interaction prediction Score from Aligned Errors).

ipSAE is a PAE-derived interface score that is more comparable across
structure predictors than native ipTM (Dunbrack, bioRxiv 2025;
https://github.com/DunbrackLab/IPSAE).

Primary score is the residue-normalized ipSAE (d0res): for each aligned
residue i in chain A, average TM terms over chain-B residues with
PAE < cutoff, using d0 from the count of those residues; then take
max_i and the max of both directions.

Protein–ligand complexes use the same formula with ligand tokens as the
second chain (centroid coords when CA is absent).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
}


def calc_d0(length: float) -> float:
    """TM-score d0 with a floor of 1.0 Å (Yang & Skolnick; Dunbrack ipSAE).

    Parameters
    ----------
    length : float
        Effective residue count used for normalization.

    Returns
    -------
    float
        d0 in Ångströms.
    """
    length = float(length)
    if length > 27:
        d0 = 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 1.0
    return max(1.0, d0)


def ptm_term(pae: np.ndarray, d0: float) -> np.ndarray:
    """Pairwise TM-style terms ``1 / (1 + (PAE / d0)^2)``.

    Parameters
    ----------
    pae : numpy.ndarray
        Predicted aligned errors (Å).
    d0 : float
        Length-dependent TM scale.

    Returns
    -------
    numpy.ndarray
        TM terms in ``(0, 1]``.
    """
    d0 = max(float(d0), 1e-6)
    pae = np.asarray(pae, dtype=float)
    return 1.0 / (1.0 + (pae / d0) ** 2)


@dataclass
class IpsaeResult:
    """Directional and max ipSAE for one chain pair."""

    ipsae: float
    ipsae_ab: float
    ipsae_ba: float
    n0res_ab: float
    n0res_ba: float
    chain_a: str
    chain_b: str
    pae_cutoff: float
    dist_cutoff: float | None
    n_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result for JSON / parquet rows."""
        return {
            "ipsae": self.ipsae,
            "ipsae_ab": self.ipsae_ab,
            "ipsae_ba": self.ipsae_ba,
            "n0res_ab": self.n0res_ab,
            "n0res_ba": self.n0res_ba,
            "chain_a": self.chain_a,
            "chain_b": self.chain_b,
            "pae_cutoff": self.pae_cutoff,
            "dist_cutoff": self.dist_cutoff,
            "n_tokens": self.n_tokens,
            **self.extra,
        }


def _direction_scores(
    pae: np.ndarray,
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    *,
    pae_cutoff: float,
    dist: np.ndarray | None,
    dist_cutoff: float | None,
) -> tuple[float, float]:
    """Return ``(ipSAE A→B, n0res at the maximizing residue)``."""
    idx_i = np.flatnonzero(mask_i)
    idx_j = np.flatnonzero(mask_j)
    if idx_i.size == 0 or idx_j.size == 0:
        return 0.0, 0.0
    best = 0.0
    best_n0 = 0.0
    for i in idx_i:
        row = pae[i, idx_j]
        valid = row < pae_cutoff
        if dist is not None and dist_cutoff is not None:
            valid = valid & (dist[i, idx_j] < dist_cutoff)
        n0 = int(np.sum(valid))
        if n0 == 0:
            continue
        d0 = calc_d0(n0)
        score = float(np.mean(ptm_term(row[valid], d0)))
        if score > best:
            best = score
            best_n0 = float(n0)
    return best, best_n0


def ipsae_pair(
    pae: np.ndarray,
    chains: np.ndarray | Iterable[str],
    *,
    chain_a: str,
    chain_b: str,
    coords: np.ndarray | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float | None = 10.0,
) -> IpsaeResult:
    """Compute Dunbrack ipSAE for one chain pair.

    Parameters
    ----------
    pae : numpy.ndarray
        Square PAE matrix, shape ``(n_tokens, n_tokens)``, in Ångströms.
    chains : array-like of str
        Per-token chain identifiers, length ``n_tokens``.
    chain_a, chain_b : str
        Chain IDs to score (protein vs protein, or protein vs ligand).
    coords : numpy.ndarray, optional
        Token coordinates ``(n_tokens, 3)`` used for the optional distance
        filter. Ignored when ``dist_cutoff`` is None.
    pae_cutoff : float, optional
        Maximum PAE (Å) for a residue pair to contribute (default 10).
    dist_cutoff : float or None, optional
        Optional CA/centroid distance cutoff (Å). Default 10. Set None to
        use PAE only (still the published d0res score).

    Returns
    -------
    IpsaeResult
        Max-of-directions ipSAE and directional scores.
    """
    pae = np.asarray(pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square; got shape {pae.shape}")
    chains = np.asarray(list(chains), dtype=object)
    if len(chains) != pae.shape[0]:
        raise ValueError(
            f"chains length {len(chains)} != PAE size {pae.shape[0]}"
        )
    dist = None
    use_dist = dist_cutoff if coords is not None else None
    if coords is not None and use_dist is not None:
        xyz = np.asarray(coords, dtype=float)
        if xyz.shape[0] != pae.shape[0]:
            raise ValueError("coords rows must match PAE size")
        delta = xyz[:, None, :] - xyz[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
    mask_a = chains == chain_a
    mask_b = chains == chain_b
    ab, n_ab = _direction_scores(
        pae, mask_a, mask_b, pae_cutoff=pae_cutoff, dist=dist, dist_cutoff=use_dist
    )
    ba, n_ba = _direction_scores(
        pae, mask_b, mask_a, pae_cutoff=pae_cutoff, dist=dist, dist_cutoff=use_dist
    )
    return IpsaeResult(
        ipsae=float(max(ab, ba)),
        ipsae_ab=float(ab),
        ipsae_ba=float(ba),
        n0res_ab=n_ab,
        n0res_ba=n_ba,
        chain_a=str(chain_a),
        chain_b=str(chain_b),
        pae_cutoff=float(pae_cutoff),
        dist_cutoff=float(use_dist) if use_dist is not None else None,
        n_tokens=int(pae.shape[0]),
    )


def load_pae_matrix(path: Path) -> np.ndarray | None:
    """Load a square PAE matrix from JSON, NPZ, or NPY.

    Parameters
    ----------
    path : pathlib.Path
        Confidence / PAE file.

    Returns
    -------
    numpy.ndarray or None
        Square float matrix, or None when the file cannot be parsed.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return _as_square_pae(arr)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            for key in ("pae", "predicted_aligned_error", "PAE"):
                if key in data:
                    return _as_square_pae(data[key])
            for key in data.files:
                got = _as_square_pae(data[key])
                if got is not None:
                    return got
            return None
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _pae_from_json(payload)
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None
    return None


def _as_square_pae(arr: Any) -> np.ndarray | None:
    try:
        pae = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return None
    if pae.ndim == 3 and pae.shape[0] == 1:
        pae = pae[0]
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.shape[0] < 2:
        return None
    return pae


def _pae_from_json(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    for key in (
        "pae",
        "predicted_aligned_error",
        "PAE",
        "pae_matrix",
        "token_pae",
    ):
        if key in payload:
            got = _as_square_pae(payload[key])
            if got is not None:
                return got
    for nested_key in ("confidences", "summary", "metrics"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            got = _pae_from_json(nested)
            if got is not None:
                return got
    return None


def find_pae_file(output_dir: Path) -> Path | None:
    """Locate a PAE JSON/NPZ/NPY under a predictor output directory."""
    output_dir = Path(output_dir)
    patterns = [
        "*pae*_model_0.npz",
        "pae_*.npz",
        "*pae*.npz",
        "*full_data*.json",
        "*confidences.json",
        "*confidence*.json",
        "*pae*.json",
        "*pae*.npy",
    ]
    skip_suffix = "summary_confidences.json"
    for pat in patterns:
        hits = [
            p
            for p in sorted(output_dir.rglob(pat))
            if not p.name.endswith(skip_suffix)
        ]
        for path in hits:
            if load_pae_matrix(path) is not None:
                return path
    return None


def find_structure_file(output_dir: Path) -> Path | None:
    """Locate a CIF/PDB under a predictor output directory."""
    output_dir = Path(output_dir)
    candidates = [
        *sorted(output_dir.rglob("*_model_0.cif")),
        *sorted(output_dir.rglob("*_model.cif")),
        *sorted(output_dir.rglob("*.cif")),
        *sorted(output_dir.rglob("*_model_0.pdb")),
        *sorted(output_dir.rglob("*.pdb")),
    ]
    return candidates[0] if candidates else None


def _parse_pdb_tokens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, chain_ids) for CA atoms plus ligand residue centroids."""
    protein: dict[tuple[str, int], list[float]] = {}
    ligand: dict[tuple[str, int], list[list[float]]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = (line[21].strip() or "A")
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            key = (chain, resseq)
            if resname in AA3:
                if name == "CA" and key not in protein:
                    protein[key] = xyz
            else:
                ligand.setdefault(key, []).append(xyz)
    coords: list[list[float]] = []
    chains: list[str] = []
    for (chain, _), xyz in protein.items():
        coords.append(xyz)
        chains.append(chain)
    for (chain, _), atoms in ligand.items():
        arr = np.asarray(atoms, dtype=float)
        coords.append(arr.mean(axis=0).tolist())
        chains.append(chain)
    if not coords:
        return np.zeros((0, 3)), np.zeros((0,), dtype=object)
    return np.asarray(coords, dtype=float), np.asarray(chains, dtype=object)


def _parse_cif_tokens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """CA + ligand centroids from mmCIF via Biopython when available."""
    try:
        from Bio.PDB import MMCIFParser  # type: ignore
    except ImportError:
        return np.zeros((0, 3)), np.zeros((0,), dtype=object)
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", str(path))
    except Exception:  # noqa: BLE001
        return np.zeros((0, 3)), np.zeros((0,), dtype=object)
    protein: list[tuple[str, list[float]]] = []
    ligand_acc: dict[tuple[str, int], list[list[float]]] = {}
    seen_ca: set[tuple[str, int]] = set()
    for atom in structure.get_atoms():
        res = atom.get_parent()
        chain = res.get_parent()
        chain_id = str(chain.id).strip() or "A"
        resname = res.get_resname().strip().upper()
        resseq = int(res.id[1])
        xyz = [float(x) for x in atom.get_coord()]
        key = (chain_id, resseq)
        if resname in AA3:
            if atom.get_name().strip() == "CA" and key not in seen_ca:
                seen_ca.add(key)
                protein.append((chain_id, xyz))
        else:
            ligand_acc.setdefault(key, []).append(xyz)
    coords: list[list[float]] = []
    chains: list[str] = []
    for chain_id, xyz in protein:
        coords.append(xyz)
        chains.append(chain_id)
    for (chain_id, _), atoms in ligand_acc.items():
        arr = np.asarray(atoms, dtype=float)
        coords.append(arr.mean(axis=0).tolist())
        chains.append(chain_id)
    if not coords:
        return np.zeros((0, 3)), np.zeros((0,), dtype=object)
    return np.asarray(coords, dtype=float), np.asarray(chains, dtype=object)


def structure_tokens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-token coordinates and chain IDs from PDB or mmCIF.

    Parameters
    ----------
    path : pathlib.Path
        Structure file.

    Returns
    -------
    coords : numpy.ndarray
        Shape ``(n_tokens, 3)``.
    chains : numpy.ndarray
        Chain ID per token.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdb":
        return _parse_pdb_tokens(path)
    return _parse_cif_tokens(path)


def infer_ligand_chain(
    chains: np.ndarray,
    *,
    protein_chain: str = "A",
    ligand_chain: str | None = None,
) -> str | None:
    """Pick the ligand/partner chain ID from token chain labels."""
    unique = [str(c) for c in dict.fromkeys(chains.tolist())]
    if ligand_chain and ligand_chain in unique:
        return ligand_chain
    others = [c for c in unique if c != protein_chain]
    if len(others) == 1:
        return others[0]
    if protein_chain not in unique and len(unique) == 2:
        return unique[1]
    return others[0] if others else None


def ipsae_from_arrays(
    pae: np.ndarray,
    chains: np.ndarray,
    *,
    coords: np.ndarray | None = None,
    protein_chain: str = "A",
    ligand_chain: str | None = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float | None = 10.0,
) -> IpsaeResult | None:
    """Score the protein–partner interface from aligned PAE and chain labels."""
    partner = infer_ligand_chain(
        chains, protein_chain=protein_chain, ligand_chain=ligand_chain
    )
    unique = {str(c) for c in chains.tolist()}
    prot = protein_chain if protein_chain in unique else next(iter(unique), None)
    if prot is None or partner is None or partner == prot:
        return None
    return ipsae_pair(
        pae,
        chains,
        chain_a=prot,
        chain_b=partner,
        coords=coords,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
    )


def _align_tokens_to_pae(
    pae: np.ndarray,
    coords: np.ndarray,
    chains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Trim or reject token arrays so they match the PAE dimension."""
    n = pae.shape[0]
    m = len(chains)
    if m == n:
        return coords, chains
    if m > n:
        return coords[:n], chains[:n]
    return None


def ipsae_from_directory(
    output_dir: Path,
    *,
    protein_chain: str = "A",
    ligand_chain: str | None = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float | None = 10.0,
) -> IpsaeResult | None:
    """Compute ipSAE from a predictor (or RF3 dock) output directory.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory containing PAE JSON/NPZ and a CIF/PDB.
    protein_chain : str, optional
        Protein chain ID (default ``A``).
    ligand_chain : str or None, optional
        Ligand / partner chain ID (default ``B``). Inferred if missing.
    pae_cutoff, dist_cutoff : float
        Dunbrack cutoffs in Ångströms.

    Returns
    -------
    IpsaeResult or None
        Interface score, or None when PAE/structure/chains are unusable
        (typical for apo single-chain models).
    """
    output_dir = Path(output_dir)
    pae_path = find_pae_file(output_dir)
    if pae_path is None:
        return None
    pae = load_pae_matrix(pae_path)
    if pae is None:
        return None
    struct = find_structure_file(output_dir)
    coords: np.ndarray | None = None
    chains: np.ndarray | None = None
    if struct is not None:
        coords, chains = structure_tokens(struct)
        if len(chains) == 0:
            coords, chains = None, None
        else:
            aligned = _align_tokens_to_pae(pae, coords, chains)
            if aligned is None:
                coords, chains = None, None
            else:
                coords, chains = aligned
    if chains is None:
        # Fallback: first tokens protein, remainder ligand.
        n = pae.shape[0]
        if n < 3:
            return None
        n_lig = max(1, n // 20)
        chains = np.array(
            [protein_chain] * (n - n_lig) + [ligand_chain or "B"] * n_lig,
            dtype=object,
        )
    return ipsae_from_arrays(
        pae,
        chains,
        coords=coords,
        protein_chain=protein_chain,
        ligand_chain=ligand_chain,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
    )
