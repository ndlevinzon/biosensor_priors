"""Normalize AlphaFold 2 / AlphaFold 3 (and stub) outputs to one schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.stage1_structures.make_jobs import structure_model_id


REQUIRED_MODEL_COLS = [
    "structure_model_id",
    "version",
    "method",
    "seed",
    "state",
    "structure_path",
    "mean_plddt",
]

REQUIRED_RESIDUE_COLS = [
    "structure_model_id",
    "version",
    "method",
    "seed",
    "state",
    "residue_index",
    "canonical_position",
    "aa",
    "plddt",
]


def _empty_residue_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_RESIDUE_COLS)


def _empty_model_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_MODEL_COLS)


def _read_pdb_bfactors(path: Path) -> list[tuple[int, str, float]]:
    """Parse CA atoms from a PDB: (resseq, aa, bfactor=pLDDT)."""
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    rows: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 66:
            continue
        atom = line[12:16].strip()
        if atom not in {"CA", "Cα"}:
            continue
        resname = line[17:20].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        if resseq in seen:
            continue
        seen.add(resseq)
        try:
            bfactor = float(line[60:66])
        except ValueError:
            bfactor = float("nan")
        rows.append((resseq, aa_map.get(resname, "X"), bfactor))
    return rows


def _read_cif_ca_bfactors(path: Path) -> list[tuple[int, str, float]]:
    """Parse CA atoms from mmCIF via Bio.PDB when available; else empty."""
    try:
        from Bio.PDB import MMCIFParser  # type: ignore
    except ImportError:
        return []
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", str(path))
    except Exception:  # noqa: BLE001
        return []
    rows: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for atom in structure.get_atoms():
        if atom.get_name().strip() != "CA":
            continue
        res = atom.get_parent()
        resseq = int(res.id[1])
        if resseq in seen:
            continue
        seen.add(resseq)
        resname = res.get_resname().strip().upper()
        bfactor = float(atom.get_bfactor())
        rows.append((resseq, aa_map.get(resname, "X"), bfactor))
    return rows


def _normalize_plddt_scale(values: list[float]) -> list[float]:
    """Map 0–1 pLDDT to 0–100 when needed (Boltz often uses 0–1)."""
    if not values:
        return values
    finite = [v for v in values if v == v]
    if not finite:
        return values
    if max(finite) <= 1.5:
        return [float(v) * 100.0 for v in values]
    return [float(v) for v in values]


def _find_boltz2_structure(output_dir: Path) -> Path | None:
    """Locate top-ranked Boltz prediction CIF/PDB under ``predictions/``."""
    candidates = [
        *sorted(output_dir.rglob("*_model_0.cif")),
        *sorted(output_dir.rglob("*_model_0.pdb")),
        *sorted(output_dir.rglob("*.cif")),
        *sorted(output_dir.rglob("*.pdb")),
    ]
    return candidates[0] if candidates else None


def _boltz_plddt_from_npz(output_dir: Path, stem_hint: str | None = None) -> list[float] | None:
    """Load per-token pLDDT from Boltz ``plddt_*_model_0.npz`` if present."""
    patterns = ["plddt_*_model_0.npz", "plddt_*.npz"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(output_dir.rglob(pat)))
    if stem_hint:
        preferred = [p for p in files if stem_hint in p.name]
        files = preferred or files
    if not files:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        data = np.load(files[0])
        arr = data[data.files[0]] if data.files else None
        if arr is None:
            return None
        return [float(x) for x in np.asarray(arr).reshape(-1)]
    except Exception:  # noqa: BLE001
        return None


def parse_Boltz2(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse Boltz-2 outputs (``predictions/*/…_model_0.cif`` + optional npz)."""
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "Boltz2", seed, state)
    cif = _find_boltz2_structure(out)
    if cif is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    ca = _read_cif_ca_bfactors(cif)
    if not ca and cif.suffix.lower() == ".pdb":
        ca = _read_pdb_bfactors(cif)
    plddts = _boltz_plddt_from_npz(out, stem_hint=cif.stem.split("_model")[0])
    if plddts:
        plddts = _normalize_plddt_scale(plddts)
    residue_rows = []
    for i, (resseq, aa, bfactor) in enumerate(ca):
        plddt = float(plddts[i]) if plddts and i < len(plddts) else float(bfactor)
        if plddt <= 1.5:
            plddt *= 100.0
        residue_rows.append(
            {
                "structure_model_id": mid,
                "version": version,
                "method": "Boltz2",
                "seed": int(seed),
                "state": state,
                "residue_index": int(resseq),
                "canonical_position": int(resseq),
                "aa": aa,
                "plddt": plddt,
                "pae_pocket": float("nan"),
            }
        )
    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "Boltz2",
                "seed": int(seed),
                "state": state,
                "structure_path": str(cif),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def parse_RF3(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse RoseTTAFold3 Foundry outputs (``*_model.cif``)."""
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "RF3", seed, state)
    candidates = [
        *sorted(out.glob("*_model.cif")),
        *sorted(out.rglob("*_model.cif")),
        *sorted(out.rglob("*.cif")),
        *sorted(out.rglob("*.pdb")),
    ]
    structure = candidates[0] if candidates else None
    if structure is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    ca = (
        _read_cif_ca_bfactors(structure)
        if structure.suffix.lower() == ".cif"
        else _read_pdb_bfactors(structure)
    )
    residue_rows = [
        {
            "structure_model_id": mid,
            "version": version,
            "method": "RF3",
            "seed": int(seed),
            "state": state,
            "residue_index": int(resseq),
            "canonical_position": int(resseq),
            "aa": aa,
            "plddt": float(plddt if plddt > 1.5 else plddt * 100.0),
            "pae_pocket": float("nan"),
        }
        for resseq, aa, plddt in ca
    ]
    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "RF3",
                "seed": int(seed),
                "state": state,
                "structure_path": str(structure),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def _find_af3_structure(output_dir: Path) -> Path | None:
    """Locate AF3 model CIF/PDB under an output tree."""
    candidates = [
        *sorted(output_dir.rglob("*_model.cif")),
        *sorted(output_dir.rglob("*model_0.cif")),
        *sorted(output_dir.rglob("*.cif")),
        *sorted(output_dir.rglob("*.pdb")),
    ]
    return candidates[0] if candidates else None


def _af3_plddt_from_confidences(path: Path) -> list[float] | None:
    """Extract per-residue pLDDT-like scores from AF3 confidence JSON if present."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("atom_plddts", "plddt", "residue_plddts", "pae"):
        if key == "pae":
            continue
        if key in data and isinstance(data[key], list) and data[key]:
            vals = data[key]
            if all(isinstance(x, (int, float)) for x in vals):
                return [float(x) for x in vals]
    # summary only
    return None


def parse_AF2(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse AlphaFold 2 outputs into model + residue tables.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        AF2 ``--output_dir`` (contains per-target subdirectory).
    version : str
        Sequence background.
    seed : int
        Model seed used for ``structure_model_id``.
    state : str, optional
        Ligand/conformational state (default ``apo``).
    structure_model_id_value : str, optional
        Override auto-built ID.

    Returns
    -------
    dict
        ``models`` and ``residues`` DataFrames.
    """
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "AF2", seed, state)
    pdb = _find_af2_structure(out)
    if pdb is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    ca = _read_pdb_bfactors(pdb)
    residue_rows = []
    for resseq, aa, plddt in ca:
        residue_rows.append(
            {
                "structure_model_id": mid,
                "version": version,
                "method": "AF2",
                "seed": int(seed),
                "state": state,
                "residue_index": int(resseq),
                "canonical_position": int(resseq),
                "aa": aa,
                "plddt": float(plddt),
                "pae_pocket": float("nan"),
            }
        )
    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "AF2",
                "seed": int(seed),
                "state": state,
                "structure_path": str(pdb),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def parse_AF3(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
    af3_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse AlphaFold 3 outputs into model + residue tables.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        AF3 ``--output_dir``.
    version : str
        Sequence background.
    seed : int
        Model seed.
    state : str, optional
        State label (default ``apo``).
    structure_model_id_value : str, optional
        Override ID.
    af3_name : str, optional
        AF3 JSON ``name`` (used to prefer matching confidence files).

    Returns
    -------
    dict
        ``models`` and ``residues`` DataFrames.
    """
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "AF3", seed, state)
    structure = _find_af3_structure(out)
    if structure is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    conf_files = sorted(out.rglob("*confidences.json"))
    if af3_name:
        prefer = [p for p in conf_files if af3_name.lower() in p.name.lower()]
        conf_files = prefer + [p for p in conf_files if p not in prefer]

    plddts: list[float] | None = None
    for conf in conf_files:
        plddts = _af3_plddt_from_confidences(conf)
        if plddts:
            break

    residue_rows = []
    if structure.suffix.lower() == ".pdb":
        ca = _read_pdb_bfactors(structure)
        for i, (resseq, aa, b) in enumerate(ca):
            plddt = float(plddts[i]) if plddts and i < len(plddts) else float(b)
            residue_rows.append(
                {
                    "structure_model_id": mid,
                    "version": version,
                    "method": "AF3",
                    "seed": int(seed),
                    "state": state,
                    "residue_index": int(resseq),
                    "canonical_position": int(resseq),
                    "aa": aa,
                    "plddt": plddt,
                    "pae_pocket": float("nan"),
                }
            )
    else:
        # mmCIF: prefer confidence vector; fall back to sequential indices.
        n = len(plddts) if plddts else 0
        for i in range(n):
            residue_rows.append(
                {
                    "structure_model_id": mid,
                    "version": version,
                    "method": "AF3",
                    "seed": int(seed),
                    "state": state,
                    "residue_index": i + 1,
                    "canonical_position": i + 1,
                    "aa": "X",
                    "plddt": float(plddts[i]),
                    "pae_pocket": float("nan"),
                }
            )
        if not residue_rows:
            # Minimal model row only when CIF present but no confidences yet.
            pass

    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "AF3",
                "seed": int(seed),
                "state": state,
                "structure_path": str(structure),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def parse_RF2(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse RoseTTAFold2 outputs (typically under ``models/``) into Stage-1 tables.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory passed to ``run_RF2.sh … -o``.
    version : str
        Sequence background.
    seed : int
        Ensemble bookkeeping seed.
    state : str, optional
        State label (default ``apo``).
    structure_model_id_value : str, optional
        Override ID.

    Returns
    -------
    dict
        ``models`` and ``residues`` DataFrames.
    """
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "RF2", seed, state)
    candidates: list[Path] = []
    models_dir = out / "models"
    if models_dir.is_dir():
        candidates.extend(sorted(models_dir.glob("*.pdb")))
    candidates.extend(sorted(out.rglob("model*.pdb")))
    candidates.extend(sorted(out.rglob("*.pdb")))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    pdb = uniq[0] if uniq else None
    if pdb is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    ca = _read_pdb_bfactors(pdb)
    residue_rows = [
        {
            "structure_model_id": mid,
            "version": version,
            "method": "RF2",
            "seed": int(seed),
            "state": state,
            "residue_index": int(resseq),
            "canonical_position": int(resseq),
            "aa": aa,
            "plddt": float(plddt),
            "pae_pocket": float("nan"),
        }
        for resseq, aa, plddt in ca
    ]
    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "RF2",
                "seed": int(seed),
                "state": state,
                "structure_path": str(pdb),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def parse_RFAA(*args: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Alias for :func:`parse_RF3` (historical RFAA / RF2 name → RF3)."""
    return parse_RF3(*args, **kwargs)


def parse_ESMFold(
    output_dir: str | Path,
    *,
    version: str,
    seed: int,
    state: str = "apo",
    structure_model_id_value: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse ESMFold PDB outputs (pLDDT in B-factors) into model + residue tables.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory written by :mod:`biosensor_priors.stage1_structures.run_esmfold`
        (``{structure_model_id}.pdb``).
    version : str
        Sequence background.
    seed : int
        Ensemble bookkeeping seed.
    state : str, optional
        State label (default ``apo``).
    structure_model_id_value : str, optional
        Override ID.

    Returns
    -------
    dict
        ``models`` and ``residues`` DataFrames.
    """
    out = Path(output_dir)
    mid = structure_model_id_value or structure_model_id(version, "ESMFold", seed, state)
    # ESMFold writes one PDB per FASTA header; prefer exact mid match.
    candidates = [
        *sorted(out.glob(f"{mid}*.pdb")),
        *sorted(out.rglob("*.pdb")),
    ]
    pdb = candidates[0] if candidates else None
    if pdb is None:
        return {"models": _empty_model_frame(), "residues": _empty_residue_frame()}

    ca = _read_pdb_bfactors(pdb)
    residue_rows = [
        {
            "structure_model_id": mid,
            "version": version,
            "method": "ESMFold",
            "seed": int(seed),
            "state": state,
            "residue_index": int(resseq),
            "canonical_position": int(resseq),
            "aa": aa,
            "plddt": float(plddt),
            "pae_pocket": float("nan"),
        }
        for resseq, aa, plddt in ca
    ]
    residues = pd.DataFrame(residue_rows) if residue_rows else _empty_residue_frame()
    mean_plddt = float(residues["plddt"].mean()) if not residues.empty else float("nan")
    models = pd.DataFrame(
        [
            {
                "structure_model_id": mid,
                "version": version,
                "method": "ESMFold",
                "seed": int(seed),
                "state": state,
                "structure_path": str(pdb),
                "mean_plddt": mean_plddt,
            }
        ]
    )
    return {"models": models, "residues": residues}


def ingest_job_registry(
    registry: pd.DataFrame,
    *,
    repo_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run adapters for each completed job row in the Stage-1 registry.

    Parameters
    ----------
    registry : pandas.DataFrame
        Job registry from :func:`make_structure_jobs`.
    repo_root : pathlib.Path, optional
        Used to resolve relative ``output_dir`` paths.

    Returns
    -------
    dict
        Concatenated ``models`` and ``residues`` tables.
    """
    from biosensor_priors.common.config import REPO_ROOT
    from biosensor_priors.stage1_structures.make_jobs import canonicalize_method

    root = repo_root or REPO_ROOT
    model_parts: list[pd.DataFrame] = []
    residue_parts: list[pd.DataFrame] = []

    for _, row in registry.iterrows():
        try:
            method = canonicalize_method(str(row["method"]))
        except ValueError:
            continue
        out_dir = Path(str(row["output_dir"]))
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        kwargs = {
            "version": str(row["version"]),
            "seed": int(row["seed"]),
            "state": str(row["state"]),
            "structure_model_id_value": str(row["structure_model_id"]),
        }
        if method == "Boltz2":
            parsed = parse_Boltz2(out_dir, **kwargs)
        elif method == "AF3":
            parsed = parse_AF3(
                out_dir,
                af3_name=str(row["af3_name"])
                if "af3_name" in row and pd.notna(row.get("af3_name"))
                else None,
                **kwargs,
            )
        elif method == "ESMFold":
            parsed = parse_ESMFold(out_dir, **kwargs)
        elif method == "RF3":
            parsed = parse_RF3(out_dir, **kwargs)
        else:
            continue
        if not parsed["models"].empty:
            model_parts.append(parsed["models"])
        if not parsed["residues"].empty:
            residue_parts.append(parsed["residues"])

    models = pd.concat(model_parts, ignore_index=True) if model_parts else _empty_model_frame()
    residues = (
        pd.concat(residue_parts, ignore_index=True) if residue_parts else _empty_residue_frame()
    )
    return {"models": models, "residues": residues}
