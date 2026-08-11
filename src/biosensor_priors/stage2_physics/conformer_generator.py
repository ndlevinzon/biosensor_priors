"""RDKit ETKDG conformer generation (OMEGA replacement for CHPC)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def rdkit_available() -> bool:
    """Return True when RDKit can be imported."""
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign

        return Chem, AllChem, rdMolAlign
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for the built-in conformer generator (no OMEGA on CHPC). "
            "Install with: pip install 'biosensor-priors[chem]' or conda install rdkit"
        ) from exc


def mol_from_smiles_or_file(
    *,
    smiles: str | None = None,
    path: Path | str | None = None,
):
    """Load an RDKit molecule from SMILES or SDF/MOL/MOL2."""
    Chem, _, _ = _require_rdkit()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        suffix = p.suffix.lower()
        if suffix in {".sdf", ".sd", ".mol"}:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False)
            mols = [m for m in suppl if m is not None]
            if not mols:
                raise ValueError(f"No molecules parsed from {p}")
            return mols[0]
        if suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(p), removeHs=False)
            if mol is None:
                raise ValueError(f"Failed to parse MOL2: {p}")
            return mol
        # Try SDF then SMILES text
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        mol = Chem.MolFromSmiles(text.splitlines()[0])
        if mol is None:
            raise ValueError(f"Unrecognized ligand file: {p}")
        return mol
    if not smiles:
        raise ValueError("Provide smiles= or path=")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def generate_conformers(
    *,
    smiles: str | None = None,
    input_path: Path | str | None = None,
    output_dir: Path | str,
    n_conformers: int = 50,
    prune_rms_thresh: float = 0.5,
    random_seed: int = 42,
    force_field: str = "MMFF94",
    add_hs: bool = True,
    minimize: bool = True,
    max_iters: int = 200,
) -> list[Path]:
    """
    Generate an ETKDG conformer ensemble and write one SDF per conformer.

    This replaces OpenEye OMEGA on CHPC. Requires RDKit.

    Parameters
    ----------
    smiles : str, optional
        Ligand SMILES when no 3D starting file is available.
    input_path : path-like, optional
        Starting SDF/MOL/MOL2.
    output_dir : path-like
        Directory for ``conf_000.sdf``, …
    n_conformers : int, optional
        Target number of embeddings before pruning (default 50).
    prune_rms_thresh : float, optional
        RMSD prune threshold in Å (default 0.5).
    random_seed : int, optional
        Embedding RNG seed.
    force_field : str, optional
        ``MMFF94`` or ``UFF`` for optional minimization.
    add_hs : bool, optional
        Add hydrogens before embedding (default True).
    minimize : bool, optional
        Run force-field minimization per conformer (default True).
    max_iters : int, optional
        Minimization iterations (default 200).

    Returns
    -------
    list of pathlib.Path
        Written SDF paths, sorted.
    """
    Chem, AllChem, rdMolAlign = _require_rdkit()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mol = mol_from_smiles_or_file(smiles=smiles, path=input_path)
    if add_hs:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.numThreads = 0

    ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params)
    if not ids:
        # Fallback single embed
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            raise RuntimeError("ETKDG embedding failed")
        ids = [cid]

    ff_name = str(force_field).upper()
    for cid in ids:
        if not minimize:
            continue
        if ff_name.startswith("MMFF"):
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
                if ff is not None:
                    ff.Minimize(maxIts=int(max_iters))
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
            if ff is not None:
                ff.Minimize(maxIts=int(max_iters))

    # Optional RMSD prune after minimize
    keep = list(ids)
    if len(keep) > 1 and prune_rms_thresh > 0:
        pruned: list[int] = []
        for cid in keep:
            if not pruned:
                pruned.append(int(cid))
                continue
            ok = True
            for kept in pruned:
                rms = rdMolAlign.GetBestRMS(mol, mol, int(cid), int(kept))
                if rms < float(prune_rms_thresh):
                    ok = False
                    break
            if ok:
                pruned.append(int(cid))
        keep = pruned

    paths: list[Path] = []
    for i, cid in enumerate(keep):
        path = out / f"conf_{i:03d}.sdf"
        w = Chem.SDWriter(str(path))
        # Write only this conformer
        conf_mol = Chem.Mol(mol)
        conf_mol.RemoveAllConformers()
        conf_mol.AddConformer(mol.GetConformer(int(cid)), assignId=True)
        conf_mol.SetProp("_Name", f"conf_{i:03d}")
        conf_mol.SetProp("conformer_index", str(i))
        w.write(conf_mol)
        w.close()
        paths.append(path)
    return paths


def cluster_conformers_rmsd(
    sdf_paths: list[Path],
    *,
    rmsd_threshold: float = 0.5,
    max_keep: int = 32,
) -> list[Path]:
    """Greedy RMSD clustering; returns representative SDF paths."""
    Chem, _, rdMolAlign = _require_rdkit()
    if not sdf_paths:
        return []
    mols = []
    for p in sdf_paths:
        m = Chem.SDMolSupplier(str(p), removeHs=False)[0]
        if m is not None:
            mols.append((p, m))
    reps: list[tuple[Path, Any]] = []
    for path, mol in mols:
        if not reps:
            reps.append((path, mol))
            continue
        if any(
            rdMolAlign.GetBestRMS(mol, rmol) < float(rmsd_threshold) for _, rmol in reps
        ):
            continue
        reps.append((path, mol))
        if len(reps) >= int(max_keep):
            break
    return [p for p, _ in reps]


def main(argv: list[str] | None = None) -> None:
    """CLI for RDKit conformer generation."""
    parser = argparse.ArgumentParser(
        description="Generate ligand conformers with RDKit ETKDG (OMEGA replacement)"
    )
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-conformers", type=int, default=50)
    parser.add_argument("--prune-rms", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ff", default="MMFF94")
    parser.add_argument("--no-minimize", action="store_true")
    args = parser.parse_args(argv)
    paths = generate_conformers(
        smiles=args.smiles,
        input_path=args.input,
        output_dir=args.out,
        n_conformers=args.n_conformers,
        prune_rms_thresh=args.prune_rms,
        random_seed=args.seed,
        force_field=args.ff,
        minimize=not args.no_minimize,
    )
    print(f"Wrote {len(paths)} conformers → {args.out}")


if __name__ == "__main__":
    main()
