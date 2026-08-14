"""RDKit conformer generation: MOL2 sanitize fallback to SMILES."""

from __future__ import annotations

from pathlib import Path

import pytest

from biosensor_priors.common.config import REPO_ROOT

rdkit = pytest.importorskip("rdkit")  # noqa: F401


def test_accoa_mol2_falls_back_to_smiles(tmp_path: Path) -> None:
    from biosensor_priors.stage2_physics.conformer_generator import (
        generate_conformers,
        mol_from_smiles_or_file,
    )
    from biosensor_priors.stage2_physics.ligand_ensemble import read_smiles_file

    mol2 = REPO_ROOT / "data" / "ligands" / "AcCoA" / "starting.mol2"
    smi_path = REPO_ROOT / "data" / "ligands" / "AcCoA" / "ligand.smi"
    assert mol2.is_file()
    smiles = read_smiles_file(smi_path)
    assert smiles

    # SMILES must parse; MOL2 may or may not depending on RDKit version.
    assert mol_from_smiles_or_file(smiles=smiles) is not None

    paths = generate_conformers(
        smiles=smiles,
        input_path=mol2,
        output_dir=tmp_path / "confs",
        n_conformers=2,
        prune_rms_thresh=0.5,
        random_seed=0,
        minimize=False,
    )
    assert len(paths) >= 1
    assert all(p.suffix == ".sdf" for p in paths)


def test_propcoa_smiles_generates_without_mol2(tmp_path: Path) -> None:
    from biosensor_priors.stage2_physics.conformer_generator import generate_conformers
    from biosensor_priors.stage2_physics.ligand_ensemble import read_smiles_file

    smiles = read_smiles_file(REPO_ROOT / "data" / "ligands" / "PropCoA" / "ligand.smi")
    assert smiles
    paths = generate_conformers(
        smiles=smiles,
        input_path=None,
        output_dir=tmp_path / "confs",
        n_conformers=2,
        minimize=False,
        random_seed=1,
    )
    assert len(paths) >= 1
