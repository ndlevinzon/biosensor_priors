"""
Stage 2 physics wrapper: CHPC **PyRosetta** mutate → pack → score.

Maps Rosetta energies onto the frozen Stage-2 / Stage-3 schema:

* ``rif_ac`` / ``rif_prop`` — ligand interface (or total) energy for AcCoA / PropCoA
* ``rpx`` — packing / total energy after local pack (apo or same pose)

Requires ``module load pyrosetta/4.0.0`` on CHPC. Without PyRosetta (or with
``--scaffold``), writes parser-compatible NaN TSVs for job wiring tests.

Config: ``configs/rosetta_physics.yaml``.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from biosensor_priors.common.config import REPO_ROOT, resolve_path
from biosensor_priors.stage2_physics.wrappers._io import (
    load_mutations_json,
    resolve_mutations_path,
    try_import,
    write_status,
)

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_PYROSETTA_INITED = False


def load_rosetta_cfg(repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/rosetta_physics.yaml`` → ``rosetta`` block."""
    root = repo_root or REPO_ROOT
    path = root / "configs" / "rosetta_physics.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data.get("rosetta") or data)


def parse_mutation_string(mutation: str) -> dict[str, Any]:
    """Parse ``Q324R`` → wt / position / mutant."""
    m = _MUT_RE.match(str(mutation).strip())
    if not m:
        return {"mutation": mutation, "wt": None, "position": None, "mutant": None}
    return {
        "mutation": mutation,
        "wt": m.group(1),
        "position": int(m.group(2)),
        "mutant": m.group(3),
    }


def _ensure_pyrosetta(init_flags: str) -> Any:
    """Import and init PyRosetta once per process."""
    global _PYROSETTA_INITED
    import pyrosetta

    if not _PYROSETTA_INITED:
        pyrosetta.init(init_flags or "-mute all")
        _PYROSETTA_INITED = True
    return pyrosetta


def _pose_from_pdb(pyrosetta: Any, pdb: Path) -> Any:
    return pyrosetta.pose_from_pdb(str(pdb))


_AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _pose_res_from_pdb_number(pose: Any, position: int) -> int:
    """Map PDB residue number → pose index (first match)."""
    for i in range(1, pose.total_residue() + 1):
        info = pose.pdb_info()
        if info is not None and info.number(i) == int(position):
            return i
    if 1 <= int(position) <= pose.total_residue():
        return int(position)
    raise ValueError(f"Residue {position} not found in pose PDB numbering")


def _mutate_and_pack(
    pyrosetta: Any,
    pose: Any,
    *,
    position: int,
    mutant_aa: str,
    pack_radius: float,
    n_cycles: int,
    scorefxn_name: str,
) -> tuple[Any, float]:
    """Apply point mutation, pack neighborhood, return (pose, total score)."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        IncludeCurrent,
        NoRepackDisulfides,
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        NeighborhoodResidueSelector,
        NotResidueSelector,
        ResidueIndexSelector,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    scorefxn = pyrosetta.create_score_function(scorefxn_name)
    pose_res = _pose_res_from_pdb_number(pose, int(position))
    aa3 = _AA1_TO_3[str(mutant_aa).upper()]
    MutateResidue(pose_res, aa3).apply(pose)

    center = ResidueIndexSelector(str(pose_res))
    nbr = NeighborhoodResidueSelector(center, float(pack_radius), True)
    outside = NotResidueSelector(nbr)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    tf.push_back(NoRepackDisulfides())
    tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), outside))
    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    for _ in range(max(1, int(n_cycles))):
        packer.apply(pose)

    return pose, float(scorefxn(pose))


def _ligand_residue_indices(pose: Any, resnames: list[str]) -> list[int]:
    """Find ligand residue indices by 3-letter name."""
    wanted = {r.strip().upper() for r in resnames if r}
    hits: list[int] = []
    for i in range(1, pose.total_residue() + 1):
        name = pose.residue(i).name3().strip().upper()
        if name in wanted:
            hits.append(i)
    return hits


def _interface_energy(pyrosetta: Any, pose: Any, ligand_res: list[int], scorefxn_name: str) -> float:
    """Approximate ligand interface energy as E(complex) − E(protein-only)."""
    scorefxn = pyrosetta.create_score_function(scorefxn_name)
    e_complex = float(scorefxn(pose))
    if not ligand_res:
        return e_complex
    stripped = pose.clone()
    for i in sorted(ligand_res, reverse=True):
        stripped.delete_residue_slow(i)
    e_protein = float(scorefxn(stripped))
    return e_complex - e_protein


def score_mutation_rosetta(
    *,
    structure_pdb: Path,
    mutation: dict[str, Any] | str,
    cfg: dict[str, Any],
    structure_model_id: str | None = None,
) -> dict[str, Any]:
    """
    Mutate + pack one variant; score AcCoA / PropCoA complexes when configured.

    Returns a row with ``rif_ac``, ``rif_prop``, ``rpx``, and metadata.
    """
    parsed = (
        parse_mutation_string(mutation)
        if isinstance(mutation, str)
        else {
            "mutation": mutation.get("mutation"),
            "wt": mutation.get("wt"),
            "position": mutation.get("position"),
            "mutant": mutation.get("mutant"),
            "version": mutation.get("version"),
        }
    )
    pyrosetta = _ensure_pyrosetta(str(cfg.get("init_flags") or "-mute all"))
    scorefxn_name = str(cfg.get("scorefxn") or "ref2015")
    pack_radius = float(cfg.get("pack_radius_A") or 8.0)
    n_cycles = int(cfg.get("n_pack_cycles") or 2)
    complexes = dict(cfg.get("complexes") or {})
    lig_names = dict(cfg.get("ligand_resnames") or {})

    rif_ac = math.nan
    rif_prop = math.nan
    rpx = math.nan

    # Apo (or primary structure): packing score → rpx
    apo = Path(structure_pdb)
    if apo.exists() and parsed.get("position") is not None and parsed.get("mutant"):
        pose = _pose_from_pdb(pyrosetta, apo)
        pose, total = _mutate_and_pack(
            pyrosetta,
            pose,
            position=int(parsed["position"]),
            mutant_aa=str(parsed["mutant"]),
            pack_radius=pack_radius,
            n_cycles=n_cycles,
            scorefxn_name=scorefxn_name,
        )
        rpx = total

    # Optional holo complexes for ligand-specific interface energies
    for ligand_key, col in (("AcCoA", "rif_ac"), ("PropCoA", "rif_prop")):
        cpath = complexes.get(ligand_key)
        if not cpath:
            continue
        cpath = resolve_path(cpath) if not Path(str(cpath)).is_absolute() else Path(cpath)
        if not cpath.exists():
            continue
        if parsed.get("position") is None or not parsed.get("mutant"):
            continue
        pose = _pose_from_pdb(pyrosetta, cpath)
        pose, _ = _mutate_and_pack(
            pyrosetta,
            pose,
            position=int(parsed["position"]),
            mutant_aa=str(parsed["mutant"]),
            pack_radius=pack_radius,
            n_cycles=n_cycles,
            scorefxn_name=scorefxn_name,
        )
        resnames = list(lig_names.get(ligand_key) or [])
        lig_res = _ligand_residue_indices(pose, resnames) if resnames else []
        energy = (
            _interface_energy(pyrosetta, pose, lig_res, scorefxn_name)
            if lig_res
            else float(pyrosetta.create_score_function(scorefxn_name)(pose))
        )
        if col == "rif_ac":
            rif_ac = energy
        else:
            rif_prop = energy

    return {
        "mutation": parsed.get("mutation"),
        "position": parsed.get("position"),
        "wt": parsed.get("wt"),
        "mutant": parsed.get("mutant"),
        "version": parsed.get("version") if isinstance(mutation, dict) else None,
        "structure_model_id": structure_model_id,
        "structure_pdb": str(structure_pdb),
        "rif_ac": rif_ac,
        "rif_prop": rif_prop,
        "rpx": rpx,
        "backend": "pyrosetta",
    }


def scaffold_rows(
    mutations: list[dict[str, Any]],
    *,
    structure_model_id: str | None,
    structure_pdb: Path,
) -> list[dict[str, Any]]:
    """Emit NaN score rows so parsers and job wiring can be tested."""
    if not mutations:
        mutations = [
            {
                "mutation": "WT",
                "position": -1,
                "wt": "X",
                "mutant": "X",
                "version": None,
            }
        ]
    rows = []
    for mut in mutations:
        if isinstance(mut, str):
            mut = parse_mutation_string(mut)
        rows.append(
            {
                "mutation": mut.get("mutation", "NA"),
                "position": mut.get("position"),
                "wt": mut.get("wt"),
                "mutant": mut.get("mutant"),
                "version": mut.get("version"),
                "structure_model_id": structure_model_id,
                "structure_pdb": str(structure_pdb),
                "rif_ac": math.nan,
                "rif_prop": math.nan,
                "rpx": math.nan,
                "backend": "scaffold",
            }
        )
    return rows


def write_interface_scores_tsv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write score TSV in columns expected by ``score_parser``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mutation",
        "position",
        "wt",
        "mutant",
        "version",
        "structure_model_id",
        "rif_ac",
        "rif_prop",
        "rpx",
        "backend",
        "structure_pdb",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, sep="\t", index=False)
    return path


def write_rpx_only_tsv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write RPX-only TSV for the packing half of Stage 2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mutation",
        "position",
        "wt",
        "mutant",
        "structure_model_id",
        "rpx",
        "backend",
        "structure_pdb",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, sep="\t", index=False)
    return path


def run(
    *,
    structure: Path,
    ligands: Path | None = None,
    ligand_name: str = "AcCoA+PropCoA",
    out: Path,
    mutations_json: Path | None = None,
    structure_model_id: str | None = None,
    force_scaffold: bool = False,
    score_filename: str = "rif_scores.tsv",
    write_rpx: bool = False,
    rpx_filename: str = "rpx_scores.tsv",
) -> Path:
    """Score mutations with PyRosetta or write scaffold TSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mut_path = resolve_mutations_path(out, mutations_json)
    mutations = load_mutations_json(mut_path)
    cfg = load_rosetta_cfg()

    ok, msg = try_import("pyrosetta")
    use_scaffold = force_scaffold or not ok

    if use_scaffold:
        rows = scaffold_rows(
            mutations, structure_model_id=structure_model_id, structure_pdb=structure
        )
        write_status(
            out,
            tool="pyrosetta",
            mode="scaffold",
            detail={
                "import_ok": ok,
                "import_message": msg,
                "n_mutations": len(mutations),
                "structure": str(structure),
                "ligands": str(ligands) if ligands else None,
                "ligand_name": ligand_name,
                "next_step": (
                    "module load pyrosetta/4.0.0; set configs/rosetta_physics.yaml "
                    "complexes.AcCoA / PropCoA to holo PDBs; drop --scaffold; "
                    "set physics.yaml backend: external and jobs.module_loads."
                ),
            },
        )
    else:
        rows = []
        errors: list[str] = []
        for mut in mutations or [{"mutation": "WT", "position": None, "wt": "X", "mutant": "X"}]:
            try:
                rows.append(
                    score_mutation_rosetta(
                        structure_pdb=structure,
                        mutation=mut,
                        cfg=cfg,
                        structure_model_id=structure_model_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — per-mutant isolation
                errors.append(f"{mut}: {type(exc).__name__}: {exc}")
                rows.append(
                    scaffold_rows(
                        [mut if isinstance(mut, dict) else parse_mutation_string(str(mut))],
                        structure_model_id=structure_model_id,
                        structure_pdb=structure,
                    )[0]
                )
        write_status(
            out,
            tool="pyrosetta",
            mode="live" if not errors else "live_partial",
            detail={"n_rows": len(rows), "errors": errors[:20]},
        )

    primary = write_interface_scores_tsv(out / score_filename, rows)
    if write_rpx:
        write_rpx_only_tsv(out / rpx_filename, rows)
    return primary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the PyRosetta Stage-2 wrapper."""
    parser = argparse.ArgumentParser(
        description="PyRosetta mutate/pack/interface scores for Stage 2"
    )
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, default=None)
    parser.add_argument("--ligand-name", default="AcCoA+PropCoA")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutations-json", type=Path, default=None)
    parser.add_argument("--structure-model-id", default=None)
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Force scaffold TSV even if pyrosetta imports",
    )
    parser.add_argument("--score-filename", default="rif_scores.tsv")
    parser.add_argument(
        "--write-rpx",
        action="store_true",
        help="Also write rpx_scores.tsv from the same packing scores",
    )
    parser.add_argument("--rpx-filename", default="rpx_scores.tsv")
    args = parser.parse_args(argv)
    path = run(
        structure=args.structure,
        ligands=args.ligands,
        ligand_name=args.ligand_name,
        out=args.out,
        mutations_json=args.mutations_json,
        structure_model_id=args.structure_model_id,
        force_scaffold=args.scaffold,
        score_filename=args.score_filename,
        write_rpx=args.write_rpx,
        rpx_filename=args.rpx_filename,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
