"""Gaussian16 QM refinement inputs and CHPC SLURM scripts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


# Periodic table (Z → symbol) for Gaussian carts from atomic numbers if needed.
_Z_TO_SYM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
}


def _atoms_from_sdf(path: Path) -> list[tuple[str, float, float, float]]:
    """Parse atom symbols + XYZ from a simple V2000 SDF (no RDKit required)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    counts_idx = None
    for i, line in enumerate(lines):
        if "V2000" in line or "V3000" in line:
            counts_idx = i
            break
        if i == 3 and re.match(r"^\s*\d+\s+\d+", line):
            counts_idx = i
            break
    if counts_idx is None:
        return []
    parts = lines[counts_idx].split()
    try:
        n_atoms = int(parts[0])
    except (ValueError, IndexError):
        return []
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[counts_idx + 1 : counts_idx + 1 + n_atoms]:
        cols = line.split()
        if len(cols) < 4:
            continue
        try:
            x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
            sym = cols[3]
        except ValueError:
            continue
        atoms.append((sym, x, y, z))
    return atoms


def _atoms_from_sdf_rdkit(path: Path) -> list[tuple[str, float, float, float]]:
    """Prefer RDKit parsing when available."""
    try:
        from rdkit import Chem
    except ImportError:
        return _atoms_from_sdf(path)
    mol = Chem.SDMolSupplier(str(path), removeHs=False)[0]
    if mol is None or mol.GetNumConformers() == 0:
        return _atoms_from_sdf(path)
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), float(pos.x), float(pos.y), float(pos.z)))
    return atoms


def write_gaussian_gjf(
    path: Path,
    *,
    atoms: list[tuple[str, float, float, float]],
    title: str,
    charge: int = 0,
    multiplicity: int = 1,
    route: str = "#p B3LYP/6-31G(d) Opt",
    nproc: int = 8,
    mem: str = "32GB",
    chk_name: str | None = None,
) -> Path:
    """
    Write a Gaussian16 input (``.gjf`` / ``.com``).

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    atoms : list of (symbol, x, y, z)
        Cartesian coordinates in Å.
    title : str
        Job title line.
    charge, multiplicity : int
        Molecular charge and spin multiplicity.
    route : str
        Gaussian route section (default B3LYP/6-31G(d) Opt).
    nproc : int
        ``%NProcShared``.
    mem : str
        ``%Mem`` (e.g. ``32GB``).
    chk_name : str, optional
        Checkpoint basename; default ``path.stem``.

    Returns
    -------
    pathlib.Path
        Written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chk = chk_name or path.stem
    lines = [
        f"%NProcShared={int(nproc)}",
        f"%Mem={mem}",
        f"%Chk={chk}.chk",
        route if route.startswith("#") else f"#{route}",
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
    ]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:<2s}  {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.extend(["", ""])  # Gaussian requires trailing blank line
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_gaussian_slurm(
    path: Path,
    *,
    gjf_path: Path | str,
    qm_cfg: dict[str, Any],
    job_name: str | None = None,
) -> Path:
    """
    Write a CHPC SLURM script that loads Gaussian16 and runs ``g16``.

    Default module: ``gaussian16/SSE4.C01`` (ember / lonepeak SSE4 build).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    job = qm_cfg.get("job") or {}
    module = str(qm_cfg.get("module", "gaussian16/SSE4.C01"))
    exe = str(qm_cfg.get("executable", "g16"))
    gjf = Path(gjf_path)
    name = job_name or path.stem[:64]
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {name}",
        f"#SBATCH -t {job.get('time', '24:00:00')}",
        f"#SBATCH -n {int(job.get('ntasks', qm_cfg.get('nproc', 8)))}",
        f"#SBATCH -N {int(job.get('nodes', 1))}",
        f"#SBATCH --mem={job.get('mem', '32G')}",
    ]
    if job.get("partition"):
        lines.append(f"#SBATCH -p {job['partition']}")
    if job.get("account"):
        lines.append(f"#SBATCH -A {job['account']}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "ml purge",
            f"ml {module}",
            "",
            f'GJF="{gjf.resolve().as_posix()}"',
            'OUT="${GJF%.gjf}.log"',
            'OUT="${OUT%.com}.log"',
            'cd "$(dirname "$GJF")"',
            f'{exe} < "$(basename "$GJF")" > "$(basename "$OUT")"',
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def prepare_gaussian_jobs_for_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    qm_cfg: dict[str, Any],
    ligand: str = "ligand",
) -> dict[str, Any]:
    """
    For each SDF in ``input_dir``, write ``.gjf`` + ``.slurm`` under ``output_dir``.

    Does not run Gaussian. Returns paths and a ``submit_all.sh`` helper.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charge = int(qm_cfg.get("charge", 0))
    multiplicity = int(qm_cfg.get("multiplicity", 1))
    route = str(qm_cfg.get("route", "#p B3LYP/6-31G(d) Opt"))
    nproc = int(qm_cfg.get("nproc", 8))
    mem = str(qm_cfg.get("mem", "32GB"))

    jobs = []
    sdf_files = sorted(in_dir.glob("*.sdf")) + sorted(in_dir.glob("*.mol"))
    for sdf in sdf_files:
        atoms = _atoms_from_sdf_rdkit(sdf)
        stem = sdf.stem
        gjf = out_dir / f"{stem}.gjf"
        slurm = out_dir / f"{stem}.slurm"
        write_gaussian_gjf(
            gjf,
            atoms=atoms,
            title=f"{ligand} {stem} Opt",
            charge=charge,
            multiplicity=multiplicity,
            route=route,
            nproc=nproc,
            mem=mem,
        )
        write_gaussian_slurm(slurm, gjf_path=gjf, qm_cfg=qm_cfg, job_name=f"g16_{ligand}_{stem}"[:64])
        jobs.append({"sdf": str(sdf), "gjf": str(gjf), "slurm": str(slurm), "n_atoms": len(atoms)})

    submit = out_dir / "submit_all.sh"
    lines = [
        "#!/bin/bash",
        "# Submit Gaussian16 Opt jobs (CHPC).",
        "set -euo pipefail",
        "",
    ]
    for j in jobs:
        lines.append(f'sbatch "{Path(j["slurm"]).as_posix()}"')
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"jobs": jobs, "submit_script": submit, "output_dir": out_dir, "n_jobs": len(jobs)}


def parse_gaussian_optimized_xyz(log_path: Path | str) -> list[tuple[str, float, float, float]]:
    """
    Extract the last Standard orientation block from a Gaussian ``.log``.

    Parameters
    ----------
    log_path : path-like
        Gaussian output log.

    Returns
    -------
    list of (symbol, x, y, z)
    """
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if "Standard orientation" in ln]
    if not starts:
        return []
    i = starts[-1]
    # Skip until the second dashed separator after the header
    dash_hits = 0
    j = i + 1
    while j < len(lines) and dash_hits < 2:
        if set(lines[j].strip()) == {"-"} and len(lines[j].strip()) >= 10:
            dash_hits += 1
        j += 1
    atoms: list[tuple[str, float, float, float]] = []
    while j < len(lines):
        if set(lines[j].strip()) == {"-"} and len(lines[j].strip()) >= 10:
            break
        cols = lines[j].split()
        if len(cols) >= 6 and cols[0].isdigit() and cols[1].isdigit():
            z = int(cols[1])
            sym = _Z_TO_SYM.get(z, f"Z{z}")
            atoms.append((sym, float(cols[3]), float(cols[4]), float(cols[5])))
        j += 1
    return atoms


def main(argv: list[str] | None = None) -> None:
    """CLI: write Gaussian inputs from an SDF directory."""
    parser = argparse.ArgumentParser(description="Write Gaussian16 GJF + CHPC SLURM scripts")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ligand", default="ligand")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--route", default="#p B3LYP/6-31G(d) Opt")
    parser.add_argument("--module", default="gaussian16/SSE4.C01")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--mem", default="32GB")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    args = parser.parse_args(argv)
    qm_cfg = {
        "module": args.module,
        "executable": "g16",
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "route": args.route,
        "nproc": args.nproc,
        "mem": args.mem,
        "job": {
            "partition": args.partition,
            "account": args.account,
            "ntasks": args.nproc,
            "mem": args.mem if args.mem.endswith("G") else args.mem.replace("GB", "G"),
            "time": "24:00:00",
            "nodes": 1,
        },
    }
    result = prepare_gaussian_jobs_for_dir(
        args.input_dir, args.out, qm_cfg=qm_cfg, ligand=args.ligand
    )
    print(f"Wrote {result['n_jobs']} Gaussian jobs → {result['output_dir']}")
    print(f"Submit: bash {result['submit_script']}")


if __name__ == "__main__":
    main()
