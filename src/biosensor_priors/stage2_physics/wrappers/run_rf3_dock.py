"""
Stage 2 physics wrapper: RoseTTAFold3 (Foundry) docking → schema scores.

Maps RF3 confidence onto frozen Stage-2 / Stage-3 columns:

* ``rif_ac`` / ``rif_prop`` — negated interface confidence for AcCoA / PropCoA
* ``delta_rif_sel`` — ``rif_ac - rif_prop`` (computed downstream)

Requires Foundry ``rf3`` on PATH (``pip install 'rc-foundry[rf3]'``). Without
RF3 (or with ``--scaffold``), writes parser-compatible NaN TSVs for job wiring.

Config: ``configs/rf3_physics.yaml`` (+ ligand SMILES from ``physics.yaml``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from biosensor_priors.common.config import REPO_ROOT, resolve_path
from biosensor_priors.stage2_physics.wrappers._io import (
    load_mutations_json,
    resolve_mutations_path,
    write_status,
)

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

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


def load_rf3_cfg(repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/rf3_physics.yaml`` → ``rf3`` block."""
    root = repo_root or REPO_ROOT
    path = root / "configs" / "rf3_physics.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data.get("rf3") or data)


def load_physics_ligand_smiles(repo_root: Path | None = None) -> dict[str, str]:
    """SMILES map from ``configs/physics.yaml`` ligands.smiles."""
    root = repo_root or REPO_ROOT
    path = root / "configs" / "physics.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    smiles = (data.get("ligands") or {}).get("smiles") or {}
    return {str(k): str(v) for k, v in smiles.items() if v}


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


def rf3_available(cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return whether the ``rf3`` runner is on PATH (after optional conda hint)."""
    cfg = cfg or {}
    runner = str(cfg.get("run") or "rf3")
    if shutil.which(runner):
        return True, f"found {runner} on PATH"
    conda = cfg.get("conda_activate")
    if conda and Path(str(conda)).exists():
        return True, f"will invoke via conda_activate={conda}"
    return False, f"{runner} not on PATH (install rc-foundry[rf3])"


def sequence_from_structure(structure: Path) -> str:
    """Extract polymer one-letter sequence from PDB/CIF via BioPython."""
    structure = Path(structure)
    if not structure.exists():
        raise FileNotFoundError(structure)
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa, three_to_one

    suffix = structure.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser: Any = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    model = parser.get_structure("s", str(structure))[0]
    seq: list[str] = []
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            try:
                seq.append(three_to_one(res.get_resname()))
            except KeyError:
                seq.append("X")
        if seq:
            break
    if not seq:
        raise ValueError(f"No standard amino acids found in {structure}")
    return "".join(seq)


def apply_mutation(
    sequence: str,
    *,
    position: int | None,
    mutant: str | None,
    wt: str | None = None,
) -> str:
    """Apply a 1-indexed point mutation to ``sequence``."""
    if position is None or mutant is None:
        return sequence
    idx = int(position) - 1
    if idx < 0 or idx >= len(sequence):
        raise ValueError(
            f"position {position} out of range for sequence length {len(sequence)}"
        )
    if wt and sequence[idx] != wt:
        raise ValueError(
            f"WT mismatch at {position}: expected {wt}, found {sequence[idx]}"
        )
    return sequence[:idx] + str(mutant).upper() + sequence[idx + 1 :]


def write_mutated_pdb(
    structure: Path,
    out_pdb: Path,
    *,
    position: int | None,
    mutant: str | None,
) -> Path:
    """Copy PDB/CIF → PDB with one residue renamed (template for RF3)."""
    structure = Path(structure)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if position is None or mutant is None:
        if structure.suffix.lower() == ".pdb":
            shutil.copy2(structure, out_pdb)
            return out_pdb
        # CIF → write PDB via BioPython
        from Bio.PDB import PDBIO, MMCIFParser

        model = MMCIFParser(QUIET=True).get_structure("s", str(structure))
        io = PDBIO()
        io.set_structure(model)
        io.save(str(out_pdb))
        return out_pdb

    from Bio.PDB import PDBIO, MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa

    suffix = structure.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser: Any = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    model = parser.get_structure("s", str(structure))
    target = _AA1_TO_3.get(str(mutant).upper())
    if not target:
        raise ValueError(f"Unknown mutant AA: {mutant}")
    mutated = False
    for chain in model[0]:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if int(res.id[1]) != int(position):
                continue
            res.resname = target
            mutated = True
            break
        if mutated:
            break
    if not mutated:
        raise ValueError(f"Residue {position} not found in {structure}")
    io = PDBIO()
    io.set_structure(model)
    io.save(str(out_pdb))
    return out_pdb


def resolve_ligand_component(
    ligand_name: str,
    *,
    cfg: dict[str, Any],
    ligands_dir: Path | None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build an RF3 ligand component from path / CCD / SMILES."""
    root = repo_root or REPO_ROOT
    lig_cfg = dict((cfg.get("ligands") or {}).get(ligand_name) or {})
    chain_id = str(cfg.get("ligand_chain_id") or "B")
    component: dict[str, Any] = {"chain_id": chain_id}

    path = lig_cfg.get("path")
    if path:
        component["path"] = str(resolve_path(str(path), root))
        return component
    ccd = lig_cfg.get("ccd_code")
    if ccd:
        component["ccd_code"] = str(ccd)
        return component
    smiles = lig_cfg.get("smiles")
    if not smiles:
        smiles = load_physics_ligand_smiles(root).get(ligand_name)
    if not smiles and ligands_dir is not None:
        # Prefer an approved SDF if present
        approved = Path(ligands_dir) / ligand_name / "approved"
        if approved.is_dir():
            sdfs = sorted(approved.glob("*.sdf")) + sorted(approved.glob("*.mol"))
            if sdfs:
                component["path"] = str(sdfs[0])
                return component
    if smiles:
        component["smiles"] = str(smiles)
        return component
    raise ValueError(
        f"No ligand source for {ligand_name}: set rf3_physics.yaml ligands."
        f"{ligand_name}.path/smiles/ccd_code or physics.yaml ligands.smiles"
    )


def write_rf3_dock_json(
    path: Path,
    *,
    name: str,
    sequence: str,
    protein_template: Path | None,
    ligand_component: dict[str, Any],
    cfg: dict[str, Any],
    apo: bool = False,
) -> Path:
    """Write Foundry RF3 JSON for apo fold or protein–ligand docking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain_id = str(cfg.get("protein_chain_id") or "A")
    protein: dict[str, Any] = {
        "seq": "".join(str(sequence).split()),
        "chain_id": chain_id,
    }
    if protein_template is not None and Path(protein_template).exists():
        protein["path"] = str(Path(protein_template).resolve())

    components: list[dict[str, Any]] = [protein]
    if not apo:
        components.append(dict(ligand_component))

    payload: dict[str, Any] = {"name": name, "components": components}
    if cfg.get("template_protein", True) and protein_template is not None:
        payload["template_selection"] = [chain_id]
    if (
        not apo
        and cfg.get("ground_truth_ligand_conformer")
        and "path" in ligand_component
    ):
        lig_chain = str(
            ligand_component.get("chain_id")
            or cfg.get("ligand_chain_id")
            or "B"
        )
        payload["ground_truth_conformer_selection"] = [lig_chain]

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_rf3_fold_command(
    input_json: Path,
    out_dir: Path,
    cfg: dict[str, Any],
) -> list[str]:
    """Build argv for ``rf3 fold`` (Hydra ``key=value`` overrides)."""
    runner = str(cfg.get("run") or "rf3")
    hydra = [
        f"inputs='{Path(input_json).as_posix()}'",
        f"out_dir='{Path(out_dir).as_posix()}'",
        "early_stopping_plddt_threshold="
        f"{float(cfg.get('early_stopping_plddt_threshold', 0.0))}",
        f"diffusion_batch_size={int(cfg.get('diffusion_batch_size', 1))}",
        f"num_steps={int(cfg.get('num_steps', 50))}",
    ]
    ckpt = cfg.get("ckpt_path")
    if ckpt:
        ckpt_abs = resolve_path(str(ckpt), REPO_ROOT)
        hydra.append(f"ckpt_path='{ckpt_abs.as_posix()}'")
    for arg in cfg.get("extra_args") or []:
        hydra.append(str(arg))
    return [runner, "fold", *hydra]


def run_rf3_fold(
    input_json: Path,
    out_dir: Path,
    cfg: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    """Execute ``rf3 fold``, optionally via ``conda_activate`` bash wrapper."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = build_rf3_fold_command(input_json, out_dir, cfg)
    conda = cfg.get("conda_activate")
    env = os.environ.copy()
    if conda:
        # Prefer bash -lc so Hydra overrides keep quotes.
        shell = (
            f'source "{Path(str(conda)).as_posix()}" && '
            + " ".join(argv[:2])
            + " "
            + " ".join(argv[2:])
        )
        return subprocess.run(
            ["bash", "-lc", shell],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    return subprocess.run(
        argv,
        cwd=str(out_dir),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Recursively collect numeric leaves from JSON-like objects."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix or "value"] = float(obj)
    return out


def parse_rf3_confidence(out_dir: Path) -> dict[str, float]:
    """Parse RF3 confidence / ranking outputs into a flat metric map."""
    out_dir = Path(out_dir)
    metrics: dict[str, float] = {}

    for path in sorted(out_dir.rglob("*summary_confidences.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics.update(_flatten_numeric(data))

    for path in sorted(out_dir.rglob("*confidences.json")):
        if path.name.endswith("summary_confidences.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics.update(_flatten_numeric(data))

    for path in sorted(out_dir.rglob("*ranking_scores.csv")) + sorted(
        out_dir.rglob("*confidences.csv")
    ):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        for col in df.columns:
            try:
                metrics[str(col)] = float(row[col])
            except (TypeError, ValueError):
                continue

    for path in sorted(out_dir.rglob("*.score")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            data = json.loads(text)
            metrics.update(_flatten_numeric(data))
        except json.JSONDecodeError:
            for line in text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                elif "=" in line:
                    k, _, v = line.partition("=")
                else:
                    continue
                try:
                    metrics[k.strip()] = float(v.strip().rstrip(","))
                except ValueError:
                    continue

    # Normalize common aliases to short keys
    aliases = {
        "iptm": ["iptm", "ipTM", "interface_ptm", "interface_iptm"],
        "ptm": ["ptm", "pTM"],
        "ranking_score": ["ranking_score", "ranking_confidence", "score"],
        "plddt": ["plddt", "mean_plddt", "pLDDT", "avg_plddt"],
    }
    lower = {k.lower(): v for k, v in metrics.items()}
    for short, names in aliases.items():
        if short in metrics:
            continue
        for name in names:
            if name in metrics:
                metrics[short] = float(metrics[name])
                break
            if name.lower() in lower:
                metrics[short] = float(lower[name.lower()])
                break
    return metrics


def pick_metric(metrics: dict[str, float], keys: list[str]) -> float:
    """Return the first available metric from ``keys`` (case-insensitive)."""
    lower = {k.lower(): v for k, v in metrics.items()}
    for key in keys:
        if key in metrics:
            return float(metrics[key])
        if key.lower() in lower:
            return float(lower[key.lower()])
        # dotted suffix match (e.g. summary.iptm)
        for mk, mv in metrics.items():
            if mk.lower().endswith("." + key.lower()) or mk.lower() == key.lower():
                return float(mv)
    return math.nan


def confidence_to_score(value: float, *, negate: bool) -> float:
    """Map RF3 confidence → Stage-2 score (optionally negated)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    # pLDDT sometimes reported 0–100
    v = float(value)
    if v > 1.5:
        v = v / 100.0
    return -v if negate else v


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
        "backend",
        "structure_pdb",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, sep="\t", index=False)
    return path


def score_mutation_rf3(
    *,
    structure_pdb: Path,
    mutation: dict[str, Any] | str,
    cfg: dict[str, Any],
    structure_model_id: str | None,
    work_dir: Path,
    ligands_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run RF3 AcCoA/PropCoA docking for one mutation; return score row."""
    root = repo_root or REPO_ROOT
    if isinstance(mutation, str):
        mutation = parse_mutation_string(mutation)
    mut_code = str(mutation.get("mutation") or "WT")
    position = mutation.get("position")
    wt = mutation.get("wt")
    mutant = mutation.get("mutant")

    work = Path(work_dir) / mut_code.replace("/", "_")
    work.mkdir(parents=True, exist_ok=True)

    wt_seq = sequence_from_structure(structure_pdb)
    mut_seq = apply_mutation(wt_seq, position=position, mutant=mutant, wt=wt)
    template = write_mutated_pdb(
        structure_pdb,
        work / "template_mut.pdb",
        position=position if mut_code.upper() != "WT" else None,
        mutant=mutant if mut_code.upper() != "WT" else None,
    )

    negate = bool(cfg.get("negate_confidence", True))
    iface_keys = list(
        cfg.get("interface_metric_keys")
        or ["iptm", "interface_ptm", "ranking_score", "ptm"]
    )

    rif_ac = math.nan
    rif_prop = math.nan
    detail: dict[str, Any] = {}

    for ligand_key, col in (("AcCoA", "rif_ac"), ("PropCoA", "rif_prop")):
        lig_dir = work / ligand_key
        lig_comp = resolve_ligand_component(
            ligand_key, cfg=cfg, ligands_dir=ligands_dir, repo_root=root
        )
        lig_json = write_rf3_dock_json(
            work / f"{ligand_key}_input.json",
            name=f"{mut_code}_{ligand_key}",
            sequence=mut_seq,
            protein_template=template,
            ligand_component=lig_comp,
            cfg=cfg,
            apo=False,
        )
        proc = run_rf3_fold(lig_json, lig_dir, cfg)
        (lig_dir / "rf3_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (lig_dir / "rf3_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        metrics = parse_rf3_confidence(lig_dir)
        raw = pick_metric(metrics, iface_keys)
        score = confidence_to_score(raw, negate=negate)
        if col == "rif_ac":
            rif_ac = score
        else:
            rif_prop = score
        detail[ligand_key] = {
            "returncode": proc.returncode,
            "raw": raw,
            "metrics": metrics,
            "ligand": lig_comp,
        }

    (work / "score_detail.json").write_text(
        json.dumps(detail, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return {
        "mutation": mut_code,
        "position": position,
        "wt": wt,
        "mutant": mutant,
        "version": mutation.get("version"),
        "structure_model_id": structure_model_id,
        "structure_pdb": str(structure_pdb),
        "rif_ac": rif_ac,
        "rif_prop": rif_prop,
        "backend": "rf3",
    }


# Back-compat aliases used by older imports
load_rosetta_cfg = load_rf3_cfg
score_mutation_rosetta = score_mutation_rf3


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
) -> Path:
    """Score mutations with RF3 docking or write scaffold TSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mut_path = resolve_mutations_path(out, mutations_json)
    mutations = load_mutations_json(mut_path)
    cfg = load_rf3_cfg()

    ok, msg = rf3_available(cfg)
    use_scaffold = force_scaffold or not ok

    if use_scaffold:
        rows = scaffold_rows(
            mutations, structure_model_id=structure_model_id, structure_pdb=structure
        )
        write_status(
            out,
            tool="rf3_dock",
            mode="scaffold",
            detail={
                "rf3_ok": ok,
                "rf3_message": msg,
                "n_mutations": len(mutations),
                "structure": str(structure),
                "ligands": str(ligands) if ligands else None,
                "ligand_name": ligand_name,
                "next_step": (
                    "pip install 'rc-foundry[rf3]'; foundry install base-models; "
                    "set configs/rf3_physics.yaml conda_activate if needed; "
                    "drop --scaffold; set physics.yaml backend: external "
                    "and jobs to granite-gpu."
                ),
            },
        )
    else:
        rows = []
        errors: list[str] = []
        work = out / "rf3_runs"
        for mut in mutations or [
            {"mutation": "WT", "position": None, "wt": "X", "mutant": "X"}
        ]:
            try:
                rows.append(
                    score_mutation_rf3(
                        structure_pdb=structure,
                        mutation=mut,
                        cfg=cfg,
                        structure_model_id=structure_model_id,
                        work_dir=work,
                        ligands_dir=ligands,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — per-mutant isolation
                errors.append(f"{mut}: {type(exc).__name__}: {exc}")
                mut_spec = (
                    mut
                    if isinstance(mut, dict)
                    else parse_mutation_string(str(mut))
                )
                rows.append(
                    scaffold_rows(
                        [mut_spec],
                        structure_model_id=structure_model_id,
                        structure_pdb=structure,
                    )[0]
                )
        write_status(
            out,
            tool="rf3_dock",
            mode="live" if not errors else "live_partial",
            detail={"n_rows": len(rows), "errors": errors[:20]},
        )

    return write_interface_scores_tsv(out / score_filename, rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for RF3 Stage-2 docking scores."""
    parser = argparse.ArgumentParser(
        description="RoseTTAFold3 docking scores for Stage 2 physics priors"
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
        help="Force scaffold TSV even if rf3 is available",
    )
    parser.add_argument("--score-filename", default="rif_scores.tsv")
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
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
