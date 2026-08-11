"""Ligand conformer ensemble generation with permanent ``conformer_id``s."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import sha256_file
from biosensor_priors.stage2_physics.jobs import PhysicsJob, run_local_job, write_shell_script


def make_conformer_id(
    ligand: str,
    *,
    content_hash: str,
    schema_version: int = 1,
    index: int | None = None,
) -> str:
    """Build permanent conformer identity stable across re-runs.

    Format: ``{ligand}:{schema_version}:{content12}[:{index:03d}]``

    Parameters
    ----------
    ligand : str
        Ligand name, e.g. ``AcCoA``.
    content_hash : str
        SHA-256 hex digest of conformer content.
    schema_version : int, optional
        Conformer ID schema version (default 1).
    index : int, optional
        Conformer index within the ensemble.

    Returns
    -------
    str
        Permanent ``conformer_id`` string.
    """
    base = f"{ligand}:v{schema_version}:{content_hash[:12]}"
    if index is not None:
        return f"{base}:{index:03d}"
    return base


def _content_hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes.

    Parameters
    ----------
    data : bytes
        Payload to hash.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def _content_hash_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hex digest from :func:`sha256_file`.
    """
    return sha256_file(path)


def _placeholder_mol_block(ligand: str, index: int) -> str:
    """Return a minimal SDF-like stub for placeholder conformers.

    Parameters
    ----------
    ligand : str
        Ligand name.
    index : int
        Conformer index within the ensemble.

    Returns
    -------
    str
        SDF-like text block.
    """
    return (
        f"{ligand}_conf_{index:03d}\n"
        f"  biosensor_priors_placeholder\n"
        f"\n"
        f"  0  0  0  0  0  0  0  0  0  0999 V2000\n"
        f"M  END\n"
        f"> <ligand>\n{ligand}\n\n"
        f"> <conformer_index>\n{index}\n\n"
        f"$$$$\n"
    )


@dataclass
class LigandPipelineResult:
    conformers: pd.DataFrame
    ligand_dirs: dict[str, Path]
    catalog_path: Path
    jobs: list[PhysicsJob]


def prepare_ligand_directories(
    physics_root: Path,
    ligands: list[str],
) -> dict[str, Path]:
    """Create ligand pipeline subdirectories under the physics root.

    Parameters
    ----------
    physics_root : pathlib.Path
        Root directory for physics data.
    ligands : list of str
        Ligand names, e.g. ``AcCoA`` and ``PropCoA``.

    Returns
    -------
    dict
        Mapping from ligand name to its root directory path.
    """
    dirs = {}
    for lig in ligands:
        d = physics_root / "ligands" / lig
        for sub in ("starting", "raw", "cleaned", "qm", "approved"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        dirs[lig] = d
    return dirs


def _stage_tool_command(
    tool: str | None,
    *,
    stage: str,
    ligand: str,
    in_dir: Path,
    out_dir: Path,
) -> list[str]:
    """Build argv for a ligand pipeline stage command.

    Parameters
    ----------
    tool : str, optional
        External tool executable; when None, returns a no-op echo command.
    stage : str
        Pipeline stage name.
    ligand : str
        Ligand identifier.
    in_dir : pathlib.Path
        Input directory for the stage.
    out_dir : pathlib.Path
        Output directory for the stage.

    Returns
    -------
    list of str
        Command argv for the stage.
    """
    if tool:
        return [tool, "--ligand", ligand, "--in", str(in_dir), "--out", str(out_dir), "--stage", stage]
    # Placeholder no-op recorded for provenance when tools are undeployed.
    return ["echo", f"STAGE={stage}", f"LIGAND={ligand}", f"OUT={out_dir}"]


def run_ligand_pipeline_stage(
    *,
    stage: str,
    ligand: str,
    in_dir: Path,
    out_dir: Path,
    tool: str | None,
    jobs_dir: Path,
    logs_dir: Path,
    backend: str = "mock",
) -> PhysicsJob:
    """Orchestrate one ligand pipeline stage (external or dry-run).

    Parameters
    ----------
    stage : str
        Pipeline stage name.
    ligand : str
        Ligand identifier.
    in_dir : pathlib.Path
        Stage input directory.
    out_dir : pathlib.Path
        Stage output directory.
    tool : str, optional
        External tool executable for this stage.
    jobs_dir : pathlib.Path
        Root directory for job work folders.
    logs_dir : pathlib.Path
        Root directory for stdout/stderr logs.
    backend : str, optional
        Physics backend; ``external`` runs real tools when configured.

    Returns
    -------
    PhysicsJob
        Completed or dry-run job record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"ligand_{ligand}_{stage}"
    work = jobs_dir / job_id
    work.mkdir(parents=True, exist_ok=True)
    cmd = _stage_tool_command(tool, stage=stage, ligand=ligand, in_dir=in_dir, out_dir=out_dir)
    job = PhysicsJob(
        job_id=job_id,
        kind="ligand",
        command=cmd,
        work_dir=work,
        stdout_path=logs_dir / f"{job_id}.out",
        stderr_path=logs_dir / f"{job_id}.err",
        metadata={"stage": stage, "ligand": ligand, "backend": backend, "tool": tool},
    )
    write_shell_script(work / "run.sh", cmd)
    # Always dry-run unless a real tool path is configured and backend=external.
    dry = backend != "external" or tool is None
    return run_local_job(job, dry_run=dry)


def build_approved_ensemble(
    ligand: str,
    ligand_dir: Path,
    *,
    schema_version: int = 1,
    n_placeholder: int = 3,
    starting_structure: Path | None = None,
    max_conformers: int = 32,
) -> pd.DataFrame:
    """Produce approved conformer files and catalog rows with permanent IDs.

    When real QM/cluster outputs are absent, writes deterministic placeholders
    so downstream RIF job construction still has an ensemble to point at.

    Parameters
    ----------
    ligand : str
        Ligand identifier.
    ligand_dir : pathlib.Path
        Ligand root directory with pipeline subfolders.
    schema_version : int, optional
        Conformer ID schema version (default 1).
    n_placeholder : int, optional
        Number of placeholder conformers to generate (default 3).
    starting_structure : pathlib.Path, optional
        Seed structure for conformer generation.
    max_conformers : int, optional
        Maximum conformers to retain (default 32).

    Returns
    -------
    pandas.DataFrame
        Catalog rows with ``conformer_id``, ``path``, and metadata.
    """
    approved = ligand_dir / "approved"
    approved.mkdir(parents=True, exist_ok=True)

    # Prefer already-approved structures if present.
    existing = sorted(approved.glob("*.sdf")) + sorted(approved.glob("*.mol2"))
    rows = []
    if existing:
        for i, path in enumerate(existing[:max_conformers]):
            digest = _content_hash_file(path)
            cid = make_conformer_id(
                ligand, content_hash=digest, schema_version=schema_version, index=i
            )
            rows.append(
                {
                    "conformer_id": cid,
                    "ligand": ligand,
                    "conformer_index": i,
                    "path": str(path),
                    "content_sha256": digest,
                    "source": "existing_approved",
                    "schema_version": schema_version,
                    "approved": True,
                }
            )
        return pd.DataFrame(rows)

    # Seed from starting structure hash if provided, else ligand name.
    if starting_structure and Path(starting_structure).exists():
        seed_hash = _content_hash_file(Path(starting_structure))
        src = "from_starting_structure"
        # Copy starting into starting/
        dest = ligand_dir / "starting" / Path(starting_structure).name
        if not dest.exists():
            dest.write_bytes(Path(starting_structure).read_bytes())
    else:
        seed_hash = _content_hash_bytes(f"{ligand}:placeholder_seed".encode())
        src = "placeholder"
        stub = ligand_dir / "starting" / f"{ligand}_starting.sdf"
        stub.write_text(_placeholder_mol_block(ligand, 0), encoding="utf-8")

    n = min(n_placeholder, max_conformers)
    for i in range(n):
        # Content includes index so each conformer_id is unique but stable.
        payload = f"{ligand}|{seed_hash}|{i}|v{schema_version}".encode()
        digest = _content_hash_bytes(payload)
        cid = make_conformer_id(
            ligand, content_hash=digest, schema_version=schema_version, index=i
        )
        path = approved / f"{cid.replace(':', '_')}.sdf"
        path.write_text(_placeholder_mol_block(ligand, i), encoding="utf-8")
        # Re-hash file bytes for catalog (permanent file identity)
        file_hash = _content_hash_file(path)
        rows.append(
            {
                "conformer_id": cid,
                "ligand": ligand,
                "conformer_index": i,
                "path": str(path),
                "content_sha256": file_hash,
                "source": src,
                "schema_version": schema_version,
                "approved": True,
                "created_at": datetime.now(tz=UTC).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def run_ligand_ensemble(
    *,
    repo_root: Path | None = None,
    physics_cfg: dict[str, Any] | None = None,
    n_placeholder: int = 3,
) -> LigandPipelineResult:
    """Stage 2A — orchestrate ligand conformer pipeline and write catalog.

    Outputs under ``data/physics/ligands/{AcCoA,PropCoA}/`` and
    ``data/physics/ligand_conformers.parquet``.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root for config resolution.
    physics_cfg : dict, optional
        Physics configuration override.
    n_placeholder : int, optional
        Placeholder conformers per ligand when real outputs absent (default 3).

    Returns
    -------
    LigandPipelineResult
        Dataclass with ``conformers``, ``ligand_dirs``, ``catalog_path``,
        and ``jobs``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    physics_cfg = physics_cfg or load_yaml(root / "configs" / "physics.yaml")
    physics_root = resolve_path(pipeline["paths"]["physics"], root)
    physics_root.mkdir(parents=True, exist_ok=True)

    lig_cfg = physics_cfg.get("ligands", {})
    ligands = list(lig_cfg.get("names") or ["AcCoA", "PropCoA"])
    schema_version = int(lig_cfg.get("conformer_id_schema_version", 1))
    max_conf = int(lig_cfg.get("clustering", {}).get("max_conformers_per_ligand", 32))
    tools = lig_cfg.get("tools") or {}
    stages = list(lig_cfg.get("pipeline") or [])
    backend = str(physics_cfg.get("backend", "mock"))
    starting = lig_cfg.get("starting_structures") or {}

    jobs_dir = physics_root / physics_cfg.get("jobs", {}).get("jobs_subdir", "jobs")
    logs_dir = physics_root / physics_cfg.get("jobs", {}).get("logs_subdir", "logs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ligand_dirs = prepare_ligand_directories(physics_root, ligands)
    all_jobs: list[PhysicsJob] = []
    catalogs: list[pd.DataFrame] = []

    stage_dirs = {
        "conformer_generation": ("starting", "raw"),
        "geometry_cleanup": ("raw", "cleaned"),
        "qm_refinement": ("cleaned", "qm"),
        "deduplication_clustering": ("qm", "approved"),
        "approve_ensemble": ("approved", "approved"),
    }
    tool_for_stage = {
        "conformer_generation": tools.get("conformer_generator"),
        "geometry_cleanup": tools.get("geometry_cleanup"),
        "qm_refinement": tools.get("qm_refinement"),
        "deduplication_clustering": tools.get("clustering"),
        "approve_ensemble": None,
    }

    for ligand in ligands:
        ldir = ligand_dirs[ligand]
        for stage in stages:
            if stage not in stage_dirs:
                continue
            in_name, out_name = stage_dirs[stage]
            job = run_ligand_pipeline_stage(
                stage=stage,
                ligand=ligand,
                in_dir=ldir / in_name,
                out_dir=ldir / out_name,
                tool=tool_for_stage.get(stage),
                jobs_dir=jobs_dir,
                logs_dir=logs_dir,
                backend=backend,
            )
            all_jobs.append(job)

        start_path = starting.get(ligand)
        start = resolve_path(start_path, root) if start_path else None
        cat = build_approved_ensemble(
            ligand,
            ldir,
            schema_version=schema_version,
            n_placeholder=n_placeholder,
            starting_structure=start,
            max_conformers=max_conf,
        )
        catalogs.append(cat)

        # Provenance sidecar per ligand
        (ldir / "ensemble_meta.json").write_text(
            json.dumps(
                {
                    "ligand": ligand,
                    "n_conformers": int(len(cat)),
                    "schema_version": schema_version,
                    "backend": backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    conformers = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    catalog_path = physics_root / "ligand_conformers.parquet"
    conformers.to_parquet(catalog_path, index=False)
    # Also CSV for human inspection
    conformers.to_csv(physics_root / "ligand_conformers.csv", index=False)

    return LigandPipelineResult(
        conformers=conformers,
        ligand_dirs=ligand_dirs,
        catalog_path=catalog_path,
        jobs=all_jobs,
    )
