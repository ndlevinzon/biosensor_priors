"""Programmatic wrapper around Stage-2 Rosetta (PyRosetta) interface scoring."""



from __future__ import annotations



import hashlib

import json

from datetime import UTC, datetime

from pathlib import Path

from typing import Any



import numpy as np

import pandas as pd



from biosensor_priors.stage2_physics.jobs import (

    PhysicsJob,

    run_local_job,

    verify_job_completion,

    write_sbatch_script,

    write_shell_script,

)

from biosensor_priors.stage2_physics.score_parser import write_mock_rif_scores





def _stable_seed(*parts: str, salt: int = 0) -> int:

    """Derive a deterministic 32-bit seed from string parts.



    Parameters
    ----------
    parts : str
        Strings hashed together for reproducibility.

    salt : int, optional

        XOR salt mixed into the digest.



    Returns
    -------
    int

        Unsigned 32-bit integer seed.

    """

    digest = hashlib.md5("|".join(parts).encode()).hexdigest()

    return (int(digest[:8], 16) ^ salt) % (2**32)





def format_rif_command(

    template: str,

    *,

    executable: str,

    structure_pdb: Path,

    ligand_dir: Path,

    ligand_name: str,

    out_dir: Path,

    structure_model_id: str,

    scan_id: str,

) -> list[str]:

    """Format an interface-scoring command template into a bash-wrapped argv list.



    Parameters
    ----------
    template : str

        Shell command template with format placeholders.

    executable : str

        Rosetta wrapper executable path or name.

    structure_pdb : pathlib.Path

        Input structure PDB path.

    ligand_dir : pathlib.Path

        Ligand ensemble directory.

    ligand_name : str

        Ligand identifier(s), e.g. ``AcCoA+PropCoA``.

    out_dir : pathlib.Path

        Output directory for interface scores.

    structure_model_id : str

        Structure model identifier.

    scan_id : str

        Physics scan batch identifier.



    Returns
    -------
    list of str

        Command argv prefixed with ``bash -lc``.

    """

    filled = template.format(

        executable=executable,

        structure_pdb=structure_pdb,

        ligand_dir=ligand_dir,

        ligand_name=ligand_name,

        out_dir=out_dir,

        structure_model_id=structure_model_id,

        scan_id=scan_id,

    )

    # Keep as a single shell string when template is multi-token; callers wrap in bash.

    return ["bash", "-lc", filled]





def build_rif_job(

    *,

    structure_model_id: str,

    ligand_name: str,

    structure_pdb: Path,

    ligand_dir: Path,

    out_dir: Path,

    physics_scan_id: str,

    physics_cfg: dict[str, Any],

    jobs_root: Path,

    logs_root: Path,

) -> PhysicsJob:

    """Construct an interface-scoring job (command + scripts + provenance sidecar).



    Parameters
    ----------
    structure_model_id : str

        Structure model identifier.

    ligand_name : str

        Ligand identifier(s) for the job.

    structure_pdb : pathlib.Path

        Input structure PDB path.

    ligand_dir : pathlib.Path

        Ligand ensemble root directory.

    out_dir : pathlib.Path

        Output directory for interface scores.

    physics_scan_id : str

        Scan batch identifier.

    physics_cfg : dict

        Physics configuration from ``physics.yaml``.

    jobs_root : pathlib.Path

        Root directory for job work folders.

    logs_root : pathlib.Path

        Root directory for stdout/stderr logs.



    Returns
    -------
    PhysicsJob

        Configured job record with scripts written.

    """

    rif_cfg = physics_cfg.get("rif", {})

    jobs_cfg = physics_cfg.get("jobs", {})

    executable = rif_cfg.get("executable") or "ROSETTA_EXECUTABLE_NOT_SET"

    template = rif_cfg.get("command_template") or "{executable} --out {out_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"rif_{structure_model_id}_{ligand_name}_{physics_scan_id}"

    work = jobs_root / job_id

    work.mkdir(parents=True, exist_ok=True)

    stdout = logs_root / f"{job_id}.out"

    stderr = logs_root / f"{job_id}.err"



    cmd = format_rif_command(

        template,

        executable=str(executable),

        structure_pdb=structure_pdb,

        ligand_dir=ligand_dir,

        ligand_name=ligand_name,

        out_dir=out_dir,

        structure_model_id=structure_model_id,

        scan_id=physics_scan_id,

    )

    job = PhysicsJob(

        job_id=job_id,

        kind="rif",

        command=cmd,

        work_dir=work,

        stdout_path=stdout,

        stderr_path=stderr,

        scheduler=str(jobs_cfg.get("scheduler", "local")),

        structure_model_id=structure_model_id,

        physics_scan_id=physics_scan_id,

        metadata={

            "ligand_name": ligand_name,

            "structure_pdb": str(structure_pdb),

            "ligand_dir": str(ligand_dir),

            "out_dir": str(out_dir),

            "executable": executable,

            "score_file": rif_cfg.get("output_score_filename", "rif_scores.tsv"),

        },

    )

    write_shell_script(

        work / "run.sh",

        cmd,

        module_loads=list(jobs_cfg.get("module_loads") or []),

    )

    if jobs_cfg.get("scheduler") == "slurm":

        write_sbatch_script(

            work / "submit.sbatch",

            cmd,

            job_name=job_id[:64],

            walltime=str(jobs_cfg.get("walltime", "04:00:00")),

            cpus=int(jobs_cfg.get("cpus", 4)),

            mem_gb=int(jobs_cfg.get("mem_gb", 16)),

            partition=jobs_cfg.get("partition"),

            account=jobs_cfg.get("account"),

            qos=jobs_cfg.get("qos"),

            module_loads=list(jobs_cfg.get("module_loads") or []),

            stdout_path=stdout,

            stderr_path=stderr,

        )

    job.write_sidecar()

    return job





def mock_rif_scores_for_mutations(

    mutations: list[dict[str, Any]],

    *,

    structure_model_id: str,

    physics_cfg: dict[str, Any],

    seed: int = 42,

) -> list[dict[str, Any]]:

    """Generate deterministic pseudo-physics scores for orchestration / Gate 2 dry-runs.

    Control mutations use configured Δ(rif_ac − rif_prop) favoring AcCoA when
    ``more_negative_is_better`` (rif_ac more negative than rif_prop).



    Parameters
    ----------
    mutations : list of dict

        Mutation specs with ``mutation``, ``position``, ``wt``, ``mutant``.

    structure_model_id : str

        Structure model identifier for seeding.

    physics_cfg : dict

        Physics config with ``mock`` subsection.

    seed : int, optional

        Base random seed (default 42).



    Returns
    -------
    list of dict

        Mock score rows with ``rif_ac``, ``rif_prop``, and metadata.

    """

    mock = physics_cfg.get("mock", {})

    base_prop = float(mock.get("base_rif_prop", -8.0))

    control_delta = dict(mock.get("control_delta_rif_sel") or {})

    noise_sd = float(mock.get("structure_noise_sd", 0.4))

    rng = np.random.default_rng(_stable_seed(structure_model_id, salt=seed))



    rows = []

    for mut in mutations:

        code = mut["mutation"]

        if code in control_delta:

            delta = float(control_delta[code])

        else:

            # Mild random selectivity for non-controls

            delta = float(rng.normal(0.0, 1.0))

        # Structure-to-structure noise

        jitter = float(rng.normal(0.0, noise_sd))

        rif_prop = base_prop + jitter

        rif_ac = rif_prop + delta + float(rng.normal(0.0, noise_sd * 0.5))

        rows.append(

            {

                "mutation": code,

                "position": mut["position"],

                "wt": mut["wt"],

                "mutant": mut["mutant"],

                "version": mut.get("version"),

                "structure_model_id": structure_model_id,

                "rif_ac": rif_ac,

                "rif_prop": rif_prop,

                "backend": "mock",

            }

        )

    return rows





def submit_or_mock_rif_job(

    job: PhysicsJob,

    *,

    mutations: list[dict[str, Any]],

    physics_cfg: dict[str, Any],

    seed: int = 42,

) -> tuple[PhysicsJob, Path]:

    """Run Rosetta scoring externally when configured; otherwise write mock scores and mark dry_run.



    Parameters
    ----------
    job : PhysicsJob

        Pre-built interface-scoring job record.

    mutations : list of dict

        Mutation specs to score in mock mode.

    physics_cfg : dict

        Physics configuration from ``physics.yaml``.

    seed : int, optional

        Random seed for mock backend (default 42).



    Returns
    -------
    job : PhysicsJob

        Updated job record.

    score_path : pathlib.Path

        Path to the interface score TSV.

    """

    backend = str(physics_cfg.get("backend", "mock"))

    executable = physics_cfg.get("rif", {}).get("executable")

    out_dir = Path(job.metadata["out_dir"])

    score_name = job.metadata.get("score_file", "rif_scores.tsv")

    score_path = out_dir / score_name



    if backend == "external" and executable:

        # Provide mutation list for wrapper scaffolds (and live runners).
        mut_path = out_dir / "mutations.json"
        mut_path.write_text(
            json.dumps({"mutations": mutations}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        job.metadata["mutations_json"] = str(mut_path)
        job = run_local_job(job, dry_run=False)

        return job, score_path



    # Mock path: do not invoke missing binaries

    assert job.structure_model_id is not None

    rows = mock_rif_scores_for_mutations(

        mutations,

        structure_model_id=job.structure_model_id,

        physics_cfg=physics_cfg,

        seed=seed,

    )

    # One job may cover both ligands; write combined table.

    write_mock_rif_scores(score_path, rows)

    job.status = "dry_run"

    job.returncode = 0

    job.stdout_path.parent.mkdir(parents=True, exist_ok=True)

    job.stdout_path.write_text(

        json.dumps({"backend": "mock", "n_rows": len(rows)}, indent=2),

        encoding="utf-8",

    )

    job.stderr_path.write_text("", encoding="utf-8")

    job.completed_at = datetime.now(tz=UTC).isoformat()

    job.write_sidecar()

    verify_job_completion(job, required_outputs=[score_path])

    return job, score_path





def prepare_rif_jobs_for_models(

    structure_models: pd.DataFrame,

    *,

    mutations: list[dict[str, Any]],

    ligand_dirs: dict[str, Path],

    physics_root: Path,

    physics_cfg: dict[str, Any],

    physics_scan_id: str,

    seed: int = 42,

) -> dict[str, Any]:

    """Build and run Rosetta interface jobs for AcCoA/PropCoA ensembles per structure model.



    ``structure_models`` needs columns: structure_model_id, pdb_path (or path).



    Parameters
    ----------
    structure_models : pandas.DataFrame

        Registry of structure models to score.

    mutations : list of dict

        Mutation specs for mock/external scoring.

    ligand_dirs : dict

        Mapping from ligand name to ensemble directory.

    physics_root : pathlib.Path

        Root directory for physics outputs.

    physics_cfg : dict

        Physics configuration from ``physics.yaml``.

    physics_scan_id : str

        Scan batch identifier.

    seed : int, optional

        Random seed for mock backend (default 42).



    Returns
    -------
    dict

        Keys ``jobs`` (list of PhysicsJob) and ``scores`` (combined DataFrame).

    """

    jobs_root = physics_root / physics_cfg.get("jobs", {}).get("jobs_subdir", "jobs")

    logs_root = physics_root / physics_cfg.get("jobs", {}).get("logs_subdir", "logs")

    jobs_root.mkdir(parents=True, exist_ok=True)

    logs_root.mkdir(parents=True, exist_ok=True)



    ligands = list(physics_cfg.get("rif", {}).get("ligands") or ["AcCoA", "PropCoA"])

    jobs: list[PhysicsJob] = []

    score_tables: list[pd.DataFrame] = []



    id_col = "structure_model_id"

    path_col = "pdb_path" if "pdb_path" in structure_models.columns else "path"

    for _, model in structure_models.iterrows():

        mid = str(model[id_col])

        pdb = Path(str(model[path_col]))

        # Combined out dir per model

        out_dir = physics_root / "rif" / mid / physics_scan_id

        # Use first ligand dir as ensemble root parent for template

        primary_ligand = ligands[0]

        ligand_dir = ligand_dirs.get(primary_ligand, physics_root / "ligands" / primary_ligand)

        job = build_rif_job(

            structure_model_id=mid,

            ligand_name="+".join(ligands),

            structure_pdb=pdb if pdb.exists() else physics_root / "structures_placeholder" / f"{mid}.pdb",

            ligand_dir=ligand_dir.parent if ligand_dir.name in ligands else ligand_dir,

            out_dir=out_dir,

            physics_scan_id=physics_scan_id,

            physics_cfg=physics_cfg,

            jobs_root=jobs_root,

            logs_root=logs_root,

        )

        # Ensure placeholder pdb path recorded

        if not pdb.exists():

            job.metadata["structure_pdb_missing"] = True

            placeholder = Path(job.metadata["structure_pdb"])

            placeholder.parent.mkdir(parents=True, exist_ok=True)

            if not placeholder.exists():

                placeholder.write_text(f"PLACEHOLDER PDB for {mid}\n", encoding="utf-8")



        job, score_path = submit_or_mock_rif_job(

            job, mutations=mutations, physics_cfg=physics_cfg, seed=seed

        )

        jobs.append(job)

        if score_path.exists():

            score_tables.append(pd.read_csv(score_path, sep="\t"))



    combined = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()

    return {"jobs": jobs, "scores": combined}


