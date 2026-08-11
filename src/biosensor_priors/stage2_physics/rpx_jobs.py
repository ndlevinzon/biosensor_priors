"""Programmatic wrapper around Rosetta packing (rpx column) jobs."""



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

from biosensor_priors.stage2_physics.score_parser import write_mock_rpx_scores





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





def build_rpx_job(

    *,

    structure_model_id: str,

    mutation: str,

    structure_pdb: Path,

    out_dir: Path,

    physics_scan_id: str,

    physics_cfg: dict[str, Any],

    jobs_root: Path,

    logs_root: Path,

) -> PhysicsJob:

    """Construct an RPX job with shell/sbatch scripts and provenance sidecar.



    Parameters
    ----------
    structure_model_id : str

        Structure model identifier.

    mutation : str

        Mutation code or ``BATCH`` for batched jobs.

    structure_pdb : pathlib.Path

        Input structure PDB path.

    out_dir : pathlib.Path

        Directory for RPX score outputs.

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

    rpx_cfg = physics_cfg.get("rpx", {})

    jobs_cfg = physics_cfg.get("jobs", {})

    executable = rpx_cfg.get("executable") or "ROSETTA_PACK_EXECUTABLE_NOT_SET"

    template = rpx_cfg.get("command_template") or "{executable} --out {out_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"rpx_{structure_model_id}_{mutation}_{physics_scan_id}"

    work = jobs_root / job_id

    work.mkdir(parents=True, exist_ok=True)

    filled = template.format(

        executable=executable,

        structure_pdb=structure_pdb,

        mutation=mutation,

        out_dir=out_dir,

        structure_model_id=structure_model_id,

        scan_id=physics_scan_id,

    )

    cmd = ["bash", "-lc", filled]

    stdout = logs_root / f"{job_id}.out"

    stderr = logs_root / f"{job_id}.err"

    job = PhysicsJob(

        job_id=job_id,

        kind="rpx",

        command=cmd,

        work_dir=work,

        stdout_path=stdout,

        stderr_path=stderr,

        scheduler=str(jobs_cfg.get("scheduler", "local")),

        structure_model_id=structure_model_id,

        physics_scan_id=physics_scan_id,

        metadata={

            "mutation": mutation,

            "structure_pdb": str(structure_pdb),

            "out_dir": str(out_dir),

            "executable": executable,

            "score_file": rpx_cfg.get("output_score_filename", "rpx_scores.tsv"),

        },

    )

    write_shell_script(work / "run.sh", cmd, module_loads=list(jobs_cfg.get("module_loads") or []))

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





def mock_rpx_scores_for_mutations(

    mutations: list[dict[str, Any]],

    *,

    structure_model_id: str,

    physics_cfg: dict[str, Any],

    seed: int = 42,

) -> list[dict[str, Any]]:

    """Generate deterministic pseudo-RPX scores for orchestration dry-runs.



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

        Mock score rows with ``rpx`` and metadata.

    """

    mock = physics_cfg.get("mock", {})

    base = float(mock.get("base_rpx", -5.0))

    noise_sd = float(mock.get("structure_noise_sd", 0.4))

    rng = np.random.default_rng(_stable_seed(structure_model_id, salt=seed + 7))

    rows = []

    for mut in mutations:

        # Mild favorable packing for Arg at control sites

        bonus = -0.8 if mut["mutation"] in {"Q324R", "A355R"} else 0.0

        rpx = base + bonus + float(rng.normal(0.0, noise_sd))

        rows.append(

            {

                "mutation": mut["mutation"],

                "position": mut["position"],

                "wt": mut["wt"],

                "mutant": mut["mutant"],

                "version": mut.get("version"),

                "structure_model_id": structure_model_id,

                "rpx": rpx,

                "backend": "mock",

            }

        )

    return rows





def submit_or_mock_rpx_batch(

    *,

    structure_model_id: str,

    structure_pdb: Path,

    mutations: list[dict[str, Any]],

    physics_scan_id: str,

    physics_cfg: dict[str, Any],

    physics_root: Path,

    seed: int = 42,

) -> tuple[PhysicsJob, pd.DataFrame]:

    """Run one batched RPX job per structure model covering the mutation list.



    Parameters
    ----------
    structure_model_id : str

        Structure model identifier.

    structure_pdb : pathlib.Path

        Input structure PDB path.

    mutations : list of dict

        Mutation specs to score.

    physics_scan_id : str

        Scan batch identifier.

    physics_cfg : dict

        Physics configuration from ``physics.yaml``.

    physics_root : pathlib.Path

        Root directory for physics outputs.

    seed : int, optional

        Random seed for mock backend (default 42).



    Returns
    -------
    job : PhysicsJob

        Completed or dry-run job record.

    scores : pandas.DataFrame

        RPX score table for all mutations.

    """

    jobs_root = physics_root / physics_cfg.get("jobs", {}).get("jobs_subdir", "jobs")

    logs_root = physics_root / physics_cfg.get("jobs", {}).get("logs_subdir", "logs")

    out_dir = physics_root / "rpx" / structure_model_id / physics_scan_id

    job = build_rpx_job(

        structure_model_id=structure_model_id,

        mutation="BATCH",

        structure_pdb=structure_pdb,

        out_dir=out_dir,

        physics_scan_id=physics_scan_id,

        physics_cfg=physics_cfg,

        jobs_root=jobs_root,

        logs_root=logs_root,

    )

    backend = str(physics_cfg.get("backend", "mock"))

    executable = physics_cfg.get("rpx", {}).get("executable")

    score_path = out_dir / job.metadata.get("score_file", "rpx_scores.tsv")



    if backend == "external" and executable:

        mut_path = out_dir / "mutations.json"
        mut_path.write_text(
            json.dumps({"mutations": mutations}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        job.metadata["mutations_json"] = str(mut_path)
        job = run_local_job(job, dry_run=False)

        scores = pd.read_csv(score_path, sep="\t") if score_path.exists() else pd.DataFrame()

        return job, scores



    rows = mock_rpx_scores_for_mutations(

        mutations,

        structure_model_id=structure_model_id,

        physics_cfg=physics_cfg,

        seed=seed,

    )

    write_mock_rpx_scores(score_path, rows)

    job.status = "dry_run"

    job.returncode = 0

    job.completed_at = datetime.now(tz=UTC).isoformat()

    job.stdout_path.parent.mkdir(parents=True, exist_ok=True)

    job.stdout_path.write_text(

        json.dumps({"backend": "mock", "n_rows": len(rows)}, indent=2),

        encoding="utf-8",

    )

    job.stderr_path.write_text("", encoding="utf-8")

    job.write_sidecar()

    verify_job_completion(job, required_outputs=[score_path])

    return job, pd.DataFrame(rows)


