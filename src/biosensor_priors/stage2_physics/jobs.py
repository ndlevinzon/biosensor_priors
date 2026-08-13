"""Shared job records, script writing, and submit/verify helpers for Stage 2."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["pending", "submitted", "running", "completed", "failed", "dry_run"]


@dataclass
class PhysicsJob:
    """One external (or mock) physics job with full provenance hooks."""

    job_id: str
    kind: str  # rif | rpx | ligand | scan
    command: list[str]
    work_dir: Path
    stdout_path: Path
    stderr_path: Path
    status: JobStatus = "pending"
    scheduler: str = "local"
    structure_model_id: str | None = None
    physics_scan_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    completed_at: str | None = None
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the job record to a JSON-compatible dict.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Job fields with paths and commands converted to plain types.
        """
        d = asdict(self)
        d["work_dir"] = str(self.work_dir)
        d["stdout_path"] = str(self.stdout_path)
        d["stderr_path"] = str(self.stderr_path)
        d["command"] = list(self.command)
        return d

    def write_sidecar(self) -> Path:
        """Write ``job.json`` provenance sidecar in the work directory.

        Parameters
        ----------
        None

        Returns
        -------
        pathlib.Path
            Path to the written sidecar file.
        """
        path = self.work_dir / "job.json"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def write_shell_script(
    path: Path,
    command: list[str] | str,
    *,
    module_loads: list[str] | None = None,
    shebang: str = "#!/usr/bin/env bash",
    extra_exports: dict[str, str] | None = None,
) -> Path:
    """Write an executable bash script for local or sbatch use.

    Parameters
    ----------
    path : pathlib.Path
        Destination script path.
    command : list of str or str
        Command to execute in the script body.
    module_loads : list of str, optional
        Environment modules to load before running.
    shebang : str, optional
        Shebang line (default ``#!/usr/bin/env bash``).
    extra_exports : dict, optional
        Environment variables exported before the command.

    Returns
    -------
    pathlib.Path
        ``path`` after writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [shebang, "set -euo pipefail", ""]
    for mod in module_loads or []:
        lines.append(f"module load {shlex.quote(mod)}")
    if module_loads:
        lines.append("")
    for k, v in (extra_exports or {}).items():
        lines.append(f"export {k}={shlex.quote(v)}")
    if extra_exports:
        lines.append("")
    if isinstance(command, list):
        lines.append(" ".join(shlex.quote(c) for c in command))
    else:
        lines.append(command)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass
    return path


def write_sbatch_script(
    path: Path,
    command: list[str] | str,
    *,
    job_name: str,
    walltime: str = "04:00:00",
    cpus: int = 4,
    mem_gb: int = 16,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    gres: str | None = None,
    module_loads: list[str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> Path:
    """Write a Slurm sbatch script (submission is opt-in).

    Parameters
    ----------
    path : pathlib.Path
        Destination sbatch script path.
    command : list of str or str
        Command to run inside the Slurm allocation.
    job_name : str
        Slurm job name.
    walltime : str, optional
        Wall-clock limit (default ``04:00:00``).
    cpus : int, optional
        CPUs per task (default 4).
    mem_gb : int, optional
        Memory in GB (default 16).
    partition : str, optional
        Slurm partition name.
    account : str, optional
        Slurm account string.
    qos : str, optional
        Slurm quality-of-service (required on Granite).
    gres : str, optional
        Slurm generic resources (e.g. ``gpu:1`` for RF3 docking).
    module_loads : list of str, optional
        Environment modules to load.
    stdout_path : pathlib.Path, optional
        Slurm stdout file path.
    stderr_path : pathlib.Path, optional
        Slurm stderr file path.

    Returns
    -------
    pathlib.Path
        ``path`` after writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if stdout_path:
        lines.append(f"#SBATCH --output={stdout_path}")
    if stderr_path:
        lines.append(f"#SBATCH --error={stderr_path}")
    lines.extend(["", "set -euo pipefail", ""])
    for mod in module_loads or []:
        lines.append(f"module load {shlex.quote(mod)}")
    if module_loads:
        lines.append("")
    if isinstance(command, list):
        lines.append(" ".join(shlex.quote(c) for c in command))
    else:
        lines.append(command)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_local_job(job: PhysicsJob, *, dry_run: bool = False) -> PhysicsJob:
    """Execute a job locally, capturing stdout/stderr.

    Parameters
    ----------
    job : PhysicsJob
        Job record with command and log paths.
    dry_run : bool, optional
        When True, skip execution and mark status ``dry_run``.

    Returns
    -------
    PhysicsJob
        Updated job with status, return code, and timestamps.
    """
    job.work_dir.mkdir(parents=True, exist_ok=True)
    job.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    job.stderr_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        job.status = "dry_run"
        job.returncode = 0
        job.completed_at = datetime.now(tz=UTC).isoformat()
        job.stdout_path.write_text("DRY_RUN\n", encoding="utf-8")
        job.stderr_path.write_text("", encoding="utf-8")
        job.write_sidecar()
        return job

    job.status = "running"
    job.write_sidecar()
    try:
        with job.stdout_path.open("w", encoding="utf-8") as out, job.stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(
                job.command,
                cwd=str(job.work_dir),
                stdout=out,
                stderr=err,
                check=False,
            )
        job.returncode = int(proc.returncode)
        job.status = "completed" if proc.returncode == 0 else "failed"
    except OSError as exc:
        job.returncode = -1
        job.status = "failed"
        job.stderr_path.write_text(str(exc), encoding="utf-8")
    job.completed_at = datetime.now(tz=UTC).isoformat()
    job.write_sidecar()
    return job


def verify_job_completion(job: PhysicsJob, *, required_outputs: list[Path] | None = None) -> dict[str, Any]:
    """Check status sidecar and optional required output files.

    Parameters
    ----------
    job : PhysicsJob
        Job to verify.
    required_outputs : list of pathlib.Path, optional
        Output files that must exist for success.

    Returns
    -------
    dict
        Verification result with ``passed``, ``checks``, and ``job_id``.
    """
    checks = []
    ok_status = job.status in {"completed", "dry_run"}
    checks.append({"name": "status_ok", "passed": ok_status, "status": job.status})
    if job.returncode is not None:
        checks.append(
            {
                "name": "returncode_zero",
                "passed": job.returncode == 0,
                "returncode": job.returncode,
            }
        )
    for path in required_outputs or []:
        checks.append({"name": f"exists:{path.name}", "passed": path.exists(), "path": str(path)})
    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks, "job_id": job.job_id}


def load_job(work_dir: Path) -> PhysicsJob:
    """Load a :class:`PhysicsJob` from its ``job.json`` sidecar.

    Parameters
    ----------
    work_dir : pathlib.Path
        Job work directory containing ``job.json``.

    Returns
    -------
    PhysicsJob
        Reconstructed job record.
    """
    data = json.loads((work_dir / "job.json").read_text(encoding="utf-8"))
    return PhysicsJob(
        job_id=data["job_id"],
        kind=data["kind"],
        command=list(data["command"]),
        work_dir=Path(data["work_dir"]),
        stdout_path=Path(data["stdout_path"]),
        stderr_path=Path(data["stderr_path"]),
        status=data.get("status", "pending"),
        scheduler=data.get("scheduler", "local"),
        structure_model_id=data.get("structure_model_id"),
        physics_scan_id=data.get("physics_scan_id"),
        metadata=data.get("metadata") or {},
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
        returncode=data.get("returncode"),
    )
