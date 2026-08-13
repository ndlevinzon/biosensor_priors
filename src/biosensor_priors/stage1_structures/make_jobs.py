"""Generate CHPC structure-prediction inputs and SLURM job scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage1_structures.slurm_templates import (
    resolve_slurm_log_paths,
    write_af3_step1_script,
    write_af3_step2_script,
    write_boltz2_script,
    write_esmfold_script,
    write_rf3_script,
)


def canonicalize_method(method: str) -> str:
    """Map predictor aliases to canonical Stage-1 method labels."""
    key = str(method).strip().upper().replace(" ", "").replace("_", "").replace("-", "")
    lookup = {
        "BOLTZ2": "Boltz2",
        "BOLTZ": "Boltz2",
        "AF3": "AF3",
        "ALPHAFOLD3": "AF3",
        "ESMFOLD": "ESMFold",
        "ESM": "ESMFold",
        "RF3": "RF3",
        "ROSETTAFOLD3": "RF3",
    }
    if key not in lookup:
        raise ValueError(
            f"Unknown structure predictor {method!r}. "
            f"Supported: Boltz2, AF3, ESMFold, RF3 "
            f"(AF2 and RF2 were replaced by Boltz2 and RF3)"
        )
    return lookup[key]


def structure_model_id(version: str, method: str, seed: int, state: str) -> str:
    """Build the canonical ``structure_model_id`` string.

    Parameters
    ----------
    version : str
        Sequence background (e.g. ``V2.4``).
    method : str
        Predictor label (``AF2``, ``AF3``, …).
    seed : int
        Model seed.
    state : str
        Ligand / conformational state (``apo``, ``AcCoA``, …).

    Returns
    -------
    str
        Identifier ``{version}_{method}_seed{seed}_{state}``.
    """
    return f"{version}_{method}_seed{int(seed)}_{state}"


def sanitize_af3_name(name: str) -> str:
    """Sanitize a job name for AlphaFold 3 JSON ``name`` field.

    Parameters
    ----------
    name : str
        Raw identifier (typically ``structure_model_id``).

    Returns
    -------
    str
        Alphanumeric / underscore / hyphen name safe for AF3 directories.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    return cleaned.strip("_") or "job"


def write_boltz2_yaml(path: Path, *, sequence: str, chain_id: str = "A") -> Path:
    """Write a Boltz-2 protein YAML input (preferred over FASTA)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": "".join(str(sequence).split()),
                }
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_rf3_input_json(
    path: Path,
    *,
    name: str,
    sequence: str,
    chain_id: str = "A",
    msa_path: str | None = None,
) -> Path:
    """Write a RoseTTAFold3 Foundry JSON input (sequence ± optional MSA)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    component: dict[str, Any] = {
        "seq": "".join(str(sequence).split()),
        "chain_id": chain_id,
    }
    if msa_path:
        component["msa_path"] = str(msa_path)
    payload = {"name": name, "components": [component]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_fasta(path: Path, *, header: str, sequence: str) -> Path:
    """Write a single-sequence FASTA file (ESMFold / generic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = "".join(str(sequence).split()).upper().rstrip("*")
    lines = [f">{header}"]
    for i in range(0, len(seq), 80):
        lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_af3_input_json(
    path: Path,
    *,
    name: str,
    sequence: str,
    seed: int,
    dialect: str = "alphafold3",
    version: int = 1,
    chain_id: str = "A",
) -> Path:
    """Write an AlphaFold 3 input JSON (protein-only / apo).

    Parameters
    ----------
    path : pathlib.Path
        Destination JSON path.
    name : str
        AF3 job name (also drives output subdirectory naming).
    sequence : str
        Protein sequence.
    seed : int
        Value for ``modelSeeds``.
    dialect : str, optional
        AF3 dialect string (default ``alphafold3``).
    version : int, optional
        AF3 JSON schema version (default 1).
    chain_id : str, optional
        Protein chain ID (default ``A``).

    Returns
    -------
    pathlib.Path
        Written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = "".join(str(sequence).split()).upper().rstrip("*")
    payload = {
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": seq,
                }
            }
        ],
        "modelSeeds": [int(seed)],
        "dialect": dialect,
        "version": int(version),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_version_sequence(
    *,
    version: str,
    repo_root: Path | None = None,
    versions: pd.DataFrame | None = None,
) -> str:
    """Resolve a protein sequence for a design background version.

    Parameters
    ----------
    version : str
        Version label (e.g. ``V2.4``).
    repo_root : pathlib.Path, optional
        Repository root for pickle lookup.
    versions : pandas.DataFrame, optional
        Pre-loaded version table; skips pickle I/O when provided.

    Returns
    -------
    str
        Clean protein sequence.

    Raises
    ------
    FileNotFoundError
        If the versions pickle is missing and ``versions`` was not passed.
    KeyError
        If ``version`` is not present in the table.
    ValueError
        If the sequence is empty.
    """
    root = repo_root or REPO_ROOT
    if versions is None:
        pipeline = load_yaml(root / "configs" / "pipeline.yaml")
        constructs = resolve_path(pipeline["paths"]["constructs"], root)
        pickle_path = constructs / pipeline["constructs"]["versions_pickle"]
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Missing versions pickle: {pickle_path}. "
                "Run Stage 0 or pass an explicit sequence / versions table."
            )
        versions = pd.read_pickle(pickle_path)

    ver_col = "Version" if "Version" in versions.columns else "version"
    seq_col = (
        "Sequence_clean"
        if "Sequence_clean" in versions.columns
        else ("Sequence" if "Sequence" in versions.columns else "sequence")
    )
    row = versions[versions[ver_col].astype(str) == str(version)]
    if row.empty:
        raise KeyError(f"Version {version!r} not found in versions table")
    seq = str(row.iloc[0][seq_col])
    seq = "".join(seq.split()).upper().rstrip("*")
    if not seq:
        raise ValueError(f"Empty sequence for version {version!r}")
    return seq


def _af3_data_json_path(output_dir: Path, af3_name: str) -> Path:
    """Expected AF3 MSA result JSON path after step 1."""
    stem = af3_name.lower()
    return Path(output_dir) / stem / f"{stem}_data.json"


def make_structure_jobs(
    *,
    version: str | None = None,
    sequence: str | None = None,
    repo_root: Path | None = None,
    structures_cfg: dict[str, Any] | None = None,
    predictors: list[str] | None = None,
    seeds: list[int] | None = None,
    states: list[str] | None = None,
    submit: bool | None = None,
) -> dict[str, Any]:
    """Generate FASTA/JSON inputs and CHPC two-step SLURM scripts.

    Does not run AlphaFold. Writes a job registry parquet and optional
    ``submit_all.sh`` that ``sbatch``es step-1 scripts (step 2 is chained
    via ``sbatch -d afterok:$SLURM_JOBID`` inside step 1 when configured).

    Parameters
    ----------
    version : str, optional
        Design background. Defaults to ``structures.default_version`` or
        ``pipeline.active_design_background``.
    sequence : str, optional
        Explicit sequence; when omitted, loaded from Stage-0 versions pickle.
    repo_root : pathlib.Path, optional
        Repository root.
    structures_cfg : dict, optional
        Parsed ``configs/structures.yaml``.
    predictors : list of str, optional
        Subset of ``AF2`` / ``AF3`` (others skipped with a note).
    seeds : list of int, optional
        Model seeds.
    states : list of str, optional
        Conformational / ligand states (apo-only AF3 JSON for now).
    submit : bool, optional
        When True, attempt ``sbatch`` of each step-1 script (HPC only).

    Returns
    -------
    dict
        Keys ``registry``, ``registry_path``, ``jobs_dir``, ``version``,
        ``n_jobs``, ``submit_script``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    cfg = structures_cfg or load_yaml(root / "configs" / "structures.yaml")

    version = version or cfg.get("default_version") or pipeline.get(
        "active_design_background", "V2.4"
    )
    version = str(version)

    thr_struct = thresholds.get("structure", {})
    predictors = list(predictors or cfg.get("predictors") or thr_struct.get("predictors") or ["Boltz2", "AF3"])
    seeds = [int(s) for s in (seeds or cfg.get("seeds") or thr_struct.get("seeds") or [1, 2, 3])]
    states = list(states or cfg.get("states") or ["apo"])

    if sequence is None:
        sequence = load_version_sequence(version=version, repo_root=root)
    else:
        sequence = "".join(str(sequence).split()).upper().rstrip("*")

    structures_root = resolve_path(pipeline["paths"]["structures"], root)
    jobs_cfg = cfg.get("jobs", {})
    jobs_dir = structures_root / str(jobs_cfg.get("jobs_subdir", "jobs")) / version
    logs_dir = structures_root / str(jobs_cfg.get("logs_subdir", "logs")) / version
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    chain_gpu = bool(jobs_cfg.get("chain_gpu_after_msa", True))
    do_submit = bool(jobs_cfg.get("submit", False) if submit is None else submit)

    af3_cfg = dict(cfg.get("af3", {}))
    esm_cfg = dict(cfg.get("esmfold", {}))
    boltz_cfg = dict(cfg.get("boltz2", {}))
    rf3_cfg = dict(cfg.get("rosettafold3", {}))

    rows: list[dict[str, Any]] = []
    step1_scripts: list[Path] = []

    for method in predictors:
        method_c = canonicalize_method(method)
        for seed in seeds:
            for state in states:
                mid = structure_model_id(version, method_c, seed, state)
                job_dir = jobs_dir / mid
                out_dir = structures_root / "raw" / mid
                job_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)

                note = None
                if state != "apo" and method_c in {"Boltz2", "AF3", "ESMFold", "RF3"}:
                    note = (
                        f"State {state!r}: protein-only input written; "
                        "ligand CCD/SMILES not yet wired."
                    )

                if method_c == "Boltz2":
                    input_yaml = write_boltz2_yaml(
                        job_dir / f"{mid}.yaml",
                        sequence=sequence,
                    )
                    script = job_dir / "boltz2_gpu.slurm"
                    stdout, stderr = resolve_slurm_log_paths(
                        logs_dir, structure_model_id=mid, script_stem=script.stem
                    )
                    write_boltz2_script(
                        script,
                        input_yaml=input_yaml.resolve(),
                        output_dir=out_dir.resolve(),
                        boltz_cfg=boltz_cfg,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    )
                    rows.append(
                        {
                            "structure_model_id": mid,
                            "version": version,
                            "method": method_c,
                            "seed": int(seed),
                            "state": state,
                            "input_path": str(input_yaml.relative_to(root)),
                            "output_dir": str(out_dir.relative_to(root)),
                            "step1_script": str(script.relative_to(root)),
                            "step2_script": None,
                            "status": "scripted",
                            "notes": note
                            or (
                                "Boltz-2 via CHPC boltz2 module + ColabFold MSA server; "
                                "seed is for ensemble bookkeeping only."
                            ),
                        }
                    )
                    step1_scripts.append(script)
                elif method_c == "AF3":
                    af3_name = sanitize_af3_name(mid)
                    input_json = write_af3_input_json(
                        job_dir / f"{af3_name}.json",
                        name=af3_name,
                        sequence=sequence,
                        seed=int(seed),
                        dialect=str(af3_cfg.get("dialect", "alphafold3")),
                        version=int(af3_cfg.get("version", 1)),
                    )
                    data_json = _af3_data_json_path(out_dir.resolve(), af3_name)
                    step2 = job_dir / "af3_step2_infer.slurm"
                    step1 = job_dir / "af3_step1_msa.slurm"
                    out2, err2 = resolve_slurm_log_paths(
                        logs_dir, structure_model_id=mid, script_stem=step2.stem
                    )
                    out1, err1 = resolve_slurm_log_paths(
                        logs_dir, structure_model_id=mid, script_stem=step1.stem
                    )
                    write_af3_step2_script(
                        step2,
                        data_json=data_json,
                        output_dir=out_dir.resolve(),
                        af3_cfg=af3_cfg,
                        stdout_path=out2,
                        stderr_path=err2,
                    )
                    write_af3_step1_script(
                        step1,
                        input_json=input_json.resolve(),
                        output_dir=out_dir.resolve(),
                        af3_cfg=af3_cfg,
                        step2_script=step2.resolve(),
                        chain_gpu=chain_gpu,
                        stdout_path=out1,
                        stderr_path=err1,
                    )
                    rows.append(
                        {
                            "structure_model_id": mid,
                            "version": version,
                            "method": method_c,
                            "seed": int(seed),
                            "state": state,
                            "af3_name": af3_name,
                            "input_path": str(input_json.relative_to(root)),
                            "output_dir": str(out_dir.relative_to(root)),
                            "expected_data_json": str(
                                Path(out_dir.relative_to(root))
                                / af3_name.lower()
                                / f"{af3_name.lower()}_data.json"
                            ),
                            "step1_script": str(step1.relative_to(root)),
                            "step2_script": str(step2.relative_to(root)),
                            "status": "scripted",
                            "notes": note,
                        }
                    )
                    step1_scripts.append(step1)
                elif method_c == "ESMFold":
                    fasta = write_fasta(
                        job_dir / f"{mid}.fasta",
                        header=mid,
                        sequence=sequence,
                    )
                    script = job_dir / "esmfold_gpu.slurm"
                    stdout, stderr = resolve_slurm_log_paths(
                        logs_dir, structure_model_id=mid, script_stem=script.stem
                    )
                    write_esmfold_script(
                        script,
                        fasta_file=fasta.resolve(),
                        output_dir=out_dir.resolve(),
                        esm_cfg=esm_cfg,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    )
                    rows.append(
                        {
                            "structure_model_id": mid,
                            "version": version,
                            "method": method_c,
                            "seed": int(seed),
                            "state": state,
                            "input_path": str(fasta.relative_to(root)),
                            "output_dir": str(out_dir.relative_to(root)),
                            "step1_script": str(script.relative_to(root)),
                            "step2_script": None,
                            "status": "scripted",
                            "notes": note
                            or (
                                "ESMFold via fair-esm Python API "
                                "(esm.pretrained.esmfold_v1); "
                                "seed is for ensemble bookkeeping only."
                            ),
                        }
                    )
                    step1_scripts.append(script)
                elif method_c == "RF3":
                    input_json = write_rf3_input_json(
                        job_dir / f"{mid}.json",
                        name=mid,
                        sequence=sequence,
                    )
                    script = job_dir / "rf3_gpu.slurm"
                    stdout, stderr = resolve_slurm_log_paths(
                        logs_dir, structure_model_id=mid, script_stem=script.stem
                    )
                    write_rf3_script(
                        script,
                        input_json=input_json.resolve(),
                        output_dir=out_dir.resolve(),
                        rf3_cfg=rf3_cfg,
                        stdout_path=stdout,
                        stderr_path=stderr,
                    )
                    rows.append(
                        {
                            "structure_model_id": mid,
                            "version": version,
                            "method": method_c,
                            "seed": int(seed),
                            "state": state,
                            "input_path": str(input_json.relative_to(root)),
                            "output_dir": str(out_dir.relative_to(root)),
                            "step1_script": str(script.relative_to(root)),
                            "step2_script": None,
                            "status": "scripted",
                            "notes": note
                            or (
                                "RoseTTAFold3 via Foundry `rf3 fold` "
                                "(install rc-foundry[rf3] if not on PATH). "
                                "Seed is for ensemble bookkeeping only."
                            ),
                        }
                    )
                    step1_scripts.append(script)

    registry = pd.DataFrame(rows)
    registry_path = structures_root / "job_registry.parquet"
    registry.to_parquet(registry_path, index=False)

    # Human-readable registry + submit helper
    registry.to_csv(structures_root / "job_registry.csv", index=False)
    submit_script = jobs_dir / "submit_all.sh"
    submit_lines = [
        "#!/bin/bash",
        "# Submit CHPC structure jobs (Boltz2 / AF3 step-1 / ESMFold / RF3).",
        "# AF3 step-2 GPU jobs are chained with sbatch -d afterok:$SLURM_JOBID when enabled.",
        "set -euo pipefail",
        "",
    ]
    for script in step1_scripts:
        submit_lines.append(f'sbatch "{script.resolve().as_posix()}"')
    submit_script.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")

    if do_submit:
        import subprocess

        for script in step1_scripts:
            subprocess.run(["sbatch", str(script)], check=True)

    return {
        "registry": registry,
        "registry_path": registry_path,
        "jobs_dir": jobs_dir,
        "version": version,
        "n_jobs": len(registry),
        "submit_script": submit_script,
        "structures_root": structures_root,
    }
