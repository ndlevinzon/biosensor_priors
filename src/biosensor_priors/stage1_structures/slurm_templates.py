"""CHPC Utah SLURM script templates for Boltz-2, AlphaFold3, ESMFold, and RF3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from biosensor_priors.common.config import REPO_ROOT, resolve_path


def _sbatch_header(
    *,
    job_name: str,
    partition: str,
    account: str,
    ntasks: int,
    nodes: int,
    mem: str,
    time: str,
    gres: str | None = None,
    qos: str | None = None,
    export_all: bool = False,
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -t {time}",
        f"#SBATCH -n {ntasks}",
        f"#SBATCH -N {nodes}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {account}",
        f"#SBATCH --mem={mem}",
    ]
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if export_all:
        lines.append("#SBATCH --export=ALL")
    if stdout_path is not None:
        out = Path(stdout_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f"#SBATCH --output={out.as_posix()}")
    if stderr_path is not None:
        err = Path(stderr_path)
        err.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f"#SBATCH --error={err.as_posix()}")
    lines.append("")
    return lines


def resolve_slurm_log_paths(
    logs_dir: Path | str | None,
    *,
    structure_model_id: str,
    script_stem: str,
) -> tuple[Path | None, Path | None]:
    """Return ``(stdout, stderr)`` under ``logs_dir`` for one SLURM script."""
    if logs_dir is None:
        return None, None
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    base = f"{structure_model_id}__{script_stem}"
    return logs / f"{base}.out", logs / f"{base}.err"


def write_af2_step1_script(
    path: Path,
    *,
    fasta_file: Path | str,
    output_dir: Path | str,
    af2_cfg: dict[str, Any],
    step2_script: Path | str | None = None,
    chain_gpu: bool = True,
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """
    Write CHPC AF2 step-1 (CPU MSA / features) SLURM script.

    Mirrors ``run_alphafold_chpc_232.slr``: load module, copy DBs to RAM disk,
    run ``run_alphafold_full.sh ... --run_feature=1``, optionally submit step 2.
    """
    step = af2_cfg["step1"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step["partition"]),
        account=str(step["account"]),
        ntasks=int(step["ntasks"]),
        nodes=int(step["nodes"]),
        mem=str(step["mem"]),
        time=str(step["time"]),
        gres=step.get("gres"),
        qos=step.get("qos"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {af2_cfg['module']}",
            "",
            f'export FASTA_FILE="{Path(fasta_file).as_posix()}"',
            f'export OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            f'SCRDB="{af2_cfg["scratch_db"]}"',
            'TMPDB="/tmp/${SLURM_JOBID}"',
            "",
        ]
    )
    if chain_gpu and step2_script is not None:
        lines.extend(
            [
                f'sbatch -d afterok:${{SLURM_JOBID}} "{Path(step2_script).as_posix()}"',
                "",
            ]
        )
    db_script = (
        af2_cfg["db_to_tmp_reduced_script"]
        if str(af2_cfg.get("db_preset", "full_dbs")) == "reduced_dbs"
        else af2_cfg["db_to_tmp_script"]
    )
    lines.append(f"{db_script}")
    lines.append("")

    use_gpu = "--use_gpu_relax" if af2_cfg.get("use_gpu_relax", True) else ""
    max_date = af2_cfg.get("max_template_date", "2022-01-01")
    preset = af2_cfg.get("model_preset", "monomer")
    if preset == "multimer" or str(af2_cfg.get("db_preset")) == "reduced_dbs":
        runner = af2_cfg.get("run_reduced", "run_alphafold_red.sh")
        lines.append(
            f"{runner} {use_gpu} --fasta_paths=$FASTA_FILE --output_dir=$OUTPUT_DIR "
            f"--max_template_date={max_date} --model_preset={preset} "
            f"--db_preset=reduced_dbs --run_feature=1"
        )
    else:
        runner = af2_cfg.get("run_full", "run_alphafold_full.sh")
        lines.append(
            f"{runner} {use_gpu} --fasta_paths=$FASTA_FILE --output_dir=$OUTPUT_DIR "
            f"--max_template_date={max_date} --run_feature=1"
        )
    lines.extend(["", "rm -rf \"$TMPDB\"", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_af2_step2_script(
    path: Path,
    *,
    fasta_file: Path | str,
    output_dir: Path | str,
    af2_cfg: dict[str, Any],
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """Write CHPC AF2 step-2 (GPU inference / relax) SLURM script."""
    step = af2_cfg["step2"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step["partition"]),
        account=str(step["account"]),
        ntasks=int(step["ntasks"]),
        nodes=int(step["nodes"]),
        mem=str(step["mem"]),
        time=str(step["time"]),
        gres=step.get("gres"),
        qos=step.get("qos"),
        export_all=True,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {af2_cfg['module']}",
            "",
            "# FASTA_FILE / OUTPUT_DIR inherited when chained; set fallbacks:",
            f': "${{FASTA_FILE:={Path(fasta_file).as_posix()}}}"',
            f': "${{OUTPUT_DIR:={Path(output_dir).as_posix()}}}"',
            f'SCRDB="{af2_cfg["scratch_db"]}"',
            f'TMPDB="{af2_cfg["scratch_db"]}"',
            "",
        ]
    )
    use_gpu = "--use_gpu_relax" if af2_cfg.get("use_gpu_relax", True) else ""
    max_date = af2_cfg.get("max_template_date", "2022-01-01")
    preset = af2_cfg.get("model_preset", "monomer")
    if preset == "multimer" or str(af2_cfg.get("db_preset")) == "reduced_dbs":
        runner = af2_cfg.get("run_reduced", "run_alphafold_red.sh")
        lines.append(
            f"{runner} {use_gpu} --fasta_paths=$FASTA_FILE --output_dir=$OUTPUT_DIR "
            f"--max_template_date={max_date} --model_preset={preset} --db_preset=reduced_dbs"
        )
    else:
        runner = af2_cfg.get("run_full", "run_alphafold_full.sh")
        lines.append(
            f"{runner} {use_gpu} --fasta_paths=$FASTA_FILE --output_dir=$OUTPUT_DIR "
            f"--max_template_date={max_date} "
            f"--data_dir=$SCRDB "
            f"--uniref90_database_path=$SCRDB/uniref90/uniref90.fasta "
            f"--uniref30_database_path=$TMPDB/uniref30/UniRef30_2021_03 "
            f"--mgnify_database_path=$SCRDB/mgnify/mgy_clusters_2022_05.fa "
            f"--bfd_database_path=$TMPDB/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt "
            f"--pdb70_database_path=$TMPDB/pdb70/pdb70 "
            f"--template_mmcif_dir=$SCRDB/pdb_mmcif/mmcif_files "
            f"--obsolete_pdbs_path=$SCRDB/pdb_mmcif/obsolete.dat"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_af3_step1_script(
    path: Path,
    *,
    input_json: Path | str,
    output_dir: Path | str,
    af3_cfg: dict[str, Any],
    step2_script: Path | str | None = None,
    chain_gpu: bool = True,
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """Write CHPC AF3 step-1 (CPU MSA, ``--norun_inference``) SLURM script."""
    step = af3_cfg["step1"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step["partition"]),
        account=str(step["account"]),
        ntasks=int(step["ntasks"]),
        nodes=int(step["nodes"]),
        mem=str(step.get("mem", "64G")),
        time=str(step["time"]),
        gres=step.get("gres"),
        qos=step.get("qos"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {af3_cfg['module']}",
            "",
            f'export INPUT_FILE="{Path(input_json).as_posix()}"',
            f'export OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            "",
        ]
    )
    if chain_gpu and step2_script is not None:
        lines.extend(
            [
                f'sbatch -d afterok:${{SLURM_JOBID}} "{Path(step2_script).as_posix()}"',
                "",
            ]
        )
    runner = af3_cfg.get("run", "run_alphafold.sh")
    lines.append(
        f"{runner} --json_path=$INPUT_FILE --output_dir=$OUTPUT_DIR --norun_inference"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_af3_step2_script(
    path: Path,
    *,
    data_json: Path | str,
    output_dir: Path | str,
    af3_cfg: dict[str, Any],
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """
    Write CHPC AF3 step-2 (GPU inference, ``--norun_data_pipeline``) script.

    ``data_json`` should be the MSA result JSON from step 1
    (e.g. ``out/<name>/<name>_data.json``).
    """
    step = af3_cfg["step2"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step["partition"]),
        account=str(step["account"]),
        ntasks=int(step["ntasks"]),
        nodes=int(step["nodes"]),
        mem=str(step["mem"]),
        time=str(step["time"]),
        gres=step.get("gres"),
        qos=step.get("qos"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {af3_cfg['module']}",
            "",
            f'INPUT_FILE="{Path(data_json).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            "",
        ]
    )
    runner = af3_cfg.get("run", "run_alphafold.sh")
    extra = ""
    flash = af3_cfg.get("flash_attention_implementation")
    if flash:
        extra = f" --flash_attention_implementation={flash}"
    lines.append(
        f"{runner} --json_path=$INPUT_FILE --output_dir=$OUTPUT_DIR "
        f"--norun_data_pipeline{extra}"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_esmfold_script(
    path: Path,
    *,
    fasta_file: Path | str,
    output_dir: Path | str,
    esm_cfg: dict[str, Any],
    runner_script: Path | str | None = None,
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """
    Write a single-step CHPC ESMFold GPU SLURM script.

    Loads ``esmfold/1.0.3`` and runs the fair-esm **Python API** via
    :mod:`biosensor_priors.stage1_structures.run_esmfold` (not ``esm-fold`` CLI).
    """
    step = esm_cfg.get("job") or esm_cfg.get("step2") or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if runner_script is None:
        # Resolve installed / source path so module Python can exec the file.
        from biosensor_priors.stage1_structures import run_esmfold as _run_esmfold

        runner = Path(_run_esmfold.__file__).resolve()
    else:
        runner = Path(runner_script).resolve()

    chunk = esm_cfg.get("chunk_size", 128)
    recycles = esm_cfg.get("num_recycles", 4)
    device = str(esm_cfg.get("device", "cuda"))
    python_bin = str(esm_cfg.get("python", "python"))

    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step.get("partition", "granite-gpu")),
        account=str(step.get("account", "cheatham")),
        ntasks=int(step.get("ntasks", 4)),
        nodes=int(step.get("nodes", 1)),
        mem=str(step.get("mem", "32G")),
        time=str(step.get("time", "4:00:00")),
        gres=step.get("gres", "gpu:1"),
        qos=step.get("qos", "granite-gpu"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {esm_cfg['module']}",
            "",
            f'FASTA_FILE="{Path(fasta_file).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            f'ESMFOLD_PY="{runner.as_posix()}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
            "# fair-esm Python API (esm.pretrained.esmfold_v1), not esm-fold CLI",
            f'{python_bin} "$ESMFOLD_PY" \\',
            '  --fasta "$FASTA_FILE" \\',
            '  --out "$OUTPUT_DIR" \\',
            f"  --chunk-size {int(chunk) if chunk is not None else 0} \\",
            f"  --num-recycles {int(recycles) if recycles is not None else 4} \\",
            f"  --device {device}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_boltz2_script(
    path: Path,
    *,
    input_yaml: Path | str,
    output_dir: Path | str,
    boltz_cfg: dict[str, Any],
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """
    Write a CHPC Boltz-2 GPU SLURM script.

    Loads ``boltz2`` and runs ``boltz predict`` with the CHPC ColabFold MSA
    server (see CHPC Alphafold docs).
    """
    step = boltz_cfg.get("job") or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step.get("partition", "granite-gpu")),
        account=str(step.get("account", "cheatham")),
        ntasks=int(step.get("ntasks", 8)),
        nodes=int(step.get("nodes", 1)),
        mem=str(step.get("mem", "64G")),
        time=str(step.get("time", "8:00:00")),
        gres=step.get("gres", "gpu:1"),
        qos=step.get("qos", "granite-gpu"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    runner = str(boltz_cfg.get("run", "boltz"))
    msa_url = str(
        boltz_cfg.get("msa_server_url")
        or "http://colabfold02.int.chpc.utah.edu:8088"
    )
    extras: list[str] = [
        f'--out_dir "{Path(output_dir).as_posix()}"',
    ]
    if boltz_cfg.get("use_msa_server", True):
        extras.append("--use_msa_server")
        extras.append(f"--msa_server_url={msa_url}")
    if boltz_cfg.get("diffusion_samples") is not None:
        extras.append(f"--diffusion_samples {int(boltz_cfg['diffusion_samples'])}")
    if boltz_cfg.get("recycling_steps") is not None:
        extras.append(f"--recycling_steps {int(boltz_cfg['recycling_steps'])}")
    if boltz_cfg.get("output_format"):
        extras.append(f"--output_format {boltz_cfg['output_format']}")
    if boltz_cfg.get("override", True):
        extras.append("--override")
    for arg in boltz_cfg.get("extra_args") or []:
        extras.append(str(arg))
    extra = " ".join(extras)
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {boltz_cfg['module']}",
            "",
            f'INPUT_YAML="{Path(input_yaml).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
            f'{runner} predict "$INPUT_YAML" {extra}',
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_rf3_script(
    path: Path,
    *,
    input_json: Path | str,
    output_dir: Path | str,
    rf3_cfg: dict[str, Any],
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
) -> Path:
    """
    Write a GPU SLURM script for RoseTTAFold3 (``rf3 fold`` / Foundry).

    CHPC does not ship RF3 as a module yet; use ``conda_activate`` or ensure
    ``rf3`` is on PATH after optional ``module`` load.
    """
    step = rf3_cfg.get("job") or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sbatch_header(
        job_name=path.stem[:64],
        partition=str(step.get("partition", "granite-gpu")),
        account=str(step.get("account", "cheatham")),
        ntasks=int(step.get("ntasks", 8)),
        nodes=int(step.get("nodes", 1)),
        mem=str(step.get("mem", "64G")),
        time=str(step.get("time", "12:00:00")),
        gres=step.get("gres", "gpu:1"),
        qos=step.get("qos", "granite-gpu"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    lines.extend(["set -euo pipefail", "ml purge"])
    if rf3_cfg.get("module"):
        lines.append(f"ml {rf3_cfg['module']}")
    conda_act = rf3_cfg.get("conda_activate")
    if conda_act:
        lines.extend(
            [
                f'source "{Path(str(conda_act)).as_posix()}"',
            ]
        )
    runner = str(rf3_cfg.get("run", "rf3"))
    early = rf3_cfg.get("early_stopping_plddt_threshold", 0.0)
    hydra_args = [
        f"inputs='{Path(input_json).as_posix()}'",
        f"out_dir='{Path(output_dir).as_posix()}'",
        f"early_stopping_plddt_threshold={float(early)}",
    ]
    ckpt = rf3_cfg.get("ckpt_path")
    if ckpt:
        ckpt_abs = resolve_path(str(ckpt), REPO_ROOT)
        hydra_args.append(f"ckpt_path='{ckpt_abs.as_posix()}'")
    for arg in rf3_cfg.get("extra_args") or []:
        hydra_args.append(str(arg))
    lines.extend(
        [
            "",
            'if ! command -v nvidia-smi >/dev/null 2>&1; then',
            '  echo "ERROR: nvidia-smi not found — need a GPU allocation" >&2',
            "  exit 1",
            "fi",
            "nvidia-smi",
            "",
            f'INPUT_JSON="{Path(input_json).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
            f"{runner} fold " + " ".join(hydra_args),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
