"""CHPC Utah SLURM script templates for AlphaFold, ESMFold, and RoseTTAFold2."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    lines.append("")
    return lines


def write_af2_step1_script(
    path: Path,
    *,
    fasta_file: Path | str,
    output_dir: Path | str,
    af2_cfg: dict[str, Any],
    step2_script: Path | str | None = None,
    chain_gpu: bool = True,
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
) -> Path:
    """
    Write a single-step CHPC ESMFold GPU SLURM script.

    Uses ``module load esmfold/1.0.3`` and the ``esm-fold`` CLI
    (``esm-fold -i FASTA -o PDB_DIR``).
    """
    step = esm_cfg.get("job") or esm_cfg.get("step2") or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {esm_cfg['module']}",
            "",
            f'FASTA_FILE="{Path(fasta_file).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
        ]
    )
    runner = esm_cfg.get("run", "esm-fold")
    extras: list[str] = []
    if esm_cfg.get("num_recycles") is not None:
        extras.append(f"--num-recycles {int(esm_cfg['num_recycles'])}")
    if esm_cfg.get("chunk_size") is not None:
        extras.append(f"--chunk-size {int(esm_cfg['chunk_size'])}")
    if esm_cfg.get("max_tokens_per_batch") is not None:
        extras.append(f"--max-tokens-per-batch {int(esm_cfg['max_tokens_per_batch'])}")
    if esm_cfg.get("cpu_offload"):
        extras.append("--cpu-offload")
    extra = (" " + " ".join(extras)) if extras else ""
    lines.append(f'{runner} -i "$FASTA_FILE" -o "$OUTPUT_DIR"{extra}')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_rosettafold2_script(
    path: Path,
    *,
    fasta_file: Path | str,
    output_dir: Path | str,
    rf2_cfg: dict[str, Any],
) -> Path:
    """
    Write a CHPC RoseTTAFold2 GPU SLURM script.

    Loads ``rosettafold2/1.0`` and runs ``run_RF2.sh FASTA -o OUTDIR``.
    """
    step = rf2_cfg.get("job") or {}
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
    )
    lines.extend(
        [
            "set -euo pipefail",
            "ml purge",
            f"ml {rf2_cfg['module']}",
            "",
            f'FASTA_FILE="{Path(fasta_file).as_posix()}"',
            f'OUTPUT_DIR="{Path(output_dir).as_posix()}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
        ]
    )
    runner = rf2_cfg.get("run", "run_RF2.sh")
    extras: list[str] = []
    if rf2_cfg.get("hhpred"):
        extras.append("--hhpred")
    if rf2_cfg.get("pair"):
        extras.append("--pair")
    symm = rf2_cfg.get("symm")
    if symm:
        extras.append(f"--symm {symm}")
    for arg in rf2_cfg.get("extra_args") or []:
        extras.append(str(arg))
    extra = (" " + " ".join(extras)) if extras else ""
    # Canonical RF2 usage: run_RF2.sh input.fasta -o outdir
    lines.append(f'{runner} "$FASTA_FILE" -o "$OUTPUT_DIR"{extra}')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
