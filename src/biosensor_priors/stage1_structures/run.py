"""Stage 1 orchestration: CHPC job scripts → adapters → confidence → Gate 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import write_manifest
from biosensor_priors.stage1_structures.adapters import ingest_job_registry
from biosensor_priors.stage1_structures.confidence import (
    compute_structural_confidence,
    write_structural_confidence,
)
from biosensor_priors.stage1_structures.gate1 import evaluate_gate1
from biosensor_priors.stage1_structures.make_jobs import make_structure_jobs


def run_stage1(
    *,
    repo_root: Path | None = None,
    version: str | None = None,
    sequence: str | None = None,
    make_jobs: bool = True,
    ingest: bool = True,
    predictors: list[str] | None = None,
    seeds: list[int] | None = None,
    states: list[str] | None = None,
    submit: bool = False,
) -> dict[str, Any]:
    """Run Stage 1 job generation and/or output ingestion.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root.
    version : str, optional
        Design background version.
    sequence : str, optional
        Explicit sequence (skips versions pickle).
    make_jobs : bool, optional
        When True, write AF2/AF3 inputs + SLURM scripts (default True).
    ingest : bool, optional
        When True, parse any available raw outputs into confidence tables
        (default True). Safe when outputs are not ready yet (empty tables).
    predictors, seeds, states
        Optional overrides for job generation.
    submit : bool, optional
        When True, ``sbatch`` step-1 scripts (HPC only).

    Returns
    -------
    dict
        Keys include ``jobs``, ``models``, ``residues``, ``confidence``,
        ``gate``, ``manifest_path``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    structures_cfg = load_yaml(root / "configs" / "structures.yaml")
    seed = int(pipeline.get("random_seed", 42))
    structures_root = resolve_path(pipeline["paths"]["structures"], root)
    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_result: dict[str, Any] | None = None
    if make_jobs:
        jobs_result = make_structure_jobs(
            version=version,
            sequence=sequence,
            repo_root=root,
            structures_cfg=structures_cfg,
            predictors=predictors,
            seeds=seeds,
            states=states,
            submit=submit,
        )
        registry = jobs_result["registry"]
    else:
        registry_path = structures_root / "job_registry.parquet"
        registry = (
            pd.read_parquet(registry_path)
            if registry_path.exists()
            else pd.DataFrame()
        )

    models = pd.DataFrame()
    residues = pd.DataFrame()
    confidence = pd.DataFrame()
    if ingest and not registry.empty:
        parsed = ingest_job_registry(registry, repo_root=root)
        models = parsed["models"]
        residues = parsed["residues"]
        models_path = structures_root / "structure_models.parquet"
        residues_path = structures_root / "structure_residues.parquet"
        models.to_parquet(models_path, index=False)
        residues.to_parquet(residues_path, index=False)
        confidence = compute_structural_confidence(
            models, residues, repo_root=root
        )
        write_structural_confidence(confidence, repo_root=root)

    # Job scripting without HPC outputs is expected on a laptop → do not fail Gate 1.
    # Ingest-only with zero models should fail so missing AF outputs are visible.
    require_models = ingest and not make_jobs
    gate = evaluate_gate1(
        registry=registry,
        models=models,
        confidence=confidence,
        repo_root=root,
        min_models=1 if require_models else 0,
    )
    if make_jobs and models.empty:
        gate = {
            **gate,
            "passed": True,
            "structure_gate": "PASS",
            "notes": "Jobs scripted; no structures ingested yet (expected before HPC).",
            "failed": [],
        }

    gate_path = out_dir / "gate1.json"
    gate_path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    (structures_root / "gate1.json").write_text(
        json.dumps(gate, indent=2, default=str), encoding="utf-8"
    )

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage1_manifest.json",
        stage="stage1_structures",
        inputs={
            "version": (jobs_result or {}).get("version") or version,
            "provider": structures_cfg.get("provider"),
            "n_jobs": int(len(registry)) if registry is not None else 0,
        },
        parameters={
            "random_seed": seed,
            "predictors": predictors or structures_cfg.get("predictors"),
            "seeds": seeds or structures_cfg.get("seeds"),
            "states": states or structures_cfg.get("states"),
            "make_jobs": make_jobs,
            "ingest": ingest,
            "submit": submit,
        },
        outputs={
            "job_registry": str((structures_root / "job_registry.parquet").relative_to(root))
            if (structures_root / "job_registry.parquet").exists()
            else None,
            "structural_confidence": str(
                (structures_root / "structural_confidence.parquet").relative_to(root)
            )
            if (structures_root / "structural_confidence.parquet").exists()
            else None,
            "gate1": str(gate_path.relative_to(root)),
        },
        random_seed=seed,
        gate=gate,
        notes=(
            "CHPC Boltz2 / AF3 / ESMFold / RF3 job scripts. AF3 requires weight "
            "access (helpdesk@chpc.utah.edu). RF3 needs Foundry (`rf3`) installed."
        ),
    )

    return {
        "jobs": jobs_result,
        "registry": registry,
        "models": models,
        "residues": residues,
        "confidence": confidence,
        "gate": gate,
        "manifest_path": manifest,
        "output_dir": out_dir,
        "structures_root": structures_root,
    }


def main() -> None:
    """CLI entry point for Stage 1 structure jobs / ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1: CHPC AlphaFold job scripts + structure confidence"
    )
    parser.add_argument("--version", default=None, help="Design background (default: pipeline)")
    parser.add_argument("--sequence", default=None, help="Override sequence (skip pickle)")
    parser.add_argument(
        "--jobs-only",
        action="store_true",
        help="Write SLURM/FASTA/JSON only (no ingest)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Parse existing raw outputs using job_registry.parquet",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="sbatch step-1 scripts (HPC only)",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=None,
        help="Subset of predictors (default: Boltz2 AF3 ESMFold RF3 from config)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Model seeds",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="States (default: apo from structures.yaml)",
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Exit non-zero if Gate 1 fails",
    )
    args = parser.parse_args()

    result = run_stage1(
        version=args.version,
        sequence=args.sequence,
        make_jobs=not args.ingest_only,
        ingest=not args.jobs_only,
        predictors=args.predictors,
        seeds=args.seeds,
        states=args.states,
        submit=args.submit,
    )
    jobs = result.get("jobs") or {}
    print(f"Version: {jobs.get('version') or args.version}")
    print(f"Jobs: {jobs.get('n_jobs', len(result.get('registry', [])))}")
    if jobs.get("submit_script"):
        print(f"Submit helper: {jobs['submit_script']}")
    print(f"Models ingested: {len(result['models'])}")
    print(f"Confidence rows: {len(result['confidence'])}")
    print(f"Gate 1: {result['gate'].get('structure_gate')}")
    print(f"Wrote: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    if args.require_gate and not result["gate"].get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
