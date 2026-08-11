"""Stage 2 orchestration: ligands → RIF/RPX scan → uncertainty → Gate 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import write_manifest
from biosensor_priors.stage2_physics.gate2 import evaluate_gate2
from biosensor_priors.stage2_physics.ligand_ensemble import run_ligand_ensemble
from biosensor_priors.stage2_physics.mutation_scan import run_mutation_scan
from biosensor_priors.stage2_physics.physics_uncertainty import run_physics_uncertainty


def run_stage2(
    *,
    repo_root: Path | None = None,
    n_placeholder_conformers: int = 3,
    skip_gate_enforce: bool = False,
) -> dict[str, Any]:
    """Run Stage 2 end-to-end under the configured backend (default: mock).

    External RIF/RPX/QM binaries are optional. With ``backend: mock`` the Python
    orchestration, permanent conformer IDs, long-format scan table, uncertainty
    aggregation, and Gate 2 controls are fully exercised.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root for config and output paths.
    n_placeholder_conformers : int, optional
        Placeholder conformers per ligand in Stage 2A (default 3).
    skip_gate_enforce : bool, optional
        When True, do not enforce Gate 2 failure policy (default False).

    Returns
    -------
    dict
        Keys ``ligands``, ``scan``, ``uncertainty``, ``gate``,
        ``manifest_path``, and ``output_dir``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    physics_cfg = load_yaml(root / "configs" / "physics.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    seed = int(pipeline.get("random_seed", 42))

    # 2A
    ligands = run_ligand_ensemble(
        repo_root=root,
        physics_cfg=physics_cfg,
        n_placeholder=n_placeholder_conformers,
    )

    # 2B + 2C
    scan = run_mutation_scan(repo_root=root)

    # 2D
    uncertainty = run_physics_uncertainty(
        scan["long_table"],
        repo_root=root,
    )

    # 2E
    gate = evaluate_gate2(uncertainty["summary"], repo_root=root)
    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "gate2.json"
    gate_path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    physics_root = resolve_path(pipeline["paths"]["physics"], root)
    (physics_root / "gate2.json").write_text(
        json.dumps(gate, indent=2, default=str), encoding="utf-8"
    )

    gate_policy = str(pipeline.get("gates", {}).get("stage2", "required_for_physics_weight"))
    if gate_policy == "required_for_physics_weight" and not skip_gate_enforce:
        # Do not raise the whole stage — record FAIL so Stage 3 can refuse weight.
        # Hard raise only when explicitly requested via CLI flag.
        pass

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage2_manifest.json",
        stage="stage2_physics",
        inputs={
            "backend": physics_cfg.get("backend"),
            "n_conformers": int(len(ligands.conformers)),
            "ligand_catalog": str(ligands.catalog_path.relative_to(root)),
        },
        parameters={
            "random_seed": seed,
            "score_direction": thresholds.get("physics", {}).get("score_direction"),
            "physics_scan_id": scan["physics_scan_id"],
            "delta_rif_sel_definition": "RIF_Ac - RIF_Prop",
        },
        outputs={
            "ligand_conformers": str(ligands.catalog_path.relative_to(root)),
            "mutation_scan": str(Path(scan["path"]).relative_to(root)),
            "physics_summary": str(Path(uncertainty["path"]).relative_to(root)),
            "processed_physics": str(Path(uncertainty["processed_path"]).relative_to(root)),
            "gate2": str(gate_path.relative_to(root)),
        },
        random_seed=seed,
        gate=gate,
        notes=(
            "External physics tools optional. Mock backend exercises orchestration. "
            "Gate 2 FAIL blocks full physics weight in Stage 3."
        ),
    )

    return {
        "ligands": ligands,
        "scan": scan,
        "uncertainty": uncertainty,
        "gate": gate,
        "manifest_path": manifest,
        "output_dir": out_dir,
    }


def main() -> None:
    """CLI entry point for Stage 2 physics landscape orchestration.

    Parameters
    ----------
    None
        Flags are parsed from ``sys.argv`` via ``argparse``.

    Returns
    -------
    None
        Prints scan/gate summary to stdout; may exit with status 1 when
        ``--require-gate`` is set and Gate 2 fails.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 physics landscape orchestration")
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Exit non-zero if Gate 2 fails (default: record FAIL and continue)",
    )
    parser.add_argument("--n-conformers", type=int, default=3)
    args = parser.parse_args()
    result = run_stage2(n_placeholder_conformers=args.n_conformers)
    gate = result["gate"]
    print(f"Backend: mock/external via configs/physics.yaml")
    print(f"Conformers: {len(result['ligands'].conformers)}")
    print(f"Scan rows: {len(result['scan']['long_table'])}")
    print(f"Mutations summarized: {result['uncertainty']['n_mutations']}")
    print(f"Gate 2: {gate['physics_gate']} failed={gate.get('failed')}")
    print(f"Wrote: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    if args.require_gate and not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
