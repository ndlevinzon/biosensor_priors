"""
Stage 2 packing / fold-quality scores via RoseTTAFold3 apo confidence.

``rpx`` is negated apo RF3 confidence after mutation (legacy column name).
Same engine as ``run_rf3_dock``; this CLI writes the packing-only TSV expected
by ``rpx_jobs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biosensor_priors.stage2_physics.wrappers._io import (
    load_mutations_json,
    resolve_mutations_path,
    write_status,
)
from biosensor_priors.stage2_physics.wrappers.run_rf3_dock import (
    load_rf3_cfg,
    parse_mutation_string,
    rf3_available,
    scaffold_rows,
    score_mutation_rf3,
    write_rpx_only_tsv,
)


def run(
    *,
    structure: Path,
    mutation: str | None,
    out: Path,
    mutations_json: Path | None = None,
    structure_model_id: str | None = None,
    force_scaffold: bool = False,
    score_filename: str = "rpx_scores.tsv",
) -> Path:
    """Score apo RF3 confidence for one mutation or a mutations.json list."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    mut_path = resolve_mutations_path(out, mutations_json)
    batch = load_mutations_json(mut_path)
    if mutation and str(mutation).upper() != "BATCH":
        targets: list[dict] = [parse_mutation_string(mutation)]
    elif batch:
        targets = [
            m if isinstance(m, dict) else parse_mutation_string(str(m)) for m in batch
        ]
    else:
        targets = [parse_mutation_string("WT")]

    cfg = load_rf3_cfg()
    ok, msg = rf3_available(cfg)
    use_scaffold = force_scaffold or not ok

    if use_scaffold:
        rows = scaffold_rows(
            targets, structure_model_id=structure_model_id, structure_pdb=structure
        )
        write_status(
            out,
            tool="rf3_apo",
            mode="scaffold",
            detail={
                "rf3_ok": ok,
                "rf3_message": msg,
                "n_mutations": len(targets),
                "structure": str(structure),
                "next_step": (
                    "pip install 'rc-foundry[rf3]'; drop --scaffold; "
                    "set physics.yaml backend: external (granite-gpu)."
                ),
            },
        )
    else:
        rows = []
        errors: list[str] = []
        work = out / "rf3_runs"
        for mut in targets:
            try:
                rows.append(
                    score_mutation_rf3(
                        structure_pdb=structure,
                        mutation=mut,
                        cfg=cfg,
                        structure_model_id=structure_model_id,
                        work_dir=work,
                        score_apo=True,
                        score_ligands=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mut}: {type(exc).__name__}: {exc}")
                rows.extend(
                    scaffold_rows(
                        [mut],
                        structure_model_id=structure_model_id,
                        structure_pdb=structure,
                    )
                )
        write_status(
            out,
            tool="rf3_apo",
            mode="live" if not errors else "live_partial",
            detail={"n_rows": len(rows), "errors": errors[:20]},
        )

    return write_rpx_only_tsv(out / score_filename, rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for RF3 apo scores (rpx column)."""
    parser = argparse.ArgumentParser(
        description="RoseTTAFold3 apo confidence → Stage 2 rpx column"
    )
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--mutation",
        default=None,
        help="Single mutation code (Q324R). Omit to use {out}/mutations.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutations-json", type=Path, default=None)
    parser.add_argument("--structure-model-id", default=None)
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--score-filename", default="rpx_scores.tsv")
    args = parser.parse_args(argv)
    path = run(
        structure=args.structure,
        mutation=args.mutation,
        out=args.out,
        mutations_json=args.mutations_json,
        structure_model_id=args.structure_model_id,
        force_scaffold=args.scaffold,
        score_filename=args.score_filename,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
