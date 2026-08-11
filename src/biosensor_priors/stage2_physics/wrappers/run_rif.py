"""
Scaffold CLI for the external RIF toolchain (willsheffler/rif).

Matches ``configs/physics.yaml`` → ``rif.command_template``:

    {executable} --structure … --ligands … --ligand-name … --out …

Until ``rif`` is installed in the active env, use ``--scaffold`` (default when
import fails) to write a parser-compatible ``rif_scores.tsv`` with null scores
and ``wrapper_status.json``.

Fill :func:`score_with_rif` once the library builds on CHPC.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.stage2_physics.wrappers._io import (
    load_mutations_json,
    resolve_mutations_path,
    try_import,
    write_status,
)


def score_with_rif(
    *,
    structure_pdb: Path,
    ligand_dir: Path,
    ligand_name: str,
    mutations: list[dict[str, Any]],
    structure_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call the real RIF API for each mutation (implement after install).

    Parameters
    ----------
    structure_pdb : pathlib.Path
        Receptor / design PDB.
    ligand_dir : pathlib.Path
        Directory of approved ligand conformers (parent of AcCoA/PropCoA).
    ligand_name : str
        Ligand label(s), e.g. ``AcCoA+PropCoA`` or ``AcCoA``.
    mutations : list of dict
        Specs with ``mutation``, ``position``, ``wt``, ``mutant``, ``version``.
    structure_model_id : str, optional
        Stage-1 model id for the score table.

    Returns
    -------
    list of dict
        Rows with at least ``mutation``, ``rif_ac``, ``rif_prop``.

    Notes
    -----
    Placeholder implementation raises ``NotImplementedError``. Suggested shape
    after ``pip``/cmake install of https://github.com/willsheffler/rif ::

        import rif  # or rif-specific docking entrypoints
        # for each mutation: apply mute, dock AcCoA / PropCoA ensembles, record scores
    """
    raise NotImplementedError(
        "score_with_rif is a scaffold. Install willsheffler/rif, then replace this "
        "function body with real docking calls. See docs/stages/stage2.md."
    )


def scaffold_rows(
    mutations: list[dict[str, Any]],
    *,
    structure_model_id: str | None,
    structure_pdb: Path,
) -> list[dict[str, Any]]:
    """Emit NaN score rows so parsers and job wiring can be tested."""
    if not mutations:
        mutations = [
            {
                "mutation": "WT",
                "position": -1,
                "wt": "X",
                "mutant": "X",
                "version": None,
            }
        ]
    rows = []
    for mut in mutations:
        rows.append(
            {
                "mutation": mut.get("mutation", "NA"),
                "position": mut.get("position"),
                "wt": mut.get("wt"),
                "mutant": mut.get("mutant"),
                "version": mut.get("version"),
                "structure_model_id": structure_model_id,
                "structure_pdb": str(structure_pdb),
                "rif_ac": math.nan,
                "rif_prop": math.nan,
                "backend": "scaffold",
            }
        )
    return rows


def write_rif_scores_tsv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rif_scores.tsv`` in the columns expected by ``score_parser``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mutation",
        "position",
        "wt",
        "mutant",
        "version",
        "structure_model_id",
        "rif_ac",
        "rif_prop",
        "backend",
        "structure_pdb",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, sep="\t", index=False)
    return path


def run(
    *,
    structure: Path,
    ligands: Path,
    ligand_name: str,
    out: Path,
    mutations_json: Path | None = None,
    structure_model_id: str | None = None,
    force_scaffold: bool = False,
    score_filename: str = "rif_scores.tsv",
) -> Path:
    """Execute scaffold or live RIF scoring; always write a score TSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mut_path = resolve_mutations_path(out, mutations_json)
    mutations = load_mutations_json(mut_path)

    ok, msg = try_import("rif")
    use_scaffold = force_scaffold or not ok

    if use_scaffold:
        rows = scaffold_rows(
            mutations, structure_model_id=structure_model_id, structure_pdb=structure
        )
        write_status(
            out,
            tool="rif",
            mode="scaffold",
            detail={
                "import_ok": ok,
                "import_message": msg,
                "n_mutations": len(mutations),
                "structure": str(structure),
                "ligands": str(ligands),
                "ligand_name": ligand_name,
                "next_step": (
                    "Install https://github.com/willsheffler/rif then implement "
                    "score_with_rif() in wrappers/run_rif.py"
                ),
            },
        )
    else:
        try:
            rows = score_with_rif(
                structure_pdb=structure,
                ligand_dir=ligands,
                ligand_name=ligand_name,
                mutations=mutations,
                structure_model_id=structure_model_id,
            )
            write_status(
                out,
                tool="rif",
                mode="live",
                detail={"import_ok": True, "n_rows": len(rows)},
            )
        except NotImplementedError as exc:
            rows = scaffold_rows(
                mutations, structure_model_id=structure_model_id, structure_pdb=structure
            )
            write_status(
                out,
                tool="rif",
                mode="scaffold_api_pending",
                detail={"import_ok": True, "error": str(exc), "n_mutations": len(mutations)},
            )

    return write_rif_scores_tsv(out / score_filename, rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the RIF wrapper scaffold."""
    parser = argparse.ArgumentParser(description="RIF docking wrapper (scaffold → live)")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, required=True)
    parser.add_argument("--ligand-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutations-json", type=Path, default=None)
    parser.add_argument("--structure-model-id", default=None)
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Force scaffold TSV even if rif imports",
    )
    parser.add_argument("--score-filename", default="rif_scores.tsv")
    args = parser.parse_args(argv)
    path = run(
        structure=args.structure,
        ligands=args.ligands,
        ligand_name=args.ligand_name,
        out=args.out,
        mutations_json=args.mutations_json,
        structure_model_id=args.structure_model_id,
        force_scaffold=args.scaffold,
        score_filename=args.score_filename,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
