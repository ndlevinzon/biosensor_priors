"""
Scaffold CLI for RPX / rpxdock (willsheffler/rpxdock).

Matches ``configs/physics.yaml`` → ``rpx.command_template``:

    {executable} --structure … --mutation … --out …

Install hint::

    pip install git+ssh://git@github.com/willsheffler/rpxdock.git

Fill :func:`score_with_rpx` once ``import rpxdock`` works on CHPC.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.stage2_physics.wrappers._io import (
    load_mutations_json,
    resolve_mutations_path,
    try_import,
    write_status,
)


_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutation_string(mutation: str) -> dict[str, Any]:
    """Parse ``Q324R`` → wt/position/mutant dict."""
    m = _MUT_RE.match(str(mutation).strip())
    if not m:
        return {"mutation": mutation, "wt": None, "position": None, "mutant": None}
    return {
        "mutation": mutation,
        "wt": m.group(1),
        "position": int(m.group(2)),
        "mutant": m.group(3),
    }


def score_with_rpx(
    *,
    structure_pdb: Path,
    mutation: str,
    structure_model_id: str | None = None,
) -> dict[str, Any]:
    """
    Call rpxdock / RPX packing for one mutation (implement after install).

    Notes
    -----
    After ``pip install git+…/rpxdock.git``::

        import rpxdock
        # score packing for mutated structure; return float rpx
    """
    raise NotImplementedError(
        "score_with_rpx is a scaffold. Install willsheffler/rpxdock, then implement "
        "this function. See docs/stages/stage2.md."
    )


def scaffold_row(
    mutation: str,
    *,
    structure_model_id: str | None,
    structure_pdb: Path,
) -> dict[str, Any]:
    """Emit one NaN RPX row for wiring tests."""
    parsed = parse_mutation_string(mutation)
    return {
        "mutation": parsed["mutation"],
        "position": parsed["position"],
        "wt": parsed["wt"],
        "mutant": parsed["mutant"],
        "structure_model_id": structure_model_id,
        "structure_pdb": str(structure_pdb),
        "rpx": math.nan,
        "backend": "scaffold",
    }


def write_rpx_scores_tsv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rpx_scores.tsv`` for ``score_parser.parse_rpx_score_table``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mutation",
        "position",
        "wt",
        "mutant",
        "structure_model_id",
        "rpx",
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
    mutation: str | None,
    out: Path,
    mutations_json: Path | None = None,
    structure_model_id: str | None = None,
    force_scaffold: bool = False,
    score_filename: str = "rpx_scores.tsv",
) -> Path:
    """Score one mutation or a mutations.json list; write RPX TSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    mut_path = resolve_mutations_path(out, mutations_json)
    batch = load_mutations_json(mut_path)
    if mutation and str(mutation).upper() != "BATCH":
        targets = [mutation]
    elif batch:
        targets = [str(m.get("mutation", m)) if isinstance(m, dict) else str(m) for m in batch]
    else:
        targets = ["WT"]

    ok, msg = try_import("rpxdock")
    # Some installs may expose a top-level name other than rpxdock
    if not ok:
        ok2, msg2 = try_import("rpx")
        ok = ok or ok2
        msg = msg if not ok2 else msg2

    use_scaffold = force_scaffold or not ok
    rows: list[dict[str, Any]] = []

    if use_scaffold:
        for mut in targets:
            rows.append(
                scaffold_row(mut, structure_model_id=structure_model_id, structure_pdb=structure)
            )
        write_status(
            out,
            tool="rpx",
            mode="scaffold",
            detail={
                "import_ok": ok,
                "import_message": msg,
                "n_mutations": len(targets),
                "structure": str(structure),
                "next_step": (
                    "pip install git+ssh://git@github.com/willsheffler/rpxdock.git "
                    "then implement score_with_rpx() in wrappers/run_rpx.py"
                ),
            },
        )
    else:
        for mut in targets:
            try:
                row = score_with_rpx(
                    structure_pdb=structure,
                    mutation=mut,
                    structure_model_id=structure_model_id,
                )
                rows.append(row)
            except NotImplementedError as exc:
                rows.append(
                    scaffold_row(
                        mut, structure_model_id=structure_model_id, structure_pdb=structure
                    )
                )
                write_status(
                    out,
                    tool="rpx",
                    mode="scaffold_api_pending",
                    detail={"import_ok": True, "error": str(exc)},
                )

        if all(r.get("backend") != "scaffold" for r in rows):
            write_status(
                out,
                tool="rpx",
                mode="live",
                detail={"import_ok": True, "n_rows": len(rows)},
            )

    return write_rpx_scores_tsv(out / score_filename, rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the RPX / rpxdock wrapper scaffold."""
    parser = argparse.ArgumentParser(description="RPX / rpxdock wrapper (scaffold → live)")
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
