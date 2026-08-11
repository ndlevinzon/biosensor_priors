"""
Run ESMFold via the fair-esm Python API (not the ``esm-fold`` CLI).

CHPC typical usage after ``module load esmfold/1.0.3``::

    python path/to/run_esmfold.py --fasta input.fasta --out out_dir

This file is intentionally runnable as a standalone script (stdlib + ``esm`` /
``torch`` only) so the module's Python can execute it without installing
``biosensor-priors`` into that interpreter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``(header, sequence)`` records."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.replace(" ", "").upper())
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return records


def predict_sequences(
    sequences: list[tuple[str, str]],
    *,
    output_dir: Path,
    chunk_size: int | None = 128,
    num_recycles: int | None = 4,
    device: str = "cuda",
) -> list[Path]:
    """
    Load ESMFold once and write one PDB per sequence.

    Parameters
    ----------
    sequences
        ``(name, aa_sequence)`` pairs; ``name`` becomes ``{name}.pdb``.
    output_dir
        Directory for PDB outputs.
    chunk_size
        Passed to ``model.set_chunk_size`` when not None (VRAM vs speed).
    num_recycles
        Recycling iterations for ``infer`` / ``infer_pdb`` when supported.
    device
        ``cuda`` or ``cpu``.

    Returns
    -------
    list of pathlib.Path
        Written PDB paths.
    """
    import torch
    import esm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        model = model.cuda()
    else:
        model = model.cpu()

    if chunk_size is not None:
        model.set_chunk_size(int(chunk_size))

    written: list[Path] = []
    for name, seq in sequences:
        if not seq:
            raise ValueError(f"Empty sequence for {name}")
        kwargs: dict = {}
        if num_recycles is not None:
            kwargs["num_recycles"] = int(num_recycles)
        with torch.no_grad():
            try:
                pdb_text = model.infer_pdb(seq, **kwargs)
            except TypeError:
                # Older fair-esm builds may not accept num_recycles on infer_pdb
                pdb_text = model.infer_pdb(seq)
        out_pdb = output_dir / f"{name}.pdb"
        out_pdb.write_text(pdb_text, encoding="utf-8")
        written.append(out_pdb)
        print(f"Wrote {out_pdb} ({len(seq)} aa)", flush=True)

    return written


def main(argv: list[str] | None = None) -> None:
    """CLI entry: FASTA → ESMFold PDBs."""
    parser = argparse.ArgumentParser(
        description="ESMFold structure prediction via esm.pretrained.esmfold_v1()"
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="model.set_chunk_size (lower = less VRAM). Use 0 to disable.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=4,
        help="Recycling iterations (if supported by this esm build)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Torch device (default: cuda)",
    )
    args = parser.parse_args(argv)

    chunk = None if int(args.chunk_size) <= 0 else int(args.chunk_size)
    records = read_fasta(args.fasta)
    print(f"ESMFold Python API | n_sequences={len(records)} | device={args.device}", flush=True)
    try:
        import esm

        print(f"esm version: {getattr(esm, '__version__', 'unknown')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import esm: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    predict_sequences(
        records,
        output_dir=args.out,
        chunk_size=chunk,
        num_recycles=int(args.num_recycles) if args.num_recycles is not None else None,
        device=args.device,
    )


if __name__ == "__main__":
    main()
