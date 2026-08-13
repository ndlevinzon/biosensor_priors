#!/usr/bin/env python3
"""Remove generated pipeline artifacts so Stages 0–6 can be rerun from scratch.

Keeps experimental inputs, construct references, ``data/ligands/``, configs,
and ``weights/`` (RF3 / Foundry checkpoints). Wipes Stage 0–6 outputs under
``data/``, ``manifests/``, and ``outputs/`` (same trees that ``.gitignore``
treats as disposable).

Usage (from repo root, on CHPC or locally)::

    # Preview
    python scripts/clean_pipeline_artifacts.py --dry-run

    # Delete (requires confirmation unless --yes)
    python scripts/clean_pipeline_artifacts.py --yes

    # Also clear local Python caches
    python scripts/clean_pipeline_artifacts.py --yes --caches

After cleaning on HPC::

    pip install -e ".[dev,chem]"
    biosensor-stage0
    biosensor-stage1 --jobs-only --version V2.4
    # submit Stage-1 SLURM jobs, then --ingest-only
    # set physics.yaml backend: external when RF3 is ready
    biosensor-stage2
    biosensor-stage3
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Never delete these trees (inputs + code + configs + model weights).
PROTECTED_PREFIXES = (
    "data/experimental",
    "data/constructs",
    "data/ligands",
    "configs",
    "src",
    "tests",
    "docs",
    "scripts",
    "weights",
)

# Wipe contents; recreate directory + .gitkeep afterward.
WIPE_DIRS = (
    "data/processed",
    "data/structures",
    "data/physics",
    "data/rounds",
    "outputs",
)

# Extra keepers under wiped trees (recreated empty).
KEEP_RELATIVE = (
    "data/processed/.gitkeep",
    "data/processed/splits/.gitkeep",
    "data/structures/.gitkeep",
    "data/physics/.gitkeep",
    "data/rounds/.gitkeep",
    "outputs/.gitkeep",
    "manifests/.gitkeep",
)

# Manifest JSONs (keep .gitkeep only).
MANIFEST_GLOB = "manifests/*.json"

# Optional local clutter.
CACHE_GLOBS = (
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "htmlcov",
    ".coverage",
)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _is_protected(path: Path) -> bool:
    rel = _rel(path)
    return any(rel == p or rel.startswith(p + "/") for p in PROTECTED_PREFIXES)


def collect_wipe_targets(*, caches: bool) -> list[Path]:
    """Return paths that would be removed (files/dirs)."""
    targets: list[Path] = []

    for rel in WIPE_DIRS:
        root = REPO_ROOT / rel
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.name == ".gitkeep":
                continue
            if _is_protected(child):
                continue
            targets.append(child)

    for path in sorted((REPO_ROOT / "manifests").glob("*.json")):
        targets.append(path)

    if caches:
        for pattern in CACHE_GLOBS:
            for path in sorted(REPO_ROOT.glob(pattern)):
                if path.exists() and not _is_protected(path):
                    targets.append(path)

    # De-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def restore_keepers() -> list[Path]:
    """Recreate empty tracked placeholders after a wipe."""
    created: list[Path] = []
    for rel in KEEP_RELATIVE:
        path = REPO_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
            created.append(path)
    return created


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wipe generated biosensor_priors artifacts for a fresh Stage 0–6 run. "
            "Keeps data/experimental, data/constructs, configs, and weights/."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation",
    )
    parser.add_argument(
        "--caches",
        action="store_true",
        help="Also remove __pycache__ / pytest / ruff caches",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)

    global REPO_ROOT
    if args.repo_root is not None:
        REPO_ROOT = Path(args.repo_root).resolve()

    if not (REPO_ROOT / "configs" / "pipeline.yaml").exists():
        print(f"ERROR: not a biosensor_priors root: {REPO_ROOT}", file=sys.stderr)
        return 2

    targets = collect_wipe_targets(caches=bool(args.caches))
    print(f"Repo: {REPO_ROOT}")
    print(f"Targets: {len(targets)}")
    for t in targets:
        kind = "dir" if t.is_dir() else "file"
        print(f"  [{kind}] {_rel(t)}")

    if not targets:
        print("Nothing to remove (already clean). Restoring keepers…")
        if not args.dry_run:
            restore_keepers()
        return 0

    if args.dry_run:
        print("Dry-run only; no files deleted.")
        return 0

    if not args.yes:
        reply = input("Delete the paths above? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    errors = 0
    for t in targets:
        try:
            remove_path(t)
            print(f"removed {_rel(t)}")
        except OSError as exc:
            errors += 1
            print(f"FAILED {_rel(t)}: {exc}", file=sys.stderr)

    keepers = restore_keepers()
    for k in keepers:
        print(f"restored {_rel(k)}")

    print("Done. Next on HPC:")
    print('  pip install -e ".[dev,chem]"')
    print("  biosensor-stage0")
    print("  biosensor-stage1 --jobs-only --version V2.4")
    print("  # submit Stage-1 jobs → biosensor-stage1 --ingest-only")
    print("  biosensor-stage2 && biosensor-stage3")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
