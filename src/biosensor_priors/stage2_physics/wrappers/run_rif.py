"""
Backward-compatible CLI alias for :mod:`run_rosetta`.

Prefer ``python -m biosensor_priors.stage2_physics.wrappers.run_rosetta``.
"""

from __future__ import annotations

from biosensor_priors.stage2_physics.wrappers.run_rosetta import (
    load_rosetta_cfg,
    main,
    run,
    scaffold_rows,
    score_mutation_rosetta,
    write_interface_scores_tsv,
)

__all__ = [
    "load_rosetta_cfg",
    "main",
    "run",
    "scaffold_rows",
    "score_mutation_rosetta",
    "write_interface_scores_tsv",
]

if __name__ == "__main__":
    main()
