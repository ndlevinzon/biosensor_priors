"""
Back-compat shim: Stage 2 interface scores now come from RoseTTAFold3 docking.

Prefer::

    python -m biosensor_priors.stage2_physics.wrappers.run_rf3_dock

This module re-exports the RF3 docking API under the historical ``run_rosetta``
name so older scripts and entry points keep working.
"""

from __future__ import annotations

from biosensor_priors.stage2_physics.wrappers.run_rf3_dock import (
    apply_mutation,
    confidence_to_score,
    load_rf3_cfg,
    load_rosetta_cfg,
    main,
    parse_mutation_string,
    parse_rf3_confidence,
    run,
    scaffold_rows,
    score_mutation_rf3,
    score_mutation_rosetta,
    sequence_from_structure,
    write_interface_scores_tsv,
    write_rf3_dock_json,
    write_rpx_only_tsv,
)

__all__ = [
    "apply_mutation",
    "confidence_to_score",
    "load_rf3_cfg",
    "load_rosetta_cfg",
    "main",
    "parse_mutation_string",
    "parse_rf3_confidence",
    "run",
    "scaffold_rows",
    "score_mutation_rf3",
    "score_mutation_rosetta",
    "sequence_from_structure",
    "write_interface_scores_tsv",
    "write_rf3_dock_json",
    "write_rpx_only_tsv",
]


if __name__ == "__main__":
    main()
