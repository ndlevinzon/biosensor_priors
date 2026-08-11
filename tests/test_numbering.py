"""Canonical numbering and mapping integrity."""

from __future__ import annotations


def test_canonical_mappings_valid(stage0_result) -> None:
    _, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert checks["canonical_mappings_valid"]["passed"]
    assert checks["canonical_mappings_valid"]["n_mapped_residues"] > 0


def test_versions_unique(stage0_result) -> None:
    from biosensor_priors.common.config import REPO_ROOT, load_pipeline_config, resolve_path
    import pandas as pd

    cfg = load_pipeline_config()
    path = resolve_path(cfg["paths"]["constructs"], REPO_ROOT) / cfg["constructs"][
        "versions_pickle"
    ]
    versions = pd.read_pickle(path)
    assert not versions["Version"].duplicated().any()
