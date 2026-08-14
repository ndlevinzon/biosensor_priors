"""Reproducibility and Stage-0 integrity checks."""

from __future__ import annotations

import numpy as np

from biosensor_priors.common.config import load_fitness_config
from biosensor_priors.stage0_ground_truth.fitness import fitness_transform


def test_constructs_uniquely_identified(stage0_result) -> None:
    master, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert checks["unique_construct_ids"]["passed"]
    assert master["construct_id"].nunique() == len(master)


def test_no_missing_required_fields(stage0_result) -> None:
    _, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert checks["required_fields_present"]["passed"]


def test_fitness_reproducible(stage0_result) -> None:
    master, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert checks["fitness_reproducible"]["passed"]

    fitness_cfg = load_fitness_config()
    cols = [
        "Affinity AcCoA__uM",
        "Affinity AcCoA__censor_direction",
        "FC AcCoA__value",
        "FC AcCoA__censor_direction",
        "FC PropCoA__value",
        "FC PropCoA__censor_direction",
        "Selectivity_Kd_Prop_over_Ac__lower",
        "Brightness__ordinal",
        "mutation_audit",
    ]
    recomputed = fitness_transform(
        master[cols].copy(),
        weights=fitness_cfg["weights"],
        min_components=int(fitness_cfg.get("min_components", 2)),
        policies=fitness_cfg.get("observations"),
    )
    assert np.allclose(
        recomputed["fitness"].to_numpy(dtype=float),
        master["fitness"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_stage0_gate_passed(stage0_result) -> None:
    _, meta = stage0_result
    assert meta["gate"]["passed"] is True


def test_preregistered_weights_sum_to_one() -> None:
    fitness_cfg = load_fitness_config()
    assert abs(sum(fitness_cfg["weights"].values()) - 1.0) < 1e-12
