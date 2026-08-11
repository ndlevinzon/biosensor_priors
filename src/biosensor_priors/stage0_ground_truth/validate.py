"""Stage-0 validation gates (IDs, mappings, fitness, splits, controls)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.align_constructs import validate_mapping
from biosensor_priors.stage0_ground_truth.fitness import fitness_transform
from biosensor_priors.stage0_ground_truth.splits import assert_no_overlap


REQUIRED_FIELDS = [
    "construct_id",
    "Construct",
    "mutation_audit",
    "Brightness__ordinal",
    "Affinity AcCoA__uM",
    "FC AcCoA__value",
    "Selectivity_Kd_Prop_over_Ac__lower",
    "fitness",
    "version",
]


def _check_unique_construct_ids(df: pd.DataFrame) -> dict[str, Any]:
    ids = df["construct_id"].astype(str)
    dupes = ids[ids.duplicated()].unique().tolist()
    return {
        "name": "unique_construct_ids",
        "passed": len(dupes) == 0,
        "n_constructs": int(ids.nunique()),
        "duplicates": dupes[:20],
    }


def _check_required_fields(df: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    return {
        "name": "required_fields_present",
        "passed": len(missing_cols) == 0,
        "missing_columns": missing_cols,
    }


def _check_canonical_mappings(
    versions: pd.DataFrame,
    residue_mapping: pd.DataFrame,
) -> dict[str, Any]:
    problems = validate_mapping(versions, residue_mapping)
    return {
        "name": "canonical_mappings_valid",
        "passed": len(problems) == 0,
        "problems": problems[:20],
        "n_versions": int(versions["Version"].nunique()),
        "n_mapped_residues": int(len(residue_mapping)),
    }


def _check_fitness_reproducible(
    df: pd.DataFrame,
    fitness_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Recompute fitness twice and require exact agreement on finite values."""
    cols_needed = [
        "Affinity AcCoA__uM",
        "Affinity AcCoA__censor_direction",
        "FC AcCoA__value",
        "FC AcCoA__censor_direction",
        "Selectivity_Kd_Prop_over_Ac__lower",
        "Brightness__ordinal",
    ]
    base = df[cols_needed].copy()
    a = fitness_transform(
        base,
        weights=fitness_cfg["weights"],
        min_components=int(fitness_cfg.get("min_components", 2)),
        policies=fitness_cfg.get("observations"),
        require_range=False,
    )
    b = fitness_transform(
        base,
        weights=fitness_cfg["weights"],
        min_components=int(fitness_cfg.get("min_components", 2)),
        policies=fitness_cfg.get("observations"),
        require_range=False,
    )
    same = np.allclose(
        a["fitness"].to_numpy(dtype=float),
        b["fitness"].to_numpy(dtype=float),
        equal_nan=True,
    )
    matches_master = np.allclose(
        a["fitness"].to_numpy(dtype=float),
        df["fitness"].to_numpy(dtype=float),
        equal_nan=True,
    )
    return {
        "name": "fitness_reproducible",
        "passed": bool(same and matches_master),
        "n_with_fitness": int(df["fitness"].notna().sum()),
        "weights": fitness_cfg["weights"],
    }


def _check_splits(splits: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for split in splits:
        try:
            assert_no_overlap(split)
        except AssertionError as exc:
            errors.append(f"{split.get('split_id')}: {exc}")
        train = set(split["train_construct_ids"])
        test = set(split["held_out_construct_ids"])
        if not train or not test:
            errors.append(f"{split.get('split_id')}: empty train or test")
    return {
        "name": "no_train_test_overlap",
        "passed": len(errors) == 0,
        "n_splits": len(splits),
        "errors": errors[:20],
    }


def _flatten_mutation_codes(df: pd.DataFrame) -> set[str]:
    found: set[str] = set()
    if "mutation_codes" in df.columns:
        for val in df["mutation_codes"]:
            if isinstance(val, list):
                found.update(str(x) for x in val)
            elif isinstance(val, str) and val.startswith("["):
                # parquet round-trip may stringify lists
                for token in ("Q324R", "A355R"):
                    if token in val:
                        found.add(token)
    for col in ("mut_codes_construct", "mut_codes_description"):
        if col not in df.columns:
            continue
        for val in df[col]:
            if isinstance(val, list):
                found.update(str(x) for x in val)
    # Also scan Construct / Description text
    for col in ("Construct", "Description"):
        if col in df.columns:
            for val in df[col].astype(str):
                for token in ("Q324R", "A355R"):
                    if token in val:
                        found.add(token)
    return found


def _check_controls(df: pd.DataFrame, required: list[str]) -> dict[str, Any]:
    found = _flatten_mutation_codes(df)
    missing = [m for m in required if m not in found]
    return {
        "name": "required_control_mutations",
        "passed": len(missing) == 0,
        "required": required,
        "missing": missing,
        "found": sorted(found.intersection(required)),
    }


def run_stage0_gates(
    df: pd.DataFrame,
    *,
    versions: pd.DataFrame,
    residue_mapping: pd.DataFrame,
    splits: list[dict[str, Any]],
    fitness_cfg: dict[str, Any],
    required_controls: list[str] | None = None,
) -> dict[str, Any]:
    """Run all Stage-0 gates and return a structured pass/fail report."""
    checks = [
        _check_unique_construct_ids(df),
        _check_required_fields(df),
        _check_canonical_mappings(versions, residue_mapping),
        _check_fitness_reproducible(df, fitness_cfg),
        _check_splits(splits),
        _check_controls(df, required_controls or ["Q324R", "A355R"]),
    ]
    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "failed": [c["name"] for c in checks if not c["passed"]],
    }
