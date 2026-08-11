"""Deterministic train/test split generator reused across all model variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

SplitStrategy = Literal["random_holdout", "leave_one_construct_out"]


def _as_sorted_ids(ids: list[str] | np.ndarray) -> list[str]:
    return sorted(str(x) for x in ids)


def make_split_record(
    *,
    split_id: str,
    train_ids: list[str],
    test_ids: list[str],
    random_seed: int,
    strategy: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train = _as_sorted_ids(train_ids)
    test = _as_sorted_ids(test_ids)
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise ValueError(f"Train/test overlap in {split_id}: {overlap[:5]}")
    record = {
        "split_id": split_id,
        "strategy": strategy,
        "random_seed": int(random_seed),
        "train_construct_ids": train,
        "held_out_construct_ids": test,
        "n_train": len(train),
        "n_held_out": len(test),
    }
    if extras:
        record.update(extras)
    return record


def generate_random_holdout_splits(
    construct_ids: list[str],
    *,
    n_splits: int,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic random hold-out splits (no replacement within a split)."""
    ids = _as_sorted_ids(construct_ids)
    if len(ids) < 2:
        raise ValueError("Need at least 2 constructs to generate hold-out splits.")

    n_test = int(test_size) if test_size is not None else max(1, int(round(len(ids) * test_fraction)))
    n_test = min(n_test, len(ids) - 1)

    rng = np.random.default_rng(random_seed)
    splits: list[dict[str, Any]] = []
    for i in range(1, n_splits + 1):
        perm = rng.permutation(len(ids))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        test_ids = [ids[j] for j in test_idx]
        train_ids = [ids[j] for j in train_idx]
        split_id = f"split_{i:03d}"
        splits.append(
            make_split_record(
                split_id=split_id,
                train_ids=train_ids,
                test_ids=test_ids,
                random_seed=random_seed,
                strategy="random_holdout",
                extras={"split_index": i, "test_fraction": test_fraction},
            )
        )
    return splits


def generate_leave_one_out_splits(
    construct_ids: list[str],
    *,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """One held-out construct per split (paired LOCO comparisons)."""
    ids = _as_sorted_ids(construct_ids)
    splits: list[dict[str, Any]] = []
    for i, held in enumerate(ids, start=1):
        train = [x for x in ids if x != held]
        splits.append(
            make_split_record(
                split_id=f"split_{i:03d}",
                train_ids=train,
                test_ids=[held],
                random_seed=random_seed,
                strategy="leave_one_construct_out",
                extras={"held_out_construct_id": held},
            )
        )
    return splits


def write_splits(
    splits: list[dict[str, Any]],
    output_dir: Path,
    *,
    clear_existing: bool = True,
) -> list[Path]:
    """Write ``split_XXX.json`` files and an index manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for old in output_dir.glob("split_*.json"):
            old.unlink()
        index_path = output_dir / "splits_index.json"
        if index_path.exists():
            index_path.unlink()

    paths: list[Path] = []
    for record in splits:
        path = output_dir / f"{record['split_id']}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        paths.append(path)

    index = {
        "n_splits": len(splits),
        "strategy": splits[0]["strategy"] if splits else None,
        "random_seed": splits[0]["random_seed"] if splits else None,
        "splits": [p.name for p in paths],
    }
    (output_dir / "splits_index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    return paths


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_overlap(split: dict[str, Any]) -> None:
    train = set(split["train_construct_ids"])
    test = set(split["held_out_construct_ids"])
    overlap = train & test
    if overlap:
        raise AssertionError(f"Train/test overlap: {sorted(overlap)[:10]}")
