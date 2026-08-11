"""Ensure train/test splits never overlap."""

from __future__ import annotations

from pathlib import Path

from biosensor_priors.stage0_ground_truth.splits import assert_no_overlap, load_split


def test_no_train_test_overlap(stage0_result) -> None:
    _, meta = stage0_result
    splits_dir = Path(meta["splits_dir"])
    paths = sorted(splits_dir.glob("split_*.json"))
    assert paths, "expected frozen split files"
    for path in paths:
        split = load_split(path)
        assert_no_overlap(split)
        assert split["train_construct_ids"]
        assert split["held_out_construct_ids"]


def test_stage0_gate_reports_no_overlap(stage0_result) -> None:
    _, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert checks["no_train_test_overlap"]["passed"]
