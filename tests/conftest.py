"""Shared fixtures for Stage-0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from biosensor_priors.common.config import REPO_ROOT
from biosensor_priors.stage0_ground_truth.load_experiments import build_experiment_master


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def stage0_result(repo_root: Path):
    """Build Stage 0 once per test session (writes processed artifacts)."""
    # Rebuild construct pickles so they match the active pandas/numpy versions.
    return build_experiment_master(repo_root=repo_root, rebuild_constructs=True)
