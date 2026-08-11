"""Configuration loading from YAML (no hard-coded analysis constants)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or REPO_ROOT) / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_pipeline_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    cfg_dir = resolve_path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return load_yaml(cfg_dir / "pipeline.yaml")


def load_fitness_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    cfg_dir = resolve_path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return load_yaml(cfg_dir / "fitness.yaml")


def load_stage0_configs(
    config_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_pipeline_config(config_dir), load_fitness_config(config_dir)
