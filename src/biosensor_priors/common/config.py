"""Configuration loading from YAML (no hard-coded analysis constants)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless absolute.

    Parameters
    ----------
    path : str | Path
        File or directory path, either absolute or relative to ``root``.
    root : Path | None, optional
        Base directory for relative paths. Defaults to ``REPO_ROOT``.

    Returns
    -------
    Path
        Absolute path when ``path`` is absolute; otherwise ``root / path``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or REPO_ROOT) / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a mapping.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file. Relative paths are resolved from ``REPO_ROOT``.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content. Empty files yield an empty dict.

    Raises
    ------
    ValueError
        If the top-level YAML value is not a mapping.
    """
    path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_pipeline_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load ``pipeline.yaml`` from the configuration directory.

    Parameters
    ----------
    config_dir : str | Path | None, optional
        Directory containing ``pipeline.yaml``. Defaults to ``DEFAULT_CONFIG_DIR``.

    Returns
    -------
    dict[str, Any]
        Parsed pipeline configuration.
    """
    cfg_dir = resolve_path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return load_yaml(cfg_dir / "pipeline.yaml")


def load_fitness_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load ``fitness.yaml`` from the configuration directory.

    Parameters
    ----------
    config_dir : str | Path | None, optional
        Directory containing ``fitness.yaml``. Defaults to ``DEFAULT_CONFIG_DIR``.

    Returns
    -------
    dict[str, Any]
        Parsed fitness configuration.
    """
    cfg_dir = resolve_path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return load_yaml(cfg_dir / "fitness.yaml")


def load_stage0_configs(
    config_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load both pipeline and fitness configs for Stage 0.

    Parameters
    ----------
    config_dir : str | Path | None, optional
        Directory containing ``pipeline.yaml`` and ``fitness.yaml``.
        Defaults to ``DEFAULT_CONFIG_DIR``.

    Returns
    -------
    pipeline_cfg : dict[str, Any]
        Parsed pipeline configuration.
    fitness_cfg : dict[str, Any]
        Parsed fitness configuration.
    """
    return load_pipeline_config(config_dir), load_fitness_config(config_dir)
