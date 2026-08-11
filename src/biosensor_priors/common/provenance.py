"""Stage manifests and provenance recording."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    """Look up installed package versions for provenance metadata.

    Parameters
    ----------
    names : list[str]
        Distribution names as registered on PyPI (e.g. ``"numpy"``).

    Returns
    -------
    dict[str, str]
        Mapping from package name to version string, or ``"not-installed"``
        when the distribution is absent.
    """
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def write_manifest(
    path: Path,
    *,
    stage: str,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    random_seed: int | None,
    gate: dict[str, Any],
    notes: str | None = None,
) -> Path:
    """Write a reconstructable stage manifest JSON file.

    The manifest captures inputs, parameters, software versions, outputs,
    random seed, gate results, and optional notes for pipeline reproducibility.

    Parameters
    ----------
    path : Path
        Destination path for ``manifest.json``. Parent directories are created
        if needed.
    stage : str
        Pipeline stage identifier (e.g. ``"stage0_ground_truth"``).
    inputs : dict[str, Any]
        Input artifact metadata, typically including paths and content hashes.
    parameters : dict[str, Any]
        Configuration and analysis parameters used for the stage run.
    outputs : dict[str, Any]
        Output artifact metadata produced by the stage.
    random_seed : int | None
        Random seed used for stochastic steps, or ``None`` if not applicable.
    gate : dict[str, Any]
        Structured pass/fail report from stage validation gates.
    notes : str | None, optional
        Free-form notes appended to the manifest.

    Returns
    -------
    Path
        The path written (same as ``path``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "inputs": inputs,
        "parameters": parameters,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "biosensor-priors",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "pyyaml",
                    "scipy",
                    "biopython",
                    "openpyxl",
                ]
            ),
        },
        "random_seed": random_seed,
        "outputs": outputs,
        "gate": gate,
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
