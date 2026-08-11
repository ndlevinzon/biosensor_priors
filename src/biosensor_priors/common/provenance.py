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
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
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
    """Write a reconstructable stage manifest.json."""
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
