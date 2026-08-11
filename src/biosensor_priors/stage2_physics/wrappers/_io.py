"""Shared helpers for RIF / RPX external wrapper scaffolds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_status(out_dir: Path, *, tool: str, mode: str, detail: dict[str, Any]) -> Path:
    """Write ``wrapper_status.json`` describing scaffold vs live backend."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "wrapper_status.json"
    payload = {"tool": tool, "mode": mode, **detail}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_mutations_json(path: Path | None) -> list[dict[str, Any]]:
    """Load mutation specs from JSON list or ``{mutations: [...]}`` object."""
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "mutations" in data:
        return list(data["mutations"])
    raise ValueError(f"Unrecognized mutations JSON shape: {path}")


def resolve_mutations_path(out_dir: Path, explicit: Path | None = None) -> Path | None:
    """Prefer ``--mutations-json``, else ``{out_dir}/mutations.json``."""
    if explicit is not None:
        return Path(explicit)
    candidate = Path(out_dir) / "mutations.json"
    return candidate if candidate.exists() else None


def try_import(module_name: str) -> tuple[bool, str]:
    """Attempt import; return (ok, message)."""
    try:
        __import__(module_name)
        return True, f"imported {module_name}"
    except Exception as exc:  # noqa: BLE001 — scaffold must never crash on missing deps
        return False, f"{type(exc).__name__}: {exc}"
