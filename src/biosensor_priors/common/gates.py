"""Pipeline gate helpers: evaluate, record, and enforce stage gates."""

from __future__ import annotations

from typing import Any


def gate_passed(gate: dict[str, Any]) -> bool:
    return bool(gate.get("passed"))


def require_gate(gate: dict[str, Any], *, stage: str) -> None:
    if not gate_passed(gate):
        failed = gate.get("failed", [])
        raise RuntimeError(f"{stage} gate failed: {failed}")
