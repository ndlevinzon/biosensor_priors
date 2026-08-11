"""Pipeline gate helpers: evaluate, record, and enforce stage gates."""

from __future__ import annotations

from typing import Any


def gate_passed(gate: dict[str, Any]) -> bool:
    """Return whether a gate report indicates overall success.

    Parameters
    ----------
    gate : dict[str, Any]
        Gate result dictionary, expected to contain a ``"passed"`` key.

    Returns
    -------
    bool
        ``True`` when ``gate["passed"]`` is truthy; otherwise ``False``.
    """
    return bool(gate.get("passed"))


def require_gate(gate: dict[str, Any], *, stage: str) -> None:
    """Raise if a stage gate did not pass.

    Parameters
    ----------
    gate : dict[str, Any]
        Gate result dictionary from a pipeline stage.
    stage : str
        Stage name included in the error message on failure.

    Returns
    -------
    None
        Returns only when the gate passed.

    Raises
    ------
    RuntimeError
        If ``gate["passed"]`` is falsy.
    """
    if not gate_passed(gate):
        failed = gate.get("failed", [])
        raise RuntimeError(f"{stage} gate failed: {failed}")
