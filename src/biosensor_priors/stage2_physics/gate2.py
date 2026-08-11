"""Gate 2: directional regression tests for Q324R and A355R controls."""

from __future__ import annotations

from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml


def _score_is_better(value: float, reference: float, score_direction: str) -> bool:
    """Return True if ``value`` is better than ``reference`` under the frozen convention.

    Parameters
    ----------
    value : float
        Candidate score to evaluate.
    reference : float
        Reference score for comparison.
    score_direction : str
        Frozen scoring convention, either ``more_negative_is_better`` or
        ``more_positive_is_better``.

    Returns
    -------
    bool
        True when ``value`` is strictly better than ``reference``.
    """
    if score_direction == "more_negative_is_better":
        return value < reference
    if score_direction == "more_positive_is_better":
        return value > reference
    raise ValueError(f"Unknown score_direction: {score_direction}")


def expected_delta_sign(expected_direction: str, score_direction: str) -> int:
    """Map control expectation onto the sign of ΔRIF_sel = RIF_Ac − RIF_Prop.

    For ``favorable_AcCoA`` + ``more_negative_is_better``:
      better Ac interaction ⇒ RIF_Ac < RIF_Prop ⇒ ΔRIF_sel < 0 ⇒ expected sign −1.

    Parameters
    ----------
    expected_direction : str
        Expected ligand preference, e.g. ``favorable_AcCoA`` or
        ``favorable_PropCoA``.
    score_direction : str
        Frozen scoring convention from ``thresholds.yaml``.

    Returns
    -------
    int
        Expected sign of ΔRIF_sel: ``-1``, ``+1``.
    """
    if expected_direction == "favorable_AcCoA":
        return -1 if score_direction == "more_negative_is_better" else +1
    if expected_direction == "favorable_PropCoA":
        return +1 if score_direction == "more_negative_is_better" else -1
    raise ValueError(f"Unknown expected_direction: {expected_direction}")


def _lookup_control_row(scores: pd.DataFrame, mutation: str) -> pd.Series | None:
    """Find the summary row for a control mutation in a physics scores table.

    Parameters
    ----------
    scores : pandas.DataFrame
        Physics summary or scan table with a ``mutation`` column.
    mutation : str
        Control mutation code, e.g. ``Q324R``.

    Returns
    -------
    pandas.Series or None
        First matching row, or None when the mutation is absent.
    """
    if scores.empty or "mutation" not in scores.columns:
        return None
    hit = scores[scores["mutation"].astype(str) == mutation]
    if hit.empty:
        return None
    # Prefer summary table with means
    return hit.iloc[0]


def _delta_from_row(row: pd.Series) -> float | None:
    """Extract ΔRIF_sel from a physics summary row using available columns.

    Parameters
    ----------
    row : pandas.Series
        Single mutation summary row.

    Returns
    -------
    float or None
        Selectivity term when derivable; otherwise None.
    """
    for col in ("delta_rif_sel_mean", "delta_rif_sel", "delta_RIF_sel"):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    if "rif_ac_mean" in row.index and "rif_prop_mean" in row.index:
        return float(row["rif_ac_mean"]) - float(row["rif_prop_mean"])
    if "rif_ac" in row.index and "rif_prop" in row.index:
        return float(row["rif_ac"]) - float(row["rif_prop"])
    return None


def test_control_direction(
    scores: pd.DataFrame,
    *,
    mutation: str,
    expected_direction: str,
    score_direction: str,
) -> dict[str, Any]:
    """Run a single control directional regression test.

    Parameters
    ----------
    scores : pandas.DataFrame
        Physics summary table with per-mutation scores.
    mutation : str
        Control mutation code to test.
    expected_direction : str
        Expected ligand preference for the control.
    score_direction : str
        Frozen scoring convention from ``thresholds.yaml``.

    Returns
    -------
    dict
        Test result with keys ``mutation``, ``passed``, ``delta_rif_sel``,
        ``expected_sign``, ``reason``, and optional metadata fields.
    """
    row = _lookup_control_row(scores, mutation)
    if row is None:
        return {
            "mutation": mutation,
            "passed": False,
            "reason": f"control {mutation} missing from physics scores",
            "expected_direction": expected_direction,
            "score_direction": score_direction,
        }
    delta = _delta_from_row(row)
    if delta is None:
        return {
            "mutation": mutation,
            "passed": False,
            "reason": "could not derive delta_rif_sel",
            "expected_direction": expected_direction,
            "score_direction": score_direction,
        }
    exp_sign = expected_delta_sign(expected_direction, score_direction)
    # Strict directional check (nonzero)
    if delta == 0:
        passed = False
        reason = "delta_rif_sel is exactly zero"
    else:
        passed = (delta < 0 and exp_sign < 0) or (delta > 0 and exp_sign > 0)
        reason = "ok" if passed else f"delta_rif_sel={delta} disagrees with expected sign {exp_sign}"

    return {
        "mutation": mutation,
        "passed": bool(passed),
        "delta_rif_sel": float(delta),
        "expected_sign": exp_sign,
        "expected_direction": expected_direction,
        "score_direction": score_direction,
        "reason": reason,
        "n_structures": int(row["n_structures"]) if "n_structures" in row.index else None,
        "structural_confidence": (
            float(row["structural_confidence"])
            if "structural_confidence" in row.index and pd.notna(row["structural_confidence"])
            else None
        ),
    }


def check_Q324R_direction(
    scores: pd.DataFrame,
    *,
    score_direction: str = "more_negative_is_better",
    expected_direction: str = "favorable_AcCoA",
) -> dict[str, Any]:
    """Regression check: Q324R must favor AcCoA under the frozen score direction.

    Parameters
    ----------
    scores : pandas.DataFrame
        Physics summary table with per-mutation scores.
    score_direction : str, optional
        Frozen scoring convention (default ``more_negative_is_better``).
    expected_direction : str, optional
        Expected ligand preference (default ``favorable_AcCoA``).

    Returns
    -------
    dict
        Directional test result from :func:`test_control_direction`.
    """
    return test_control_direction(
        scores,
        mutation="Q324R",
        expected_direction=expected_direction,
        score_direction=score_direction,
    )


def check_A355R_direction(
    scores: pd.DataFrame,
    *,
    score_direction: str = "more_negative_is_better",
    expected_direction: str = "favorable_AcCoA",
) -> dict[str, Any]:
    """Regression check: A355R must favor AcCoA under the frozen score direction.

    Parameters
    ----------
    scores : pandas.DataFrame
        Physics summary table with per-mutation scores.
    score_direction : str, optional
        Frozen scoring convention (default ``more_negative_is_better``).
    expected_direction : str, optional
        Expected ligand preference (default ``favorable_AcCoA``).

    Returns
    -------
    dict
        Directional test result from :func:`test_control_direction`.
    """
    return test_control_direction(
        scores,
        mutation="A355R",
        expected_direction=expected_direction,
        score_direction=score_direction,
    )


def evaluate_gate2(
    scores: pd.DataFrame,
    *,
    repo_root=None,
    score_direction: str | None = None,
    controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gate 2 — hard directional regression on required controls.

    If either control fails: ``physics_gate = FAIL`` and Stage 3 must not
    quietly use physics at full weight.

    Parameters
    ----------
    scores : pandas.DataFrame
        Mutation-level physics summary table.
    repo_root : pathlib.Path, optional
        Repository root for loading ``thresholds.yaml``.
    score_direction : str, optional
        Override score direction; defaults to config value.
    controls : list of dict, optional
        Control specifications with ``mutation`` and ``expected_direction``.

    Returns
    -------
    dict
        Gate verdict with ``passed``, ``physics_gate``, ``tests``, ``failed``,
        and ``allow_full_physics_weight``.
    """
    root = repo_root or REPO_ROOT
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    phys = thresholds.get("physics", {})
    score_direction = score_direction or str(phys.get("score_direction", "more_negative_is_better"))
    controls = controls or list(phys.get("controls") or [])
    if not controls:
        controls = [
            {"mutation": "Q324R", "expected_direction": "favorable_AcCoA"},
            {"mutation": "A355R", "expected_direction": "favorable_AcCoA"},
        ]

    tests = []
    for ctrl in controls:
        mut = str(ctrl["mutation"])
        exp = str(ctrl.get("expected_direction", "favorable_AcCoA"))
        if mut == "Q324R":
            tests.append(
                check_Q324R_direction(scores, score_direction=score_direction, expected_direction=exp)
            )
        elif mut == "A355R":
            tests.append(
                check_A355R_direction(scores, score_direction=score_direction, expected_direction=exp)
            )
        else:
            tests.append(
                test_control_direction(
                    scores,
                    mutation=mut,
                    expected_direction=exp,
                    score_direction=score_direction,
                )
            )

    passed = all(t["passed"] for t in tests)
    failed = [t["mutation"] for t in tests if not t["passed"]]
    return {
        "passed": passed,
        "physics_gate": "PASS" if passed else "FAIL",
        "score_direction": score_direction,
        "tests": tests,
        "failed": failed,
        "allow_full_physics_weight": passed,
        "notes": (
            "Stage 3 must not use physics at full weight when physics_gate=FAIL."
            if not passed
            else "Controls passed directional checks."
        ),
    }


def assert_physics_weight_allowed(gate: dict[str, Any]) -> None:
    """Raise if Stage 3 attempts full physics weight after Gate 2 failure.

    Parameters
    ----------
    gate : dict
        Gate 2 result dict from :func:`evaluate_gate2`.

    Returns
    -------
    None
        Returns only when full physics weight is allowed.

    Raises
    ------
    RuntimeError
        When ``allow_full_physics_weight`` is False.
    """
    if not gate.get("allow_full_physics_weight", False):
        raise RuntimeError(
            "Gate 2 FAIL: physics may not be used at full weight. "
            f"Failed controls: {gate.get('failed')}"
        )
