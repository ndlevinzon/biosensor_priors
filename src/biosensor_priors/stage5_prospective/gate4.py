"""Gate 4 / Stage-5 gate before model update after prospective evaluation."""

from __future__ import annotations

from typing import Any

from biosensor_priors.stage5_prospective.freeze_predictions import verify_freeze_integrity


def evaluate_gate4(
    validation: dict[str, Any],
    *,
    rounds_dir=None,
    round_id=None,
    min_matched: int = 1,
    require_finite_rmse: bool = True,
    require_freeze_integrity: bool = True,
) -> dict[str, Any]:
    """
    Stage-5 gate: freeze integrity + prospective validation sanity.

    Required before model update when ``pipeline.gates.stage5`` is
    ``required_before_model_update``.
    """
    checks = []

    if require_freeze_integrity and rounds_dir is not None and round_id is not None:
        integrity = verify_freeze_integrity(rounds_dir, round_id)
        checks.append(
            {
                "name": "freeze_integrity",
                "passed": bool(integrity["ok"]),
                "details": integrity,
            }
        )

    overall = validation.get("overall", {})
    n_matched = int(overall.get("n_matched", validation.get("n_matched", 0)) or 0)
    checks.append(
        {
            "name": "matched_observations",
            "passed": n_matched >= min_matched,
            "n_matched": n_matched,
            "min_matched": min_matched,
        }
    )

    rmse = overall.get("rmse")
    checks.append(
        {
            "name": "finite_rmse",
            "passed": (not require_finite_rmse) or (rmse is not None and rmse == rmse),
            "rmse": rmse,
        }
    )

    checks.append(
        {
            "name": "validation_report_passed",
            "passed": bool(validation.get("passed", False)),
        }
    )

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "failed": [c["name"] for c in checks if not c["passed"]],
        "overall": overall,
        "by_algorithm": validation.get("by_algorithm", []),
        "physics_revalidation": validation.get("physics_revalidation", {}),
    }
