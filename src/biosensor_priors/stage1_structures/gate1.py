"""Stage-1 structural quality / completeness gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml


def evaluate_gate1(
    *,
    registry: pd.DataFrame | None = None,
    models: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    repo_root: Path | None = None,
    min_models: int = 1,
    min_reliable_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate Gate 1 on the structure ensemble and confidence table.

    Checks
    ------
    * At least ``min_models`` ingested structure models exist.
    * Confidence table is non-empty when models exist.
    * Fraction of reliable positions ≥ ``min_reliable_fraction``.

    Parameters
    ----------
    registry : pandas.DataFrame, optional
        Job registry (informational).
    models : pandas.DataFrame, optional
        Ingested model table.
    confidence : pandas.DataFrame, optional
        ``structural_confidence`` table.
    repo_root : pathlib.Path, optional
        Loads gate policy from ``pipeline.yaml``.
    min_models : int, optional
        Minimum accepted models (default 1).
    min_reliable_fraction : float, optional
        Minimum fraction of reliable residues (default 0.0).

    Returns
    -------
    dict
        Gate payload with ``passed``, ``structure_gate``, ``checks``, ``failed``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    policy = str(pipeline.get("gates", {}).get("stage1", "advisory"))

    checks: list[dict[str, Any]] = []
    failed: list[str] = []

    n_models = 0 if models is None or models.empty else int(len(models))
    ok_models = n_models >= int(min_models)
    checks.append(
        {
            "name": "min_models",
            "passed": ok_models,
            "n_models": n_models,
            "min_models": int(min_models),
        }
    )
    if not ok_models:
        failed.append("min_models")

    conf_ok = True
    reliable_frac = float("nan")
    if n_models > 0:
        conf_ok = confidence is not None and not confidence.empty
        checks.append({"name": "confidence_table_nonempty", "passed": conf_ok})
        if not conf_ok:
            failed.append("confidence_table_nonempty")
        elif "reliable" in confidence.columns or "Reliable" in confidence.columns:
            col = "reliable" if "reliable" in confidence.columns else "Reliable"
            series = confidence[col]
            if series.dtype == object:
                rel = series.astype(str).str.lower().isin(["yes", "true", "1"])
            else:
                rel = series.astype(bool)
            reliable_frac = float(rel.mean()) if len(rel) else 0.0
            ok_frac = reliable_frac >= float(min_reliable_fraction)
            checks.append(
                {
                    "name": "reliable_fraction",
                    "passed": ok_frac,
                    "reliable_fraction": reliable_frac,
                    "min_reliable_fraction": float(min_reliable_fraction),
                }
            )
            if not ok_frac:
                failed.append("reliable_fraction")

    n_jobs = 0 if registry is None or registry.empty else int(len(registry))
    passed = len(failed) == 0
    return {
        "passed": passed,
        "structure_gate": "PASS" if passed else "FAIL",
        "policy": policy,
        "failed": failed,
        "checks": checks,
        "n_jobs": n_jobs,
        "n_models": n_models,
        "reliable_fraction": reliable_frac,
    }
