"""Cross-model ipSAE tables for holo Stage-1 predictions.

Native ipTM is not comparable across AF3 / Boltz2 / RF3. ipSAE is computed
from each model's PAE with the same Dunbrack formula so interface confidence
can be compared (and disagreement quantified) on one scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.ipsae import ipsae_from_directory


def _ipsae_cfg(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    structures = load_yaml(root / "configs" / "structures.yaml")
    return dict(structures.get("ipsae") or {})


def score_model_ipsae(
    output_dir: Path,
    *,
    protein_chain: str = "A",
    ligand_chain: str | None = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float | None = 10.0,
    apo_states: tuple[str, ...] = ("apo",),
    state: str = "apo",
) -> dict[str, Any] | None:
    """Compute ipSAE for one predictor output directory.

    Apo (single-chain) jobs return None. Holo jobs return a metric dict.
    """
    if str(state).lower() in {s.lower() for s in apo_states}:
        return None
    result = ipsae_from_directory(
        output_dir,
        protein_chain=protein_chain,
        ligand_chain=ligand_chain,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
    )
    if result is None:
        return None
    return result.as_dict()


def compute_model_ipsae_table(
    registry: pd.DataFrame,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Score every non-apo Stage-1 job that has PAE outputs.

    Parameters
    ----------
    registry : pandas.DataFrame
        Job registry with ``output_dir``, ``method``, ``version``, ``seed``,
        ``state``, and ``structure_model_id``.
    repo_root : pathlib.Path, optional
        Used to resolve relative output paths and ipSAE cutoffs.

    Returns
    -------
    pandas.DataFrame
        One row per scored model.
    """
    root = repo_root or REPO_ROOT
    cfg = _ipsae_cfg(root)
    pae_cutoff = float(cfg.get("pae_cutoff", 10.0))
    dist_cutoff = cfg.get("dist_cutoff", 10.0)
    dist_cutoff = None if dist_cutoff is None else float(dist_cutoff)
    protein_chain = str(cfg.get("protein_chain", "A"))
    ligand_chain = cfg.get("ligand_chain", "B")
    apo_states = tuple(cfg.get("apo_states") or ("apo",))

    rows: list[dict[str, Any]] = []
    columns = [
        "structure_model_id",
        "version",
        "method",
        "seed",
        "state",
        "output_dir",
        "ipsae",
        "ipsae_ab",
        "ipsae_ba",
    ]
    if registry is None or registry.empty:
        return pd.DataFrame(columns=columns)
    for _, job in registry.iterrows():
        state = str(job.get("state", "apo"))
        out_dir = Path(str(job["output_dir"]))
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        scored = score_model_ipsae(
            out_dir,
            protein_chain=protein_chain,
            ligand_chain=None if ligand_chain is None else str(ligand_chain),
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
            apo_states=apo_states,
            state=state,
        )
        if scored is None:
            continue
        rows.append(
            {
                "structure_model_id": str(job.get("structure_model_id", "")),
                "version": str(job.get("version", "")),
                "method": str(job.get("method", "")),
                "seed": job.get("seed"),
                "state": state,
                "output_dir": str(out_dir),
                **scored,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def summarize_ipsae_across_models(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ipSAE mean / std / range across predictors per version×state.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`compute_model_ipsae_table`.

    Returns
    -------
    pandas.DataFrame
        Cross-model comparison rows. ``ipsae_std`` is the disagreement
        measure used in place of ipTM scatter.
    """
    if table.empty or "ipsae" not in table.columns:
        return pd.DataFrame(
            columns=[
                "version",
                "state",
                "n_models",
                "n_methods",
                "ipsae_mean",
                "ipsae_std",
                "ipsae_min",
                "ipsae_max",
                "ipsae_range",
            ]
        )
    rows: list[dict[str, Any]] = []
    grouped = table.groupby(["version", "state"], dropna=False)
    for (version, state), group in grouped:
        vals = pd.to_numeric(group["ipsae"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "version": version,
                "state": state,
                "n_models": int(len(group)),
                "n_methods": int(group["method"].nunique()) if "method" in group else 0,
                "ipsae_mean": float(vals.mean()),
                "ipsae_std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                "ipsae_min": float(vals.min()),
                "ipsae_max": float(vals.max()),
                "ipsae_range": float(vals.max() - vals.min()),
            }
        )
    return pd.DataFrame(rows)


def write_ipsae_tables(
    per_model: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write ipSAE parquet tables under ``data/structures``."""
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    out_dir = resolve_path(pipeline["paths"]["structures"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_model": out_dir / "ipsae_by_model.parquet",
        "summary": out_dir / "ipsae_across_models.parquet",
    }
    per_model.to_parquet(paths["per_model"], index=False)
    summary.to_parquet(paths["summary"], index=False)
    return paths
