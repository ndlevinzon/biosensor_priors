"""Propagate physics-score uncertainty across structural models."""



from __future__ import annotations



from pathlib import Path

from typing import Any



import numpy as np

import pandas as pd



from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path





SCORE_COLS = ("rif_ac", "rif_prop", "rpx", "delta_rif_sel")





def _agg_stats(series: pd.Series) -> dict[str, float]:

    """Compute distributional summary statistics for a numeric series.



    Parameters
    ----------
    series : pandas.Series

        Per-structure score values for one mutation.



    Returns
    -------
    dict

        Keys ``mean``, ``std``, ``n``, and optionally ``min``/``max``.

    """

    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)

    n = int(len(x))

    if n == 0:

        return {"mean": float("nan"), "std": float("nan"), "n": 0}

    return {

        "mean": float(np.mean(x)),

        "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,

        "n": n,

        "min": float(np.min(x)),

        "max": float(np.max(x)),

    }





def load_structural_confidence(

    path: Path | None,

) -> pd.DataFrame:

    """Load Stage 1 structural confidence table from parquet.



    Parameters
    ----------
    path : pathlib.Path, optional

        Parquet path; when missing or absent, returns an empty frame.



    Returns
    -------
    pandas.DataFrame

        Confidence table or empty DataFrame.

    """

    if path is None or not Path(path).exists():

        return pd.DataFrame()

    return pd.read_parquet(path)





def attach_mutation_confidence(

    summary: pd.DataFrame,

    confidence: pd.DataFrame,

) -> pd.DataFrame:

    """Attach structural confidence for the mutated canonical position when available.



    Confidence table expected columns (Stage 1): version, canonical_position /

    position, structural_confidence (or Confidence).



    Parameters
    ----------
    summary : pandas.DataFrame

        Mutation-level physics summary table.

    confidence : pandas.DataFrame

        Stage 1 per-position confidence table.



    Returns
    -------
    pandas.DataFrame

        Summary with ``structural_confidence`` and ``structural_confidence_source``.

    """

    out = summary.copy()

    if confidence.empty:

        out["structural_confidence"] = np.nan

        out["structural_confidence_source"] = "missing"

        return out



    conf = confidence.copy()

    cols = {c.lower(): c for c in conf.columns}

    ver = cols.get("version")

    pos = cols.get("canonical_position") or cols.get("canonical_pos") or cols.get("position")

    ccol = (

        cols.get("structural_confidence")

        or cols.get("confidence")

        or cols.get("conf")

    )

    if not (ver and pos and ccol):

        out["structural_confidence"] = np.nan

        out["structural_confidence_source"] = "unrecognized_schema"

        return out



    conf = conf.rename(columns={ver: "version", pos: "position", ccol: "structural_confidence"})

    conf["version"] = conf["version"].astype(str)

    conf["position"] = pd.to_numeric(conf["position"], errors="coerce")

    merged = out.merge(

        conf[["version", "position", "structural_confidence"]],

        on=["version", "position"],

        how="left",

        suffixes=("", "_conf"),

    )

    if "structural_confidence_conf" in merged.columns:

        merged["structural_confidence"] = merged["structural_confidence_conf"]

        merged = merged.drop(columns=["structural_confidence_conf"])

    merged["structural_confidence_source"] = np.where(

        merged["structural_confidence"].notna(), "stage1_table", "missing"

    )

    return merged





def aggregate_physics_uncertainty(

    long_table: pd.DataFrame,

    *,

    confidence: pd.DataFrame | None = None,

) -> pd.DataFrame:

    """Collapse per-structure scores into distributional summaries per mutation.



    Example output row::



        mutation = Q324R

        rif_ac_mean = -12.5

        rif_ac_std = 1.8

        n_structures = 7

        structural_confidence = 0.91



    Parameters
    ----------
    long_table : pandas.DataFrame

        Long-format scan table with one row per mutation × structure.

    confidence : pandas.DataFrame, optional

        Stage 1 structural confidence table to merge.



    Returns
    -------
    pandas.DataFrame

        Mutation-level summary with mean/std columns and confidence.



    Raises
    ------
    ValueError

        When required grouping columns are missing from ``long_table``.

    """

    if long_table.empty:

        return pd.DataFrame()



    group_cols = ["version", "position", "wt", "mutant", "mutation"]

    for c in group_cols:

        if c not in long_table.columns:

            raise ValueError(f"long_table missing required column {c}")



    rows = []

    for keys, group in long_table.groupby(group_cols, dropna=False):

        version, position, wt, mutant, mutation = keys

        n_structures = int(group["structure_model_id"].nunique()) if "structure_model_id" in group else len(group)

        row: dict[str, Any] = {

            "version": version,

            "position": int(position),

            "wt": wt,

            "mutant": mutant,

            "mutation": mutation,

            "n_structures": n_structures,

            "structure_model_ids": sorted(

                {str(x) for x in group.get("structure_model_id", pd.Series(dtype=str)).dropna()}

            ),

        }

        for col in SCORE_COLS:

            if col not in group.columns:

                continue

            stats = _agg_stats(group[col])

            row[f"{col}_mean"] = stats["mean"]

            row[f"{col}_std"] = stats["std"]

            row[f"{col}_n"] = stats["n"]

            # Convenience aliases matching the writeup narrative

            if col == "rif_ac":

                row["mean_RIF"] = stats["mean"]

                row["SD_RIF"] = stats["std"]

        rows.append(row)



    summary = pd.DataFrame(rows)

    summary = attach_mutation_confidence(summary, confidence if confidence is not None else pd.DataFrame())



    # Stage-3 friendly aliases (point estimates = means)

    summary["rif_ac"] = summary.get("rif_ac_mean")

    summary["rif_prop"] = summary.get("rif_prop_mean")

    summary["rpx"] = summary.get("rpx_mean")

    summary["delta_rif_sel"] = summary.get("delta_rif_sel_mean")

    summary["rif_ac_sd"] = summary.get("rif_ac_std")

    summary["delta_rif_sel_sd"] = summary.get("delta_rif_sel_std")

    summary["rpx_sd"] = summary.get("rpx_std")

    return summary





def run_physics_uncertainty(

    long_table: pd.DataFrame | None = None,

    *,

    long_table_path: Path | None = None,

    repo_root: Path | None = None,

    out_path: Path | None = None,

) -> dict[str, Any]:

    """Stage 2D — build mutation-level physics uncertainty table.



    Parameters
    ----------
    long_table : pandas.DataFrame, optional

        Long-format scan table; alternative to ``long_table_path``.

    long_table_path : pathlib.Path, optional

        Parquet path for the long scan table.

    repo_root : pathlib.Path, optional

        Repository root for config resolution.

    out_path : pathlib.Path, optional

        Override output parquet path.



    Returns
    -------
    dict

        Keys ``summary``, ``path``, ``processed_path``, and ``n_mutations``.



    Raises
    ------
    ValueError

        When neither ``long_table`` nor ``long_table_path`` is provided.

    """

    root = repo_root or REPO_ROOT

    pipeline = load_yaml(root / "configs" / "pipeline.yaml")

    physics_cfg = load_yaml(root / "configs" / "physics.yaml")



    if long_table is None:

        if long_table_path is None:

            raise ValueError("Provide long_table or long_table_path")

        long_table = pd.read_parquet(long_table_path)



    conf_path = physics_cfg.get("uncertainty", {}).get("structural_confidence_path")

    conf = load_structural_confidence(

        resolve_path(conf_path, root) if conf_path else None

    )

    summary = aggregate_physics_uncertainty(long_table, confidence=conf)



    physics_root = resolve_path(pipeline["paths"]["physics"], root)

    out_path = Path(out_path) if out_path else physics_root / "physics_scores_summary.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_parquet(out_path, index=False)

    summary.to_csv(out_path.with_suffix(".csv"), index=False)



    # Also write Stage-3 drop-in under data/processed when useful

    processed = resolve_path(pipeline["paths"]["processed"], root)

    processed.mkdir(parents=True, exist_ok=True)

    dropin = processed / "physics_mutation_scores.parquet"

    summary.to_parquet(dropin, index=False)



    return {

        "summary": summary,

        "path": out_path,

        "processed_path": dropin,

        "n_mutations": int(len(summary)),

    }


