"""Import new experimental results through the Stage-0 cleaning pathway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import load_fitness_config, load_pipeline_config, resolve_path
from biosensor_priors.stage0_ground_truth.clean import (
    load_raw_experimental_workbook,
    prepare_database,
)
from biosensor_priors.stage0_ground_truth.fitness import fitness_transform
from biosensor_priors.stage0_ground_truth.version_resolve import (
    attach_resolved_versions,
    get_row_mutations,
)


def _attach_mutation_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized ``mutation_codes`` strings to cleaned result rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experimental table with resolvable mutation metadata.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a ``mutation_codes`` list column added.
    """
    out = df.copy()
    codes = []
    for _, row in out.iterrows():
        muts = get_row_mutations(row)
        if muts is None:
            codes.append(None)
        else:
            codes.append([f"{a}{p}{b}" for a, p, b in muts])
    out["mutation_codes"] = codes
    return out


def clean_new_results(
    raw: pd.DataFrame,
    *,
    versions: pd.DataFrame,
    fitness_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    experimental_round: int | str | None = None,
) -> pd.DataFrame:
    """Clean new wet-lab rows through the same Stage-0 pathway as historical data.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw plate export before cleaning.
    versions : pd.DataFrame
        Construct version table for sequence resolution.
    fitness_cfg : dict
        Parsed ``fitness.yaml`` weights and observation policies.
    pipeline_cfg : dict
        Parsed ``pipeline.yaml`` experimental settings.
    experimental_round : int, str, or None, optional
        Round label to attach to cleaned rows.

    Returns
    -------
    pd.DataFrame
        Cleaned constructs with fitness and ``mutation_codes`` columns.
    """
    clean = prepare_database(
        raw,
        assume_unitless_affinity_um=bool(
            pipeline_cfg.get("experimental", {}).get("assume_unitless_affinity_um", False)
        ),
    )
    clean = attach_resolved_versions(
        clean,
        versions,
        version_aliases=pipeline_cfg.get("version_aliases") or {},
    )
    clean = _attach_mutation_codes(clean)
    clean = fitness_transform(
        clean,
        weights=fitness_cfg["weights"],
        min_components=int(fitness_cfg.get("min_components", 2)),
        policies=fitness_cfg.get("observations"),
        require_range=False,
    )
    if experimental_round is not None:
        clean["experimental_round"] = experimental_round
    return clean


def load_and_clean_results_file(
    path: Path,
    *,
    repo_root: Path | None = None,
    experimental_round: int | str | None = None,
) -> pd.DataFrame:
    """Load an Excel/CSV plate export and clean it via Stage 0.

    Parameters
    ----------
    path : Path
        Path to raw experimental results file.
    repo_root : Path or None, optional
        Repository root for construct/version artifacts.
    experimental_round : int, str, or None, optional
        Round label to attach to cleaned rows.

    Returns
    -------
    pd.DataFrame
        Cleaned results ready for validation and master append.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    from biosensor_priors.common.config import REPO_ROOT

    root = repo_root or REPO_ROOT
    pipeline_cfg = load_pipeline_config()
    fitness_cfg = load_fitness_config()
    constructs_dir = resolve_path(pipeline_cfg["paths"]["constructs"], root)
    versions = pd.read_pickle(
        constructs_dir / pipeline_cfg["constructs"]["versions_pickle"]
    )

    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = load_raw_experimental_workbook(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported results format: {path}")

    return clean_new_results(
        raw,
        versions=versions,
        fitness_cfg=fitness_cfg,
        pipeline_cfg=pipeline_cfg,
        experimental_round=experimental_round,
    )


def append_to_experiment_master(
    new_rows: pd.DataFrame,
    *,
    master_path: Path,
    master_pickle_path: Path | None = None,
) -> pd.DataFrame:
    """Append cleaned new rows to the authoritative experiment master artifacts.

    Deduplicates on ``construct_id`` keeping the newest row.

    Parameters
    ----------
    new_rows : pd.DataFrame
        Cleaned constructs to append.
    master_path : Path
        Parquet path for the experiment master.
    master_pickle_path : Path or None, optional
        Pickle path; defaults to ``master_path`` with ``.pkl`` suffix.

    Returns
    -------
    pd.DataFrame
        Combined master table after append and deduplication.
    """
    master_path = Path(master_path)
    if master_pickle_path is None:
        master_pickle_path = master_path.with_suffix(".pkl")

    if master_pickle_path.exists():
        master = pd.read_pickle(master_pickle_path)
    elif master_path.exists():
        master = pd.read_parquet(master_path)
    else:
        master = pd.DataFrame()

    combined = pd.concat([master, new_rows], ignore_index=True)
    if "construct_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["construct_id"], keep="last")

    combined.to_pickle(master_pickle_path)
    store = combined.copy()
    for col in store.columns:
        if store[col].dtype == object:
            store[col] = store[col].map(lambda x: None if x is None else str(x))
    store.to_parquet(master_path, index=False)
    return combined
