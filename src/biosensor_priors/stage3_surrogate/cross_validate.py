"""Split runner for leave-one-construct-out and paired model comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from biosensor_priors.stage0_ground_truth.splits import (
    generate_leave_one_out_splits,
    load_split,
)
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate, ModelKind


def _subset(df: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    """Filter construct table to a set of construct IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        Full construct table.
    ids : iterable of str
        Construct IDs to retain.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``df``.
    """
    id_set = {str(x) for x in ids}
    return df[df["construct_id"].astype(str).isin(id_set)].copy()


def load_frozen_splits(splits_dir: Path) -> list[dict[str, Any]]:
    """Load all frozen split JSON files from a directory.

    Parameters
    ----------
    splits_dir : pathlib.Path
        Directory containing ``split_*.json`` files.

    Returns
    -------
    list of dict
        Parsed split records from Stage 0.

    Raises
    ------
    FileNotFoundError
        When no split files are found.
    """
    paths = sorted(splits_dir.glob("split_*.json"))
    if not paths:
        raise FileNotFoundError(f"No split_*.json files in {splits_dir}")
    return [load_split(p) for p in paths]


def ensure_splits_for_fitness(
    df: pd.DataFrame,
    splits_dir: Path | None = None,
    *,
    prefer_loco: bool = True,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Load Stage-0 splits filtered to fitness rows, or build LOCO splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment master table with ``fitness`` and ``construct_id``.
    splits_dir : pathlib.Path, optional
        Directory of frozen split JSON files.
    prefer_loco : bool, optional
        When True, generate leave-one-construct-out splits if none load.
    random_seed : int, optional
        Random seed for generated splits (default 42).

    Returns
    -------
    list of dict
        Split records with ``train_construct_ids`` and ``held_out_construct_ids``.
    """
    eligible = df.loc[df["fitness"].notna(), "construct_id"].astype(str)
    eligible_set = set(eligible)

    if splits_dir is not None and splits_dir.exists():
        splits = []
        for split in load_frozen_splits(splits_dir):
            train = [i for i in split["train_construct_ids"] if i in eligible_set]
            test = [i for i in split["held_out_construct_ids"] if i in eligible_set]
            if not train or not test:
                continue
            record = dict(split)
            record["train_construct_ids"] = train
            record["held_out_construct_ids"] = test
            splits.append(record)
        if splits:
            return splits

    ids = eligible.tolist()
    if prefer_loco:
        return generate_leave_one_out_splits(ids, random_seed=random_seed)
    from biosensor_priors.stage0_ground_truth.splits import generate_random_holdout_splits

    return generate_random_holdout_splits(
        ids, n_splits=min(10, max(1, len(ids) // 5)), random_seed=random_seed
    )


def run_split_evaluation(
    df: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    kinds: list[ModelKind] | None = None,
    use_confidence_weighting: bool = True,
    random_seed: int = 42,
    encoding: str = "hybrid",
) -> pd.DataFrame:
    """Fit and evaluate each model kind across all CV splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment master with fitness labels.
    splits : list of dict
        Cross-validation split specifications.
    kinds : list of str, optional
        Model kinds to evaluate; defaults to all three baselines.
    use_confidence_weighting : bool, optional
        Apply structural confidence weighting to physics features.
    random_seed : int, optional
        Random seed for GP fitting (default 42).
    encoding : str, optional
        Feature encoding mode (default ``hybrid``).

    Returns
    -------
    pandas.DataFrame
        Long-format prediction rows with errors and metadata.
    """
    kinds = kinds or ["physics_only", "gp_zero_mean", "physics_gp"]
    work = df[df["fitness"].notna()].copy()
    rows: list[dict[str, Any]] = []

    for split in splits:
        train_df = _subset(work, split["train_construct_ids"])
        test_df = _subset(work, split["held_out_construct_ids"])
        if train_df.empty or test_df.empty:
            continue
        y_train = train_df["fitness"].to_numpy(dtype=float)

        for kind in kinds:
            model = FusedSurrogate(
                kind=kind,
                use_confidence_weighting=use_confidence_weighting,
                random_state=random_seed,
                encoding=encoding,
            )
            model.fit(train_df, y_train)
            pred = model.predict(test_df)
            for i, cid in enumerate(pred.construct_ids):
                y_true = float(test_df.loc[test_df["construct_id"].astype(str) == cid, "fitness"].iloc[0])
                rows.append(
                    {
                        "split_id": split["split_id"],
                        "strategy": split.get("strategy"),
                        "model_kind": kind,
                        "encoding": encoding,
                        "construct_id": cid,
                        "y_true": y_true,
                        "fitness_mean": float(pred.fitness_mean[i]),
                        "fitness_std": float(pred.fitness_std[i]),
                        "physics_mean": float(pred.physics_mean[i]),
                        "gp_residual_mean": float(pred.gp_residual_mean[i]),
                        "abs_error": abs(y_true - float(pred.fitness_mean[i])),
                        "sq_error": (y_true - float(pred.fitness_mean[i])) ** 2,
                        **{
                            f"meta_{k}": v
                            for k, v in model.metadata().items()
                            if k in {"has_physics_features", "n_features"}
                        },
                    }
                )
    return pd.DataFrame(rows)


def save_cv_predictions(predictions: pd.DataFrame, path: Path) -> Path:
    """Write cross-validation predictions to a parquet file.

    Parameters
    ----------
    predictions : pandas.DataFrame
        CV prediction table from :func:`run_split_evaluation`.
    path : pathlib.Path
        Output parquet path.

    Returns
    -------
    pathlib.Path
        ``path`` after writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path
