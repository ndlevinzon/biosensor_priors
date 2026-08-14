"""Split runner for leave-one-construct-out and paired model comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.stage0_ground_truth.fitness import FoldFitnessScaler
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
    want = "leave_one_construct_out" if prefer_loco else "random_holdout"

    if splits_dir is not None and splits_dir.exists():
        splits = []
        try:
            loaded = load_frozen_splits(splits_dir)
        except FileNotFoundError:
            loaded = []
        for split in loaded:
            if split.get("strategy") != want:
                continue
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
    from biosensor_priors.stage0_ground_truth.splits import (
        generate_random_holdout_splits,
    )

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
    encoding: str = "mutation_bag",
    surrogate_kwargs: dict[str, Any] | None = None,
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
        Feature encoding mode (default ``mutation_bag``).
    surrogate_kwargs : dict, optional
        Extra :class:`FusedSurrogate` constructor kwargs (kernel, shrinkage, …).

    Returns
    -------
    pandas.DataFrame
        Long-format prediction rows with errors and metadata.
    """
    kinds = kinds or ["physics_only", "gp_zero_mean", "physics_gp"]
    extra = dict(surrogate_kwargs or {})
    extra.setdefault("encoding", encoding)
    min_components = int(extra.pop("min_components", 2))
    scaler = FoldFitnessScaler(
        weights=extra.get("phenotype_weights"),
        min_components=min_components,
    )
    work = df[df["fitness"].notna()].copy()
    rows: list[dict[str, Any]] = []

    for split in splits:
        train_df = scaler.fit_transform(_subset(work, split["train_construct_ids"]))
        test_df = scaler.transform(_subset(work, split["held_out_construct_ids"]))
        train_df = train_df[train_df["fitness"].notna()].copy()
        test_df = test_df[test_df["fitness"].notna()].copy()
        if train_df.empty or test_df.empty:
            continue
        y_train = train_df["fitness"].to_numpy(dtype=float)

        for kind in kinds:
            model = FusedSurrogate(
                kind=kind,
                use_confidence_weighting=use_confidence_weighting,
                random_state=random_seed,
                **extra,
            )
            model.fit(train_df, y_train)
            pred = model.predict(test_df)
            for i, cid in enumerate(pred.construct_ids):
                row_t = test_df.loc[test_df["construct_id"].astype(str) == cid].iloc[0]
                y_true = float(row_t["fitness"])
                if "structural_confidence" in test_df.columns:
                    conf = pd.to_numeric(
                        row_t.get("structural_confidence"), errors="coerce"
                    )
                    sigma_s = 0.0 if pd.isna(conf) else float(1.0 - conf)
                else:
                    sigma_s = 0.0
                if "physics_score_std" in test_df.columns:
                    raw_p = pd.to_numeric(
                        row_t.get("physics_score_std"), errors="coerce"
                    )
                    sigma_p = float(raw_p) if pd.notna(raw_p) else 0.0
                else:
                    sigma_p = 0.0
                rows.append(
                    {
                        "split_id": split["split_id"],
                        "strategy": split.get("strategy"),
                        "model_kind": kind,
                        "encoding": extra.get("encoding", encoding),
                        "construct_id": cid,
                        "y_true": y_true,
                        "fitness_mean": float(pred.fitness_mean[i]),
                        "fitness_std": float(pred.fitness_std[i]),
                        "physics_mean": float(pred.physics_mean[i]),
                        "gp_residual_mean": float(pred.gp_residual_mean[i]),
                        "sigma_structure": sigma_s,
                        "sigma_physics": sigma_p,
                        "physics_alpha": float(pred.physics_alpha),
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
