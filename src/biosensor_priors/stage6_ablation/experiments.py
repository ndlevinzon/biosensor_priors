"""Ablation configuration matrix runner (shared splits and seeds)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage3_surrogate.cross_validate import ensure_splits_for_fitness
from biosensor_priors.stage3_surrogate.features import FeatureBuilder
from biosensor_priors.stage3_surrogate.gate3 import metrics_for_predictions
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate, ModelKind
from biosensor_priors.stage4_search.prefilter import PrefilterCategory, physics_prefilter


@dataclass(frozen=True)
class AblationConfig:
    """One cell of the Stage-6 ablation matrix."""

    id: str
    physics: bool
    gp: bool
    confidence_weighting: bool | None
    structure_source: str | None
    prefilter: bool
    label: str | None = None

    def model_kind(self) -> ModelKind:
        """Map physics/GP flags to a fused surrogate kind.

        Parameters
        ----------
        None

        Returns
        -------
        ModelKind
            One of ``"physics_gp"``, ``"physics_only"``, or ``"gp_zero_mean"``.

        Raises
        ------
        ValueError
            If both physics and GP are disabled.
        """
        if self.physics and self.gp:
            return "physics_gp"
        if self.physics and not self.gp:
            return "physics_only"
        if not self.physics and self.gp:
            return "gp_zero_mean"
        raise ValueError(f"Invalid ablation {self.id}: need physics and/or GP")

    def use_confidence_weighting(self) -> bool:
        """Whether structural confidence weighting is active for this cell.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            ``True`` only when confidence weighting is explicitly enabled and
            physics features are included.
        """
        if self.confidence_weighting is None:
            return False
        return bool(self.confidence_weighting) and bool(self.physics)

    def as_row(self) -> dict[str, Any]:
        """Serialize the config plus derived fields for reporting.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, Any]
            Dataclass fields plus ``model_kind`` and
            ``confidence_weighting_effective``.
        """
        row = asdict(self)
        row["model_kind"] = self.model_kind()
        row["confidence_weighting_effective"] = self.use_confidence_weighting()
        return row


def _parse_optional_bool(value: Any) -> bool | None:
    """Parse YAML-style optional booleans and sentinel placeholders.

    Parameters
    ----------
    value : Any
        Raw value from an ablation config entry (bool, str, or null sentinel).

    Returns
    -------
    bool or None
        Parsed boolean, or ``None`` when the value denotes missing/disabled.
    """
    if value is None or value == "—" or value == "-" or value == "":
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"yes", "true", "y", "1"}:
            return True
        if low in {"no", "false", "n", "0"}:
            return False
        if low in {"null", "none", "—", "-"}:
            return None
    return bool(value)


def _parse_optional_str(value: Any) -> str | None:
    """Parse YAML-style optional strings and sentinel placeholders.

    Parameters
    ----------
    value : Any
        Raw value from an ablation config entry.

    Returns
    -------
    str or None
        String form of the value, or ``None`` when empty or a null sentinel.
    """
    if value is None or value == "—" or value == "-" or value == "":
        return None
    return str(value)


def load_ablation_matrix(path: Path | None = None) -> tuple[list[AblationConfig], dict[str, Any]]:
    """Load ``configs/ablation.yaml`` into typed configs and raw settings.

    Parameters
    ----------
    path : Path, optional
        Path to the ablation YAML file. Defaults to ``configs/ablation.yaml``
        under the repository root.

    Returns
    -------
    configs : list[AblationConfig]
        Parsed ablation matrix cells.
    settings : dict[str, Any]
        Full raw YAML document (seeds, paths, reference config, etc.).
    """
    root = REPO_ROOT
    cfg_path = path or (root / "configs" / "ablation.yaml")
    raw = load_yaml(cfg_path)
    configs: list[AblationConfig] = []
    for item in raw.get("configs", []):
        configs.append(
            AblationConfig(
                id=str(item["id"]),
                physics=bool(_parse_optional_bool(item.get("physics")) or False),
                gp=bool(_parse_optional_bool(item.get("gp")) or False),
                confidence_weighting=_parse_optional_bool(item.get("confidence_weighting")),
                structure_source=_parse_optional_str(item.get("structure_source")),
                prefilter=bool(_parse_optional_bool(item.get("prefilter")) or False),
                label=item.get("label"),
            )
        )
    if not configs:
        raise ValueError(f"No ablation configs in {cfg_path}")
    return configs, raw


def default_ablation_matrix() -> list[AblationConfig]:
    """Return the built-in ablation matrix matching the Stage-6 writeup.

    Parameters
    ----------
    None

    Returns
    -------
    list[AblationConfig]
        Six preset configurations spanning physics-only, GP-only, fused models,
        confidence weighting, and AF2/AF3 prefilter variants.
    """
    return [
        AblationConfig("physics_only_consensus", True, False, None, "consensus", False, "Physics only"),
        AblationConfig("gp_only", False, True, None, None, False, "GP only"),
        AblationConfig("physics_gp_no_conf", True, True, False, "consensus", False, "Physics+GP (no conf)"),
        AblationConfig(
            "physics_gp_conf_consensus", True, True, True, "consensus", False, "Physics+GP + conf"
        ),
        AblationConfig(
            "physics_gp_conf_af2_prefilter",
            True,
            True,
            True,
            "Boltz2",
            True,
            "Physics+GP + Boltz2 + prefilter",
        ),
        AblationConfig(
            "physics_gp_conf_af3_prefilter",
            True,
            True,
            True,
            "AF3",
            True,
            "Physics+GP + AF3 + prefilter",
        ),
    ]


def _subset(df: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    """Filter a construct table to a set of construct IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing a ``construct_id`` column.
    ids : Iterable[str]
        Construct identifiers to retain.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` restricted to rows whose ``construct_id`` is in ``ids``.
    """
    id_set = {str(x) for x in ids}
    return df[df["construct_id"].astype(str).isin(id_set)].copy()


def attach_structure_confidence(
    df: pd.DataFrame,
    structure_source: str | None,
    *,
    repo_root: Path | None = None,
    confidence_dir: str | Path = "data/processed",
    pattern: str = "structural_confidence_{source}.parquet",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach ``structural_confidence`` for a named structure source.

    Looks for Stage-1 parquet artifacts; otherwise uses an existing consensus
    column or a deterministic hash-based proxy and records availability in
    metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Construct table with ``construct_id`` (and optionally
        ``structural_confidence``).
    structure_source : str or None
        Structure predictor label (e.g. ``"consensus"``, ``"AF2"``, ``"AF3"``).
        ``None`` sets unit confidence.
    repo_root : Path, optional
        Repository root for resolving relative paths.
    confidence_dir : str or Path, default ``"data/processed"``
        Directory containing per-source confidence parquet files.
    pattern : str, default ``"structural_confidence_{source}.parquet"``
        Filename pattern with ``{source}`` placeholder.

    Returns
    -------
    out : pd.DataFrame
        Copy of ``df`` with a ``structural_confidence`` column.
    meta : dict[str, Any]
        Provenance flags: ``structure_source``, ``structure_available``,
        ``structure_path``, and optional ``structure_proxy``.
    """
    out = df.copy()
    meta: dict[str, Any] = {
        "structure_source": structure_source,
        "structure_available": False,
        "structure_path": None,
    }
    if structure_source is None:
        out["structural_confidence"] = 1.0
        return out, meta

    root = repo_root or REPO_ROOT
    conf_dir = resolve_path(confidence_dir, root)
    path = conf_dir / pattern.format(source=structure_source)
    meta["structure_path"] = str(path)

    if path.exists():
        table = pd.read_parquet(path)
        id_col = "construct_id" if "construct_id" in table.columns else None
        conf_col = "structural_confidence" if "structural_confidence" in table.columns else None
        if id_col and conf_col:
            merged = out.merge(
                table[[id_col, conf_col]].rename(columns={conf_col: "_src_conf"}),
                on=id_col,
                how="left",
            )
            out["structural_confidence"] = (
                pd.to_numeric(merged["_src_conf"], errors="coerce").fillna(0.0).to_numpy()
            )
            meta["structure_available"] = True
            return out, meta

    # Consensus / fallback: keep existing confidence or ones.
    if "structural_confidence" in out.columns and structure_source.lower() == "consensus":
        out["structural_confidence"] = pd.to_numeric(
            out["structural_confidence"], errors="coerce"
        ).fillna(0.0)
        meta["structure_available"] = True
        meta["structure_path"] = "inline_consensus"
        return out, meta

    # Source-specific deterministic proxy so AF2 vs AF3 slots remain distinct
    # until Stage-1 tables land (documented in meta).
    conf = []
    for cid in out["construct_id"].astype(str):
        digest = hashlib.md5(f"{structure_source}:{cid}".encode()).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        conf.append(float(rng.uniform(0.55, 0.98)))
    out["structural_confidence"] = conf
    meta["structure_available"] = False
    meta["structure_proxy"] = "deterministic_hash_uniform"
    return out, meta


def apply_prefilter_mask(
    df: pd.DataFrame,
    *,
    enabled: bool,
    score_direction: str = "more_negative_is_better",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply physics prefilter and optionally drop HARD_FAIL constructs.

    Parameters
    ----------
    df : pd.DataFrame
        Construct table with physics scores for prefilter tagging.
    enabled : bool
        When ``False``, return a copy of ``df`` unchanged.
    score_direction : str, default ``"more_negative_is_better"``
        Passed to :func:`physics_prefilter`.

    Returns
    -------
    filtered : pd.DataFrame
        Rows retained after prefilter (HARD_FAIL removed when enabled).
    meta : dict[str, Any]
        ``prefilter_enabled``, ``n_dropped_hard_fail``, and optionally
        ``n_remaining``.
    """
    if not enabled:
        return df.copy(), {"prefilter_enabled": False, "n_dropped_hard_fail": 0}
    tagged = physics_prefilter(df, score_direction=score_direction)
    keep = tagged["prefilter"] != PrefilterCategory.HARD_FAIL.value
    dropped = int((~keep).sum())
    return tagged.loc[keep].copy(), {
        "prefilter_enabled": True,
        "n_dropped_hard_fail": dropped,
        "n_remaining": int(keep.sum()),
    }


def run_ablation_config(
    df: pd.DataFrame,
    splits: list[dict[str, Any]],
    config: AblationConfig,
    *,
    encoding: str = "hybrid",
    random_seed: int = 42,
    score_direction: str = "more_negative_is_better",
    repo_root: Path | None = None,
    structure_confidence_dir: str = "data/processed",
    structure_confidence_pattern: str = "structural_confidence_{source}.parquet",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit and predict one ablation configuration over shared CV splits.

    Parameters
    ----------
    df : pd.DataFrame
        Fitness-labeled construct table.
    splits : list[dict[str, Any]]
        Cross-validation splits with ``split_id``, ``train_construct_ids``, and
        ``held_out_construct_ids``.
    config : AblationConfig
        Ablation cell defining model flags and data options.
    encoding : str, default ``"hybrid"``
        Feature encoding passed to :class:`FeatureBuilder`.
    random_seed : int, default 42
        Random seed for the fused surrogate GP.
    score_direction : str, default ``"more_negative_is_better"``
        Physics score direction for prefiltering.
    repo_root : Path, optional
        Repository root for structure-confidence artifacts.
    structure_confidence_dir : str, default ``"data/processed"``
        Directory for per-source confidence parquet files.
    structure_confidence_pattern : str
        Filename pattern with ``{source}`` placeholder.

    Returns
    -------
    preds : pd.DataFrame
        Per-construct held-out predictions with ablation metadata and errors.
    meta : dict[str, Any]
        Config row, structure/prefilter provenance, prediction count, and
        summary metrics.
    """
    work, struct_meta = attach_structure_confidence(
        df,
        config.structure_source,
        repo_root=repo_root,
        confidence_dir=structure_confidence_dir,
        pattern=structure_confidence_pattern,
    )
    work, pref_meta = apply_prefilter_mask(
        work, enabled=config.prefilter, score_direction=score_direction
    )
    work = work[work["fitness"].notna()].copy()
    kind = config.model_kind()
    use_conf = config.use_confidence_weighting()
    include_physics = bool(config.physics)

    rows: list[dict[str, Any]] = []
    for split in splits:
        train_df = _subset(work, split["train_construct_ids"])
        test_df = _subset(work, split["held_out_construct_ids"])
        if train_df.empty or test_df.empty:
            continue
        fb = FeatureBuilder(
            encoding=encoding,  # type: ignore[arg-type]
            include_physics=include_physics,
            include_struct_confidence=config.structure_source is not None,
        )
        model = FusedSurrogate(
            kind=kind,
            use_confidence_weighting=use_conf,
            random_state=random_seed,
            encoding=encoding,
            feature_builder=fb,
        )
        model.fit(train_df, train_df["fitness"].to_numpy(dtype=float))
        pred = model.predict(test_df)
        for i, cid in enumerate(pred.construct_ids):
            y_true = float(
                test_df.loc[test_df["construct_id"].astype(str) == cid, "fitness"].iloc[0]
            )
            y_pred = float(pred.fitness_mean[i])
            rows.append(
                {
                    "ablation_id": config.id,
                    "ablation_label": config.label or config.id,
                    "physics": config.physics,
                    "gp": config.gp,
                    "confidence_weighting": config.confidence_weighting,
                    "structure_source": config.structure_source,
                    "prefilter": config.prefilter,
                    "model_kind": kind,
                    "encoding": encoding,
                    "split_id": split["split_id"],
                    "construct_id": cid,
                    "y_true": y_true,
                    "fitness_mean": y_pred,
                    "fitness_std": float(pred.fitness_std[i]),
                    "physics_mean": float(pred.physics_mean[i]),
                    "gp_residual_mean": float(pred.gp_residual_mean[i]),
                    "abs_error": abs(y_true - y_pred),
                    "sq_error": (y_true - y_pred) ** 2,
                    "random_seed": random_seed,
                }
            )

    preds = pd.DataFrame(rows)
    summary = summarize_ablation_predictions(preds, config)
    meta = {
        **config.as_row(),
        **struct_meta,
        **pref_meta,
        "n_prediction_rows": int(len(preds)),
        "summary": summary,
    }
    return preds, meta


def summarize_ablation_predictions(
    preds: pd.DataFrame, config: AblationConfig | None = None
) -> dict[str, Any]:
    """Compute aggregate regression metrics for ablation predictions.

    Parameters
    ----------
    preds : pd.DataFrame
        Prediction table with ``y_true`` and ``fitness_mean`` columns.
    config : AblationConfig, optional
        When provided, ``ablation_id`` is included in the summary dict.

    Returns
    -------
    dict[str, Any]
        Keys include ``n``, ``rmse``, ``mae``, and gate-3 metrics from
        :func:`metrics_for_predictions`; NaN metrics when ``preds`` is empty.
    """
    if preds.empty:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    y_true = preds["y_true"].to_numpy(dtype=float)
    y_pred = preds["fitness_mean"].to_numpy(dtype=float)
    metrics = metrics_for_predictions(y_true, y_pred)
    out: dict[str, Any] = {"n": int(len(preds)), **metrics}
    if config is not None:
        out["ablation_id"] = config.id
    return out


def run_ablation_matrix(
    df: pd.DataFrame,
    *,
    configs: list[AblationConfig] | None = None,
    splits: list[dict[str, Any]] | None = None,
    splits_dir: Path | None = None,
    encoding: str = "hybrid",
    random_seed: int = 42,
    score_direction: str = "more_negative_is_better",
    repo_root: Path | None = None,
    ablation_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every ablation configuration on identical splits and seed.

    Parameters
    ----------
    df : pd.DataFrame
        Fitness-labeled construct table.
    configs : list[AblationConfig], optional
        Ablation cells to evaluate. Loaded from YAML when ``None``.
    splits : list[dict[str, Any]], optional
        Precomputed CV splits. Generated via :func:`ensure_splits_for_fitness`
        when ``None``.
    splits_dir : Path, optional
        Directory for persisted split artifacts.
    encoding : str, default ``"hybrid"``
        Feature encoding for all configs.
    random_seed : int, default 42
        Shared random seed across configs.
    score_direction : str, default ``"more_negative_is_better"``
        Physics score direction for optional prefiltering.
    repo_root : Path, optional
        Repository root for artifact resolution.
    ablation_settings : dict[str, Any], optional
        Raw ablation YAML settings (structure paths, etc.).

    Returns
    -------
    dict[str, Any]
        Keys: ``predictions``, ``configs``, ``config_meta``, ``metrics_table``,
        ``n_splits``, ``random_seed``, ``encoding``, ``settings``.
    """
    root = repo_root or REPO_ROOT
    settings = ablation_settings or {}
    if configs is None:
        configs, settings = load_ablation_matrix()

    if splits is None:
        splits = ensure_splits_for_fitness(
            df,
            splits_dir,
            prefer_loco=True,
            random_seed=random_seed,
        )

    conf_dir = str(settings.get("structure_confidence_dir", "data/processed"))
    conf_pat = str(
        settings.get("structure_confidence_pattern", "structural_confidence_{source}.parquet")
    )

    all_preds: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    for cfg in configs:
        preds, meta = run_ablation_config(
            df,
            splits,
            cfg,
            encoding=encoding,
            random_seed=random_seed,
            score_direction=score_direction,
            repo_root=root,
            structure_confidence_dir=conf_dir,
            structure_confidence_pattern=conf_pat,
        )
        all_preds.append(preds)
        metas.append(meta)

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metric_rows = []
    for m in metas:
        row = {
            "ablation_id": m.get("id"),
            "label": m.get("label"),
            "physics": m.get("physics"),
            "gp": m.get("gp"),
            "confidence_weighting": m.get("confidence_weighting"),
            "structure_source": m.get("structure_source"),
            "prefilter": m.get("prefilter"),
            "model_kind": m.get("model_kind"),
            "structure_available": m.get("structure_available"),
            "n_dropped_hard_fail": m.get("n_dropped_hard_fail"),
        }
        row.update(m.get("summary") or {})
        metric_rows.append(row)
    metrics_table = pd.DataFrame(metric_rows)

    return {
        "predictions": predictions,
        "configs": configs,
        "config_meta": metas,
        "metrics_table": metrics_table,
        "n_splits": len(splits),
        "random_seed": random_seed,
        "encoding": encoding,
        "settings": settings,
    }
