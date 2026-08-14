"""Stage 4 orchestration: design space + prefilter + search policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import write_manifest
from biosensor_priors.stage0_ground_truth.fitness import FoldFitnessScaler
from biosensor_priors.stage3_surrogate.attach_priors import (
    attach_physics_and_confidence,
    resolve_multi_mutant,
)
from biosensor_priors.stage3_surrogate.surrogate import (
    FusedSurrogate,
    surrogate_kwargs_from_cfg,
)
from biosensor_priors.stage4_search.design_space import build_design_library
from biosensor_priors.stage4_search.policy import build_search_policies
from biosensor_priors.stage4_search.prefilter import (
    physics_prefilter,
    select_search_pools,
)
from biosensor_priors.stage4_search.proposals import (
    split_exploit_explore,
    write_stage4_proposals,
)


def _load_master(root: Path) -> pd.DataFrame:
    """Load the processed experiment master table from disk.

    Parameters
    ----------
    root : Path
        Repository root containing ``data/processed/``.

    Returns
    -------
    pd.DataFrame
        Experiment master table preferring pickle over parquet when both exist.
    """
    pkl = root / "data" / "processed" / "experiment_master.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    return pd.read_parquet(root / "data" / "processed" / "experiment_master.parquet")


def _build_policies(search_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Instantiate all configured Stage-4 search policies."""
    return build_search_policies(search_cfg, seed)


def run_stage4(
    *,
    repo_root: Path | None = None,
    mutable_positions: list[int] | None = None,
    use_measured_holdout_pool: bool = True,
    freeze_round: int | str | None = None,
    freeze_strategy: str = "exploit",
) -> dict[str, Any]:
    """Fit fused surrogate on observed fitness rows and propose batches.

    By default, also runs a measured-landscape benchmark using unobserved
    constructs with fitness as an oracle pool (retrospective). When mutable
    positions are provided (or set in fitness.yaml design), also generates a
    combinatorial design-space batch.

    If ``freeze_round`` is set, the exploit proposal batch (or
    ``freeze_strategy``) is written immutably under ``data/rounds/``
    (Stage 5A) before synthesis.

    Parameters
    ----------
    repo_root : Path or None, optional
        Repository root; defaults to configured ``REPO_ROOT``.
    mutable_positions : list of int or None, optional
        Canonical mutable positions for combinatorial design; falls back to config.
    use_measured_holdout_pool : bool, optional
        When True, run retrospective holdout-pool proposals (default True).
    freeze_round : int, str, or None, optional
        Round identifier for immutable prediction freeze before wet lab.
        freeze_strategy : str, optional
        Design-space batch to freeze (default ``"exploit"``).

    Returns
    -------
    dict
        Keys include ``batches``, ``manifest_path``, ``output_dir``, ``surrogate``,
        ``n_design_candidates``, and optional ``freeze`` metadata.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    search_cfg = load_yaml(root / "configs" / "search.yaml")
    fitness_cfg = load_yaml(root / "configs" / "fitness.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    seed = int(pipeline.get("random_seed", 42))
    batch_size = int(search_cfg.get("batch_size", 8))
    priors_cfg = thresholds.get("priors", {})
    multi_mutant = resolve_multi_mutant(priors_cfg.get("multi_mutant"))

    master = attach_physics_and_confidence(
        _load_master(root),
        repo_root=root,
        multi_mutant=multi_mutant,
    )
    observed = master[master["fitness"].notna()].copy()
    if observed.empty:
        raise RuntimeError("No constructs with fitness available for Stage 4.")
    observed = FoldFitnessScaler(
        weights=fitness_cfg.get("weights"),
        min_components=int(fitness_cfg.get("min_components", 2)),
    ).fit_transform(observed)
    observed = observed[observed["fitness"].notna()].copy()

    surrogate = FusedSurrogate(
        kind="physics_gp",
        random_state=seed,
        **surrogate_kwargs_from_cfg(thresholds.get("gp", {}), fitness_cfg),
    )
    surrogate.fit(observed, observed["fitness"].to_numpy(dtype=float))
    cal_path = (
        resolve_path(pipeline["paths"]["outputs"], root)
        / "stage3"
        / "uncertainty_calibration.json"
    )
    if cal_path.exists():
        import json

        from biosensor_priors.stage3_surrogate.calibration import UncertaintyCalibrator

        payload = json.loads(cal_path.read_text(encoding="utf-8"))
        fields = UncertaintyCalibrator.__dataclass_fields__
        surrogate.calibrator_ = UncertaintyCalibrator(
            **{k: v for k, v in payload.items() if k in fields}
        )

    policies = _build_policies(search_cfg, seed)
    strategy_names = list(
        search_cfg.get("strategies", ["random", "adalead", "mcmc", "bo", "thompson"])
    )

    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage4"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_tables = []

    if use_measured_holdout_pool:
        # Retrospective pool: constructs lacking fitness cannot be scored by oracle,
        # so use a random subset of fitness constructs as "unobserved" for demo search.
        # For true retrospective campaigns, Stage 5 freezes predictions; here we
        # leave out 30% as candidate pool.
        ids = observed["construct_id"].astype(str).tolist()
        rng_ids = pd.Series(ids).sample(frac=0.3, random_state=seed).tolist()
        pool = observed[observed["construct_id"].astype(str).isin(rng_ids)].copy()
        train = observed[~observed["construct_id"].astype(str).isin(rng_ids)].copy()
        # Refit on train only for fair proposal
        surrogate.fit(train, train["fitness"].to_numpy(dtype=float))
        pool = physics_prefilter(
            pool,
            score_direction=thresholds.get("physics", {}).get(
                "score_direction", "more_negative_is_better"
            ),
        )
        pools = select_search_pools(
            pool,
            hard_fail_exclude=bool(search_cfg.get("prefilter", {}).get("hard_fail_exclude", True)),
        )
        main_pool = pools["main"]
        # Reserve exploration budget
        explore_frac = float(search_cfg.get("prefilter", {}).get("exploration_budget_fraction", 0.15))
        n_explore = max(0, int(round(batch_size * explore_frac)))
        n_main = max(1, batch_size - n_explore)

        for name in strategy_names:
            policy = policies[name]
            batch = policy.propose(train, main_pool, surrogate, n_main)
            if n_explore and not pools["exploration"].empty:
                extra = policies["random"].propose(train, pools["exploration"], surrogate, n_explore)
                batch = pd.concat([batch, extra], ignore_index=True)
            batch = batch.head(batch_size).copy()
            batch["search_strategy"] = name
            batch["pool_type"] = "measured_holdout"
            batch_tables.append(batch)
            batch.to_csv(out_dir / f"batch_measured_{name}.csv", index=False)

    # Combinatorial design space (multi-parent substitutions + indels)
    positions = mutable_positions or list(
        fitness_cfg.get("design", {}).get("allowed_mutable_positions") or []
    )
    if mutable_positions:
        fitness_cfg = dict(fitness_cfg)
        design_cfg = dict(fitness_cfg.get("design") or {})
        design_cfg["allowed_mutable_positions"] = list(mutable_positions)
        fitness_cfg["design"] = design_cfg

    versions_path = resolve_path(pipeline["paths"]["constructs"], root) / pipeline[
        "constructs"
    ]["versions_pickle"]
    mapping_path = resolve_path(pipeline["paths"]["constructs"], root) / pipeline[
        "constructs"
    ]["residue_mapping_pickle"]
    versions = pd.read_pickle(versions_path)
    residue_mapping = pd.read_pickle(mapping_path)
    parent = pipeline.get("active_design_background", "V2.4")
    if parent not in set(versions["Version"].astype(str)):
        parent = str(versions["Version"].iloc[-1])

    design = build_design_library(
        versions,
        residue_mapping,
        fitness_cfg,
        default_parent=parent,
    )
    design = attach_physics_and_confidence(
        design,
        repo_root=root,
        multi_mutant=multi_mutant,
    )
    # Exclude already-observed mutation sets
    observed_keys = {
        (
            str(row.get("version") or ""),
            tuple(sorted(map(str, row["mutation_codes"]))),
        )
        for _, row in observed.iterrows()
        if isinstance(row.get("mutation_codes"), list)
    }
    if observed_keys and not design.empty:
        keep = []
        for _, row in design.iterrows():
            key = (
                str(row.get("parent_version") or row.get("version") or ""),
                tuple(sorted(map(str, row.get("mutation_codes") or []))),
            )
            keep.append(key not in observed_keys)
        design = design.loc[keep].reset_index(drop=True)

    design = physics_prefilter(
        design,
        score_direction=thresholds.get("physics", {}).get(
            "score_direction", "more_negative_is_better"
        ),
    )
    pools = select_search_pools(design)
    main_pool = pools["main"]
    # Refit on all observed for design proposals
    surrogate.fit(observed, observed["fitness"].to_numpy(dtype=float))
    exploit = pd.DataFrame()
    explore = pd.DataFrame()
    proposal_paths: dict[str, str] = {}
    if not main_pool.empty:
        exploit, explore = split_exploit_explore(
            observed,
            main_pool,
            surrogate,
            search_cfg=search_cfg,
            fitness_cfg=fitness_cfg,
        )
        proposal_paths = write_stage4_proposals(exploit, explore, out_dir)
        for table, role in ((exploit, "exploit"), (explore, "explore")):
            if table.empty:
                continue
            tagged = table.copy()
            tagged["search_strategy"] = role
            tagged["pool_type"] = "design_space"
            batch_tables.append(tagged)
    for name in strategy_names:
        policy = policies[name]
        batch = policy.propose(observed, main_pool, surrogate, batch_size)
        batch["search_strategy"] = name
        batch["pool_type"] = "design_space"
        batch_tables.append(batch)
        batch.to_csv(out_dir / f"batch_design_{name}.csv", index=False)

    all_batches = pd.concat(batch_tables, ignore_index=True) if batch_tables else pd.DataFrame()
    all_path = out_dir / "all_batches.parquet"
    # stringify object cols for parquet
    store = all_batches.copy()
    for col in store.columns:
        if store[col].dtype == object:
            store[col] = store[col].map(lambda x: None if x is None else str(x))
    store.to_parquet(all_path, index=False)

    freeze_meta = None
    if freeze_round is not None:
        from biosensor_priors.stage5_prospective.run import freeze_round_batch

        to_freeze = None
        if freeze_strategy == "exploit" and not exploit.empty:
            to_freeze = exploit.copy()
        elif freeze_strategy == "explore" and not explore.empty:
            to_freeze = explore.copy()
        if to_freeze is None and not exploit.empty:
            to_freeze = exploit.copy()
        if to_freeze is None:
            design_batches = [
                b
                for b in batch_tables
                if "pool_type" in b.columns and (b["pool_type"] == "design_space").all()
            ]
            for b in design_batches:
                if "search_strategy" in b.columns and (
                    b["search_strategy"] == freeze_strategy
                ).all():
                    to_freeze = b.copy()
                    break
            if to_freeze is None and design_batches:
                to_freeze = design_batches[0].copy()
        if to_freeze is not None:
            if "selection_algorithm" not in to_freeze.columns:
                to_freeze["selection_algorithm"] = to_freeze.get(
                    "search_strategy", freeze_strategy
                )
            if "selection_rank" not in to_freeze.columns:
                to_freeze["selection_rank"] = range(1, len(to_freeze) + 1)
            freeze_meta = freeze_round_batch(
                to_freeze, round_id=freeze_round, repo_root=root
            )

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage4_manifest.json",
        stage="stage4_search",
        inputs={
            "n_observed": int(len(observed)),
            "parent_version": parent,
            "mutable_positions": positions,
            "n_design_candidates": int(len(design)),
        },
        parameters={
            "search": search_cfg,
            "batch_size": batch_size,
            "freeze_round": freeze_round,
            "freeze_strategy": freeze_strategy,
        },
        outputs={
            "all_batches": str(all_path.relative_to(root)),
            "n_batch_rows": int(len(all_batches)),
            "n_exploit": int(len(exploit)),
            "n_explore": int(len(explore)),
            "proposals": {
                k: str(Path(v).relative_to(root)) if Path(v).is_absolute() else v
                for k, v in proposal_paths.items()
            },
            "freeze": freeze_meta,
        },
        random_seed=seed,
        gate={"passed": True, "notes": "Stage 4 advisory; proposals generated."},
    )

    return {
        "batches": all_batches,
        "manifest_path": manifest,
        "output_dir": out_dir,
        "surrogate": surrogate,
        "n_design_candidates": int(len(design)),
        "exploit": exploit,
        "explore": explore,
        "freeze": freeze_meta,
    }


def main() -> None:
    """CLI entry point for Stage 4 active-learning proposal generation.

    Parameters
    ----------
    None
        Arguments are parsed from ``sys.argv`` via ``argparse``.

    Returns
    -------
    None
        Prints batch counts and output paths to stdout.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4 active-learning proposals")
    parser.add_argument(
        "--freeze-round",
        default=None,
        help="If set, freeze the design batch for this round under data/rounds/",
    )
    parser.add_argument(
        "--freeze-strategy",
        default="exploit",
        help="Which design-space batch to freeze (default: exploit)",
    )
    args = parser.parse_args()
    result = run_stage4(freeze_round=args.freeze_round, freeze_strategy=args.freeze_strategy)
    print(f"Design candidates: {result['n_design_candidates']}")
    print(f"Batch rows: {len(result['batches'])}")
    print(f"Wrote: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    if result.get("freeze"):
        print(f"Frozen: {result['freeze']}")


if __name__ == "__main__":
    main()
