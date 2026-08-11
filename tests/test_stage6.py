"""Stage 6 ablation matrix + statistics tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biosensor_priors.stage6_ablation.experiments import (
    AblationConfig,
    default_ablation_matrix,
    load_ablation_matrix,
    run_ablation_matrix,
)
from biosensor_priors.stage6_ablation.report import write_report
from biosensor_priors.stage6_ablation.statistics import (
    cliffs_delta,
    cohens_d_paired,
    compare_ablation_pair,
    holm_adjust,
    paired_bootstrap_delta,
    run_ablation_statistics,
)


def test_load_ablation_matrix() -> None:
    configs, raw = load_ablation_matrix()
    assert len(configs) >= 5
    assert "configs" in raw
    ids = {c.id for c in configs}
    assert "physics_only_consensus" in ids
    assert "gp_only" in ids
    assert "physics_gp_conf_consensus" in ids


def test_holm_and_effect_sizes() -> None:
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert len(adj) == 3
    assert adj[0] <= adj[1] or adj[0] <= 0.03
    assert all(0 <= p <= 1 for p in adj)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.1, 2.2, 2.8, 3.5])
    assert np.isfinite(cohens_d_paired(a, b))
    assert -1.0 <= cliffs_delta(a, b) <= 1.0
    boot = paired_bootstrap_delta(a - 0.5, b - 0.5, statistic="rmse", n_boot=200, seed=0)
    assert "ci_low" in boot and "ci_high" in boot


def test_ablation_matrix_shared_splits(stage0_result, tmp_path: Path) -> None:
    master, _ = stage0_result
    fit = master[master["fitness"].notna()].copy()
    ids = fit["construct_id"].astype(str).tolist()
    # Tiny fixed splits for speed (still shared across configs)
    splits = [
        {
            "split_id": "s0",
            "strategy": "holdout",
            "train_construct_ids": ids[2:],
            "held_out_construct_ids": ids[:2],
        },
        {
            "split_id": "s1",
            "strategy": "holdout",
            "train_construct_ids": ids[:2] + ids[4:],
            "held_out_construct_ids": ids[2:4],
        },
    ]
    configs = default_ablation_matrix()[:4]
    result = run_ablation_matrix(
        fit,
        configs=configs,
        splits=splits,
        encoding="hybrid",
        random_seed=0,
    )
    preds = result["predictions"]
    assert not preds.empty
    assert set(preds["ablation_id"]) == {c.id for c in configs}
    # Same seed + splits → every config shares split_ids
    split_sets = [
        set(preds.loc[preds["ablation_id"] == c.id, "split_id"]) for c in configs
    ]
    assert all(s == split_sets[0] for s in split_sets)

    stats = run_ablation_statistics(
        preds,
        reference_config_id=configs[-1].id,
        n_boot=100,
        seed=0,
    )
    assert stats["ok"]
    assert stats["n_comparisons"] == len(configs) - 1
    pair = compare_ablation_pair(preds, configs[0].id, configs[-1].id, n_boot=50, seed=0)
    assert pair["ok"]

    arts = write_report(
        out_dir=tmp_path / "stage6",
        metrics=result["metrics_table"],
        stats_report=stats,
        predictions=preds,
        meta={"random_seed": 0, "encoding": "hybrid", "n_splits": result["n_splits"]},
    )
    assert Path(arts["report_md"]).exists()
    assert Path(arts["metrics_csv"]).exists()
    assert Path(arts["comparisons_csv"]).exists()


def test_ablation_config_model_kinds() -> None:
    assert AblationConfig("a", True, False, None, "consensus", False).model_kind() == "physics_only"
    assert AblationConfig("b", False, True, None, None, False).model_kind() == "gp_zero_mean"
    assert AblationConfig("c", True, True, True, "AF2", True).model_kind() == "physics_gp"
    assert AblationConfig("d", True, True, None, "consensus", False).use_confidence_weighting() is False
