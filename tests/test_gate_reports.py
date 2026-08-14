"""Stage-gate visual reports under outputs/gate_reports/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biosensor_priors.common.gate_reports import (
    gate_reports_root,
    write_stage1_report,
    write_stage2_report,
    write_stage3_report,
    write_stage4_report,
    write_stage5_report,
    write_stage6_report,
)


def test_stage0_gate_report_written(stage0_result, repo_root: Path) -> None:
    master, meta = stage0_result
    report = meta.get("gate_report") or {}
    folder = repo_root / report["directory"]
    assert folder.is_dir()
    assert (folder / "index.md").is_file()
    assert (folder / "gate.json").is_file()
    assert (folder / "stats.json").is_file()
    overview = folder / "overview.png"
    pytest.importorskip("matplotlib")
    assert overview.is_file()
    assert int(report["stats"]["n_constructs"]) == len(master)


def test_other_stage_reports_write_empty_safe(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    from biosensor_priors.common import gate_reports as gr

    monkeypatch.setattr(gr, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pipeline.yaml").write_text(
        "paths:\n  outputs: outputs\n  gate_reports: outputs/gate_reports\n",
        encoding="utf-8",
    )

    r1 = write_stage1_report({"passed": True, "checks": []}, repo_root=tmp_path)
    r2 = write_stage2_report({"passed": False, "tests": []}, repo_root=tmp_path)
    pred = pd.DataFrame(
        {
            "model_kind": ["physics_gp", "gp_zero_mean"],
            "y_true": [0.2, 0.2],
            "fitness_mean": [0.3, 0.4],
            "fitness_std": [0.1, 0.2],
        }
    )
    r3 = write_stage3_report(
        {"passed": False, "fused_kind": "physics_gp", "summary": []},
        predictions=pred,
        repo_root=tmp_path,
    )
    r4 = write_stage4_report(
        {
            "passed": True,
            "checks": [{"name": "exploit_batch", "passed": True}],
        },
        exploit=pd.DataFrame(
            {
                "pred_fitness_mean": [0.7],
                "pred_fitness_std": [0.1],
                "mutation_cost": [1.0],
            }
        ),
        explore=pd.DataFrame(
            {
                "pred_fitness_mean": [0.4],
                "pred_fitness_std": [0.5],
                "mutation_cost": [3.0],
            }
        ),
        design=pd.DataFrame({"parent_version": ["V1.0", "V2.4"]}),
        repo_root=tmp_path,
    )
    r5 = write_stage5_report(
        {"passed": True, "checks": [{"name": "freeze_integrity", "passed": True}]},
        validation={"overall": {"n_matched": 0, "rmse": None}},
        repo_root=tmp_path,
    )
    r6 = write_stage6_report(
        {"passed": True, "checks": [{"name": "ablation_metrics", "passed": True}]},
        metrics=pd.DataFrame({"ablation_id": ["fused"], "rmse": [0.12]}),
        comparisons=pd.DataFrame(
            {
                "config_a": ["fused"],
                "config_b": ["gp"],
                "delta_rmse": [-0.04],
                "delta_rmse_ci_low": [-0.08],
                "delta_rmse_ci_high": [-0.01],
                "cohens_d_abs_error": [0.3],
            }
        ),
        repo_root=tmp_path,
    )
    root = gate_reports_root(tmp_path)
    assert (root / "README.md").is_file()
    for report in (r1, r2, r3, r4, r5, r6):
        folder = tmp_path / report["directory"]
        assert (folder / "index.md").is_file()
        assert (folder / "overview.png").is_file()
