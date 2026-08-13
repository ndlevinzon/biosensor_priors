"""Stage 2 physics orchestration + Gate 2 tests."""

from __future__ import annotations

import pandas as pd
import pytest

from biosensor_priors.common.config import REPO_ROOT, load_yaml
from biosensor_priors.stage2_physics.gate2 import (
    assert_physics_weight_allowed,
    check_A355R_direction,
    check_Q324R_direction,
    evaluate_gate2,
    expected_delta_sign,
)
from biosensor_priors.stage2_physics.ligand_ensemble import (
    make_conformer_id,
    run_ligand_ensemble,
)
from biosensor_priors.stage2_physics.mutation_scan import (
    default_structure_models,
    generate_mutation_specs,
    run_mutation_scan,
)
from biosensor_priors.stage2_physics.physics_uncertainty import aggregate_physics_uncertainty
from biosensor_priors.stage2_physics.score_parser import compute_delta_rif_sel


def test_conformer_id_stable() -> None:
    a = make_conformer_id("AcCoA", content_hash="abc123def456", schema_version=1, index=0)
    b = make_conformer_id("AcCoA", content_hash="abc123def456", schema_version=1, index=0)
    assert a == b
    assert a.startswith("AcCoA:v1:")


def test_delta_rif_sel_definition() -> None:
    assert compute_delta_rif_sel(-10.0, -8.0) == pytest.approx(-2.0)
    assert expected_delta_sign("favorable_AcCoA", "more_negative_is_better") == -1


def test_ligand_ensemble_writes_catalog() -> None:
    result = run_ligand_ensemble(repo_root=REPO_ROOT, n_placeholder=2)
    assert result.catalog_path.exists()
    assert not result.conformers.empty
    assert set(result.conformers["ligand"]) >= {"AcCoA", "PropCoA"}
    assert result.conformers["conformer_id"].is_unique
    assert (REPO_ROOT / "data" / "physics" / "ligands" / "AcCoA" / "approved").exists()
    # Gaussian16 template SLURM written even under mock (no OMEGA; QM via g16)
    qm_dir = REPO_ROOT / "data" / "physics" / "ligands" / "AcCoA" / "qm"
    assert any(qm_dir.glob("*.slurm"))
    slurm_text = next(qm_dir.glob("*.slurm")).read_text(encoding="utf-8")
    assert "gaussian16/SSE4.C01" in slurm_text
    assert "g16" in slurm_text


def test_gaussian_gjf_and_log_parser(tmp_path) -> None:
    from biosensor_priors.stage2_physics.gaussian_qm import (
        parse_gaussian_optimized_xyz,
        write_gaussian_gjf,
        write_gaussian_slurm,
    )

    gjf = write_gaussian_gjf(
        tmp_path / "m.gjf",
        atoms=[("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.1)],
        title="test",
        charge=0,
        route="#p B3LYP/6-31G(d) Opt",
    )
    text = gjf.read_text(encoding="utf-8")
    assert "%NProcShared=" in text
    assert "#p B3LYP/6-31G(d) Opt" in text
    slurm = write_gaussian_slurm(
        tmp_path / "m.slurm",
        gjf_path=gjf,
        qm_cfg={
            "module": "gaussian16/SSE4.C01",
            "executable": "g16",
            "job": {"partition": "notchpeak", "account": "test", "time": "1:00:00", "ntasks": 4, "mem": "8G"},
        },
    )
    s = slurm.read_text(encoding="utf-8")
    assert "ml gaussian16/SSE4.C01" in s
    log = tmp_path / "m.log"
    log.write_text(
        """
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    1.090000
 ---------------------------------------------------------------------
""",
        encoding="utf-8",
    )
    atoms = parse_gaussian_optimized_xyz(log)
    assert atoms[0][0] == "C"
    assert atoms[1][0] == "H"


def test_mutation_scan_and_gate2(stage0_result) -> None:
    _ = stage0_result
    # Small scan for speed
    models = default_structure_models(version="V2.4", predictors=["Boltz2"], seeds=[1, 2])
    scan = run_mutation_scan(
        repo_root=REPO_ROOT,
        structure_models=models,
        positions=[324, 355],
        amino_acids=["A", "R", "Q"],
    )
    long_table = scan["long_table"]
    assert not long_table.empty
    assert {
        "rif_ac",
        "rif_prop",
        "delta_rif_sel",
        "mutation",
    }.issubset(long_table.columns)
    assert "rpx" not in long_table.columns
    # Controls present
    assert "Q324R" in set(long_table["mutation"])
    assert "A355R" in set(long_table["mutation"])

    summary = aggregate_physics_uncertainty(long_table)
    assert "rif_ac_mean" in summary.columns
    assert "n_structures" in summary.columns
    assert summary.loc[summary["mutation"] == "Q324R", "n_structures"].iloc[0] == 2

    q = check_Q324R_direction(summary)
    a = check_A355R_direction(summary)
    assert q["passed"], q
    assert a["passed"], a

    gate = evaluate_gate2(summary, repo_root=REPO_ROOT)
    assert gate["physics_gate"] == "PASS"
    assert gate["allow_full_physics_weight"] is True
    assert_physics_weight_allowed(gate)


def test_gate2_fails_on_wrong_direction() -> None:
    bad = pd.DataFrame(
        [
            {
                "mutation": "Q324R",
                "delta_rif_sel_mean": 3.0,  # wrong sign for favorable_AcCoA / more_negative
                "n_structures": 3,
            },
            {
                "mutation": "A355R",
                "delta_rif_sel_mean": -2.0,
                "n_structures": 3,
            },
        ]
    )
    gate = evaluate_gate2(bad, repo_root=REPO_ROOT)
    assert gate["physics_gate"] == "FAIL"
    assert "Q324R" in gate["failed"]
    with pytest.raises(RuntimeError):
        assert_physics_weight_allowed(gate)


def test_generate_mutation_specs_20aa() -> None:
    versions = pd.DataFrame({"Version": ["V2.4"], "Sequence": ["M" * 400]})
    specs = generate_mutation_specs(
        version="V2.4",
        positions=[324],
        amino_acids=list("ACDEFGHIKLMNPQRSTVWY"),
        versions=versions,
        include_wt=True,
    )
    assert len(specs) == 20
    assert all(s["position"] == 324 for s in specs)


def test_score_direction_frozen_in_config() -> None:
    thresholds = load_yaml(REPO_ROOT / "configs" / "thresholds.yaml")
    assert thresholds["physics"]["score_direction"] == "more_negative_is_better"
