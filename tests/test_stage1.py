"""Tests for Stage 1 CHPC AlphaFold job generation and adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biosensor_priors.stage1_structures.adapters import parse_AF2, parse_AF3
from biosensor_priors.stage1_structures.confidence import compute_structural_confidence
from biosensor_priors.stage1_structures.make_jobs import (
    make_structure_jobs,
    sanitize_af3_name,
    structure_model_id,
    write_af3_input_json,
    write_fasta,
)
from biosensor_priors.stage1_structures.run import run_stage1


SEQ = "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDRED"


def test_structure_model_id_format():
    assert structure_model_id("V2.4", "AF2", 1, "apo") == "V2.4_AF2_seed1_apo"


def test_make_jobs_writes_chpc_af2_af3_scripts(tmp_path: Path, monkeypatch):
    # Isolate writes under tmp by pointing REPO_ROOT via make_structure_jobs repo_root
    # and a minimal config tree.
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data" / "structures").mkdir(parents=True)
    (root / "data" / "constructs").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "outputs").mkdir()

    # Copy real configs (structures + pipeline/thresholds) by writing minimal YAML
    (root / "configs" / "pipeline.yaml").write_text(
        """
canonical_reference: "V1.0"
active_design_background: "V2.4"
random_seed: 42
paths:
  experimental: "data/experimental"
  constructs: "data/constructs"
  structures: "data/structures"
  physics: "data/physics"
  rounds: "data/rounds"
  processed: "data/processed"
  splits: "data/processed/splits"
  manifests: "manifests"
  outputs: "outputs"
constructs:
  versions_pickle: "biosensor_versions_clean.pkl"
gates:
  stage1: advisory
""".strip(),
        encoding="utf-8",
    )
    (root / "configs" / "thresholds.yaml").write_text(
        """
structure:
  predictors: [AF2, AF3]
  seeds: [1]
  states: [apo]
  confidence:
    plDDT_min_reliable: 70.0
    rmsd_max_reliable: 2.0
    pae_pocket_max_reliable: 10.0
    confidence_min_reliable: 0.5
""".strip(),
        encoding="utf-8",
    )
    # Reuse real structures.yaml content from repo via import path — write CHPC essentials
    from biosensor_priors.common.config import REPO_ROOT
    import shutil

    shutil.copy(REPO_ROOT / "configs" / "structures.yaml", root / "configs" / "structures.yaml")

    result = make_structure_jobs(
        version="V2.4",
        sequence=SEQ,
        repo_root=root,
        predictors=["AF2", "AF3"],
        seeds=[1],
        states=["apo"],
        submit=False,
    )
    assert result["n_jobs"] == 2
    registry = result["registry"]
    assert set(registry["method"]) == {"AF2", "AF3"}

    af2_row = registry[registry["method"] == "AF2"].iloc[0]
    step1 = root / af2_row["step1_script"]
    step2 = root / af2_row["step2_script"]
    text1 = step1.read_text(encoding="utf-8")
    text2 = step2.read_text(encoding="utf-8")
    assert "ml alphafold/2.3.2" in text1
    assert "db_to_tmp_232.sh" in text1
    assert "--run_feature=1" in text1
    assert "sbatch -d afterok:${SLURM_JOBID}" in text1
    assert "run_alphafold_full.sh" in text1
    assert "ml alphafold/2.3.2" in text2
    assert "--run_feature=1" not in text2
    assert "--gres=gpu:t4:1" in text2

    af3_row = registry[registry["method"] == "AF3"].iloc[0]
    a3s1 = (root / af3_row["step1_script"]).read_text(encoding="utf-8")
    a3s2 = (root / af3_row["step2_script"]).read_text(encoding="utf-8")
    assert "ml alphafold/3.0.0" in a3s1
    assert "--norun_inference" in a3s1
    assert "sbatch -d afterok:${SLURM_JOBID}" in a3s1
    assert "--norun_data_pipeline" in a3s2
    assert "notchpeak-gpu" in a3s2
    assert "_data.json" in a3s2

    fasta = (root / af2_row["input_path"]).read_text(encoding="utf-8")
    assert fasta.startswith(">")
    assert "GMRESYANEN" in fasta

    import json as _json

    js_obj = _json.loads((root / af3_row["input_path"]).read_text(encoding="utf-8"))
    assert js_obj["dialect"] == "alphafold3"
    assert js_obj["modelSeeds"] == [1]


def test_parse_af2_from_pdb(tmp_path: Path):
    pdb = tmp_path / "ranked_0.pdb"
    # Minimal CA atoms with pLDDT in B-factor
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",
        "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 80.00           C",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = parse_AF2(tmp_path, version="V2.4", seed=1, state="apo")
    assert len(parsed["models"]) == 1
    assert len(parsed["residues"]) == 2
    assert parsed["residues"]["plddt"].tolist() == [90.0, 80.0]


def test_parse_af3_confidences(tmp_path: Path):
    cif = tmp_path / "job_model.cif"
    cif.write_text("data_model\n# placeholder\n", encoding="utf-8")
    conf = tmp_path / "job_confidences.json"
    conf.write_text('{"residue_plddts": [91.0, 88.5, 70.0]}\n', encoding="utf-8")
    parsed = parse_AF3(tmp_path, version="V2.4", seed=2, state="apo", af3_name="job")
    assert len(parsed["models"]) == 1
    assert list(parsed["residues"]["plddt"]) == [91.0, 88.5, 70.0]


def test_confidence_and_run_stage1_jobs_only(tmp_path: Path):
    root = tmp_path
    for d in ("configs", "data/structures", "data/constructs", "manifests", "outputs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    from biosensor_priors.common.config import REPO_ROOT
    import shutil

    shutil.copy(REPO_ROOT / "configs" / "structures.yaml", root / "configs" / "structures.yaml")
    (root / "configs" / "pipeline.yaml").write_text(
        """
active_design_background: "V2.4"
random_seed: 1
paths:
  constructs: "data/constructs"
  structures: "data/structures"
  manifests: "manifests"
  outputs: "outputs"
  experimental: "data/experimental"
  physics: "data/physics"
  rounds: "data/rounds"
  processed: "data/processed"
  splits: "data/processed/splits"
constructs:
  versions_pickle: "biosensor_versions_clean.pkl"
gates:
  stage1: advisory
""".strip(),
        encoding="utf-8",
    )
    (root / "configs" / "thresholds.yaml").write_text(
        "structure:\n  confidence:\n    plDDT_min_reliable: 70.0\n",
        encoding="utf-8",
    )

    result = run_stage1(
        repo_root=root,
        version="V2.4",
        sequence=SEQ,
        make_jobs=True,
        ingest=True,
        predictors=["AF2"],
        seeds=[1, 2],
        states=["apo"],
    )
    assert result["jobs"]["n_jobs"] == 2
    assert result["gate"]["passed"] is True
    assert (root / "data" / "structures" / "job_registry.parquet").exists()


def test_make_jobs_esmfold_and_rf2(tmp_path: Path):
    root = tmp_path
    for d in ("configs", "data/structures", "data/constructs", "manifests", "outputs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    from biosensor_priors.common.config import REPO_ROOT
    import shutil

    shutil.copy(REPO_ROOT / "configs" / "structures.yaml", root / "configs" / "structures.yaml")

    (root / "configs" / "pipeline.yaml").write_text(
        """
active_design_background: "V2.4"
random_seed: 1
paths:
  constructs: "data/constructs"
  structures: "data/structures"
  manifests: "manifests"
  outputs: "outputs"
  experimental: "data/experimental"
  physics: "data/physics"
  rounds: "data/rounds"
  processed: "data/processed"
  splits: "data/processed/splits"
constructs:
  versions_pickle: "biosensor_versions_clean.pkl"
gates:
  stage1: advisory
""".strip(),
        encoding="utf-8",
    )
    (root / "configs" / "thresholds.yaml").write_text("structure: {}\n", encoding="utf-8")

    result = make_structure_jobs(
        version="V2.4",
        sequence=SEQ,
        repo_root=root,
        predictors=["ESMFold", "RF2"],
        seeds=[1],
        states=["apo"],
    )
    assert result["n_jobs"] == 2
    reg = result["registry"]
    esm = reg[reg["method"] == "ESMFold"].iloc[0]
    esm_script = (root / esm["step1_script"]).read_text(encoding="utf-8")
    assert "ml esmfold/1.0.3" in esm_script
    assert "esm-fold -i" in esm_script
    assert "--chunk-size 128" in esm_script

    rf = reg[reg["method"] == "RF2"].iloc[0]
    rf_script = (root / rf["step1_script"]).read_text(encoding="utf-8")
    assert "ml rosettafold2/1.0" in rf_script
    assert "run_RF2.sh" in rf_script
    assert ' -o "$OUTPUT_DIR"' in rf_script


def test_parse_esmfold(tmp_path: Path):
    from biosensor_priors.stage1_structures.adapters import parse_ESMFold

    mid = "V2.4_ESMFold_seed1_apo"
    pdb = tmp_path / f"{mid}.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 85.00           C\n",
        encoding="utf-8",
    )
    parsed = parse_ESMFold(tmp_path, version="V2.4", seed=1, state="apo")
    assert parsed["models"].iloc[0]["method"] == "ESMFold"
    assert parsed["residues"].iloc[0]["plddt"] == 85.0


def test_parse_rf2(tmp_path: Path):
    from biosensor_priors.stage1_structures.adapters import parse_RF2

    models = tmp_path / "models"
    models.mkdir()
    (models / "model_00.pdb").write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 77.00           C\n",
        encoding="utf-8",
    )
    parsed = parse_RF2(tmp_path, version="V2.4", seed=1, state="apo")
    assert parsed["models"].iloc[0]["method"] == "RF2"
    assert parsed["residues"].iloc[0]["plddt"] == 77.0


def test_sanitize_and_writers(tmp_path: Path):
    assert sanitize_af3_name("V2.4_AF3_seed1_apo") == "V2_4_AF3_seed1_apo"
    write_fasta(tmp_path / "x.fasta", header="h", sequence="ACDE")
    write_af3_input_json(tmp_path / "x.json", name="x", sequence="ACDE", seed=3)
    conf = compute_structural_confidence(
        pd.DataFrame(
            [
                {
                    "structure_model_id": "m1",
                    "version": "V2.4",
                    "method": "AF2",
                    "seed": 1,
                    "state": "apo",
                    "structure_path": str(tmp_path / "missing.pdb"),
                    "mean_plddt": 90.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "structure_model_id": "m1",
                    "version": "V2.4",
                    "method": "AF2",
                    "seed": 1,
                    "state": "apo",
                    "residue_index": 1,
                    "canonical_position": 324,
                    "aa": "Q",
                    "plddt": 91.0,
                    "pae_pocket": 3.0,
                }
            ]
        ),
        conf_cfg={
            "plDDT_min_reliable": 70.0,
            "rmsd_max_reliable": 2.0,
            "pae_pocket_max_reliable": 10.0,
            "confidence_min_reliable": 0.5,
            "w_plddt": 0.5,
            "w_rmsd": 0.3,
            "w_pae": 0.2,
        },
    )
    assert len(conf) == 1
    assert conf.iloc[0]["Reliable"] == "yes"
