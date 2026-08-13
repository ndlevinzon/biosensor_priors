"""Tests for Stage 1 CHPC structure job generation and adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biosensor_priors.stage1_structures.adapters import parse_AF3, parse_Boltz2, parse_RF3
from biosensor_priors.stage1_structures.make_jobs import (
    make_structure_jobs,
    sanitize_af3_name,
    structure_model_id,
    write_af3_input_json,
    write_boltz2_yaml,
    write_fasta,
)
from biosensor_priors.stage1_structures.run import run_stage1

SEQ = "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDRED"


def test_structure_model_id_format():
    assert structure_model_id("V2.4", "Boltz2", 1, "apo") == "V2.4_Boltz2_seed1_apo"


def test_write_boltz2_yaml(tmp_path: Path):
    path = write_boltz2_yaml(tmp_path / "in.yaml", sequence=SEQ)
    text = path.read_text(encoding="utf-8")
    assert "protein:" in text
    assert "GMRESYANEN" in text


def test_boltz2_shares_msa_across_seeds(tmp_path: Path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data" / "structures").mkdir(parents=True)
    (root / "data" / "constructs").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "outputs").mkdir()
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
  predictors: [Boltz2]
  seeds: [1, 2]
  states: [apo]
  confidence:
    plDDT_min_reliable: 70.0
    rmsd_max_reliable: 2.0
    pae_pocket_max_reliable: 10.0
    confidence_min_reliable: 0.5
""".strip(),
        encoding="utf-8",
    )
    from biosensor_priors.common.config import REPO_ROOT
    import shutil

    shutil.copy(REPO_ROOT / "configs" / "structures.yaml", root / "configs" / "structures.yaml")

    result = make_structure_jobs(
        version="V2.4",
        sequence=SEQ,
        repo_root=root,
        predictors=["Boltz2"],
        seeds=[1, 2],
        states=["apo"],
        submit=False,
    )
    reg = result["registry"].sort_values("seed")
    s1, s2 = reg.iloc[0], reg.iloc[1]
    script1 = (root / s1["step1_script"]).read_text(encoding="utf-8")
    script2 = (root / s2["step1_script"]).read_text(encoding="utf-8")
    assert "--use_msa_server" in script1
    assert "--use_msa_server" not in script2
    assert str(s1["input_path"]).endswith(".fasta")
    assert str(s2["input_path"]).endswith(".yaml")
    yaml2 = (root / s2["input_path"]).read_text(encoding="utf-8")
    assert "msa:" in yaml2
    assert "boltz_results_V2_4_Boltz2_seed1_apo" in yaml2.replace("\\", "/")
    submit = (result["submit_script"]).read_text(encoding="utf-8")
    assert "sbatch --parsable" in submit
    assert "afterok:" in submit


def test_make_jobs_writes_boltz2_af3_scripts(tmp_path: Path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data" / "structures").mkdir(parents=True)
    (root / "data" / "constructs").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "outputs").mkdir()

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
  predictors: [Boltz2, AF3]
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
    from biosensor_priors.common.config import REPO_ROOT
    import shutil

    shutil.copy(REPO_ROOT / "configs" / "structures.yaml", root / "configs" / "structures.yaml")

    result = make_structure_jobs(
        version="V2.4",
        sequence=SEQ,
        repo_root=root,
        predictors=["Boltz2", "AF3"],
        seeds=[1],
        states=["apo"],
        submit=False,
    )
    assert result["n_jobs"] == 2
    registry = result["registry"]
    assert set(registry["method"]) == {"Boltz2", "AF3"}

    bz = registry[registry["method"] == "Boltz2"].iloc[0]
    bz_script = (root / bz["step1_script"]).read_text(encoding="utf-8")
    assert "ml boltz2" in bz_script
    assert "boltz predict" in bz_script
    assert "--use_msa_server" in bz_script
    assert "colabfold02.int.chpc.utah.edu" in bz_script
    assert "#SBATCH -p granite-gpu" in bz_script
    assert "#SBATCH --ntasks-per-node=1" in bz_script
    assert "#SBATCH --cpus-per-task=16" in bz_script
    assert "#SBATCH -n " not in bz_script
    assert "#SBATCH --output=" in bz_script
    assert "/logs/V2.4/V2.4_Boltz2_seed1_apo__boltz2_gpu.out" in bz_script.replace("\\", "/")
    assert "#SBATCH --error=" in bz_script
    assert bz["step2_script"] is None or (isinstance(bz["step2_script"], float) and pd.isna(bz["step2_script"]))

    # CHPC-style FASTA input (sanitized stem; no dots)
    assert str(bz["input_path"]).endswith(".fasta")
    fasta_text = (root / bz["input_path"]).read_text(encoding="utf-8")
    assert fasta_text.startswith(">A|protein")
    assert "GMRESYANEN" in fasta_text

    af3_row = registry[registry["method"] == "AF3"].iloc[0]
    a3s1 = (root / af3_row["step1_script"]).read_text(encoding="utf-8")
    a3s2 = (root / af3_row["step2_script"]).read_text(encoding="utf-8")
    assert "ml alphafold/3.0.0" in a3s1
    assert "--norun_inference" in a3s1
    assert "sbatch -d afterok:${SLURM_JOBID}" in a3s1
    assert "--norun_data_pipeline" in a3s2
    assert "granite-gpu" in a3s2
    assert "#SBATCH --ntasks-per-node=1" in a3s2
    assert "/logs/V2.4/V2.4_AF3_seed1_apo__af3_step1_msa.out" in a3s1.replace("\\", "/")
    assert "/logs/V2.4/V2.4_AF3_seed1_apo__af3_step2_infer.out" in a3s2.replace("\\", "/")
    import json as _json

    js_obj = _json.loads((root / af3_row["input_path"]).read_text(encoding="utf-8"))
    assert js_obj["dialect"] == "alphafold3"
    assert js_obj["modelSeeds"] == [1]


def test_parse_boltz2_from_pdb_tree(tmp_path: Path):
    pred = tmp_path / "predictions" / "job"
    pred.mkdir(parents=True)
    pdb = pred / "job_model_0.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 80.00           C\n",
        encoding="utf-8",
    )
    parsed = parse_Boltz2(tmp_path, version="V2.4", seed=1, state="apo")
    assert len(parsed["models"]) == 1
    assert parsed["models"].iloc[0]["method"] == "Boltz2"
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
        predictors=["Boltz2"],
        seeds=[1, 2],
        states=["apo"],
    )
    assert result["jobs"]["n_jobs"] == 2
    assert result["gate"]["passed"] is True
    assert (root / "data" / "structures" / "job_registry.parquet").exists()


def test_make_jobs_esmfold_and_rf3(tmp_path: Path):
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
        predictors=["ESMFold", "RF3"],
        seeds=[1],
        states=["apo"],
    )
    assert result["n_jobs"] == 2
    reg = result["registry"]
    esm = reg[reg["method"] == "ESMFold"].iloc[0]
    esm_script = (root / esm["step1_script"]).read_text(encoding="utf-8")
    assert "ml esmfold/1.0.3" in esm_script
    assert "run_esmfold.py" in esm_script

    rf = reg[reg["method"] == "RF3"].iloc[0]
    rf_script = (root / rf["step1_script"]).read_text(encoding="utf-8")
    assert "rf3 fold" in rf_script
    assert "early_stopping_plddt_threshold=0" in rf_script
    assert "nvidia-smi" in rf_script
    assert "#SBATCH --ntasks-per-node=1" in rf_script
    assert "#SBATCH -n " not in rf_script
    js = (root / rf["input_path"]).read_text(encoding="utf-8")
    assert '"seq"' in js


def test_parse_rf3(tmp_path: Path):
    pdb = tmp_path / "example_model.pdb"
    pdb.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 77.00           C\n",
        encoding="utf-8",
    )
    parsed = parse_RF3(tmp_path, version="V2.4", seed=1, state="apo")
    assert parsed["models"].iloc[0]["method"] == "RF3"
    assert parsed["residues"].iloc[0]["plddt"] == 77.0


def test_sanitize_af3_name():
    assert sanitize_af3_name("V2.4_AF3_seed1_apo") == "V2_4_AF3_seed1_apo"


def test_write_af3_and_fasta(tmp_path: Path):
    fa = write_fasta(tmp_path / "t.fa", header="h", sequence="ACDE")
    assert fa.read_text(encoding="utf-8").startswith(">h")
    js = write_af3_input_json(tmp_path / "j.json", name="job", sequence="ACDE", seed=3)
    assert "alphafold3" in js.read_text(encoding="utf-8")
