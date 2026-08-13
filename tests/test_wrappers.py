"""Tests for RF3 docking wrapper scaffolds."""

from __future__ import annotations

import json
from pathlib import Path

from biosensor_priors.stage2_physics.score_parser import parse_rif_score_table
from biosensor_priors.stage2_physics.wrappers.run_rf3_dock import (
    confidence_to_score,
    pick_metric,
    write_rf3_dock_json,
)
from biosensor_priors.stage2_physics.wrappers.run_rf3_dock import run as run_rf3_dock


def test_rf3_scaffold_writes_parsable_tsv(tmp_path: Path) -> None:
    muts = tmp_path / "mutations.json"
    muts.write_text(
        json.dumps(
            {
                "mutations": [
                    {
                        "mutation": "Q324R",
                        "position": 324,
                        "wt": "Q",
                        "mutant": "R",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pdb = tmp_path / "model.pdb"
    pdb.write_text("ATOM\n", encoding="utf-8")
    out = tmp_path / "rf3_out"
    path = run_rf3_dock(
        structure=pdb,
        ligands=tmp_path / "ligands",
        ligand_name="AcCoA+PropCoA",
        out=out,
        mutations_json=muts,
        structure_model_id="V2.4_Boltz2_seed1_apo",
        force_scaffold=True,
    )
    assert path.exists()
    assert (out / "wrapper_status.json").exists()
    df = parse_rif_score_table(path)
    assert "rif_ac" in df.columns
    assert "rif_prop" in df.columns
    assert "rpx" not in df.columns
    assert df.iloc[0]["mutation"] == "Q324R"


def test_rf3_dock_json_and_score_mapping(tmp_path: Path) -> None:
    cfg = {
        "protein_chain_id": "A",
        "ligand_chain_id": "B",
        "template_protein": True,
        "ground_truth_ligand_conformer": False,
    }
    template = tmp_path / "t.pdb"
    template.write_text("ATOM\n", encoding="utf-8")
    path = write_rf3_dock_json(
        tmp_path / "dock.json",
        name="Q324R_AcCoA",
        sequence="ACDE",
        protein_template=template,
        ligand_component={"smiles": "CCO", "chain_id": "B"},
        cfg=cfg,
        apo=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["components"][0]["seq"] == "ACDE"
    assert payload["components"][1]["smiles"] == "CCO"
    assert payload["template_selection"] == ["A"]
    assert confidence_to_score(0.8, negate=True) == -0.8
    assert pick_metric({"iptm": 0.9, "ptm": 0.7}, ["iptm", "ptm"]) == 0.9
