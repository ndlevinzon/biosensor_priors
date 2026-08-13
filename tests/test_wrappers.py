"""Tests for PyRosetta wrapper scaffolds (interface + packing)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biosensor_priors.stage2_physics.score_parser import parse_rif_score_table, parse_rpx_score_table
from biosensor_priors.stage2_physics.wrappers.run_rosetta import run as run_rosetta
from biosensor_priors.stage2_physics.wrappers.run_rpx import run as run_rpx


def test_rosetta_scaffold_writes_parsable_tsv(tmp_path: Path) -> None:
    muts = tmp_path / "mutations.json"
    muts.write_text(
        '{"mutations": [{"mutation": "Q324R", "position": 324, "wt": "Q", "mutant": "R"}]}',
        encoding="utf-8",
    )
    pdb = tmp_path / "model.pdb"
    pdb.write_text("ATOM\n", encoding="utf-8")
    out = tmp_path / "rosetta_out"
    path = run_rosetta(
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
    assert df.iloc[0]["mutation"] == "Q324R"


def test_rpx_scaffold_batch_and_single(tmp_path: Path) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text("ATOM\n", encoding="utf-8")
    out = tmp_path / "rpx_out"
    out.mkdir()
    (out / "mutations.json").write_text(
        '[{"mutation": "Q324R"}, {"mutation": "A355R"}]',
        encoding="utf-8",
    )
    path = run_rpx(
        structure=pdb,
        mutation="BATCH",
        out=out,
        structure_model_id="m1",
        force_scaffold=True,
    )
    df = parse_rpx_score_table(path)
    assert set(df["mutation"]) == {"Q324R", "A355R"}
    assert "rpx" in df.columns

    path2 = run_rpx(
        structure=pdb,
        mutation="Q324R",
        out=tmp_path / "rpx_one",
        force_scaffold=True,
    )
    df2 = pd.read_csv(path2, sep="\t")
    assert list(df2["mutation"]) == ["Q324R"]
