"""Data-hygiene: priors join, fold labels, MISMATCH, LOCO, physchem."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.fitness import FoldFitnessScaler
from biosensor_priors.stage0_ground_truth.splits import load_split
from biosensor_priors.stage3_surrogate.attach_priors import (
    attach_physics_and_confidence,
)
from biosensor_priors.stage3_surrogate.features import (
    FeatureBuilder,
    _aa_lookup_with_z,
    is_binary_feature_name,
)
from biosensor_priors.stage4_search.landscape import parse_mutation_list
from biosensor_priors.stage4_search.prefilter import physics_prefilter


def test_mismatch_not_parsed_and_unlabeled(stage0_result) -> None:
    master, _ = stage0_result
    row = master.loc[master["construct_id"].astype(str) == "Pan1.0 Q324R"].iloc[0]
    assert str(row["mutation_audit"]) == "MISMATCH"
    assert pd.isna(row["fitness"])
    assert parse_mutation_list(row) == []

    fake = pd.Series(
        {
            "mutation_audit": "MISMATCH",
            "mut_from_construct": [("Q", 324, "R")],
            "mut_codes_construct": ["Q324R"],
            "mutation_codes": ["Q324R"],
        }
    )
    assert parse_mutation_list(fake) == []


def test_d104_insertion_is_an_edit() -> None:
    row = pd.Series(
        {
            "mutation_audit": "match",
            "Construct": "PancACe 2.0 D104 insertion AS-AS Linker",
            "Description": "D104 insertion",
            "mut_from_construct": [],
            "mut_from_description": [],
        }
    )
    assert parse_mutation_list(row) == [("+", 104, "X")]


def test_frozen_splits_are_loco(stage0_result) -> None:
    _, meta = stage0_result
    paths = sorted(Path(meta["splits_dir"]).glob("split_*.json"))
    assert paths
    for path in paths:
        split = load_split(path)
        assert split["strategy"] == "leave_one_construct_out"
        assert split["n_held_out"] == 1
        assert len(split["held_out_construct_ids"]) == 1


def test_fc_prop_in_fitness(stage0_result) -> None:
    master, _ = stage0_result
    assert "_fitness_fc_prop_raw" in master.columns
    assert "_fitness_fc_prop_score" in master.columns
    assert "Fitness_weight_fc_prop" in master.columns
    assert np.isclose(float(master["Fitness_weight_fc_prop"].iloc[0]), 0.20)
    assert np.isclose(float(master["Fitness_weight_brightness"].iloc[0]), 0.25)


def test_fold_scaler_does_not_leak_held_out() -> None:
    train_vals = [1.0, 2.0, 3.0, 4.0]
    test_vals = [100.0]

    def _frame(vals: list[float], audit: str = "match") -> pd.DataFrame:
        n = len(vals)
        return pd.DataFrame(
            {
                "_fitness_selectivity_raw": vals,
                "_fitness_affinity_raw": vals,
                "_fitness_fc_raw": vals,
                "_fitness_brightness_raw": vals,
                "mutation_audit": [audit] * n,
                "construct_id": [f"c{i}" for i in range(n)],
            }
        )

    train = _frame(train_vals)
    test = _frame(test_vals)
    scaler = FoldFitnessScaler().fit(train)
    honest = scaler.transform(train)["fitness"].to_numpy(dtype=float)
    with_test = scaler.transform(pd.concat([train, test], ignore_index=True))
    assert np.allclose(honest, with_test["fitness"].iloc[: len(train)].to_numpy())
    leaky = FoldFitnessScaler().fit_transform(
        pd.concat([train, test], ignore_index=True)
    )
    assert not np.allclose(honest, leaky["fitness"].iloc[: len(train)].to_numpy())

    mismatch = _frame(train_vals)
    mismatch.loc[0, "mutation_audit"] = "MISMATCH"
    labeled = FoldFitnessScaler().fit_transform(mismatch)
    assert pd.isna(labeled.loc[0, "fitness"])


def test_attach_sum_max_abs_and_missing_confidence() -> None:
    muts = [("Q", 324, "R"), ("A", 355, "R")]
    rows = pd.DataFrame(
        {
            "version": ["V2.4"],
            "mutation_audit": ["match"],
            "mut_from_construct": [muts],
            "mut_from_description": [muts],
            "mutation_codes": [["Q324R", "A355R"]],
        }
    )
    physics = pd.DataFrame(
        {
            "mutation": ["Q324R", "A355R"],
            "version": ["V2.4", "V2.4"],
            "rif_ac": [-1.0, -2.0],
            "rif_prop": [0.0, 0.0],
            "delta_rif_sel": [-1.0, -4.0],
        }
    )
    conf = pd.DataFrame(
        {
            "version": ["V2.4", "V2.4"],
            "canonical_position": [324, 355],
            "structural_confidence": [0.9, 0.2],
        }
    )
    summed = attach_physics_and_confidence(
        rows, physics_table=physics, confidence_table=conf, multi_mutant="sum"
    )
    assert np.isclose(summed.iloc[0]["rif_ac"], -3.0)
    assert np.isclose(summed.iloc[0]["delta_rif_sel"], -5.0)
    assert np.isclose(summed.iloc[0]["structural_confidence"], 0.2)

    absed = attach_physics_and_confidence(
        rows, physics_table=physics, confidence_table=conf, multi_mutant="max_abs"
    )
    assert np.isclose(absed.iloc[0]["rif_ac"], -2.0)
    assert np.isclose(absed.iloc[0]["delta_rif_sel"], -4.0)
    assert np.isclose(absed.iloc[0]["structural_confidence"], 0.2)

    missing = attach_physics_and_confidence(
        rows,
        physics_table=pd.DataFrame(),
        confidence_table=pd.DataFrame(),
    )
    assert pd.isna(missing.iloc[0]["structural_confidence"])
    assert pd.isna(missing.iloc[0]["rif_ac"])


def test_missing_confidence_is_zero_not_one() -> None:
    df = pd.DataFrame(
        {
            "mutation_audit": ["match"] * 4,
            "mut_from_construct": [[("Q", 324, "R")]] * 4,
            "version": ["V2.4"] * 4,
            "construct_id": [f"c{i}" for i in range(4)],
            "fitness": [0.1, 0.3, 0.5, 0.8],
        }
    )
    fb = FeatureBuilder(encoding="mutation_bag", include_physics=True)
    X = fb.fit_transform(df)
    conf = fb.confidence_vector(X)
    assert np.allclose(conf, 0.0)
    assert not np.allclose(conf, 1.0)


def test_binary_physchem_not_zscored_georgiev_z_real() -> None:
    assert is_binary_feature_name("delta_charge")
    assert is_binary_feature_name("geo_s0_polar")
    assert not is_binary_feature_name("delta_hydrophobicity_KD")
    assert not is_binary_feature_name("geo_s0_hydrophobicity_KD_z")
    props = _aa_lookup_with_z()
    assert "hydrophobicity_KD_z" in props["A"]
    assert props["A"]["hydrophobicity_KD_z"] != props["A"]["hydrophobicity_KD"]


def test_prefilter_missing_physics_passes() -> None:
    all_missing = pd.DataFrame(
        {
            "delta_rif_sel": [np.nan, np.nan],
            "structural_confidence": [np.nan, 0.9],
        }
    )
    out = physics_prefilter(all_missing)
    assert (out["prefilter"] == "PASS").all()

    mixed = pd.DataFrame(
        {
            "delta_rif_sel": [-10.0, -1.0, 20.0, np.nan],
            "structural_confidence": [0.9, 0.9, np.nan, 0.9],
        }
    )
    out2 = physics_prefilter(mixed)
    assert out2.iloc[3]["prefilter"] == "PASS"
    assert out2.iloc[2]["prefilter"] != "HARD_FAIL"
