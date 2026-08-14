"""Insertions/deletions, mutation cost, and exploit/explore proposals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage0_ground_truth.edits import (
    compose_canonical,
    format_edit,
    mutation_cost,
    parse_edit_code,
    parse_mutation_list,
    scaffold_edits,
)
from biosensor_priors.stage3_surrogate.surrogate import SurrogatePrediction
from biosensor_priors.stage4_search.design_space import (
    build_design_library,
    design_space_from_config,
)
from biosensor_priors.stage4_search.proposals import split_exploit_explore


def test_edit_roundtrip_and_costs() -> None:
    assert format_edit("Q", 324, "R") == "Q324R"
    assert format_edit("+", 104, "X") == "ins104"
    assert format_edit("+", 0, "N") == "insNterm"
    assert format_edit("-", 0, "N") == "delNterm"
    assert parse_edit_code("ins104") == ("+", 104, "X")
    assert parse_edit_code("insNterm") == ("+", 0, "N")
    sub = mutation_cost([("Q", 324, "R")])
    ins = mutation_cost([("+", 104, "X")])
    block = mutation_cost([("+", 0, "N")])
    assert ins > sub
    assert block > ins


def test_compose_delnterm_drops_insertion() -> None:
    composed = compose_canonical(["insNterm", "Q324R"], ["delNterm", "A355R"])
    assert "insNterm" not in composed
    assert "delNterm" in composed
    assert "A355R" in composed
    assert "Q324R" in composed


def test_v24_scaffold_encodes_movable_sites() -> None:
    root = REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    mapping = pd.read_pickle(
        constructs / pipeline["constructs"]["residue_mapping_pickle"]
    )
    codes = [format_edit(*e) for e in scaffold_edits(mapping, "V2.4")]
    assert "insNterm" in codes
    assert "Q324R" in codes


def test_design_space_emits_indels() -> None:
    root = REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    versions = pd.read_pickle(
        constructs / pipeline["constructs"]["versions_pickle"]
    )
    mapping = pd.read_pickle(
        constructs / pipeline["constructs"]["residue_mapping_pickle"]
    )
    design = design_space_from_config(
        versions,
        parent_version="V1.0",
        mutable_positions=[324],
        allowed_amino_acids=["R"],
        max_mutations=1,
        residue_mapping=mapping,
        indel_events=[{"code": "ins104", "on_parents": ["V1.0"]}],
        missing_ok=True,
    )
    bags = {"/".join(m) for m in design["mutation_codes"]}
    assert any("324" in b for b in bags)
    assert "ins104" in bags
    assert design["mutation_cost"].max() >= 3.0


def test_design_library_has_two_parents() -> None:
    root = REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    fitness_cfg = load_yaml(root / "configs" / "fitness.yaml")
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    versions = pd.read_pickle(
        constructs / pipeline["constructs"]["versions_pickle"]
    )
    mapping = pd.read_pickle(
        constructs / pipeline["constructs"]["residue_mapping_pickle"]
    )
    slim = dict(fitness_cfg)
    design_cfg = dict(slim.get("design") or {})
    design_cfg["allowed_mutable_positions"] = [324]
    design_cfg["allowed_amino_acids"] = ["R", "A"]
    slim["design"] = design_cfg
    library = build_design_library(versions, mapping, slim, default_parent="V2.4")
    assert set(library["parent_version"].astype(str)) >= {"V1.0", "V2.4"}
    joined = library["mutation_codes"].map(
        lambda codes: "/".join(map(str, codes)) if isinstance(codes, list) else ""
    )
    assert joined.str.contains("ins104|insNterm|delNterm", regex=True).any()


def test_d104_canonical_on_master(stage0_result) -> None:
    master, _ = stage0_result
    hits = master[
        master["construct_id"].astype(str).str.contains("D104", case=False, na=False)
    ]
    assert not hits.empty
    codes = hits.iloc[0]["canonical_edit_codes"]
    assert codes is not None
    assert any(str(c).startswith("ins104") for c in codes)
    parsed = parse_mutation_list(hits.iloc[0])
    assert any(a in {"+", "I"} and p == 104 for a, p, _ in parsed)


class _StubSurrogate:
    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        n = len(df)
        return SurrogatePrediction(
            fitness_mean=df["mu"].to_numpy(dtype=float),
            fitness_std=df["std"].to_numpy(dtype=float),
            physics_mean=np.zeros(n),
            gp_residual_mean=np.zeros(n),
            gp_residual_std=np.zeros(n),
            phenotype_mean={
                "brightness": df["b"].to_numpy(dtype=float),
                "fc_prop": df["p"].to_numpy(dtype=float),
            },
            phenotype_std={
                "brightness": np.full(n, 0.02),
                "fc_prop": np.full(n, 0.02),
            },
        )


def test_exploit_cost_and_explore_uncertainty() -> None:
    observed = pd.DataFrame({"fitness": [0.50], "construct_id": ["obs"]})
    pool = pd.DataFrame(
        {
            "construct_id": ["cheap", "costly", "dim", "uncertain"],
            "mu": [0.70, 0.56, 0.90, 0.40],
            "std": [0.05, 0.05, 0.05, 0.40],
            "b": [0.80, 0.80, 0.10, 0.80],
            "p": [0.80, 0.80, 0.80, 0.80],
            "mutation_cost": [1.0, 4.0, 1.0, 3.0],
        }
    )
    search_cfg = {
        "batch_size": 2,
        "thompson": {
            "constraints": {
                "brightness": {"min": 0.55, "min_prob": 0.50},
                "fc_prop": {"min": 0.50, "min_prob": 0.50},
            }
        },
        "proposals": {"exploit_size": 2, "explore_size": 2},
    }
    fitness_cfg = {"mutation_cost": {"lambda": 0.08}}
    exploit, explore = split_exploit_explore(
        observed,
        pool,
        _StubSurrogate(),
        search_cfg=search_cfg,
        fitness_cfg=fitness_cfg,
    )
    assert list(exploit["construct_id"]) == ["cheap"]
    assert "uncertain" in set(explore["construct_id"].astype(str))
    assert "cheap" not in set(explore["construct_id"].astype(str))
    assert (exploit["proposal_role"] == "exploit").all()
    assert (explore["proposal_role"] == "explore").all()
