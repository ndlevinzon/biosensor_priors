"""Stage 4 search and campaign tests."""

from __future__ import annotations

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage3_surrogate.features import FeatureBuilder
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.adalead import AdaLeadPolicy
from biosensor_priors.stage4_search.bo import BOPolicy
from biosensor_priors.stage4_search.campaign import run_campaign_benchmark
from biosensor_priors.stage4_search.design_space import design_space_from_config
from biosensor_priors.stage4_search.mcmc import MCMCPolicy
from biosensor_priors.stage4_search.prefilter import PrefilterCategory, physics_prefilter
from biosensor_priors.stage4_search.random_search import RandomSearchPolicy
from biosensor_priors.stage4_search.thompson import ThompsonPolicy


def test_design_space_canonical_mapping() -> None:
    root = REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    versions = pd.read_pickle(constructs / pipeline["constructs"]["versions_pickle"])
    mapping = pd.read_pickle(constructs / pipeline["constructs"]["residue_mapping_pickle"])
    parent = pipeline.get("active_design_background", "V2.4")
    design = design_space_from_config(
        versions,
        parent_version=parent,
        mutable_positions=[324, 355],
        allowed_amino_acids=list("AR"),
        max_mutations=1,
        residue_mapping=mapping,
        positions_are_canonical=True,
    )
    assert not design.empty
    assert design["n_mutations"].max() == 1
    assert all("324" in "/".join(m) or "355" in "/".join(m) for m in design["mutations"])


def test_prefilter_categories() -> None:
    df = pd.DataFrame(
        {
            "delta_rif_sel": [-10.0, -1.0, 5.0, 20.0],
            "structural_confidence": [0.9, 0.9, 0.2, 0.9],
        }
    )
    out = physics_prefilter(df, score_direction="more_negative_is_better")
    assert set(out["prefilter"]).issubset({c.value for c in PrefilterCategory})


def test_encodings_onehot_georgiev_hybrid(stage0_result) -> None:
    master, _ = stage0_result
    df = master[master["fitness"].notna()].head(12).copy()
    for enc in ("onehot", "georgiev", "hybrid", "mutation_bag"):
        fb = FeatureBuilder(encoding=enc, include_physics=False)
        X = fb.fit_transform(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] > 0


def test_search_policies_propose(stage0_result) -> None:
    master, _ = stage0_result
    observed = master[master["fitness"].notna()].copy()
    pool = observed.sample(n=min(8, len(observed)), random_state=0).copy()
    train = observed.drop(pool.index)
    surrogate = FusedSurrogate(kind="gp_zero_mean", random_state=0, encoding="hybrid")
    surrogate.fit(train, train["fitness"].to_numpy(dtype=float))

    policies = [
        RandomSearchPolicy(candidate_m=32, random_seed=0),
        AdaLeadPolicy(kappa=0.1),
        MCMCPolicy(n_steps=20, n_chains=2, candidate_m=32, random_seed=0),
        BOPolicy(kappa=1.5),
        ThompsonPolicy(random_seed=0, primary="fitness"),
    ]
    for policy in policies:
        batch = policy.propose(train, pool, surrogate, batch_size=3)
        assert 1 <= len(batch) <= 3
        assert "pred_fitness_mean" in batch.columns


def test_campaign_benchmark_smoke(stage0_result) -> None:
    _ = stage0_result
    result = run_campaign_benchmark(
        n_repeats=2,
        n_rounds=2,
        batch_size=3,
        initial_n=5,
        encoding="hybrid",
    )
    assert not result["summary"].empty
    assert {"Success_ratio", "Final_cumulative_best_mean"}.issubset(result["summary"].columns)
    assert set(result["round_table"]["Algorithm"]) >= {
        "random",
        "adalead",
        "mcmc",
        "bo",
        "thompson",
    }
