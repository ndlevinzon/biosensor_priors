"""Unit tests for ipSAE, Hamming GP, shrinkage μ₀, conformal σ, Thompson."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biosensor_priors.common.ipsae import (
    calc_d0,
    ipsae_from_directory,
    ipsae_pair,
    ptm_term,
)
from biosensor_priors.stage1_structures.ipsae_compare import (
    summarize_ipsae_across_models,
)
from biosensor_priors.stage3_surrogate.calibration import (
    UncertaintyCalibrator,
    conformal_quantile,
    fit_uncertainty_calibration,
)
from biosensor_priors.stage3_surrogate.construct_intercept import GroupIntercept
from biosensor_priors.stage3_surrogate.kernels import HammingPlusMaternKernel
from biosensor_priors.stage3_surrogate.phenotypes import combine_phenotype_means
from biosensor_priors.stage3_surrogate.physics_mean import PhysicsMeanModel
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.thompson import ThompsonPolicy


def test_ipsae_high_for_confident_interface() -> None:
    pae = np.array(
        [
            [0.0, 0.4, 0.5, 0.8],
            [0.4, 0.0, 0.5, 0.7],
            [0.5, 0.5, 0.0, 0.6],
            [0.8, 0.7, 0.6, 0.0],
        ],
        dtype=float,
    )
    chains = np.array(["A", "A", "A", "B"], dtype=object)
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.5, 0, 0]],
        dtype=float,
    )
    result = ipsae_pair(
        pae, chains, chain_a="A", chain_b="B", coords=coords, pae_cutoff=10.0
    )
    assert result.ipsae > 0.5
    assert 0.0 <= result.ipsae <= 1.0
    noisy = pae.copy()
    noisy[:3, 3] = 40.0
    noisy[3, :3] = 40.0
    weak = ipsae_pair(noisy, chains, chain_a="A", chain_b="B", pae_cutoff=10.0)
    assert weak.ipsae < result.ipsae
    assert calc_d0(10) == 1.0
    assert ptm_term(np.array([0.0]), 1.0)[0] == 1.0


def test_ipsae_from_directory_protein_ligand(tmp_path: Path) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",  # noqa: E501
                "ATOM      2  CA  ALA A   2       1.500   0.000   0.000  1.00 90.00           C",  # noqa: E501
                "HETATM    3  C1  LIG B   1       3.000   0.000   0.000  1.00 80.00           C",  # noqa: E501
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pae = [[0.0, 0.4, 0.5], [0.4, 0.0, 0.6], [0.5, 0.6, 0.0]]
    (tmp_path / "confidences.json").write_text(
        json.dumps({"pae": pae}), encoding="utf-8"
    )
    scored = ipsae_from_directory(tmp_path, protein_chain="A", ligand_chain="B")
    assert scored is not None
    assert scored.ipsae > 0.4


def test_ipsae_cross_model_summary() -> None:
    table = pd.DataFrame(
        {
            "version": ["V2.4", "V2.4", "V2.4"],
            "state": ["AcCoA", "AcCoA", "AcCoA"],
            "method": ["AF3", "Boltz2", "RF3"],
            "ipsae": [0.7, 0.65, 0.4],
        }
    )
    summary = summarize_ipsae_across_models(table)
    assert len(summary) == 1
    assert summary.iloc[0]["n_methods"] == 3
    assert summary.iloc[0]["ipsae_std"] > 0


def test_hamming_kernel_diag_and_identical() -> None:
    k = HammingPlusMaternKernel(n_hamming=3, matern_variance=0.0)
    X = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    K = k(X)
    assert np.allclose(np.diag(K), k.diag(X))
    assert K[0, 1] > K[0, 2]


def test_physics_shrinkage_downweights() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = 0.5 * X[:, 0] + rng.normal(scale=0.3, size=40)
    ridge = PhysicsMeanModel(shrinkage="ridge_cv").fit(X, y)
    horse = PhysicsMeanModel(shrinkage="horseshoe").fit(X, y)
    assert ridge.coefficients_ is not None
    assert horse.coefficients_ is not None
    horse_n = np.linalg.norm(horse.coefficients_[1:])
    ridge_n = np.linalg.norm(ridge.coefficients_[1:])
    assert horse_n <= ridge_n + 0.5


def test_version_intercept_does_not_leak() -> None:
    df = pd.DataFrame({"version": ["V1", "V1", "V2", "V2"]})
    residual = np.array([10.0, 12.0, -4.0, -6.0])
    model = GroupIntercept().fit(df, residual)
    pred = model.predict(pd.DataFrame({"version": ["V1", "V3"]}))
    assert pred[0] == 11.0
    assert pred[1] == 0.0


def test_combine_phenotypes_redistributes() -> None:
    means = {
        "selectivity": np.array([1.0, np.nan]),
        "affinity": np.array([0.0, 0.5]),
        "fc": np.array([np.nan, 0.5]),
        "brightness": np.array([np.nan, np.nan]),
    }
    out = combine_phenotype_means(means)
    assert np.isclose(out[0], 1.0 * 0.40 / 0.65)
    assert np.isfinite(out[1])


def test_conformal_and_lambda_calibration() -> None:
    n = 40
    y = np.linspace(0, 1, n)
    mu = y + 0.05
    cv = pd.DataFrame(
        {
            "y_true": y,
            "fitness_mean": mu,
            "fitness_std": np.full(n, 0.02),
            "sigma_structure": np.zeros(n),
            "sigma_physics": np.zeros(n),
        }
    )
    cal = fit_uncertainty_calibration(cv, target_coverage=0.9)
    assert cal.conformal_quantile >= 1.0
    sig = cal.sigma_calibrated(np.full(n, 0.02))
    assert np.all(sig >= 0.02)
    assert conformal_quantile(np.abs(y - mu) / 0.02, alpha=0.1) > 0


def test_multi_output_and_thompson(stage0_result) -> None:
    master, _ = stage0_result
    df = master[master["fitness"].notna()].copy()
    train = df.iloc[: max(6, len(df) // 2)]
    pool = df.iloc[max(6, len(df) // 2) :].head(8)
    model = FusedSurrogate(
        kind="physics_gp",
        random_state=0,
        encoding="mutation_bag",
        kernel="hamming",
        multi_output=True,
        shrinkage="ridge_cv",
    )
    model.fit(train, train["fitness"].to_numpy(dtype=float))
    pred = model.predict(pool)
    assert len(pred.fitness_mean) == len(pool)
    assert pred.physics_alpha >= 0.0
    model.calibrator_ = UncertaintyCalibrator(conformal_quantile=1.2)
    batch = ThompsonPolicy(random_seed=0, primary="fitness").propose(
        train, pool, model, batch_size=3
    )
    assert 1 <= len(batch) <= 3
    assert "acquisition" in batch.columns
