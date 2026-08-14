"""Fused physics-informed GP surrogate with prediction decomposition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.calibration import UncertaintyCalibrator
from biosensor_priors.stage3_surrogate.confidence_weighting import (
    apply_confidence_weighting,
)
from biosensor_priors.stage3_surrogate.construct_intercept import GroupIntercept
from biosensor_priors.stage3_surrogate.features import (
    PHYSICS_FEATURE_COLUMNS,
    FeatureBuilder,
)
from biosensor_priors.stage3_surrogate.gp_residual import GPResidualModel, KernelKind
from biosensor_priors.stage3_surrogate.phenotypes import (
    DEFAULT_WEIGHTS,
    PHENOTYPES,
    combine_phenotype_means,
    combine_phenotype_std,
    labeled_mask,
    minmax_from_train,
    phenotype_score_matrix,
    phenotype_weights,
)
from biosensor_priors.stage3_surrogate.physics_mean import (
    PhysicsMeanModel,
    ShrinkageKind,
)

ModelKind = Literal["physics_only", "gp_zero_mean", "physics_gp"]


@dataclass
class _Head:
    """One physics + intercept + GP stack for a scalar target."""

    physics: PhysicsMeanModel
    intercept: GroupIntercept
    gp: GPResidualModel | None
    alpha: float
    n_train: int


@dataclass
class SurrogatePrediction:
    fitness_mean: np.ndarray
    fitness_std: np.ndarray
    physics_mean: np.ndarray
    gp_residual_mean: np.ndarray
    gp_residual_std: np.ndarray
    construct_ids: list[str] = field(default_factory=list)
    phenotype_mean: dict[str, np.ndarray] = field(default_factory=dict)
    phenotype_std: dict[str, np.ndarray] = field(default_factory=dict)
    version_intercept: np.ndarray | None = None
    physics_alpha: float = 1.0


@dataclass
class FusedSurrogate:
    """
    Configurable surrogate:

    * ``physics_only`` — μ₀ only
    * ``gp_zero_mean`` — GP on y (μ₀ = 0)
    * ``physics_gp`` — α μ₀ + version intercept + GP(residual)

    Physics features are the mean only (not extra kernel dimensions).
    The residual kernel defaults to Hamming on mutation sets plus a small
    physicochemical Matérn. Optional multi-output heads model S, A, FC, B
    and combine with preregistered weights.
    """

    kind: ModelKind = "physics_gp"
    use_confidence_weighting: bool = True
    random_state: int = 42
    encoding: str = "mutation_bag"
    kernel: KernelKind = "hamming"
    shrinkage: ShrinkageKind = "ridge_cv"
    physics_in_gp: bool = False
    fit_physics_alpha: bool = True
    version_intercept: bool = True
    group_col: str = "version"
    multi_output: bool = True
    phenotype_weights: dict[str, float] | None = None
    feature_builder: FeatureBuilder | None = None
    physics_model: PhysicsMeanModel = field(default_factory=PhysicsMeanModel)
    gp_model: GPResidualModel | None = None
    fitted_: bool = False
    calibrator_: UncertaintyCalibrator | None = None
    scalar_head_: _Head | None = None
    phenotype_heads_: dict[str, _Head] = field(default_factory=dict)
    train_raw_lo_: float = 0.0
    train_raw_hi_: float = 1.0
    physics_alpha_: float = 1.0

    def __post_init__(self) -> None:
        """Initialize default feature builder when none is provided."""
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(encoding=self.encoding)  # type: ignore[arg-type]
        self.physics_model.shrinkage = self.shrinkage

    def _prepare_features(
        self, df: pd.DataFrame, *, fit: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        assert self.feature_builder is not None
        if fit:
            X = self.feature_builder.fit_transform(df)
        else:
            X = self.feature_builder.transform(df)
        conf = self.feature_builder.confidence_vector(X)
        X_raw, X_w = apply_confidence_weighting(
            X, self.feature_builder.feature_names_, conf
        )
        X_used = X_w if self.use_confidence_weighting else X_raw
        X_physics = self.feature_builder.physics_block(X_used)
        if self.physics_in_gp:
            X_gp, n_h = X_used, 0
        else:
            X_gp, n_h = self.feature_builder.gp_block(X_used)
        return X_used, X_physics, X_gp, n_h

    def _zero_physics(self) -> PhysicsMeanModel:
        model = PhysicsMeanModel(shrinkage=self.shrinkage)
        model.mode_ = "intercept"
        model.intercept_ = 0.0
        model.n_features_ = 0
        model.coefficients_ = np.zeros(0)
        return model

    def _fit_alpha(self, mu0: np.ndarray, y: np.ndarray) -> float:
        if not self.fit_physics_alpha or self.kind == "gp_zero_mean":
            return 0.0 if self.kind == "gp_zero_mean" else 1.0
        finite = np.isfinite(mu0) & np.isfinite(y)
        if int(finite.sum()) < 2:
            return 1.0
        num = float(np.dot(mu0[finite], y[finite]))
        den = float(np.dot(mu0[finite], mu0[finite]))
        if den < 1e-12:
            return 1.0
        return float(np.clip(num / den, 0.0, 1.0))

    def _fit_head(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        X_physics: np.ndarray,
        X_gp: np.ndarray,
        n_hamming: int,
    ) -> _Head:
        y = np.asarray(y, dtype=float)
        if self.kind == "gp_zero_mean":
            physics = self._zero_physics()
            mu0 = np.zeros(len(y), dtype=float)
            alpha = 0.0
        else:
            physics = PhysicsMeanModel(shrinkage=self.shrinkage)
            physics.fit(X_physics, y)
            mu0 = physics.predict(X_physics)
            alpha = self._fit_alpha(mu0, y)
        residual = y - alpha * mu0
        intercept = GroupIntercept(column=self.group_col)
        if self.version_intercept:
            intercept.fit(df, residual)
            residual = residual - intercept.predict(df)
        gp: GPResidualModel | None = None
        if self.kind in {"gp_zero_mean", "physics_gp"}:
            gp = GPResidualModel(
                random_state=self.random_state,
                kernel=self.kernel,
                n_hamming=n_hamming,
            )
            gp.fit(X_gp, residual)
        return _Head(
            physics=physics, intercept=intercept, gp=gp, alpha=alpha, n_train=len(y)
        )

    def _predict_head(
        self,
        head: _Head,
        df: pd.DataFrame,
        X_physics: np.ndarray,
        X_gp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu0 = head.physics.predict(X_physics)
        if self.version_intercept:
            inter = head.intercept.predict(df)
        else:
            inter = np.zeros(len(df))
        if head.gp is None:
            gp_mean = np.zeros(len(df), dtype=float)
            gp_std = np.zeros(len(df), dtype=float)
        else:
            gp_mean, gp_std = head.gp.predict(X_gp)
        phys = head.alpha * mu0
        mean = phys + inter + gp_mean
        return mean, gp_std, phys, gp_mean, inter

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> FusedSurrogate:
        """Fit physics mean and GP residual models on training data.

        Parameters
        ----------
        df : pandas.DataFrame
            Training construct table with features and metadata.
        y : numpy.ndarray
            Observed fitness values aligned with ``df`` rows.

        Returns
        -------
        FusedSurrogate
            Fitted surrogate (``self``).
        """
        y = np.asarray(y, dtype=float)
        _, X_physics, X_gp, n_h = self._prepare_features(df, fit=True)
        self.scalar_head_ = self._fit_head(df, y, X_physics, X_gp, n_h)
        self.physics_model = self.scalar_head_.physics
        self.gp_model = self.scalar_head_.gp
        self.physics_alpha_ = self.scalar_head_.alpha
        if "Fitness_raw_weighted" in df.columns:
            raw = pd.to_numeric(df["Fitness_raw_weighted"], errors="coerce")
            finite = raw[raw.notna()]
            if len(finite) >= 2:
                self.train_raw_lo_ = float(finite.min())
                self.train_raw_hi_ = float(finite.max())
            else:
                self.train_raw_lo_, self.train_raw_hi_ = 0.0, 1.0
        else:
            finite_y = y[np.isfinite(y)]
            if len(finite_y) >= 2:
                self.train_raw_lo_ = float(finite_y.min())
                self.train_raw_hi_ = float(finite_y.max())
            else:
                self.train_raw_lo_, self.train_raw_hi_ = 0.0, 1.0

        self.phenotype_heads_ = {}
        if self.multi_output:
            scores = phenotype_score_matrix(df)
            for name in PHENOTYPES:
                mask = labeled_mask(scores[name], min_n=3)
                if mask is None:
                    continue
                sub = df.iloc[np.flatnonzero(mask)].reset_index(drop=True)
                self.phenotype_heads_[name] = self._fit_head(
                    sub,
                    scores[name][mask],
                    X_physics[mask],
                    X_gp[mask],
                    n_h,
                )
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict fitness with decomposed physics and GP components.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table to predict.

        Returns
        -------
        SurrogatePrediction
            Dataclass with mean/std predictions and component breakdown.

        Raises
        ------
        RuntimeError
            When called before :meth:`fit`.
        """
        if not self.fitted_ or self.scalar_head_ is None:
            raise RuntimeError("Surrogate must be fit before predict.")
        ids = (
            df["construct_id"].astype(str).tolist()
            if "construct_id" in df.columns
            else [str(i) for i in range(len(df))]
        )
        _, X_physics, X_gp, _ = self._prepare_features(df, fit=False)
        _mean, _std, phys, gp_mean, inter = self._predict_head(
            self.scalar_head_, df, X_physics, X_gp
        )
        gp_std = (
            self.scalar_head_.gp.predict(X_gp)[1]
            if self.scalar_head_.gp is not None
            else np.zeros(len(df), dtype=float)
        )
        phys_out = phys + inter
        mean = phys_out + gp_mean
        std = gp_std
        pheno_mu: dict[str, np.ndarray] = {}
        pheno_std: dict[str, np.ndarray] = {}
        if self.phenotype_heads_:
            for name, head in self.phenotype_heads_.items():
                p_mean, p_std, _, _, _ = self._predict_head(
                    head, df, X_physics, X_gp
                )
                pheno_mu[name] = p_mean
                pheno_std[name] = p_std
            weights = phenotype_weights(self.phenotype_weights)
            combined = combine_phenotype_means(pheno_mu, weights=weights)
            combined_std = combine_phenotype_std(pheno_std, weights=weights)
            mean = minmax_from_train(
                combined, lo=self.train_raw_lo_, hi=self.train_raw_hi_
            )
            span = max(self.train_raw_hi_ - self.train_raw_lo_, 1e-8)
            std = combined_std / span
            phys_out = mean - gp_mean
        elif self.kind == "physics_only":
            mean = phys_out
            std = np.zeros(len(df), dtype=float)
            gp_mean = np.zeros(len(df), dtype=float)
        elif self.kind == "gp_zero_mean":
            phys_out = inter
            mean = inter + gp_mean
        if self.kind == "physics_only":
            std = np.zeros(len(df), dtype=float)

        return SurrogatePrediction(
            fitness_mean=np.asarray(mean, dtype=float),
            fitness_std=np.asarray(std, dtype=float),
            physics_mean=np.asarray(phys_out, dtype=float),
            gp_residual_mean=np.asarray(gp_mean, dtype=float),
            gp_residual_std=np.asarray(gp_std, dtype=float),
            construct_ids=ids,
            phenotype_mean=pheno_mu,
            phenotype_std=pheno_std,
            version_intercept=inter,
            physics_alpha=self.physics_alpha_,
        )

    def sample_fitness(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Draw one posterior sample of fitness and phenotype heads.

        Parameters
        ----------
        df : pandas.DataFrame
            Candidate table.
        rng : numpy.random.Generator
            RNG for Thompson sampling.

        Returns
        -------
        fitness_sample : numpy.ndarray
            One draw of combined / scalar fitness.
        phenotype_samples : dict
            Per-phenotype draws when multi-output heads exist.
        """
        pred = self.predict(df)
        pheno_s: dict[str, np.ndarray] = {}
        std = pred.fitness_std
        if self.calibrator_ is not None:
            from biosensor_priors.stage3_surrogate.calibration import (
                structural_and_physics_sigma,
            )

            ss, sp = structural_and_physics_sigma(df)
            std = self.calibrator_.sigma_calibrated(std, ss, sp)
        if pred.phenotype_mean:
            for name, mu in pred.phenotype_mean.items():
                sig = pred.phenotype_std.get(name, np.full(len(mu), 1e-6))
                pheno_s[name] = rng.normal(mu, np.maximum(sig, 1e-8))
            weights = phenotype_weights(self.phenotype_weights)
            combined = combine_phenotype_means(pheno_s, weights=weights)
            fitness = minmax_from_train(
                combined, lo=self.train_raw_lo_, hi=self.train_raw_hi_
            )
            return fitness, pheno_s
        return rng.normal(pred.fitness_mean, np.maximum(std, 1e-8)), pheno_s

    def metadata(self) -> dict[str, Any]:
        """Return fitted model metadata for provenance and inspection."""
        assert self.feature_builder is not None
        return {
            "kind": self.kind,
            "encoding": self.encoding,
            "kernel": self.kernel,
            "shrinkage": self.shrinkage,
            "use_confidence_weighting": self.use_confidence_weighting,
            "physics_in_gp": self.physics_in_gp,
            "fit_physics_alpha": self.fit_physics_alpha,
            "physics_alpha": self.physics_alpha_,
            "version_intercept": self.version_intercept,
            "multi_output": self.multi_output,
            "phenotype_heads": sorted(self.phenotype_heads_),
            "physics_weights": self.physics_model.as_weight_dict(
                list(PHYSICS_FEATURE_COLUMNS)
            ),
            "n_features": len(self.feature_builder.feature_names_),
            "feature_names": list(self.feature_builder.feature_names_),
            "has_physics_features": self.feature_builder.has_physics_,
            "mutation_vocab_size": len(self.feature_builder.mutation_vocab),
            "calibrator": (
                None if self.calibrator_ is None else self.calibrator_.as_dict()
            ),
        }


def surrogate_kwargs_from_cfg(
    gp_cfg: Mapping[str, Any] | None = None,
    fitness_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build FusedSurrogate kwargs from thresholds.yaml / fitness.yaml."""
    gp_cfg = dict(gp_cfg or {})
    fitness_cfg = dict(fitness_cfg or {})
    weights = fitness_cfg.get("weights") or DEFAULT_WEIGHTS
    return {
        "encoding": str(gp_cfg.get("encoding", "mutation_bag")),
        "kernel": str(gp_cfg.get("kernel", "hamming")),
        "shrinkage": str(gp_cfg.get("shrinkage", "ridge_cv")),
        "physics_in_gp": bool(gp_cfg.get("physics_in_gp", False)),
        "fit_physics_alpha": bool(gp_cfg.get("fit_physics_alpha", True)),
        "version_intercept": bool(gp_cfg.get("version_intercept", True)),
        "group_col": str(gp_cfg.get("group_col", "version")),
        "multi_output": bool(gp_cfg.get("multi_output", True)),
        "phenotype_weights": dict(weights),
    }
