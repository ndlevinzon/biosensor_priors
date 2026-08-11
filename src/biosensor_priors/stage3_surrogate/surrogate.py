"""Fused physics-informed GP surrogate with prediction decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from biosensor_priors.stage3_surrogate.confidence_weighting import apply_confidence_weighting
from biosensor_priors.stage3_surrogate.features import PHYSICS_FEATURE_COLUMNS, FeatureBuilder
from biosensor_priors.stage3_surrogate.gp_residual import GPResidualModel
from biosensor_priors.stage3_surrogate.physics_mean import PhysicsMeanModel

ModelKind = Literal["physics_only", "gp_zero_mean", "physics_gp"]


@dataclass
class SurrogatePrediction:
    fitness_mean: np.ndarray
    fitness_std: np.ndarray
    physics_mean: np.ndarray
    gp_residual_mean: np.ndarray
    gp_residual_std: np.ndarray
    construct_ids: list[str] = field(default_factory=list)


@dataclass
class FusedSurrogate:
    """
    Configurable surrogate:

    * ``physics_only`` — μ₀ only
    * ``gp_zero_mean`` — GP on y (μ₀ = 0)
    * ``physics_gp`` — μ₀ + GP(residual)
    """

    kind: ModelKind = "physics_gp"
    use_confidence_weighting: bool = True
    random_state: int = 42
    encoding: str = "hybrid"
    feature_builder: FeatureBuilder | None = None
    physics_model: PhysicsMeanModel = field(default_factory=PhysicsMeanModel)
    gp_model: GPResidualModel | None = None
    fitted_: bool = False

    def __post_init__(self) -> None:
        """Initialize default feature builder when none is provided.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(encoding=self.encoding)  # type: ignore[arg-type]

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
        assert self.feature_builder is not None
        y = np.asarray(y, dtype=float)
        X = self.feature_builder.fit_transform(df)
        conf = self.feature_builder.confidence_vector(X)
        X_raw, X_w = apply_confidence_weighting(
            X, self.feature_builder.feature_names_, conf
        )
        X_used = X_w if self.use_confidence_weighting else X_raw
        X_physics = self.feature_builder.physics_block(X_used)

        if self.kind == "gp_zero_mean":
            self.physics_model = PhysicsMeanModel()
            self.physics_model.mode_ = "intercept"
            self.physics_model.intercept_ = 0.0
            self.physics_model.n_features_ = 0
            self.physics_model.coefficients_ = np.zeros(0)
            residual = y.copy()
        else:
            self.physics_model.fit(X_physics, y)
            mu0 = self.physics_model.predict(X_physics)
            residual = y - mu0

        if self.kind in {"gp_zero_mean", "physics_gp"}:
            self.gp_model = GPResidualModel(random_state=self.random_state)
            self.gp_model.fit(X_used, residual)
        else:
            self.gp_model = None

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
        if not self.fitted_:
            raise RuntimeError("Surrogate must be fit before predict.")
        assert self.feature_builder is not None
        ids = df["construct_id"].astype(str).tolist() if "construct_id" in df.columns else [
            str(i) for i in range(len(df))
        ]
        X = self.feature_builder.transform(df)
        conf = self.feature_builder.confidence_vector(X)
        X_raw, X_w = apply_confidence_weighting(
            X, self.feature_builder.feature_names_, conf
        )
        X_used = X_w if self.use_confidence_weighting else X_raw
        X_physics = self.feature_builder.physics_block(X_used)
        mu0 = self.physics_model.predict(X_physics)

        if self.gp_model is None:
            residual_mean = np.zeros(len(df), dtype=float)
            residual_std = np.zeros(len(df), dtype=float)
        else:
            residual_mean, residual_std = self.gp_model.predict(X_used)

        if self.kind == "physics_only":
            fitness_mean = mu0
            fitness_std = np.zeros(len(df), dtype=float)
        elif self.kind == "gp_zero_mean":
            fitness_mean = residual_mean
            fitness_std = residual_std
            mu0 = np.zeros(len(df), dtype=float)
        else:
            fitness_mean = mu0 + residual_mean
            fitness_std = residual_std

        return SurrogatePrediction(
            fitness_mean=fitness_mean,
            fitness_std=fitness_std,
            physics_mean=mu0,
            gp_residual_mean=residual_mean,
            gp_residual_std=residual_std,
            construct_ids=ids,
        )

    def metadata(self) -> dict[str, Any]:
        """Return fitted model metadata for provenance and inspection.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Model kind, encoding, physics weights, and feature summary.
        """
        assert self.feature_builder is not None
        return {
            "kind": self.kind,
            "encoding": self.encoding,
            "use_confidence_weighting": self.use_confidence_weighting,
            "physics_weights": self.physics_model.as_weight_dict(
                list(PHYSICS_FEATURE_COLUMNS)
            ),
            "n_features": len(self.feature_builder.feature_names_),
            "feature_names": list(self.feature_builder.feature_names_),
            "has_physics_features": self.feature_builder.has_physics_,
            "mutation_vocab_size": len(self.feature_builder.mutation_vocab),
        }
