"""Map experimental phenotypes to a preregistered scalar fitness in [0, 1]."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PHENOTYPES = ("selectivity", "affinity", "fc", "brightness", "fc_prop")
DEFAULT_WEIGHTS: dict[str, float] = {
    "selectivity": 0.20,
    "affinity": 0.20,
    "fc": 0.15,
    "brightness": 0.25,
    "fc_prop": 0.20,
}


def finite_positive(value: Any) -> bool:
    """Return whether a value is a finite positive number.

    Parameters
    ----------
    value : Any
        Value to test, coerced to ``float`` when possible.

    Returns
    -------
    bool
        ``True`` when ``value`` is finite and strictly greater than zero.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > 0)


def measured_component_values(
    clean: pd.DataFrame,
    *,
    policies: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build conservative, measured-only phenotype component scores.

    All returned component values are oriented so **higher is better**.

    Default censoring policy (preregistered in ``configs/fitness.yaml``):

    * Affinity: exact → use; ``<x`` → use x; ``>x`` → omit
    * FC: exact → use; ``>x`` → use x; ``<x`` → omit
    * Selectivity: use positive lower bound of Kd(Prop)/Kd(Ac); omit if ≤0
    * Brightness: measured ordinal

    Parameters
    ----------
    clean : pandas.DataFrame
        Cleaned experimental table with parsed phenotype columns.
    policies : dict[str, Any] | None, optional
        Reserved for forward-compatible censoring policy overrides.
        Currently unused.

    Returns
    -------
    pandas.DataFrame
        Copy of ``clean`` with ``_fitness_*_raw`` columns for affinity, FC,
        selectivity, and brightness components.
    """
    _ = policies  # explicit for forward-compatible policy branching
    df = clean.copy()

    affinity_score: list[float] = []
    for _, row in df.iterrows():
        value = row.get("Affinity AcCoA__uM")
        censor = row.get("Affinity AcCoA__censor_direction")
        if finite_positive(value) and censor != "above":
            affinity_score.append(-np.log10(float(value)))
        else:
            affinity_score.append(np.nan)
    df["_fitness_affinity_raw"] = affinity_score

    fc_score: list[float] = []
    for _, row in df.iterrows():
        value = row.get("FC AcCoA__value")
        censor = row.get("FC AcCoA__censor_direction")
        if finite_positive(value) and censor != "below":
            fc_score.append(np.log10(float(value)))
        else:
            fc_score.append(np.nan)
    df["_fitness_fc_raw"] = fc_score

    selectivity_score: list[float] = []
    for _, row in df.iterrows():
        lo = row.get("Selectivity_Kd_Prop_over_Ac__lower")
        if finite_positive(lo):
            selectivity_score.append(np.log10(float(lo)))
        else:
            selectivity_score.append(np.nan)
    df["_fitness_selectivity_raw"] = selectivity_score

    df["_fitness_brightness_raw"] = pd.to_numeric(
        df.get("Brightness__ordinal", np.nan),
        errors="coerce",
    )

    # Off-target FC PropCoA: higher is better = less PropCoA response.
    fc_prop: list[float] = []
    has_fc_prop = "FC PropCoA__value" in df.columns
    for _, row in df.iterrows():
        if not has_fc_prop:
            fc_prop.append(np.nan)
            continue
        value = row.get("FC PropCoA__value")
        censor = row.get("FC PropCoA__censor_direction")
        if finite_positive(value) and censor != "below":
            fc_prop.append(-np.log10(float(value)))
        else:
            fc_prop.append(np.nan)
    df["_fitness_fc_prop_raw"] = fc_prop
    return df


def percentile_score(series: pd.Series) -> pd.Series:
    """Convert measured values to [0, 1] empirical percentile ranks.

    Uses mean ranks for ties. A single valid observation receives score 0.5.

    Parameters
    ----------
    series : pandas.Series
        Raw component values, possibly containing NaN.

    Returns
    -------
    pandas.Series
        Percentile scores in [0, 1] for non-missing entries; NaN elsewhere.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(valid.sum())
    if n == 1:
        out.loc[valid] = 0.5
        return out
    if n >= 2:
        ranks = s.loc[valid].rank(method="average")
        out.loc[valid] = (ranks - 1) / (n - 1)
    return out


def fitness_transform(
    clean: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    min_components: int = 2,
    policies: dict[str, Any] | None = None,
    require_range: bool = True,
) -> pd.DataFrame:
    """Compute weighted measured-only fitness in [0, 1].

    Missing phenotype components are **not** imputed. Their weights are
    redistributed across available components when
    ``missing_phenotype: redistribute_weights``.

    Parameters
    ----------
    clean : pandas.DataFrame
        Cleaned experimental table with parsed phenotype columns.
    weights : dict[str, float] | None, optional
        Component weights for ``selectivity``, ``affinity``, ``fc``,
        ``brightness``, and ``fc_prop``. Must sum to 1.0. Defaults to
        preregistered weights.
    min_components : int, optional
        Minimum number of measured components required to assign fitness.
        Default is 2.
    policies : dict[str, Any] | None, optional
        Censoring policies forwarded to :func:`measured_component_values`.
    require_range : bool, optional
        When ``True``, require at least five constructs with fitness and a
        non-degenerate raw score range. Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``clean`` with fitness component scores, metadata columns,
        and final ``fitness`` in [0, 1].

    Raises
    ------
    ValueError
        If weights are incomplete, do not sum to 1.0, or if
        ``require_range`` checks fail.
    """
    weight_map = weights or dict(DEFAULT_WEIGHTS)
    expected = set(weight_map)
    if expected != set(PHENOTYPES):
        raise ValueError(
            "Fitness weights must define selectivity, affinity, fc, "
            f"brightness, and fc_prop (got {sorted(weight_map)})."
        )
    if not np.isclose(sum(weight_map.values()), 1.0):
        raise ValueError(
            f"Fitness weights must sum to 1.0 (got {sum(weight_map.values()):.6f})."
        )

    df = measured_component_values(clean, policies=policies)

    mismatch = _mismatch_mask(df)
    raw_cols = {
        "selectivity": "_fitness_selectivity_raw",
        "affinity": "_fitness_affinity_raw",
        "fc": "_fitness_fc_raw",
        "brightness": "_fitness_brightness_raw",
        "fc_prop": "_fitness_fc_prop_raw",
    }
    score_cols: dict[str, str] = {}
    for name, col in raw_cols.items():
        score_col = f"_fitness_{name}_score"
        series = pd.to_numeric(df[col], errors="coerce").copy()
        series.loc[mismatch] = np.nan
        df[score_col] = percentile_score(series)
        score_cols[name] = score_col

    fitness: list[float] = []
    n_components: list[int] = []
    effective_weight: list[float] = []

    for _, row in df.iterrows():
        numerator = 0.0
        denominator = 0.0
        count = 0
        for phenotype, weight in weight_map.items():
            value = row[score_cols[phenotype]]
            if pd.notna(value):
                numerator += weight * float(value)
                denominator += weight
                count += 1
        n_components.append(count)
        effective_weight.append(denominator)
        if count >= min_components and denominator > 0:
            fitness.append(numerator / denominator)
        else:
            fitness.append(np.nan)

    df["Fitness_components"] = n_components
    df["Fitness_available_weight"] = effective_weight
    df["Fitness_raw_weighted"] = fitness
    df["Fitness_weight_selectivity"] = weight_map["selectivity"]
    df["Fitness_weight_affinity"] = weight_map["affinity"]
    df["Fitness_weight_fc"] = weight_map["fc"]
    df["Fitness_weight_brightness"] = weight_map["brightness"]
    df["Fitness_weight_fc_prop"] = weight_map["fc_prop"]

    valid = df["Fitness_raw_weighted"].notna()
    if require_range and int(valid.sum()) < 5:
        raise ValueError(
            "Too few constructs have sufficient measured phenotypes to define fitness."
        )

    if valid.any():
        lo = float(df.loc[valid, "Fitness_raw_weighted"].min())
        hi = float(df.loc[valid, "Fitness_raw_weighted"].max())
        if require_range and (
            not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi)
        ):
            raise ValueError("Measured scalar fitness has no usable range.")
        df["fitness"] = (df["Fitness_raw_weighted"] - lo) / (hi - lo)
    else:
        df["fitness"] = np.nan

    df.loc[mismatch, "fitness"] = np.nan
    df.loc[mismatch, "Fitness_raw_weighted"] = np.nan
    return df


def _mismatch_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose mutation identity failed Construct vs Description audit."""
    if "mutation_audit" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["mutation_audit"].astype(str) == "MISMATCH"


def percentile_against_reference(
    values: pd.Series | np.ndarray,
    reference: pd.Series | np.ndarray,
) -> np.ndarray:
    """Map values onto the empirical percentile scale of a train-only reference.

    Uses the same average-rank convention as :func:`percentile_score`. Scores
    outside the train range are clipped to ``[0, 1]``.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    out = np.full(len(vals), np.nan, dtype=float)
    n = int(len(ref))
    if n == 0:
        return out
    if n == 1:
        out[np.isfinite(vals)] = 0.5
        return out
    for i, x in enumerate(vals):
        if not np.isfinite(x):
            continue
        n_lt = float(np.sum(ref < x))
        n_eq = float(np.sum(ref == x))
        rank = n_lt + (n_eq + 1.0) / 2.0
        out[i] = float(np.clip((rank - 1.0) / (n - 1.0), 0.0, 1.0))
    return out


class FoldFitnessScaler:
    """Train-only phenotype percentiles and fitness minmax for CV / Stage 3-4.

    Stage 0 may still write a global catalog ``fitness`` column. Modeling must
    call :meth:`fit` on the training fold and :meth:`transform` on train and
    test so held-out raw values never enter the label scale.
    """

    FITNESS_PHENOTYPES = PHENOTYPES
    AUX_PHENOTYPES: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        min_components: int = 2,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.min_components = int(min_components)
        self.reference_: dict[str, np.ndarray] = {}
        self.raw_lo_: float = 0.0
        self.raw_hi_: float = 1.0
        self.fitted_: bool = False

    def _raw_col(self, name: str) -> str:
        return f"_fitness_{name}_raw"

    def _score_col(self, name: str) -> str:
        return f"_fitness_{name}_score"

    def fit(self, df: pd.DataFrame) -> FoldFitnessScaler:
        """Store train raw references and combined min/max."""
        work = df
        if "_fitness_fc_raw" not in work.columns:
            work = measured_component_values(work)
        mismatch = _mismatch_mask(work)
        tmp = work.copy()
        for name in (*self.FITNESS_PHENOTYPES, *self.AUX_PHENOTYPES):
            col = self._raw_col(name)
            if col not in tmp.columns:
                self.reference_[name] = np.zeros(0, dtype=float)
                continue
            series = pd.to_numeric(tmp[col], errors="coerce").copy()
            series.loc[mismatch] = np.nan
            vals = series.to_numpy(dtype=float)
            self.reference_[name] = vals[np.isfinite(vals)]
            tmp[self._score_col(name)] = percentile_score(series)
        scored = self._combine(tmp, mismatch)
        finite = scored[np.isfinite(scored)]
        if len(finite) >= 2:
            self.raw_lo_ = float(np.min(finite))
            self.raw_hi_ = float(np.max(finite))
        else:
            self.raw_lo_, self.raw_hi_ = 0.0, 1.0
        self.fitted_ = True
        return self

    def _combine(self, df: pd.DataFrame, mismatch: pd.Series) -> np.ndarray:
        n = len(df)
        combined = np.full(n, np.nan, dtype=float)
        for i in range(n):
            if bool(mismatch.iloc[i]):
                continue
            numer = 0.0
            denom = 0.0
            count = 0
            for name in self.FITNESS_PHENOTYPES:
                if name not in self.weights:
                    continue
                col = self._score_col(name)
                if col not in df.columns:
                    continue
                value = df.iloc[i][col]
                if pd.notna(value):
                    numer += self.weights[name] * float(value)
                    denom += self.weights[name]
                    count += 1
            if count >= self.min_components and denom > 0:
                combined[i] = numer / denom
        return combined

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write fold scores, ``Fitness_raw_weighted``, and ``fitness``."""
        if not self.fitted_:
            raise RuntimeError("FoldFitnessScaler must be fit before transform.")
        out = df.copy()
        if "_fitness_fc_raw" not in out.columns:
            out = measured_component_values(out)
        mismatch = _mismatch_mask(out)
        for name in (*self.FITNESS_PHENOTYPES, *self.AUX_PHENOTYPES):
            raw_col = self._raw_col(name)
            if raw_col not in out.columns:
                continue
            series = pd.to_numeric(out[raw_col], errors="coerce").copy()
            series.loc[mismatch] = np.nan
            out[self._score_col(name)] = percentile_against_reference(
                series, self.reference_.get(name, np.zeros(0))
            )
        combined = self._combine(out, mismatch)
        out["Fitness_raw_weighted"] = combined
        span = self.raw_hi_ - self.raw_lo_
        if not np.isfinite(span) or abs(span) < 1e-12:
            out["fitness"] = combined
        else:
            fitness = (combined - self.raw_lo_) / span
            out["fitness"] = fitness
        out.loc[mismatch, ["fitness", "Fitness_raw_weighted"]] = np.nan
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return labeled copy."""
        return self.fit(df).transform(df)
