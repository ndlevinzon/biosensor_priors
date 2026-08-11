"""Map experimental phenotypes to a preregistered scalar fitness in [0, 1]."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def finite_positive(value: Any) -> bool:
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
    """
    Build conservative, measured-only phenotype scores.

    All returned component values are oriented so **higher is better**.

    Default censoring policy (preregistered in ``configs/fitness.yaml``):

    * Affinity: exact → use; ``<x`` → use x; ``>x`` → omit
    * FC: exact → use; ``>x`` → use x; ``<x`` → omit
    * Selectivity: use positive lower bound of Kd(Prop)/Kd(Ac); omit if ≤0
    * Brightness: measured ordinal
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
    return df


def percentile_score(series: pd.Series) -> pd.Series:
    """Convert measured values to [0, 1] empirical percentile ranks (mean ranks)."""
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
    """
    Weighted measured-only fitness in [0, 1].

    Missing phenotype components are **not** imputed. Their weights are
    redistributed across available components when
    ``missing_phenotype: redistribute_weights``.
    """
    weight_map = weights or {
        "selectivity": 0.40,
        "affinity": 0.25,
        "fc": 0.20,
        "brightness": 0.15,
    }
    expected = set(weight_map)
    if expected != {"selectivity", "affinity", "fc", "brightness"}:
        raise ValueError(
            "Fitness weights must define selectivity, affinity, fc, and brightness "
            f"(got {sorted(weight_map)})."
        )
    if not np.isclose(sum(weight_map.values()), 1.0):
        raise ValueError(
            f"Fitness weights must sum to 1.0 (got {sum(weight_map.values()):.6f})."
        )

    df = measured_component_values(clean, policies=policies)

    raw_cols = {
        "selectivity": "_fitness_selectivity_raw",
        "affinity": "_fitness_affinity_raw",
        "fc": "_fitness_fc_raw",
        "brightness": "_fitness_brightness_raw",
    }
    score_cols: dict[str, str] = {}
    for name, col in raw_cols.items():
        score_col = f"_fitness_{name}_score"
        df[score_col] = percentile_score(df[col])
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

    return df
