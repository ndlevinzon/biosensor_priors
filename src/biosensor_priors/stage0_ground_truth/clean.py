"""Clean and normalize the experimental mutant workbook."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biosensor_priors.common.identifiers import (
    extract_mutations,
    make_construct_id,
    mutation_codes,
)
from biosensor_priors.stage0_ground_truth.parsing import (
    BRIGHTNESS_SCALE,
    apply_numeric_parse,
    format_ratio_bound,
    map_to_ordinal,
    normalize_improvement,
    normalize_to_uM,
    ratio_bounds,
)


def mutation_audit_status(row: pd.Series) -> str:
    """Compare mutations parsed from Construct vs Description columns.

    Parameters
    ----------
    row : pandas.Series
        Row containing ``mut_from_construct`` and ``mut_from_description``.

    Returns
    -------
    str
        Audit status: ``"match"``, ``"MISMATCH"``, ``"construct_only"``,
        ``"description_only"``, or ``"no_mutation_found"``.
    """
    c = row["mut_from_construct"]
    d = row["mut_from_description"]
    if c and d:
        return "match" if set(c) == set(d) else "MISMATCH"
    if c and not d:
        return "construct_only"
    if d and not c:
        return "description_only"
    return "no_mutation_found"


def prepare_database(
    df: pd.DataFrame,
    *,
    assume_unitless_affinity_um: bool = False,
) -> pd.DataFrame:
    """Normalize phenotypes, mutations, and selectivity intervals.

    Parses numeric fields, converts affinities to µM, maps brightness ordinals,
    audits mutation consistency, and derives selectivity ratio bounds.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw experimental workbook loaded as a DataFrame.
    assume_unitless_affinity_um : bool, optional
        When ``True``, treat unitless affinity values as micromolar.
        Default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Cleaned table with normalized columns, mutation audit fields, and
        ``construct_id``.
    """
    df = df.copy()

    numeric_cols = [
        "FC AcCoA",
        "Affinity AcCoA",
        "FC PropCoA",
        "Affinity PropCoA",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = apply_numeric_parse(df, col)

    for col in ["Affinity AcCoA", "Affinity PropCoA"]:
        if f"{col}__value" not in df.columns:
            continue
        df[f"{col}__uM"] = [
            normalize_to_uM(value, unit, assume_unitless_affinity_um)
            for value, unit in zip(df[f"{col}__value"], df[f"{col}__unit"], strict=True)
        ]

    brightness_results = df["Brightness"].apply(
        lambda x: map_to_ordinal(x, BRIGHTNESS_SCALE)
    )
    df["Brightness__ordinal"] = brightness_results.str[0]
    df["Brightness__mapping_status"] = brightness_results.str[1]

    df["Improvement__status"] = df["Improvement from Baseline?"].apply(
        normalize_improvement
    )

    df["mut_from_construct"] = df["Construct"].apply(extract_mutations)
    df["mut_from_description"] = df["Description"].apply(extract_mutations)
    df["mut_codes_construct"] = df["mut_from_construct"].apply(mutation_codes)
    df["mut_codes_description"] = df["mut_from_description"].apply(mutation_codes)
    df["mutation_audit"] = df.apply(mutation_audit_status, axis=1)

    ratio_results = df.apply(
        lambda r: ratio_bounds(
            r["Affinity PropCoA__uM"],
            r["Affinity PropCoA__censor_direction"],
            r["Affinity AcCoA__uM"],
            r["Affinity AcCoA__censor_direction"],
        ),
        axis=1,
    )
    df["Selectivity_Kd_Prop_over_Ac__lower"] = ratio_results.str[0]
    df["Selectivity_Kd_Prop_over_Ac__upper"] = ratio_results.str[1]
    df["Selectivity_Kd_Prop_over_Ac__display"] = [
        format_ratio_bound(lo, hi)
        for lo, hi in zip(
            df["Selectivity_Kd_Prop_over_Ac__lower"],
            df["Selectivity_Kd_Prop_over_Ac__upper"],
            strict=True,
        )
    ]

    exact_ratio = np.isclose(
        df["Selectivity_Kd_Prop_over_Ac__lower"],
        df["Selectivity_Kd_Prop_over_Ac__upper"],
        equal_nan=False,
    )
    df["Selectivity_Kd_Prop_over_Ac__exact"] = np.where(
        exact_ratio,
        df["Selectivity_Kd_Prop_over_Ac__lower"],
        np.nan,
    )
    df["Selectivity__conservative_score"] = df["Selectivity_Kd_Prop_over_Ac__lower"]

    prop_raw = df["Affinity PropCoA"].astype(str).str.strip().str.lower()
    qualitative_evidence = pd.Series(pd.NA, index=df.index, dtype="object")
    similar_mask = prop_raw.str.contains(r"\bsimilar\b.*\baccoa\b", regex=True, na=False)
    qualitative_evidence.loc[similar_mask] = "qualitative_similar"
    manual_review_mask = (df["Affinity PropCoA__parse_status"] == "text") & ~similar_mask
    qualitative_evidence.loc[manual_review_mask] = "qualitative_manual_review"
    df["Selectivity__qualitative_evidence"] = qualitative_evidence

    df["construct_id"] = df["Construct"].map(make_construct_id)
    return df


def load_raw_experimental_workbook(path) -> pd.DataFrame:
    """Load the raw experimental mutant workbook from Excel.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the ``.xlsx`` or ``.xls`` workbook.

    Returns
    -------
    pandas.DataFrame
        Unmodified sheet contents as loaded by ``pandas.read_excel``.
    """
    return pd.read_excel(path)
