"""Canonical physicochemical residue annotations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_AA_JSON = Path(__file__).with_name("aa_properties.json")


def load_aa_properties(path: Path | None = None) -> dict[str, dict]:
    """Load amino-acid physicochemical property definitions from JSON.

    Parameters
    ----------
    path : pathlib.Path | None, optional
        Path to the properties JSON file. Defaults to ``aa_properties.json``
        alongside this module.

    Returns
    -------
    dict[str, dict]
        Mapping from single-letter amino-acid code to property dictionaries.
    """
    src = path or _AA_JSON
    return json.loads(src.read_text(encoding="utf-8"))


def build_aa_property_table(
    aa_properties: dict[str, dict] | None = None,
    *,
    create_zscores: bool = True,
) -> pd.DataFrame:
    """Build a lookup table of amino-acid physicochemical properties.

    Parameters
    ----------
    aa_properties : dict[str, dict] | None, optional
        Property definitions keyed by amino-acid code. Loaded from JSON when
        ``None``.
    create_zscores : bool, optional
        When ``True``, append z-scored continuous property columns.
        Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Sorted lookup table with one row per standard amino acid.

    Raises
    ------
    ValueError
        If the table is missing standard amino acids or contains unexpected
        codes.
    """
    props = aa_properties or load_aa_properties()
    rows = [{"AA": aa, **values} for aa, values in props.items()]
    lookup = pd.DataFrame(rows).sort_values("AA").reset_index(drop=True)

    standard = set("ACDEFGHIKLMNPQRSTVWY")
    observed = set(lookup["AA"])
    if standard - observed:
        raise ValueError(f"Missing amino acids: {sorted(standard - observed)}")
    if observed - standard:
        raise ValueError(f"Unexpected amino acids: {sorted(observed - standard)}")

    if create_zscores:
        continuous = [
            "hydrophobicity_KD",
            "side_chain_volume_A3",
            "molecular_weight_Da",
            "polarity_Grantham",
        ]
        for col in continuous:
            mean = lookup[col].mean()
            sd = lookup[col].std(ddof=0)
            lookup[f"{col}_z"] = (lookup[col] - mean) / sd
    return lookup


def add_physicochemical_features(
    mapping: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach version and canonical amino-acid properties to a mapping table.

    Parameters
    ----------
    mapping : pandas.DataFrame
        Residue mapping with ``Version_AA`` and ``Canonical_AA`` columns.
    lookup : pandas.DataFrame
        Amino-acid property table from :func:`build_aa_property_table`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``mapping`` with merged ``AA_*`` and ``Canonical_*`` property
        columns.
    """
    df = mapping.copy()
    version_lookup = lookup.rename(
        columns={col: (f"AA_{col}" if col != "AA" else "Version_AA") for col in lookup.columns}
    )
    df = df.merge(version_lookup, on="Version_AA", how="left", validate="many_to_one")

    canonical_lookup = lookup.rename(
        columns={
            col: (f"Canonical_{col}" if col != "AA" else "Canonical_AA")
            for col in lookup.columns
        }
    )
    return df.merge(canonical_lookup, on="Canonical_AA", how="left", validate="many_to_one")


def add_delta_vs_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Compute version-minus-canonical deltas for physicochemical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with ``AA_*`` and ``Canonical_*`` property columns from
        :func:`add_physicochemical_features`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``Delta_*_vs_canonical`` columns for continuous
        and binary properties.
    """
    out = df.copy()
    continuous = [
        "charge",
        "hydrophobicity_KD",
        "side_chain_volume_A3",
        "molecular_weight_Da",
        "polarity_Grantham",
    ]
    for feature in continuous:
        out[f"Delta_{feature}_vs_canonical"] = (
            out[f"AA_{feature}"] - out[f"Canonical_{feature}"]
        )

    binary_features = [
        "polar",
        "aromatic",
        "hbond_donor",
        "hbond_acceptor",
        "positive",
        "negative",
        "branched",
        "sulfur_containing",
        "Gly",
        "Pro",
    ]
    for feature in binary_features:
        out[f"Delta_{feature}_vs_canonical"] = (
            out[f"AA_{feature}"] - out[f"Canonical_{feature}"]
        )
    return out


def add_residue_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable residue and mutation labels relative to canonical.

    Parameters
    ----------
    df : pandas.DataFrame
        Physicochemical residue table with version and canonical AA columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``Canonical_residue_label`` and
        ``Change_vs_canonical`` columns.
    """
    out = df.copy()
    out["Canonical_residue_label"] = (
        out["Version_AA"].astype(str) + out["Canonical_key"].astype(str)
    )

    def canonical_change(row: pd.Series):
        """Format a canonical mutation code for one residue row.

        Parameters
        ----------
        row : pandas.Series
            Row with ``Canonical_AA``, ``Canonical_key``, and ``Version_AA``.

        Returns
        -------
        str | pandas._libs.missing.NAType
            Mutation string such as ``"Q324R"``, or NA for matches and gaps.
        """
        if pd.isna(row["Canonical_AA"]) or row["Canonical_AA"] == "-":
            return pd.NA
        if row["Canonical_AA"] == row["Version_AA"]:
            return pd.NA
        return f"{row['Canonical_AA']}{row['Canonical_key']}{row['Version_AA']}"

    out["Change_vs_canonical"] = out.apply(canonical_change, axis=1)
    return out


def build_physchem_residue_database(residue_mapping: pd.DataFrame) -> pd.DataFrame:
    """Build the full per-residue physicochemical annotation database.

    Parameters
    ----------
    residue_mapping : pandas.DataFrame
        Canonical residue mapping from :func:`build_canonical_mapping`.

    Returns
    -------
    pandas.DataFrame
        Annotated residue table sorted by version and version position.
    """
    lookup = build_aa_property_table()
    db = add_physicochemical_features(residue_mapping, lookup)
    db = add_delta_vs_canonical(db)
    db = add_residue_labels(db)
    return db.sort_values(["Version", "Version_position"]).reset_index(drop=True)
