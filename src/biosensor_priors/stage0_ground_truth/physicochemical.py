"""Canonical physicochemical residue annotations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_AA_JSON = Path(__file__).with_name("aa_properties.json")


def load_aa_properties(path: Path | None = None) -> dict[str, dict]:
    src = path or _AA_JSON
    return json.loads(src.read_text(encoding="utf-8"))


def build_aa_property_table(
    aa_properties: dict[str, dict] | None = None,
    *,
    create_zscores: bool = True,
) -> pd.DataFrame:
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
    out = df.copy()
    out["Canonical_residue_label"] = (
        out["Version_AA"].astype(str) + out["Canonical_key"].astype(str)
    )

    def canonical_change(row: pd.Series):
        if pd.isna(row["Canonical_AA"]) or row["Canonical_AA"] == "-":
            return pd.NA
        if row["Canonical_AA"] == row["Version_AA"]:
            return pd.NA
        return f"{row['Canonical_AA']}{row['Canonical_key']}{row['Version_AA']}"

    out["Change_vs_canonical"] = out.apply(canonical_change, axis=1)
    return out


def build_physchem_residue_database(residue_mapping: pd.DataFrame) -> pd.DataFrame:
    lookup = build_aa_property_table()
    db = add_physicochemical_features(residue_mapping, lookup)
    db = add_delta_vs_canonical(db)
    db = add_residue_labels(db)
    return db.sort_values(["Version", "Version_position"]).reset_index(drop=True)
