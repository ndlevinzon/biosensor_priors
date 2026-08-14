"""Generate the constrained candidate universe from mutable positions."""

from __future__ import annotations

import itertools
from typing import Iterable

import pandas as pd

from biosensor_priors.stage0_ground_truth.physicochemical import load_aa_properties

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")


def canonical_to_version_positions(
    residue_mapping: pd.DataFrame,
    *,
    version: str,
    canonical_positions: Iterable[int | str],
) -> dict[int, int]:
    """Map canonical residue keys to version-local positions for one background.

    Parameters
    ----------
    residue_mapping : pd.DataFrame
        Residue mapping table with ``Version``, ``Canonical_key``, and
        ``Version_position`` columns.
    version : str
        Background version identifier.
    canonical_positions : Iterable of int or str
        Canonical positions to resolve.

    Returns
    -------
    dict of int to int
        Mapping from canonical position to version-local index.

    Raises
    ------
    ValueError
        If any requested canonical position is missing from the mapping.
    """
    want = {str(p) for p in canonical_positions}
    sub = residue_mapping[
        (residue_mapping["Version"].astype(str) == str(version))
        & (residue_mapping["Canonical_key"].astype(str).isin(want))
        & (residue_mapping["Version_position"].notna())
    ]
    mapping: dict[int, int] = {}
    for _, row in sub.iterrows():
        mapping[int(str(row["Canonical_key"]).split("i")[0])] = int(row["Version_position"])
    missing = sorted({int(p) for p in want} - set(mapping))
    if missing:
        raise ValueError(
            f"Canonical positions {missing} not mapped for version {version}."
        )
    return mapping


def generate_design_space(
    *,
    parent_version: str,
    parent_sequence: str,
    mutable_positions: Iterable[int],
    allowed_amino_acids: Iterable[str] | None = None,
    max_mutations: int = 2,
    exclude_wt: bool = True,
    position_labels: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Enumerate single and multi-site mutants up to ``max_mutations``.

    ``mutable_positions`` are version-local indices into ``parent_sequence``.
    Optional ``position_labels`` maps version-local indices to canonical positions
    for mutation IDs.

    Parameters
    ----------
    parent_version : str
        Background version label for generated construct IDs.
    parent_sequence : str
        Parent amino-acid sequence (1-indexed positions in ``mutable_positions``).
    mutable_positions : Iterable of int
        Version-local indices that may be mutated.
    allowed_amino_acids : Iterable of str or None, optional
        Allowed destination amino acids (default standard 20).
    max_mutations : int, optional
        Maximum number of simultaneous mutations per construct (default 2).
    exclude_wt : bool, optional
        When True, skip mutations that restore the parent amino acid (default True).
    position_labels : dict of int to int or None, optional
        Version-local to canonical position labels for mutation codes.

    Returns
    -------
    pd.DataFrame
        Design-space table with mutation codes, physicochemical deltas, and
        placeholder physics columns (NaN until Stage 2/1 priors are joined).

    Raises
    ------
    ValueError
        If any mutable position falls outside the parent sequence length.
    """
    allowed = [a.upper() for a in (allowed_amino_acids or STANDARD_AA)]
    positions = sorted({int(p) for p in mutable_positions})
    wt = {i + 1: aa for i, aa in enumerate(parent_sequence)}
    missing = [p for p in positions if p not in wt]
    if missing:
        raise ValueError(f"Mutable positions outside parent sequence length: {missing}")

    labels = position_labels or {p: p for p in positions}
    rows: list[dict] = []
    aa_props = load_aa_properties()

    for n_mut in range(1, max_mutations + 1):
        for pos_combo in itertools.combinations(positions, n_mut):
            aa_choices = []
            for pos in pos_combo:
                opts = [aa for aa in allowed if (not exclude_wt) or aa != wt[pos]]
                aa_choices.append(opts)
            for aa_combo in itertools.product(*aa_choices):
                mutations = []
                canonical_positions = []
                for pos, aa_to in zip(pos_combo, aa_combo, strict=True):
                    canon = int(labels.get(pos, pos))
                    mutations.append(f"{wt[pos]}{canon}{aa_to}")
                    canonical_positions.append(canon)
                cand_id = f"{parent_version}|" + "/".join(mutations)
                deltas = {
                    f"delta_{k}": 0.0
                    for k in (
                        "hydrophobicity_KD",
                        "side_chain_volume_A3",
                        "polarity_Grantham",
                        "charge",
                    )
                }
                for mut, aa_to in zip(mutations, aa_combo, strict=True):
                    aa_from = mut[0]
                    for k in deltas:
                        key = k.replace("delta_", "")
                        deltas[k] += float(aa_props[aa_to][key]) - float(
                            aa_props[aa_from][key]
                        )
                rows.append(
                    {
                        "candidate_id": cand_id,
                        "construct_id": cand_id,
                        "parent_version": parent_version,
                        "version": parent_version,
                        "mutations": mutations,
                        "mutation_codes": mutations,
                        "canonical_positions": canonical_positions,
                        "version_positions": list(pos_combo),
                        "n_mutations": n_mut,
                        "rif_ac": float("nan"),
                        "rif_prop": float("nan"),
                        "delta_rif_sel": float("nan"),
                        "structural_confidence": float("nan"),
                        **deltas,
                    }
                )

    return pd.DataFrame(rows)


def design_space_from_config(
    versions: pd.DataFrame,
    *,
    parent_version: str,
    mutable_positions: list[int],
    allowed_amino_acids: list[str],
    max_mutations: int,
    residue_mapping: pd.DataFrame | None = None,
    positions_are_canonical: bool = True,
) -> pd.DataFrame:
    """Build a combinatorial design space from pipeline configuration inputs.

    Parameters
    ----------
    versions : pd.DataFrame
        Version table containing ``Version`` and ``Sequence_clean`` columns.
    parent_version : str
        Background version to mutate.
    mutable_positions : list of int
        Canonical or version-local mutable positions depending on
        ``positions_are_canonical``.
    allowed_amino_acids : list of str
        Allowed destination amino acids at each mutable site.
    max_mutations : int
        Maximum simultaneous mutations per construct.
    residue_mapping : pd.DataFrame or None, optional
        Required when ``positions_are_canonical`` is True.
    positions_are_canonical : bool, optional
        When True, resolve ``mutable_positions`` via ``residue_mapping`` (default True).

    Returns
    -------
    pd.DataFrame
        Combinatorial design-space table from :func:`generate_design_space`.

    Raises
    ------
    ValueError
        If the parent version is missing or canonical mapping is required but absent.
    """
    row = versions.loc[versions["Version"].astype(str) == parent_version]
    if row.empty:
        raise ValueError(f"Parent version {parent_version} not found in versions table.")
    seq = str(row.iloc[0]["Sequence_clean"])

    if positions_are_canonical:
        if residue_mapping is None:
            raise ValueError("residue_mapping required when positions_are_canonical=True")
        canon_to_version = canonical_to_version_positions(
            residue_mapping,
            version=parent_version,
            canonical_positions=mutable_positions,
        )
        version_positions = list(canon_to_version.values())
        labels = {v: c for c, v in canon_to_version.items()}
    else:
        version_positions = list(mutable_positions)
        labels = {p: p for p in version_positions}

    return generate_design_space(
        parent_version=parent_version,
        parent_sequence=seq,
        mutable_positions=version_positions,
        allowed_amino_acids=allowed_amino_acids,
        max_mutations=max_mutations,
        position_labels=labels,
    )
