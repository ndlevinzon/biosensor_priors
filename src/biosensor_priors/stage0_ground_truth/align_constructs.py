"""Version-aware canonical residue numbering (align every version to V1.0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

GAP_OPEN_SCORE = -10.0
GAP_EXTEND_SCORE = -0.5


def clean_protein_sequence(seq) -> str | None:
    """Normalize a protein sequence string for alignment.

    Parameters
    ----------
    seq : Any
        Raw sequence value, possibly NaN.

    Returns
    -------
    str | None
        Uppercase sequence with whitespace and gaps removed and trailing ``*``
        stripped, or ``None`` when missing or empty.
    """
    if pd.isna(seq):
        return None
    seq = str(seq).upper()
    seq = "".join(seq.split()).replace("-", "").rstrip("*")
    return seq or None


def load_version_database(filename: Path, canonical_version: str) -> pd.DataFrame:
    """Load and validate the biosensor version sequence database.

    Parameters
    ----------
    filename : pathlib.Path
        Excel workbook with version metadata and sequences.
    canonical_version : str
        Version label used as the canonical numbering reference.

    Returns
    -------
    pandas.DataFrame
        Validated version table with cleaned sequences and length checks.

    Raises
    ------
    ValueError
        If required columns are missing, sequences are empty, version names
        are duplicated, or the canonical version is absent.
    """
    df = pd.read_excel(filename)
    required = ["Version", "Parent", "Sequence", "Length"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Version"] = df["Version"].astype(str).str.strip()
    df["Parent"] = df["Parent"].where(df["Parent"].notna(), pd.NA)
    df["Parent"] = df["Parent"].apply(
        lambda x: str(x).strip() if pd.notna(x) else pd.NA
    )
    df["Sequence_clean"] = df["Sequence"].apply(clean_protein_sequence)
    if df["Sequence_clean"].isna().any():
        bad = df.loc[df["Sequence_clean"].isna(), "Version"].tolist()
        raise ValueError(f"Missing/empty sequences for: {bad}")

    df["Length_actual"] = df["Sequence_clean"].str.len()
    df["Length_reported"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Length_matches"] = df["Length_reported"] == df["Length_actual"]

    if df["Version"].duplicated().any():
        duplicates = df.loc[df["Version"].duplicated(keep=False), "Version"].tolist()
        raise ValueError(f"Duplicate version names: {sorted(set(duplicates))}")
    if canonical_version not in set(df["Version"]):
        raise ValueError(f"Canonical version '{canonical_version}' not found.")
    return df


def build_aligner() -> PairwiseAligner:
    """Create a global BLOSUM62 pairwise aligner with gap penalties.

    Returns
    -------
    Bio.Align.PairwiseAligner
        Configured aligner for canonical-to-version sequence alignment.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = GAP_OPEN_SCORE
    aligner.extend_gap_score = GAP_EXTEND_SCORE
    return aligner


def validate_sequence_alphabet(sequence: str, version: str, matrix) -> None:
    """Ensure all residues in a sequence are supported by the substitution matrix.

    Parameters
    ----------
    sequence : str
        Protein sequence to validate.
    version : str
        Version label included in error messages.
    matrix
        Biopython substitution matrix with an ``alphabet`` attribute.

    Raises
    ------
    ValueError
        If ``sequence`` contains characters outside ``matrix.alphabet``.
    """
    invalid = sorted(set(sequence) - set(matrix.alphabet))
    if invalid:
        raise ValueError(
            f"{version} contains amino-acid characters not supported by BLOSUM62: {invalid}"
        )


def build_mapping_from_alignment(
    canonical_version: str,
    canonical_sequence: str,
    version: str,
    parent,
    version_sequence: str,
    alignment,
) -> tuple[list[dict], list[dict], dict, str, str]:
    """Derive per-residue canonical mapping rows from one pairwise alignment.

    Parameters
    ----------
    canonical_version : str
        Label of the canonical reference version.
    canonical_sequence : str
        Unaligned canonical sequence.
    version : str
        Label of the version being mapped.
    parent
        Parent version name, or pandas NA.
    version_sequence : str
        Unaligned version sequence.
    alignment
        Biopython alignment object for canonical vs version sequences.

    Returns
    -------
    alignment_rows : list[dict]
        Full alignment trace including deletions.
    residue_rows : list[dict]
        Mapped version residues (matches, substitutions, insertions).
    summary : dict
        Alignment statistics and identity metrics.
    canonical_aligned : str
        Gapped canonical sequence string from the alignment.
    version_aligned : str
        Gapped version sequence string from the alignment.

    Raises
    ------
    RuntimeError
        If aligned sequence strings differ in length.
    """
    canonical_aligned = str(alignment[0])
    version_aligned = str(alignment[1])
    if len(canonical_aligned) != len(version_aligned):
        raise RuntimeError("Aligned sequence lengths differ.")

    canonical_position = 0
    version_position = 0
    insertion_counter: dict[int, int] = {}
    alignment_rows: list[dict] = []
    residue_rows: list[dict] = []
    matches = substitutions = insertions = deletions = aligned_pairs = 0

    for alignment_column, (canonical_aa, version_aa) in enumerate(
        zip(canonical_aligned, version_aligned, strict=True),
        start=1,
    ):
        if canonical_aa != "-":
            canonical_position += 1
        if version_aa != "-":
            version_position += 1

        if canonical_aa != "-" and version_aa != "-":
            canonical_key = str(canonical_position)
            if canonical_aa == version_aa:
                relation = "match"
                matches += 1
            else:
                relation = "substitution"
                substitutions += 1
            aligned_pairs += 1
            row = {
                "Canonical_version": canonical_version,
                "Version": version,
                "Parent": parent,
                "Alignment_column": alignment_column,
                "Version_position": version_position,
                "Version_AA": version_aa,
                "Canonical_position": canonical_position,
                "Canonical_key": canonical_key,
                "Canonical_AA": canonical_aa,
                "Insertion_after_canonical": pd.NA,
                "Insertion_index": pd.NA,
                "Relation": relation,
            }
            alignment_rows.append(row)
            residue_rows.append(row.copy())

        elif canonical_aa == "-" and version_aa != "-":
            boundary = canonical_position
            insertion_counter[boundary] = insertion_counter.get(boundary, 0) + 1
            insertion_index = insertion_counter[boundary]
            canonical_key = f"{boundary}i{insertion_index}"
            insertions += 1
            row = {
                "Canonical_version": canonical_version,
                "Version": version,
                "Parent": parent,
                "Alignment_column": alignment_column,
                "Version_position": version_position,
                "Version_AA": version_aa,
                "Canonical_position": pd.NA,
                "Canonical_key": canonical_key,
                "Canonical_AA": "-",
                "Insertion_after_canonical": boundary,
                "Insertion_index": insertion_index,
                "Relation": "insertion",
            }
            alignment_rows.append(row)
            residue_rows.append(row.copy())

        elif canonical_aa != "-" and version_aa == "-":
            deletions += 1
            alignment_rows.append(
                {
                    "Canonical_version": canonical_version,
                    "Version": version,
                    "Parent": parent,
                    "Alignment_column": alignment_column,
                    "Version_position": pd.NA,
                    "Version_AA": "-",
                    "Canonical_position": canonical_position,
                    "Canonical_key": str(canonical_position),
                    "Canonical_AA": canonical_aa,
                    "Insertion_after_canonical": pd.NA,
                    "Insertion_index": pd.NA,
                    "Relation": "deletion",
                }
            )

    summary = {
        "Canonical_version": canonical_version,
        "Version": version,
        "Parent": parent,
        "Canonical_length": len(canonical_sequence),
        "Version_length": len(version_sequence),
        "Alignment_length": len(canonical_aligned),
        "Alignment_score": alignment.score,
        "Matches": matches,
        "Substitutions": substitutions,
        "Insertions": insertions,
        "Deletions": deletions,
        "Aligned_residue_pairs": aligned_pairs,
        "Sequence_identity": matches / aligned_pairs if aligned_pairs else np.nan,
    }
    return alignment_rows, residue_rows, summary, canonical_aligned, version_aligned


def build_canonical_mapping(
    versions: pd.DataFrame,
    canonical_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Align all versions to a canonical reference and build mapping tables.

    Parameters
    ----------
    versions : pandas.DataFrame
        Version database from :func:`load_version_database`.
    canonical_version : str
        Reference version label for canonical numbering.

    Returns
    -------
    residue_mapping : pandas.DataFrame
        Per-residue mapping rows for all versions.
    alignment_detail : pandas.DataFrame
        Full alignment trace including deletions.
    summaries : pandas.DataFrame
        Per-version alignment summary statistics.
    alignment_text : list[str]
        Human-readable alignment strings for logging or export.
    """
    canonical_row = versions.loc[versions["Version"] == canonical_version].iloc[0]
    canonical_sequence = canonical_row["Sequence_clean"]
    aligner = build_aligner()
    matrix = substitution_matrices.load("BLOSUM62")
    validate_sequence_alphabet(canonical_sequence, canonical_version, matrix)

    all_alignment_rows: list[dict] = []
    all_residue_rows: list[dict] = []
    all_summaries: list[dict] = []
    alignment_text: list[str] = []

    for _, row in versions.iterrows():
        version = row["Version"]
        parent = row["Parent"]
        sequence = row["Sequence_clean"]
        validate_sequence_alphabet(sequence, version, matrix)
        alignment = aligner.align(canonical_sequence, sequence)[0]
        alignment_rows, residue_rows, summary, _, _ = build_mapping_from_alignment(
            canonical_version,
            canonical_sequence,
            version,
            parent,
            sequence,
            alignment,
        )
        all_alignment_rows.extend(alignment_rows)
        all_residue_rows.extend(residue_rows)
        all_summaries.append(summary)
        alignment_text.extend(
            [
                "=" * 80,
                f"{version} vs {canonical_version}",
                f"Score: {alignment.score:.2f}",
                "=" * 80,
                str(alignment),
                "",
            ]
        )

    return (
        pd.DataFrame(all_residue_rows),
        pd.DataFrame(all_alignment_rows),
        pd.DataFrame(all_summaries),
        alignment_text,
    )


def validate_mapping(versions: pd.DataFrame, residue_mapping: pd.DataFrame) -> list[str]:
    """Run QC checks on a canonical residue mapping table.

    Parameters
    ----------
    versions : pandas.DataFrame
        Version database with ``Version`` and ``Length_actual`` columns.
    residue_mapping : pandas.DataFrame
        Per-residue mapping from :func:`build_canonical_mapping`.

    Returns
    -------
    list[str]
        Human-readable problem descriptions; empty when all checks pass.
    """
    problems: list[str] = []
    for _, version_row in versions.iterrows():
        version = version_row["Version"]
        expected_length = int(version_row["Length_actual"])
        mapped = residue_mapping[residue_mapping["Version"] == version]
        if len(mapped) != expected_length:
            problems.append(
                f"{version}: sequence length={expected_length}, mapped residues={len(mapped)}"
            )
        if mapped["Version_position"].duplicated().any():
            problems.append(f"{version}: duplicate Version_position values.")
        positions = (
            mapped["Version_position"].dropna().astype(int).sort_values().tolist()
        )
        if positions != list(range(1, expected_length + 1)):
            problems.append(f"{version}: version residue numbering is not continuous.")
    return problems
