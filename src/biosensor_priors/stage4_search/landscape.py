"""Measured-landscape utilities for paper-faithful search policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.edits import parse_mutation_list


@dataclass
class LandscapeView:
    """Indexed measured landscape with variable-site sequence strings."""

    frame: pd.DataFrame
    sequences: list[str]
    site_positions: list[int]
    site_alphabets: list[list[str]]
    distance: np.ndarray

    @property
    def n_sites(self) -> int:
        """Number of variable sites in the landscape representation.

        Parameters
        ----------
        None
            This is a read-only property.

        Returns
        -------
        int
            Length of ``site_positions``.
        """
        return len(self.site_positions)


def hamming(a: str, b: str) -> int:
    """Count differing characters between two equal-length strings.

    Parameters
    ----------
    a : str
        First sequence string.
    b : str
        Second sequence string (same length as ``a``).

    Returns
    -------
    int
        Hamming distance between ``a`` and ``b``.
    """
    return sum(x != y for x, y in zip(a, b, strict=True))


def pairwise_hamming(sequences: list[str]) -> np.ndarray:
    """Build a symmetric pairwise Hamming distance matrix.

    Parameters
    ----------
    sequences : list of str
        Variable-site sequence strings for each construct.

    Returns
    -------
    np.ndarray
        Square ``(n, n)`` integer matrix of pairwise Hamming distances.
    """
    n = len(sequences)
    d = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming(sequences[i], sequences[j])
            d[i, j] = dist
            d[j, i] = dist
    return d


def build_landscape_view(df: pd.DataFrame) -> LandscapeView:
    """Build variable-site sequences and distance matrix from mutation codes.

    Sites are the union of mutated canonical positions in ``df``. Each construct
    sequence is the WT-at-site string with trusted mutations applied.

    Parameters
    ----------
    df : pd.DataFrame
        Table of constructs with mutation-code columns.

    Returns
    -------
    LandscapeView
        Indexed landscape with sequences, site alphabets, and Hamming distances.
    """
    work = df.reset_index(drop=True).copy()
    mut_lists = [parse_mutation_list(row) for _, row in work.iterrows()]

    wt_at_pos: dict[int, str] = {}
    observed_at_pos: dict[int, set[str]] = {}
    for muts in mut_lists:
        for aa_from, pos, aa_to in muts:
            if aa_from in {"+", "I"}:
                wt_at_pos.setdefault(pos, "-")
                observed_at_pos.setdefault(pos, set()).update({"-", "I"})
            elif aa_from in {"-", "D"}:
                wt = aa_to if aa_to.isalpha() else "X"
                wt_at_pos.setdefault(pos, wt)
                observed_at_pos.setdefault(pos, set()).update({wt, "-"})
            else:
                wt_at_pos.setdefault(pos, aa_from)
                observed_at_pos.setdefault(pos, set()).add(aa_from)
                observed_at_pos[pos].add(aa_to)

    if not wt_at_pos:
        # Degenerate: single-site placeholder so policies still run.
        site_positions = [0]
        sequences = ["X"] * len(work)
        alphabets = [["X"]]
        dist = pairwise_hamming(sequences)
        work["sequence"] = sequences
        return LandscapeView(work, sequences, site_positions, alphabets, dist)

    site_positions = sorted(wt_at_pos)
    alphabets = [
        sorted(observed_at_pos.get(p, {wt_at_pos[p]}) | {wt_at_pos[p]})
        for p in site_positions
    ]

    sequences: list[str] = []
    for muts in mut_lists:
        state = {p: wt_at_pos[p] for p in site_positions}
        for aa_from, pos, aa_to in muts:
            if pos not in state:
                continue
            if aa_from in {"+", "I"}:
                state[pos] = "I"
            elif aa_from in {"-", "D"}:
                state[pos] = "-"
            else:
                state[pos] = aa_to
        sequences.append("".join(state[p] for p in site_positions))

    work["sequence"] = sequences
    dist = pairwise_hamming(sequences)
    return LandscapeView(work, sequences, site_positions, alphabets, dist)


def top_b_by_score(frame: pd.DataFrame, score_col: str, batch_size: int) -> pd.DataFrame:
    """Return the top-scoring rows from a ranked candidate table.

    Parameters
    ----------
    frame : pd.DataFrame
        Candidate table containing ``score_col``.
    score_col : str
        Column to sort by in descending order.
    batch_size : int
        Maximum number of rows to return.

    Returns
    -------
    pd.DataFrame
        Top ``batch_size`` rows by ``score_col``, or empty when inputs are invalid.
    """
    if frame.empty or batch_size <= 0:
        return frame.iloc[0:0].copy()
    return frame.sort_values(score_col, ascending=False).head(batch_size).copy()
