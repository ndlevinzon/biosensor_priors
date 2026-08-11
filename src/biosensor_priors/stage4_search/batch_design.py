"""Batch diversification after acquisition ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _mutation_positions(row: pd.Series) -> set[int]:
    """Extract canonical mutation positions from a candidate row.

    Parameters
    ----------
    row : pd.Series
        Candidate row with ``canonical_positions`` or mutation-code columns.

    Returns
    -------
    set of int
        Canonical positions touched by the candidate's mutations.
    """
    pos = row.get("canonical_positions")
    if isinstance(pos, list):
        return {int(p) for p in pos}
    codes = row.get("mutation_codes") or row.get("mutations") or []
    out = set()
    if isinstance(codes, list):
        for code in codes:
            s = str(code)
            if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha():
                try:
                    out.add(int(s[1:-1]))
                except ValueError:
                    continue
    return out


def _sequence_distance(a: pd.Series, b: pd.Series) -> int:
    """Approximate sequence distance as symmetric difference of mutation codes.

    Parameters
    ----------
    a : pd.Series
        First candidate row.
    b : pd.Series
        Second candidate row.

    Returns
    -------
    int
        Size of the symmetric difference between mutation-code sets.
    """
    ca = set(map(str, a.get("mutation_codes") or a.get("mutations") or []))
    cb = set(map(str, b.get("mutation_codes") or b.get("mutations") or []))
    return len(ca.symmetric_difference(cb))


def diversify_batch(
    ranked: pd.DataFrame,
    *,
    batch_size: int,
    max_candidates_per_position: int = 2,
    min_sequence_distance: int = 1,
    exploitation_fraction: float = 0.7,
) -> pd.DataFrame:
    """Greedy diversification over an already-ranked candidate table.

    Selects a batch balancing high acquisition scores with position caps and
    minimum sequence distance between picks.

    Parameters
    ----------
    ranked : pd.DataFrame
        Candidate table sorted or sortable by ``acquisition``.
    batch_size : int
        Target number of candidates to select.
    max_candidates_per_position : int, optional
        Maximum selections sharing any single mutable position (default 2).
    min_sequence_distance : int, optional
        Minimum mutation-code distance between selected pairs (default 1).
    exploitation_fraction : float, optional
        Fraction of the batch filled by top acquisition under constraints (default 0.7).

    Returns
    -------
    pd.DataFrame
        Diversified subset of up to ``batch_size`` rows from ``ranked``.
    """
    if ranked.empty or batch_size <= 0:
        return ranked.iloc[0:0].copy()

    work = ranked.reset_index(drop=True)
    if "acquisition" in work.columns:
        work = work.sort_values("acquisition", ascending=False).reset_index(drop=True)

    n_exploit = max(1, int(round(batch_size * exploitation_fraction)))
    n_explore = max(0, batch_size - n_exploit)

    selected_idx: list[int] = []
    pos_counts: dict[int, int] = {}

    def can_add(i: int, selected: list[int]) -> bool:
        """Check whether candidate ``i`` satisfies diversification constraints.

        Parameters
        ----------
        i : int
            Row index in ``work`` to evaluate.
        selected : list of int
            Indices already chosen for the batch.

        Returns
        -------
        bool
            True if adding ``i`` respects position caps and distance rules.
        """
        row = work.iloc[i]
        positions = _mutation_positions(row)
        for p in positions:
            if pos_counts.get(p, 0) >= max_candidates_per_position:
                return False
        for j in selected:
            if _sequence_distance(row, work.iloc[j]) < min_sequence_distance:
                return False
        return True

    def add(i: int) -> None:
        """Append candidate ``i`` to the selection and update position counts.

        Parameters
        ----------
        i : int
            Row index in ``work`` to add.

        Returns
        -------
        None
        """
        selected_idx.append(i)
        for p in _mutation_positions(work.iloc[i]):
            pos_counts[p] = pos_counts.get(p, 0) + 1

    # Exploitation: top acquisition under constraints
    for i in range(len(work)):
        if len(selected_idx) >= n_exploit:
            break
        if can_add(i, selected_idx):
            add(i)

    # Exploration: prefer high predictive uncertainty among remaining
    remaining = [i for i in range(len(work)) if i not in selected_idx]
    if n_explore and "pred_fitness_std" in work.columns:
        remaining = sorted(remaining, key=lambda i: float(work.iloc[i]["pred_fitness_std"]), reverse=True)
    for i in remaining:
        if len(selected_idx) >= batch_size:
            break
        if can_add(i, selected_idx):
            add(i)

    # Fill if constraints were too strict
    if len(selected_idx) < batch_size:
        for i in range(len(work)):
            if len(selected_idx) >= batch_size:
                break
            if i not in selected_idx:
                selected_idx.append(i)

    return work.iloc[selected_idx[:batch_size]].copy()
