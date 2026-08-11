"""Master identifier helpers for constructs and mutations."""

from __future__ import annotations

import re

MUT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z])\s*(\d+)\s*([A-Za-z])"
    r"(?![A-Za-z0-9])"
)


def make_construct_id(construct: str) -> str:
    """Build a stable construct identifier from a construct label.

    Parameters
    ----------
    construct : str
        Raw construct name or label from experimental records.

    Returns
    -------
    str
        Stripped construct string used as the canonical ``construct_id``.
    """
    return str(construct).strip()


def extract_mutations(text: object) -> list[tuple[str, int, str]]:
    """Parse one or more point mutations from free text.

    Supports formats such as ``Q324R``, ``Q324R/A355R``, and ``Q324R + A355R``.

    Parameters
    ----------
    text : object
        Cell value or string containing mutation notation. ``None`` and NaN
        yield an empty list.

    Returns
    -------
    list[tuple[str, int, str]]
        Parsed mutations as ``(from_aa, position, to_aa)`` tuples with
        uppercase amino-acid letters.
    """
    if text is None:
        return []
    try:
        import pandas as pd

        if pd.isna(text):
            return []
    except Exception:
        pass

    return [
        (aa1.upper(), int(pos), aa2.upper())
        for aa1, pos, aa2 in MUT_RE.findall(str(text))
    ]


def mutation_codes(muts: list[tuple[str, int, str]]) -> list[str]:
    """Convert mutation tuples to standard single-letter codes.

    Parameters
    ----------
    muts : list[tuple[str, int, str]]
        Mutations as ``(from_aa, position, to_aa)`` tuples.

    Returns
    -------
    list[str]
        Codes such as ``"Q324R"`` for each mutation.
    """
    return [f"{aa1}{pos}{aa2}" for aa1, pos, aa2 in muts]


def mutation_to_code(mutation: tuple[str, int, str]) -> str:
    """Format a single mutation tuple as a standard code.

    Parameters
    ----------
    mutation : tuple[str, int, str]
        Mutation as ``(from_aa, position, to_aa)``.

    Returns
    -------
    str
        Standard mutation code (e.g. ``"Q324R"``).
    """
    aa_from, position, aa_to = mutation
    return f"{aa_from}{int(position)}{aa_to}"
