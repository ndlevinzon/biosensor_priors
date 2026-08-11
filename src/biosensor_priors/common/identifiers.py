"""Master identifier helpers for constructs and mutations."""

from __future__ import annotations

import re

MUT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z])\s*(\d+)\s*([A-Za-z])"
    r"(?![A-Za-z0-9])"
)


def make_construct_id(construct: str) -> str:
    """Stable construct identifier (currently the Construct label)."""
    return str(construct).strip()


def extract_mutations(text: object) -> list[tuple[str, int, str]]:
    """Parse one or more mutations: Q324R, Q324R/A355R, Q324R + A355R, …"""
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
    return [f"{aa1}{pos}{aa2}" for aa1, pos, aa2 in muts]


def mutation_to_code(mutation: tuple[str, int, str]) -> str:
    aa_from, position, aa_to = mutation
    return f"{aa_from}{int(position)}{aa_to}"
