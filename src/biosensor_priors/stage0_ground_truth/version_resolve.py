"""Resolve experimental rows onto biosensor version backgrounds."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def extract_numeric_version_token(text: Any) -> str | None:
    """Extract a ``major.minor`` version token from free text.

    Parameters
    ----------
    text : Any
        String or cell value potentially containing a version number.

    Returns
    -------
    str | None
        Matched token such as ``"1.0"``, or ``None`` when absent or NaN.
    """
    if pd.isna(text):
        return None
    m = re.search(r"(?<!\d)(\d+\.\d+)(?!\d)", str(text))
    return m.group(1) if m else None


def build_version_numeric_map(known_versions: list[str]) -> dict[str, list[str]]:
    """Group known version names by their numeric ``major.minor`` token.

    Parameters
    ----------
    known_versions : list[str]
        Full version labels from the version database.

    Returns
    -------
    dict[str, list[str]]
        Mapping from numeric token to all version names sharing that token.
    """
    mapping: dict[str, list[str]] = {}
    for version in known_versions:
        token = extract_numeric_version_token(version)
        if token:
            mapping.setdefault(token, []).append(version)
    return mapping


def resolve_version_for_row(
    row: pd.Series,
    known_versions: list[str],
    numeric_map: dict[str, list[str]],
    version_aliases: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Resolve the background biosensor version for one experimental row.

    Priority: exact column match → aliases → version substring → unique
    numeric token.

    Parameters
    ----------
    row : pandas.Series
        Experimental record row.
    known_versions : list[str]
        Valid version names from the construct database.
    numeric_map : dict[str, list[str]]
        Numeric token → version names mapping from
        :func:`build_version_numeric_map`.
    version_aliases : dict[str, str] | None, optional
        Substring aliases mapping matched text to a canonical version name.

    Returns
    -------
    version : str | None
        Resolved version name, or ``None`` when unresolved.
    method : str
        Resolution method tag (e.g. ``"exact:Construct"``, ``"unresolved"``).

    Raises
    ------
    ValueError
        If an alias target is not present in ``known_versions``.
    """
    aliases = version_aliases or {}
    fields: list[tuple[str, str]] = []
    for col in [
        "Version",
        "Sensor Version",
        "Base Version",
        "Base construct",
        "Construct",
    ]:
        if col in row.index and pd.notna(row[col]):
            fields.append((col, str(row[col]).strip()))

    for col, text in fields:
        if text in known_versions:
            return text, f"exact:{col}"

    for alias, target in aliases.items():
        for col, text in fields:
            if alias.lower() in text.lower():
                if target not in known_versions:
                    raise ValueError(
                        f"VERSION alias target {target!r} not in version database"
                    )
                return target, f"alias:{col}:{alias}"

    for col, text in fields:
        lower = text.lower()
        for version in sorted(known_versions, key=len, reverse=True):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(version.lower())}(?![A-Za-z0-9])"
            if re.search(pattern, lower):
                return version, f"substring:{col}"

    for col, text in fields:
        token = extract_numeric_version_token(text)
        if token and token in numeric_map and len(numeric_map[token]) == 1:
            return numeric_map[token][0], f"numeric:{col}"

    return None, "unresolved"


def attach_resolved_versions(
    clean: pd.DataFrame,
    versions: pd.DataFrame,
    version_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach resolved biosensor version metadata to experimental rows.

    Parameters
    ----------
    clean : pandas.DataFrame
        Cleaned experimental table.
    versions : pandas.DataFrame
        Version database with at least a ``Version`` column and optional
        ``Parent`` column.
    version_aliases : dict[str, str] | None, optional
        Substring aliases for version resolution.

    Returns
    -------
    pandas.DataFrame
        Copy of ``clean`` with ``version``, ``version_resolution``, and
        ``parent_version`` columns added.
    """
    known = sorted(versions["Version"].astype(str).unique())
    numeric_map = build_version_numeric_map(known)
    parent_map = (
        versions.set_index("Version")["Parent"].astype("string").to_dict()
        if "Parent" in versions.columns
        else {}
    )

    out = clean.copy()
    resolved = []
    methods = []
    parents = []
    for _, row in out.iterrows():
        version, method = resolve_version_for_row(
            row, known, numeric_map, version_aliases
        )
        resolved.append(version)
        methods.append(method)
        parents.append(parent_map.get(version) if version else pd.NA)

    out["version"] = resolved
    out["version_resolution"] = methods
    out["parent_version"] = parents
    return out


def get_row_mutations(row: pd.Series) -> list[tuple[str, int, str]] | None:
    """Return trusted mutations for a row after audit checks.

    Parameters
    ----------
    row : pandas.Series
        Row with ``mutation_audit``, ``mut_from_construct``, and
        ``mut_from_description`` fields.

    Returns
    -------
    list[tuple[str, int, str]] | None
        ``None`` when audit status is ``"MISMATCH"``; empty list for baseline
        constructs; otherwise mutation tuples from construct or description.
    """
    if str(row.get("mutation_audit", "") or "") == "MISMATCH":
        return None
    c = row.get("mut_from_construct", [])
    d = row.get("mut_from_description", [])
    if isinstance(c, list) and c:
        return c
    if isinstance(d, list) and d:
        return d
    return []
