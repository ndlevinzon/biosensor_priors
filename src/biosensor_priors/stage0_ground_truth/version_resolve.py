"""Resolve experimental rows onto biosensor version backgrounds."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def extract_numeric_version_token(text: Any) -> str | None:
    if pd.isna(text):
        return None
    m = re.search(r"(?<!\d)(\d+\.\d+)(?!\d)", str(text))
    return m.group(1) if m else None


def build_version_numeric_map(known_versions: list[str]) -> dict[str, list[str]]:
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
    """
    Resolve background biosensor version.

    Priority: exact column match → aliases → version substring → unique numeric token.
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
    """Trusted mutations: None = mismatch, [] = baseline, else mutation tuples."""
    if row.get("mutation_audit") == "MISMATCH":
        return None
    c = row.get("mut_from_construct", [])
    d = row.get("mut_from_description", [])
    if isinstance(c, list) and c:
        return c
    if isinstance(d, list) and d:
        return d
    return []
