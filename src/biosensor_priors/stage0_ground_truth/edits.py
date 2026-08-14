"""Canonical edit codes: substitutions plus insertions/deletions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from biosensor_priors.stage0_ground_truth.version_resolve import get_row_mutations

Edit = tuple[str, int, str]

_SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_INDEL_RE = re.compile(r"^(ins|del)(?:Nterm|(\d+)([A-Z])?)$", re.IGNORECASE)
_D104_RE = re.compile(r"D\s*104\s*insertion", re.IGNORECASE)

DEFAULT_COSTS = {
    "substitution": 1.0,
    "insertion": 3.0,
    "deletion": 3.0,
    "block": 4.0,
    "lambda": 0.08,
}


def format_edit(aa_from: str, position: int, aa_to: str) -> str:
    """Format an edit triple as a stable code (``Q324R``, ``ins104``, ``insNterm``)."""
    src = str(aa_from)
    dst = str(aa_to)
    pos = int(position)
    if src in {"+", "I"} or src.lower() == "ins":
        if pos == 0:
            return "insNterm"
        if dst.isalpha() and dst not in {"X", "N", "+"}:
            return f"ins{pos}{dst}"
        return f"ins{pos}"
    if src in {"-", "D"} or src.lower() == "del":
        if pos == 0:
            return "delNterm"
        return f"del{pos}"
    return f"{src}{pos}{dst}"


def parse_edit_code(code: str) -> Edit | None:
    """Parse a substitution or indel code into an ``(aa_from, pos, aa_to)`` triple."""
    s = str(code).strip()
    if not s:
        return None
    sub = _SUB_RE.match(s)
    if sub:
        return (sub.group(1), int(sub.group(2)), sub.group(3))
    indel = _INDEL_RE.match(s)
    if indel:
        kind = indel.group(1).lower()
        if indel.group(2) is None:
            pos = 0
            aa = "N"
        else:
            pos = int(indel.group(2))
            aa = (indel.group(3) or "X").upper()
        return ("+", pos, aa) if kind == "ins" else ("-", pos, aa)
    return None


def edit_kind(edit: Edit) -> str:
    """Return ``substitution``, ``insertion``, or ``deletion``."""
    src = edit[0]
    if src in {"+", "I"}:
        return "insertion"
    if src in {"-", "D"}:
        return "deletion"
    return "substitution"


def is_block_edit(edit: Edit) -> bool:
    """True for the N-terminal 242-residue insertion/deletion event."""
    return int(edit[1]) == 0 and edit[0] in {"+", "-", "I", "D"}


def mutation_cost(
    edits: Iterable[Edit | str],
    *,
    costs: dict[str, float] | None = None,
) -> float:
    """Sum per-edit costs (indels cost more than substitutions; N-term is a block)."""
    cfg = {**DEFAULT_COSTS, **(costs or {})}
    total = 0.0
    for item in edits:
        edit = parse_edit_code(item) if isinstance(item, str) else item
        if edit is None:
            continue
        if is_block_edit(edit):
            total += float(cfg.get("block", 4.0))
        else:
            total += float(cfg.get(edit_kind(edit), 1.0))
    return float(total)


def edits_from_text(text: object) -> list[Edit]:
    """Parse D104-style insertions from free text."""
    if text is None:
        return []
    try:
        if pd.isna(text):
            return []
    except (TypeError, ValueError):
        pass
    blob = str(text)
    if _D104_RE.search(blob):
        return [("+", 104, "X")]
    return []


def scaffold_edits(
    mapping: pd.DataFrame,
    version: str,
) -> list[Edit]:
    """Version-vs-V1.0 substitutions plus a single N-term insertion event if present."""
    if mapping.empty or "Version" not in mapping.columns:
        return []
    sub = mapping[mapping["Version"].astype(str) == str(version)]
    if sub.empty:
        return []
    edits: list[Edit] = []
    if "Relation" in sub.columns:
        ins = sub[sub["Relation"].astype(str) == "insertion"]
        if not ins.empty:
            edits.append(("+", 0, "N"))
        subs = sub[sub["Relation"].astype(str) == "substitution"]
        for _, row in subs.iterrows():
            try:
                pos = int(str(row["Canonical_key"]).split("i")[0])
            except (TypeError, ValueError):
                continue
            aa_from = str(row.get("Canonical_AA", "")).upper()
            aa_to = str(row.get("Version_AA", "")).upper()
            if (
                aa_from.isalpha()
                and aa_to.isalpha()
                and len(aa_from) == 1
                and len(aa_to) == 1
            ):
                edits.append((aa_from, pos, aa_to))
    return _unique_edits(edits)


def _unique_edits(edits: Iterable[Edit]) -> list[Edit]:
    seen: set[str] = set()
    out: list[Edit] = []
    for edit in edits:
        code = format_edit(*edit)
        if code in seen:
            continue
        seen.add(code)
        out.append(edit)
    return out


def merge_edits(*groups: Iterable[Edit]) -> list[Edit]:
    """Concatenate edit groups, dropping duplicate codes."""
    merged: list[Edit] = []
    for group in groups:
        merged.extend(group)
    return _unique_edits(merged)


def compose_canonical(
    scaffold: Iterable[str],
    proposed: Iterable[str],
) -> list[str]:
    """Apply proposed edits onto a parent scaffold bag.

    Substitutions replace any scaffold substitution at the same canonical
    position. ``delNterm`` removes ``insNterm`` (and vice versa) so the bag
    does not carry both the block and its deletion.
    """
    codes = [str(c) for c in scaffold]

    def _sub_pos(code: str) -> int | None:
        parsed = parse_edit_code(code)
        if parsed is None or edit_kind(parsed) != "substitution":
            return None
        return int(parsed[1])

    for raw in proposed:
        code = str(raw)
        parsed = parse_edit_code(code)
        if parsed is None:
            if code not in codes:
                codes.append(code)
            continue
        kind = edit_kind(parsed)
        pos = int(parsed[1])
        if kind == "deletion" and pos == 0:
            codes = [c for c in codes if c != "insNterm"]
            if "delNterm" not in codes:
                codes.append("delNterm")
            continue
        if kind == "insertion" and pos == 0:
            codes = [c for c in codes if c != "delNterm"]
            if "insNterm" not in codes:
                codes.append("insNterm")
            continue
        if kind == "substitution":
            codes = [c for c in codes if _sub_pos(c) != pos]
            formatted = format_edit(*parsed)
            if formatted not in codes:
                codes.append(formatted)
            continue
        formatted = format_edit(*parsed)
        if formatted not in codes:
            codes.append(formatted)
    return codes


def construct_edits(row: pd.Series) -> list[Edit]:
    """Trusted construct substitutions plus text-mined indels (not scaffold)."""
    if str(row.get("mutation_audit", "") or "") == "MISMATCH":
        return []
    muts = get_row_mutations(row)
    local: list[Edit] = list(muts) if muts else []
    text_bits = [
        row.get("Construct"),
        row.get("Description"),
        row.get("construct_id"),
    ]
    for blob in text_bits:
        local.extend(edits_from_text(blob))
    for col in ("mut_codes_construct", "mut_codes_description", "mutation_codes"):
        val = row.get(col)
        if isinstance(val, list):
            for code in val:
                parsed = parse_edit_code(str(code))
                if parsed is not None:
                    local.append(parsed)
    return _unique_edits(local)


def parse_mutation_list(row: pd.Series) -> list[Edit]:
    """All canonical edits for a row: scaffold + construct + explicit codes.

    ``mutation_audit == MISMATCH`` returns an empty list. Prefers a precomputed
    ``canonical_edit_codes`` column when present.
    """
    if str(row.get("mutation_audit", "") or "") == "MISMATCH":
        return []
    pre = row.get("canonical_edit_codes")
    if isinstance(pre, list) and pre:
        parsed = [parse_edit_code(str(c)) for c in pre]
        return [e for e in parsed if e is not None]
    sc_edits: list[Edit] = []
    scaffold = row.get("scaffold_edits")
    if isinstance(scaffold, list):
        for item in scaffold:
            if isinstance(item, tuple) and len(item) == 3:
                sc_edits.append((str(item[0]), int(item[1]), str(item[2])))
            else:
                parsed = parse_edit_code(str(item))
                if parsed is not None:
                    sc_edits.append(parsed)
    return merge_edits(sc_edits, construct_edits(row))


def attach_canonical_edits(
    df: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write ``scaffold_edits`` and ``canonical_edit_codes`` (lists of codes)."""
    out = df.copy()
    by_version: dict[str, list[str]] = {}
    if mapping is not None and not mapping.empty:
        for ver in mapping["Version"].astype(str).unique():
            by_version[str(ver)] = [
                format_edit(*e) for e in scaffold_edits(mapping, ver)
            ]
    scaffold_col: list[list[str]] = []
    canonical_col: list[list[str] | None] = []
    for _, row in out.iterrows():
        if str(row.get("mutation_audit", "") or "") == "MISMATCH":
            scaffold_col.append([])
            canonical_col.append(None)
            continue
        version = None if pd.isna(row.get("version")) else str(row.get("version"))
        sc = list(by_version.get(version or "", []))
        local = [format_edit(*e) for e in construct_edits(row)]
        merged = compose_canonical(sc, local)
        scaffold_col.append(sc)
        canonical_col.append(merged)
    out["scaffold_edits"] = scaffold_col
    out["canonical_edit_codes"] = canonical_col
    return out


def load_residue_mapping(repo_root: Any | None = None) -> pd.DataFrame:
    """Load the construct residue-mapping pickle, or empty if missing."""
    from pathlib import Path

    from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path

    root = Path(repo_root or REPO_ROOT)
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    path = constructs / pipeline["constructs"]["residue_mapping_pickle"]
    if path.exists():
        return pd.read_pickle(path)
    return pd.DataFrame()
