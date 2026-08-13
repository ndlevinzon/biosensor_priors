"""Parse and standardize Stage-2 physics score outputs; retain raw terms.

Legacy column names ``rif_ac`` / ``rif_prop`` are kept for Stage 3.
Values are typically negated RF3 docking confidence (higher → more negative).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_RAW_COLUMNS = (
    "rif_ac",
    "rif_prop",
    "delta_rif_sel",
)


def compute_delta_rif_sel(rif_ac: float, rif_prop: float) -> float:
    """Selectivity term: ``rif_ac - rif_prop`` (sign convention elsewhere)."""
    return float(rif_ac) - float(rif_prop)


def parse_rif_score_table(path: Path) -> pd.DataFrame:
    """Parse an RF3 / interface score file into a standardized table.

    Parameters
    ----------
    path : pathlib.Path
        Path to a score TSV/CSV file.

    Returns
    -------
    pandas.DataFrame
        Table with canonical column names (``rif_ac``, ``rif_prop``, etc.).
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    if df.empty:
        return df

    rename: dict[str, str] = {}
    lower_map = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "mutation": ["mutation", "mut", "variant"],
        "position": ["position", "pos", "resi"],
        "wt": ["wt", "wildtype", "from_aa"],
        "mutant": ["mutant", "to_aa", "aa"],
        "version": ["version", "construct_version"],
        "structure_model_id": ["structure_model_id", "model_id", "structure_id"],
        "rif_ac": ["rif_ac", "rif_accoa", "rif ac", "score_ac", "accoa"],
        "rif_prop": ["rif_prop", "rif_propcoa", "rif prop", "score_prop", "propcoa"],
        "backend": ["backend", "engine"],
    }
    for canon, names in aliases.items():
        for name in names:
            if name in lower_map:
                rename[lower_map[name]] = canon
                break
    out = df.rename(columns=rename)
    if "delta_rif_sel" not in out.columns and {"rif_ac", "rif_prop"}.issubset(
        out.columns
    ):
        out["delta_rif_sel"] = [
            compute_delta_rif_sel(a, b)
            for a, b in zip(out["rif_ac"], out["rif_prop"], strict=False)
        ]
    return out


def standardize_scan_row(
    *,
    version: str,
    position: int,
    wt: str,
    mutant: str,
    rif_ac: float,
    rif_prop: float,
    structure_model_id: str | None = None,
    physics_scan_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one long-format physics scan row with raw and derived terms."""
    row = {
        "version": version,
        "position": int(position),
        "wt": str(wt),
        "mutant": str(mutant),
        "mutation": f"{wt}{int(position)}{mutant}",
        "rif_ac": float(rif_ac),
        "rif_prop": float(rif_prop),
        "delta_rif_sel": compute_delta_rif_sel(rif_ac, rif_prop),
        "structure_model_id": structure_model_id,
        "physics_scan_id": physics_scan_id,
    }
    if extra:
        for k, v in extra.items():
            if k not in row:
                row[k] = v
    return row


def write_mock_rif_scores(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write mock interface score rows to a TSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mutation",
        "position",
        "wt",
        "mutant",
        "version",
        "structure_model_id",
        "rif_ac",
        "rif_prop",
        "backend",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, sep="\t", index=False)
    return path
