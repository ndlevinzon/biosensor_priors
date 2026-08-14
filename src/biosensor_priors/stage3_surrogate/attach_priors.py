"""Join Stage-1 confidence and Stage-2 physics onto construct / design rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT
from biosensor_priors.stage0_ground_truth.edits import (
    edit_kind,
    format_edit,
    parse_edit_code,
    parse_mutation_list,
)

MultiMutant = Literal["sum", "max_abs"]

PHYSICS_SCORE_COLS = ("rif_ac", "rif_prop", "delta_rif_sel")
_STD_CANDIDATES = (
    "physics_score_std",
    "delta_rif_sel_std",
    "delta_rif_sel_sd",
    "rif_ac_std",
    "rif_ac_sd",
)


def resolve_multi_mutant(value: str | None) -> MultiMutant:
    """Return a valid multi-mutant aggregation mode (default ``sum``)."""
    return "max_abs" if str(value or "").strip() == "max_abs" else "sum"


def mutation_codes_from_row(row: pd.Series) -> list[str]:
    """Return construct-local mutation codes (not the parent scaffold bag)."""
    val = row.get("mutation_codes")
    if isinstance(val, list):
        return [str(c) for c in val]
    muts = parse_mutation_list(row)
    return [format_edit(a, p, b) for a, p, b in muts]


def load_physics_lookup(repo_root: Path | None = None) -> pd.DataFrame:
    """Load mutation-level physics scores from processed, summary, or scan tables."""
    root = Path(repo_root or REPO_ROOT)
    candidates = [
        root / "data" / "processed" / "physics_mutation_scores.parquet",
        root / "data" / "physics" / "physics_scores_summary.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    scan_dir = root / "data" / "physics" / "scans"
    if scan_dir.exists():
        tables = sorted(scan_dir.glob("*/mutation_scan_long.parquet"))
        if tables:
            long = pd.concat([pd.read_parquet(p) for p in tables], ignore_index=True)
            from biosensor_priors.stage2_physics.physics_uncertainty import (
                aggregate_physics_uncertainty,
            )

            return aggregate_physics_uncertainty(long)
    return pd.DataFrame()


def load_confidence_lookup(repo_root: Path | None = None) -> pd.DataFrame:
    """Load Stage-1 per-position structural confidence, or empty if absent."""
    root = Path(repo_root or REPO_ROOT)
    path = root / "data" / "structures" / "structural_confidence.parquet"
    if path.exists():
        return pd.read_parquet(path)
    processed = root / "data" / "processed" / "structural_confidence.parquet"
    if processed.exists():
        return pd.read_parquet(processed)
    return pd.DataFrame()


def _row_float(row: pd.Series, names: tuple[str, ...]) -> float:
    for name in names:
        if name in row.index:
            val = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                return float(val)
    return float("nan")


def _lookup_mutation(
    table: pd.DataFrame,
    *,
    mutation: str,
    version: str | None,
) -> pd.Series | None:
    if table.empty or "mutation" not in table.columns:
        return None
    sub = table[table["mutation"].astype(str) == str(mutation)]
    if sub.empty:
        return None
    if version is not None and "version" in sub.columns:
        matched = sub[sub["version"].astype(str) == str(version)]
        if not matched.empty:
            sub = matched
    return sub.iloc[0]


def _confidence_for_position(
    table: pd.DataFrame,
    *,
    version: str | None,
    position: int,
) -> float:
    if table.empty:
        return float("nan")
    cols = {c.lower(): c for c in table.columns}
    pos_col = (
        cols.get("canonical_position")
        or cols.get("canonical key")
        or cols.get("canonical_key")
        or cols.get("position")
    )
    conf_col = (
        cols.get("structural_confidence")
        or cols.get("confidence")
        or cols.get("conf")
    )
    if pos_col is None or conf_col is None:
        return float("nan")
    work = table.copy()
    work["_pos"] = pd.to_numeric(work[pos_col], errors="coerce")
    hit = work[work["_pos"] == int(position)]
    if version is not None:
        ver_col = cols.get("version")
        if ver_col is not None:
            matched = hit[hit[ver_col].astype(str) == str(version)]
            if not matched.empty:
                hit = matched
    if hit.empty:
        return float("nan")
    return _row_float(hit.iloc[0], (conf_col,))


def _aggregate_mutant_physics(
    parts: list[dict[str, float]],
    *,
    method: MultiMutant,
) -> dict[str, float]:
    confs = [
        p["structural_confidence"]
        for p in parts
        if np.isfinite(p["structural_confidence"])
    ]
    conf = float(np.min(confs)) if confs else float("nan")
    finite = [p for p in parts if any(np.isfinite(p[c]) for c in PHYSICS_SCORE_COLS)]
    empty = {
        "rif_ac": float("nan"),
        "rif_prop": float("nan"),
        "delta_rif_sel": float("nan"),
        "physics_score_std": float("nan"),
        "structural_confidence": conf,
        "n_physics_mutations": 0,
    }
    if not finite:
        return empty
    if method == "max_abs":

        def _key(p: dict[str, float]) -> float:
            d = p.get("delta_rif_sel", float("nan"))
            return abs(d) if np.isfinite(d) else -1.0

        best = max(finite, key=_key)
        out = dict(empty)
        for col in PHYSICS_SCORE_COLS:
            out[col] = best[col]
        out["physics_score_std"] = best["physics_score_std"]
        out["structural_confidence"] = conf
        out["n_physics_mutations"] = 1
        return out
    out = dict(empty)
    out["n_physics_mutations"] = len(finite)
    for col in PHYSICS_SCORE_COLS:
        vals = [p[col] for p in finite if np.isfinite(p[col])]
        out[col] = float(np.sum(vals)) if vals else float("nan")
    stds = [
        p["physics_score_std"]
        for p in finite
        if np.isfinite(p["physics_score_std"])
    ]
    if stds:
        out["physics_score_std"] = float(np.sqrt(np.sum(np.square(stds))))
    out["structural_confidence"] = conf
    return out


def attach_physics_and_confidence(
    df: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    physics_table: pd.DataFrame | None = None,
    confidence_table: pd.DataFrame | None = None,
    multi_mutant: MultiMutant = "sum",
) -> pd.DataFrame:
    """Attach physics scores and confidence; missing values stay NaN (never 1.0).

    Multi-mutants aggregate per-mutation scores with ``sum`` (default) or
    ``max_abs`` (the member with largest |delta_rif_sel|). Confidence is the
    minimum across mutated positions (weakest link).
    """
    out = df.copy()
    physics = (
        physics_table if physics_table is not None else load_physics_lookup(repo_root)
    )
    confidence = (
        confidence_table
        if confidence_table is not None
        else load_confidence_lookup(repo_root)
    )
    records: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        version = None if pd.isna(row.get("version")) else str(row.get("version"))
        codes = mutation_codes_from_row(row)
        parts: list[dict[str, float]] = []
        for code in codes:
            parsed = parse_edit_code(str(code))
            if parsed is not None and edit_kind(parsed) != "substitution":
                continue
            hit = _lookup_mutation(physics, mutation=code, version=version)
            pos = int(code[1:-1]) if len(code) >= 3 and code[1:-1].isdigit() else None
            conf = (
                _confidence_for_position(confidence, version=version, position=pos)
                if pos is not None
                else float("nan")
            )
            if hit is None:
                parts.append(
                    {
                        "rif_ac": float("nan"),
                        "rif_prop": float("nan"),
                        "delta_rif_sel": float("nan"),
                        "physics_score_std": float("nan"),
                        "structural_confidence": conf,
                    }
                )
                continue
            parts.append(
                {
                    "rif_ac": _row_float(hit, ("rif_ac", "rif_ac_mean")),
                    "rif_prop": _row_float(hit, ("rif_prop", "rif_prop_mean")),
                    "delta_rif_sel": _row_float(
                        hit, ("delta_rif_sel", "delta_rif_sel_mean")
                    ),
                    "physics_score_std": _row_float(hit, _STD_CANDIDATES),
                    "structural_confidence": conf
                    if np.isfinite(conf)
                    else _row_float(hit, ("structural_confidence",)),
                }
            )
        if not codes:
            conf = float("nan")
            if version is not None and not confidence.empty:
                cols = {c.lower(): c for c in confidence.columns}
                ver_col = cols.get("version")
                conf_col = (
                    cols.get("structural_confidence")
                    or cols.get("confidence")
                    or cols.get("conf")
                )
                if ver_col and conf_col:
                    sub = confidence[confidence[ver_col].astype(str) == version]
                    vals = pd.to_numeric(sub[conf_col], errors="coerce").dropna()
                    if not vals.empty:
                        conf = float(vals.mean())
            agg = {
                "rif_ac": float("nan"),
                "rif_prop": float("nan"),
                "delta_rif_sel": float("nan"),
                "physics_score_std": float("nan"),
                "structural_confidence": conf,
                "n_physics_mutations": 0,
            }
        else:
            agg = _aggregate_mutant_physics(parts, method=multi_mutant)
        records.append(agg)
    attached = pd.DataFrame.from_records(records, index=out.index)
    for col in attached.columns:
        out[col] = attached[col]
    return out
