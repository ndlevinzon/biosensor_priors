"""20-AA mutation scan engine over allowed canonical positions."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.stage2_physics.rif_jobs import prepare_rif_jobs_for_models
from biosensor_priors.stage2_physics.rpx_jobs import submit_or_mock_rpx_batch
from biosensor_priors.stage2_physics.score_parser import compute_delta_rif_sel, standardize_scan_row


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def make_physics_scan_id(
    *,
    version: str,
    positions: list[int],
    seed: int,
    backend: str,
) -> str:
    """Build a stable physics scan identifier from scan parameters.

    Parameters
    ----------
    version : str
        Design background version.
    positions : list of int
        Canonical positions included in the scan.
    seed : int
        Random seed for reproducibility.
    backend : str
        Physics backend name, e.g. ``mock`` or ``external``.

    Returns
    -------
    str
        Scan ID of the form ``scan_{version}_{digest12}``.
    """
    raw = f"{version}|{','.join(map(str, positions))}|{seed}|{backend}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"scan_{version}_{digest}"


def wildtype_aa_at_canonical(
    versions: pd.DataFrame,
    residue_mapping: pd.DataFrame | None,
    *,
    version: str,
    canonical_position: int,
    fallback_wt: dict[int, str] | None = None,
) -> str:
    """Resolve WT amino acid at a canonical position for a version background.

    Fallback map defaults to control hotspots Q324 / A355 when mapping is incomplete.

    Parameters
    ----------
    versions : pandas.DataFrame
        Version/sequence table from Stage 0 constructs.
    residue_mapping : pandas.DataFrame, optional
        Canonical position to amino acid mapping.
    version : str
        Design background version identifier.
    canonical_position : int
        Canonical residue position.
    fallback_wt : dict, optional
        Position-to-WT map used when mapping is incomplete.

    Returns
    -------
    str
        Single-letter wild-type amino acid code.
    """
    fallback_wt = fallback_wt or {324: "Q", 355: "A"}
    if residue_mapping is not None and not residue_mapping.empty:
        # Expect columns like Version, Canonical_position / canonical_position, AA
        cols = {c.lower(): c for c in residue_mapping.columns}
        ver_col = cols.get("version")
        pos_col = cols.get("canonical_position") or cols.get("canonical_pos") or cols.get("position")
        aa_col = cols.get("aa") or cols.get("amino_acid") or cols.get("residue")
        if ver_col and pos_col and aa_col:
            sub = residue_mapping[
                (residue_mapping[ver_col].astype(str) == str(version))
                & (pd.to_numeric(residue_mapping[pos_col], errors="coerce") == canonical_position)
            ]
            if not sub.empty:
                return str(sub.iloc[0][aa_col])

    # Sequence index fallback if Version + Sequence present
    if "Version" in versions.columns and "Sequence" in versions.columns:
        row = versions[versions["Version"].astype(str) == str(version)]
        if not row.empty and residue_mapping is not None:
            cols = {c.lower(): c for c in residue_mapping.columns}
            # optional version_position mapping
            _ = cols
    return fallback_wt.get(canonical_position, "X")


def generate_mutation_specs(
    *,
    version: str,
    positions: list[int],
    amino_acids: list[str],
    versions: pd.DataFrame,
    residue_mapping: pd.DataFrame | None = None,
    include_wt: bool = True,
) -> list[dict[str, Any]]:
    """Emit mutation specs for every allowed position × amino acid.

    Example: position 324 → A,C,D,...,Y (and WT Q retained when include_wt).

    Parameters
    ----------
    version : str
        Design background version identifier.
    positions : list of int
        Canonical positions to scan.
    amino_acids : list of str
        Allowed mutant amino acids.
    versions : pandas.DataFrame
        Version/sequence table for WT resolution.
    residue_mapping : pandas.DataFrame, optional
        Canonical position to amino acid mapping.
    include_wt : bool, optional
        When True, include wild-type entries (default True).

    Returns
    -------
    list of dict
        Mutation specs with ``version``, ``position``, ``wt``, ``mutant``,
        and ``mutation`` keys.
    """
    specs = []
    for pos in positions:
        wt = wildtype_aa_at_canonical(
            versions, residue_mapping, version=version, canonical_position=int(pos)
        )
        for aa in amino_acids:
            if not include_wt and aa == wt:
                continue
            specs.append(
                {
                    "version": version,
                    "position": int(pos),
                    "wt": wt,
                    "mutant": aa,
                    "mutation": f"{wt}{int(pos)}{aa}",
                }
            )
    return specs


def default_structure_models(
    *,
    version: str,
    predictors: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Build placeholder structure model registry until Stage 1 tables exist.

    Parameters
    ----------
    version : str
        Design background version identifier.
    predictors : list of str, optional
        Structure prediction methods (default ``AF2``, ``AF3``).
    seeds : list of int, optional
        Random seeds per predictor (default ``[1, 2]``).

    Returns
    -------
    pandas.DataFrame
        Registry with ``structure_model_id``, ``pdb_path``, and metadata.
    """
    predictors = predictors or ["AF2", "AF3"]
    seeds = seeds or [1, 2]
    rows = []
    for method in predictors:
        for seed in seeds:
            mid = f"{version}_{method}_seed{seed}_apo"
            rows.append(
                {
                    "structure_model_id": mid,
                    "version": version,
                    "method": method,
                    "seed": seed,
                    "state": "apo",
                    "pdb_path": f"data/structures/{mid}.pdb",
                }
            )
    return pd.DataFrame(rows)


def merge_rif_rpx_to_long(
    rif_scores: pd.DataFrame,
    rpx_scores: pd.DataFrame,
    *,
    physics_scan_id: str,
) -> pd.DataFrame:
    """Join RIF and RPX on mutation × structure_model_id.

    Parameters
    ----------
    rif_scores : pandas.DataFrame
        RIF score table with ``rif_ac`` and ``rif_prop``.
    rpx_scores : pandas.DataFrame
        RPX score table with ``rpx`` column.
    physics_scan_id : str
        Scan batch identifier attached to each row.

    Returns
    -------
    pandas.DataFrame
        Long-format table with raw scores and ``delta_rif_sel``.
    """
    if rif_scores.empty:
        return pd.DataFrame()
    rif = rif_scores.copy()
    rpx = rpx_scores.copy() if rpx_scores is not None else pd.DataFrame()
    if not rpx.empty:
        merged = rif.merge(
            rpx[["mutation", "structure_model_id", "rpx"]],
            on=["mutation", "structure_model_id"],
            how="left",
        )
    else:
        merged = rif.copy()
        merged["rpx"] = float("nan")

    rows = []
    for _, row in merged.iterrows():
        rif_ac = float(row["rif_ac"])
        rif_prop = float(row["rif_prop"])
        rows.append(
            standardize_scan_row(
                version=str(row.get("version") or ""),
                position=int(row["position"]),
                wt=str(row["wt"]),
                mutant=str(row["mutant"]),
                rif_ac=rif_ac,
                rif_prop=rif_prop,
                rpx=float(row["rpx"]) if pd.notna(row.get("rpx")) else float("nan"),
                structure_model_id=str(row.get("structure_model_id")),
                physics_scan_id=physics_scan_id,
                extra={
                    "backend": row.get("backend"),
                    # Explicitly store derived term (also inside standardize)
                    "delta_rif_sel": compute_delta_rif_sel(rif_ac, rif_prop),
                },
            )
        )
    return pd.DataFrame(rows)


def run_mutation_scan(
    *,
    repo_root: Path | None = None,
    version: str | None = None,
    structure_models: pd.DataFrame | None = None,
    positions: list[int] | None = None,
    amino_acids: list[str] | None = None,
    physics_scan_id: str | None = None,
) -> dict[str, Any]:
    """Stage 2C — generate specs, score via RIF/RPX, and write long table.

    Long-format columns include Version, Position, WT, Mutant, RIF Ac,
    RIF Prop, RPX, and delta_rif_sel.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root for config and output paths.
    version : str, optional
        Design background version override.
    structure_models : pandas.DataFrame, optional
        Structure model registry; built from defaults when omitted.
    positions : list of int, optional
        Canonical positions to scan.
    amino_acids : list of str, optional
        Allowed mutant amino acids.
    physics_scan_id : str, optional
        Pre-assigned scan ID; generated when omitted.

    Returns
    -------
    dict
        Keys ``physics_scan_id``, ``specs``, ``long_table``, ``path``,
        ``meta``, ``rif_jobs``, ``rpx_jobs``, and ``structure_models``.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    fitness = load_yaml(root / "configs" / "fitness.yaml")
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    physics_cfg = load_yaml(root / "configs" / "physics.yaml")

    version = version or str(pipeline.get("active_design_background", "V2.4"))
    design = fitness.get("design", {})
    scan_cfg = physics_cfg.get("scan", {})
    positions = positions or scan_cfg.get("positions") or list(
        design.get("allowed_mutable_positions") or [324, 355]
    )
    amino_acids = amino_acids or scan_cfg.get("amino_acids") or list(
        design.get("allowed_amino_acids") or AA20
    )
    include_wt = bool(scan_cfg.get("include_wt", True))
    seed = int(pipeline.get("random_seed", 42))
    backend = str(physics_cfg.get("backend", "mock"))

    physics_root = resolve_path(pipeline["paths"]["physics"], root)
    physics_root.mkdir(parents=True, exist_ok=True)
    constructs = resolve_path(pipeline["paths"]["constructs"], root)
    versions = pd.read_pickle(constructs / pipeline["constructs"]["versions_pickle"])
    mapping_path = constructs / pipeline["constructs"]["residue_mapping_pickle"]
    residue_mapping = pd.read_pickle(mapping_path) if mapping_path.exists() else None

    if structure_models is None:
        predictors = list(thresholds.get("structure", {}).get("predictors") or ["AF2", "AF3"])
        seeds = list(thresholds.get("structure", {}).get("seeds") or [1, 2])
        # Keep mock scans modest by default
        structure_models = default_structure_models(
            version=version,
            predictors=predictors[:2],
            seeds=seeds[:2],
        )

    scan_id = physics_scan_id or make_physics_scan_id(
        version=version, positions=list(positions), seed=seed, backend=backend
    )
    specs = generate_mutation_specs(
        version=version,
        positions=[int(p) for p in positions],
        amino_acids=[str(a) for a in amino_acids],
        versions=versions,
        residue_mapping=residue_mapping,
        include_wt=include_wt,
    )

    # Ligand dirs (may already exist from 2A)
    ligand_dirs = {
        lig: physics_root / "ligands" / lig
        for lig in (physics_cfg.get("ligands", {}).get("names") or ["AcCoA", "PropCoA"])
    }

    rif_result = prepare_rif_jobs_for_models(
        structure_models,
        mutations=specs,
        ligand_dirs=ligand_dirs,
        physics_root=physics_root,
        physics_cfg=physics_cfg,
        physics_scan_id=scan_id,
        seed=seed,
    )

    rpx_frames = []
    rpx_jobs = []
    path_col = "pdb_path" if "pdb_path" in structure_models.columns else "path"
    for _, model in structure_models.iterrows():
        mid = str(model["structure_model_id"])
        pdb = Path(str(model[path_col]))
        if not pdb.is_absolute():
            pdb = root / pdb
        job, scores = submit_or_mock_rpx_batch(
            structure_model_id=mid,
            structure_pdb=pdb,
            mutations=specs,
            physics_scan_id=scan_id,
            physics_cfg=physics_cfg,
            physics_root=physics_root,
            seed=seed,
        )
        rpx_jobs.append(job)
        rpx_frames.append(scores)

    rpx_scores = pd.concat(rpx_frames, ignore_index=True) if rpx_frames else pd.DataFrame()
    long_table = merge_rif_rpx_to_long(
        rif_result["scores"], rpx_scores, physics_scan_id=scan_id
    )

    out_dir = physics_root / "scans" / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "mutation_scan_long.parquet"
    long_table.to_parquet(long_path, index=False)
    # Human-readable columns matching the writeup table
    display = long_table.rename(
        columns={
            "version": "Version",
            "position": "Position",
            "wt": "WT",
            "mutant": "Mutant",
            "rif_ac": "RIF_Ac",
            "rif_prop": "RIF_Prop",
            "rpx": "RPX",
            "delta_rif_sel": "delta_RIF_sel",
        }
    )
    display.to_csv(out_dir / "mutation_scan_long.csv", index=False)

    meta = {
        "physics_scan_id": scan_id,
        "version": version,
        "positions": list(positions),
        "amino_acids": list(amino_acids),
        "n_specs": len(specs),
        "n_structure_models": int(len(structure_models)),
        "n_rows": int(len(long_table)),
        "backend": backend,
        "score_direction": thresholds.get("physics", {}).get("score_direction"),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "delta_rif_sel_definition": "RIF_Ac - RIF_Prop",
        "note": "Score direction is declared in thresholds.yaml; parsers do not guess.",
    }
    (out_dir / "scan_meta.json").write_text(
        __import__("json").dumps(meta, indent=2),
        encoding="utf-8",
    )

    return {
        "physics_scan_id": scan_id,
        "specs": specs,
        "long_table": long_table,
        "path": long_path,
        "meta": meta,
        "rif_jobs": rif_result["jobs"],
        "rpx_jobs": rpx_jobs,
        "structure_models": structure_models,
    }
