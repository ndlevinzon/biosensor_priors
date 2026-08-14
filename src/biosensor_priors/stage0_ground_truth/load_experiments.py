"""Build the authoritative Stage-0 ``experiment_master`` dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biosensor_priors.common.config import (
    REPO_ROOT,
    load_stage0_configs,
    resolve_path,
)
from biosensor_priors.common.gate_reports import write_stage0_report
from biosensor_priors.common.provenance import sha256_file, write_manifest
from biosensor_priors.stage0_ground_truth.align_constructs import (
    build_canonical_mapping,
    load_version_database,
    validate_mapping,
)
from biosensor_priors.stage0_ground_truth.clean import (
    load_raw_experimental_workbook,
    prepare_database,
)
from biosensor_priors.stage0_ground_truth.edits import (
    attach_canonical_edits,
    construct_edits,
    format_edit,
)
from biosensor_priors.stage0_ground_truth.fitness import fitness_transform
from biosensor_priors.stage0_ground_truth.physicochemical import (
    build_aa_property_table,
    build_physchem_residue_database,
)
from biosensor_priors.stage0_ground_truth.splits import (
    generate_leave_one_out_splits,
    generate_random_holdout_splits,
    write_splits,
)
from biosensor_priors.stage0_ground_truth.validate import run_stage0_gates
from biosensor_priors.stage0_ground_truth.version_resolve import (
    attach_resolved_versions,
)


def _read_table(path: Path) -> pd.DataFrame:
    """Load a persisted table from pickle, parquet, or Excel.

    Parameters
    ----------
    path : pathlib.Path
        Path to the table file.

    Returns
    -------
    pandas.DataFrame
        Loaded table contents.

    Raises
    ------
    ValueError
        If the file suffix is not supported.
    """
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def _ensure_construct_artifacts(
    constructs_dir: Path,
    *,
    epochs_workbook: str,
    versions_pickle: str,
    residue_mapping_pickle: str,
    physchem_pickle: str,
    aa_lookup_pickle: str,
    canonical_reference: str,
    rebuild_from_epochs: bool,
) -> dict[str, Path]:
    """Ensure construct-database artifacts exist, rebuilding when requested.

    Parameters
    ----------
    constructs_dir : pathlib.Path
        Directory for version, mapping, and physicochemical artifacts.
    epochs_workbook : str
        Filename of the epochs Excel workbook within ``constructs_dir``.
    versions_pickle : str
        Filename for the serialized version database.
    residue_mapping_pickle : str
        Filename for the serialized residue mapping.
    physchem_pickle : str
        Filename for the serialized physicochemical residue database.
    aa_lookup_pickle : str
        Filename for the serialized amino-acid property lookup.
    canonical_reference : str
        Canonical version label for alignment.
    rebuild_from_epochs : bool
        When ``True``, force rebuild from the epochs workbook.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths to ``versions``, ``residue_mapping``, ``physchem``, ``aa_lookup``,
        and ``epochs`` artifacts.

    Raises
    ------
    ValueError
        If canonical mapping QC fails during rebuild.
    """
    versions_path = constructs_dir / versions_pickle
    mapping_path = constructs_dir / residue_mapping_pickle
    physchem_path = constructs_dir / physchem_pickle
    aa_path = constructs_dir / aa_lookup_pickle
    epochs_path = constructs_dir / epochs_workbook

    need_rebuild = rebuild_from_epochs or not versions_path.exists() or not mapping_path.exists()
    if need_rebuild:
        versions = load_version_database(epochs_path, canonical_reference)
        residue_mapping, _, _, alignment_text = build_canonical_mapping(
            versions, canonical_reference
        )
        problems = validate_mapping(versions, residue_mapping)
        if problems:
            raise ValueError("Canonical mapping QC failed:\n" + "\n".join(problems))
        versions.to_pickle(versions_path)
        residue_mapping.to_pickle(mapping_path)
        (constructs_dir / "canonical_alignments.txt").write_text(
            "\n".join(alignment_text),
            encoding="utf-8",
        )

    if rebuild_from_epochs or not physchem_path.exists() or not aa_path.exists():
        residue_mapping = _read_table(mapping_path)
        aa_lookup = build_aa_property_table()
        physchem = build_physchem_residue_database(residue_mapping)
        aa_lookup.to_pickle(aa_path)
        physchem.to_pickle(physchem_path)

    return {
        "versions": versions_path,
        "residue_mapping": mapping_path,
        "physchem": physchem_path,
        "aa_lookup": aa_path,
        "epochs": epochs_path,
    }


def _attach_mutation_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach standardized mutation code lists to each experimental row.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with rows processable by :func:`get_row_mutations`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a ``mutation_codes`` column (list or ``None``).
    """
    out = df.copy()
    codes = []
    for _, row in out.iterrows():
        if str(row.get("mutation_audit", "") or "") == "MISMATCH":
            codes.append(None)
        else:
            codes.append([format_edit(*e) for e in construct_edits(row)])
    out["mutation_codes"] = codes
    return out


def _dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to strings for pyarrow parquet serialization.

    Mixed-type object columns (lists, dicts, scalars) are stringified so
    parquet writers do not fail on heterogeneous cells.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with potentially mixed object-dtype columns.

    Returns
    -------
    pandas.DataFrame
        Copy safe for ``to_parquet`` export.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        out[col] = out[col].map(
            lambda value: None
            if value is None
            or (not isinstance(value, (list, dict, tuple)) and pd.isna(value))
            else str(value)
        )
    return out


def build_experiment_master(
    *,
    config_dir: Path | None = None,
    repo_root: Path | None = None,
    rebuild_constructs: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load experimental and construct data and build the master table.

    Applies cleaning, version resolution, fitness scoring, split generation,
    and Stage-0 gate validation. Writes artifacts under ``data/processed``
    and a Stage-0 manifest.

    Parameters
    ----------
    config_dir : pathlib.Path | None, optional
        Directory containing ``pipeline.yaml`` and ``fitness.yaml``.
    repo_root : pathlib.Path | None, optional
        Repository root for resolving configured paths. Defaults to
        ``REPO_ROOT``.
    rebuild_constructs : bool | None, optional
        When set, overrides the ``rebuild_from_epochs`` pipeline flag.

    Returns
    -------
    master : pandas.DataFrame
        Authoritative experiment master table with fitness and metadata.
    meta : dict[str, Any]
        Run metadata including paths, gate report, and row counts.

    Raises
    ------
    ValueError
        If construct artifact rebuild fails canonical mapping QC.
    RuntimeError
        If Stage-0 validation gates do not pass.
    """
    root = repo_root or REPO_ROOT
    pipeline_cfg, fitness_cfg = load_stage0_configs(config_dir)
    if rebuild_constructs is not None:
        pipeline_cfg = dict(pipeline_cfg)
        constructs_cfg = dict(pipeline_cfg.get("constructs", {}))
        constructs_cfg["rebuild_from_epochs"] = bool(rebuild_constructs)
        pipeline_cfg["constructs"] = constructs_cfg
    paths = pipeline_cfg["paths"]

    experimental_dir = resolve_path(paths["experimental"], root)
    constructs_dir = resolve_path(paths["constructs"], root)
    processed_dir = resolve_path(paths["processed"], root)
    splits_dir = resolve_path(paths["splits"], root)
    manifests_dir = resolve_path(paths["manifests"], root)
    processed_dir.mkdir(parents=True, exist_ok=True)

    construct_paths = _ensure_construct_artifacts(
        constructs_dir,
        epochs_workbook=pipeline_cfg["constructs"]["epochs_workbook"],
        versions_pickle=pipeline_cfg["constructs"]["versions_pickle"],
        residue_mapping_pickle=pipeline_cfg["constructs"]["residue_mapping_pickle"],
        physchem_pickle=pipeline_cfg["constructs"]["physchem_pickle"],
        aa_lookup_pickle=pipeline_cfg["constructs"]["aa_lookup_pickle"],
        canonical_reference=pipeline_cfg["canonical_reference"],
        rebuild_from_epochs=bool(pipeline_cfg["constructs"].get("rebuild_from_epochs")),
    )

    raw_path = experimental_dir / pipeline_cfg["experimental"]["workbook"]
    raw = load_raw_experimental_workbook(raw_path)
    clean = prepare_database(
        raw,
        assume_unitless_affinity_um=bool(
            pipeline_cfg["experimental"].get("assume_unitless_affinity_um", False)
        ),
    )

    versions = _read_table(construct_paths["versions"])
    residue_mapping = _read_table(construct_paths["residue_mapping"])
    physchem = _read_table(construct_paths["physchem"])

    clean = attach_resolved_versions(
        clean,
        versions,
        version_aliases=pipeline_cfg.get("version_aliases") or {},
    )
    clean = _attach_mutation_codes(clean)
    clean = attach_canonical_edits(clean, residue_mapping)
    clean = fitness_transform(
        clean,
        weights=fitness_cfg["weights"],
        min_components=int(fitness_cfg.get("min_components", 2)),
        policies=fitness_cfg.get("observations"),
    )

    master_path = processed_dir / "experiment_master.parquet"
    parquet_df = _dataframe_for_parquet(clean)
    # Also keep a pickle with full Python objects for in-pipeline reuse.
    pickle_path = processed_dir / "experiment_master.pkl"
    clean.to_pickle(pickle_path)
    parquet_df.to_parquet(master_path, index=False)

    # Splits over constructs with usable fitness
    eligible = clean.loc[clean["fitness"].notna(), "construct_id"].astype(str).tolist()
    split_cfg = pipeline_cfg.get("splits", {})
    strategy = split_cfg.get("strategy", "random_holdout")
    seed = int(pipeline_cfg.get("random_seed", 42))
    if strategy == "leave_one_construct_out":
        splits = generate_leave_one_out_splits(eligible, random_seed=seed)
    else:
        splits = generate_random_holdout_splits(
            eligible,
            n_splits=int(split_cfg.get("n_splits", 10)),
            test_fraction=float(split_cfg.get("test_fraction", 0.2)),
            test_size=split_cfg.get("test_size"),
            random_seed=seed,
        )
    split_paths = write_splits(splits, splits_dir)

    gate = run_stage0_gates(
        clean,
        versions=versions,
        residue_mapping=residue_mapping,
        splits=splits,
        fitness_cfg=fitness_cfg,
        required_controls=fitness_cfg.get("required_control_mutations", ["Q324R", "A355R"]),
    )
    gate_report = write_stage0_report(
        clean, gate, splits=splits, repo_root=root
    )

    input_hashes = {
        "experimental_workbook": {
            "path": str(raw_path.relative_to(root)),
            "sha256": sha256_file(raw_path),
        },
        "versions": {
            "path": str(construct_paths["versions"].relative_to(root)),
            "sha256": sha256_file(construct_paths["versions"]),
        },
        "residue_mapping": {
            "path": str(construct_paths["residue_mapping"].relative_to(root)),
            "sha256": sha256_file(construct_paths["residue_mapping"]),
        },
        "physchem": {
            "path": str(construct_paths["physchem"].relative_to(root)),
            "sha256": sha256_file(construct_paths["physchem"]),
        },
    }
    output_hashes = {
        "experiment_master": {
            "path": str(master_path.relative_to(root)),
            "sha256": sha256_file(master_path),
            "n_rows": int(len(clean)),
            "n_with_fitness": int(clean["fitness"].notna().sum()),
        },
        "experiment_master_pickle": {
            "path": str(pickle_path.relative_to(root)),
            "sha256": sha256_file(pickle_path),
        },
        "splits": {
            "dir": str(splits_dir.relative_to(root)),
            "files": [p.name for p in split_paths],
        },
        "gate_report": gate_report,
    }
    manifest_path = manifests_dir / "stage0_manifest.json"
    write_manifest(
        manifest_path,
        stage="stage0_ground_truth",
        inputs=input_hashes,
        parameters={
            "pipeline": pipeline_cfg,
            "fitness": fitness_cfg,
            "n_physchem_residues": int(len(physchem)),
        },
        outputs=output_hashes,
        random_seed=seed,
        gate=gate,
        notes="Authoritative experimental+fitness dataset for paired Stage 3/6 comparisons.",
    )

    if not gate.get("passed", False):
        raise RuntimeError(f"Stage 0 gate failed: {gate}")

    meta = {
        "master_path": master_path,
        "splits_dir": splits_dir,
        "manifest_path": manifest_path,
        "gate": gate,
        "gate_report": gate_report,
        "n_rows": len(clean),
    }
    return clean, meta


def main() -> None:
    """Run Stage-0 master build from default configs and print summary paths.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints row counts and artifact paths to stdout.
    """
    master, meta = build_experiment_master()
    print(f"experiment_master rows: {meta['n_rows']}")
    print(f"wrote: {meta['master_path']}")
    print(f"splits: {meta['splits_dir']}")
    print(f"manifest: {meta['manifest_path']}")
    print(f"gate passed: {meta['gate']['passed']}")
    report = meta.get("gate_report") or {}
    if report.get("directory"):
        print(f"gate report: {report['directory']}")


if __name__ == "__main__":
    main()
