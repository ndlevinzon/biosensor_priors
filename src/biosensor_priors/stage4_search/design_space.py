"""Generate the constrained candidate universe from mutable positions."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import pandas as pd

from biosensor_priors.stage0_ground_truth.edits import (
    DEFAULT_COSTS,
    compose_canonical,
    format_edit,
    mutation_cost,
    parse_edit_code,
    scaffold_edits,
)
from biosensor_priors.stage0_ground_truth.physicochemical import load_aa_properties

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
_DELTA_KEYS = (
    "hydrophobicity_KD",
    "side_chain_volume_A3",
    "polarity_Grantham",
    "charge",
)


def canonical_to_version_positions(
    residue_mapping: pd.DataFrame,
    *,
    version: str,
    canonical_positions: Iterable[int | str],
    missing_ok: bool = False,
) -> dict[int, int]:
    """Map canonical residue keys to version-local positions for one background.

    Parameters
    ----------
    residue_mapping : pd.DataFrame
        Residue mapping table with ``Version``, ``Canonical_key``, and
        ``Version_position`` columns.
    version : str
        Background version identifier.
    canonical_positions : Iterable of int or str
        Canonical positions to resolve.
    missing_ok : bool, optional
        When True, skip canonical sites that are absent on this version
        instead of raising (default False).

    Returns
    -------
    dict of int to int
        Mapping from canonical position to version-local index.

    Raises
    ------
    ValueError
        If any requested canonical position is missing and ``missing_ok``
        is False.
    """
    want = {str(p) for p in canonical_positions}
    sub = residue_mapping[
        (residue_mapping["Version"].astype(str) == str(version))
        & (residue_mapping["Canonical_key"].astype(str).isin(want))
        & (residue_mapping["Version_position"].notna())
    ]
    mapping: dict[int, int] = {}
    for _, row in sub.iterrows():
        mapping[int(str(row["Canonical_key"]).split("i")[0])] = int(
            row["Version_position"]
        )
    missing = sorted({int(p) for p in want} - set(mapping))
    if missing and not missing_ok:
        raise ValueError(
            f"Canonical positions {missing} not mapped for version {version}."
        )
    return mapping


def _empty_deltas() -> dict[str, float]:
    return {f"delta_{k}": 0.0 for k in _DELTA_KEYS}


def _sub_deltas(
    mutations: list[str],
    aa_combo: Iterable[str],
    aa_props: dict,
) -> dict[str, float]:
    deltas = _empty_deltas()
    for mut, aa_to in zip(mutations, aa_combo, strict=True):
        aa_from = mut[0]
        if aa_from not in aa_props or aa_to not in aa_props:
            continue
        for k in _DELTA_KEYS:
            deltas[f"delta_{k}"] += float(aa_props[aa_to][k]) - float(
                aa_props[aa_from][k]
            )
    return deltas


def _iter_sub_combos(
    positions: list[int],
    *,
    wt: dict[int, str],
    allowed: list[str],
    exclude_wt: bool,
    n_mut: int,
):
    for pos_combo in itertools.combinations(positions, n_mut):
        aa_choices = []
        empty = False
        for pos in pos_combo:
            opts = [aa for aa in allowed if (not exclude_wt) or aa != wt[pos]]
            if not opts:
                empty = True
                break
            aa_choices.append(opts)
        if empty:
            continue
        for aa_combo in itertools.product(*aa_choices):
            yield pos_combo, aa_combo


def generate_design_space(
    *,
    parent_version: str,
    parent_sequence: str,
    mutable_positions: Iterable[int],
    allowed_amino_acids: Iterable[str] | None = None,
    max_mutations: int = 2,
    exclude_wt: bool = True,
    position_labels: dict[int, int] | None = None,
    indel_events: Iterable[str] | None = None,
    cost_cfg: dict[str, float] | None = None,
    scaffold_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Enumerate single and multi-site mutants up to ``max_mutations``.

    ``mutable_positions`` are version-local indices into ``parent_sequence``.
    Optional ``position_labels`` maps version-local indices to canonical
    positions for mutation IDs. Insertion/deletion events (``ins104``,
    ``insNterm``, ``delNterm``) occupy a mutation slot and count toward
    ``max_mutations``.

    Parameters
    ----------
    parent_version : str
        Background version label for generated construct IDs.
    parent_sequence : str
        Parent amino-acid sequence (1-indexed positions in
        ``mutable_positions``).
    mutable_positions : Iterable of int
        Version-local indices that may be mutated.
    allowed_amino_acids : Iterable of str or None, optional
        Allowed destination amino acids (default standard 20).
    max_mutations : int, optional
        Maximum number of simultaneous edits per construct (default 2).
    exclude_wt : bool, optional
        When True, skip mutations that restore the parent amino acid
        (default True).
    position_labels : dict of int to int or None, optional
        Version-local to canonical position labels for mutation codes.
    indel_events : Iterable of str or None, optional
        Indel codes allowed on this parent (count toward ``max_mutations``).
    cost_cfg : dict or None, optional
        Per-edit cost table forwarded to :func:`mutation_cost`.
    scaffold_codes : Iterable of str or None, optional
        Parent-vs-V1.0 edits prepended into ``canonical_edit_codes``.

    Returns
    -------
    pd.DataFrame
        Design-space table with mutation codes, physicochemical deltas, and
        placeholder physics columns (NaN until Stage 2/1 priors are joined).

    Raises
    ------
    ValueError
        If any mutable position falls outside the parent sequence length.
    """
    allowed = [a.upper() for a in (allowed_amino_acids or STANDARD_AA)]
    positions = sorted({int(p) for p in mutable_positions})
    wt = {i + 1: aa for i, aa in enumerate(parent_sequence)}
    missing = [p for p in positions if p not in wt]
    if missing:
        raise ValueError(
            f"Mutable positions outside parent sequence length: {missing}"
        )

    labels = position_labels or {p: p for p in positions}
    indels = [str(c) for c in (indel_events or [])]
    scaffold = [str(c) for c in (scaffold_codes or [])]
    costs = {**DEFAULT_COSTS, **(cost_cfg or {})}
    aa_props = load_aa_properties()
    rows: list[dict] = []

    def _append(
        sub_codes: list[str],
        sub_positions: list[int],
        version_positions: list[int],
        indel_codes: list[str],
        deltas: dict[str, float],
    ) -> None:
        proposed = [*sub_codes, *indel_codes]
        if not proposed:
            return
        n_ins = 0
        n_del = 0
        for code in proposed:
            parsed = parse_edit_code(code)
            if parsed is None:
                continue
            if parsed[0] in {"+", "I"}:
                n_ins += 1
            elif parsed[0] in {"-", "D"}:
                n_del += 1
        cand_id = f"{parent_version}|" + "/".join(proposed)
        rows.append(
            {
                "candidate_id": cand_id,
                "construct_id": cand_id,
                "parent_version": parent_version,
                "version": parent_version,
                "mutations": proposed,
                "mutation_codes": proposed,
                "scaffold_edits": scaffold,
                "canonical_edit_codes": compose_canonical(scaffold, proposed),
                "canonical_positions": sub_positions,
                "version_positions": version_positions,
                "n_mutations": len(proposed),
                "n_insertions": n_ins,
                "n_deletions": n_del,
                "mutation_cost": mutation_cost(proposed, costs=costs),
                "rif_ac": float("nan"),
                "rif_prop": float("nan"),
                "delta_rif_sel": float("nan"),
                "structural_confidence": float("nan"),
                **deltas,
            }
        )

    max_n = int(max_mutations)
    for n_mut in range(1, max_n + 1):
        if not positions:
            break
        for pos_combo, aa_combo in _iter_sub_combos(
            positions,
            wt=wt,
            allowed=allowed,
            exclude_wt=exclude_wt,
            n_mut=n_mut,
        ):
            sub_codes: list[str] = []
            canon_pos: list[int] = []
            for pos, aa_to in zip(pos_combo, aa_combo, strict=True):
                canon = int(labels.get(pos, pos))
                sub_codes.append(f"{wt[pos]}{canon}{aa_to}")
                canon_pos.append(canon)
            _append(
                sub_codes,
                canon_pos,
                list(pos_combo),
                [],
                _sub_deltas(sub_codes, aa_combo, aa_props),
            )

    for k_indel in range(1, min(len(indels), max_n) + 1):
        for indel_combo in itertools.combinations(indels, k_indel):
            indel_list = list(indel_combo)
            _append([], [], [], indel_list, _empty_deltas())
            remaining = max_n - k_indel
            for n_mut in range(1, remaining + 1):
                if not positions:
                    break
                for pos_combo, aa_combo in _iter_sub_combos(
                    positions,
                    wt=wt,
                    allowed=allowed,
                    exclude_wt=exclude_wt,
                    n_mut=n_mut,
                ):
                    sub_codes = []
                    canon_pos = []
                    for pos, aa_to in zip(pos_combo, aa_combo, strict=True):
                        canon = int(labels.get(pos, pos))
                        sub_codes.append(f"{wt[pos]}{canon}{aa_to}")
                        canon_pos.append(canon)
                    _append(
                        sub_codes,
                        canon_pos,
                        list(pos_combo),
                        indel_list,
                        _sub_deltas(sub_codes, aa_combo, aa_props),
                    )

    return pd.DataFrame(rows)


def _indel_codes_for_parent(
    events: Iterable[Any],
    *,
    parent: str,
    scaffold_codes: list[str],
) -> list[str]:
    out: list[str] = []
    for ev in events or []:
        if isinstance(ev, str):
            code = ev
            parents = None
        else:
            code = str(ev.get("code") or ev.get("id") or "")
            parents = ev.get("on_parents") or ev.get("parents")
        if not code:
            continue
        if parents is not None and str(parent) not in {str(p) for p in parents}:
            continue
        if code == "insNterm" and "insNterm" in scaffold_codes:
            continue
        if code == "delNterm" and "insNterm" not in scaffold_codes:
            continue
        if code != "delNterm" and code in scaffold_codes:
            continue
        out.append(code)
    return out


def design_space_from_config(
    versions: pd.DataFrame,
    *,
    parent_version: str,
    mutable_positions: list[int],
    allowed_amino_acids: list[str],
    max_mutations: int,
    residue_mapping: pd.DataFrame | None = None,
    positions_are_canonical: bool = True,
    indel_events: Iterable[Any] | None = None,
    cost_cfg: dict[str, float] | None = None,
    missing_ok: bool = True,
) -> pd.DataFrame:
    """Build a combinatorial design space from pipeline configuration inputs.

    Parameters
    ----------
    versions : pd.DataFrame
        Version table containing ``Version`` and ``Sequence_clean`` columns.
    parent_version : str
        Background version to mutate.
    mutable_positions : list of int
        Canonical or version-local mutable positions depending on
        ``positions_are_canonical``.
    allowed_amino_acids : list of str
        Allowed destination amino acids at each mutable site.
    max_mutations : int
        Maximum simultaneous mutations per construct.
    residue_mapping : pd.DataFrame or None, optional
        Required when ``positions_are_canonical`` is True.
    positions_are_canonical : bool, optional
        When True, resolve ``mutable_positions`` via ``residue_mapping``
        (default True).
    indel_events : Iterable or None, optional
        Insertion/deletion events from ``fitness.yaml`` ``design.indel_events``.
    cost_cfg : dict or None, optional
        Mutation-cost table.
    missing_ok : bool, optional
        Skip canonical sites unmapped on this parent (default True).

    Returns
    -------
    pd.DataFrame
        Combinatorial design-space table from :func:`generate_design_space`.

    Raises
    ------
    ValueError
        If the parent version is missing or canonical mapping is required
        but absent.
    """
    row = versions.loc[versions["Version"].astype(str) == parent_version]
    if row.empty:
        raise ValueError(
            f"Parent version {parent_version} not found in versions table."
        )
    seq = str(row.iloc[0]["Sequence_clean"])
    mapping = residue_mapping if residue_mapping is not None else pd.DataFrame()
    scaffold = [format_edit(*e) for e in scaffold_edits(mapping, parent_version)]

    if positions_are_canonical:
        if residue_mapping is None:
            raise ValueError(
                "residue_mapping required when positions_are_canonical=True"
            )
        canon_to_version = canonical_to_version_positions(
            residue_mapping,
            version=parent_version,
            canonical_positions=mutable_positions,
            missing_ok=missing_ok,
        )
        version_positions = list(canon_to_version.values())
        labels = {v: c for c, v in canon_to_version.items()}
    else:
        version_positions = list(mutable_positions)
        labels = {p: p for p in version_positions}

    indels = _indel_codes_for_parent(
        indel_events or [],
        parent=parent_version,
        scaffold_codes=scaffold,
    )
    return generate_design_space(
        parent_version=parent_version,
        parent_sequence=seq,
        mutable_positions=version_positions,
        allowed_amino_acids=allowed_amino_acids,
        max_mutations=max_mutations,
        position_labels=labels,
        indel_events=indels,
        cost_cfg=cost_cfg,
        scaffold_codes=scaffold,
    )


def build_design_library(
    versions: pd.DataFrame,
    residue_mapping: pd.DataFrame,
    fitness_cfg: dict[str, Any],
    *,
    default_parent: str | None = None,
) -> pd.DataFrame:
    """Enumerate the configured multi-parent design library.

    Parameters
    ----------
    versions : pd.DataFrame
        Version table with ``Version`` and ``Sequence_clean``.
    residue_mapping : pd.DataFrame
        Canonical residue mapping.
    fitness_cfg : dict
        Parsed ``fitness.yaml`` (uses ``design`` and ``mutation_cost``).
    default_parent : str or None, optional
        Fallback parent when ``design.parents`` is empty.

    Returns
    -------
    pd.DataFrame
        Concatenated design space over all configured parents.
    """
    design_cfg = fitness_cfg.get("design", {}) or {}
    parents = [str(p) for p in (design_cfg.get("parents") or [])]
    if not parents:
        parents = [str(default_parent or "V2.4")]
    positions = [int(p) for p in (design_cfg.get("allowed_mutable_positions") or [])]
    allowed = list(design_cfg.get("allowed_amino_acids") or STANDARD_AA)
    max_mut = min(int(design_cfg.get("maximum_mutations_per_construct", 2)), 2)
    cost_cfg = dict(fitness_cfg.get("mutation_cost") or {})
    frames: list[pd.DataFrame] = []
    available = set(versions["Version"].astype(str))
    for parent in parents:
        if parent not in available:
            continue
        frame = design_space_from_config(
            versions,
            parent_version=parent,
            mutable_positions=positions,
            allowed_amino_acids=allowed,
            max_mutations=max_mut,
            residue_mapping=residue_mapping,
            positions_are_canonical=True,
            indel_events=design_cfg.get("indel_events") or [],
            cost_cfg=cost_cfg,
            missing_ok=True,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
