"""Build feature vectors; fit preprocessing inside each training split."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.physicochemical import load_aa_properties
from biosensor_priors.stage4_search.landscape import build_landscape_view, parse_mutation_list

EncodingMode = Literal["mutation_bag", "onehot", "georgiev", "hybrid"]

PHYSICS_FEATURE_COLUMNS = (
    "rif_ac",
    "rif_prop",
    "delta_rif_sel",
    "rpx",
)

STRUCT_CONF_COLUMN = "structural_confidence"

PHYSCHEM_DELTA_KEYS = (
    "hydrophobicity_KD",
    "side_chain_volume_A3",
    "polarity_Grantham",
    "charge",
    "polar",
    "aromatic",
    "positive",
    "negative",
    "branched",
)

# Georgiev-like per-residue physicochemical vector (AAIndex-style stand-in).
# Paper Georgiev is 19-D PCA of AAIndex; we use a fixed 19-D property vector
# from our curated AA table (continuous + binary descriptors). [https://doi.org/10.1089/cmb.2008.0173]
GEORGIEV_KEYS = (
    "hydrophobicity_KD",
    "side_chain_volume_A3",
    "molecular_weight_Da",
    "polarity_Grantham",
    "charge",
    "polar",
    "aromatic",
    "hbond_donor",
    "hbond_acceptor",
    "positive",
    "negative",
    "branched",
    "sulfur_containing",
    "Gly",
    "Pro",
    "hydrophobicity_KD_z",
    "side_chain_volume_A3_z",
    "molecular_weight_Da_z",
    "polarity_Grantham_z",
)


def _mutation_delta_vector(muts: list[tuple[str, int, str]], aa_props: dict) -> np.ndarray:
    if not muts:
        return np.zeros(len(PHYSCHEM_DELTA_KEYS) + 1, dtype=float)
    deltas = []
    for aa_from, _, aa_to in muts:
        if aa_from not in aa_props or aa_to not in aa_props:
            continue
        d = [float(aa_props[aa_to][k]) - float(aa_props[aa_from][k]) for k in PHYSCHEM_DELTA_KEYS]
        deltas.append(d)
    if not deltas:
        return np.zeros(len(PHYSCHEM_DELTA_KEYS) + 1, dtype=float)
    mean_delta = np.mean(np.asarray(deltas, dtype=float), axis=0)
    return np.concatenate([mean_delta, [float(len(muts))]])


def _aa_georgiev_vector(aa: str, aa_props: dict) -> np.ndarray:
    """19-D physicochemical vector; z-keys fall back to raw values if absent."""
    if aa not in aa_props:
        return np.zeros(19, dtype=float)
    props = aa_props[aa]
    vals = []
    for k in GEORGIEV_KEYS:
        if k in props:
            vals.append(float(props[k]))
        elif k.endswith("_z") and k[:-2] in props:
            vals.append(float(props[k[:-2]]))
        else:
            vals.append(0.0)
    return np.asarray(vals, dtype=float)


@dataclass
class FeatureBuilder:
    """
    Fit-transform feature pipeline.

    Standardization statistics are fit on **training rows only**.

    Encoding modes (BO-EVO paper + pipeline extensions):
      * ``onehot`` — per variable site, classical 20-AA one-hot (N×20)
      * ``georgiev`` — per-site 19-D physicochemical vector
      * ``hybrid`` — onehot + georgiev
      * ``mutation_bag`` — mutation physchem deltas + mutation-code one-hots
    """

    encoding: EncodingMode = "hybrid"
    include_physics: bool = True
    include_struct_confidence: bool = True
    onehot_mutations: bool = True
    mutation_vocab: list[str] = field(default_factory=list)
    site_positions: list[int] = field(default_factory=list)
    means_: np.ndarray | None = None
    stds_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)
    has_physics_: bool = False
    aa_props: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.aa_props:
            self.aa_props = load_aa_properties()

    def _sequences_and_sites(self, df: pd.DataFrame) -> tuple[list[str], list[int]]:
        view = build_landscape_view(df)
        # Prefer fitted sites when transforming
        if self.site_positions:
            # Rebuild sequences on fitted site order using mutation lists
            sequences = []
            wt = {}
            for muts in (parse_mutation_list(row) for _, row in df.iterrows()):
                for aa_from, pos, _ in muts:
                    wt.setdefault(pos, aa_from)
            for _, row in df.iterrows():
                muts = parse_mutation_list(row)
                state = {p: wt.get(p, "X") for p in self.site_positions}
                # Fill missing WT from fitted training if needed
                for aa_from, pos, aa_to in muts:
                    if pos in state:
                        state[pos] = aa_to
                        wt.setdefault(pos, aa_from)
                sequences.append("".join(state[p] for p in self.site_positions))
            return sequences, self.site_positions
        return view.sequences, view.site_positions

    def _encode_sequence_modes(self, sequences: list[str]) -> tuple[np.ndarray, list[str]]:
        n = len(sequences)
        n_sites = len(self.site_positions) if self.site_positions else (len(sequences[0]) if sequences else 0)
        parts: list[np.ndarray] = []
        names: list[str] = []

        if self.encoding in {"onehot", "hybrid"}:
            # Paper Onehot: N × 20 classical amino acids
            aa_order = list("ACDEFGHIKLMNPQRSTVWY")
            block = np.zeros((n, n_sites * 20), dtype=float)
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    if j >= n_sites:
                        break
                    if aa in aa_order:
                        block[i, j * 20 + aa_order.index(aa)] = 1.0
            parts.append(block)
            names.extend([f"oh_s{s}_{aa}" for s in range(n_sites) for aa in aa_order])

        if self.encoding in {"georgiev", "hybrid"}:
            gdim = len(GEORGIEV_KEYS)
            block = np.zeros((n, n_sites * gdim), dtype=float)
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    if j >= n_sites:
                        break
                    block[i, j * gdim : (j + 1) * gdim] = _aa_georgiev_vector(aa, self.aa_props)
            parts.append(block)
            names.extend([f"geo_s{s}_{k}" for s in range(n_sites) for k in GEORGIEV_KEYS])

        if not parts:
            return np.zeros((n, 0), dtype=float), []
        return np.concatenate(parts, axis=1), names

    def _encode_mutation_bag(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        rows = []
        base_names = [f"delta_{k}" for k in PHYSCHEM_DELTA_KEYS] + ["n_mutations"]
        mut_names = [f"mut_{code}" for code in self.mutation_vocab] if self.onehot_mutations else []
        names = base_names + mut_names
        for _, row in df.iterrows():
            muts = parse_mutation_list(row)
            vec = list(_mutation_delta_vector(muts, self.aa_props))
            if self.onehot_mutations:
                codes = {f"{a}{p}{b}" for a, p, b in muts}
                vec.extend(1.0 if code in codes else 0.0 for code in self.mutation_vocab)
            rows.append(vec)
        return np.asarray(rows, dtype=float), names

    def _append_physics_confidence(
        self, df: pd.DataFrame, X: np.ndarray, names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        extras = []
        extra_names = []
        physics_present = any(c in df.columns for c in PHYSICS_FEATURE_COLUMNS)
        self.has_physics_ = bool(physics_present)

        if self.include_physics:
            for col in PHYSICS_FEATURE_COLUMNS:
                if col in df.columns:
                    extras.append(
                        pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    )
                else:
                    extras.append(np.zeros(len(df), dtype=float))
                extra_names.append(col)

        if self.include_struct_confidence:
            if STRUCT_CONF_COLUMN in df.columns:
                extras.append(
                    pd.to_numeric(df[STRUCT_CONF_COLUMN], errors="coerce").fillna(1.0).to_numpy(
                        dtype=float
                    )
                )
            else:
                extras.append(np.ones(len(df), dtype=float))
            extra_names.append(STRUCT_CONF_COLUMN)

        if extras:
            X = np.column_stack([X] + extras) if X.size else np.column_stack(extras)
            names = names + extra_names
        return X, names

    def _raw_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if self.encoding == "mutation_bag":
            X, names = self._encode_mutation_bag(df)
        else:
            sequences, sites = self._sequences_and_sites(df)
            if not self.site_positions:
                self.site_positions = list(sites)
            X, names = self._encode_sequence_modes(sequences)
        return self._append_physics_confidence(df, X, names)

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        if self.encoding == "mutation_bag":
            vocab: set[str] = set()
            if self.onehot_mutations:
                for _, row in df.iterrows():
                    for a, p, b in parse_mutation_list(row):
                        vocab.add(f"{a}{p}{b}")
            self.mutation_vocab = sorted(vocab)
            self.site_positions = []
        else:
            view = build_landscape_view(df)
            self.site_positions = list(view.site_positions)
            self.mutation_vocab = []

        X, names = self._raw_matrix(df)
        self.feature_names_ = names
        if X.size == 0:
            self.means_ = np.zeros(0)
            self.stds_ = np.zeros(0)
            return self
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=0)
        self.stds_ = np.where(self.stds_ < 1e-12, 1.0, self.stds_)
        self.means_ = np.nan_to_num(self.means_, nan=0.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        X, _ = self._raw_matrix(df)
        X = np.nan_to_num(X, nan=0.0)
        if X.shape[1] != len(self.means_):
            # Pad / trim for safety if site sets differ
            out = np.zeros((len(df), len(self.means_)), dtype=float)
            cols = min(X.shape[1], len(self.means_))
            out[:, :cols] = X[:, :cols]
            X = out
        return (X - self.means_) / self.stds_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def physics_block(self, X: np.ndarray) -> np.ndarray:
        if not self.include_physics or not self.feature_names_:
            return np.zeros((len(X), 0), dtype=float)
        idx = [i for i, n in enumerate(self.feature_names_) if n in PHYSICS_FEATURE_COLUMNS]
        if not idx:
            return np.zeros((len(X), 0), dtype=float)
        return X[:, idx]

    def confidence_vector(self, X: np.ndarray) -> np.ndarray:
        if STRUCT_CONF_COLUMN not in self.feature_names_:
            return np.ones(len(X), dtype=float)
        i = self.feature_names_.index(STRUCT_CONF_COLUMN)
        raw = X[:, i] * self.stds_[i] + self.means_[i]
        return np.clip(raw, 0.0, 1.0)
