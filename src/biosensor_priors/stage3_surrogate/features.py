"""Build feature vectors; fit preprocessing inside each training split."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from biosensor_priors.stage0_ground_truth.edits import (
    format_edit,
    parse_mutation_list,
)
from biosensor_priors.stage0_ground_truth.physicochemical import (
    build_aa_property_table,
)
from biosensor_priors.stage4_search.landscape import (
    build_landscape_view,
)

EncodingMode = Literal["mutation_bag", "onehot", "georgiev", "hybrid"]

PHYSICS_FEATURE_COLUMNS = (
    "rif_ac",
    "rif_prop",
    "delta_rif_sel",
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

BINARY_PHYSCHEM_KEYS = frozenset(
    {
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
    }
)
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
    """Compute mean physicochemical delta vector across a mutation list.

    Parameters
    ----------
    muts : list of tuple
        Parsed mutations as ``(wt_aa, position, mut_aa)`` triples.
    aa_props : dict
        Amino acid property lookup table.

    Returns
    -------
    numpy.ndarray
        Mean delta features plus mutation count as final element.
    """
    if not muts:
        return np.zeros(len(PHYSCHEM_DELTA_KEYS) + 1, dtype=float)
    deltas = []
    for aa_from, _, aa_to in muts:
        if aa_from not in aa_props or aa_to not in aa_props:
            continue
        d = [float(aa_props[aa_to][k]) - float(aa_props[aa_from][k]) for k in PHYSCHEM_DELTA_KEYS]
        deltas.append(d)
    if not deltas:
        zeros = np.zeros(len(PHYSCHEM_DELTA_KEYS), dtype=float)
        return np.concatenate([zeros, [float(len(muts))]])
    mean_delta = np.mean(np.asarray(deltas, dtype=float), axis=0)
    return np.concatenate([mean_delta, [float(len(muts))]])


def _aa_lookup_with_z() -> dict[str, dict]:
    """Amino-acid properties including continuous z-scores from the AA table."""
    table = build_aa_property_table(create_zscores=True)
    out: dict[str, dict] = {}
    for _, row in table.iterrows():
        rec = row.to_dict()
        aa = str(rec.pop("AA"))
        out[aa] = rec
    return out


def is_binary_feature_name(name: str) -> bool:
    """True for mutation/one-hot bits and 0/1 (or -1/0/1) physchem flags."""
    if name.startswith("mut_") or name.startswith("oh_"):
        return True
    if name.startswith("delta_") and name[len("delta_") :] in BINARY_PHYSCHEM_KEYS:
        return True
    if name.startswith("geo_"):
        return any(name.endswith(f"_{key}") for key in BINARY_PHYSCHEM_KEYS)
    return False


def _aa_georgiev_vector(aa: str, aa_props: dict) -> np.ndarray:
    """Build 19-D physicochemical vector; z-keys fall back to raw values.

    Parameters
    ----------
    aa : str
        Single-letter amino acid code.
    aa_props : dict
        Amino acid property lookup table.

    Returns
    -------
    numpy.ndarray
        19-dimensional Georgiev-like feature vector.
    """
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
    binary_names_: list[str] = field(default_factory=list)
    physics_fill_: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load amino acid properties when not provided at construction."""
        if not self.aa_props:
            self.aa_props = _aa_lookup_with_z()

    def _sequences_and_sites(self, df: pd.DataFrame) -> tuple[list[str], list[int]]:
        """Extract aligned sequences and site positions from construct rows.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table with mutation lists.

        Returns
        -------
        sequences : list of str
            Per-row sequence strings over site positions.
        site_positions : list of int
            Canonical site order used for encoding.
        """
        view = build_landscape_view(df)
        # Prefer fitted sites when transforming
        if self.site_positions:
            # Rebuild sequences on fitted site order using mutation lists
            sequences = []
            wt = {}
            for muts in (parse_mutation_list(row) for _, row in df.iterrows()):
                for aa_from, pos, aa_to in muts:
                    if aa_from in {"+", "I"}:
                        wt.setdefault(pos, "-")
                    elif aa_from in {"-", "D"}:
                        wt.setdefault(pos, aa_to if aa_to.isalpha() else "X")
                    else:
                        wt.setdefault(pos, aa_from)
            for _, row in df.iterrows():
                muts = parse_mutation_list(row)
                state = {p: wt.get(p, "X") for p in self.site_positions}
                for aa_from, pos, aa_to in muts:
                    if pos not in state:
                        continue
                    if aa_from in {"+", "I"}:
                        state[pos] = "I"
                        wt.setdefault(pos, "-")
                    elif aa_from in {"-", "D"}:
                        state[pos] = "-"
                        wt.setdefault(pos, aa_to if aa_to.isalpha() else "X")
                    else:
                        state[pos] = aa_to
                        wt.setdefault(pos, aa_from)
                sequences.append("".join(state[p] for p in self.site_positions))
            return sequences, self.site_positions
        return view.sequences, view.site_positions

    def _encode_sequence_modes(self, sequences: list[str]) -> tuple[np.ndarray, list[str]]:
        """Encode sequences using one-hot and/or Georgiev feature blocks.

        Parameters
        ----------
        sequences : list of str
            Per-construct sequence strings over fitted sites.

        Returns
        -------
        X : numpy.ndarray
            Encoded feature matrix.
        names : list of str
            Feature names aligned with columns of ``X``.
        """
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
        """Encode mutation-bag features with physchem deltas and optional one-hots.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table with mutation lists.

        Returns
        -------
        X : numpy.ndarray
            Encoded feature matrix.
        names : list of str
            Feature names aligned with columns of ``X``.
        """
        rows = []
        base_names = (
            [f"delta_{k}" for k in PHYSCHEM_DELTA_KEYS]
            + ["n_mutations", "n_insertions", "n_deletions"]
        )
        mut_names = [f"mut_{code}" for code in self.mutation_vocab] if self.onehot_mutations else []
        names = base_names + mut_names
        for _, row in df.iterrows():
            muts = parse_mutation_list(row)
            vec = list(_mutation_delta_vector(muts, self.aa_props))
            n_ins = sum(1 for a, _, _ in muts if a in {"+", "I"})
            n_del = sum(1 for a, _, _ in muts if a in {"-", "D"})
            vec.extend([float(n_ins), float(n_del)])
            if self.onehot_mutations:
                codes = {format_edit(a, p, b) for a, p, b in muts}
                vec.extend(1.0 if code in codes else 0.0 for code in self.mutation_vocab)
            rows.append(vec)
        return np.asarray(rows, dtype=float), names

    def _append_physics_confidence(
        self, df: pd.DataFrame, X: np.ndarray, names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Append physics scores and structural confidence columns to features.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table with optional physics columns.
        X : numpy.ndarray
            Base encoded feature matrix.
        names : list of str
            Base feature names.

        Returns
        -------
        X : numpy.ndarray
            Feature matrix with physics/confidence columns appended.
        names : list of str
            Extended feature names.
        """
        extras = []
        extra_names = []
        physics_present = False
        if self.include_physics:
            for col in PHYSICS_FEATURE_COLUMNS:
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce")
                    extras.append(series.to_numpy(dtype=float))
                    if bool(np.isfinite(series.to_numpy(dtype=float)).any()):
                        physics_present = True
                else:
                    extras.append(np.full(len(df), np.nan, dtype=float))
                extra_names.append(col)
        self.has_physics_ = bool(physics_present)

        if self.include_struct_confidence:
            if STRUCT_CONF_COLUMN in df.columns:
                extras.append(
                    pd.to_numeric(df[STRUCT_CONF_COLUMN], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
            else:
                extras.append(np.full(len(df), np.nan, dtype=float))
            extra_names.append(STRUCT_CONF_COLUMN)

        if extras:
            X = np.column_stack([X] + extras) if X.size else np.column_stack(extras)
            names = names + extra_names
        return X, names

    def _raw_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build unstandardized feature matrix for a construct table.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table to encode.

        Returns
        -------
        X : numpy.ndarray
            Raw feature matrix before standardization.
        names : list of str
            Feature names aligned with columns of ``X``.
        """
        if self.encoding == "mutation_bag":
            X, names = self._encode_mutation_bag(df)
        else:
            sequences, sites = self._sequences_and_sites(df)
            if not self.site_positions:
                self.site_positions = list(sites)
            X, names = self._encode_sequence_modes(sequences)
        return self._append_physics_confidence(df, X, names)

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Fit vocabulary, site order, and standardization on training rows.

        Parameters
        ----------
        df : pandas.DataFrame
            Training construct table.

        Returns
        -------
        FeatureBuilder
            Fitted builder (``self``).
        """
        if self.encoding == "mutation_bag":
            vocab: set[str] = set()
            if self.onehot_mutations:
                for _, row in df.iterrows():
                    for a, p, b in parse_mutation_list(row):
                        vocab.add(format_edit(a, p, b))
            self.mutation_vocab = sorted(vocab)
            self.site_positions = []
        else:
            view = build_landscape_view(df)
            self.site_positions = list(view.site_positions)
            self.mutation_vocab = []

        X, names = self._raw_matrix(df)
        self.feature_names_ = names
        self.binary_names_ = [n for n in names if is_binary_feature_name(n)]
        self.physics_fill_ = {}
        if X.size == 0:
            self.means_ = np.zeros(0)
            self.stds_ = np.zeros(0)
            return self
        for i, name in enumerate(names):
            if name not in PHYSICS_FEATURE_COLUMNS:
                continue
            col = X[:, i]
            finite = np.isfinite(col)
            self.physics_fill_[name] = (
                float(np.mean(col[finite])) if finite.any() else float("nan")
            )
        filled = self._fill_physics_confidence(X)
        self.means_ = np.nanmean(filled, axis=0)
        self.stds_ = np.nanstd(filled, axis=0, ddof=0)
        self.stds_ = np.where(self.stds_ < 1e-12, 1.0, self.stds_)
        self.means_ = np.nan_to_num(self.means_, nan=0.0)
        return self

    def _fill_physics_confidence(self, X: np.ndarray) -> np.ndarray:
        """Impute physics with train means; missing confidence becomes 0 (not 1)."""
        filled = np.array(X, dtype=float, copy=True)
        for i, name in enumerate(self.feature_names_):
            col = filled[:, i]
            if name in PHYSICS_FEATURE_COLUMNS:
                fill = self.physics_fill_.get(name, float("nan"))
                replacement = fill if np.isfinite(fill) else 0.0
                filled[:, i] = np.where(np.isfinite(col), col, replacement)
            elif name == STRUCT_CONF_COLUMN:
                filled[:, i] = np.where(np.isfinite(col), col, 0.0)
        return np.nan_to_num(filled, nan=0.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardize features using statistics fit on training data.

        Parameters
        ----------
        df : pandas.DataFrame
            Construct table to transform.

        Returns
        -------
        numpy.ndarray
            Standardized feature matrix.

        Raises
        ------
        RuntimeError
            When called before :meth:`fit`.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        X, _ = self._raw_matrix(df)
        X = self._fill_physics_confidence(X)
        if X.shape[1] != len(self.means_):
            # Pad / trim for safety if site sets differ
            out = np.zeros((len(df), len(self.means_)), dtype=float)
            cols = min(X.shape[1], len(self.means_))
            out[:, :cols] = X[:, :cols]
            X = out
        binary = set(self.binary_names_)
        scaled = (X - self.means_) / self.stds_
        for i, name in enumerate(self.feature_names_):
            if name in binary:
                scaled[:, i] = X[:, i]
        return scaled

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on ``df`` and return standardized features.

        Parameters
        ----------
        df : pandas.DataFrame
            Training construct table.

        Returns
        -------
        numpy.ndarray
            Standardized feature matrix.
        """
        return self.fit(df).transform(df)

    def physics_block(self, X: np.ndarray) -> np.ndarray:
        """Extract physics feature columns from a standardized matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Standardized feature matrix.

        Returns
        -------
        numpy.ndarray
            Submatrix containing physics columns only.
        """
        if not self.include_physics or not self.feature_names_:
            return np.zeros((len(X), 0), dtype=float)
        idx = [i for i, n in enumerate(self.feature_names_) if n in PHYSICS_FEATURE_COLUMNS]
        if not idx:
            return np.zeros((len(X), 0), dtype=float)
        return X[:, idx]

    def gp_exclude_names(self) -> set[str]:
        """Feature names that belong in μ₀ / σ_eff, not the residual kernel."""
        return set(PHYSICS_FEATURE_COLUMNS) | {STRUCT_CONF_COLUMN}

    def gp_block(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        """Residual-GP features: Hamming binaries first, then physchem.

        Physics and structural confidence are excluded (mean / σ_eff only).

        Parameters
        ----------
        X : numpy.ndarray
            Standardized feature matrix from :meth:`transform`.

        Returns
        -------
        X_gp : numpy.ndarray
            Reordered residual features.
        n_hamming : int
            Number of leading Hamming (mutation / one-hot) columns.
        """
        if not self.feature_names_:
            return np.zeros((len(X), 0), dtype=float), 0
        exclude = self.gp_exclude_names()
        hamming_idx = [
            i
            for i, n in enumerate(self.feature_names_)
            if n in self.binary_names_ and n not in exclude
        ]
        other_idx = [
            i
            for i, n in enumerate(self.feature_names_)
            if n not in exclude and i not in set(hamming_idx)
        ]
        idx = hamming_idx + other_idx
        if not idx:
            return np.zeros((len(X), 0), dtype=float), 0
        return X[:, idx], len(hamming_idx)

    def confidence_vector(self, X: np.ndarray) -> np.ndarray:
        """Recover per-row structural confidence from standardized features.

        Parameters
        ----------
        X : numpy.ndarray
            Standardized feature matrix.

        Returns
        -------
        numpy.ndarray
            Confidence values clipped to ``[0, 1]``; defaults to zeros.
        """
        if STRUCT_CONF_COLUMN not in self.feature_names_:
            return np.zeros(len(X), dtype=float)
        i = self.feature_names_.index(STRUCT_CONF_COLUMN)
        raw = X[:, i] * self.stds_[i] + self.means_[i]
        return np.clip(raw, 0.0, 1.0)
