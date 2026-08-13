"""Parse and standardize Stage-2 physics score outputs; retain raw terms.

Legacy column names ``rif_ac`` / ``rif_prop`` / ``rpx`` are kept for Stage 3.
Values are typically negated RF3 confidence (higher → more negative).
"""



from __future__ import annotations



from pathlib import Path

from typing import Any



import pandas as pd





# Canonical long-format / raw score columns retained end-to-end.

RAW_SCORE_COLUMNS = (

    "rif_ac",

    "rif_prop",

    "rpx",

    "delta_rif_sel",

)





def compute_delta_rif_sel(rif_ac: float, rif_prop: float) -> float:

    """Compute selectivity term ΔRIF_sel = RIF_Ac − RIF_Prop.



    Score direction (whether more negative is better) is **not** inferred

    here — it is declared in ``thresholds.yaml`` → ``physics.score_direction``.



    Parameters
    ----------
    rif_ac : float

        RIF score for the AcCoA ligand ensemble.

    rif_prop : float

        RIF score for the PropCoA ligand ensemble.



    Returns
    -------
    float

        Selectivity term RIF_Ac minus RIF_Prop.

    """

    return float(rif_ac) - float(rif_prop)





def parse_rif_score_table(path: Path) -> pd.DataFrame:

    """Parse a RIF score file into a standardized table.



    Accepted formats: TSV/CSV with columns that map to ligand scores.

    Flexible column aliases are normalized.



    Parameters
    ----------
    path : pathlib.Path

        Path to a RIF score TSV/CSV file.



    Returns
    -------
    pandas.DataFrame

        Table with canonical column names (``rif_ac``, ``rif_prop``, etc.).



    Raises
    ------
    FileNotFoundError

        When ``path`` does not exist.

    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(path)

    if path.suffix.lower() in {".tsv", ".txt"}:

        df = pd.read_csv(path, sep="\t")

    else:

        df = pd.read_csv(path)



    rename = {}

    lower = {c.lower().strip(): c for c in df.columns}

    aliases = {

        "rif_ac": ["rif_ac", "rif_accoa", "rif ac", "score_ac", "accoa"],

        "rif_prop": ["rif_prop", "rif_propcoa", "rif prop", "score_prop", "propcoa"],

        "mutation": ["mutation", "mut", "mutant"],

        "position": ["position", "pos", "canonical_position"],

        "wt": ["wt", "wildtype", "aa_wt"],

        "mutant": ["mutant_aa", "aa_mut", "to"],

        "version": ["version", "background"],

        "structure_model_id": ["structure_model_id", "model_id", "structure_id"],

    }

    for canon, names in aliases.items():

        for name in names:

            if name in lower:

                rename[lower[name]] = canon

                break

    out = df.rename(columns=rename)

    return out





def parse_rpx_score_table(path: Path) -> pd.DataFrame:

    """Parse an RPX packing score file into a standardized table.



    Parameters
    ----------
    path : pathlib.Path

        Path to an RPX score TSV/CSV file.



    Returns
    -------
    pandas.DataFrame

        Table with canonical column names (``rpx``, ``mutation``, etc.).



    Raises
    ------
    FileNotFoundError

        When ``path`` does not exist.

    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(path)

    if path.suffix.lower() in {".tsv", ".txt"}:

        df = pd.read_csv(path, sep="\t")

    else:

        df = pd.read_csv(path)

    rename = {}

    lower = {c.lower().strip(): c for c in df.columns}

    for canon, names in {

        "rpx": ["rpx", "rpx_score", "packing", "score"],

        "mutation": ["mutation", "mut"],

        "position": ["position", "pos", "canonical_position"],

        "structure_model_id": ["structure_model_id", "model_id"],

        "version": ["version"],

    }.items():

        for name in names:

            if name in lower:

                rename[lower[name]] = canon

                break

    return df.rename(columns=rename)





def standardize_scan_row(

    *,

    version: str,

    position: int,

    wt: str,

    mutant: str,

    rif_ac: float,

    rif_prop: float,

    rpx: float,

    structure_model_id: str | None = None,

    physics_scan_id: str | None = None,

    extra: dict[str, Any] | None = None,

) -> dict[str, Any]:

    """Build one long-format physics scan row with raw and derived terms.



    Parameters
    ----------
    version : str

        Design background version identifier.

    position : int

        Canonical mutation position.

    wt : str

        Wild-type amino acid at ``position``.

    mutant : str

        Mutant amino acid at ``position``.

    rif_ac : float

        RIF score for AcCoA.

    rif_prop : float

        RIF score for PropCoA.

    rpx : float

        RPX packing score.

    structure_model_id : str, optional

        Structure model identifier for this row.

    physics_scan_id : str, optional

        Scan batch identifier.

    extra : dict, optional

        Additional columns merged when not already present.



    Returns
    -------
    dict

        Standardized long-format scan row.

    """

    row = {

        "version": version,

        "position": int(position),

        "wt": str(wt),

        "mutant": str(mutant),

        "mutation": f"{wt}{int(position)}{mutant}",

        "rif_ac": float(rif_ac),

        "rif_prop": float(rif_prop),

        "rpx": float(rpx),

        "delta_rif_sel": compute_delta_rif_sel(rif_ac, rif_prop),

        "structure_model_id": structure_model_id,

        "physics_scan_id": physics_scan_id,

    }

    if extra:

        for k, v in extra.items():

            if k not in row:

                row[k] = v

    return row





def write_mock_rif_scores(

    path: Path,

    rows: list[dict[str, Any]],

) -> Path:

    """Write mock RIF score rows to a TSV file.



    Parameters
    ----------
    path : pathlib.Path

        Output TSV path.

    rows : list of dict

        Score rows to serialize.



    Returns
    -------
    pathlib.Path

        ``path`` after writing.

    """

    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

    return path





def write_mock_rpx_scores(

    path: Path,

    rows: list[dict[str, Any]],

) -> Path:

    """Write mock RPX score rows to a TSV file.



    Parameters
    ----------
    path : pathlib.Path

        Output TSV path.

    rows : list of dict

        Score rows to serialize.



    Returns
    -------
    pathlib.Path

        ``path`` after writing.

    """

    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

    return path


