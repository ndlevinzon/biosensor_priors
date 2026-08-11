"""Numeric / qualitative parsing for wet-lab phenotype fields."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

MISSING_TOKENS = {
    "nd",
    "n.d.",
    "n/a",
    "na",
    "",
    "nan",
    "none",
    "not determined",
}

NUMERIC_RE = re.compile(
    r"^\s*(?P<censor><=|>=|<|>|~|≈)?\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)\s*"
    r"(?P<unit>[a-zA-Zµμ%]*)\s*$"
)

UNIT_TO_UM = {
    "um": 1.0,
    "µm": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "m": 1e6,
}

BRIGHTNESS_SCALE = {
    "much dimmer": -2,
    "less than 20% of pan1.0": -2,
    "dimmer": -1,
    "similar": 0,
    "slightly brighter": 1,
    "brighter": 1,
    "much brighter": 2,
    "significantly brighter": 2,
}

IMPROVEMENT_SCALE = {
    "yes": "improved",
    "no": "not_improved",
    "yes and no": "mixed",
}


def parse_numeric_field(val: Any) -> dict[str, Any]:
    """Parse quantitative fields without silently converting qualitative text.

    Recognizes censored values (``<``, ``>``, etc.), approximate markers,
    units, and explicit missing tokens.

    Parameters
    ----------
    val : Any
        Raw cell value from an experimental workbook column.

    Returns
    -------
    dict[str, Any]
        Parsed record with keys ``raw``, ``value``, ``unit``, ``censored``,
        ``censor_direction``, ``approximate``, ``is_missing``, and
        ``parse_status`` (``"missing"``, ``"text"``, or ``"numeric"``).
    """
    raw = "" if pd.isna(val) else str(val).strip()
    result: dict[str, Any] = {
        "raw": raw,
        "value": np.nan,
        "unit": None,
        "censored": False,
        "censor_direction": None,
        "approximate": False,
        "is_missing": False,
        "parse_status": None,
    }

    key = raw.lower().strip()
    if key in MISSING_TOKENS:
        result["is_missing"] = True
        result["parse_status"] = "missing"
        return result

    s = raw.replace("≤", "<=").replace("≥", ">=").replace("μ", "µ")
    m = NUMERIC_RE.match(s)
    if not m:
        result["is_missing"] = None
        result["parse_status"] = "text"
        return result

    censor = m.group("censor")
    value = float(m.group("value"))
    unit = m.group("unit").strip() or None
    result["value"] = value
    result["unit"] = unit
    result["parse_status"] = "numeric"

    if censor in ("<", "<=", ">", ">="):
        result["censored"] = True
        result["censor_direction"] = "below" if censor in ("<", "<=") else "above"
    elif censor in ("~", "≈"):
        result["approximate"] = True

    return result


def apply_numeric_parse(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Append parsed numeric columns for a raw phenotype field.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing ``col``.
    col : str
        Column name whose values are passed to :func:`parse_numeric_field`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with additional ``{col}__*`` columns from parsing.
    """
    parsed = df[col].apply(parse_numeric_field).apply(pd.Series).add_prefix(f"{col}__")
    return pd.concat([df, parsed], axis=1)


def normalize_to_uM(
    value: Any,
    unit: Any,
    assume_unitless_um: bool = False,
) -> float:
    """Convert affinity values to micromolar (µM) units.

    Unitless values are not assumed to be µM unless explicitly requested.

    Parameters
    ----------
    value : Any
        Numeric affinity magnitude.
    unit : Any
        Unit string (e.g. ``"nM"``, ``"µM"``). ``None`` triggers the
        unitless policy.
    assume_unitless_um : bool, optional
        When ``True`` and ``unit`` is ``None``, treat ``value`` as already
        in µM. Default is ``False``.

    Returns
    -------
    float
        Affinity in µM, or ``nan`` when conversion is not possible.
    """
    if pd.isna(value):
        return np.nan
    if unit is None:
        return float(value) if assume_unitless_um else np.nan

    unit_key = str(unit).strip().lower().replace("μ", "µ")
    factor = UNIT_TO_UM.get(unit_key)
    if factor is None:
        return np.nan
    return float(value) * factor


def map_to_ordinal(text: Any, scale_dict: dict[str, int]) -> tuple[float, str]:
    """Map qualitative text to an ordinal score using a lookup scale.

    Parameters
    ----------
    text : Any
        Qualitative label from an experimental record.
    scale_dict : dict[str, int]
        Lowercase phrase → ordinal integer mapping. Longer phrases are
        matched before shorter ones for fuzzy substring matching.

    Returns
    -------
    score : float
        Mapped ordinal value, or ``nan`` when unmapped or missing.
    status : str
        Mapping status: ``"missing"``, ``"mapped"``, ``"fuzzy_mapped"``,
        or ``"UNMAPPED"``.
    """
    if pd.isna(text):
        return np.nan, "missing"
    key = str(text).strip().lower()
    if key in MISSING_TOKENS:
        return np.nan, "missing"
    if key in scale_dict:
        return scale_dict[key], "mapped"
    for phrase in sorted(scale_dict, key=len, reverse=True):
        if phrase in key:
            return scale_dict[phrase], "fuzzy_mapped"
    return np.nan, "UNMAPPED"


def normalize_improvement(val: Any) -> str:
    """Normalize improvement-from-baseline responses to standard labels.

    Parameters
    ----------
    val : Any
        Raw improvement field value.

    Returns
    -------
    str
        One of ``"improved"``, ``"not_improved"``, ``"mixed"``, ``"missing"``,
        or ``"UNMAPPED"``.
    """
    if pd.isna(val):
        return "missing"
    key = str(val).strip().lower()
    return IMPROVEMENT_SCALE.get(key, "UNMAPPED")


def bounds_from_value(value: Any, censor_direction: Any) -> tuple[float, float]:
    """Convert a reported value into a conservative numeric interval.

    Parameters
    ----------
    value : Any
        Reported numeric measurement.
    censor_direction : Any
        Censoring direction from parsing: ``"below"``, ``"above"``, or other.

    Returns
    -------
    lower : float
        Lower bound of the interval (``0.0`` for below-censored values).
    upper : float
        Upper bound of the interval (``inf`` for above-censored values).
    """
    if pd.isna(value):
        return np.nan, np.nan
    value = float(value)
    if censor_direction == "below":
        return 0.0, value
    if censor_direction == "above":
        return value, np.inf
    return value, value


def ratio_bounds(
    numerator_value: Any,
    numerator_censor: Any,
    denominator_value: Any,
    denominator_censor: Any,
) -> tuple[float, float]:
    """Compute interval bounds for a ratio of censored measurements.

    Used for selectivity as Kd(Prop) / Kd(Ac).

    Parameters
    ----------
    numerator_value : Any
        Numerator affinity value (PropCoA).
    numerator_censor : Any
        Numerator censor direction.
    denominator_value : Any
        Denominator affinity value (AcCoA).
    denominator_censor : Any
        Denominator censor direction.

    Returns
    -------
    lower : float
        Conservative lower bound of the ratio interval.
    upper : float
        Conservative upper bound of the ratio interval.
    """
    if pd.isna(numerator_value) or pd.isna(denominator_value):
        return np.nan, np.nan

    nl, nu = bounds_from_value(numerator_value, numerator_censor)
    dl, du = bounds_from_value(denominator_value, denominator_censor)

    if np.isinf(du):
        lower = 0.0
    else:
        lower = nl / du if du > 0 else np.nan

    if dl == 0:
        upper = np.inf
    else:
        upper = nu / dl

    return lower, upper


def format_ratio_bound(lower: Any, upper: Any) -> str | None:
    """Format ratio interval bounds for human-readable display.

    Parameters
    ----------
    lower : Any
        Lower bound of the ratio interval.
    upper : Any
        Upper bound of the ratio interval.

    Returns
    -------
    str | None
        Display string (exact value, bounded range, or inequality), or
        ``None`` when bounds are missing.
    """
    if pd.isna(lower) or pd.isna(upper):
        return None
    if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
        return f"{lower:.3g}"
    if lower > 0 and np.isinf(upper):
        return f">{lower:.3g}"
    if lower == 0 and np.isfinite(upper):
        return f"<{upper:.3g}"
    if lower > 0 and np.isfinite(upper):
        return f"{lower:.3g}-{upper:.3g}"
    return "unbounded"
