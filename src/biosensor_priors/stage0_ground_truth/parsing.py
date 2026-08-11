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
    """Parse quantitative fields without silently converting qualitative text."""
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
    parsed = df[col].apply(parse_numeric_field).apply(pd.Series).add_prefix(f"{col}__")
    return pd.concat([df, parsed], axis=1)


def normalize_to_uM(
    value: Any,
    unit: Any,
    assume_unitless_um: bool = False,
) -> float:
    """Convert affinity values to µM. Unitless values are not assumed µM by default."""
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
    if pd.isna(val):
        return "missing"
    key = str(val).strip().lower()
    return IMPROVEMENT_SCALE.get(key, "UNMAPPED")


def bounds_from_value(value: Any, censor_direction: Any) -> tuple[float, float]:
    """Convert a reported value into an interval."""
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
    """Interval bounds for numerator/denominator (Kd Prop / Kd Ac)."""
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
