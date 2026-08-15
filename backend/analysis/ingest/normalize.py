"""Normalization and parsing utilities"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional, Union


def _is_nan_like(x: Any) -> bool:
    """True for None / NaN / NaT scalars (never raising on arrays/lists)."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=200_000)
def _normalize_text_core(text: str) -> str:
    """Pure, cached text normalization over a plain string.

    The public :func:`normalize_text` handles NaN/None and delegates here, so the
    hot path (the same handful of well/field/header strings normalized millions of
    times across a build) is memoized.
    """
    text = text.strip().replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()


def normalize_text(x: Any) -> str:
    """
    Normalize text: lowercase, strip spaces, normalize Cyrillic, remove duplicates.

    Args:
        x: Input text

    Returns:
        Normalized text string
    """
    if _is_nan_like(x):
        return ""
    return _normalize_text_core(str(x))


def strip_header_prefix(x: Any) -> str:
    """
    Remove Excel-like column prefixes such as ``A.`` or ``BW.`` from headers.
    
    Args:
        x: Header text
        
    Returns:
        Header without positional prefix
    """
    text = normalize_text(x)
    return re.sub(r"^[a-z]{1,3}\.\s+", "", text, flags=re.IGNORECASE)


@lru_cache(maxsize=200_000)
def _normalize_well_core(text: str) -> str:
    well_id = text.strip()
    # Remove trailing .0 from Excel numbers
    if well_id.endswith(".0"):
        well_id = well_id[:-2]
    well_id = _normalize_text_core(well_id)
    # Keep side-track suffixes (e.g., "1сч", "2сч", "1пб") but normalize spacing.
    return re.sub(r"\s+([а-яa-z]+)$", r"\1", well_id)


def normalize_well(x: Any) -> str:
    """
    Normalize well identifiers.

    Args:
        x: Input well identifier

    Returns:
        Normalized well ID
    """
    if _is_nan_like(x):
        return ""
    return _normalize_well_core(str(x))


def parse_date(x: Any) -> Optional[pd.Timestamp]:
    """
    Parse dates from various formats.
    
    Supports:
    - dd.mm.yyyy
    - yyyy-mm-dd
    - Excel serials
    - pandas Timestamps
    - datetime objects
    
    Args:
        x: Input date value
        
    Returns:
        pandas Timestamp or None
    """
    if _is_nan_like(x):
        return None

    # Already a timestamp
    if isinstance(x, pd.Timestamp):
        return x

    # datetime object
    if isinstance(x, datetime):
        return pd.Timestamp(x)

    # Excel serial number. Bound the accepted range to plausible dates
    # (~1954..2119) so a small runtime/count integer like 5 is NOT silently read
    # as 1900-01-04.
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if _EXCEL_SERIAL_MIN <= x <= _EXCEL_SERIAL_MAX:
            try:
                # Excel epoch 1899-12-30 (accounts for the 1900 leap-year bug).
                return pd.Timestamp("1899-12-30") + pd.Timedelta(days=x)
            except Exception:
                return None
        return None

    # String parsing (cached — the same dates recur across whole columns).
    if isinstance(x, str):
        return _parse_date_str(x.strip())

    return None


# Excel serials for ~1954-08-14 .. ~2119-01-27 — outside this a bare number is
# treated as not-a-date rather than an Excel serial.
_EXCEL_SERIAL_MIN = 20000
_EXCEL_SERIAL_MAX = 80000


@lru_cache(maxsize=200_000)
def _parse_date_str(text: str) -> Optional[pd.Timestamp]:
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(text, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(text, dayfirst=True)
    except Exception:
        return None


def parse_date_series(series: pd.Series) -> pd.Series:
    """Vectorized whole-column date parsing: ``%d.%m.%Y`` -> ISO -> dayfirst.

    Much faster than ``series.apply(parse_date)`` for large columns. Already-typed
    datetime values pass straight through ``pd.to_datetime``; unparseable cells
    become ``NaT``. (Excel serials are not handled here — use :func:`parse_date`
    per-cell for those.)
    """
    result = pd.to_datetime(series, format="%d.%m.%Y", errors="coerce")
    remaining = result.isna() & series.notna()
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(series[remaining], format="%Y-%m-%d", errors="coerce")
    remaining = result.isna() & series.notna()
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(series[remaining], dayfirst=True, errors="coerce")
    return result


def parse_number(x: Any) -> Optional[float]:
    """
    Parse numeric values with various separators.
    
    Args:
        x: Input numeric value
        
    Returns:
        float or None
    """
    if pd.isna(x) or x is None or x == "":
        return None
    
    if isinstance(x, (int, float)):
        return float(x)
    
    # String parsing
    if isinstance(x, str):
        x = x.strip()
        
        if x == "" or x.lower() in ["-", "н/д", "нет"]:
            return None
        
        # Replace comma with dot for decimal
        x = x.replace(",", ".")
        
        # Remove spaces
        x = x.replace(" ", "")
        
        try:
            return float(x)
        except Exception:
            return None

    return None


def normalize_column_name(col: str, synonyms: dict = None) -> str:
    """
    Normalize column name using synonyms.
    
    Args:
        col: Input column name
        synonyms: Dictionary of {canonical_name: [synonyms]}
        
    Returns:
        Normalized column name
    """
    col_norm = normalize_text(col)
    
    if synonyms:
        for canonical, syns in synonyms.items():
            if col_norm in [normalize_text(s) for s in syns]:
                return canonical
    
    return col


def find_column_index(df: pd.DataFrame, canonical_name: str, synonyms: dict) -> Optional[int]:
    """
    Find column index by canonical name using synonyms.
    
    Args:
        df: DataFrame
        canonical_name: Canonical column name
        synonyms: Synonym dictionary
        
    Returns:
        Column index or None
    """
    if canonical_name in synonyms:
        possible_names = [normalize_text(s) for s in synonyms[canonical_name]]
    else:
        possible_names = [normalize_text(canonical_name)]
    
    for i, col in enumerate(df.columns):
        if normalize_text(col) in possible_names:
            return i
    
    return None


def get_column_by_synonym(df: pd.DataFrame, canonical_name: str, synonyms: dict) -> Optional[pd.Series]:
    """
    Get column from DataFrame by canonical name.
    
    Args:
        df: DataFrame
        canonical_name: Canonical column name
        synonyms: Synonym dictionary
        
    Returns:
        Column Series or None
    """
    idx = find_column_index(df, canonical_name, synonyms)
    if idx is not None:
        return df.iloc[:, idx]
    return None


def safe_fillna(series: pd.Series, value: Any = None) -> pd.Series:
    """Safely fill NA values in a series."""
    if series is None:
        return pd.Series(dtype=object)
    return series.fillna(value)
