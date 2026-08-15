"""Local normalization helpers for TechRegime utilities."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""

    text = str(value).strip()
    text = text.replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()


def normalize_well(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""

    well_id = str(value).strip()
    if well_id.endswith(".0"):
        well_id = well_id[:-2]
    well_id = normalize_text(well_id)
    return re.sub(r"\s+([а-яa-z]+)$", r"\1", well_id)


def parse_date(value: Any):
    if pd.isna(value) or value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, (int, float)) and value > 0:
        try:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=value)
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(text, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(text, dayfirst=True)
        except Exception:
            pass
    return None
