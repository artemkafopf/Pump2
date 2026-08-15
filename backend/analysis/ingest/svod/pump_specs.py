"""Pump-type parsing and a most-likely nominal-parameter database.

The register's ``Тип УЭЦН`` encodes the pump's nominal design in one of three
conventions:

* **Russian ЭЦН** -- ``gabarit-flow-head`` e.g. ``5а-200-2350`` (gabarit ``5А``,
  design flow 200 m³/d, head 2350 m), optionally with a series prefix
  (``ЭЦНДИК`` ...) and/or a leading OD (``10.2ЭЦНДИК5а-80-1600``).
* **Modular MT** -- ``MT5A-100DP`` (gabarit ``5A``, flow 100 m³/d).
* **Western REDA** -- ``DN460`` / ``SN8000`` (series + flow in bbl/d → ×0.159 m³/d).

:func:`parse_pump_type` recovers ``(gabarit, flow, head)`` from the code, and
:func:`build_pump_nominal_db` aggregates the fleet into the most-likely nominal
parameters per pump type so :func:`fill_missing_pump_specs` can backfill rows
whose nominal columns are blank.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..normalize import parse_number

# bbl/d → m³/d (REDA design flow is quoted in barrels).
_BBL_TO_M3 = 0.159

# Russian ESP series prefixes (longest first for a greedy match).
_RUSSIAN_SERIES_PREFIXES = [
    "УЭЦВН", "ЭЦНДИКЭ", "ЭЦНДИКэ", "ЭЦНДИК", "ЭЦНМИКэ", "ЭЦНМИК", "ЭЦНДИэ",
    "ЭЦНДИ", "ЭЦНМИэ", "ЭЦНМИ", "УЭЦНДИК", "УЭЦН", "УВНН", "ВННП", "ВНН",
    "ЭЦНА", "ЭЦН", "ЭОВНБ",
]

# Cyrillic letters that visually equal Latin (for gabarit / MT normalisation).
_CYR_TO_LAT = str.maketrans({
    "А": "A", "а": "A", "В": "B", "С": "C", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "Т": "T", "Х": "X",
})

_RU_CORE = re.compile(r"(\d+\s*[аАaA]?)\s*[- ]\s*(\d+)\s*[- ]\s*(\d+)")
_MT_CORE = re.compile(r"^MT\s*-?\s*(\d+A?)\s*[-/ ]\s*(\d+)")
_REDA_CORE = re.compile(r"^([DGSA])(N)?\s*0*(\d+)")


@dataclass(frozen=True)
class PumpTypeParse:
    """Structured nominal design recovered from a pump-type string."""

    gabarit: Optional[str]
    flow_m3d: Optional[float]
    head_m: Optional[float]
    parsed: bool


def _norm_gabarit(raw: str) -> str:
    return raw.strip().translate(_CYR_TO_LAT).upper().replace(" ", "")


def parse_pump_type(raw) -> PumpTypeParse:
    """Parse a pump-type string into ``(gabarit, flow, head)``. Never raises."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return PumpTypeParse(None, None, None, False)
    text = str(raw).strip()
    if not text:
        return PumpTypeParse(None, None, None, False)

    normalized = unicodedata.normalize("NFKC", text)
    latin = normalized.translate(_CYR_TO_LAT)

    mt = _MT_CORE.match(latin.upper())
    if mt:
        return PumpTypeParse(_norm_gabarit(mt.group(1)), float(mt.group(2)), None, True)

    reda = _REDA_CORE.match(latin.upper())
    if reda and not latin[:1].isdigit():
        flow = round(float(reda.group(3)) * _BBL_TO_M3, 1)
        return PumpTypeParse(None, flow, None, True)

    core = _RU_CORE.search(normalized)
    if core:
        return PumpTypeParse(
            _norm_gabarit(core.group(1)), float(core.group(2)), float(core.group(3)), True
        )

    return PumpTypeParse(None, None, None, False)


# Target columns backfilled from the pump type / nominal database.
NOMINAL_FLOW_COLUMN = "Ном. Произв. м₃/сут"
NOMINAL_HEAD_COLUMN = "Ном.напор (50Гц)"
PRODUCTIVITY_COLUMN = "Производительность"
GABARIT_COLUMN = "Габарит"
STAGES_COLUMN = "Кол.ступеней"

# Provenance note appended to the comment column when nominal params are filled.
NOMINAL_FILL_NOTE = "Ном. параметры из базы ЭЦН"


def append_comment(existing, note: str) -> str:
    """Append a provenance note to a comment cell, de-duplicating."""
    parts = [part.strip() for part in str(existing).split(";")] if existing not in (None, "") else []
    parts = [part for part in parts if part and part.lower() not in {"nan", "none"}]
    if note not in parts:
        parts.append(note)
    return "; ".join(parts)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() in {"", "-", "–", "nan", "none"}
    return False


def _pump_type_key(value) -> str:
    """Normalized key for grouping identical pump types (case/space/homoglyph)."""
    if _is_missing(value):
        return ""
    return str(value).strip().translate(_CYR_TO_LAT).upper().replace(" ", "")


def _most_likely(series: pd.Series):
    """Most-likely value for a nominal column: mode of the numeric values."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    modes = numeric.mode()
    return float(modes.iloc[0]) if not modes.empty else float(numeric.median())


def build_pump_nominal_db(
    df: pd.DataFrame,
    *,
    type_columns=("Тип УЭЦН", "esp_type"),
) -> dict:
    """Aggregate a fleet frame into most-likely nominal params per pump type.

    Returns ``{pump_type_key: {flow, head, gabarit, stages}}`` using the mode of
    each nominal column across all rows sharing that pump type.
    """
    if df is None or df.empty:
        return {}

    type_column = next((column for column in type_columns if column in df.columns), None)
    if type_column is None:
        return {}

    frame = df.copy()
    frame["_pump_key"] = frame[type_column].apply(_pump_type_key)
    frame = frame[frame["_pump_key"] != ""]
    if frame.empty:
        return {}

    database: dict = {}
    for pump_key, group in frame.groupby("_pump_key", sort=False):
        entry: dict = {}
        if NOMINAL_FLOW_COLUMN in group.columns:
            entry["flow"] = _most_likely(group[NOMINAL_FLOW_COLUMN])
        if NOMINAL_HEAD_COLUMN in group.columns:
            entry["head"] = _most_likely(group[NOMINAL_HEAD_COLUMN])
        if STAGES_COLUMN in group.columns:
            entry["stages"] = _most_likely(group[STAGES_COLUMN])
        if GABARIT_COLUMN in group.columns:
            gabarit_values = group[GABARIT_COLUMN].dropna().astype(str)
            if not gabarit_values.empty:
                entry["gabarit"] = gabarit_values.mode().iloc[0]
        database[pump_key] = entry
    return database


def fill_missing_pump_specs(
    records: pd.DataFrame,
    database: Optional[dict] = None,
    *,
    comment_column: Optional[str] = None,
) -> pd.DataFrame:
    """Backfill blank nominal columns from the pump-type code, then the fleet DB.

    For each row the ``Тип УЭЦН`` code is parsed for ``(gabarit, flow, head)``;
    any still-missing nominal column is then filled from ``database`` (built with
    :func:`build_pump_nominal_db`) keyed by the same pump type. When
    ``comment_column`` is given, a provenance note is recorded on rows whose
    nominal params were filled.
    """
    records = records.copy()
    if records.empty:
        return records
    if database is None:
        database = build_pump_nominal_db(records)

    for column in (NOMINAL_FLOW_COLUMN, NOMINAL_HEAD_COLUMN, PRODUCTIVITY_COLUMN, GABARIT_COLUMN, STAGES_COLUMN):
        if column not in records.columns:
            records[column] = None
    if comment_column is not None and comment_column not in records.columns:
        records[comment_column] = None

    type_series = (
        records["Тип УЭЦН"] if "Тип УЭЦН" in records.columns
        else records.get("esp_type", pd.Series([None] * len(records), index=records.index))
    )

    for idx, type_value in type_series.items():
        parsed = parse_pump_type(type_value)
        entry = database.get(_pump_type_key(type_value), {})

        flow = parsed.flow_m3d if parsed.flow_m3d is not None else entry.get("flow")
        head = parsed.head_m if parsed.head_m is not None else entry.get("head")
        gabarit = parsed.gabarit if parsed.gabarit is not None else entry.get("gabarit")
        stages = entry.get("stages")

        filled_any = False
        if flow is not None:
            if _is_missing(records.at[idx, PRODUCTIVITY_COLUMN]):
                records.at[idx, PRODUCTIVITY_COLUMN] = flow
                filled_any = True
            if _is_missing(records.at[idx, NOMINAL_FLOW_COLUMN]):
                records.at[idx, NOMINAL_FLOW_COLUMN] = flow
                filled_any = True
        if head is not None and _is_missing(records.at[idx, NOMINAL_HEAD_COLUMN]):
            records.at[idx, NOMINAL_HEAD_COLUMN] = head
            filled_any = True
        if gabarit is not None and _is_missing(records.at[idx, GABARIT_COLUMN]):
            records.at[idx, GABARIT_COLUMN] = gabarit
            filled_any = True
        if stages is not None and _is_missing(records.at[idx, STAGES_COLUMN]):
            records.at[idx, STAGES_COLUMN] = stages
            filled_any = True

        if filled_any and comment_column is not None:
            records.at[idx, comment_column] = append_comment(records.at[idx, comment_column], NOMINAL_FILL_NOTE)

    return records


__all__ = [
    "PumpTypeParse",
    "parse_pump_type",
    "build_pump_nominal_db",
    "fill_missing_pump_specs",
    "append_comment",
    "NOMINAL_FILL_NOTE",
]
