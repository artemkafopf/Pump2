"""OPZ dataset processing."""

import re

import pandas as pd

from ..io import load_dataset_frames
from ..normalize import normalize_text, normalize_well, parse_date, parse_number
from .enrich import combine_dataset_frames


def _is_two_stage_operation(opz_text: str) -> bool:
    normalized = normalize_text(opz_text)
    if not normalized:
        return False
    if "двух" in normalized or "двухэтап" in normalized:
        return True
    return re.search(r"(^|\W)2\s*[-й]?\s*этап", opz_text.lower()) is not None


def _resolve_record_lookback_days(record: pd.Series, default_days: int | float) -> float:
    for column_name in ("runtime_nno", "Наработка (сут)", "I. Наработка (сут)"):
        runtime_days = parse_number(record.get(column_name))
        if runtime_days is not None and runtime_days > 0:
            return float(runtime_days)
    return float(default_days)


def enrich_from_opz(
    records: pd.DataFrame,
    opz_path,
    well_col: str = "well",
    failure_date_col: str = "failure_date",
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Enrich OPZ flags from one or more OPZ datasets."""
    records = records.copy()
    frames = load_dataset_frames(opz_path, "opz")
    opz_df = combine_dataset_frames(frames)
    if opz_df.empty:
        return records

    well_col_opz = next((c for c in opz_df.columns if "скв" in str(c).lower()), None)
    if well_col_opz:
        opz_df["normalized_well"] = opz_df[well_col_opz].apply(normalize_well)
    else:
        return records

    opz_flag_cols = {
        "opz": "ОПЗ",
        "sko_opz": "СКО ОПЗ",
        "gk_opz": "ГК ОПЗ",
        "gko_opz": "ГКО ОПЗ",
        "sko_esp": "СКО УЭЦН",
        "gko_esp": "ГКО УЭЦН",
        "two_stage": "2-х этапка",
    }
    for col in opz_flag_cols.values():
        if col not in records.columns:
            records[col] = 0

    opz_type_col = next((c for c in opz_df.columns if any(x in str(c).lower() for x in ["тип", "операция", "вид"])), None)

    for idx, record in records.iterrows():
        if well_col not in record or pd.isna(record[well_col]):
            continue
        well_norm = normalize_well(record[well_col])
        fail_date = parse_date(record.get(failure_date_col))
        record_lookback_days = _resolve_record_lookback_days(record, lookback_days)
        matching_opz = opz_df[opz_df.get("normalized_well", pd.Series()).eq(well_norm)]
        if not matching_opz.empty and fail_date:
            for _, opz_row in matching_opz.iterrows():
                opz_date = parse_date(opz_row.get(next((c for c in opz_df.columns if "дата" in str(c).lower()), None)))
                if opz_date is None:
                    continue
                days_before_failure = (fail_date - opz_date).days
                if days_before_failure < 0 or days_before_failure > record_lookback_days:
                    continue

                opz_text = str(opz_row.get(opz_type_col, "")) if opz_type_col else ""
                normalized_opz_text = normalize_text(opz_text)
                if "опз" in normalized_opz_text:
                    records.loc[idx, "ОПЗ"] = 1
                if "ско" in normalized_opz_text and "уэцн" not in normalized_opz_text:
                    records.loc[idx, "СКО ОПЗ"] = 1
                if "гк" in normalized_opz_text and "гко" not in normalized_opz_text:
                    records.loc[idx, "ГК ОПЗ"] = 1
                if "гко" in normalized_opz_text and "уэцн" not in normalized_opz_text:
                    records.loc[idx, "ГКО ОПЗ"] = 1
                if "ско" in normalized_opz_text and "уэцн" in normalized_opz_text:
                    records.loc[idx, "СКО УЭЦН"] = 1
                if "гко" in normalized_opz_text and "уэцн" in normalized_opz_text:
                    records.loc[idx, "ГКО УЭЦН"] = 1
                if _is_two_stage_operation(opz_text):
                    records.loc[idx, "2-х этапка"] = 1
    return records


__all__ = ["enrich_from_opz"]
