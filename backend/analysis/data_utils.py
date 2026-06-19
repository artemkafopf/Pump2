from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


_FAIL_TOKENS = {"1", "true", "yes", "y", "fail", "failed", "failure"}
_CENSOR_TOKENS = {"0", "false", "no", "n", "censored", "running", "planned_stop", "stop"}


def read_excel_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    workbook = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return {sheet_name: workbook.parse(sheet_name=sheet_name) for sheet_name in workbook.sheet_names}


def load_excel_sheet(file_bytes: bytes, sheet_name: str | int) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl")


def _coerce_filter_operand(series: pd.Series, value: Any) -> tuple[pd.Series, Any]:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return pd.to_numeric(series, errors="coerce"), float(numeric_value)
    return series.astype("string"), str(value)


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    filtered = df.copy()
    for filter_config in filters or []:
        column = filter_config.get("column")
        operator = filter_config.get("operator")
        value = filter_config.get("value")

        if not column or operator is None:
            continue
        if column not in filtered.columns:
            raise ValueError(f"Filter column '{column}' was not found.")

        series = filtered[column]
        if operator == "is_null":
            mask = series.isna()
        elif operator == "not_null":
            mask = series.notna()
        elif operator == "contains":
            mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        elif operator == "not_contains":
            mask = ~series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        else:
            left, right = _coerce_filter_operand(series, value)
            if operator == "==":
                mask = left == right
            elif operator == "!=":
                mask = left != right
            elif operator == "<":
                mask = left < right
            elif operator == "<=":
                mask = left <= right
            elif operator == ">":
                mask = left > right
            elif operator == ">=":
                mask = left >= right
            else:
                raise ValueError(f"Unsupported filter operator '{operator}'.")

        filtered = filtered.loc[mask.fillna(False)].copy()
    return filtered


def normalize_event_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    non_null_numeric = numeric[series.notna()]
    if not non_null_numeric.empty and non_null_numeric.dropna().isin([0, 1]).all():
        return numeric.astype("Int64")

    normalized = series.astype("string").str.strip().str.casefold()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[normalized.isin(_FAIL_TOKENS)] = 1.0
    result.loc[normalized.isin(_CENSOR_TOKENS)] = 0.0

    return result.astype("Int64")


def _group_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)


def _serialize_group_key(values: tuple[str, ...]) -> str:
    if not values:
        return "GLOBAL"
    return " | ".join(values)


def assign_group_fallback(
    df: pd.DataFrame,
    group_columns: list[str] | None,
    min_group_size: int = 20,
) -> pd.DataFrame:
    group_columns = [column for column in (group_columns or []) if column in df.columns]
    assigned = df.copy()
    assigned["analysis_original_group"] = "GLOBAL"
    assigned["analysis_group_key"] = "GLOBAL"
    assigned["analysis_group_level"] = 0

    if not group_columns:
        return assigned

    level_counts: dict[int, dict[tuple[str, ...], int]] = {}
    for level in range(1, len(group_columns) + 1):
        keys = [
            tuple(_group_value(row[column]) for column in group_columns[:level])
            for _, row in assigned[group_columns[:level]].iterrows()
        ]
        level_counts[level] = pd.Series(keys).value_counts().to_dict()
        if level == len(group_columns):
            assigned["analysis_original_group"] = [_serialize_group_key(key) for key in keys]

    chosen_keys: list[str] = []
    chosen_levels: list[int] = []
    for _, row in assigned.iterrows():
        selected_key = "GLOBAL"
        selected_level = 0
        for level in range(len(group_columns), 0, -1):
            key = tuple(_group_value(row[column]) for column in group_columns[:level])
            if level_counts[level].get(key, 0) >= max(1, int(min_group_size)):
                selected_key = _serialize_group_key(key)
                selected_level = level
                break
        chosen_keys.append(selected_key)
        chosen_levels.append(selected_level)

    assigned["analysis_group_key"] = chosen_keys
    assigned["analysis_group_level"] = chosen_levels
    return assigned


def build_representative_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)

    row_data: dict[str, Any] = {}
    for column in df.columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(1, int(len(series) * 0.5)):
            row_data[column] = float(numeric.median())
            continue

        mode = series.mode(dropna=True)
        if not mode.empty:
            row_data[column] = mode.iloc[0]
        else:
            row_data[column] = None
    return pd.Series(row_data)


def prepare_modeling_dataframe(
    df: pd.DataFrame,
    duration_column: str,
    event_column: str,
    group_columns: list[str] | None = None,
    stress_terms: list[dict] | None = None,
    filters: list[dict] | None = None,
    min_group_size: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if duration_column not in df.columns:
        raise ValueError(f"Duration column '{duration_column}' was not found.")
    if event_column not in df.columns:
        raise ValueError(f"Event column '{event_column}' was not found.")

    selected = apply_filters(df, filters)
    summary: dict[str, Any] = {
        "rows_before_filtering": int(len(df)),
        "rows_after_filtering": int(len(selected)),
        "notes": [],
    }

    prepared = selected.copy()
    prepared[duration_column] = pd.to_numeric(prepared[duration_column], errors="coerce")
    original_event = prepared[event_column].copy()
    prepared[event_column] = normalize_event_series(prepared[event_column])

    stress_terms = stress_terms or []
    required_numeric_columns = {duration_column}
    for term in stress_terms:
        column = term.get("column")
        if column:
            required_numeric_columns.add(column)
        if term.get("reference_mode") == "column" and term.get("reference_column"):
            required_numeric_columns.add(term["reference_column"])

    for column in required_numeric_columns:
        if column not in prepared.columns:
            raise ValueError(f"Required modeling column '{column}' was not found.")
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid_duration = prepared[duration_column].isna() | (prepared[duration_column] <= 0)
    if invalid_duration.any():
        summary["notes"].append(f"Removed {int(invalid_duration.sum())} rows with missing or non-positive duration.")
        prepared = prepared.loc[~invalid_duration].copy()

    missing_event = prepared[event_column].isna()
    if missing_event.any():
        invalid_original_event = original_event.notna() & missing_event
        if invalid_original_event.any():
            sample_values = sorted({str(value) for value in original_event.loc[invalid_original_event].dropna().tolist()})
            summary["notes"].append(
                "Removed "
                f"{int(invalid_original_event.sum())} rows with event flag different from 0/1 or unsupported labels. "
                f"Examples: {sample_values[:10]}"
            )
        remaining_missing_event = missing_event & original_event.isna()
        if remaining_missing_event.any():
            summary["notes"].append(f"Removed {int(remaining_missing_event.sum())} rows with missing event flag.")
        prepared = prepared.loc[~missing_event].copy()

    missing_numeric = pd.Series(False, index=prepared.index)
    for column in sorted(required_numeric_columns):
        missing_numeric = missing_numeric | prepared[column].isna()
    if missing_numeric.any():
        summary["notes"].append(
            f"Removed {int(missing_numeric.sum())} rows with missing stress or reference values."
        )
        prepared = prepared.loc[~missing_numeric].copy()

    prepared = assign_group_fallback(prepared, group_columns, min_group_size=min_group_size)
    summary["rows_after_validation"] = int(len(prepared))
    summary["failure_count"] = int(prepared[event_column].sum())
    summary["censored_count"] = int(len(prepared) - prepared[event_column].sum())
    summary["group_columns"] = [column for column in (group_columns or []) if column in df.columns]

    if prepared.empty:
        raise ValueError("No rows remain after filtering and validation.")

    return prepared, summary
