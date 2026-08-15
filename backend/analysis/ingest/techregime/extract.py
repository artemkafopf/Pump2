"""Query and aggregate TechRegime values from Excel exports or SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ._normalize import normalize_text, normalize_well, parse_date
from .processor import DATE_COLUMN, WELL_ID_COLUMN, merge_techregime_exports
from .sqlite_store import SQLITE_COLUMN_MAP_TABLE, SQLITE_DEFAULT_TABLE


WELL_NUMBER_COLUMN = "№ скважины"
SQLITE_SUFFIXES = {".sqlite", ".db"}
DEFAULT_QUERY_INTERVAL = pd.Timedelta(days=7)


@dataclass(frozen=True)
class TechRegimeExtractResult:
    """Aggregated TechRegime values for one well and date window."""

    source_kind: str
    well_id: Any
    query_date: pd.Timestamp | None
    interval_start: pd.Timestamp | None
    interval_end: pd.Timestamp | None
    matched_rows: int
    values: dict[str, Any]


def _build_empty_result(
    *,
    source_kind: str,
    well_id: Any,
    query_date: Any,
    normalized_columns: Sequence[str],
    interval: Any,
) -> TechRegimeExtractResult:
    query_ts = parse_date(query_date)
    interval_end = query_ts.normalize() if query_ts is not None else None
    interval_delta = _parse_interval(interval)
    interval_start = interval_end - interval_delta if interval_end is not None else None
    return TechRegimeExtractResult(
        source_kind=source_kind,
        well_id=well_id,
        query_date=query_ts,
        interval_start=interval_start,
        interval_end=interval_end,
        matched_rows=0,
        values={normalized_column: None for normalized_column in normalized_columns},
    )


def _normalize_column_request(columns: str | Sequence[str]) -> list[str]:
    if isinstance(columns, str):
        requested = [columns]
    else:
        requested = list(columns)
    normalized = [normalize_text(column) for column in requested]
    return list(dict.fromkeys(column for column in normalized if column))


def _normalize_filter_values(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple, set, frozenset, pd.Index)):
        return list(values)
    return [values]


def _parse_interval(interval: Any) -> pd.Timedelta:
    if interval is None:
        return DEFAULT_QUERY_INTERVAL
    if isinstance(interval, pd.Timedelta):
        parsed = interval
    elif isinstance(interval, timedelta):
        parsed = pd.Timedelta(interval)
    elif isinstance(interval, (int, float)):
        parsed = pd.Timedelta(days=float(interval))
    else:
        parsed = pd.to_timedelta(interval)

    if pd.isna(parsed) or parsed < pd.Timedelta(0):
        raise ValueError("Interval must be a non-negative duration.")
    return parsed


def _build_column_lookup(columns: Sequence[object]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for column in columns:
        normalized = normalize_text(column)
        if normalized:
            lookup.setdefault(normalized, []).append(str(column))
    return lookup


def _resolve_column_names(
    available_columns: Sequence[object],
    normalized_columns: Sequence[str],
) -> dict[str, str]:
    lookup = _build_column_lookup(available_columns)
    resolved: dict[str, str] = {}
    for normalized_column in normalized_columns:
        matches = lookup.get(normalized_column, [])
        if not matches:
            raise KeyError(f"TechRegime column not found: {normalized_column}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous TechRegime column '{normalized_column}'. Matches: {matches}"
            )
        resolved[normalized_column] = matches[0]
    return resolved


def _is_missing_value(value: Any) -> bool:
    if pd.isna(value) or value is None:
        return True
    if isinstance(value, str):
        return normalize_text(value) in {"", "nan", "none", "-", "–"}
    return False


def _prepare_query_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"Expected '{DATE_COLUMN}' column in TechRegime data.")

    prepared = frame.copy()
    prepared["_parsed_date"] = prepared[DATE_COLUMN].apply(parse_date)

    if WELL_ID_COLUMN in prepared.columns:
        prepared["_normalized_well_id"] = prepared[WELL_ID_COLUMN].apply(normalize_well)
    else:
        prepared["_normalized_well_id"] = ""

    if WELL_NUMBER_COLUMN in prepared.columns:
        prepared["_normalized_well_number"] = prepared[WELL_NUMBER_COLUMN].apply(normalize_well)
    else:
        prepared["_normalized_well_number"] = ""

    return prepared


def _filter_query_rows(
    frame: pd.DataFrame,
    *,
    well_id: Any,
    query_date: Any,
    interval: Any,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    prepared = _prepare_query_frame(frame)
    normalized_well = normalize_well(well_id)
    query_ts = parse_date(query_date)

    mask = pd.Series(True, index=prepared.index)
    if normalized_well:
        mask = mask & (
            (prepared["_normalized_well_id"] == normalized_well)
            | (prepared["_normalized_well_number"] == normalized_well)
        )

    interval_end = query_ts.normalize() if query_ts is not None else None
    interval_delta = _parse_interval(interval)
    interval_start = interval_end - interval_delta if interval_end is not None else None

    if interval_end is not None:
        mask = mask & prepared["_parsed_date"].notna()
        mask = mask & (prepared["_parsed_date"].dt.normalize() >= interval_start)
        mask = mask & (prepared["_parsed_date"].dt.normalize() <= interval_end)

    return prepared.loc[mask].copy(), query_ts, interval_start, interval_end


def _apply_normalized_filters(
    frame: pd.DataFrame,
    filters: Mapping[str, Sequence[Any]] | None,
) -> pd.DataFrame:
    if not filters:
        return frame

    normalized_filters = {
        normalize_text(key): _normalize_filter_values(values)
        for key, values in filters.items()
        if normalize_text(key)
    }
    resolved_filters = _resolve_column_names(frame.columns, list(normalized_filters))
    filtered = frame
    for normalized_key, original_column in resolved_filters.items():
        allowed_values = normalized_filters[normalized_key]
        allowed_normalized = {normalize_text(value) for value in allowed_values}
        allow_missing = "" in allowed_normalized
        allowed_normalized.discard("")
        filtered = filtered[
            filtered[original_column].apply(
                lambda value: allow_missing if _is_missing_value(value) else normalize_text(value) in allowed_normalized
            )
        ]
    return filtered


def _ordered_unique_values(values: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for value in values:
        marker = normalize_text(value)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def _aggregate_series(values: pd.Series) -> Any:
    non_empty = values[~values.apply(_is_missing_value)]
    if non_empty.empty:
        return None

    numeric = pd.to_numeric(non_empty, errors="coerce")
    numeric_values = numeric[numeric.notna()]
    if not numeric_values.empty:
        return float(numeric_values.mean())

    unique_values = _ordered_unique_values(non_empty.tolist())
    if len(unique_values) == 1:
        return unique_values[0]
    return unique_values


def extract_techregime_values_from_frame(
    frame: pd.DataFrame,
    well_id: Any,
    query_date: Any,
    normalized_columns: str | Sequence[str],
    filters: Mapping[str, Sequence[Any]] | None = None,
    interval: Any = DEFAULT_QUERY_INTERVAL,
    source_kind: str = "dataframe",
) -> TechRegimeExtractResult:
    """Aggregate requested normalized TechRegime columns from an in-memory dataframe."""
    requested_columns = _normalize_column_request(normalized_columns)
    if frame.empty and DATE_COLUMN not in frame.columns:
        return _build_empty_result(
            source_kind=source_kind,
            well_id=well_id,
            query_date=query_date,
            normalized_columns=requested_columns,
            interval=interval,
        )
    matching_rows, query_ts, interval_start, interval_end = _filter_query_rows(
        frame,
        well_id=well_id,
        query_date=query_date,
        interval=interval,
    )
    filtered_rows = _apply_normalized_filters(matching_rows, filters)
    resolved_columns = _resolve_column_names(filtered_rows.columns, requested_columns)
    values = {
        normalized_column: _aggregate_series(filtered_rows[original_column])
        for normalized_column, original_column in resolved_columns.items()
    }
    return TechRegimeExtractResult(
        source_kind=source_kind,
        well_id=well_id,
        query_date=query_ts,
        interval_start=interval_start,
        interval_end=interval_end,
        matched_rows=len(filtered_rows),
        values=values,
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _load_sqlite_column_map(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        f"SELECT storage_name, original_name, column_order FROM {SQLITE_COLUMN_MAP_TABLE} ORDER BY column_order",
        connection,
    )


def _resolve_sqlite_source(source: Any) -> Path | None:
    if isinstance(source, Path):
        return source if source.suffix.lower() in SQLITE_SUFFIXES else None
    if isinstance(source, str):
        path = Path(source)
        return path if path.suffix.lower() in SQLITE_SUFFIXES else None
    if isinstance(source, Sequence) and not isinstance(source, pd.DataFrame):
        if len(source) != 1:
            return None
        item = source[0]
        if isinstance(item, (str, Path)):
            path = Path(item)
            return path if path.suffix.lower() in SQLITE_SUFFIXES else None
    return None


def extract_techregime_values_from_sqlite(
    sqlite_path: Path | str,
    well_id: Any,
    query_date: Any,
    normalized_columns: str | Sequence[str],
    filters: Mapping[str, Sequence[Any]] | None = None,
    interval: Any = DEFAULT_QUERY_INTERVAL,
    table_name: str = SQLITE_DEFAULT_TABLE,
) -> TechRegimeExtractResult:
    """Aggregate requested normalized TechRegime columns from the SQLite store."""
    sqlite_path = Path(sqlite_path)
    requested_columns = _normalize_column_request(normalized_columns)
    filter_columns = [normalize_text(key) for key in filters] if filters else []
    query_ts = parse_date(query_date)
    interval_end = query_ts.normalize() if query_ts is not None else None
    interval_delta = _parse_interval(interval)
    interval_start = interval_end - interval_delta if interval_end is not None else None
    normalized_well = normalize_well(well_id)

    with sqlite3.connect(sqlite_path) as connection:
        column_map = _load_sqlite_column_map(connection)
        original_by_normalized = _resolve_column_names(
            column_map["original_name"].tolist(),
            requested_columns + filter_columns,
        )
        storage_by_original = {
            str(row["original_name"]): str(row["storage_name"])
            for _, row in column_map.iterrows()
        }

        selected_storage = {
            storage_by_original[original_column]
            for original_column in original_by_normalized.values()
        }
        selected_columns = sorted(selected_storage)
        select_list = [
            *(_quote_identifier(column) for column in selected_columns),
            "_meta_report_date",
            "_meta_normalized_well_id",
            "_meta_normalized_well_number",
        ]

        where_clauses: list[str] = []
        params: list[Any] = []
        if normalized_well:
            where_clauses.append(
                "(_meta_normalized_well_id = ? OR _meta_normalized_well_number = ?)"
            )
            params.extend([normalized_well, normalized_well])
        if interval_start is not None and interval_end is not None:
            where_clauses.append("_meta_report_date >= ?")
            where_clauses.append("_meta_report_date <= ?")
            params.extend(
                [
                    interval_start.strftime("%Y-%m-%d"),
                    interval_end.strftime("%Y-%m-%d"),
                ]
            )

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = (
            f"SELECT {', '.join(select_list)} "
            f"FROM {_quote_identifier(table_name)} "
            f"{where_sql} "
            f"ORDER BY _meta_report_date ASC"
        )
        rows = pd.read_sql_query(sql, connection, params=params)

    if rows.empty:
        return TechRegimeExtractResult(
            source_kind="sqlite",
            well_id=well_id,
            query_date=query_ts,
            interval_start=interval_start,
            interval_end=interval_end,
            matched_rows=0,
            values={normalized_column: None for normalized_column in requested_columns},
        )

    restored_columns = {
        storage_by_original[original_column]: original_column
        for original_column in original_by_normalized.values()
    }
    restored = rows.rename(columns=restored_columns)
    filtered_rows = _apply_normalized_filters(restored, filters)
    values = {
        normalized_column: _aggregate_series(filtered_rows[original_column])
        for normalized_column, original_column in original_by_normalized.items()
        if normalized_column in requested_columns
    }
    return TechRegimeExtractResult(
        source_kind="sqlite",
        well_id=well_id,
        query_date=query_ts,
        interval_start=interval_start,
        interval_end=interval_end,
        matched_rows=len(filtered_rows),
        values=values,
    )


def extract_techregime_values(
    source: pd.DataFrame | Path | str | Sequence[Path | str],
    well_id: Any,
    query_date: Any,
    normalized_columns: str | Sequence[str],
    filters: Mapping[str, Sequence[Any]] | None = None,
    interval: Any = DEFAULT_QUERY_INTERVAL,
    table_name: str = SQLITE_DEFAULT_TABLE,
    verbose: bool = False,
) -> TechRegimeExtractResult:
    """Dispatch TechRegime extraction to SQLite, Excel source files, or an in-memory dataframe."""
    if isinstance(source, pd.DataFrame):
        return extract_techregime_values_from_frame(
            frame=source,
            well_id=well_id,
            query_date=query_date,
            normalized_columns=normalized_columns,
            filters=filters,
            interval=interval,
            source_kind="dataframe",
        )

    sqlite_path = _resolve_sqlite_source(source)
    if sqlite_path is not None:
        return extract_techregime_values_from_sqlite(
            sqlite_path=sqlite_path,
            well_id=well_id,
            query_date=query_date,
            normalized_columns=normalized_columns,
            filters=filters,
            interval=interval,
            table_name=table_name,
        )

    frame = merge_techregime_exports(source, verbose=verbose)
    if frame.empty and DATE_COLUMN not in frame.columns:
        return _build_empty_result(
            source_kind="excel",
            well_id=well_id,
            query_date=query_date,
            normalized_columns=_normalize_column_request(normalized_columns),
            interval=interval,
        )
    return extract_techregime_values_from_frame(
        frame=frame,
        well_id=well_id,
        query_date=query_date,
        normalized_columns=normalized_columns,
        filters=filters,
        interval=interval,
        source_kind="excel",
    )


__all__ = [
    "DEFAULT_QUERY_INTERVAL",
    "TechRegimeExtractResult",
    "extract_techregime_values",
    "extract_techregime_values_from_frame",
    "extract_techregime_values_from_sqlite",
]
