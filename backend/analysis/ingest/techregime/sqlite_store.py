"""SQLite storage and lookup helpers for TechRegime exports."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis.paths import resolve_techregime_db_path

from ._normalize import normalize_well, parse_date
from ..store_sync import dedup_across_files, guard_nonempty_source
from .processor import (
    DATE_COLUMN,
    DEFAULT_SOURCE_DIR,
    FIELD_COLUMN,
    WELL_ID_COLUMN,
    _prepare_partition_columns,
    _render_progress,
    _verbose_print,
    merge_techregime_exports,
    resolve_techregime_files,
)


def _techregime_dedup_key(merged: pd.DataFrame) -> pd.Series:
    """Natural key per row: (normalized_well_id, report_date)."""
    if WELL_ID_COLUMN in merged.columns:
        well = merged[WELL_ID_COLUMN].apply(normalize_well).astype(str)
    else:
        well = pd.Series([""] * len(merged), index=merged.index)
    if DATE_COLUMN in merged.columns:
        date = merged[DATE_COLUMN].apply(
            lambda value: (lambda d: d.strftime("%Y-%m-%d") if d is not None and pd.notna(d) else "")(parse_date(value))
        ).astype(str)
    else:
        date = pd.Series([""] * len(merged), index=merged.index)
    key = well.str.cat(date, sep="|")
    return key.mask((well == "") & (date == ""), "")


WELL_NUMBER_COLUMN = "№ скважины"
SQLITE_DEFAULT_PATH = resolve_techregime_db_path()
SQLITE_DEFAULT_TABLE = "techregime_records"
SQLITE_COLUMN_MAP_TABLE = "techregime_column_map"
SQLITE_METADATA_COLUMNS = {
    "_meta_row_id",
    "_meta_report_date",
    "_meta_report_year",
    "_meta_field",
    "_meta_well_field",
    "_meta_well_id",
    "_meta_well_number",
    "_meta_normalized_well_id",
    "_meta_normalized_well_number",
    "_meta_source_file",
    "_meta_source_path",
}


@dataclass(frozen=True)
class SQLiteBuildResult:
    """Summary of a TechRegime SQLite build."""

    sqlite_path: Path
    table_name: str
    source_files: tuple[Path, ...]
    total_rows: int
    indexed_rows: int


def _storage_column_name(index: int) -> str:
    return f"col_{index:04d}"


def _prepare_sqlite_frames(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = _prepare_partition_columns(
        merged,
        date_column=DATE_COLUMN,
        field_column=FIELD_COLUMN,
    )

    storage_df = pd.DataFrame(index=prepared.index)
    column_map_rows = []
    for index, column in enumerate(merged.columns, start=1):
        storage_name = _storage_column_name(index)
        storage_df[storage_name] = prepared[column]
        column_map_rows.append(
            {
                "storage_name": storage_name,
                "original_name": str(column),
                "column_order": index,
            }
        )

    well_id_series = prepared[WELL_ID_COLUMN] if WELL_ID_COLUMN in prepared.columns else pd.Series(index=prepared.index, dtype=object)
    well_number_series = prepared[WELL_NUMBER_COLUMN] if WELL_NUMBER_COLUMN in prepared.columns else pd.Series(index=prepared.index, dtype=object)

    storage_df["_meta_row_id"] = range(1, len(prepared) + 1)
    storage_df["_meta_report_date"] = prepared["_parsed_date"].apply(
        lambda value: value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
    )
    storage_df["_meta_report_year"] = prepared["_parsed_year"]
    storage_df["_meta_field"] = prepared["_field_value"]
    storage_df["_meta_well_field"] = prepared["_well_field_value"]
    storage_df["_meta_well_id"] = well_id_series
    storage_df["_meta_well_number"] = well_number_series
    storage_df["_meta_normalized_well_id"] = well_id_series.apply(normalize_well)
    storage_df["_meta_normalized_well_number"] = well_number_series.apply(normalize_well)
    storage_df["_meta_source_file"] = prepared["_source_file"] if "_source_file" in prepared.columns else None
    storage_df["_meta_source_path"] = prepared["_source_path"] if "_source_path" in prepared.columns else None

    column_map = pd.DataFrame(column_map_rows)
    return storage_df, column_map


def _create_sqlite_indexes(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_id_date "
        f"ON {table_name} (_meta_normalized_well_id, _meta_report_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_number_date "
        f"ON {table_name} (_meta_normalized_well_number, _meta_report_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_field_date "
        f"ON {table_name} (_meta_field, _meta_report_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_field_date "
        f"ON {table_name} (_meta_well_field, _meta_report_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_year ON {table_name} (_meta_report_year)"
    )


def build_techregime_sqlite(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    sqlite_path: Path | str = SQLITE_DEFAULT_PATH,
    table_name: str = SQLITE_DEFAULT_TABLE,
    verbose: bool = False,
) -> SQLiteBuildResult:
    """Import merged TechRegime exports into a SQLite lookup store."""
    sqlite_path = Path(sqlite_path)
    source_files = tuple(Path(path) for path in resolve_techregime_files(source))
    # Refuse to wipe a populated store when no source files were found.
    guard_nonempty_source(source_files, sqlite_path, table_name, label="techregime")
    merged = merge_techregime_exports(source_files, verbose=verbose)

    if not merged.empty:
        merged, dropped = dedup_across_files(merged, _techregime_dedup_key(merged))
        if dropped:
            print(f"[techregime] dedup dropped {dropped} row(s) duplicated across files (kept newest)")

    if merged.empty:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")
        return SQLiteBuildResult(
            sqlite_path=sqlite_path,
            table_name=table_name,
            source_files=source_files,
            total_rows=0,
            indexed_rows=0,
        )

    storage_df, column_map = _prepare_sqlite_frames(merged)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    _verbose_print(f"Building SQLite store at {sqlite_path}", verbose=verbose)

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")

        chunk_size = 5000
        total_chunks = max(1, (len(storage_df) + chunk_size - 1) // chunk_size)
        for chunk_index, start in enumerate(range(0, len(storage_df), chunk_size), start=1):
            chunk = storage_df.iloc[start : start + chunk_size]
            chunk.to_sql(
                table_name,
                connection,
                if_exists="append",
                index=False,
            )
            _render_progress("Writing SQLite rows", chunk_index, total_chunks, verbose=verbose)

        column_map.to_sql(
            SQLITE_COLUMN_MAP_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        _create_sqlite_indexes(connection, table_name)
        connection.commit()

    return SQLiteBuildResult(
        sqlite_path=sqlite_path,
        table_name=table_name,
        source_files=source_files,
        total_rows=len(merged),
        indexed_rows=len(storage_df),
    )


def _load_column_mapping(connection: sqlite3.Connection) -> dict[str, str]:
    column_map = pd.read_sql_query(
        f"SELECT storage_name, original_name FROM {SQLITE_COLUMN_MAP_TABLE} ORDER BY column_order",
        connection,
    )
    return {
        str(row["storage_name"]): str(row["original_name"])
        for _, row in column_map.iterrows()
    }


def query_techregime_sqlite(
    sqlite_path: Path | str,
    well_value: Any | None = None,
    failure_date: Any | None = None,
    table_name: str = SQLITE_DEFAULT_TABLE,
    limit: int = 20,
) -> pd.DataFrame:
    """Query TechRegime rows from SQLite and restore original column names."""
    sqlite_path = Path(sqlite_path)
    normalized_well = normalize_well(well_value) if well_value is not None else ""
    failure_ts = parse_date(failure_date)

    where_clauses = []
    params: list[Any] = []

    if normalized_well:
        where_clauses.append(
            "(_meta_normalized_well_id = ? OR _meta_normalized_well_number = ?)"
        )
        params.extend([normalized_well, normalized_well])

    if failure_ts is not None:
        where_clauses.append("_meta_report_date <= ?")
        params.append(failure_ts.strftime("%Y-%m-%d"))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        f"SELECT * FROM {table_name} "
        f"{where_sql} "
        f"ORDER BY _meta_report_date DESC, _meta_row_id DESC "
        f"LIMIT ?"
    )
    params.append(limit)

    with sqlite3.connect(sqlite_path) as connection:
        df = pd.read_sql_query(sql, connection, params=params)
        if df.empty:
            return df
        mapping = _load_column_mapping(connection)

    restored_columns = {
        column: mapping.get(column, column)
        for column in df.columns
        if column not in SQLITE_METADATA_COLUMNS
    }
    original_df = df.rename(columns=restored_columns)

    ordered_columns = [mapping[key] for key in sorted(mapping.keys()) if mapping[key] in original_df.columns]
    metadata_columns = [column for column in original_df.columns if column.startswith("_meta_")]
    return original_df[ordered_columns + metadata_columns]


def lookup_latest_techregime_sqlite_row(
    sqlite_path: Path | str,
    well_value: Any,
    failure_date: Any | None = None,
    table_name: str = SQLITE_DEFAULT_TABLE,
) -> dict[str, Any] | None:
    """Return the latest TechRegime row for a well on or before the given date."""
    result = query_techregime_sqlite(
        sqlite_path=sqlite_path,
        well_value=well_value,
        failure_date=failure_date,
        table_name=table_name,
        limit=1,
    )
    if result.empty:
        return None
    return result.iloc[0].to_dict()
