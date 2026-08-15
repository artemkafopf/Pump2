"""SQLite storage and lookup helpers for lab chemistry exports."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from analysis.paths import resolve_lab_db_path

from ..normalize import normalize_well, parse_date
from ..store_sync import dedup_across_files, guard_nonempty_source
from .processor import (
    DEFAULT_SOURCE_DIR,
    WELL_COLUMN,
    aggregate_lab_column,
    merge_lab_exports,
    resolve_lab_files,
    resolve_sample_date_column,
    select_lab_rows_for_window,
)


def _lab_dedup_key(merged: pd.DataFrame) -> pd.Series:
    """Natural key per lab row: (normalized_well, sample_timestamp, row hash).

    The row hash is taken over every column except the source-file provenance
    columns, so it is *source-file-independent*: the same measurement exported in
    two overlapping files collapses to one row, while two genuinely different
    measurements for the same well+timestamp are kept (their hashes differ).
    """
    if WELL_COLUMN in merged.columns:
        well = merged[WELL_COLUMN].apply(normalize_well).astype(str)
    else:
        well = pd.Series([""] * len(merged), index=merged.index)

    date_column = resolve_sample_date_column(merged.columns)
    if date_column is not None:
        ts = merged[date_column].apply(
            lambda value: (lambda d: d.strftime("%Y-%m-%d %H:%M:%S") if d is not None and pd.notna(d) else "")(parse_date(value))
        ).astype(str)
    else:
        ts = pd.Series([""] * len(merged), index=merged.index)

    data_columns = [c for c in merged.columns if c not in ("_source_file", "_source_path")]
    row_hash = pd.util.hash_pandas_object(merged[data_columns], index=False).astype(str)
    return well.str.cat([ts, row_hash], sep="|")


SQLITE_DEFAULT_PATH = resolve_lab_db_path()
SQLITE_RAW_TABLE = "lab_raw"
SQLITE_COLUMN_MAP_TABLE = "lab_column_map"
SQLITE_SUFFIXES = {".sqlite", ".db"}
SQLITE_METADATA_COLUMNS = {
    "_meta_row_id",
    "_meta_sample_timestamp",
    "_meta_sample_date",
    "_meta_normalized_well",
    "_meta_source_file",
    "_meta_source_path",
}


@dataclass(frozen=True)
class LabSQLiteBuildResult:
    """Summary of a lab SQLite build."""

    sqlite_path: Path
    table_name: str
    source_files: tuple[Path, ...]
    total_rows: int
    indexed_rows: int


def _storage_column_name(index: int) -> str:
    return f"col_{index:04d}"


def _prepare_storage_frames(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    storage_df = pd.DataFrame(index=merged.index)
    column_map_rows = []
    for index, column in enumerate(merged.columns, start=1):
        storage_name = _storage_column_name(index)
        storage_df[storage_name] = merged[column]
        column_map_rows.append(
            {
                "storage_name": storage_name,
                "original_name": str(column),
                "column_order": index,
            }
        )

    date_column = resolve_sample_date_column(merged.columns)
    if date_column is not None:
        parsed_dates = merged[date_column].apply(parse_date)
    else:
        parsed_dates = pd.Series([None] * len(merged), index=merged.index, dtype=object)

    if WELL_COLUMN in merged.columns:
        normalized_wells = merged[WELL_COLUMN].apply(normalize_well)
    else:
        normalized_wells = pd.Series([""] * len(merged), index=merged.index, dtype=object)

    storage_df["_meta_row_id"] = range(1, len(merged) + 1)
    storage_df["_meta_sample_timestamp"] = parsed_dates.apply(
        lambda value: value.strftime("%Y-%m-%d %H:%M:%S") if value is not None and pd.notna(value) else None
    )
    storage_df["_meta_sample_date"] = parsed_dates.apply(
        lambda value: value.normalize().strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
    )
    storage_df["_meta_normalized_well"] = normalized_wells
    storage_df["_meta_source_file"] = merged["_source_file"] if "_source_file" in merged.columns else None
    storage_df["_meta_source_path"] = merged["_source_path"] if "_source_path" in merged.columns else None

    column_map = pd.DataFrame(column_map_rows)
    return storage_df, column_map


def _create_indexes(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_date "
        f"ON {table_name} (_meta_normalized_well, _meta_sample_date)"
    )


def build_lab_sqlite(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    sqlite_path: Path | str = SQLITE_DEFAULT_PATH,
    table_name: str = SQLITE_RAW_TABLE,
    verbose: bool = False,
) -> LabSQLiteBuildResult:
    """Import merged lab exports into a SQLite lookup store."""
    sqlite_path = Path(sqlite_path)
    source_files = tuple(Path(path) for path in resolve_lab_files(source))
    # Refuse to wipe a populated store when no source files were found.
    guard_nonempty_source(source_files, sqlite_path, table_name, label="lab")
    merged = merge_lab_exports(source_files, verbose=verbose)

    if not merged.empty:
        merged, dropped = dedup_across_files(merged, _lab_dedup_key(merged))
        if dropped:
            print(f"[lab] dedup dropped {dropped} row(s) duplicated across files (kept newest)")

    if merged.empty:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")
        return LabSQLiteBuildResult(
            sqlite_path=sqlite_path,
            table_name=table_name,
            source_files=source_files,
            total_rows=0,
            indexed_rows=0,
        )

    storage_df, column_map = _prepare_storage_frames(merged)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Building lab SQLite store at {sqlite_path}")

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")

        chunk_size = 5000
        total_chunks = max(1, (len(storage_df) + chunk_size - 1) // chunk_size)
        for chunk_index, start in enumerate(range(0, len(storage_df), chunk_size), start=1):
            chunk = storage_df.iloc[start : start + chunk_size]
            chunk.to_sql(table_name, connection, if_exists="append", index=False)
            if verbose:
                print(f"Writing lab rows: {chunk_index}/{total_chunks}")

        column_map.to_sql(SQLITE_COLUMN_MAP_TABLE, connection, if_exists="replace", index=False)
        _create_indexes(connection, table_name)
        connection.commit()

    return LabSQLiteBuildResult(
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


def _restore_original_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    restored_columns = {
        column: mapping.get(column, column)
        for column in frame.columns
        if column not in SQLITE_METADATA_COLUMNS
    }
    original_df = frame.rename(columns=restored_columns)
    ordered_columns = [mapping[key] for key in sorted(mapping.keys()) if mapping[key] in original_df.columns]
    metadata_columns = [column for column in original_df.columns if column.startswith("_meta_")]
    return original_df[ordered_columns + metadata_columns]


def query_lab_sqlite(
    sqlite_path: Path | str,
    *,
    well_value: Any | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    table_name: str = SQLITE_RAW_TABLE,
    limit: int = 5000,
) -> pd.DataFrame:
    """Query lab rows from SQLite and restore original column names."""
    sqlite_path = Path(sqlite_path)
    normalized_well = normalize_well(well_value) if well_value is not None else ""
    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date)

    where_clauses = []
    params: list[Any] = []
    if normalized_well:
        where_clauses.append("_meta_normalized_well = ?")
        params.append(normalized_well)
    if start_ts is not None:
        where_clauses.append("_meta_sample_date >= ?")
        params.append(start_ts.normalize().strftime("%Y-%m-%d"))
    if end_ts is not None:
        where_clauses.append("_meta_sample_date <= ?")
        params.append(end_ts.normalize().strftime("%Y-%m-%d"))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        f"SELECT * FROM {table_name} "
        f"{where_sql} "
        f"ORDER BY _meta_sample_date ASC, _meta_row_id ASC "
        f"LIMIT ?"
    )
    params.append(limit)

    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
        if frame.empty:
            return frame
        mapping = _load_column_mapping(connection)

    return _restore_original_columns(frame, mapping)


def aggregate_lab_chemistry_sqlite(
    sqlite_path: Path | str,
    well_value: Any,
    failure_date: Any | None = None,
    av_window: int = -1,
    column_rules: Optional[dict[str, str]] = None,
    table_name: str = SQLITE_RAW_TABLE,
) -> dict[str, float]:
    """Return windowed, averaged lab chemistry values for one well.

    Mirrors the file-based ``enrich_from_lab`` behaviour exactly: rows for the
    well on or before ``failure_date`` are windowed via
    :func:`select_lab_rows_for_window` and averaged per ``column_rules``
    (defaults to ``LAB_COLUMN_RULES``).
    """
    if column_rules is None:
        from ..config import LAB_COLUMN_RULES

        column_rules = LAB_COLUMN_RULES

    rows = query_lab_sqlite(
        sqlite_path,
        well_value=well_value,
        end_date=failure_date,
        table_name=table_name,
    )
    if rows.empty:
        return {}

    # Window on the normalized metadata timestamp rather than the restored raw
    # date column: the latter round-trips through SQLite as an ISO datetime
    # *string* that parse_date misreads with dayfirst semantics. The metadata
    # timestamp was parsed from a real Timestamp at build time, so materializing
    # it back to Timestamps keeps the window selection correct.
    rows = rows.copy()
    rows["_lab_sample_dt"] = pd.to_datetime(rows["_meta_sample_timestamp"], errors="coerce")
    selected_rows = select_lab_rows_for_window(
        rows,
        failure_date,
        av_window=av_window,
        date_col="_lab_sample_dt",
    )

    values: dict[str, float] = {}
    for target_col, source_col in column_rules.items():
        if source_col not in rows.columns:
            continue
        value = aggregate_lab_column(selected_rows, source_col)
        if value is not None:
            values[target_col] = value
    return values


def resolve_lab_sqlite_path(source: Path | str | Sequence[Path | str]) -> Path | None:
    """Resolve a lab SQLite store from a file, directory, or list."""
    candidates: list[Path] = []
    items = [source] if isinstance(source, (str, Path)) else list(source)
    for item in items:
        path = Path(item)
        if path.exists() and path.is_file() and path.suffix.lower() in SQLITE_SUFFIXES:
            candidates.append(path.resolve())
            continue
        if path.exists() and path.is_dir():
            for suffix in sorted(SQLITE_SUFFIXES):
                candidates.extend(sorted(item.resolve() for item in path.glob(f"*{suffix}") if item.is_file()))
    unique_candidates = sorted(set(candidates), key=lambda item: (item.name.lower(), str(item).lower()))
    if not unique_candidates:
        return None
    preferred = [path for path in unique_candidates if path.stem.lower() == "lab"]
    if len(preferred) == 1:
        return preferred[0]
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return max(unique_candidates, key=lambda item: item.stat().st_mtime)
