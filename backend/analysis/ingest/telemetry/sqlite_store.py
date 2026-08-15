"""SQLite storage and lookup helpers for telemetry exports."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis.paths import resolve_telemetry_db_path

from ..column_resolution import build_specs, resolve_columns
from ..normalize import normalize_well, parse_date
from ..store_sync import dedup_across_files, guard_nonempty_source
from .processor import DEFAULT_SOURCE_DIR, merge_telemetry_exports, resolve_telemetry_files


# Metadata anchor columns the raw-storage indexer keys on. Resolving these lets
# a renamed/reordered header still be recognized; a plain substring fallback
# keeps the previous tolerant behaviour for broad headers.
_TELEMETRY_META_SPECS = build_specs(
    {
        "дата": ["дата", "date", "дата время", "дата и время"],
        "скважина": ["скважина", "скваж", "well", "id скважины", "№ скважины"],
        "run_id": ["run", "run_id", "id запуска", "запуск"],
    }
)


SQLITE_DEFAULT_PATH = resolve_telemetry_db_path()
SQLITE_RAW_TABLE = "telemetry_raw"
SQLITE_DAILY_TABLE = "telemetry_daily"
SQLITE_COLUMN_MAP_TABLE = "telemetry_column_map"
SQLITE_SUFFIXES = {".sqlite", ".db"}
SQLITE_METADATA_COLUMNS = {
    "_meta_row_id",
    "_meta_record_timestamp",
    "_meta_record_date",
    "_meta_normalized_well",
    "_meta_normalized_run_id",
    "_meta_source_file",
    "_meta_source_path",
}


@dataclass(frozen=True)
class TelemetrySQLiteBuildResult:
    """Summary of a telemetry SQLite build."""

    sqlite_path: Path
    raw_table: str
    daily_table: str
    source_files: tuple[Path, ...]
    total_rows: int
    raw_indexed_rows: int
    daily_rows: int


def _storage_column_name(index: int) -> str:
    return f"col_{index:04d}"


#: SQLite's INTEGER is 64-bit. Python's is not, so a single absurd cell kills the
#: whole load with ``OverflowError: Python int too large to convert to SQLite
#: INTEGER`` — and it does happen: ``Газовый фактор (ОЗНА), м3/т`` carries 13
#: values up to 2.3e24 m³/t across three field exports, the ОЗНА meter dividing by
#: a near-zero oil rate.
_SQLITE_INT_MAX = 2**63 - 1
_SQLITE_INT_MIN = -(2**63)


def _fits_sqlite_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and not (
        _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX
    )


def _coerce_oversized_integers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Store out-of-range integers as their exact decimal TEXT.

    Lossless: SQLite is dynamically typed, so the cell keeps its exact digits and
    can be read back. Dropping the row or clamping the value would silently edit
    the raw layer, which is meant to be the export as it arrived.
    """
    result = frame
    affected: dict[str, int] = {}
    for column in frame.columns:
        series = frame[column]
        if series.dtype != object:
            continue
        oversized = series.map(_fits_sqlite_integer)
        if not oversized.any():
            continue
        if result is frame:
            result = frame.copy()
        result[column] = series.mask(oversized, series.where(oversized).map(
            lambda value: str(value) if value is not None and not pd.isna(value) else value
        ))
        affected[str(column)] = int(oversized.sum())
    return result, affected


def _detect_anchor_columns(merged: pd.DataFrame) -> tuple[object, object, object]:
    """Resolve the (date, well, run_id) metadata columns from the merged headers.

    Uses the shared column resolver, then falls back to the previous substring
    heuristic so broad headers (e.g. "ID скважины (номер)") are never dropped by
    the stricter match. Shared by dedup and raw-storage preparation so both key
    on the same columns.
    """
    anchor_report = resolve_columns(
        [str(column) for column in merged.columns],
        _TELEMETRY_META_SPECS,
        dataset="telemetry",
        source_file="telemetry",
    )
    resolved_anchor: dict[str, object] = {}
    for column, resolution in zip(merged.columns, anchor_report.resolutions):
        if resolution.canonical is not None and resolution.canonical not in resolved_anchor:
            resolved_anchor[resolution.canonical] = column

    date_column = resolved_anchor.get("дата") or next(
        (column for column in merged.columns if str(column).strip().lower() == "дата"), None
    )
    well_column = resolved_anchor.get("скважина") or next(
        (column for column in merged.columns if "скваж" in str(column).lower()), None
    )
    run_id_column = resolved_anchor.get("run_id") or next(
        (column for column in merged.columns if "run" in str(column).lower()), None
    )
    return date_column, well_column, run_id_column


def _telemetry_dedup_key(merged: pd.DataFrame) -> pd.Series:
    """Natural key per row: (normalized_well, record_timestamp)."""
    date_column, well_column, _ = _detect_anchor_columns(merged)
    if well_column is not None:
        well = merged[well_column].apply(normalize_well)
    else:
        well = pd.Series([""] * len(merged), index=merged.index)
    if date_column is not None:
        ts = merged[date_column].apply(
            lambda value: (lambda d: d.strftime("%Y-%m-%d %H:%M:%S") if d is not None and pd.notna(d) else "")(parse_date(value))
        )
    else:
        ts = pd.Series([""] * len(merged), index=merged.index)
    well = well.astype(str)
    ts = ts.astype(str)
    key = well.str.cat(ts, sep="|")
    # A row with neither a well nor a timestamp has no natural identity — leave
    # its key empty so dedup never collapses such rows together.
    return key.mask((well == "") & (ts == ""), "")


def _prepare_raw_storage_frames(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe, oversized = _coerce_oversized_integers(merged)
    if oversized:
        detail = ", ".join(f"{column}: {count}" for column, count in oversized.items())
        print(
            "[telemetry] значения вне 64-битного диапазона сохранены текстом "
            f"(без потери цифр) — {detail}"
        )

    storage_df = pd.DataFrame(
        {
            _storage_column_name(index): safe[column]
            for index, column in enumerate(safe.columns, start=1)
        },
        index=safe.index,
    )
    column_map_rows = [
        {
            "storage_name": _storage_column_name(index),
            "original_name": str(column),
            "column_order": index,
        }
        for index, column in enumerate(safe.columns, start=1)
    ]

    date_column, well_column, run_id_column = _detect_anchor_columns(merged)

    if date_column is not None:
        parsed_dates = merged[date_column].apply(parse_date)
    else:
        parsed_dates = pd.Series([None] * len(merged), index=merged.index, dtype=object)
    if well_column is not None:
        normalized_wells = merged[well_column].apply(normalize_well)
    else:
        normalized_wells = pd.Series([""] * len(merged), index=merged.index, dtype=object)
    if run_id_column is not None:
        normalized_run_ids = merged[run_id_column].astype(str).str.strip().str.casefold()
    else:
        normalized_run_ids = pd.Series([""] * len(merged), index=merged.index, dtype=object)

    storage_df["_meta_row_id"] = range(1, len(merged) + 1)
    storage_df["_meta_record_timestamp"] = parsed_dates.apply(
        lambda value: value.strftime("%Y-%m-%d %H:%M:%S") if value is not None and pd.notna(value) else None
    )
    storage_df["_meta_record_date"] = parsed_dates.apply(
        lambda value: value.normalize().strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
    )
    storage_df["_meta_normalized_well"] = normalized_wells
    storage_df["_meta_normalized_run_id"] = normalized_run_ids
    storage_df["_meta_source_file"] = merged["_source_file"] if "_source_file" in merged.columns else None
    storage_df["_meta_source_path"] = merged["_source_path"] if "_source_path" in merged.columns else None

    column_map = pd.DataFrame(column_map_rows)
    return storage_df, column_map


def _prepare_daily_storage_frame(merged: pd.DataFrame) -> pd.DataFrame:
    from ..operating_data import canonicalize_operating_source

    daily, _ = canonicalize_operating_source(
        merged,
        source_kind="telemetry",
        target_runs=None,
    )
    if daily.empty:
        return daily

    daily = daily.copy()
    daily["_meta_record_date"] = daily["date"].apply(
        lambda value: value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
    )
    daily["_meta_normalized_well"] = daily["well"].apply(normalize_well) if "well" in daily.columns else ""
    daily["_meta_normalized_run_id"] = (
        daily["run_id"].fillna("").astype(str).str.strip().str.casefold()
        if "run_id" in daily.columns
        else ""
    )
    # The canonical numeric columns are float64 by now, but pass-through object
    # columns (field, contractor, event) can still carry an oversized integer.
    daily, _ = _coerce_oversized_integers(daily)
    return daily


def _create_raw_indexes(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_date "
        f"ON {table_name} (_meta_normalized_well, _meta_record_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_run_id_date "
        f"ON {table_name} (_meta_normalized_run_id, _meta_record_date)"
    )


def _create_daily_indexes(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_well_date "
        f"ON {table_name} (_meta_normalized_well, _meta_record_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_run_id_date "
        f"ON {table_name} (_meta_normalized_run_id, _meta_record_date)"
    )


def build_telemetry_sqlite(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    sqlite_path: Path | str = SQLITE_DEFAULT_PATH,
    raw_table: str = SQLITE_RAW_TABLE,
    daily_table: str = SQLITE_DAILY_TABLE,
    verbose: bool = False,
) -> TelemetrySQLiteBuildResult:
    """Import merged telemetry exports into a SQLite lookup store."""
    sqlite_path = Path(sqlite_path)
    source_files = tuple(Path(path) for path in resolve_telemetry_files(source))
    # Refuse to wipe a populated store when no source files were found.
    guard_nonempty_source(source_files, sqlite_path, raw_table, label="telemetry")
    merged = merge_telemetry_exports(source_files, verbose=verbose)

    if not merged.empty:
        merged, dropped = dedup_across_files(merged, _telemetry_dedup_key(merged))
        if dropped:
            print(f"[telemetry] dedup dropped {dropped} row(s) duplicated across files (kept newest)")

    if merged.empty:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            connection.execute(f"DROP TABLE IF EXISTS {raw_table}")
            connection.execute(f"DROP TABLE IF EXISTS {daily_table}")
            connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")
        return TelemetrySQLiteBuildResult(
            sqlite_path=sqlite_path,
            raw_table=raw_table,
            daily_table=daily_table,
            source_files=source_files,
            total_rows=0,
            raw_indexed_rows=0,
            daily_rows=0,
        )

    raw_storage_df, column_map = _prepare_raw_storage_frames(merged)
    daily_storage_df = _prepare_daily_storage_frame(merged)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Building telemetry SQLite store at {sqlite_path}")

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(f"DROP TABLE IF EXISTS {raw_table}")
        connection.execute(f"DROP TABLE IF EXISTS {daily_table}")
        connection.execute(f"DROP TABLE IF EXISTS {SQLITE_COLUMN_MAP_TABLE}")

        chunk_size = 50000
        total_raw_chunks = max(1, (len(raw_storage_df) + chunk_size - 1) // chunk_size)
        for chunk_index, start in enumerate(range(0, len(raw_storage_df), chunk_size), start=1):
            chunk = raw_storage_df.iloc[start : start + chunk_size]
            chunk.to_sql(raw_table, connection, if_exists="append", index=False)
            if verbose:
                print(f"Writing telemetry raw rows: {chunk_index}/{total_raw_chunks}")

        total_daily_chunks = max(1, (len(daily_storage_df) + chunk_size - 1) // chunk_size)
        for chunk_index, start in enumerate(range(0, len(daily_storage_df), chunk_size), start=1):
            chunk = daily_storage_df.iloc[start : start + chunk_size]
            chunk.to_sql(daily_table, connection, if_exists="append", index=False)
            if verbose:
                print(f"Writing telemetry daily rows: {chunk_index}/{total_daily_chunks}")

        column_map.to_sql(SQLITE_COLUMN_MAP_TABLE, connection, if_exists="replace", index=False)
        _create_raw_indexes(connection, raw_table)
        _create_daily_indexes(connection, daily_table)
        connection.commit()

    return TelemetrySQLiteBuildResult(
        sqlite_path=sqlite_path,
        raw_table=raw_table,
        daily_table=daily_table,
        source_files=source_files,
        total_rows=len(merged),
        raw_indexed_rows=len(raw_storage_df),
        daily_rows=len(daily_storage_df),
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


def load_telemetry_sqlite_frame(
    sqlite_path: Path | str,
    *,
    daily: bool = True,
    raw_table: str = SQLITE_RAW_TABLE,
    daily_table: str = SQLITE_DAILY_TABLE,
) -> pd.DataFrame:
    """Load the telemetry SQLite store back into a dataframe."""
    sqlite_path = Path(sqlite_path)
    table_name = daily_table if daily else raw_table
    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        if daily:
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            return frame
        if frame.empty:
            return frame
        mapping = _load_column_mapping(connection)

    restored_columns = {
        column: mapping.get(column, column)
        for column in frame.columns
        if column not in SQLITE_METADATA_COLUMNS
    }
    original_df = frame.rename(columns=restored_columns)
    ordered_columns = [mapping[key] for key in sorted(mapping.keys()) if mapping[key] in original_df.columns]
    metadata_columns = [column for column in original_df.columns if column.startswith("_meta_")]
    return original_df[ordered_columns + metadata_columns]


def query_telemetry_sqlite(
    sqlite_path: Path | str,
    *,
    well_value: Any | None = None,
    run_id: Any | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    daily: bool = True,
    raw_table: str = SQLITE_RAW_TABLE,
    daily_table: str = SQLITE_DAILY_TABLE,
    limit: int = 500,
) -> pd.DataFrame:
    """Query telemetry rows from SQLite."""
    sqlite_path = Path(sqlite_path)
    table_name = daily_table if daily else raw_table
    normalized_well = normalize_well(well_value) if well_value is not None else ""
    normalized_run_id = str(run_id).strip().casefold() if run_id is not None else ""
    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date)

    where_clauses = []
    params: list[Any] = []
    if normalized_well:
        where_clauses.append("_meta_normalized_well = ?")
        params.append(normalized_well)
    if normalized_run_id:
        where_clauses.append("_meta_normalized_run_id = ?")
        params.append(normalized_run_id)
    if start_ts is not None:
        where_clauses.append("_meta_record_date >= ?")
        params.append(start_ts.normalize().strftime("%Y-%m-%d"))
    if end_ts is not None:
        where_clauses.append("_meta_record_date <= ?")
        params.append(end_ts.normalize().strftime("%Y-%m-%d"))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        f"SELECT * FROM {table_name} "
        f"{where_sql} "
        f"ORDER BY _meta_record_date ASC "
        f"LIMIT ?"
    )
    params.append(limit)

    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
        if daily:
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            return frame
        if frame.empty:
            return frame
        mapping = _load_column_mapping(connection)

    restored_columns = {
        column: mapping.get(column, column)
        for column in frame.columns
        if column not in SQLITE_METADATA_COLUMNS
    }
    original_df = frame.rename(columns=restored_columns)
    ordered_columns = [mapping[key] for key in sorted(mapping.keys()) if mapping[key] in original_df.columns]
    metadata_columns = [column for column in original_df.columns if column.startswith("_meta_")]
    return original_df[ordered_columns + metadata_columns]


def resolve_telemetry_sqlite_path(source: Path | str | Sequence[Path | str]) -> Path | None:
    """Resolve a telemetry SQLite store from a file, directory, or list."""
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
    preferred = [path for path in unique_candidates if path.stem.lower() == "telemetry"]
    if len(preferred) == 1:
        return preferred[0]
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return max(unique_candidates, key=lambda item: item.stat().st_mtime)
