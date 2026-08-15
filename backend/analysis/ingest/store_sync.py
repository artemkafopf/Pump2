"""Shared build-time helpers for the source SQLite caches (lab/techregime/telemetry).

These stores are derived caches rebuilt by DROP + reload. Two hazards come with
that: an empty source folder silently wipes the store, and overlapping exports
in one folder duplicate rows (the per-file frames are ``concat``'ed with no
dedup). This module supplies the two guards used by all three stores:

* :func:`guard_nonempty_source` — refuse to rebuild (which would DROP the tables)
  when discovery found no files but the store still holds data.
* :func:`dedup_across_files` — drop rows whose natural key also appears in a
  *newer* source file, keeping the newest file's version. Rows sharing a key
  within a single file are left untouched — only cross-file collisions collapse,
  so a rebuild is idempotent and duplicate-free for overlapping inputs without
  discarding legitimately distinct same-key readings.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class EmptySourceError(RuntimeError):
    """Raised when a rebuild would wipe a populated store but found no sources."""


def _file_mtime(path: object) -> float:
    """Modification time of a source file; ``-inf`` when it can't be read."""
    try:
        return Path(str(path)).stat().st_mtime
    except (OSError, TypeError, ValueError):
        return float("-inf")


def dedup_across_files(
    frame: pd.DataFrame,
    key: pd.Series,
    *,
    source_path_col: str = "_source_path",
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose natural ``key`` appears in a newer source file.

    For each key value, the newest contributing file wins: rows from any older
    file with that key are dropped. Rows from the newest file for a key are all
    kept (so two distinct same-key readings inside one export survive), and rows
    with an empty key are never dropped. Row order is preserved.

    Returns the deduplicated frame and the number of rows dropped.
    """
    if frame.empty:
        return frame, 0

    key = pd.Series(list(key), index=frame.index).astype("string").fillna("")
    has_key = key.str.len() > 0

    if source_path_col in frame.columns:
        mtime = frame[source_path_col].map(_file_mtime)
    else:
        mtime = pd.Series(0.0, index=frame.index)

    # Newest source mtime per key; keep rows that match it (and all unkeyed rows).
    newest = mtime.where(has_key).groupby(key.where(has_key)).transform("max")
    keep_mask = (~has_key) | (mtime >= newest)
    dropped = int((~keep_mask).sum())
    return frame[keep_mask].reset_index(drop=True), dropped


def store_table_rowcount(sqlite_path: Path | str, table: str) -> int:
    """Row count of ``table`` in the store, or 0 if the store/table is absent."""
    path = Path(sqlite_path)
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def guard_nonempty_source(
    source_files: object,
    sqlite_path: Path | str,
    table: str,
    *,
    label: str,
) -> None:
    """Abort a rebuild that would wipe a populated store after finding no files.

    A rebuild DROPs the tables first, so running it against an empty source
    folder would silently destroy the cache. If discovery returned no files but
    the store still has rows, raise instead of proceeding.
    """
    if len(tuple(source_files)) > 0:
        return
    existing = store_table_rowcount(sqlite_path, table)
    if existing > 0:
        raise EmptySourceError(
            f"{label}: refusing to rebuild {sqlite_path} — no source files were "
            f"discovered but the store still holds {existing} row(s). Point at the "
            f"source folder, or delete the store to intentionally reset it."
        )


__all__ = [
    "EmptySourceError",
    "dedup_across_files",
    "guard_nonempty_source",
    "store_table_rowcount",
]
