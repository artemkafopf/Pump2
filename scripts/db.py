"""Central warehouse connection and helper utilities for pump2.db."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_DIR = REPO_ROOT / "data" / "warehouse"
WAREHOUSE_PATH = WAREHOUSE_DIR / "pump2.db"

_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS meta__pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT    NOT NULL,
    run_at      TEXT    NOT NULL,
    duration_seconds REAL,
    row_count   INTEGER,
    status      TEXT    NOT NULL DEFAULT 'ok',
    error_msg   TEXT
);

CREATE TABLE IF NOT EXISTS meta__source_hashes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name  TEXT NOT NULL,
    identifier   TEXT,
    checked_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta__quality_flags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    well_key   TEXT,
    row_id     INTEGER,
    flag_type  TEXT NOT NULL,
    detail     TEXT,
    logged_at  TEXT NOT NULL
);
"""


def get_warehouse_conn(path: Path = WAREHOUSE_PATH) -> sqlite3.Connection:
    """Return an open connection to pump2.db with WAL mode and metadata tables ensured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_METADATA_DDL)
    conn.commit()
    return conn


def upsert_df(
    df: pd.DataFrame,
    table: str,
    conn: sqlite3.Connection,
    *,
    pk_cols: list[str] | None = None,
    if_exists: str = "replace",
) -> int:
    """Write df to table.

    if_exists='replace': DROP + recreate (full refresh, the default for pipeline builds).
    if_exists='append':  INSERT into existing table.
    pk_cols: when supplied, also creates a UNIQUE index over those columns after write.

    Returns the number of rows written.
    """
    if df.empty:
        return 0

    # Normalize date columns to ISO strings so SQLite stores them consistently.
    date_df = df.copy()
    for col in date_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]", "datetimetz"]).columns:
        date_df[col] = date_df[col].dt.strftime("%Y-%m-%d")

    date_df.to_sql(table, conn, if_exists=if_exists, index=False)

    if pk_cols:
        index_name = f"idx_{table}_{'_'.join(pk_cols)}"
        cols_sql = ", ".join(pk_cols)
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table} ({cols_sql})")
        conn.commit()

    return len(date_df)


def log_pipeline_run(
    table: str,
    *,
    conn: sqlite3.Connection,
    status: str = "ok",
    row_count: int = 0,
    duration_seconds: float = 0.0,
    error_msg: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO meta__pipeline_runs (table_name, run_at, duration_seconds, row_count, status, error_msg)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (table, _now(), duration_seconds, row_count, status, error_msg),
    )
    conn.commit()


def log_source_hash(source_name: str, identifier: str, *, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO meta__source_hashes (source_name, identifier, checked_at) VALUES (?, ?, ?)",
        (source_name, identifier, _now()),
    )
    conn.commit()


def log_quality_flag(
    table: str,
    flag_type: str,
    detail: str,
    *,
    conn: sqlite3.Connection,
    well_key: str | None = None,
    row_id: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO meta__quality_flags (table_name, well_key, row_id, flag_type, detail, logged_at) VALUES (?, ?, ?, ?, ?, ?)",
        (table, well_key, row_id, flag_type, detail, _now()),
    )
    conn.commit()


def get_last_pipeline_run(table: str, *, conn: sqlite3.Connection) -> dict | None:
    """Return the most recent pipeline run record for a table, or None."""
    row = conn.execute(
        "SELECT table_name, run_at, row_count, status FROM meta__pipeline_runs WHERE table_name = ? ORDER BY id DESC LIMIT 1",
        (table,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(["table_name", "run_at", "row_count", "status"], row))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StepTimer:
    """Context manager that times a pipeline step and logs the result."""

    def __init__(self, table: str, conn: sqlite3.Connection) -> None:
        self.table = table
        self.conn = conn
        self._start: float = 0.0
        self.row_count: int = 0

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        print(f"[pipeline] Building {self.table} ...", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            log_pipeline_run(self.table, conn=self.conn, status="ok", row_count=self.row_count, duration_seconds=elapsed)
            print(f"[pipeline] {self.table}: {self.row_count:,} rows in {elapsed:.1f}s", flush=True)
        else:
            log_pipeline_run(self.table, conn=self.conn, status="error", error_msg=str(exc_val), duration_seconds=elapsed)
            print(f"[pipeline] {self.table}: FAILED after {elapsed:.1f}s — {exc_val}", flush=True)
        return False  # re-raise exceptions


__all__ = [
    "WAREHOUSE_PATH",
    "StepTimer",
    "file_sha256",
    "get_last_pipeline_run",
    "get_warehouse_conn",
    "log_pipeline_run",
    "log_quality_flag",
    "log_source_hash",
    "upsert_df",
]
