"""Build proc__daily_merged: telemetry-first daily operational data with techregime fallback.

Ports load_daily_merged() from scripts/data_utils.py into the warehouse.
Processes wells in chunks to avoid loading all data into memory at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.data_utils import CANONICAL_COLUMNS, load_daily_merged, normalize_well_key
from scripts.db import StepTimer, get_warehouse_conn, upsert_df, log_quality_flag

_CHUNK_SIZE = 200  # wells per pipeline chunk (smaller than 400 to keep memory reasonable)


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        wells_df = pd.read_sql("SELECT DISTINCT well, well_key FROM raw__v03_runs WHERE well IS NOT NULL", conn)
        if wells_df.empty:
            print("[build_daily_merged] raw__v03_runs is empty — run ingest_v03 first.")
            return

        all_wells = sorted(wells_df["well"].dropna().astype(str).str.strip().unique().tolist())
        print(f"[build_daily_merged] Processing {len(all_wells)} wells in chunks of {_CHUNK_SIZE}...")

        with StepTimer("proc__daily_merged", conn) as timer:
            # Drop and recreate so we get a clean table + index.
            conn.execute("DROP TABLE IF EXISTS proc__daily_merged")
            conn.commit()

            total_rows = 0
            first_chunk = True
            for start in range(0, len(all_wells), _CHUNK_SIZE):
                chunk_wells = all_wells[start : start + _CHUNK_SIZE]
                chunk_df = load_daily_merged(chunk_wells)
                if chunk_df.empty:
                    continue

                # Normalize well_id → well_key for consistent joins downstream.
                chunk_df["well_key"] = chunk_df["well_id"].map(normalize_well_key)
                chunk_df = chunk_df.drop(columns=["well_id"])

                # Reorder columns.
                cols = ["well_key", "dt", *CANONICAL_COLUMNS, "source"]
                chunk_df = chunk_df[[c for c in cols if c in chunk_df.columns]]

                if_exists = "replace" if first_chunk else "append"
                upsert_df(chunk_df, "proc__daily_merged", conn, if_exists=if_exists)
                total_rows += len(chunk_df)
                first_chunk = False
                print(f"  wells {start}–{start + len(chunk_wells) - 1}: {len(chunk_df):,} rows (total so far: {total_rows:,})")

            # Create index after full load for performance.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_daily_merged_key_dt ON proc__daily_merged (well_key, dt)")
            conn.commit()
            timer.row_count = total_rows

            # Quality check: flag wells with zero rows.
            merged_wells = set(pd.read_sql("SELECT DISTINCT well_key FROM proc__daily_merged", conn)["well_key"].tolist())
            for w in all_wells:
                if normalize_well_key(w) not in merged_wells:
                    log_quality_flag("proc__daily_merged", "no_daily_rows", f"Well {w!r} has no rows in merged daily data", conn=conn, well_key=normalize_well_key(w))
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
