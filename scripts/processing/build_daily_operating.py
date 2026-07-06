"""Build proc__daily_operating: binary in-operation flag per well per day.

Priority:
  1. telemetry qliq > 0               → op_source = "telemetry_qliq"
  2. techregime status == "В работе"  → op_source = "techregime_status"
  3. no data                          → in_operation = 0, op_source = "missing"

Input: telemetry_daily SQLite (direct, no fallback) + raw techregime SQLite (for status).
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

from scripts.analyze_true_ttf import load_techregime_status_daily
from scripts.data_utils import _load_telemetry_daily, _numeric, normalize_well_key
from scripts.db import StepTimer, get_warehouse_conn, upsert_df

_CHUNK_SIZE = 200


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        wells_df = pd.read_sql("SELECT DISTINCT well FROM raw__v03_runs WHERE well IS NOT NULL", conn)
        if wells_df.empty:
            print("[build_daily_operating] raw__v03_runs is empty — run ingest_v03 first.")
            return

        all_wells = sorted(wells_df["well"].dropna().astype(str).str.strip().unique().tolist())
        print(f"[build_daily_operating] Processing {len(all_wells)} wells...")

        with StepTimer("proc__daily_operating", conn) as timer:
            conn.execute("DROP TABLE IF EXISTS proc__daily_operating")
            conn.commit()

            total_rows = 0
            first_chunk = True
            for start in range(0, len(all_wells), _CHUNK_SIZE):
                chunk_wells = all_wells[start : start + _CHUNK_SIZE]
                chunk_wkeys = [normalize_well_key(w) for w in chunk_wells if normalize_well_key(w)]
                placeholders = ",".join([f"'{wk}'" for wk in chunk_wkeys])

                # Load telemetry qliq directly from source — avoids techregime fallback in merged table.
                tel_raw = _load_telemetry_daily(chunk_wells)
                tel_raw["well_key"] = tel_raw["well_id"].map(normalize_well_key)
                tel_raw["tele_op"] = (_numeric(tel_raw["qliq"]) > 0).fillna(False)
                tel_df = tel_raw[["well_key", "dt", "tele_op"]]

                # Load techregime status from source SQLite.
                treg_df = load_techregime_status_daily(chunk_wells)
                treg_df["well_key"] = treg_df["well_id"].map(normalize_well_key)

                # Outer-join on (well_key, dt).
                merged = tel_df[["well_key", "dt", "tele_op"]].merge(
                    treg_df[["well_key", "dt", "treg_in_operation"]],
                    on=["well_key", "dt"],
                    how="outer",
                )
                merged["tele_op"] = merged["tele_op"].fillna(False)
                merged["treg_in_operation"] = merged["treg_in_operation"].fillna(False)

                # Apply priority rule: telemetry first, techregime as fallback.
                merged["in_operation"] = 0
                merged["op_source"] = "missing"

                tele_mask = merged["tele_op"]
                treg_mask = (~tele_mask) & merged["treg_in_operation"]
                merged.loc[tele_mask, "in_operation"] = 1
                merged.loc[tele_mask, "op_source"] = "telemetry_qliq"
                merged.loc[treg_mask, "in_operation"] = 1
                merged.loc[treg_mask, "op_source"] = "techregime_status"

                result = merged[["well_key", "dt", "in_operation", "op_source"]].copy()
                result = result.loc[result["well_key"].notna() & result["dt"].notna()]
                upsert_df(result, "proc__daily_operating", conn, if_exists="replace" if first_chunk else "append")
                total_rows += len(result)
                first_chunk = False

            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_daily_op_key_dt ON proc__daily_operating (well_key, dt)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_daily_op_flag ON proc__daily_operating (in_operation)")
            conn.commit()
            timer.row_count = total_rows
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
