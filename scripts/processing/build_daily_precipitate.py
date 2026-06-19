"""Build proc__daily_precipitate: daily and cumulative salt/gypsum proxy loads per well.

Joins proc__daily_merged (qliq) with proc__daily_lab (chemistry) and computes:
  - daily_*_load_kg = ion_mg_l × qliq_m3d / 1000
  - daily_gypsum_scale_proxy = calcium × sulfate × qliq / 1,000,000
  - cum_* = cumulative sum per well (sorted by dt)

Ports add_dynamic_salt_proxies() from scripts/data_utils.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.data_utils import _numeric, normalize_well_key
from scripts.db import StepTimer, get_warehouse_conn, upsert_df

_CHUNK_SIZE = 200


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        wells_df = pd.read_sql("SELECT DISTINCT well_key FROM proc__daily_merged WHERE well_key IS NOT NULL", conn)
        if wells_df.empty:
            print("[build_daily_precipitate] proc__daily_merged is empty — run build_daily_merged first.")
            return

        all_wkeys = sorted(wells_df["well_key"].dropna().astype(str).unique().tolist())
        print(f"[build_daily_precipitate] Processing {len(all_wkeys)} wells...")

        with StepTimer("proc__daily_precipitate", conn) as timer:
            conn.execute("DROP TABLE IF EXISTS proc__daily_precipitate")
            conn.commit()

            total_rows = 0
            first_chunk = True
            for start in range(0, len(all_wkeys), _CHUNK_SIZE):
                chunk = all_wkeys[start : start + _CHUNK_SIZE]
                placeholders = ",".join([f"'{wk}'" for wk in chunk])

                merged = pd.read_sql(
                    f"SELECT well_key, dt, qliq FROM proc__daily_merged WHERE well_key IN ({placeholders})",
                    conn,
                    parse_dates=["dt"],
                )
                lab = pd.read_sql(
                    f"SELECT well_key, dt, calcium_mg_l, chloride_mg_l, sulfate_mg_l FROM proc__daily_lab WHERE well_key IN ({placeholders})",
                    conn,
                    parse_dates=["dt"],
                )

                combined = merged.merge(lab, on=["well_key", "dt"], how="left")
                qliq = _numeric(combined["qliq"]).clip(lower=0)
                calcium = _numeric(combined["calcium_mg_l"])
                chloride = _numeric(combined["chloride_mg_l"])
                sulfate = _numeric(combined["sulfate_mg_l"])

                combined["daily_calcium_load_kg"] = (calcium * qliq) / 1000.0
                combined["daily_chloride_load_kg"] = (chloride * qliq) / 1000.0
                combined["daily_sulfate_load_kg"] = (sulfate * qliq) / 1000.0
                combined["daily_salt_load_kg"] = ((calcium + chloride + sulfate) * qliq) / 1000.0
                combined["daily_gypsum_scale_proxy"] = (calcium * sulfate * qliq) / 1_000_000.0

                combined = combined.sort_values(["well_key", "dt"]).reset_index(drop=True)
                grp = combined.groupby("well_key", sort=False)
                for daily_col, cum_col in [
                    ("daily_calcium_load_kg", "cum_calcium_load_kg"),
                    ("daily_chloride_load_kg", "cum_chloride_load_kg"),
                    ("daily_sulfate_load_kg", "cum_sulfate_load_kg"),
                    ("daily_salt_load_kg", "cum_salt_load_kg"),
                    ("daily_gypsum_scale_proxy", "cum_gypsum_scale_proxy"),
                ]:
                    combined[cum_col] = grp[daily_col].cumsum()

                keep = [
                    "well_key", "dt",
                    "daily_salt_load_kg", "daily_gypsum_scale_proxy",
                    "daily_calcium_load_kg", "daily_chloride_load_kg", "daily_sulfate_load_kg",
                    "cum_salt_load_kg", "cum_gypsum_scale_proxy",
                    "cum_calcium_load_kg", "cum_chloride_load_kg", "cum_sulfate_load_kg",
                ]
                result = combined[[c for c in keep if c in combined.columns]]
                upsert_df(result, "proc__daily_precipitate", conn, if_exists="replace" if first_chunk else "append")
                total_rows += len(result)
                first_chunk = False

            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_daily_ppt_key_dt ON proc__daily_precipitate (well_key, dt)")
            conn.commit()
            timer.row_count = total_rows
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
