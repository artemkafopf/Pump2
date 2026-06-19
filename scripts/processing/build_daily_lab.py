"""Build proc__daily_lab: lab chemistry expanded to daily grain via stepwise forward-fill.

Ports attach_lab_chemistry() from scripts/data_utils.py.
For each well in raw__v03_runs, we generate a date spine covering all run intervals,
then attach the nearest lab sample using merge_asof (backward + forward fill).
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

from scripts.data_utils import LAB_CHEMISTRY_COLUMNS, _numeric, normalize_well_key
from scripts.db import StepTimer, get_warehouse_conn, upsert_df, log_quality_flag

# Re-use the internal lab loader from data_utils.
from scripts.data_utils import _load_lab_samples

_CHUNK_SIZE = 200


def _date_spine_for_well(install_dates: pd.Series, stop_dates: pd.Series) -> pd.DatetimeIndex:
    """Return sorted unique dates covering all [install, stop] intervals for a well."""
    dates: set[pd.Timestamp] = set()
    for install, stop in zip(install_dates, stop_dates):
        if pd.notna(install) and pd.notna(stop):
            dates.update(pd.date_range(install, stop, freq="D"))
    return pd.DatetimeIndex(sorted(dates))


def _expand_well_lab(well_key: str, spine: pd.DatetimeIndex, lab_df: pd.DataFrame) -> pd.DataFrame:
    """Apply merge_asof (backward + optional forward) for one well."""
    frame = pd.DataFrame({"well_key": well_key, "dt": spine})
    chemistry = lab_df.loc[lab_df["well_id"].map(normalize_well_key) == well_key].copy()
    if chemistry.empty:
        for col in LAB_CHEMISTRY_COLUMNS:
            frame[col] = np.nan
        frame["lab_sample_date"] = pd.NaT
        return frame

    chemistry = chemistry.sort_values("sample_date").reset_index(drop=True)
    frame = frame.sort_values("dt").reset_index(drop=True)

    backward = pd.merge_asof(
        frame,
        chemistry[["sample_date", *LAB_CHEMISTRY_COLUMNS]],
        left_on="dt",
        right_on="sample_date",
        direction="backward",
    )
    forward = pd.merge_asof(
        frame,
        chemistry[["sample_date", *LAB_CHEMISTRY_COLUMNS]],
        left_on="dt",
        right_on="sample_date",
        direction="forward",
    )
    for col in LAB_CHEMISTRY_COLUMNS:
        backward[col] = backward[col].combine_first(forward[col])
    backward["lab_sample_date"] = backward["sample_date"].combine_first(forward["sample_date"])
    return backward.drop(columns=["sample_date"])


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        runs = pd.read_sql(
            "SELECT well, well_key, install_date, stop_date FROM raw__v03_runs WHERE well IS NOT NULL",
            conn,
            parse_dates=["install_date", "stop_date"],
        )
        if runs.empty:
            print("[build_daily_lab] raw__v03_runs is empty — run ingest_v03 first.")
            return

        all_wells = sorted(runs["well"].dropna().astype(str).str.strip().unique().tolist())
        print(f"[build_daily_lab] Processing {len(all_wells)} wells for lab chemistry expansion...")

        with StepTimer("proc__daily_lab", conn) as timer:
            conn.execute("DROP TABLE IF EXISTS proc__daily_lab")
            conn.commit()

            total_rows = 0
            first_chunk = True
            for start in range(0, len(all_wells), _CHUNK_SIZE):
                chunk_wells = all_wells[start : start + _CHUNK_SIZE]
                lab = _load_lab_samples(chunk_wells)
                chunk_rows = runs.loc[runs["well"].isin(chunk_wells)].copy()

                frames: list[pd.DataFrame] = []
                for well in chunk_wells:
                    well_runs = chunk_rows.loc[chunk_rows["well"].astype(str).str.strip() == well]
                    spine = _date_spine_for_well(well_runs["install_date"], well_runs["stop_date"])
                    if len(spine) == 0:
                        continue
                    wk = normalize_well_key(well)
                    frames.append(_expand_well_lab(wk, spine, lab))

                if not frames:
                    continue
                chunk_df = pd.concat(frames, ignore_index=True)
                upsert_df(chunk_df, "proc__daily_lab", conn, if_exists="replace" if first_chunk else "append")
                total_rows += len(chunk_df)
                first_chunk = False
                print(f"  wells {start}–{start + len(chunk_wells) - 1}: {len(chunk_df):,} rows")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_daily_lab_key_dt ON proc__daily_lab (well_key, dt)")
            conn.commit()
            timer.row_count = total_rows

            # Quality: flag wells with no lab coverage.
            if total_rows > 0:
                covered = set(pd.read_sql(
                    f"SELECT DISTINCT well_key FROM proc__daily_lab WHERE {LAB_CHEMISTRY_COLUMNS[0]} IS NOT NULL",
                    conn,
                )["well_key"].tolist())
                for w in all_wells:
                    if normalize_well_key(w) not in covered:
                        log_quality_flag("proc__daily_lab", "no_lab_samples", f"Well {w!r} has no lab samples", conn=conn, well_key=normalize_well_key(w))
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
