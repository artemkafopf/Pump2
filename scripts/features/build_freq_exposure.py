"""Build feat__run_freq_exposure: frequency exposure statistics per run.

For each run in raw__v03_runs, compute over the run's daily interval:
  - freq_above_55hz_pct   : fraction of valid freq days with freq > 55 Hz
  - freq_below_45hz_pct   : fraction of valid freq days with freq < 45 Hz
  - freq_signed_exposure   : frac_above − frac_below
  - freq_w_mean           : mean freq over the whole run interval (valid days)
  - n_freq_valid_days      : total days with valid freq reading
  - n_freq_above_55hz      : count of days above 55 Hz
  - n_freq_below_45hz      : count of days below 45 Hz
  - total_liquid_m3        : sum of qliq over the run (TLF)
  - total_freq_hz_days     : sum of freq over valid days (TRF)
  - avg_glf               : whole-run mean of gas_factor (all days with valid data)
  - avg_kpod              : whole-run mean of qliq / nominal_flow_m3d
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

from scripts.data_utils import _numeric
from scripts.db import StepTimer, get_warehouse_conn, upsert_df

FREQ_HIGH_HZ = 55.0
FREQ_LOW_HZ = 45.0
MIN_VALID_DAYS = 7


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        runs = pd.read_sql(
            """
            SELECT row_id, well, well_key, install_date, stop_date, nominal_flow_m3d
            FROM raw__v03_runs WHERE well IS NOT NULL
            """,
            conn,
            parse_dates=["install_date", "stop_date"],
        )
        if runs.empty:
            print("[build_freq_exposure] raw__v03_runs is empty — run ingest_v03 first.")
            return

        all_wkeys = sorted(runs["well_key"].dropna().astype(str).unique().tolist())
        placeholders = ",".join([f"'{wk}'" for wk in all_wkeys])
        daily = pd.read_sql(
            f"SELECT well_key, dt, freq, qliq, gas_factor FROM proc__daily_merged WHERE well_key IN ({placeholders})",
            conn,
            parse_dates=["dt"],
        )
        daily["freq"] = _numeric(daily["freq"])
        daily["qliq"] = _numeric(daily["qliq"])
        daily["gas_factor"] = _numeric(daily["gas_factor"])

        print(f"[build_freq_exposure] Computing freq exposure for {len(runs)} runs...")
        rows: list[dict] = []
        for run_row in runs.to_dict(orient="records"):
            wkey = run_row["well_key"]
            install = pd.Timestamp(run_row["install_date"])
            stop = pd.Timestamp(run_row["stop_date"])
            nom_flow = _numeric(pd.Series([run_row["nominal_flow_m3d"]])).iloc[0]

            run_daily = daily.loc[
                (daily["well_key"] == wkey)
                & (daily["dt"] >= install)
                & (daily["dt"] <= stop)
            ].copy()

            freq = run_daily["freq"]
            qliq = run_daily["qliq"]
            gas_factor = run_daily["gas_factor"]

            valid_mask = freq.notna() & (freq > 0)
            n_valid = int(valid_mask.sum())
            valid_freq = freq.loc[valid_mask]

            n_above = int((valid_freq > FREQ_HIGH_HZ).sum()) if n_valid > 0 else 0
            n_below = int((valid_freq < FREQ_LOW_HZ).sum()) if n_valid > 0 else 0
            frac_above = float(n_above / n_valid) if n_valid >= MIN_VALID_DAYS else np.nan
            frac_below = float(n_below / n_valid) if n_valid >= MIN_VALID_DAYS else np.nan
            signed_exposure = float(frac_above - frac_below) if n_valid >= MIN_VALID_DAYS else np.nan
            total_liq = float(qliq.clip(lower=0).sum(min_count=1))
            total_freq_val = float(valid_freq.sum()) if n_valid > 0 else np.nan
            freq_w_mean = float(valid_freq.mean()) if n_valid > 0 else np.nan

            # Whole-run avg_glf and avg_kpod — same definitions as original load_run_stats().
            glf_valid = gas_factor.dropna()
            avg_glf = float(glf_valid.mean()) if len(glf_valid) > 0 else np.nan

            if pd.notna(nom_flow) and nom_flow > 0:
                qliq_valid = qliq.dropna()
                avg_kpod = float((qliq_valid / nom_flow).mean()) if len(qliq_valid) > 0 else np.nan
            else:
                avg_kpod = np.nan

            rows.append({
                "row_id": int(run_row["row_id"]),
                "freq_above_55hz_pct": frac_above,
                "freq_below_45hz_pct": frac_below,
                "freq_signed_exposure": signed_exposure,
                "freq_w_mean": freq_w_mean,
                "n_freq_valid_days": n_valid,
                "n_freq_above_55hz": n_above,
                "n_freq_below_45hz": n_below,
                "total_liquid_m3": total_liq,
                "total_freq_hz_days": total_freq_val,
                "avg_glf": avg_glf,
                "avg_kpod": avg_kpod,
            })

        with StepTimer("feat__run_freq_exposure", conn) as timer:
            result = pd.DataFrame(rows)
            timer.row_count = upsert_df(result, "feat__run_freq_exposure", conn, pk_cols=["row_id"])
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
