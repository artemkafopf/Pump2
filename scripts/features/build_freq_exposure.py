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
  - avg_glf               : whole-run mean of gas_factor (valid days only)
  - avg_kpod              : whole-run mean of qliq / nominal_flow_m3d

All per-property validity bounds below are confirmed against the actual data distribution in
proc__daily_merged. Values outside these ranges are physically impossible for ESP operations.
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

# Hard physical validity bounds — values outside cannot be real ESP readings.
# Data evidence: freq max=1700 (real VFD limit ~70 Hz, absolute HW limit ~100 Hz),
# qliq max=40000 (p99.9%=1333; no ESP well produces 5000+ m³/day),
# gas_factor max=4.9M (p99%=10K; GLF>5000 on oil well is sensor error).
FREQ_VALID_MAX_HZ: float = 100.0    # absolute VFD hardware limit
QLIQ_VALID_MAX_M3D: float = 5000.0  # generous upper bound for ESP liquid rate
GLF_VALID_MAX: float = 5000.0       # GLF above this on an oil ESP is sensor error


def _valid_freq(s: pd.Series) -> pd.Series:
    """Return a boolean mask: freq values that are physically plausible."""
    return s.notna() & (s > 0.0) & (s <= FREQ_VALID_MAX_HZ)


def _valid_qliq(s: pd.Series) -> pd.Series:
    """Return a boolean mask: qliq values that are physically plausible."""
    return s.notna() & (s >= 0.0) & (s <= QLIQ_VALID_MAX_M3D)


def _valid_glf(s: pd.Series) -> pd.Series:
    """Return a boolean mask: gas_factor values that are physically plausible."""
    return s.notna() & (s >= 0.0) & (s <= GLF_VALID_MAX)


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

        op_daily = pd.read_sql(
            f"SELECT well_key, dt, in_operation FROM proc__daily_operating WHERE well_key IN ({placeholders})",
            conn,
            parse_dates=["dt"],
        )
        daily = daily.merge(op_daily[["well_key", "dt", "in_operation"]], on=["well_key", "dt"], how="left")
        daily["in_operation"] = daily["in_operation"].fillna(0).astype(int)

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

            # Frequency validity: all physically plausible readings (0 < freq <= 100 Hz).
            freq_ok = _valid_freq(freq)
            n_valid = int(freq_ok.sum())
            valid_freq = freq.loc[freq_ok]  # used for TRF (total_freq_hz_days) only

            # Operating-day filter: exclude standby days where VFD holds a setpoint (~33–35 Hz).
            freq_ok_op = freq_ok & (run_daily["in_operation"] == 1)
            n_valid_op = int(freq_ok_op.sum())
            valid_freq_op = freq.loc[freq_ok_op]

            n_above = int((valid_freq_op > FREQ_HIGH_HZ).sum()) if n_valid_op > 0 else 0
            n_below = int((valid_freq_op < FREQ_LOW_HZ).sum()) if n_valid_op > 0 else 0
            frac_above = float(n_above / n_valid_op) if n_valid_op >= MIN_VALID_DAYS else np.nan
            frac_below = float(n_below / n_valid_op) if n_valid_op >= MIN_VALID_DAYS else np.nan
            signed_exposure = float(frac_above - frac_below) if n_valid_op >= MIN_VALID_DAYS else np.nan
            freq_w_mean = float(valid_freq_op.mean()) if n_valid_op > 0 else np.nan
            total_freq_val = float(valid_freq.sum()) if n_valid > 0 else np.nan  # TRF: all valid days

            # Liquid: only non-negative values within plausible ESP rate range.
            qliq_ok = _valid_qliq(qliq)
            valid_qliq = qliq.loc[qliq_ok]
            total_liq = float(valid_qliq.sum()) if len(valid_qliq) > 0 else 0.0

            # Gas-liquid factor: non-negative and below sensor-error threshold.
            glf_ok = _valid_glf(gas_factor)
            valid_glf = gas_factor.loc[glf_ok]
            avg_glf = float(valid_glf.mean()) if len(valid_glf) > 0 else np.nan

            # KPod = qliq / nominal_flow, using the same valid qliq mask.
            if pd.notna(nom_flow) and nom_flow > 0:
                avg_kpod = float((valid_qliq / nom_flow).mean()) if len(valid_qliq) > 0 else np.nan
            else:
                avg_kpod = np.nan

            rows.append({
                "row_id": int(run_row["row_id"]),
                "freq_above_55hz_pct": frac_above,
                "freq_below_45hz_pct": frac_below,
                "freq_signed_exposure": signed_exposure,
                "freq_w_mean": freq_w_mean,
                "n_freq_valid_days": n_valid_op,
                "n_freq_op_days": n_valid_op,
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
