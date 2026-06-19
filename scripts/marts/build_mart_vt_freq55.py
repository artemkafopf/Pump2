"""Build mart__vt_freq55: one row per run, replacing load_run_stats() in vt_freq55_app.py.

Joins:
  raw__v03_runs               — identifiers, dates, equipment specs
  feat__run_freq_exposure     — frequency stats, TLF, TRF
  feat__run_mature            — mature-window KPod/GLF/load/freq features
  proc__h2s_proxy             — H2S proxy
  proc__ttf_true              — TTF_true_best days
  proc__daily_precipitate     — cumulative salt/gypsum at run stop date
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

from scripts.db import StepTimer, get_warehouse_conn, upsert_df


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        with StepTimer("mart__vt_freq55", conn) as timer:
            # Base: runs.
            base = pd.read_sql(
                """
                SELECT row_id, well, well_key, field, pad, contractor, failed_node,
                       install_date, stop_date, event, run_days,
                       pump_type, nominal_flow_m3d, nominal_freq_hz,
                       motor_power_kw, stages, pbubble_atm
                FROM raw__v03_runs
                """,
                conn,
                parse_dates=["install_date", "stop_date"],
            )

            # Frequency exposure.
            freq = pd.read_sql("SELECT * FROM feat__run_freq_exposure", conn)
            base = base.merge(freq, on="row_id", how="left")

            # TTF true.
            ttf = pd.read_sql(
                "SELECT row_id, ttf_true_best_days, ttf_true_source, ttf_tele_days, ttf_treg_days FROM proc__ttf_true",
                conn,
            )
            base = base.merge(ttf, on="row_id", how="left")

            # H2S proxy.
            h2s = pd.read_sql(
                "SELECT row_id, h2s_proxy_mg_l, h2s_proxy_source FROM proc__h2s_proxy",
                conn,
            )
            base = base.merge(h2s, on="row_id", how="left")

            # Mature features (selected columns).
            # freq_m_mean excluded here — feat__run_freq_exposure already provides freq_w_mean
            # and avg_glf/avg_kpod; including freq_m_mean would collide if feat__run_mature
            # also exports it, producing _x/_y suffixes.
            mature_cols = [
                "row_id", "kpod_m_mean", "frac_kpod_below_0p7_m", "glf_m_mean",
                "frac_glf_above_thr_m", "load_m_mean",
                "salt_proxy_m_mean", "integrated_salt_proxy_m", "gypsum_proxy_m_mean",
            ]
            try:
                mature = pd.read_sql("SELECT * FROM feat__run_mature", conn)
                avail = [c for c in mature_cols if c in mature.columns]
                base = base.merge(mature[avail], on="row_id", how="left")
            except Exception:
                pass  # feat__run_mature may not exist yet; mart still useful without it

            # Cumulative precipitate at the last available day ≤ stop_date.
            # Use a subquery approach: join on well_key + last date ≤ stop_date.
            try:
                ppt = pd.read_sql(
                    """
                    SELECT p.well_key, r.row_id,
                           p.cum_salt_load_kg, p.cum_gypsum_scale_proxy
                    FROM proc__daily_precipitate p
                    JOIN raw__v03_runs r ON r.well_key = p.well_key
                    WHERE p.dt = (
                        SELECT MAX(p2.dt)
                        FROM proc__daily_precipitate p2
                        WHERE p2.well_key = p.well_key AND p2.dt <= r.stop_date
                    )
                    """,
                    conn,
                )
                if not ppt.empty:
                    base = base.merge(ppt[["row_id", "cum_salt_load_kg", "cum_gypsum_scale_proxy"]], on="row_id", how="left")
            except Exception:
                pass  # proc__daily_precipitate may not exist yet

            timer.row_count = upsert_df(base, "mart__vt_freq55", conn, pk_cols=["row_id"])
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
