"""Build mart__weibull_input: one row per run for weibull.py (project-data mode).

Joins raw__v03_runs + proc__ttf_true + proc__h2s_proxy + feat__run_mature.
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
        with StepTimer("mart__weibull_input", conn) as timer:
            base = pd.read_sql(
                """
                SELECT row_id, well, well_key, field, pad, contractor,
                       install_date, stop_date, event, run_days,
                       pump_type, nominal_flow_m3d, nominal_freq_hz, pbubble_atm
                FROM raw__v03_runs
                """,
                conn,
                parse_dates=["install_date", "stop_date"],
            )

            for tbl, id_col, cols in [
                ("proc__ttf_true", "row_id", ["row_id", "ttf_true_best_days", "ttf_true_source"]),
                ("proc__h2s_proxy", "row_id", ["row_id", "h2s_proxy_mg_l", "h2s_proxy_source"]),
            ]:
                try:
                    df = pd.read_sql(f"SELECT * FROM {tbl}", conn)[cols]
                    base = base.merge(df, on="row_id", how="left")
                except Exception:
                    pass

            try:
                mature = pd.read_sql("SELECT * FROM feat__run_mature", conn)
                base = base.merge(mature, on="row_id", how="left", suffixes=("", "_m"))
            except Exception:
                pass

            timer.row_count = upsert_df(base, "mart__weibull_input", conn, pk_cols=["row_id"])
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
