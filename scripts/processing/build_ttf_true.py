"""Build proc__ttf_true: TTF variants per run.

TTF columns:
  ttf_true_best_days  — productive operating days (qliq > 0), union of tele + treg sources
  ttf_motor_days      — motor-running days (freq > STANDBY_FREQ_HZ OR treg "В работе"),
                        includes standby; represents actual wear accumulation
  ttf_tele_days       — productive days per telemetry qliq only (transparency)
  ttf_treg_days       — productive days per techregime status only (transparency)
  ttf_union_days      — alias for ttf_true_best_days (kept for downstream compatibility)
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

from scripts.analyze_true_ttf import load_techregime_status_daily
from scripts.data_utils import _load_telemetry_daily, _numeric
from scripts.db import StepTimer, get_warehouse_conn, upsert_df

# Frequency above this threshold means motor is running (excludes standby ~33-35 Hz).
STANDBY_FREQ_HZ: float = 40.0


def build_ttf_true(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute proc__ttf_true rows for all runs.

    Replicates compare_ttf_sources() from analyze_true_ttf.py but returns only the
    minimal columns needed by the warehouse table.
    """
    wells = sorted(runs["well"].dropna().astype(str).str.strip().unique().tolist())
    tel_daily = _load_telemetry_daily(wells)
    tel_daily["tele_in_operation"] = (_numeric(tel_daily.get("qliq", pd.Series(index=tel_daily.index, dtype=float))) > 0).fillna(False)
    tel_daily["motor_on"] = (_numeric(tel_daily.get("freq", pd.Series(index=tel_daily.index, dtype=float))) > STANDBY_FREQ_HZ).fillna(False)

    treg = load_techregime_status_daily(wells)

    rows: list[dict] = []
    for run in runs.to_dict(orient="records"):
        well_id = str(run["well"]).strip()
        install = pd.Timestamp(run["install_date"])
        stop = pd.Timestamp(run["stop_date"])

        tel_run = tel_daily.loc[
            (tel_daily["well_id"].astype("string") == well_id)
            & (tel_daily["dt"] >= install)
            & (tel_daily["dt"] <= stop)
        ].copy()
        tele_days = int(tel_run["dt"].nunique())

        treg_run = pd.DataFrame()
        treg_days = 0
        if not treg.empty:
            treg_run = treg.loc[
                (treg["well_id"].astype("string") == well_id)
                & (treg["dt"] >= install)
                & (treg["dt"] <= stop)
            ].copy()
            treg_days = int(treg_run["dt"].nunique())

        treg_op_dates = set(treg_run.loc[treg_run["treg_in_operation"], "dt"].dt.normalize()) if not treg_run.empty else set()
        tele_op_dates = set(tel_run.loc[tel_run["tele_in_operation"], "dt"].dt.normalize())
        motor_on_dates = set(tel_run.loc[tel_run["motor_on"], "dt"].dt.normalize())

        treg_operating_days = len(treg_op_dates)
        tele_operating_days = len(tele_op_dates)
        union_op_days = len(treg_op_dates | tele_op_dates)
        motor_days = len(motor_on_dates | treg_op_dates)  # freq>40Hz OR techregime "В работе"

        ttf_true_best = float(union_op_days) if union_op_days > 0 else np.nan
        ttf_true_source = (
            "union" if treg_op_dates and tele_op_dates
            else "techregime" if treg_op_dates
            else "telemetry" if tele_op_dates
            else "missing"
        )

        rows.append({
            "row_id": int(run["row_id"]),
            "ttf_true_best_days": ttf_true_best,
            "ttf_true_source": ttf_true_source,
            "ttf_tele_days": float(tele_operating_days) if tele_days > 0 else np.nan,
            "ttf_treg_days": float(treg_operating_days) if treg_days > 0 else np.nan,
            "ttf_union_days": float(union_op_days) if union_op_days > 0 else np.nan,
            "ttf_motor_days": float(motor_days) if motor_days > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def run(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_warehouse_conn()
    try:
        runs = pd.read_sql("SELECT row_id, well, install_date, stop_date FROM raw__v03_runs", conn)
        if runs.empty:
            print("[build_ttf_true] raw__v03_runs is empty — run ingest_v03 first.")
            return

        with StepTimer("proc__ttf_true", conn) as timer:
            result = build_ttf_true(runs)
            timer.row_count = upsert_df(result, "proc__ttf_true", conn, pk_cols=["row_id"])
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run()
