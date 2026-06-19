"""Build proc__ttf_true: operational-day TTF per run with techregime-first, telemetry fallback.

Ports compare_ttf_sources() from scripts/analyze_true_ttf.py into the warehouse.
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
from scripts.data_utils import _numeric, load_daily_merged
from scripts.db import StepTimer, get_warehouse_conn, upsert_df


def build_ttf_true(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute proc__ttf_true rows for all runs.

    Replicates compare_ttf_sources() from analyze_true_ttf.py but returns only the
    minimal columns needed by the warehouse table.
    """
    wells = sorted(runs["well"].dropna().astype(str).str.strip().unique().tolist())
    daily = load_daily_merged(wells)
    daily["tele_in_operation"] = (_numeric(daily.get("qliq", pd.Series(index=daily.index, dtype=float))) > 0).fillna(False)

    treg = load_techregime_status_daily(wells)

    rows: list[dict] = []
    for run in runs.to_dict(orient="records"):
        well_id = str(run["well"]).strip()
        install = pd.Timestamp(run["install_date"])
        stop = pd.Timestamp(run["stop_date"])

        run_daily = daily.loc[
            (daily["well_id"].astype("string") == well_id)
            & (daily["dt"] >= install)
            & (daily["dt"] <= stop)
        ].copy()
        tele_days = int(run_daily["dt"].nunique())
        tele_operating_days = int(run_daily.loc[run_daily["tele_in_operation"], "dt"].nunique())

        treg_operating_days = 0
        treg_days = 0
        if not treg.empty:
            treg_run = treg.loc[
                (treg["well_id"].astype("string") == well_id)
                & (treg["dt"] >= install)
                & (treg["dt"] <= stop)
            ].copy()
            treg_days = int(treg_run["dt"].nunique())
            treg_operating_days = int(treg_run.loc[treg_run["treg_in_operation"], "dt"].nunique())

        if treg_operating_days > 0:
            ttf_true_best = float(treg_operating_days)
            ttf_true_source = "techregime_status"
        elif tele_operating_days > 0:
            ttf_true_best = float(tele_operating_days)
            ttf_true_source = "telemetry_qliq_gt_0"
        else:
            ttf_true_best = np.nan
            ttf_true_source = "missing"

        rows.append({
            "row_id": int(run["row_id"]),
            "ttf_true_best_days": ttf_true_best,
            "ttf_true_source": ttf_true_source,
            "ttf_tele_days": float(tele_operating_days) if tele_days > 0 else np.nan,
            "ttf_treg_days": float(treg_operating_days) if treg_days > 0 else np.nan,
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
