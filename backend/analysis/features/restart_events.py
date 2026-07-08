"""Reconciled pump restart / on-off events (directive #4, 2026-07-08).

Phase D inferred restarts from telemetry ``qliq: 0 → >0`` alone, which cannot tell a
genuine shutdown from a telemetry outage (sensor/SCADA down while the pump runs). The
techregime «Состояние» field carries an administrative operating status («В работе»)
from an independent source; using it as a second opinion removes the outage false
positives:

  operating(day) := (qliq > 0)  OR  (techregime status == "В работе")

so a day counts as *down* only when **both** sources agree it is down (or the sole
present source says down). A raw qliq restart whose gap techregime spanned as
"В работе" is reclassified as an outage, not a restart.

Primary function :func:`reconcile_restarts` works on one run's merged daily frame and is
pure/tested. :func:`build_restart_features` runs it fleet-wide off the warehouse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _count_up_transitions(op: np.ndarray) -> int:
    """Number of False→True transitions in a boolean sequence (observed days only)."""
    if len(op) < 2:
        return 0
    prev = op[:-1]
    cur = op[1:]
    return int(np.sum((~prev) & cur))


def reconcile_restarts(frame: pd.DataFrame) -> dict:
    """Reconcile restarts for a single run.

    ``frame`` columns (one row per observed day within the run window, any subset of
    sources present per day):
      * ``qliq``            — telemetry liquid rate (NaN if no telemetry that day)
      * ``treg_in_operation`` — techregime "В работе" flag (NaN if no techregime that day)

    Returns per-run counts:
      n_restarts_raw        — telemetry-only qliq 0→>0 (the Phase D definition)
      n_restarts_reconciled — after OR-combining with techregime (outages removed)
      n_outages_removed     — raw − reconciled (qliq restarts techregime covered)
      n_restarts_both       — reconciled restarts confirmed by both sources on the up-day
      restart_reliable      — True if the two sources broadly agree (see below)
      n_op_days_tel / _treg / _union — operating-day tallies for transparency
    """
    if frame.empty:
        return {
            "n_restarts_raw": 0, "n_restarts_reconciled": 0, "n_outages_removed": 0,
            "n_restarts_both": 0, "restart_reliable": False,
            "n_op_days_tel": 0, "n_op_days_treg": 0, "n_op_days_union": 0,
        }
    df = frame.sort_values("dt") if "dt" in frame.columns else frame.copy()

    if "qliq" in df.columns:
        qliq = pd.to_numeric(df["qliq"], errors="coerce")
        op_tel = qliq.gt(0).where(qliq.notna(), other=pd.NA)          # True/False/NA
    else:
        op_tel = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if "treg_in_operation" in df.columns:
        op_treg = df["treg_in_operation"].astype("boolean")
    else:
        op_treg = pd.Series(pd.NA, index=df.index, dtype="boolean")

    tel_true = op_tel.fillna(False).to_numpy(dtype=bool)
    treg_true = op_treg.fillna(False).to_numpy(dtype=bool)
    has_info = (op_tel.notna() | op_treg.notna()).to_numpy(dtype=bool)

    # raw = telemetry-only, over days where telemetry is present
    tel_present = op_tel.notna().to_numpy(dtype=bool)
    n_raw = _count_up_transitions(tel_true[tel_present]) if tel_present.any() else 0

    # reconciled = OR-combined operating state over days with any info
    op_union = (tel_true | treg_true)[has_info]
    n_rec = _count_up_transitions(op_union) if has_info.any() else 0

    # "both"-confirmed up-days: qliq>0 AND treg in operation, following a down day
    both_true = (tel_true & treg_true)[has_info]
    prev_down = ~op_union[:-1] if len(op_union) > 1 else np.array([], dtype=bool)
    n_both = int(np.sum(prev_down & both_true[1:])) if len(both_true) > 1 else 0

    n_op_tel = int(tel_true[tel_present].sum())
    n_op_treg = int(treg_true[op_treg.notna().to_numpy(dtype=bool)].sum())
    n_op_union = int(op_union.sum())

    # reliability: telemetry and techregime operating-day counts within 15% where both exist
    both_days = (op_tel.notna() & op_treg.notna()).to_numpy(dtype=bool)
    reliable = False
    if both_days.sum() >= 20:
        a = int(tel_true[both_days].sum())
        b = int(treg_true[both_days].sum())
        denom = max(a, b, 1)
        reliable = abs(a - b) / denom <= 0.15

    return {
        "n_restarts_raw": int(n_raw),
        "n_restarts_reconciled": int(n_rec),
        "n_outages_removed": int(max(n_raw - n_rec, 0)),
        "n_restarts_both": int(n_both),
        "restart_reliable": bool(reliable),
        "n_op_days_tel": n_op_tel,
        "n_op_days_treg": n_op_treg,
        "n_op_days_union": n_op_union,
    }


def _load_techregime_operation(well_keys: list[str]) -> pd.DataFrame:
    """Daily «В работе» flag per (normalized well_key, dt) from the techregime DB.

    Self-contained (avoids a pandas-version quirk in scripts.analyze_true_ttf); the
    raw techregime well id is normalized with the same key as the warehouse so it
    joins to proc__daily_merged / raw__v03_runs.
    """
    import sqlite3

    from analysis.paths import resolve_techregime_db_path
    from scripts.data_utils import normalize_well_key

    tr_path = resolve_techregime_db_path()
    if not Path(tr_path).exists():
        return pd.DataFrame(columns=["well_key", "dt", "treg_in_operation"])
    con = sqlite3.connect(tr_path)
    raw = pd.read_sql(
        "SELECT col_0003 AS well_raw, col_0010 AS dt, col_0004 AS status FROM techregime_records",
        con)
    con.close()
    raw["well_key"] = raw["well_raw"].map(normalize_well_key)
    raw["dt"] = pd.to_datetime(raw["dt"], errors="coerce", dayfirst=True)
    raw = raw[raw["well_key"].isin(set(well_keys)) & raw["dt"].notna()]
    raw["op"] = raw["status"].astype("string").str.strip().eq("В работе")
    # one flag per well-day: operating if any record that day says so
    g = raw.groupby(["well_key", "dt"], as_index=False)["op"].max()
    g = g.rename(columns={"op": "treg_in_operation"})
    return g


def build_restart_features() -> pd.DataFrame:
    """Fleet-wide reconciled restart counts, one row per raw__v03_runs run."""
    import sqlite3

    from analysis.paths import WAREHOUSE_DIR

    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    runs = pd.read_sql(
        "SELECT row_id, well_key, install_date, stop_date FROM raw__v03_runs", con)
    runs["install_date"] = pd.to_datetime(runs["install_date"])
    runs["stop_date"] = pd.to_datetime(runs["stop_date"])
    daily = pd.read_sql(
        "SELECT well_key, dt, qliq FROM proc__daily_merged WHERE source='telemetry'", con)
    con.close()
    daily["dt"] = pd.to_datetime(daily["dt"])

    treg = _load_techregime_operation(sorted(runs["well_key"].dropna().unique().tolist()))

    tel_by_well = {w: g for w, g in daily.groupby("well_key")}
    treg_by_well = {w: g for w, g in treg.groupby("well_key")}

    out = []
    for r in runs.itertuples(index=False):
        t = tel_by_well.get(r.well_key)
        g = treg_by_well.get(r.well_key)
        parts = []
        if t is not None:
            parts.append(t[(t.dt >= r.install_date) & (t.dt <= r.stop_date)][["dt", "qliq"]])
        if g is not None:
            parts.append(g[(g.dt >= r.install_date) & (g.dt <= r.stop_date)][["dt", "treg_in_operation"]])
        if parts:
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.merge(p, on="dt", how="outer")
        else:
            merged = pd.DataFrame(columns=["dt", "qliq", "treg_in_operation"])
        summ = reconcile_restarts(merged)
        summ["row_id"] = int(r.row_id)
        out.append(summ)
    return pd.DataFrame(out)


__all__ = ["reconcile_restarts", "build_restart_features"]
