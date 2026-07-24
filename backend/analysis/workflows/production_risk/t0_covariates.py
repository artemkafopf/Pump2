"""t0-window covariates for the Ya->Mc Cox transfer.

Only covariates that are computable on BOTH sides survive here:

* fitting side  -- ``proc__daily_merged`` (techregime dailies: ``qgas``, ``qliq``,
  ``watercut``);
* forecast side -- the ТМ-06 plan (``crosswalk.PlanData``: ``gas_volume``,
  ``liquid_volume``, ``oil_volume_m3``, ``op_days``).

A covariate absent from ТМ-06 cannot be deployed on planned wells, so it is out of
scope regardless of significance.  See ``agents/analyses/cox_extended_hazard_handoff.md``
§3.1.

**Why a t0 window and not a run mean.**  Rates are contaminated by the outcome: a pump
that failed on day 20 has a "run mean" over 20 days, and the last of those days are
already degradation (a supply breakdown drops qliq BEFORE the failure).  Regressing on
that recovers the failure read backwards, not a cause.  The window is therefore the
first ``window_op_days`` OPERATING days of the run, frozen as a time-fixed covariate.

**Definitions kept identical to the plan side** (§3.1, trap 1):

* ``glf = qgas / qliq`` -- gas per LIQUID.  The techregime ``gas_factor`` column is gas
  per OIL and diverges with water cut; mixing the two would fit one covariate and deploy
  another.  ``gas_factor`` is loaded only as a control.
* ``wcut`` -- taken from the techregime column directly; the plan side derives it as
  ``(Ql - Qo_m3) / Ql``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .esp_population import WAREHOUSE_DIR

DEFAULT_WINDOW_OP_DAYS = 30

#: Covariates carried to the Cox stage.  ``Qg``/``glf`` are mandatory (user decision).
T0_COVARIATES = ["glf", "qg", "ql", "qo", "wcut"]


@dataclass
class T0Coverage:
    """How much of the population actually got a window -- report, never drop silently."""

    n_runs: int
    n_with_any_daily: int
    n_with_full_window: int
    n_with_glf: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "n_runs": self.n_runs,
                    "n_with_any_daily": self.n_with_any_daily,
                    "n_with_full_window": self.n_with_full_window,
                    "n_with_glf": self.n_with_glf,
                    "share_full_window": self.n_with_full_window / self.n_runs if self.n_runs else np.nan,
                    "share_glf": self.n_with_glf / self.n_runs if self.n_runs else np.nan,
                }
            ]
        )


def _warehouse_conn(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))


def load_dailies(
    well_keys: list[str],
    *,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Daily techregime rows for ``well_keys`` (casefolded well_key space)."""
    if not well_keys:
        return pd.DataFrame(columns=["well_key", "dt", "qliq", "qgas", "watercut", "gas_factor"])
    conn = _warehouse_conn(db_path)
    try:
        placeholders = ",".join("?" * len(well_keys))
        sql = (
            "select well_key, dt, qliq, qgas, watercut, gas_factor "
            "from proc__daily_merged "
            f"where well_key in ({placeholders})"
        )
        df = pd.read_sql(sql, conn, params=list(well_keys))
    finally:
        conn.close()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    return df.dropna(subset=["dt"])


def build(
    pop: pd.DataFrame,
    *,
    window_op_days: int = DEFAULT_WINDOW_OP_DAYS,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, T0Coverage]:
    """Attach t0-window covariates to ``pop`` (an ``esp_population`` selection).

    Returns ``(pop_with_covariates, coverage)``.  Runs whose window is missing or
    incomplete keep NaN covariates -- the caller reports coverage rather than dropping,
    because short runs are exactly the infant failures and dropping them biases c=0.
    """
    out = pop.copy()
    out["well_key"] = out["code"].astype(str).str.casefold()

    dailies = load_dailies(sorted(out["well_key"].unique().tolist()), db_path=db_path)
    # An operating day is a day the pump actually lifted fluid.  Mirrors build_ttf_true's
    # telemetry arm (qliq > 0); the techregime "В работе" arm is not needed here because a
    # day with no flow carries no rate to average anyway.
    dailies = dailies[dailies["qliq"].notna() & (dailies["qliq"] > 0)]
    by_well = {k: g.sort_values("dt") for k, g in dailies.groupby("well_key", sort=False)}

    rows: list[dict] = []
    n_any = n_full = 0
    for _, run in out.iterrows():
        g = by_well.get(run["well_key"])
        rec: dict = {"_idx": run.name}
        if g is None or pd.isna(run["install"]):
            rows.append(rec)
            continue
        install = pd.Timestamp(run["install"])
        stop = pd.Timestamp(run["end"]) if pd.notna(run["end"]) else pd.Timestamp.max
        win = g[(g["dt"] >= install) & (g["dt"] <= stop)]
        if win.empty:
            rows.append(rec)
            continue
        n_any += 1
        win = win.head(window_op_days)
        rec["t0_n_days"] = int(len(win))
        if len(win) >= window_op_days:
            n_full += 1
        ql = float(win["qliq"].mean())
        qg = float(win["qgas"].mean()) if win["qgas"].notna().any() else np.nan
        wc = float(win["watercut"].mean()) if win["watercut"].notna().any() else np.nan
        rec["ql"] = ql
        rec["qg"] = qg
        rec["wcut"] = wc
        rec["glf"] = qg / ql if (ql > 0 and np.isfinite(qg)) else np.nan
        # Techregime has no qoil column; oil rate follows from liquid and water cut.
        rec["qo"] = ql * (1.0 - wc / 100.0) if np.isfinite(wc) else np.nan
        # Control only -- gas per OIL, must not be used as glf (see module docstring).
        rec["gas_factor_ctl"] = (
            float(win["gas_factor"].mean()) if win["gas_factor"].notna().any() else np.nan
        )
        rows.append(rec)

    feat = pd.DataFrame(rows).set_index("_idx")
    out = out.join(feat)
    cov = T0Coverage(
        n_runs=int(len(out)),
        n_with_any_daily=int(n_any),
        n_with_full_window=int(n_full),
        n_with_glf=int(out["glf"].notna().sum()) if "glf" in out.columns else 0,
    )
    return out, cov


def plan_covariates(plan, months: list[str] | None = None) -> pd.DataFrame:
    """Deployment-side counterpart: per-well t0 covariates from the ТМ-06 plan.

    Monthly plan volumes are divided by ``Отработанное время`` to become per-operating-day
    rates -- otherwise a monthly volume would be compared against a daily rate.  Units are
    already reconciled: ``gas_volume`` is scaled to м3 at parse time, so ``glf`` is
    directly comparable to the fitting side (measured plan/hist ratio 0.84 on 606 wells).
    """
    cols = list(months or plan.fwd_months)
    op = plan.op_days[cols].sum(axis=1)
    gas = plan.gas_volume[cols].sum(axis=1)
    liq = plan.liquid_volume[cols].sum(axis=1)
    oil_m3 = plan.oil_volume_m3[cols].sum(axis=1)

    ok = op > 0
    df = pd.DataFrame(index=plan.gas_volume.index)
    df["qg"] = np.where(ok, gas / op.where(ok, np.nan), np.nan)
    df["ql"] = np.where(ok, liq / op.where(ok, np.nan), np.nan)
    df["glf"] = np.where(df["ql"] > 0, df["qg"] / df["ql"], np.nan)
    # wcut needs oil in m3: subtracting тн from м3 would bias it by oil density (~0.85).
    df["wcut"] = np.where(
        (liq > 0) & (oil_m3 > 0), (liq - oil_m3) / liq.where(liq > 0, np.nan) * 100.0, np.nan
    )
    df["qo"] = np.where(ok & (oil_m3 > 0), oil_m3 / op.where(ok, np.nan), np.nan)
    return df
