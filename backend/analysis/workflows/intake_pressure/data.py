"""Assemble the daily panel on which Рприем is reconstructed.

What the warehouse actually supplies (measured 2026-08-05 on ``pump2.db``)
-------------------------------------------------------------------------
``proc__daily_merged`` has 1 251 826 rows and ``rpump_intake`` is positive on only
**56.4 %** of them.  That headline is misleading and acting on it would waste most of the
effort.  Split by whether the well was actually producing:

===========================  =========  ==================
subset                       rows       ``rpump_intake`` >0
===========================  =========  ==================
all rows                     1 251 826  56.4 %
**operating** (qliq>0,freq>0)  584 531  **95.7 %**
idle / stopped                 667 295  ~20 %
===========================  =========  ==================

So the pressure is missing chiefly *because the pump is off*, and on an idle day there
is no intake flow, no gas passing the intake, and nothing the downstream β wants.  The
real target is the **25 336 operating days (4.33 %)** that lack it.  Everything in this
package is therefore defined on the operating-day frame; imputing idle days is declared
out of scope rather than silently attempted.

The gap is bimodal, and that decides the method
-----------------------------------------------
Consecutive missing days form 1 608 blocks within wells:

* **8.1 %** of gap days sit in blocks ≤7 days (median block length is 1 day).  These are
  telemetry dropouts inside an otherwise-instrumented well — neighbouring days carry the
  answer, and within-well interpolation beats any cross-well model.
* **85.4 %** of gap days sit in 126 blocks longer than 30 days, concentrated in **17
  wells** that are 0-50 % covered.  Two — ``vt_5608`` and ``au_315`` — have *no* observed
  intake pressure at all, though only ``vt_5608`` survives into the panel:
  :attr:`PanelSpec.min_days_per_well` drops ``au_315`` at 16 operating days.  Nothing
  within the well helps; this is genuine cross-well extrapolation and it is where the two
  realizations must be judged.

Two consequences are wired into :func:`build_panel` and must not be undone:

**1. The nearest physical predictor is co-missing.**  ``rzab`` (Рзаб) is present on
94 % of *observed* operating days but on only **10.3 %** of the gap.  A model trained
with Рзаб learns to lean on it and then meets a gap where it is absent — the
train/apply covariate mismatch this project has already been bitten by.  So Рзаб is
excluded from the deployable feature set and kept only as a diagnostic; the physics
routes around it through the inflow-performance relation instead.  What *does* survive
into the gap: ``rpl`` 97.3 %, ``watercut`` 99.8 %, ``gas_factor`` 98.3 %, ``qliq`` 100 %,
``kprod`` 32.3 %.

**2. Validation must group by well.**  Daily pressure is near-continuous in time, so a
random row split puts day *t−1* in train and day *t* in test and reports an R² that has
nothing to do with the deployment case (whole missing wells).  :func:`split_groups`
returns well-level folds; :mod:`analysis.workflows.intake_pressure.validate` reports the
random-split number too, purely to show the size of the lie.

Column semantics (confirmed against the physical ordering)
-----------------------------------------------------------
``rpl`` ≥ ``rzab`` ≥ ``rpump_intake`` holds on 98.7 / 96.8 % of guarded operating rows,
which is what fixes the reading Рпл (reservoir) → Рзаб (bottomhole flowing) → Рприем
(intake, up-hole of the perforations and therefore lower).  ``rpl`` is a genuine time
series, not a static annotation — median 52 distinct values per well, within-well
sd 33 atm — unlike ``pbubble_atm``, which this project has established is a *field*
label.  ``vg_m`` is the pump's true vertical depth (always ≤ ``pump_depth_m``, the
measured depth; median difference 295 m).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.features.free_gas import (
    GOR_RANGE,
    P_INTAKE_RANGE,
    P_RES_RANGE,
    WATERCUT_RANGE,
)
from analysis.paths import WAREHOUSE_DIR

#: Guards beyond the ones :mod:`analysis.features.free_gas` already defines.  The raw
#: columns carry impossible values (``rpump_intake`` max 8990 atm, ``rzab`` min ≤0), and
#: an unguarded ``Pzab − Pintake`` spans −3257 … +16 962 atm.  Guarded it is a sane
#: −242 … +559 with median 10.5.
FREQ_RANGE = (1.0, 90.0)            # Hz
QLIQ_RANGE = (0.1, 5000.0)          # m3/d
LOAD_RANGE = (0.0, 200.0)           # % of nominal
KPROD_RANGE = (1e-4, 1000.0)        # m3/d/atm

#: Minimum observed intake-pressure days before a well may calibrate its own physical
#: constants.  Below this the per-well least squares is fitting noise and the pooled
#: field-level fallback is more honest.
MIN_WELL_CALIB_DAYS = 20

DEFAULT_DB = WAREHOUSE_DIR / "pump2.db"

TARGET = "rpump_intake"

#: The feature set deliberately lives in :mod:`analysis.models.ml.intake_pressure_ml`
#: (``DAILY_FEATURES`` / ``EQUIP_FEATURES`` / ``FORBIDDEN``) and **not** here.  Two lists
#: of predictors in two modules is how a forbidden column such as ``rzab`` gets
#: reintroduced on one side and not the other — the exact train/apply mismatch this
#: package is built to avoid.  This module supplies columns; that one decides which are
#: admissible.


def _guard(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.where((s >= lo) & (s <= hi))


@dataclass(frozen=True)
class PanelSpec:
    """Which rows enter the frame.

    Parameters
    ----------
    operating_only
        Keep only days with ``qliq>0`` and ``freq>0``.  Default True — an idle day has no
        intake flow, so neither realization is defined there (see module docstring).
    min_days_per_well
        Drop wells with fewer operating days than this; they cannot support a per-well
        calibration nor a meaningful held-out fold.
    """

    operating_only: bool = True
    min_days_per_well: int = 30


def load_daily(db_path=None, spec: PanelSpec = PanelSpec()) -> pd.DataFrame:
    """Read the guarded daily operating frame joined to run equipment.

    Equipment is joined **as-of** the run: ``feat__run_equipment`` is one row per run, so
    each daily row takes the equipment of the run whose ``install_date`` is the latest one
    at or before that day.  Joining on ``well_key`` alone would smear a workover across
    the pump that preceded it and put the wrong setting depth on half the panel.
    """
    path = str(db_path or DEFAULT_DB)
    con = sqlite3.connect(path)
    try:
        where = "1=1"
        if spec.operating_only:
            where = "qliq > 0 AND freq > 0"
        daily = pd.read_sql(
            f"""SELECT well_key, dt, freq, load, rpl, rpump_intake, rzab,
                       qliq, watercut, gas_factor, qgas, kprod, source
                FROM proc__daily_merged
                WHERE {where}""",
            con,
        )
        equip = pd.read_sql(
            """SELECT well_key, install_date, pump_depth_m, vg_m, q_nom_m3d,
                      head_nom_m, stages, pump_od_mm
               FROM feat__run_equipment""",
            con,
        )
        pb = pd.read_sql(
            """SELECT well_key, field, MAX(pbubble_atm) AS pbubble_atm
               FROM mart__weibull_input GROUP BY well_key, field""",
            con,
        )
    finally:
        con.close()

    daily["dt"] = pd.to_datetime(daily["dt"], errors="coerce")
    equip["install_date"] = pd.to_datetime(equip["install_date"], errors="coerce")
    daily = daily.dropna(subset=["dt", "well_key"])

    equip = equip.dropna(subset=["install_date"]).sort_values("install_date")
    daily = daily.sort_values("dt")
    daily = pd.merge_asof(
        daily,
        equip,
        left_on="dt",
        right_on="install_date",
        by="well_key",
        direction="backward",
    )
    daily = daily.merge(pb, on="well_key", how="left")
    if "field" not in daily.columns:
        daily["field"] = np.nan
    daily["field"] = daily["field"].fillna(
        daily["well_key"].astype(str).str.split("_").str[0]
    )
    return daily


def build_panel(db_path=None, spec: PanelSpec = PanelSpec()) -> pd.DataFrame:
    """Guarded, feature-complete daily panel with the target and its observation mask.

    Adds:

    ``rho_mix_kgm3``
        Liquid-phase mixture density from water cut — the honest part of the annulus
        gradient.  The *gas* in the annulus is what makes the observed gradient far
        lighter than this (median Pzab−Pintake is 10.5 atm across a ~300 m column, where
        a liquid-full column would give ~30), and that is exactly the term the physical
        model calibrates per well rather than assumes.
    ``kpod``
        ``qliq / q_nom_m3d`` — the project's standard loading ratio, restated here so the
        ML side sees pump loading without re-deriving it.
    ``op_day``
        Days since the run's install date.  Carries pressure drift over a run's life.
    ``is_observed``
        True where the guarded target exists.  Both realizations train on it and are
        scored against it; rows where it is False are the deliverable.
    """
    df = load_daily(db_path, spec)

    df["rpump_intake"] = _guard(df["rpump_intake"], *P_INTAKE_RANGE)
    df["rzab"] = _guard(df["rzab"], *P_RES_RANGE)
    df["rpl"] = _guard(df["rpl"], *P_RES_RANGE)
    df["watercut"] = _guard(df["watercut"], *WATERCUT_RANGE)
    df["gas_factor"] = _guard(df["gas_factor"], *GOR_RANGE)
    df["freq"] = _guard(df["freq"], *FREQ_RANGE)
    df["qliq"] = _guard(df["qliq"], *QLIQ_RANGE)
    df["load"] = _guard(df["load"], *LOAD_RANGE)
    df["kprod"] = _guard(df["kprod"], *KPROD_RANGE)

    fw = df["watercut"] / 100.0
    df["qwater"] = df["qliq"] * fw
    df["qoil"] = df["qliq"] * (1.0 - fw)
    df["rho_mix_kgm3"] = 1010.0 * fw + 850.0 * (1.0 - fw)
    df["kpod"] = df["qliq"] / df["q_nom_m3d"].replace(0, np.nan)
    df["op_day"] = (df["dt"] - df["install_date"]).dt.days
    df["month"] = df["dt"].dt.month
    df["is_observed"] = df["rpump_intake"].notna()

    if spec.min_days_per_well > 0:
        keep = df.groupby("well_key")["dt"].transform("size") >= spec.min_days_per_well
        df = df[keep]

    return df.reset_index(drop=True).sort_values(["well_key", "dt"]).reset_index(drop=True)


def gap_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-well observation coverage and the length of its longest missing block.

    The two columns answer different questions: ``coverage`` says whether a per-well
    physical calibration is possible at all, ``max_gap_days`` says whether the missing
    days are dropouts (short — interpolate) or structural (long — extrapolate).
    """
    out = []
    for wk, g in df.groupby("well_key", sort=False):
        miss = ~g["is_observed"].to_numpy()
        blocks, cur = [], 0
        for m in miss:
            if m:
                cur += 1
            elif cur:
                blocks.append(cur)
                cur = 0
        if cur:
            blocks.append(cur)
        out.append(
            {
                "well_key": wk,
                "field": g["field"].iloc[0],
                "n_days": len(g),
                "n_observed": int(g["is_observed"].sum()),
                "coverage": float(g["is_observed"].mean()),
                "n_gap_blocks": len(blocks),
                "max_gap_days": int(max(blocks)) if blocks else 0,
                "calibratable": bool(g["is_observed"].sum() >= MIN_WELL_CALIB_DAYS),
            }
        )
    return pd.DataFrame(out).sort_values("coverage")


def split_groups(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each row a fold index by **well**, balanced on observed-day count.

    Greedy longest-first bin packing rather than a hash: well sizes span 30 … ~2000 days,
    and a hash split leaves folds with wildly different weight, which makes fold-to-fold
    scatter a property of the split instead of the model.  Used for the cold regime, where
    whole wells must move together — see
    :mod:`analysis.workflows.intake_pressure.validate`.
    """
    sizes = (
        df[df["is_observed"]].groupby("well_key").size().sort_values(ascending=False)
    )
    rng = np.random.default_rng(seed)
    order = sizes.index.to_numpy()
    # jitter ties so the packing is not alphabetical
    order = order[rng.permutation(len(order))]
    order = sorted(order, key=lambda w: -sizes[w])

    loads = np.zeros(n_folds)
    assign: dict[str, int] = {}
    for w in order:
        k = int(np.argmin(loads))
        assign[w] = k
        loads[k] += sizes[w]
    for w in df["well_key"].unique():
        assign.setdefault(w, int(rng.integers(n_folds)))
    return df["well_key"].map(assign).to_numpy()


__all__ = [
    "PanelSpec",
    "load_daily",
    "build_panel",
    "gap_profile",
    "split_groups",
    "TARGET",
    "MIN_WELL_CALIB_DAYS",
    "DEFAULT_DB",
]
