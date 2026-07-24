"""True operating-day clock for ESP runs, from daily telemetry.

Why this exists
---------------
The survival fit runs on Свод «Наработка (сут)», which is **not** operating time —
on well-observed runs the true op-days / «Наработка» ratio is ~0.84 (Mc) / ~0.91
(Ya) at the median, with a long tail (p10 ≈ 0.16 for Mc).  Meanwhile the forward
projection consumes `planned_op_days` (real op-days).  Fitting on one clock and
projecting on the other under-states the forecast by roughly the uptime factor.

A leakage-proof landmark test (Кэкспл over days 0–90 → failures after day 90) shows
the failure rate per OPERATING day is constant across utilization (CV 0.028) while
the per-CALENDAR-day rate is not (CV 0.209).  Damage accrues while running.  So
op-days is the physically correct clock, and this module measures it.

The two traps in `proc__daily_operating` — read before using
------------------------------------------------------------
1. **"no data" is coded as `in_operation = 0`** (43% of rows carry
   `op_source='missing'`).  `in_operation` is a perfect proxy for `op_source`, so a
   0 means "no source affirmed operation", never "confirmed idle".
2. **Absent well-days have NO ROW** — the builder outer-joins telemetry and
   techregime, so a day neither source saw simply does not appear.  Median well
   coverage is 0.759.

⇒ Counting rows as the denominator conflates *idle* with *unobserved* and silently
under-states Кэкспл.  This module therefore always uses a **true calendar
denominator**, reports `coverage` explicitly, and falls back to a field-median
ratio when a run is too poorly observed to measure.  Telemetry starts 2018-01-01,
so pre-2018 runs always take the fallback.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from analysis.paths import WAREHOUSE_DIR
from analysis.workflows.production_risk import crosswalk

MIN_COVERAGE = 0.90
MIN_CALENDAR_DAYS = 10
DAY_ZERO = 0.5  # day-0 startup failures are real events, not zero-length runs


def load_daily_operating() -> pd.DataFrame:
    """well-day operating flags, keyed by the canonical well code."""
    with sqlite3.connect(WAREHOUSE_DIR / "pump2.db") as con:
        df = pd.read_sql(
            "SELECT well_key, dt, in_operation FROM proc__daily_operating",
            con,
            parse_dates=["dt"],
        )
    df["code"] = df["well_key"].map(crosswalk.norm_well)
    df["in_operation"] = pd.to_numeric(df["in_operation"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["code"])


def _index_by_code(daily: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        code: (g["dt"].values, g["in_operation"].values)
        for code, g in daily.groupby("code")
    }


def measure(
    pop: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """Attach measured op-days to a population built by :mod:`esp_population`.

    Adds:
      ``cal_days``      true calendar span of the run (NOT a row count)
      ``obs_days``      well-days actually present in the daily table
      ``coverage``      obs_days / cal_days — the honesty check
      ``op_days``       Σ in_operation over the run's calendar span
      ``kexp``          op_days / cal_days (a LOWER bound: unobserved ⇒ counted idle)
      ``tte_op``        the op-day clock: measured when coverage is good, else imputed
      ``tte_op_source`` "measured" | "imputed_field_ratio" | "imputed_global_ratio"
    """
    daily = load_daily_operating() if daily is None else daily
    idx = _index_by_code(daily)
    d0, d1 = daily["dt"].min(), daily["dt"].max()
    as_of = pd.Timestamp(as_of) if as_of is not None else d1

    out = pop.copy()
    end_eff = out["end"].fillna(as_of).clip(upper=as_of)
    cal, obs, opd = [], [], []
    for code, install, end in zip(out["code"], out["install"], end_eff):
        span = int((end - install).days) + 1 if pd.notna(install) and pd.notna(end) else 0
        cal.append(max(span, 0))
        v = idx.get(code)
        if v is None or span <= 0 or install < d0:
            # No telemetry, or the run predates the daily table -> unmeasurable.
            obs.append(0)
            opd.append(np.nan)
            continue
        dts, ops = v
        m = (dts >= np.datetime64(install)) & (dts <= np.datetime64(end))
        obs.append(int(m.sum()))
        opd.append(float(ops[m].sum()))

    out["cal_days"] = cal
    out["obs_days"] = obs
    out["op_days"] = opd
    out["coverage"] = np.where(out["cal_days"] > 0, out["obs_days"] / out["cal_days"], 0.0)
    out["kexp"] = np.where(out["cal_days"] > 0, out["op_days"] / out["cal_days"], np.nan)

    measured = (
        out["op_days"].notna()
        & (out["coverage"] >= min_coverage)
        & (out["cal_days"] >= MIN_CALENDAR_DAYS)
    )
    # Impute the rest by the ratio measured on the well-observed runs of the same
    # field — the ratio is what transfers, not the absolute op-days.
    ratio = np.where(out["tte_full"] > 0, out["op_days"] / out["tte_full"], np.nan)
    out["_ratio"] = np.where(measured, ratio, np.nan)
    field_ratio = out.groupby("field")["_ratio"].transform("median")
    global_ratio = float(np.nanmedian(out["_ratio"])) if measured.any() else 1.0

    tte_op = np.where(measured, out["op_days"], np.nan)
    src = np.where(measured, "measured", "")
    fallback_field = out["tte"] * field_ratio
    use_field = ~measured & field_ratio.notna()
    tte_op = np.where(use_field, fallback_field, tte_op)
    src = np.where(use_field, "imputed_field_ratio", src)
    still = np.isnan(tte_op)
    tte_op = np.where(still, out["tte"] * global_ratio, tte_op)
    src = np.where(still, "imputed_global_ratio", src)

    out["tte_op"] = np.clip(tte_op, 0.5, None)  # keep day-0 startup failures
    out["tte_op_source"] = src
    out = out.drop(columns=["_ratio"])
    if "t_cal" not in out.columns:
        # Order-independent: `measure` may run before `esp_population.add_time_scales`.
        # `cal_days` is the same span (+1 day, inclusive of the install day).
        out["t_cal"] = np.clip(out["cal_days"].astype(float), DAY_ZERO, None)
    return add_op_scales(out, measured)


def add_op_scales(out: pd.DataFrame, measured: pd.Series | np.ndarray) -> pd.DataFrame:
    """Split the op clock into the honest measurement and the servable one.

    ``t_op``        measured operating days, ``NaN`` where telemetry cannot support a
                    measurement (no coverage, run predates the 2018 daily table, span
                    too short).  Never imputed — this is the column to audit against
                    and the one to use when you want *only* real measurements.
    ``t_mix``       ``t_op`` where measured, else ``t_cal * Кэкспл`` — the fit clock.
                    Blank op-time is not optional to handle: measured coverage is only
                    ~40-50% of runs, so a measured-only fit would silently drop half
                    the population (and not at random — poorly-instrumented wells).
    ``t_mix_source`` "measured" | "imputed_field_kexp" | "imputed_global_kexp"
    ``kexp_used``   the ratio actually applied, so any imputed row can be undone.

    The imputation base is **t_cal**, not «Наработка».  The transferable quantity is
    Кэкспл = op-days / calendar-days — a physical utilisation share bounded by 1.
    Imputing off ННО instead (what the legacy ``tte_op`` does) inherits ННО's mixed
    clock and its reporting noise, and the resulting ratio has no physical meaning.
    """
    measured = np.asarray(measured, dtype=bool)
    if "t_cal" not in out.columns:
        raise KeyError("add_op_scales needs t_cal — call esp_population.add_time_scales first")

    out = out.copy()
    out["t_op"] = np.where(measured, out["op_days"], np.nan)

    # Кэкспл measured on well-observed runs only; the median transfers, the absolute does not.
    kexp = np.where(measured & (out["t_cal"] > 0), out["t_op"] / out["t_cal"], np.nan)
    out["_kexp"] = kexp
    field_kexp = out.groupby("field")["_kexp"].transform("median")
    global_kexp = float(np.nanmedian(kexp)) if np.isfinite(kexp).any() else 1.0

    use_field = ~measured & field_kexp.notna()
    ratio = np.where(measured, 1.0, np.where(use_field, field_kexp, global_kexp))
    t_mix = np.where(measured, out["t_op"], out["t_cal"] * ratio)
    src = np.where(measured, "measured",
                   np.where(use_field, "imputed_field_kexp", "imputed_global_kexp"))

    out["t_mix"] = np.clip(t_mix, 0.5, None)   # keep day-0 startup failures
    out["t_mix_source"] = src
    out["kexp_used"] = np.where(measured, out["_kexp"], ratio)
    return out.drop(columns=["_kexp"])


def audit(measured_pop: pd.DataFrame) -> pd.DataFrame:
    """Per-field imputation share and the measured op-day / «Наработка» ratio."""
    rows = []
    for field, g in measured_pop.groupby("field"):
        m = g[g["tte_op_source"] == "measured"]
        rows.append({
            "field": field,
            "runs": len(g),
            "events": int(g["event"].sum()),
            "measured": len(m),
            "measured_share": round(len(m) / max(len(g), 1), 3),
            "median_coverage": round(float(g["coverage"].median()), 3),
            "median_kexp": round(float(m["kexp"].median()), 3) if len(m) else np.nan,
            "median_op_over_svod": (
                round(float((m["op_days"] / m["tte_full"]).median()), 3) if len(m) else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values("runs", ascending=False)
