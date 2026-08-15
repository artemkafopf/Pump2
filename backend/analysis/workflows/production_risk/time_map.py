"""Data-based calendar-to-operating-day map for production-risk replay.

The survival baseline is fitted on operating days.  Historical intervals and
future plans arrive in calendar months, so this module provides a small,
bundle-loadable map from calendar month + operating age to expected operating
exposure.  Missing bundles are neutral: callers keep the legacy behaviour.
"""
from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import WAREHOUSE_DIR
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk


AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("000_030", 0.0, 30.0),
    ("030_090", 30.0, 90.0),
    ("090_180", 90.0, 180.0),
    ("180_365", 180.0, 365.0),
    ("365_730", 365.0, 730.0),
    ("730_inf", 730.0, np.inf),
)

MIN_FIELD_MONTH_ROWS = 20
MIN_FIELD_AGE_ROWS = 30


@dataclass(frozen=True)
class IntervalSlice:
    month: str
    calendar_days: float
    op_days: float
    age_start: float
    producing: bool | None
    source: str


class TimeMap:
    """Bundle-backed uptime and idle-hazard lookup.

    ``calendar_month`` is 1..12 for seasonal rows and 0 for all-month fallback.
    ``field`` is the model field code (Ya/Vt/Mc/...) or ``GLOBAL``.
    """

    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = frame.copy() if frame is not None else pd.DataFrame()
        if self.frame.empty:
            self.uptime = pd.DataFrame()
            self.idle = pd.DataFrame()
            self._uptime_lookup: dict[tuple[str, str, int], float] = {}
            self._idle_lookup: dict[str, float] = {}
            return
        self.uptime = self.frame[self.frame["row_type"].eq("uptime")].copy()
        self.idle = self.frame[self.frame["row_type"].eq("idle_hazard")].copy()
        # Precompute O(1) lookups once: predict_uptime/idle_fraction are called hundreds of
        # thousands of times during a replay, and per-call DataFrame boolean-mask scans were
        # the dominant cost.  These dicts preserve the exact first-match precedence.
        self._uptime_lookup = self._build_uptime_lookup(self.uptime)
        self._idle_lookup = self._build_idle_lookup(self.idle)

    @staticmethod
    def _build_uptime_lookup(uptime: pd.DataFrame) -> dict[tuple[str, str, int], float]:
        lookup: dict[tuple[str, str, int], float] = {}
        if uptime.empty:
            return lookup
        months = pd.to_numeric(uptime["calendar_month"], errors="coerce").fillna(-1).astype(int)
        values = pd.to_numeric(uptime["uptime"], errors="coerce")
        for field, band, cmonth, val in zip(
            uptime["field"].astype(str), uptime["age_band"].astype(str), months, values
        ):
            # first row wins, matching the original ``.iloc[0]`` (uptime is not dropna'd)
            lookup.setdefault((field, band, int(cmonth)), float(np.clip(val, 0.0, 1.0)))
        return lookup

    @staticmethod
    def _build_idle_lookup(idle: pd.DataFrame) -> dict[str, float]:
        lookup: dict[str, float] = {}
        if idle.empty:
            return lookup
        values = pd.to_numeric(idle["idle_hazard_fraction"], errors="coerce")
        for field, val in zip(idle["field"].astype(str), values):
            if not np.isfinite(val):  # matches the original ``.dropna()`` (first non-null wins)
                continue
            lookup.setdefault(field, float(np.clip(val, 0.0, 1.0)))
        return lookup

    @classmethod
    def missing(cls) -> "TimeMap":
        return cls(None)

    @classmethod
    def from_csv(cls, path: Path | None) -> "TimeMap":
        if path is None or not Path(path).exists():
            return cls.missing()
        df = pd.read_csv(path, encoding="utf-8-sig")
        return cls(df)

    @property
    def available(self) -> bool:
        return not self.uptime.empty

    def predict_uptime(
        self,
        model_field: str | None,
        age_start: float,
        calendar_month: int,
        *,
        fallback: float,
    ) -> tuple[float, str]:
        if not self._uptime_lookup:
            return float(np.clip(fallback, 0.0, 1.0)), "legacy_uptime_factor"
        field = str(model_field or "GLOBAL")
        age_band = age_band_label(age_start)
        cm = int(calendar_month)
        candidates = (
            (field, age_band, cm),
            (field, age_band, 0),
            (field, "ALL", cm),
            (field, "ALL", 0),
            ("GLOBAL", age_band, cm),
            ("GLOBAL", age_band, 0),
            ("GLOBAL", "ALL", cm),
            ("GLOBAL", "ALL", 0),
        )
        for key in candidates:
            value = self._uptime_lookup.get(key)
            if value is not None:
                return value, f"map:{key[0]}:{key[1]}:{key[2]}"
        return float(np.clip(fallback, 0.0, 1.0)), "legacy_uptime_factor"

    def idle_fraction(self, reporting_field: str | None, model_field: str | None) -> tuple[float, str]:
        if not self._idle_lookup:
            return 0.0, "none"
        for key in (str(reporting_field or ""), str(model_field or ""), "GLOBAL"):
            if not key:
                continue
            value = self._idle_lookup.get(key)
            if value is not None:
                return value, f"idle:{key}"
        return 0.0, "none"


def age_band_label(age_start: float) -> str:
    age = max(0.0, float(age_start)) if np.isfinite(float(age_start)) else 0.0
    for label, lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return label
    return AGE_BANDS[-1][0]


@lru_cache(maxsize=4096)
def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    p = pd.Period(month, freq="M")
    return p.start_time, p.start_time + pd.offsets.MonthBegin(1)


@lru_cache(maxsize=4096)
def _month_meta(month: str) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    """(month_start, month_end, calendar_month 1..12, days_in_month) — cached per month."""
    p = pd.Period(month, freq="M")
    return p.start_time, p.start_time + pd.offsets.MonthBegin(1), int(p.month), int(p.days_in_month)


def interval_slices(
    *,
    months: list[str],
    start: datetime,
    end: datetime,
    code: str,
    reporting_field: str,
    model_field: str | None,
    total_op: float | None,
    fallback_uptime: float,
    time_map: TimeMap,
    plan=None,
    active: pd.DataFrame | None = None,
) -> list[IntervalSlice]:
    """Build mapped monthly exposure slices for one observed interval.

    If no time map is available, this intentionally returns an empty list so the
    caller can keep the legacy branch byte-identical.
    """
    if not time_map.available:
        return []

    # Only iterate months that can overlap [start, end]; ``months`` is sorted "YYYY-MM"
    # strings, so the calendar month of start/end bounds the relevant slice.  Months
    # outside this range have overlap <= 0 and contributed nothing (byte-identical).
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    lo = bisect.bisect_left(months, ts_start.strftime("%Y-%m"))
    hi = bisect.bisect_right(months, ts_end.strftime("%Y-%m"))
    relevant_months = months[lo:hi]

    plan_months = getattr(plan, "months", ()) if plan is not None else ()
    plan_months_set = set(plan_months)
    op_frame = getattr(plan, "op_days", getattr(plan, "op_days_raw", None)) if plan is not None else None
    cal_days_series = getattr(plan, "cal_days", None) if plan is not None else None
    # idle fraction depends only on (reporting_field, model_field) -> constant per interval
    idle_frac, idle_source = time_map.idle_fraction(reporting_field, model_field)

    raw: list[dict] = []
    running_age = 0.0
    for month in relevant_months:
        m_start, m_end, cal_month, days_in_month = _month_meta(month)
        ov_start = max(ts_start, m_start)
        ov_end = min(ts_end, m_end)
        overlap = float((ov_end - ov_start).days)
        if overlap <= 0:
            continue

        in_plan_month = month in plan_months_set
        producing: bool | None = None
        plan_op = np.nan
        if in_plan_month:
            producing = bool(
                active is not None
                and code in active.index
                and month in active.columns
                and bool(active.at[code, month])
            )
            try:
                cal_days = float(cal_days_series.get(month, days_in_month)) if cal_days_series is not None else float(days_in_month)
                month_op = float(op_frame.at[code, month]) if op_frame is not None and code in op_frame.index and month in op_frame.columns else 0.0
                plan_op = month_op * overlap / cal_days if cal_days > 0 else 0.0
            except Exception:
                plan_op = np.nan

        uptime, source = time_map.predict_uptime(
            model_field,
            running_age,
            cal_month,
            fallback=fallback_uptime,
        )
        if in_plan_month and producing is True and np.isfinite(plan_op):
            weight = max(0.0, float(plan_op))
            source = "plan_actual_op_days"
        elif in_plan_month and producing is False:
            weight = max(0.0, overlap * uptime * idle_frac)
            source = f"{source}+{idle_source}"
        else:
            weight = max(0.0, overlap * uptime)
        raw.append(
            {
                "month": month,
                "calendar_days": overlap,
                "weight": weight,
                "producing": producing,
                "source": source,
            }
        )
        running_age += max(0.0, weight)

    if not raw:
        return []
    weights = np.array([r["weight"] for r in raw], dtype=float)
    if total_op is not None and np.isfinite(float(total_op)) and float(total_op) > 0 and weights.sum() > 0:
        op = weights * (float(total_op) / weights.sum())
    else:
        op = weights

    out: list[IntervalSlice] = []
    age = 0.0
    for rec, op_days in zip(raw, op):
        if op_days <= 0:
            continue
        out.append(
            IntervalSlice(
                month=str(rec["month"]),
                calendar_days=float(rec["calendar_days"]),
                op_days=float(op_days),
                age_start=float(age),
                producing=rec["producing"],
                source=str(rec["source"]),
            )
        )
        age += float(op_days)
    return out


def load_daily_operating(db_path: Path | None = None) -> pd.DataFrame:
    db = Path(db_path or (WAREHOUSE_DIR / "pump2.db"))
    con = sqlite3.connect(db)
    try:
        return pd.read_sql(
            """
            SELECT o.well_key, o.dt, o.in_operation, m.qliq
            FROM proc__daily_operating o
            LEFT JOIN proc__daily_merged m
              ON o.well_key = m.well_key AND o.dt = m.dt
            """,
            con,
            parse_dates=["dt"],
        )
    finally:
        con.close()


def build_run_month_dataset(population: pd.DataFrame, daily: pd.DataFrame, plan=None) -> pd.DataFrame:
    pop = population.copy()
    pop["well_code"] = pop["well_key"].map(lambda x: crosswalk.norm_well(x) or "").str.upper()
    pop["model_field"] = pop["well_code"].map(crosswalk.map_model_field_from_well)
    pop["install_date"] = pd.to_datetime(pop["install_date"], errors="coerce")
    pop["end_date"] = pd.to_datetime(pop["end_date"], errors="coerce")
    pop["tte"] = pd.to_numeric(pop["tte"], errors="coerce")

    d = daily.copy()
    d["well_code"] = d["well_key"].map(lambda x: crosswalk.norm_well(x) or "").str.upper()
    d["dt"] = pd.to_datetime(d["dt"], errors="coerce")
    d["in_operation"] = pd.to_numeric(d["in_operation"], errors="coerce").fillna(0).astype(int)
    d["qliq_positive"] = pd.to_numeric(d["qliq"], errors="coerce").fillna(0.0) > 0.0
    daily_by_well = {k: g.sort_values("dt") for k, g in d.dropna(subset=["dt"]).groupby("well_code", sort=False)}

    plan_active = None
    if plan is not None:
        op = plan.op_days_raw
        oil = plan.oil_volume.reindex(index=op.index, columns=op.columns, fill_value=0.0)
        liq = plan.liquid_volume.reindex(index=op.index, columns=op.columns, fill_value=0.0)
        plan_active = (op > 0) & ((oil > 0) | (liq > 0))

    rows: list[dict] = []
    cutoff = pd.Timestamp(C.DEFAULT_FACT_THROUGH_MONTH) + pd.offsets.MonthEnd(0) + pd.offsets.Day(1)
    for run_idx, r in pop.dropna(subset=["install_date", "tte"]).iterrows():
        start = pd.Timestamp(r["install_date"])
        end = pd.Timestamp(r["end_date"]) if pd.notna(r["end_date"]) else cutoff
        end = min(end, cutoff)
        if end <= start:
            continue
        code = str(r["well_code"])
        g = daily_by_well.get(code, pd.DataFrame())
        run_daily = g[(g["dt"] >= start) & (g["dt"] < end)].copy() if not g.empty else pd.DataFrame()
        cum_op = 0.0
        for p in pd.period_range(start.to_period("M"), (end - pd.Timedelta(days=1)).to_period("M"), freq="M"):
            ms, me = p.start_time, p.start_time + pd.offsets.MonthBegin(1)
            ov_start = max(start, ms)
            ov_end = min(end, me)
            cal = float((ov_end - ov_start).days)
            if cal <= 0:
                continue
            md = run_daily[(run_daily["dt"] >= ov_start) & (run_daily["dt"] < ov_end)] if not run_daily.empty else pd.DataFrame()
            treg_op = float(md["in_operation"].sum()) if not md.empty else np.nan
            qliq_op = float(md["qliq_positive"].sum()) if not md.empty else np.nan
            coverage = float(len(md) / cal) if cal > 0 else 0.0
            month = p.strftime("%Y-%m")
            producing = np.nan
            if plan_active is not None and code in plan_active.index and month in plan_active.columns:
                producing = bool(plan_active.at[code, month])
            rows.append(
                {
                    "run_idx": int(run_idx),
                    "well_code": code,
                    "field": r.get("field"),
                    "model_field": r.get("model_field"),
                    "stratum": r.get("stratum"),
                    "source": r.get("source"),
                    "install_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "month": month,
                    "calendar_month": int(p.month),
                    "calendar_days": cal,
                    "age_op_start_observed": cum_op,
                    "daily_coverage": coverage,
                    "op_days_treg": treg_op,
                    "op_days_qliq": qliq_op,
                    "tte": float(r["tte"]),
                    "event": int(r.get("event", 0)),
                    "is_failure_month": bool(int(r.get("event", 0)) == 1 and pd.notna(r.get("end_date")) and pd.Timestamp(r["end_date"]).strftime("%Y-%m") == month),
                    "plan_producing": producing,
                }
            )
            if np.isfinite(treg_op):
                cum_op += max(0.0, treg_op)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["age_band"] = df["age_op_start_observed"].map(age_band_label)
    df["uptime_treg"] = np.divide(
        df["op_days_treg"].to_numpy(dtype=float),
        df["calendar_days"].to_numpy(dtype=float),
        out=np.full(len(df), np.nan),
        where=df["calendar_days"].to_numpy(dtype=float) > 0,
    )
    df["uptime_qliq"] = np.divide(
        df["op_days_qliq"].to_numpy(dtype=float),
        df["calendar_days"].to_numpy(dtype=float),
        out=np.full(len(df), np.nan),
        where=df["calendar_days"].to_numpy(dtype=float) > 0,
    )
    return df


def _agg_uptime(df: pd.DataFrame, field: str, age_band: str, calendar_month: int, rows: pd.DataFrame) -> dict:
    uptime = float(rows["uptime_treg"].clip(0, 1).mean())
    return {
        "row_type": "uptime",
        "field": field,
        "age_band": age_band,
        "calendar_month": int(calendar_month),
        "uptime": round(uptime, 6),
        "n_run_months": int(len(rows)),
        "idle_hazard_fraction": np.nan,
        "source": "treg_in_operation",
    }


def fit_time_map(run_months: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = run_months[
        run_months["daily_coverage"].ge(0.5)
        & run_months["uptime_treg"].notna()
        & run_months["model_field"].notna()
    ].copy()
    if fit.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    rows.append(_agg_uptime(fit, "GLOBAL", "ALL", 0, fit))
    for (band, month), g in fit.groupby(["age_band", "calendar_month"]):
        if len(g) >= MIN_FIELD_MONTH_ROWS:
            rows.append(_agg_uptime(fit, "GLOBAL", str(band), int(month), g))
    for band, g in fit.groupby("age_band"):
        if len(g) >= MIN_FIELD_AGE_ROWS:
            rows.append(_agg_uptime(fit, "GLOBAL", str(band), 0, g))

    for field, fg in fit.groupby("model_field"):
        rows.append(_agg_uptime(fit, str(field), "ALL", 0, fg))
        for band, bg in fg.groupby("age_band"):
            if len(bg) >= MIN_FIELD_AGE_ROWS:
                rows.append(_agg_uptime(fit, str(field), str(band), 0, bg))
            for month, mg in bg.groupby("calendar_month"):
                if len(mg) >= MIN_FIELD_MONTH_ROWS:
                    rows.append(_agg_uptime(fit, str(field), str(band), int(month), mg))

    time_map = pd.DataFrame(rows)
    metrics = reconstruction_metrics(fit, time_map)
    return time_map, metrics


def _predict_rows_with_map(rows: pd.DataFrame, tmap: TimeMap, field: str | None = None) -> np.ndarray:
    out = []
    for _, r in rows.iterrows():
        f = field if field is not None else r.get("model_field")
        u, _ = tmap.predict_uptime(f, float(r.get("age_op_start_observed", 0.0)), int(r["calendar_month"]), fallback=1.0)
        out.append(float(r["calendar_days"]) * u)
    return np.asarray(out, dtype=float)


def reconstruction_metrics(fit: pd.DataFrame, time_map: pd.DataFrame) -> pd.DataFrame:
    tmap = TimeMap(time_map)
    rows: list[dict] = []
    fit = fit.copy()
    fit["split"] = np.where((pd.util.hash_pandas_object(fit["well_code"], index=False) % 5) == 0, "holdout", "train")
    for split, g in fit.groupby("split"):
        pred_map = _predict_rows_with_map(g, tmap)
        pred_scalar = g.groupby("model_field")["uptime_treg"].transform("mean").to_numpy(dtype=float) * g["calendar_days"].to_numpy(dtype=float)
        actual = g["op_days_treg"].to_numpy(dtype=float)
        scaled = pd.DataFrame({"run_idx": g["run_idx"].to_numpy(), "pred": pred_map, "tte": g["tte"].to_numpy(dtype=float)})
        scale = scaled.groupby("run_idx").agg(pred_sum=("pred", "sum"), tte=("tte", "first"))
        scale["factor"] = np.divide(
            scale["tte"].to_numpy(dtype=float),
            scale["pred_sum"].to_numpy(dtype=float),
            out=np.ones(len(scale), dtype=float),
            where=scale["pred_sum"].to_numpy(dtype=float) > 0,
        )
        pred_map_scaled = pred_map * g["run_idx"].map(scale["factor"]).to_numpy(dtype=float)
        for name, pred in [
            ("scalar_field", pred_scalar),
            ("age_season_map_unscaled", pred_map),
            ("age_season_map_scaled_to_run_nno", pred_map_scaled),
        ]:
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "n_run_months": int(len(g)),
                    "monthly_op_day_mae": float(np.mean(np.abs(pred - actual))),
                }
            )
            by_run = pd.DataFrame({"run_idx": g["run_idx"].to_numpy(), "actual": actual, "pred": pred})
            recon = by_run.groupby("run_idx", as_index=False).sum()
            if name == "age_season_map_scaled_to_run_nno":
                actual_total = g.groupby("run_idx")["tte"].first().rename("actual_total")
                recon = recon.merge(actual_total, on="run_idx", how="left")
                denom = recon["actual_total"].replace(0, np.nan)
                rel = ((recon["pred"] - recon["actual_total"]).abs() / denom).replace([np.inf, -np.inf], np.nan).dropna()
            else:
                denom = recon["actual"].replace(0, np.nan)
                rel = ((recon["pred"] - recon["actual"]).abs() / denom).replace([np.inf, -np.inf], np.nan).dropna()
            rows[-1]["median_total_nno_abs_pct"] = float(rel.median()) if not rel.empty else np.nan
    return pd.DataFrame(rows)


def idle_hazard_table(run_months: pd.DataFrame, plan_field_by_well: dict[str, str] | None = None) -> pd.DataFrame:
    df = run_months.copy()
    if plan_field_by_well:
        df["reporting_field"] = df["well_code"].map(plan_field_by_well).fillna("")
    else:
        df["reporting_field"] = df["model_field"].fillna("GLOBAL")
    df = df[df["plan_producing"].notna()].copy()
    df["status"] = np.where(df["plan_producing"].astype(bool), "producing", "idle")
    rows = []
    for field, g in list(df.groupby("reporting_field")) + [("GLOBAL", df)]:
        piv = g.groupby("status").agg(months=("is_failure_month", "size"), failures=("is_failure_month", "sum"))
        prod_h = float(piv.loc["producing", "failures"] / piv.loc["producing", "months"]) if "producing" in piv.index and piv.loc["producing", "months"] > 0 else np.nan
        idle_h = float(piv.loc["idle", "failures"] / piv.loc["idle", "months"]) if "idle" in piv.index and piv.loc["idle", "months"] > 0 else np.nan
        frac = idle_h / prod_h if np.isfinite(idle_h) and np.isfinite(prod_h) and prod_h > 0 else 0.0
        rows.append(
            {
                "row_type": "idle_hazard",
                "field": str(field),
                "age_band": "ALL",
                "calendar_month": 0,
                "uptime": np.nan,
                "n_run_months": int(len(g)),
                "idle_hazard_fraction": round(float(np.clip(frac, 0.0, 1.0)), 6),
                "source": "plan_active_producing_split",
                "idle_months": int(piv.loc["idle", "months"]) if "idle" in piv.index else 0,
                "idle_failures": int(piv.loc["idle", "failures"]) if "idle" in piv.index else 0,
                "idle_hazard_per_month": idle_h,
                "producing_months": int(piv.loc["producing", "months"]) if "producing" in piv.index else 0,
                "producing_failures": int(piv.loc["producing", "failures"]) if "producing" in piv.index else 0,
                "producing_hazard_per_month": prod_h,
                "decision": "fit_idle_hazard_fraction" if frac >= 0.1 else "attribute_to_last_producing_month",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "AGE_BANDS",
    "IntervalSlice",
    "TimeMap",
    "age_band_label",
    "build_run_month_dataset",
    "fit_time_map",
    "idle_hazard_table",
    "interval_slices",
    "load_daily_operating",
]
