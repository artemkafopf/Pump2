"""Monthly ESP failure-rate analysis, per УН (license area) and fleet-global.

Deliverable definitions
------------------------
* **Fleet size** (denominator) — from the master file ``Сводные данные`` sheet: a
  well counts toward its field's (and the global) fleet in month *M* when it has
  positive ``Отработанное время`` **and** positive production (oil or liquid)
  that month.  This avoids counting non-producing runtime rows as ESP wells at
  risk; in the late-2026 plan those rows otherwise inflate the fleet far above
  the operating oil-well count visible in tech-regime snapshots.

* **Observed failures** (numerator, history) — actual ESP failures from «Свод»
  plus failure dates in ``WellsArtificialLiftBig``, deduplicated by well/date and
  restricted to wells present in the master fleet so numerator and denominator
  share a population.

* **Model-predicted failures** (numerator, whole timeline) — driven purely by pump
  **installation dates**:
    - History: actual observed pump intervals are replayed from installation dates.
      ``WellsArtificialLiftBig`` backfills missing installations from ``Свод``;
      it supplies real starts/boundaries, but the displayed history line is a
      conditional active-fleet month-start rate, matching the forecast chart
      interpretation.  Future replacement renewal starts only at forecast.
    - Forecast (>= forecast_start): the sanctioned forward projection's
      ``expected_failures`` aggregated by field/month.

* **УН ↔ Weibull stratum** — the reporting group is the production-plan ``УН``
  value for the well.  The survival model stratum remains per-well:
  well-code prefix -> model field code
  (``Ya``/``Vt``/``Za``/``Ic``/``Az``/``Mc``/``Da``) by its well-code prefix
  (``config.FIELD_PREFIX_MAP``).  A single ``УН`` may span several strata.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
from typing import Callable, Collection
import warnings

import numpy as np
import pandas as pd

from openpyxl.chart import LineChart, Reference, Series
from openpyxl.chart.axis import ChartLines, DateAxis
from openpyxl.utils import get_column_letter

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk.survival import (
    HazardLayer,
    StrataModel,
    current_pump_p_fail,
    infant_hazard_theta,
    kpod_hazard_theta,
    ql_hazard_theta,
    scenario_params,
)
from analysis.workflows.production_risk.time_map import TimeMap, interval_slices

GLOBAL_LABEL = "ГЛОБАЛЬНО"
RATE_SHEET = "Интенсивность отказов"
DATA_SHEET = "Отказы_данные"
# The model is computed from HISTORY_FIRST_MONTH so pumps are aged with real
# warm-up (the 2024-01 model value is a proper mid-life rate, not zero), but the
# sheet and charts only DISPLAY from DISPLAY_FIRST_MONTH.  Both the model and the
# factual line therefore start at 2024-01; earlier months feed the model but are
# not shown.
HISTORY_FIRST_MONTH = "2018-01"
DISPLAY_FIRST_MONTH = "2024-01"
FACT_DISPLAY_FIRST_MONTH = "2024-01"
# Chart materiality only affects the visual grid.  The wide rate matrix and the
# long audit sheet still include every УН so totals and auditability are intact.
# Threshold is on mean active fleet, not a failure count, so a full-history span
# does not promote a tiny long-lived field just for accumulating failures slowly.
CHART_MIN_MEAN_FLEET = 12.0

# Optional legacy audit hook for model-field strata whose history line is
# survival-weighted (mass decays as it fails).  The shipped validation chart now
# leaves this empty so history and forecast share the same conditional
# active-at-month-start interpretation.
_SURVIVAL_WEIGHT_FIELDS: frozenset[str] = frozenset()

# Legacy empirical calibration factors retired by Workstreams D/E on 2026-07-15.
# The accepted bundle (`2026-07-15-mc2023plus`) passes the Mc/Ya/Vt/global
# 2024-01..2026-06 fact/model gate with manual factors disabled.
_CALIBRATION_FACTORS: dict[str, float] = {}
_REPORTING_FIELD_CALIBRATION_FACTORS: dict[str, float] = {}


@dataclass
class FailureRateResult:
    months: list[str]
    display_months: list[str]
    fields: list[str]
    chart_fields: list[str]
    forecast_first_month: str
    monthly: pd.DataFrame          # long: field, month, fleet_size, observed_failures, predicted_failures, *_rate
    coverage: dict[str, object]    # diagnostics (well-population coverage etc.)


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    period = pd.Period(month, freq="M")
    return period.start_time.to_pydatetime(), period.end_time.floor("D").to_pydatetime()


def _month_range(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _active_interval_rows(
    esp_source,
    well_field: dict[str, str],
    months: list[str],
    equipment_big_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Count active ESP intervals by field/month, restricted to master wells."""
    month_periods = [pd.Period(m, freq="M") for m in months]
    month_start = {m: p.start_time.to_pydatetime() for m, p in zip(months, month_periods)}
    month_end_excl = {m: (p.start_time + pd.offsets.MonthBegin(1)).to_pydatetime() for m, p in zip(months, month_periods)}
    active = pd.DataFrame(False, index=pd.Index(sorted(well_field), name="wid"), columns=months)
    join_to_code = {_join_key(code): code for code in well_field}
    big_key = str(equipment_big_path) if equipment_big_path else ""
    big_by_key = _big_runs_by_well(big_key)
    used_big_keys: set[str] = set()
    interval_count = 0

    def add_interval(code: str, start: datetime | None, end: datetime | None) -> None:
        nonlocal interval_count
        if code not in active.index or start is None:
            return
        interval_count += 1
        stop = end or month_end_excl[months[-1]]
        if stop <= start:
            return
        for month in months:
            if start < month_end_excl[month] and stop > month_start[month]:
                active.at[code, month] = True

    for key, records in big_by_key.items():
        code = join_to_code.get(key)
        if code is None:
            continue
        used_big_keys.add(key)
        records = sorted(records, key=lambda r: r["start"])
        for idx, record in enumerate(records):
            next_start = records[idx + 1]["start"] if idx + 1 < len(records) else None
            end_candidates = []
            if record.get("end") is not None:
                end_candidates.append(record["end"])
            if next_start is not None:
                end_candidates.append(next_start)
            add_interval(code, record.get("start"), min(end_candidates) if end_candidates else None)

    for code, runs in esp_source.runs_by_well.items():
        key = _join_key(code)
        if key in used_big_keys:
            continue
        for run in sorted([r for r in runs if r.mount is not None], key=lambda r: r.mount):
            add_interval(code, run.mount, run.stop or run.demo)

    rows: list[dict] = []
    field_series = pd.Series({wid: well_field.get(wid, "") for wid in active.index}, name="field")
    for field, mask in active.groupby(field_series):
        if not str(field).strip():
            field = "Без УН"
        counts = mask.sum(axis=0)
        for month in months:
            rows.append({"field": str(field), "month": month, "fleet_size_interval": int(counts.get(month, 0))})
    global_counts = active.sum(axis=0)
    for month in months:
        rows.append({"field": GLOBAL_LABEL, "month": month, "fleet_size_interval": int(global_counts.get(month, 0))})
    return pd.DataFrame(rows), {
        "intervals_used": interval_count,
        "big_interval_wells_matched": len(used_big_keys),
        "big_load_ok": _big_load_ok(big_key),
    }


def _fleet_size_by_field(plan, well_field: dict[str, str], months: list[str]) -> pd.DataFrame:
    """Count active producing wells per (field, month) + global.

    ``Отработанное время`` alone is too broad in the production-plan master: some
    months carry runtime rows for wells with zero oil and zero liquid, which does
    not match the operational ESP-at-risk fleet.  The failure-rate denominator is
    therefore active production: runtime > 0 and either oil or liquid > 0.
    """
    op = plan.op_days_raw.reindex(columns=months, fill_value=0.0)
    oil = plan.oil_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    liq = plan.liquid_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    active = (op > 0) & ((oil > 0) | (liq > 0))
    field_series = pd.Series({wid: well_field.get(wid, "") for wid in active.index}, name="field")
    rows: list[dict] = []
    grouped = active.groupby(field_series)
    for field, mask in grouped:
        if not str(field).strip():
            field = "Без УН"
        counts = mask.sum(axis=0)
        for month in months:
            rows.append({"field": str(field), "month": month, "fleet_size": int(counts.get(month, 0))})
    global_counts = active.sum(axis=0)
    for month in months:
        rows.append({"field": GLOBAL_LABEL, "month": month, "fleet_size": int(global_counts.get(month, 0))})
    return pd.DataFrame(rows)


def _fleet_size_hybrid(
    plan,
    esp_source,
    well_field: dict[str, str],
    months: list[str],
    equipment_big_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fleet denominator with interval warm-up before the plan window.

    Plan months retain the active-producing definition needed for the future
    forecast denominator.  Earlier warm-up months use interval counts because the
    production master has no pre-2024 columns.
    """
    plan_months = set(plan.months)
    interval, info = _active_interval_rows(esp_source, well_field, months, equipment_big_path)
    plan_fleet = _fleet_size_by_field(plan, well_field, [m for m in months if m in plan_months])
    out = interval.rename(columns={"fleet_size_interval": "fleet_size"}).copy()
    out["fleet_basis"] = "interval"
    if not plan_fleet.empty:
        key_cols = ["field", "month"]
        plan_fleet = plan_fleet.rename(columns={"fleet_size": "fleet_size_plan"})
        out = out.merge(plan_fleet, on=key_cols, how="left")
        use_plan = out["month"].isin(plan_months) & out["fleet_size_plan"].notna()
        out.loc[use_plan, "fleet_size"] = out.loc[use_plan, "fleet_size_plan"].astype(int)
        out.loc[use_plan, "fleet_basis"] = "plan_active_producing"
        out = out.drop(columns=["fleet_size_plan"])

    seam = out[(out["field"] == GLOBAL_LABEL) & (out["month"].isin(["2024-01", "2024-02"]))]
    interval_global = interval[(interval["field"] == GLOBAL_LABEL) & (interval["month"] == "2024-01")]
    plan_global = plan_fleet[(plan_fleet["field"] == GLOBAL_LABEL) & (plan_fleet["month"] == "2024-01")] if not plan_fleet.empty else pd.DataFrame()
    if not interval_global.empty and not plan_global.empty:
        i = float(interval_global["fleet_size_interval"].iloc[0])
        p = float(plan_global["fleet_size_plan"].iloc[0])
        info["denominator_2024_01_interval"] = i
        info["denominator_2024_01_plan"] = p
        info["denominator_2024_01_diff_pct"] = abs(i - p) / p if p > 0 else float("nan")
    info["denominator_basis"] = "interval_before_plan_then_plan_active_producing"
    info["denominator_seam_rows"] = seam.to_dict("records")
    return out[["field", "month", "fleet_size", "fleet_basis"]], info


# Planned-intervention pull reasons in WellsArtificialLiftBig that are NOT ESP
# failures: ГТМ (геолого-техническое мероприятие) and ППР (планово-предупредительный
# ремонт).  Big fills a «Дата отказа» for these workovers too, so counting every
# fail_date over-states the failure numerator (≈2× for Мирнинский).  The survival
# model trains on genuine failures only (event=1 requires a failed component), so
# the factual numerator must exclude these.  «Свод» (the failures register,
# Failure Flag=1) is already clean; this filter applies to the Big supplement only.
_WORKOVER_PULL_REASONS: frozenset[str] = frozenset({"ГТМ", "ППР"})


def _is_failure_pull(reason: object) -> bool:
    """True if a Big pull with a fail_date is a genuine failure (not a workover)."""
    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return True  # dated pull, no recorded reason → treat as a failure
    return str(reason).strip().upper() not in _WORKOVER_PULL_REASONS


def _observed_failures_by_field(
    esp_source, well_field: dict[str, str], months: set[str], equipment_big_path: Path | None = None
) -> pd.DataFrame:
    """Actual **failures** by month (event=1), restricted to master wells.

    ``Свод`` (Failure Flag=1) is the curated failure register and is used as-is.
    It is supplemented with Big failure dates (Big is also used for historical
    install intervals, keeping fact and model populations aligned), but **only for
    genuine failures** — Big pulls coded as planned workovers (ГТМ / ППР) carry a
    «Дата отказа» yet are not ESP failures, so they are excluded (see
    ``_WORKOVER_PULL_REASONS``).  Censored runs (no fail_date) are excluded already.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for code, runs in esp_source.runs_by_well.items():
        field = well_field.get(code)
        if field is None:  # not in master fleet -> keep numerator/denominator aligned
            continue
        for run in runs:
            if run.failure_flag == 1 and run.stop is not None:
                month = run.stop.strftime("%Y-%m")
                day = run.stop.strftime("%Y-%m-%d")
                key = (code, day)
                if month in months and key not in seen:
                    seen.add(key)
                    rows.append({"field": field, "month": month, "observed_failures": 1.0})
    join_to_code = {_join_key(code): code for code in well_field}
    big_key = str(equipment_big_path) if equipment_big_path else ""
    for key, records in _big_runs_by_well(big_key).items():
        code = join_to_code.get(key)
        if code is None:
            continue
        field = well_field.get(code)
        if field is None:
            continue
        for record in records:
            fail = record.get("fail")
            if fail is None:
                continue
            if not _is_failure_pull(record.get("pull_reason")):
                continue  # planned workover (ГТМ/ППР), not a failure
            month = fail.strftime("%Y-%m")
            day = fail.strftime("%Y-%m-%d")
            seen_key = (code, day)
            if month in months and seen_key not in seen:
                seen.add(seen_key)
                rows.append({"field": field, "month": month, "observed_failures": 1.0})
    if not rows:
        return pd.DataFrame(columns=["field", "month", "observed_failures"])
    df = pd.DataFrame(rows)
    field_agg = df.groupby(["field", "month"], as_index=False)["observed_failures"].sum()
    global_agg = df.groupby("month", as_index=False)["observed_failures"].sum().assign(field=GLOBAL_LABEL)
    return pd.concat([field_agg, global_agg], ignore_index=True)


def _bundle_observed_failures(bundle_date: str, months: set[str]) -> pd.DataFrame | None:
    path = C.observed_failures_path(bundle_date)
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"field", "month", "observed_failures"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    out = df[["field", "month", "observed_failures"]].copy()
    out["month"] = out["month"].astype(str)
    out = out[out["month"].isin(months)].copy()
    out["observed_failures"] = pd.to_numeric(out["observed_failures"], errors="coerce").fillna(0.0)
    return out.groupby(["field", "month"], as_index=False)["observed_failures"].sum()


def _active_producing_mask(plan, months: list[str]) -> pd.DataFrame:
    op = plan.op_days_raw.reindex(columns=months, fill_value=0.0)
    oil = plan.oil_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    liq = plan.liquid_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    return (op > 0) & ((oil > 0) | (liq > 0))


def _join_key(code: str) -> str:
    return str(code).strip().casefold()


_BIG_LOAD_STATUS: dict[str, dict[str, object]] = {}


def _big_load_ok(path_key: str) -> bool:
    return bool(_BIG_LOAD_STATUS.get(path_key or "", {}).get("ok", False))


@lru_cache(maxsize=8)
def _big_runs_by_well(path_key: str = "") -> dict[str, list[dict]]:
    """ESP run intervals from WellsArtificialLiftBig, keyed compatibly with plan wells.

    Big is an external validation/backfill source.  If it is absent or its layout
    changes, the historical model line falls back to ``Свод`` intervals only.
    """
    try:
        from analysis.data.equipment_big import load_equipment_big
        df = load_equipment_big(Path(path_key) if path_key else None)
    except Exception as exc:
        _BIG_LOAD_STATUS[path_key or ""] = {"ok": False, "error": str(exc)}
        warnings.warn(f"WellsArtificialLiftBig load failed; falling back to Свод-only history: {exc}")
        return {}
    if df.empty or "well_key" not in df.columns or "install_date" not in df.columns:
        _BIG_LOAD_STATUS[path_key or ""] = {"ok": False, "error": "empty_or_missing_required_columns"}
        warnings.warn("WellsArtificialLiftBig load produced no usable ESP intervals; falling back to Свод-only history.")
        return {}
    focus = df[
        df.get("is_esp", pd.Series(False, index=df.index)).fillna(False)
        & df["well_key"].notna()
        & df["install_date"].notna()
    ].copy()
    if focus.empty:
        _BIG_LOAD_STATUS[path_key or ""] = {"ok": False, "error": "no_esp_rows"}
        warnings.warn("WellsArtificialLiftBig has no usable ESP rows; falling back to Свод-only history.")
        return {}
    focus["join_key"] = focus["well_key"].map(_join_key)
    out: dict[str, list[dict]] = {}
    for key, grp in focus.groupby("join_key"):
        records: list[dict] = []
        grp = grp.sort_values(["install_date", "row_id"], na_position="last")
        for _, row in grp.iterrows():
            start = pd.to_datetime(row.get("install_date"), errors="coerce")
            if pd.isna(start):
                continue
            fail = pd.to_datetime(row.get("fail_date"), errors="coerce")
            pull = pd.to_datetime(row.get("pull_date"), errors="coerce")
            end_candidates = [
                dt.to_pydatetime()
                for dt in (fail, pull)
                if pd.notna(dt) and dt.to_pydatetime() > start.to_pydatetime()
            ]
            records.append(
                {
                    "start": start.to_pydatetime(),
                    "fail": fail.to_pydatetime() if pd.notna(fail) else None,
                    "end": min(end_candidates) if end_candidates else None,
                    "nno_days": row.get("nno_days"),
                    "pull_reason": row.get("pull_reason"),
                }
            )
        if records:
            out[str(key)] = records
    _BIG_LOAD_STATUS[path_key or ""] = {"ok": True, "rows": int(len(focus)), "wells": int(len(out))}
    return out


def _as_positive_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) and out > 0 else None


def _append_interval_predictions(
    rows: list[dict],
    *,
    plan,
    active: pd.DataFrame,
    model: StrataModel,
    field: str,
    code: str,
    months: list[str],
    start: datetime,
    end: datetime,
    params: dict[str, float],
    total_op: float | None,
    global_pooled: bool = False,
    p_fail_fn: Callable[[dict[int, float], float], float] | None = None,
    survival_weight: bool = False,
    calibration: float = 1.0,
    use_ql_hazard: bool = False,
    use_kpod_hazard: bool = False,
    qnominal: float | None = None,
    ql_audit_rows: list[dict] | None = None,
    time_map: TimeMap | None = None,
    emit_rows: list[dict] | None = None,
    emit_stratum: str | None = None,
    emit_covariates: dict[str, float] | None = None,
    emit_event_month: str | None = None,
) -> None:
    """Append non-renewal expected failures over one observed pump interval.

    ``emit_rows`` is an additive, off-by-default instrumentation hook for the
    Workstream-C hazard refit: when a list is supplied, one per-run-month row is
    appended for every replayed slice carrying the *baseline* expected failures
    (``mu_baseline``), operating age/exposure, resolved stratum, hazard covariates
    and the A-population event flag.  It never alters ``rows`` or the shipped output.

    ``p_fail_fn(age_pmf, op_days) -> float`` is the pluggable survival hook: its default is
    the Weibull ``current_pump_p_fail`` closure (so the Weibull line stays byte-identical),
    and the CatBoost comparison injects an ``S_cb``-backed closure.  Only the survival curve
    differs between the two model lines — every other line here (month slicing, aging,
    exposure via ``uptime_factor``/plan op-days) is shared, by construction.

    ``survival_weight`` (experimental, per-stratum): when True the monthly conditional
    failure probability is weighted by the running survival mass and that mass is decayed,
    so a pump's expected failures over its interval telescope to ``F(age)=1-S(age)`` instead
    of the inflated conditional-hazard sum (``≈ -ln S``).  This matches how the forward
    ``project_well`` decays age-mass and removes the systematic history over-count.
    """
    ql_daily_available = p_fail_fn is None
    if p_fail_fn is None:
        def p_fail_fn(age_pmf: dict[int, float], op_days: float) -> float:
            return current_pump_p_fail(age_pmf, params, op_days, model)

    def _ql_inputs(month: str) -> tuple[float | None, str]:
        if month not in getattr(plan, "months", []):
            return None, "not_plan_month"
        try:
            liq = float(plan.liquid_volume.at[code, month])
            op = float(plan.op_days.at[code, month])
        except Exception:
            return None, "lookup_failed"
        if not np.isfinite(liq) or not np.isfinite(op) or liq <= 0.0 or op <= 0.0:
            return None, "nonpositive_or_nonfinite_ql"
        return liq / op, "ok"

    def _kpod_inputs(month: str) -> tuple[float | None, str]:
        ql, status = _ql_inputs(month)
        if ql is None:
            return None, status
        qnom = _as_positive_float(qnominal)
        if qnom is None:
            return None, "missing_qnominal"
        return ql / qnom, "ok"

    def _kpod_field() -> str | None:
        text = str(field or "").casefold()
        if "большетир" in text:
            return "Bt"
        return crosswalk.map_model_field_from_well(code)

    def _audit_ql(month: str, age0: float, op_days: float, status: str, theta: float | None) -> None:
        if ql_audit_rows is None:
            return
        ql_audit_rows.append(
            {
                "field": field,
                "code": code,
                "month": month,
                "in_plan_month": bool(month in getattr(plan, "months", [])),
                "model_field": crosswalk.map_model_field_from_well(code) or "",
                "age_start": float(age0),
                "op_days": float(op_days),
                "status": status,
                "theta": float(theta) if theta is not None and np.isfinite(theta) else np.nan,
            }
        )

    def _stress_daily_probability(month: str, age0: float, op_days: float) -> tuple[float | None, str, float | None]:
        if not ql_daily_available:
            return None, "monthly_fallback", None
        ql, status = _ql_inputs(month)
        kpod, kpod_status = _kpod_inputs(month)
        if use_ql_hazard and ql is None:
            return None, status, None
        if use_kpod_hazard and kpod is None:
            return None, kpod_status, None
        model_field = crosswalk.map_model_field_from_well(code)
        age = max(0, int(round(float(age0))))
        remaining = max(0.0, float(op_days))
        q_arr = model.daily_fail_prob_array(params, age + int(math.ceil(remaining)) + 2)
        survived = 1.0
        theta_first: float | None = None
        while remaining > 1e-12 and survived > 0.0:
            step = min(1.0, remaining)
            q_base = float(q_arr[min(age, len(q_arr) - 1)])
            theta = 1.0
            if use_ql_hazard and ql is not None:
                theta *= float(ql_hazard_theta(float(np.log1p(ql)), model_field, float(age)))
            if use_kpod_hazard and kpod is not None:
                theta *= float(kpod_hazard_theta(float(kpod), float(age), _kpod_field()))
            if theta_first is None:
                theta_first = theta
            q_eff = float(np.clip(1.0 - (1.0 - q_base) ** theta, 0.0, 1.0))
            survived *= max(0.0, 1.0 - step * q_eff)
            remaining -= step
            age += 1
        return float(np.clip(1.0 - survived, 0.0, 1.0)), "applied_daily", theta_first

    def _apply_infant_dial(age0: float, op_days: float, p_base: float) -> float:
        """Measured infant-mortality multiplier — must hit the REPLAY as well as the
        forecast, or history and forecast silently use different hazards and the
        fact-vs-model check becomes meaningless.

        The replay is MONTHLY but tau is ~11 days, so theta falls from 2.18 to ~1.07
        *inside* the first month.  Sampling it at the month's start age would apply
        the age-0 peak to all 30 days and over-predict badly (Vt 2024: 1.09 -> 1.33).
        So integrate theta over the month's age span instead:

            mean theta = 1 + u0*tau/D * (exp(-a0/tau) - exp(-(a0+D)/tau))

        `project_well` does not need this — it indexes theta by age day-by-day.
        """
        if not C.INFANT_HAZARD_ENABLED or p_base <= 0.0:
            return p_base
        p = C.infant_hazard_params(crosswalk.map_model_field_from_well(code))
        u0, tau, cap = float(p["u0"]), float(p["tau_days"]), float(p["cap"])
        a0 = float(max(age0, 0.0))
        d = float(max(op_days, 0.0))
        if u0 <= 0.0 or tau <= 0.0:
            return p_base
        if d <= 0.0:
            theta = 1.0 + u0 * math.exp(-a0 / tau)
        else:
            theta = 1.0 + (u0 * tau / d) * (math.exp(-a0 / tau) - math.exp(-(a0 + d) / tau))
        theta = float(np.clip(theta, 1.0, max(cap, 1.0)))
        if theta <= 0.0 or not np.isfinite(theta):
            return p_base
        return float(np.clip(1.0 - (1.0 - float(p_base)) ** theta, 0.0, 1.0))

    def _apply_stress_dials(month: str, age0: float, op_days: float, p_base: float) -> float:
        p_base = _apply_infant_dial(age0, op_days, p_base)
        if (not use_ql_hazard and not use_kpod_hazard) or p_base <= 0.0:
            return p_base
        p_daily, status, theta_audit = _stress_daily_probability(month, age0, op_days)
        if p_daily is not None:
            _audit_ql(month, age0, op_days, status, theta_audit)
            return p_daily
        ql, ql_status = _ql_inputs(month)
        kpod, kpod_status = _kpod_inputs(month)
        if use_ql_hazard and ql is None:
            _audit_ql(month, age0, op_days, ql_status, None)
            return p_base
        if use_kpod_hazard and kpod is None:
            _audit_ql(month, age0, op_days, kpod_status, None)
            return p_base
        theta = 1.0
        model_field = crosswalk.map_model_field_from_well(code)
        if use_ql_hazard and ql is not None:
            theta *= float(ql_hazard_theta(float(np.log1p(ql)), model_field, float(max(age0, 0.0))))
        if use_kpod_hazard and kpod is not None:
            theta *= float(kpod_hazard_theta(float(kpod), float(max(age0, 0.0)), _kpod_field()))
        if theta <= 0.0 or not np.isfinite(theta):
            _audit_ql(month, age0, op_days, "nonfinite_theta", None)
            return p_base
        _audit_ql(month, age0, op_days, "applied_monthly", theta)
        return float(np.clip(1.0 - (1.0 - float(p_base)) ** theta, 0.0, 1.0))

    def _emit(month: str, age_start: float, op_days: float) -> None:
        if emit_rows is None:
            return
        mu = float(p_fail_fn({int(round(float(age_start))): 1.0}, float(op_days)))
        row = {
            "code": code,
            "field": field,
            "model_field": crosswalk.map_model_field_from_well(code),
            "stratum": emit_stratum,
            "month": month,
            "age_start": float(age_start),
            "op_days": float(op_days),
            "mu_baseline": mu,
            "event": 1 if (emit_event_month is not None and month == emit_event_month) else 0,
        }
        if emit_covariates:
            for _k, _v in emit_covariates.items():
                row[f"cov_{_k}"] = _v
        emit_rows.append(row)

    mapped = interval_slices(
        months=months,
        start=start,
        end=end,
        code=code,
        reporting_field=field,
        model_field=crosswalk.map_model_field_from_well(code),
        total_op=total_op,
        fallback_uptime=_as_positive_float(params.get("uptime_factor")) or 1.0,
        time_map=time_map or TimeMap.missing(),
        plan=plan,
        active=active,
    )
    if mapped:
        surv = 1.0
        for sl in mapped:
            _emit(sl.month, sl.age_start, sl.op_days)
            p_cond = _apply_stress_dials(
                sl.month,
                sl.age_start,
                sl.op_days,
                p_fail_fn({int(round(sl.age_start)): 1.0}, sl.op_days),
            )
            p = (surv * p_cond if survival_weight else p_cond) * calibration
            if p > 0:
                rows.append({
                    "field": field,
                    "month": sl.month,
                    "predicted_failures": float(p),
                    "global_pooled_predicted_failures": float(p) if global_pooled else 0.0,
                })
            if survival_weight:
                surv *= max(0.0, 1.0 - p_cond)
        return

    slices: list[tuple[str, float, float]] = []
    for month in months:
        m_start = pd.Period(month, freq="M").start_time.to_pydatetime()
        m_end = (pd.Period(month, freq="M").start_time + pd.offsets.MonthBegin(1)).to_pydatetime()
        ov_start = max(pd.Timestamp(start).to_pydatetime(), m_start)
        ov_end = min(pd.Timestamp(end).to_pydatetime(), m_end)
        overlap = float((pd.Timestamp(ov_end) - pd.Timestamp(ov_start)).days)
        if overlap <= 0:
            continue
        op_days_month = overlap
        if month in getattr(plan, "months", []):
            if code not in active.index or month not in active.columns or not bool(active.at[code, month]):
                continue
            op_frame = getattr(plan, "op_days", plan.op_days_raw)
            cal_days = float(
                getattr(plan, "cal_days", pd.Series(dtype=float)).get(month, pd.Period(month, freq="M").days_in_month)
            )
            month_op = float(op_frame.at[code, month]) if code in op_frame.index and month in op_frame.columns else 0.0
            op_days_month = month_op * overlap / cal_days if cal_days > 0 else 0.0
            if op_days_month <= 0:
                continue
        slices.append((month, overlap, op_days_month))
    if not slices:
        return
    if total_op is not None:
        span_days = max((end - start).days, 0)
        if span_days <= 0:
            return
        surv = 1.0
        for month, overlap, _ in slices:
            op_days_month = float(total_op) * (overlap / span_days)
            m_start = pd.Period(month, freq="M").start_time.to_pydatetime()
            age_start = float(total_op) * (max((m_start - start).days, 0) / span_days)
            _emit(month, age_start, op_days_month)
            p_cond = _apply_stress_dials(
                month,
                age_start,
                op_days_month,
                p_fail_fn({int(round(age_start)): 1.0}, op_days_month),
            )
            p = (surv * p_cond if survival_weight else p_cond) * calibration
            if p > 0:
                rows.append({
                    "field": field,
                    "month": month,
                    "predicted_failures": float(p),
                    "global_pooled_predicted_failures": float(p) if global_pooled else 0.0,
                })
            if survival_weight:
                surv *= max(0.0, 1.0 - p_cond)
        return

    uptime = _as_positive_float(params.get("uptime_factor")) or 1.0
    age = 0.0
    surv = 1.0
    for month, overlap, _ in slices:
        op_days_month = max(0.0, overlap * uptime)
        _emit(month, age, op_days_month)
        p_cond = _apply_stress_dials(month, age, op_days_month, p_fail_fn({int(round(age)): 1.0}, op_days_month))
        p = (surv * p_cond if survival_weight else p_cond) * calibration
        if p > 0:
            rows.append({
                "field": field,
                "month": month,
                "predicted_failures": float(p),
                "global_pooled_predicted_failures": float(p) if global_pooled else 0.0,
            })
        if survival_weight:
            surv *= max(0.0, 1.0 - p_cond)
        age += op_days_month


def _hist_predicted_failures_by_field(
    plan,
    esp_source,
    projection: pd.DataFrame,
    scenario_id: str,
    model: StrataModel,
    well_field: dict[str, str],
    months: list[str],
    forecast_first: str,
    equipment_big_path: Path | None = None,
    p_fail_provider: Callable[[str, datetime, str], Callable[[dict[int, float], float], float]] | None = None,
    hazard: HazardLayer | None = None,
    hazard_mode: str = "baseline",
    bundle_date: str = C.BUNDLE_DATE,
    emit_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Weibull-predicted failures per month over history from observed intervals.

    ``p_fail_provider(code, start, field) -> p_fail_fn`` swaps the survival curve for the
    CatBoost comparison line; when ``None`` the Weibull ``current_pump_p_fail`` closure is
    used, so the two lines share identical month-slicing/aging/exposure by construction.

    ``WellsArtificialLiftBig`` is primary when available because it carries many
    replacement starts that are absent from ``Свод``.  ``Свод`` is still used for
    wells missing from Big and for stratum refinements (sour/contractor) when mount
    dates match.  No renewal is simulated before forecast start.
    """
    hist_months = [m for m in months if m < forecast_first]
    if not hist_months:
        return pd.DataFrame(columns=["field", "month", "predicted_failures"])
    active = _active_producing_mask(plan, hist_months)
    cutoff = esp_source.source_cutoff or datetime(2026, 6, 1)
    forecast_start = pd.Period(forecast_first, freq="M").start_time.to_pydatetime()
    rows: list[dict] = []
    ql_audit_rows: list[dict] = []
    use_hazard = hazard_mode == "stress"
    emit = emit_rows is not None
    hazard = hazard or HazardLayer(bundle_date=bundle_date)
    run_cov_map = crosswalk._load_run_covariates(bundle_date) if (use_hazard or emit) else {}
    time_map = getattr(model, "time_map", TimeMap.missing())

    svod_by_key: dict[str, list] = {}
    for code, runs in esp_source.runs_by_well.items():
        svod_by_key[_join_key(code)] = sorted([r for r in runs if r.mount is not None], key=lambda r: r.mount)

    def _matching_svod_run(code: str, start: datetime):
        candidates = svod_by_key.get(_join_key(code), [])
        best = None
        best_gap = None
        for run in candidates:
            if run.mount is None:
                continue
            gap = abs((run.mount - start).days)
            if gap <= 7 and (best_gap is None or gap < best_gap):
                best = run
                best_gap = gap
        return best

    def _run_ordinal(code: str, run) -> int | None:
        if run is None:
            return None
        candidates = [r for r in svod_by_key.get(_join_key(code), []) if r.mount is not None]
        for idx, candidate in enumerate(candidates, start=1):
            if candidate is run:
                return idx
            if candidate.mount is not None and run.mount is not None and abs((candidate.mount - run.mount).days) <= 7:
                return idx
        return None

    def _previous_svod_run(code: str, ordinal: int | None):
        if ordinal is None or ordinal <= 1:
            return None
        candidates = [r for r in svod_by_key.get(_join_key(code), []) if r.mount is not None]
        return candidates[ordinal - 2] if ordinal - 2 < len(candidates) else None

    def _hazard_covariates(code: str, run, ordinal: int | None) -> dict[str, float]:
        if run is None or (not use_hazard and not emit):
            return {}
        norm_code = crosswalk.norm_well(code) or code
        cov = dict(run_cov_map.get((norm_code, int(ordinal)), {})) if ordinal is not None else {}
        fallback = crosswalk._fallback_covariates(run, _previous_svod_run(code, ordinal), ordinal)
        for key, value in fallback.items():
            cov.setdefault(key, value)
        return cov

    # Big-backed observed intervals.  Big adds real missing starts, but we do not
    # generate additional replacements beyond the observed Big/Svod interval list.
    join_to_code = {_join_key(code): code for code in well_field}
    big_key = str(equipment_big_path) if equipment_big_path else ""
    big_by_key = _big_runs_by_well(big_key)
    for key, records in big_by_key.items():
        code = join_to_code.get(key)
        if code is None:
            continue
        field = well_field.get(code)
        if field is None:
            continue
        records = sorted(records, key=lambda r: r["start"])
        model_field = crosswalk.map_model_field_from_well(code)
        for idx, record in enumerate(records):
            start = record["start"]
            if start >= forecast_start:
                continue
            next_start = records[idx + 1]["start"] if idx + 1 < len(records) else None
            end_candidates = [forecast_start]
            if record.get("end") is not None and record["end"] > start:
                end_candidates.append(record["end"])
            if next_start is not None and next_start > start:
                end_candidates.append(next_start)
            end = min(end_candidates)
            if end <= start:
                continue
            match = _matching_svod_run(code, start)
            sour = crosswalk.sour_group(match.sour_raw) if match is not None else "nonsour"
            ctr = crosswalk.contractor_group(match.ctr_raw) if match is not None else "Pooled"
            params, stratum = model.resolve(
                model_field,
                sour,
                ctr,
            )
            ordinal = _run_ordinal(code, match)
            if use_hazard:
                params, _theta = scenario_params(
                    params,
                    _hazard_covariates(code, match, ordinal),
                    hazard,
                    "stress",
                )

            total_op = _as_positive_float(record.get("nno_days"))
            if total_op is None and match is not None:
                total_op = _as_positive_float(match.age_op)
            emit_cov = _hazard_covariates(code, match, ordinal) if emit else None
            emit_event_month = None
            if emit:
                _fail = record.get("fail")
                if _fail is not None and _is_failure_pull(record.get("pull_reason")):
                    _m = _fail.strftime("%Y-%m")
                    if _m in hist_months:
                        emit_event_month = _m
            _append_interval_predictions(
                rows,
                plan=plan,
                active=active,
                model=model,
                field=field,
                code=code,
                months=hist_months,
                start=start,
                end=end,
                params=params,
                total_op=total_op,
                global_pooled=stratum == "Global_Pooled",
                p_fail_fn=p_fail_provider(code, start, field) if p_fail_provider else None,
                survival_weight=model_field in _SURVIVAL_WEIGHT_FIELDS,
                calibration=_CALIBRATION_FACTORS.get(model_field, 1.0),
                use_ql_hazard=use_hazard and C.QL_HAZARD_ENABLED,
                use_kpod_hazard=use_hazard and C.KPOD_HAZARD_ENABLED,
                qnominal=crosswalk.resolve_qnominal(code, model_field, equipment_big_path),
                ql_audit_rows=ql_audit_rows if use_hazard and C.QL_HAZARD_ENABLED else None,
                time_map=time_map,
                emit_rows=emit_rows,
                emit_stratum=stratum,
                emit_covariates=emit_cov,
                emit_event_month=emit_event_month,
            )

    # Свод-only fallback for wells absent from Big.
    for code, runs in esp_source.runs_by_well.items():
        if _join_key(code) in big_by_key:
            continue
        field = well_field.get(code)
        if field is None:
            continue
        model_field = crosswalk.map_model_field_from_well(code)
        sorted_runs = sorted([r for r in runs if r.mount is not None], key=lambda r: r.mount)
        for ordinal, run in enumerate(sorted_runs, start=1):
            start = run.mount
            if start is None or start >= forecast_start:
                continue
            end = min(run.stop or run.demo or cutoff, forecast_start)
            if end <= start:
                continue
            params, stratum = model.resolve(
                model_field,
                crosswalk.sour_group(run.sour_raw),
                crosswalk.contractor_group(run.ctr_raw),
            )
            if use_hazard:
                params, _theta = scenario_params(
                    params,
                    _hazard_covariates(code, run, ordinal),
                    hazard,
                    "stress",
                )
            emit_cov = _hazard_covariates(code, run, ordinal) if emit else None
            emit_event_month = None
            if emit and run.failure_flag == 1 and run.stop is not None:
                _m = run.stop.strftime("%Y-%m")
                if _m in hist_months:
                    emit_event_month = _m
            _append_interval_predictions(
                rows,
                plan=plan,
                active=active,
                model=model,
                field=field,
                code=code,
                months=hist_months,
                start=start,
                end=end,
                params=params,
                total_op=_as_positive_float(run.age_op),
                global_pooled=stratum == "Global_Pooled",
                p_fail_fn=p_fail_provider(code, start, field) if p_fail_provider else None,
                survival_weight=model_field in _SURVIVAL_WEIGHT_FIELDS,
                calibration=_CALIBRATION_FACTORS.get(model_field, 1.0),
                use_ql_hazard=use_hazard and C.QL_HAZARD_ENABLED,
                use_kpod_hazard=use_hazard and C.KPOD_HAZARD_ENABLED,
                qnominal=crosswalk.resolve_qnominal(code, model_field, equipment_big_path),
                ql_audit_rows=ql_audit_rows if use_hazard and C.QL_HAZARD_ENABLED else None,
                time_map=time_map,
                emit_rows=emit_rows,
                emit_stratum=stratum,
                emit_covariates=emit_cov,
                emit_event_month=emit_event_month,
            )
    if not rows:
        out = pd.DataFrame(columns=["field", "month", "predicted_failures"])
        out.attrs["ql_audit"] = pd.DataFrame(ql_audit_rows)
        return out
    df = pd.DataFrame(rows)
    if _REPORTING_FIELD_CALIBRATION_FACTORS:
        # MODIFIED: empirical УН-level calibration is applied before aggregation so
        # the global row remains the sum of the calibrated field contributions.
        cal = df["field"].map(_REPORTING_FIELD_CALIBRATION_FACTORS).fillna(1.0).astype(float)
        df["predicted_failures"] = df["predicted_failures"] * cal.to_numpy()
        df["global_pooled_predicted_failures"] = df["global_pooled_predicted_failures"] * cal.to_numpy()
    agg_cols = ["predicted_failures", "global_pooled_predicted_failures"]
    field_agg = df.groupby(["field", "month"], as_index=False)[agg_cols].sum()
    global_agg = df.groupby("month", as_index=False)[agg_cols].sum().assign(field=GLOBAL_LABEL)
    out = pd.concat([field_agg, global_agg], ignore_index=True)
    out.attrs["ql_audit"] = pd.DataFrame(ql_audit_rows)
    return out


def _fwd_predicted_failures_by_field(
    projection: pd.DataFrame,
    scenario_id: str,
    forecast_first: str,
    well_field: dict[str, str],
    model: StrataModel,
) -> pd.DataFrame:
    """Forward model expected failures, aggregated by master УН/month."""
    if projection.empty:
        return pd.DataFrame(columns=["field", "month", "predicted_failures", "global_pooled_predicted_failures"])
    focus = projection[projection["scenario"] == scenario_id].copy()
    focus = focus[focus["month"] >= forecast_first]
    focus["field"] = focus["wid"].astype(str).map(well_field).fillna("Без УН")
    focus["field"] = focus["field"].astype(str).replace("", "Без УН")
    focus["expected_failures"] = pd.to_numeric(focus["expected_failures"], errors="coerce").fillna(0.0)
    if _CALIBRATION_FACTORS:
        # MODIFIED: apply the same manual Ya/Vt calibration to the forecast so the
        # single model line is coherent across the history/forecast boundary.
        cal = focus["wid"].astype(str).map(
            lambda w: _CALIBRATION_FACTORS.get(crosswalk.map_model_field_from_well(w), 1.0)
        )
        focus["expected_failures"] = focus["expected_failures"] * cal.to_numpy()
    if _REPORTING_FIELD_CALIBRATION_FACTORS:
        # MODIFIED: carry the УН-level empirical calibration through forecast months
        # as a fixed multiplier, rather than recomputing it from future observations.
        cal = focus["field"].map(_REPORTING_FIELD_CALIBRATION_FACTORS).fillna(1.0).astype(float)
        focus["expected_failures"] = focus["expected_failures"] * cal.to_numpy()
    def _is_global(wid: object) -> bool:
        model_field = crosswalk.map_model_field_from_well(str(wid))
        _, stratum = model.resolve(model_field, "nonsour", "Pooled")
        return stratum == "Global_Pooled"
    focus["global_pooled_predicted_failures"] = np.where(
        focus["wid"].map(_is_global),
        focus["expected_failures"],
        0.0,
    )
    field_agg = (
        focus.groupby(["field", "month"], as_index=False)[["expected_failures", "global_pooled_predicted_failures"]].sum()
        .rename(columns={"expected_failures": "predicted_failures"})
    )
    global_agg = (
        focus.groupby("month", as_index=False)[["expected_failures", "global_pooled_predicted_failures"]].sum()
        .rename(columns={"expected_failures": "predicted_failures"})
        .assign(field=GLOBAL_LABEL)
    )
    return pd.concat([field_agg, global_agg], ignore_index=True)


def _ql_audit_summary(audit: pd.DataFrame) -> dict[str, object]:
    if audit is None or audit.empty:
        return {"total_slices": 0, "status_counts": {}, "plan_month_status_counts": {}}
    status_counts = audit["status"].value_counts().sort_index()
    plan_audit = audit[audit["in_plan_month"].astype(bool)]
    plan_status_counts = plan_audit["status"].value_counts().sort_index()
    by_field_month = (
        audit.groupby(["field", "month", "status"], as_index=False)
        .size()
        .rename(columns={"size": "slices"})
        .sort_values(["month", "field", "status"])
    )
    return {
        "total_slices": int(len(audit)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "plan_month_status_counts": {str(k): int(v) for k, v in plan_status_counts.items()},
        "first_plan_month": (
            str(audit.loc[audit["in_plan_month"].astype(bool), "month"].min())
            if bool(audit["in_plan_month"].any())
            else ""
        ),
        "by_field_month_sample": by_field_month.head(50).to_dict("records"),
    }


def build_well_field(plan) -> dict[str, str]:
    """well code -> master УН reporting group (only master wells feed a group's fleet)."""
    meta = plan.producer_meta
    well_field: dict[str, str] = {}
    for wid in meta.index:
        raw = str(meta.at[wid, "license_area"]) if "license_area" in meta.columns else ""
        if not raw.strip() and "plan_field" in meta.columns:
            raw = str(meta.at[wid, "plan_field"])
        well_field[str(wid)] = raw.strip() if raw and raw.strip() else "Без УН"
    return well_field


def _attach_catboost_columns(
    monthly: pd.DataFrame,
    *,
    plan,
    esp_source,
    projection: pd.DataFrame,
    scenario_id: str,
    model: StrataModel,
    well_field: dict[str, str],
    months: list[str],
    forecast_first: str,
    equipment_big_path: Path | None,
    fleet_pos: np.ndarray,
    catboost_model=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add the parallel CatBoost failure-rate columns; return (monthly, diagnostics).

    Runs the identical history replay with the survival curve swapped for the CatBoost
    ``S_cb`` (in-sample and cross-fit variants).  CatBoost columns are NaN from
    ``forecast_first`` on — there is no CatBoost renewal forecast (§1 of the handoff).  The
    Weibull columns already on ``monthly`` are never touched.
    """
    from analysis.workflows.production_risk import failure_rate_catboost as cbfr

    cb_model = catboost_model
    if cb_model is None:
        cb_model = cbfr.build_model(well_field)

    fleet_pos = np.asarray(fleet_pos, dtype=bool)
    forecast_mask = monthly["month"].to_numpy() >= forecast_first
    diag_out: dict[str, object] = {}

    variants = (("xfit", "_catboost_xfit"), ("insample", "_catboost"))
    for variant, suffix in variants:
        diag = cbfr._ReplayDiag()
        provider = (
            lambda code, start, field, _v=variant, _d=diag: cb_model.p_fail_fn(code, start, field, _v, _d)
        )
        agg = _hist_predicted_failures_by_field(
            plan, esp_source, projection, scenario_id, model, well_field, months,
            forecast_first, equipment_big_path, p_fail_provider=provider,
        )
        fail_col = f"predicted_failures{suffix}"
        rate_col = f"predicted_rate{suffix}"
        if agg.empty:
            merged = monthly.assign(**{fail_col: 0.0})
        else:
            take = agg[["field", "month", "predicted_failures"]].rename(columns={"predicted_failures": fail_col})
            merged = monthly.merge(take, on=["field", "month"], how="left")
            merged[fail_col] = merged[fail_col].fillna(0.0)
        # CatBoost only spans history; blank the forecast window (no ML renewal line).
        merged.loc[forecast_mask, fail_col] = np.nan
        merged[rate_col] = np.where(fleet_pos, merged[fail_col] / merged["fleet_size"], np.nan)
        merged.loc[forecast_mask, rate_col] = np.nan
        monthly = merged
        if variant == "xfit":  # headline diagnostics come from the cross-fit pass
            cov = diag.covariate_share_by_field()
            tail = diag.past_b90_share_by_field()
            diag_out["catboost_covariate_share"] = float(cov.get("__global__", float("nan")))
            diag_out["catboost_covariate_share_by_field"] = {k: v for k, v in cov.items() if k != "__global__"}
            diag_out["catboost_past_b90_share"] = float(tail.get("__global__", float("nan")))
            diag_out["catboost_past_b90_share_by_field"] = {k: v for k, v in tail.items() if k != "__global__"}
    return monthly, diag_out


def compute(
    plan,
    esp_source,
    projection: pd.DataFrame,
    cfg: C.RunConfig,
    scenario_id: str = C.PRIMARY_SCENARIO_ID,
    model: StrataModel | None = None,
    hazard: HazardLayer | None = None,
    catboost_model=None,
    only_fields: Collection[str] | None = None,
) -> FailureRateResult:
    model = model or StrataModel(bundle_date=cfg.bundle_date)
    hazard = hazard or HazardLayer(bundle_date=cfg.bundle_date)
    forecast_last = max(str(cfg.horizon_end.strftime("%Y-%m")), max(plan.months))
    months = _month_range(HISTORY_FIRST_MONTH, forecast_last)
    display_months = [m for m in months if m >= DISPLAY_FIRST_MONTH]
    forecast_first = cfg.forecast_start.strftime("%Y-%m")

    # well -> master УН (only master wells contribute to a reporting group's fleet)
    well_field = build_well_field(plan)
    if only_fields is not None:
        # Every fleet/observed/predicted path keys off well_field, so restricting it
        # here scopes the whole computation to the requested УН — the tuning app
        # recomputes one reporting group instead of the full fleet.  GLOBAL_LABEL
        # then aggregates only the retained wells.
        keep = set(only_fields)
        well_field = {wid: field for wid, field in well_field.items() if field in keep}
        if not well_field:
            raise ValueError(f"only_fields={sorted(keep)} matched no wells in the plan.")

    fleet, fleet_info = _fleet_size_hybrid(plan, esp_source, well_field, months, cfg.equipment_big_path)
    observed = _bundle_observed_failures(cfg.bundle_date, set(months))
    if observed is None:
        observed = _observed_failures_by_field(esp_source, well_field, set(months), cfg.equipment_big_path)
    hist_pred = _hist_predicted_failures_by_field(
        plan,
        esp_source,
        projection,
        scenario_id,
        model,
        well_field,
        months,
        forecast_first,
        cfg.equipment_big_path,
        hazard=hazard,
        hazard_mode="stress" if scenario_id == C.STRESS_SCENARIO_ID else "baseline",
        bundle_date=cfg.bundle_date,
    )
    ql_audit = hist_pred.attrs.get("ql_audit", pd.DataFrame())
    ql_audit_summary = _ql_audit_summary(ql_audit)
    fwd_pred = _fwd_predicted_failures_by_field(projection, scenario_id, forecast_first, well_field, model)
    predicted = pd.concat([hist_pred, fwd_pred], ignore_index=True)
    if not predicted.empty:
        predicted = predicted.groupby(["field", "month"], as_index=False)[
            ["predicted_failures", "global_pooled_predicted_failures"]
        ].sum()

    monthly = fleet.merge(observed, on=["field", "month"], how="left")
    monthly = monthly.merge(predicted, on=["field", "month"], how="left")
    monthly["observed_failures"] = monthly["observed_failures"].fillna(0.0)
    monthly["predicted_failures"] = monthly["predicted_failures"].fillna(0.0)
    monthly["global_pooled_predicted_failures"] = monthly["global_pooled_predicted_failures"].fillna(0.0)
    monthly["global_pooled_share"] = np.divide(
        monthly["global_pooled_predicted_failures"],
        monthly["predicted_failures"],
        out=np.zeros(len(monthly), dtype=float),
        where=monthly["predicted_failures"].to_numpy(dtype=float) > 0,
    )
    # Observed rate is undefined once we run out of complete actuals; predicted rate spans all.
    last_obs_month = None
    obs_nonzero = monthly[(monthly["field"] == GLOBAL_LABEL) & (monthly["observed_failures"] > 0)]
    if not obs_nonzero.empty:
        last_obs_month = obs_nonzero["month"].max()
        fact_through = str(getattr(cfg, "fact_through_month", C.DEFAULT_FACT_THROUGH_MONTH) or "").strip()
        if fact_through and fact_through in set(months):
            last_obs_month = min(str(last_obs_month), fact_through)

    fleet_pos = monthly["fleet_size"] > 0
    monthly["observed_rate"] = np.where(fleet_pos, monthly["observed_failures"] / monthly["fleet_size"], np.nan)
    monthly["predicted_rate"] = np.where(fleet_pos, monthly["predicted_failures"] / monthly["fleet_size"], np.nan)

    # --- Parallel CatBoost line (comparison only; Weibull columns above untouched) ---
    catboost_diag: dict[str, object] = {}
    if getattr(cfg, "enable_catboost_compare", False):
        monthly, catboost_diag = _attach_catboost_columns(
            monthly,
            plan=plan,
            esp_source=esp_source,
            projection=projection,
            scenario_id=scenario_id,
            model=model,
            well_field=well_field,
            months=months,
            forecast_first=forecast_first,
            equipment_big_path=cfg.equipment_big_path,
            fleet_pos=fleet_pos,
            catboost_model=catboost_model,
        )

    if last_obs_month is not None:
        beyond = monthly["month"] > last_obs_month
        monthly.loc[beyond, "observed_rate"] = np.nan
        monthly.loc[beyond, "observed_failures"] = np.nan
    # Factual is only shown from FACT_DISPLAY_FIRST_MONTH; earlier actual coverage
    # for the current master fleet is incomplete, so blank it (the installation-
    # driven model line still spans the full history).
    before_fact = monthly["month"] < FACT_DISPLAY_FIRST_MONTH
    monthly.loc[before_fact, "observed_rate"] = np.nan
    monthly.loc[before_fact, "observed_failures"] = np.nan

    fields_present = [f for f in monthly["field"].unique() if f != GLOBAL_LABEL]
    fields = [GLOBAL_LABEL] + sorted(fields_present)
    display_mask = monthly["month"].isin(display_months)
    # Materiality: mean active fleet over the ACTUAL operating record (fact window),
    # not the display span or forecast.  A field barely operating today but projected
    # to grow (e.g. Большетирский, ~0 producing wells in history) must not be charted
    # off the back of forecast growth; fleet size over real months keeps the charted
    # set stable and matches the "suppress minor/new fields" intent.
    fact_window_end = last_obs_month or (display_months[-1] if display_months else "")
    material_mask = (
        monthly["month"].isin(display_months)
        & (monthly["month"] <= fact_window_end)
        & (monthly["field"] != GLOBAL_LABEL)
    )
    mean_fleet = monthly[material_mask].groupby("field")["fleet_size"].mean()
    chart_fields = [GLOBAL_LABEL] + [f for f in sorted(fields_present) if float(mean_fleet.get(f, 0.0)) >= CHART_MIN_MEAN_FLEET]
    monthly = monthly.sort_values(["field", "month"]).reset_index(drop=True)

    master_wells = set(well_field)
    svod_wells = set(esp_source.runs_by_well)
    fallback_by_field = (
        monthly[display_mask]
        .groupby("field")[["predicted_failures", "global_pooled_predicted_failures"]]
        .sum()
    )
    fallback_share_by_field = {
        str(field): (
            float(row["global_pooled_predicted_failures"] / row["predicted_failures"])
            if float(row["predicted_failures"]) > 0
            else 0.0
        )
        for field, row in fallback_by_field.iterrows()
    }
    coverage = {
        "master_wells": float(len(master_wells)),
        "svod_wells": float(len(svod_wells)),
        "svod_in_master": float(len(svod_wells & master_wells)),
        "last_observed_month": last_obs_month if last_obs_month else "",
        "fleet_denominator": fleet_info.get("denominator_basis", ""),
        "display_first_month": DISPLAY_FIRST_MONTH,
        "history_first_month": HISTORY_FIRST_MONTH,
        "fact_display_first_month": FACT_DISPLAY_FIRST_MONTH,
        "chart_min_mean_fleet": CHART_MIN_MEAN_FLEET,
        "chart_fields": chart_fields,
        "scenario_id": scenario_id,
        "historical_hazard_layer": (
            "static+Ql(field-ref clip5)+raw-Kpod(U-shape)"
            if scenario_id == C.STRESS_SCENARIO_ID and C.QL_HAZARD_ENABLED and C.KPOD_HAZARD_ENABLED
            else (
                "static+Ql(field-ref clip5)"
                if scenario_id == C.STRESS_SCENARIO_ID and C.QL_HAZARD_ENABLED
                else (
                    "static+raw-Kpod(U-shape)"
                    if scenario_id == C.STRESS_SCENARIO_ID and C.KPOD_HAZARD_ENABLED
                    else ("static only; Ql/Kpod disabled" if scenario_id == C.STRESS_SCENARIO_ID else "none")
                )
            )
        ),
        "global_pooled_share_by_field": fallback_share_by_field,
        # MODIFIED parameters, surfaced so the manual adjustments are never silent:
        "survival_weighted_fields": sorted(_SURVIVAL_WEIGHT_FIELDS),
        "calibration_factors_modified": dict(_CALIBRATION_FACTORS),
        "reporting_field_calibration_factors_modified": dict(_REPORTING_FIELD_CALIBRATION_FACTORS),
        "global_calibration_modified": "sum_of_reporting_field_calibrations",
        "ql_hazard_firing_audit": ql_audit_summary,
        **fleet_info,
    }
    if catboost_diag:
        coverage.update(catboost_diag)
    return FailureRateResult(
        months=months,
        display_months=display_months,
        fields=fields,
        chart_fields=chart_fields,
        forecast_first_month=forecast_first,
        monthly=monthly,
        coverage=coverage,
    )


# --------------------------------------------------------------------------- #
# Excel rendering (native openpyxl charts — Cyrillic-safe, no matplotlib)      #
# --------------------------------------------------------------------------- #

def _rate_matrix(result: FailureRateResult, value: str) -> pd.DataFrame:
    """months (index) × fields (columns) matrix of a rate/count column."""
    pivot = result.monthly.pivot_table(index="month", columns="field", values=value, aggfunc="first")
    pivot = pivot.reindex(index=result.months, columns=result.fields)
    return pivot


def _excel_serial(month: str) -> int:
    """Excel 1900-system serial for the first day of ``month`` (YYYY-MM)."""
    d = datetime(int(month[:4]), int(month[5:7]), 1)
    return (d - datetime(1899, 12, 30)).days


def _line_chart(title: str, ws, *, cat_ref, obs_ref, pred_ref,
                view_min: int | None = None, view_max: int | None = None) -> LineChart:
    chart = LineChart()
    chart.title = title
    chart.style = 2
    chart.height = 7.5
    chart.width = 16
    chart.y_axis.title = "отказов / скв. в мес"
    # Date axis: the series carry the full modelled history, but the visible window
    # is clipped to [view_min, view_max] so the model line enters at its real
    # mid-life value instead of appearing to start from zero.  The category cells
    # are dates (first of month), so this limits the view without touching the data.
    chart.x_axis = DateAxis(axId=10, crossAx=100)
    chart.y_axis.axId = 100
    chart.y_axis.crossAx = 10
    chart.x_axis.axPos = "b"   # DateAxis defaults to "l"; the x-axis must sit at the bottom
    chart.y_axis.axPos = "l"
    chart.x_axis.number_format = "yyyy-mm"
    chart.x_axis.majorTimeUnit = "months"
    chart.x_axis.baseTimeUnit = "months"
    chart.x_axis.title = "Месяц"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.x_axis.majorGridlines = None
    chart.y_axis.majorGridlines = ChartLines()
    chart.x_axis.tickLblPos = "low"
    chart.x_axis.txPr = None
    if view_min is not None:
        chart.x_axis.scaling.min = view_min
    if view_max is not None:
        chart.x_axis.scaling.max = view_max

    s_obs = Series(obs_ref, title="Факт")
    s_obs.graphicalProperties.line.width = 20000
    s_obs.graphicalProperties.line.solidFill = "1F77B4"
    s_pred = Series(pred_ref, title="Прогноз модели")
    s_pred.graphicalProperties.line.width = 20000
    s_pred.graphicalProperties.line.solidFill = "D62728"
    s_pred.graphicalProperties.line.dashStyle = "dash"
    chart.series = [s_obs, s_pred]
    chart.set_categories(cat_ref)
    return chart


def write_sheets(workbook, result: FailureRateResult) -> None:
    """Append the failure-rate sheet (charts) and the long-form data sheet."""
    obs = _rate_matrix(result, "observed_rate")
    pred = _rate_matrix(result, "predicted_rate")
    fields = result.fields
    # The table and chart series carry the FULL modelled history; the charts clip
    # the visible x-axis to the display window (see _line_chart).  The factual rate
    # is already blank before FACT_DISPLAY_FIRST_MONTH, so no factual is drawn there.
    months = result.months
    n = len(months)
    view_min = _excel_serial(result.display_months[0]) if result.display_months else None
    view_max = _excel_serial(result.display_months[-1]) if result.display_months else None

    ws = workbook.create_sheet(RATE_SHEET)
    ws.append(["Интенсивность отказов УЭЦН — факт vs прогноз модели (по УН и по флоту)"])
    ws.append([
        "Интенсивность = отказы / парк в работе. Знаменатель: история — активные "
        "интервалы УЭЦН (Big/Свод); прогноз (с "
        f"{result.forecast_first_month}) — активный добывающий парк плана "
        "(«Отработанное время» > 0 и добыча нефти или жидкости > 0). "
        f"Факт — по «Свод» (Failure Flag=1), показан с "
        f"{result.coverage.get('fact_display_first_month', FACT_DISPLAY_FIRST_MONTH)}. "
        "Валидационный график показывает условный месячный риск активного на начало "
        "месяца парка до и после forecast_start; это не симуляция замен. "
        f"Слой hazard: {result.coverage.get('historical_hazard_layer', 'none')}."
    ])
    ws.append([
        "Прогноз/прогнозис добычи считается отдельно: renewal-симуляция с возвратом "
        "скважины после downtime, расчетом потерь нефти и нагрузки ремонтов."
    ])
    ws.append([f"Покрытие: master={int(result.coverage['master_wells'])}, "
               f"Свод={int(result.coverage['svod_wells'])}, "
               f"Свод∩master={int(result.coverage['svod_in_master'])}, "
               f"последний факт-месяц={result.coverage['last_observed_month']}"])
    ws.append([f"Графики: только УН со средним парком >= {CHART_MIN_MEAN_FLEET:g} скв.; "
               f"модель — вся история, ось графиков — с {result.display_months[0] if result.display_months else ''}."])
    cal = result.coverage.get("calibration_factors_modified") or {}
    field_cal = result.coverage.get("reporting_field_calibration_factors_modified") or {}
    sw = result.coverage.get("survival_weighted_fields") or []
    if cal or field_cal or sw:
        parts = []
        if sw:
            parts.append(f"survival-weight: {', '.join(sw)}")
        if cal:
            parts.append("ручная калибровка (МОДИФИЦИРОВАНО): "
                         + ", ".join(f"{k}×{v:g}" for k, v in cal.items()))
        if field_cal:
            parts.append(
                "УН-калибровка (МОДИФИЦИРОВАНО): "
                + ", ".join(f"{k}×{v:g}" for k, v in field_cal.items())
            )
            parts.append("ГЛОБАЛЬНО = сумма откалиброванных УН")
        ws.append([
            "ВНИМАНИЕ — прогноз модели содержит фиксированные прозрачные ручные "
            "поправки; коэффициенты не пересчитываются при запуске. "
            + "; ".join(parts) + "."
        ])
    ws.append([])

    # ---- wide rate matrix: month | <field>_факт | <field>_прогноз ... ----
    header = ["Месяц"]
    for field in fields:
        header += [f"{field} · факт", f"{field} · прогноз"]
    ws.append(header)
    # Anchor to where the header actually landed: ws.append([]) above writes an
    # empty row that ws.max_row does not count, so max_row+1 is off by one and the
    # chart refs would start on the header (a text cell → a spurious leading zero).
    header_row = ws.max_row
    for i, month in enumerate(months):
        # Column A is a real date (first of month) so the charts can use a date
        # axis and clip the visible range without dropping data points.
        row = [datetime(int(month[:4]), int(month[5:7]), 1)]
        for field in fields:
            ov = obs.at[month, field] if field in obs.columns else np.nan
            pv = pred.at[month, field] if field in pred.columns else np.nan
            row += [None if pd.isna(ov) else float(ov), None if pd.isna(pv) else float(pv)]
        ws.append(row)
    first_data_row = header_row + 1
    last_data_row = header_row + n
    for r in range(first_data_row, last_data_row + 1):
        ws.cell(row=r, column=1).number_format = "yyyy-mm"
        for c in range(2, 2 + 2 * len(fields)):
            ws.cell(row=r, column=c).number_format = "0.0000"
    ws.column_dimensions["A"].width = 10

    cat_ref = Reference(ws, min_col=1, min_row=first_data_row, max_row=last_data_row)

    # charts arranged in a 2-column grid below the table
    charts_top = last_data_row + 3
    chart_rows = 16   # vertical cells per chart
    chart_cols = 9    # horizontal cells per chart
    for chart_idx, field in enumerate(result.chart_fields):
        if field not in fields:
            continue
        idx = fields.index(field)
        obs_col = 2 + 2 * idx
        pred_col = obs_col + 1
        obs_ref = Reference(ws, min_col=obs_col, min_row=first_data_row, max_row=last_data_row)
        pred_ref = Reference(ws, min_col=pred_col, min_row=first_data_row, max_row=last_data_row)
        label = "Флот (все УН)" if field == GLOBAL_LABEL else field
        chart = _line_chart(label, ws, cat_ref=cat_ref, obs_ref=obs_ref, pred_ref=pred_ref,
                            view_min=view_min, view_max=view_max)
        grid_row = chart_idx // 2
        grid_col = chart_idx % 2
        anchor = f"{get_column_letter(1 + grid_col * chart_cols)}{charts_top + grid_row * chart_rows}"
        ws.add_chart(chart, anchor)

    # ---- long-form data sheet for auditing ----
    dws = workbook.create_sheet(DATA_SHEET)
    cols = ["field", "month", "fleet_size", "fleet_basis", "observed_failures", "predicted_failures",
            "global_pooled_predicted_failures", "global_pooled_share", "observed_rate", "predicted_rate"]
    dws.append(["УН", "Месяц", "Активный добывающий парк", "Отказы (факт)",
                "Отказы (прогноз)", "Global_Pooled отказы", "Global_Pooled доля",
                "Интенсивность (факт)", "Интенсивность (прогноз)", "Основа парка"])
    # Full modelled history; factual columns are already blank before
    # FACT_DISPLAY_FIRST_MONTH.
    for _, row in result.monthly[cols].iterrows():
        dws.append([
            row["field"], row["month"],
            None if pd.isna(row["fleet_size"]) else int(row["fleet_size"]),
            None if pd.isna(row["observed_failures"]) else float(row["observed_failures"]),
            None if pd.isna(row["predicted_failures"]) else float(row["predicted_failures"]),
            None if pd.isna(row["global_pooled_predicted_failures"]) else float(row["global_pooled_predicted_failures"]),
            None if pd.isna(row["global_pooled_share"]) else float(row["global_pooled_share"]),
            None if pd.isna(row["observed_rate"]) else float(row["observed_rate"]),
            None if pd.isna(row["predicted_rate"]) else float(row["predicted_rate"]),
            row["fleet_basis"],
        ])
    for r in range(2, dws.max_row + 1):
        for c in (7, 8, 9):
            dws.cell(row=r, column=c).number_format = "0.0000"
    dws.column_dimensions["A"].width = 34
    dws.column_dimensions["B"].width = 10
