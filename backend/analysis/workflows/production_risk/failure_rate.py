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
      it supplies real starts/boundaries, but the history line does not simulate
      renewal chains.  Future replacement renewal starts only at forecast.
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
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from openpyxl.chart import LineChart, Reference, Series
from openpyxl.chart.axis import ChartLines
from openpyxl.utils import get_column_letter

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk.survival import StrataModel, current_pump_p_fail

GLOBAL_LABEL = "ГЛОБАЛЬНО"
RATE_SHEET = "Интенсивность отказов"
DATA_SHEET = "Отказы_данные"
DISPLAY_FIRST_MONTH = "2024-01"
HISTORY_FIRST_MONTH = "2018-01"
# Chart materiality only affects the visual grid.  The wide rate matrix and the
# long audit sheet still include every УН so totals and auditability are intact.
CHART_MIN_TOTAL_FAILURES = 20.0


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


def _observed_failures_by_field(
    esp_source, well_field: dict[str, str], months: set[str], equipment_big_path: Path | None = None
) -> pd.DataFrame:
    """Actual failures by month, restricted to master wells.

    ``Свод`` is supplemented with Big failure dates because Big is also used for
    historical install intervals.  This keeps fact and model populations aligned.
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


def _active_producing_mask(plan, months: list[str]) -> pd.DataFrame:
    op = plan.op_days_raw.reindex(columns=months, fill_value=0.0)
    oil = plan.oil_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    liq = plan.liquid_volume.reindex(index=op.index, columns=months, fill_value=0.0)
    return (op > 0) & ((oil > 0) | (liq > 0))


def _segment_month_slices(plan, active: pd.DataFrame, wid: str, months: list[str],
                          start: datetime, end: datetime) -> list[tuple[str, float, float]]:
    """Return (month, calendar overlap days, planned op-days) for an active segment."""
    op_frame = getattr(plan, "op_days", plan.op_days_raw)
    if wid not in op_frame.index or start >= end:
        return []
    out: list[tuple[str, float, float]] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for month in months:
        m_start = pd.Period(month, freq="M").start_time
        m_end = m_start + pd.offsets.MonthBegin(1)
        ov_start = max(start_ts, m_start)
        ov_end = min(end_ts, m_end)
        overlap = float((ov_end - ov_start).days)
        if overlap <= 0:
            continue
        if wid not in active.index or month not in active.columns or not bool(active.at[wid, month]):
            continue
        cal_days = float(
            getattr(plan, "cal_days", pd.Series(dtype=float)).get(month, pd.Period(month, freq="M").days_in_month)
        )
        month_op = float(op_frame.at[wid, month]) if month in op_frame.columns else 0.0
        op_days = month_op * overlap / cal_days if cal_days > 0 else 0.0
        if op_days > 0:
            out.append((month, overlap, op_days))
    return out


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
) -> None:
    """Append non-renewal expected failures over one observed pump interval."""
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
        for month, overlap, _ in slices:
            op_days_month = float(total_op) * (overlap / span_days)
            m_start = pd.Period(month, freq="M").start_time.to_pydatetime()
            age_start = float(total_op) * (max((m_start - start).days, 0) / span_days)
            p = current_pump_p_fail({int(round(age_start)): 1.0}, params, op_days_month, model)
            if p > 0:
                rows.append({
                    "field": field,
                    "month": month,
                    "predicted_failures": float(p),
                    "global_pooled_predicted_failures": float(p) if global_pooled else 0.0,
                })
        return

    uptime = _as_positive_float(params.get("uptime_factor")) or 1.0
    age = 0.0
    for month, overlap, _ in slices:
        op_days_month = max(0.0, overlap * uptime)
        p = current_pump_p_fail({int(round(age)): 1.0}, params, op_days_month, model)
        if p > 0:
            rows.append({
                "field": field,
                "month": month,
                "predicted_failures": float(p),
                "global_pooled_predicted_failures": float(p) if global_pooled else 0.0,
            })
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
) -> pd.DataFrame:
    """Weibull-predicted failures per month over history from observed intervals.

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

            total_op = _as_positive_float(record.get("nno_days"))
            if total_op is None and match is not None:
                total_op = _as_positive_float(match.age_op)
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
            )

    # Свод-only fallback for wells absent from Big.
    for code, runs in esp_source.runs_by_well.items():
        if _join_key(code) in big_by_key:
            continue
        field = well_field.get(code)
        if field is None:
            continue
        model_field = crosswalk.map_model_field_from_well(code)
        for run in sorted([r for r in runs if r.mount is not None], key=lambda r: r.mount):
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
            )
    if not rows:
        return pd.DataFrame(columns=["field", "month", "predicted_failures"])
    df = pd.DataFrame(rows)
    agg_cols = ["predicted_failures", "global_pooled_predicted_failures"]
    field_agg = df.groupby(["field", "month"], as_index=False)[agg_cols].sum()
    global_agg = df.groupby("month", as_index=False)[agg_cols].sum().assign(field=GLOBAL_LABEL)
    return pd.concat([field_agg, global_agg], ignore_index=True)


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


def compute(
    plan,
    esp_source,
    projection: pd.DataFrame,
    cfg: C.RunConfig,
    scenario_id: str = C.PRIMARY_SCENARIO_ID,
    model: StrataModel | None = None,
) -> FailureRateResult:
    model = model or StrataModel(bundle_date=cfg.bundle_date)
    forecast_last = max(str(cfg.horizon_end.strftime("%Y-%m")), max(plan.months))
    months = _month_range(HISTORY_FIRST_MONTH, forecast_last)
    display_months = [m for m in months if m >= DISPLAY_FIRST_MONTH]
    forecast_first = cfg.forecast_start.strftime("%Y-%m")

    # well -> master УН (only master wells contribute to a reporting group's fleet)
    meta = plan.producer_meta
    well_field: dict[str, str] = {}
    for wid in meta.index:
        raw = str(meta.at[wid, "license_area"]) if "license_area" in meta.columns else ""
        if not raw.strip() and "plan_field" in meta.columns:
            raw = str(meta.at[wid, "plan_field"])
        well_field[str(wid)] = raw.strip() if raw and raw.strip() else "Без УН"

    fleet, fleet_info = _fleet_size_hybrid(plan, esp_source, well_field, months, cfg.equipment_big_path)
    observed = _observed_failures_by_field(esp_source, well_field, set(months), cfg.equipment_big_path)
    hist_pred = _hist_predicted_failures_by_field(
        plan, esp_source, projection, scenario_id, model, well_field, months, forecast_first, cfg.equipment_big_path
    )
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
    if last_obs_month is not None:
        beyond = monthly["month"] > last_obs_month
        monthly.loc[beyond, "observed_rate"] = np.nan
        monthly.loc[beyond, "observed_failures"] = np.nan

    fields_present = [f for f in monthly["field"].unique() if f != GLOBAL_LABEL]
    fields = [GLOBAL_LABEL] + sorted(fields_present)
    display_mask = monthly["month"].isin(display_months)
    material = (
        monthly[display_mask & (monthly["field"] != GLOBAL_LABEL)]
        .assign(total_failures=lambda d: d["observed_failures"].fillna(0.0) + d["predicted_failures"].fillna(0.0))
        .groupby("field")["total_failures"]
        .sum()
    )
    chart_fields = [GLOBAL_LABEL] + [f for f in sorted(fields_present) if float(material.get(f, 0.0)) >= CHART_MIN_TOTAL_FAILURES]
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
        "chart_min_total_failures": CHART_MIN_TOTAL_FAILURES,
        "chart_fields": chart_fields,
        "global_pooled_share_by_field": fallback_share_by_field,
        **fleet_info,
    }
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


def _line_chart(title: str, ws, *, cat_ref, obs_ref, pred_ref) -> LineChart:
    chart = LineChart()
    chart.title = title
    chart.style = 2
    chart.height = 7.5
    chart.width = 16
    chart.y_axis.title = "отказов / скв. в мес"
    chart.x_axis.title = "Месяц"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.x_axis.majorGridlines = None
    chart.y_axis.majorGridlines = ChartLines()
    chart.x_axis.tickLblPos = "low"
    chart.x_axis.txPr = None

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
    months = result.display_months
    n = len(months)

    ws = workbook.create_sheet(RATE_SHEET)
    ws.append(["Интенсивность отказов УЭЦН — факт vs прогноз модели (по УН и по флоту)"])
    ws.append([
        "Интенсивность = отказы / активный добывающий парк "
        "(до плана — активные интервалы Big/Свод; в плане — «Отработанное время» > 0 "
        "и добыча нефти или жидкости > 0 в «Сводные данные»). "
        f"Факт — по «Свод» (Failure Flag=1). Прогноз модели (Weibull) по датам монтажа насосов; "
        f"с {result.forecast_first_month} — прогнозный горизонт."
    ])
    ws.append([f"Покрытие: master={int(result.coverage['master_wells'])}, "
               f"Свод={int(result.coverage['svod_wells'])}, "
               f"Свод∩master={int(result.coverage['svod_in_master'])}, "
               f"последний факт-месяц={result.coverage['last_observed_month']}"])
    ws.append([f"Графики: только УН с факт+прогноз отказов >= {CHART_MIN_TOTAL_FAILURES:g}; "
               "таблица и аудит содержат все УН."])
    ws.append([])

    # ---- wide rate matrix: month | <field>_факт | <field>_прогноз ... ----
    header_row = ws.max_row + 1
    header = ["Месяц"]
    for field in fields:
        header += [f"{field} · факт", f"{field} · прогноз"]
    ws.append(header)
    for i, month in enumerate(months):
        row = [month]
        for field in fields:
            ov = obs.at[month, field] if field in obs.columns else np.nan
            pv = pred.at[month, field] if field in pred.columns else np.nan
            row += [None if pd.isna(ov) else float(ov), None if pd.isna(pv) else float(pv)]
        ws.append(row)
    first_data_row = header_row + 1
    last_data_row = header_row + n
    for r in range(first_data_row, last_data_row + 1):
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
        chart = _line_chart(label, ws, cat_ref=cat_ref, obs_ref=obs_ref, pred_ref=pred_ref)
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
