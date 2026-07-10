"""Shape-compatible export for the legacy ``Прогноз ремонтов`` view.

The legacy UI expected a row-per-well object with daily 1/0 statuses.  The
production-risk model is probabilistic and monthly, so this module preserves the
old contract while marking values as survival-derived rather than CatBoost.
"""
from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk.survival import WellState


FIXED_COLUMNS = [
    "Категория",
    "Участок недр",
    "Куст",
    "Скважина",
    "Прогноз CatBoost",
    "Вероятностный прогноз",
    "Использовано",
    "Факт ННО",
    "ДебН, т/сут",
    "Дата активации",
]


def _horizon_end_day(horizon_month: date) -> date:
    period = pd.Period(horizon_month.strftime("%Y-%m"), freq="M")
    return period.end_time.date()


def _daily_dates(start: date, end_month: date) -> list[date]:
    end = _horizon_end_day(end_month)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _stable_unit(*parts: object) -> float:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _month_spread_date(month: str, well_code: str, event_index: int = 0, event_count: int = 1) -> date:
    period = pd.Period(month, freq="M")
    if event_count <= 1:
        # Keep dates deterministic, but spread wells across the whole month.
        day = 1 + int(_stable_unit(well_code, month, event_index) * period.days_in_month)
    else:
        # Multiple events for one well/month are placed on an evenly spaced grid
        # with a small stable phase shift so wells do not stack on identical days.
        phase = _stable_unit(well_code, month) - 0.5
        position = (event_index + 1 + 0.35 * phase) / (event_count + 1)
        day = 1 + int(np.clip(position, 0.0, 0.999999) * period.days_in_month)
    return period.start_time.date() + timedelta(days=max(min(day, period.days_in_month) - 1, 0))


def _distribute_event_counts(expected_by_well: dict[str, float]) -> dict[str, int]:
    """Fleet-preserving deterministic integerisation (largest-remainder method).

    The previous per-well ``floor(cumulative)`` walk dropped each well's fractional
    remainder (~16% of all events fleet-wide) and, because most wells share similar
    hazard profiles, crossed integer thresholds in the same months — recreating the
    artificial event clustering at month scale.  Here the fleet total equals
    ``round(sum(expected))`` exactly: every well gets ``floor(E_w)`` events and the
    remaining events go to the wells with the largest fractional parts (ties broken
    by a stable per-well hash so output is deterministic run-to-run).
    """
    floors = {wid: int(np.floor(max(e, 0.0))) for wid, e in expected_by_well.items()}
    fleet_total = int(round(sum(max(e, 0.0) for e in expected_by_well.values())))
    residual = fleet_total - sum(floors.values())
    if residual > 0:
        by_fraction = sorted(
            expected_by_well,
            key=lambda wid: (
                -(max(expected_by_well[wid], 0.0) - floors[wid]),
                _stable_unit(wid, "remainder"),
            ),
        )
        for wid in by_fraction[:residual]:
            floors[wid] += 1
    return floors


def _allocate_event_dates(rows: pd.DataFrame, well_code: str, n_events: int) -> list[date]:
    """Place ``n_events`` failure markers along the well's own monthly hazard profile.

    Event k lands where the well's cumulative expected-failure curve crosses
    ``(k - u) * E_total / n_events`` with a stable per-well phase ``u`` — i.e. a
    deterministic inverse-CDF (stratified) sample of the failure-time distribution.
    The random phase keeps wells with similar profiles from piling onto the same
    month; in aggregate, monthly fleet counts track monthly expected totals.
    """
    if n_events <= 0 or rows.empty:
        return []
    ordered = rows.sort_values("month")
    monthly = np.maximum(
        pd.to_numeric(ordered["expected_failures"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0
    )
    total = float(monthly.sum())
    if total <= 0:
        return []
    cum = np.cumsum(monthly)
    months = ordered["month"].astype(str).tolist()
    phase = _stable_unit(well_code, "phase")
    per_month_index: dict[str, int] = {}
    placements: list[str] = []
    for k in range(1, n_events + 1):
        target = (k - phase) * total / n_events
        mi = int(np.searchsorted(cum, min(target, total - 1e-12), side="left"))
        month = months[min(mi, len(months) - 1)]
        placements.append(month)
    counts = {m: placements.count(m) for m in dict.fromkeys(placements)}
    event_dates: list[date] = []
    for month in placements:
        idx = per_month_index.get(month, 0)
        per_month_index[month] = idx + 1
        event_dates.append(_month_spread_date(month, well_code, idx, counts[month]))
    return event_dates


def _first_active_date(rows: pd.DataFrame, fallback: date) -> date | None:
    active = rows[pd.to_numeric(rows["planned_op_days"], errors="coerce").fillna(0.0) > 0]
    if active.empty:
        return None
    first_month = str(active.sort_values("month").iloc[0]["month"])
    month_start = pd.Period(first_month, freq="M").start_time.date()
    return max(month_start, fallback)


def _planned_oil_rate(rows: pd.DataFrame) -> float | None:
    oil = pd.to_numeric(rows["planned_oil_t"], errors="coerce").fillna(0.0).sum()
    op = pd.to_numeric(rows["planned_op_days"], errors="coerce").fillna(0.0).sum()
    if op <= 0:
        return None
    return float(oil / op)


def _predicted_interval(rows: pd.DataFrame) -> float | None:
    expected = pd.to_numeric(rows["expected_failures"], errors="coerce").fillna(0.0).sum()
    op = pd.to_numeric(rows["planned_op_days"], errors="coerce").fillna(0.0).sum()
    if expected <= 0 or op <= 0:
        return None
    return float(op / expected)


def _downtime_days(rows: pd.DataFrame) -> int:
    if "downtime_days" not in rows.columns:
        return 1
    values = pd.to_numeric(rows["downtime_days"], errors="coerce").dropna()
    if values.empty:
        return 1
    return max(int(round(float(values.max()))), 1)


def _state_map(states: list[WellState]) -> dict[str, WellState]:
    return {state.code: state for state in states}


def build(
    projection: pd.DataFrame,
    states: list[WellState],
    cfg: C.RunConfig,
    scenario_id: str = C.PRIMARY_SCENARIO_ID,
) -> dict:
    """Build a legacy-compatible repair forecast payload from survival output."""
    dates = _daily_dates(cfg.forecast_start, cfg.horizon_end)
    date_index = {item: idx for idx, item in enumerate(dates)}
    state_by_code = _state_map(states)
    focus = projection[projection["scenario"] == scenario_id].copy()

    expected_by_well = {
        str(wid): float(pd.to_numeric(grp["expected_failures"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
        for wid, grp in focus.groupby("wid", sort=True)
    }
    events_by_well = _distribute_event_counts(expected_by_well)

    result_rows: list[dict] = []
    monthly_summary: dict[str, dict[str, float | int]] = {}
    for wid, grp in focus.groupby("wid", sort=True):
        state = state_by_code.get(str(wid))
        event_dates = [
            item
            for item in _allocate_event_dates(grp, str(wid), events_by_well.get(str(wid), 0))
            if dates[0] <= item <= dates[-1]
        ]
        downtime_days = _downtime_days(grp)
        statuses = [1] * len(dates)
        if state is not None and state.current_in_operation is False and dates:
            # Stopped in the current techregime report: treat as a repair in progress —
            # down for a full scenario downtime block from forecast start, extended to
            # the first scheduled ГТМ/КРС date when one exists in the horizon.
            resume = dates[0] + timedelta(days=downtime_days)
            if state.first_gtm_date is not None:
                gtm_day = pd.Timestamp(state.first_gtm_date).date()
                if gtm_day > dates[0]:
                    resume = max(resume, min(gtm_day, dates[-1] + timedelta(days=1)))
            for idx in range((min(resume, dates[-1] + timedelta(days=1)) - dates[0]).days):
                statuses[idx] = 0
        for item in event_dates:
            for offset in range(downtime_days):
                down_date = item + timedelta(days=offset)
                if down_date in date_index:
                    statuses[date_index[down_date]] = 0

        predicted_nno = _predicted_interval(grp)
        for event_date in event_dates:
            month_key = f"{event_date.year}-{event_date.month:02d}"
            bucket = monthly_summary.setdefault(month_key, {"total_nno": 0.0, "failure_count": 0})
            bucket["total_nno"] = float(bucket["total_nno"]) + float(predicted_nno or 0.0)
            bucket["failure_count"] = int(bucket["failure_count"]) + 1

        first_active = _first_active_date(grp, cfg.forecast_start)
        first_row = grp.sort_values("month").iloc[0]
        first_op = float(first_row.get("planned_op_days", 0.0) or 0.0)
        first_oil = float(first_row.get("planned_oil_t", 0.0) or 0.0)
        result_rows.append(
            {
                "category": "ВНС" if state is not None and state.current_in_operation is False else ("БАЗА" if first_op > 0 and first_oil > 0 else "ВНС"),
                "field_name": str(first_row.get("plan_field", "") or ""),
                "license_area": str(first_row.get("plan_field", "") or ""),
                "cluster_name": None,
                "well_name": str(first_row.get("raw_id", wid) or wid),
                "catboost_nno": None,
                "probabilistic_nno": predicted_nno,
                "predicted_nno": predicted_nno,
                "used_prediction_source": "survival",
                "downtime_days": downtime_days,
                "current_status": state.current_status if state is not None else None,
                "current_in_operation": state.current_in_operation if state is not None else None,
                "current_status_date": state.current_status_date if state is not None else None,
                "actual_nno": float(state.age_mean) if state is not None else None,
                "oil_rate": _planned_oil_rate(grp),
                "oil_rate_series": [],
                "source_dates": [f"{month}-01" for month in grp.sort_values("month")["month"].astype(str).tolist()],
                "source_oil_rate_series": [],
                "activation_date": first_active.isoformat() if first_active else None,
                "runtime_days": float(state.age_mean) if state is not None else None,
                "event_dates": [item.isoformat() for item in event_dates],
                "statuses": statuses,
            }
        )

    summary_rows = [
        {
            "month": month,
            "total_nno": float(values["total_nno"]),
            "failure_count": int(values["failure_count"]),
        }
        for month, values in sorted(monthly_summary.items())
    ]
    return {
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "dates": [item.isoformat() for item in dates],
        "used_feature_columns": ["stratum", "operating_age", "planned_op_days", "planned_oil_t"],
        "missing_feature_columns": [],
        "source_datasets": [
            {"label": "Production plan", "dataset": None},
            {"label": "ESP survival state", "dataset": None},
        ],
        "monthly_summary": summary_rows,
        "rows": result_rows,
        "sampling_logs": [],
        "tail_diagnostic_image": None,
        "notes": [
            "Shape-compatible export for the legacy repair forecast view.",
            "Values are survival-derived; CatBoost-specific fields are intentionally empty.",
            "Event counts use fleet-preserving largest-remainder rounding: total events == round(sum of expected failures).",
            "Each well's events are placed by deterministic inverse-CDF sampling of its own monthly hazard profile, with a stable per-well phase to avoid artificial date piles.",
            "Each event paints a downtime block using scenario downtime_days.",
            "Wells stopped in the current techregime report start with a downtime block (extended to the first scheduled GTM date when present).",
        ],
    }


def rows_frame(payload: dict) -> pd.DataFrame:
    rows = []
    for row in payload.get("rows", []):
        base = {key: value for key, value in row.items() if key not in {"statuses"}}
        base["event_count"] = len(row.get("event_dates") or [])
        rows.append(base)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except PermissionError:
        stamped = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        stamped.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return stamped


def write_excel(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Repair forecast"
    sheet.append(FIXED_COLUMNS + list(payload.get("dates", [])))

    for row in payload.get("rows", []):
        statuses = list(row.get("statuses") or [])
        date_count = len(payload.get("dates", []))
        if len(statuses) < date_count:
            statuses.extend([1] * (date_count - len(statuses)))
        elif len(statuses) > date_count:
            statuses = statuses[:date_count]
        sheet.append(
            [
                row.get("category"),
                row.get("license_area"),
                row.get("cluster_name"),
                row.get("well_name"),
                row.get("catboost_nno"),
                row.get("probabilistic_nno"),
                row.get("used_prediction_source"),
                row.get("actual_nno"),
                row.get("oil_rate"),
                row.get("activation_date"),
                *statuses,
            ]
        )

    summary = workbook.create_sheet("Summary")
    summary.append(["Start date", payload.get("start_date")])
    summary.append(["End date", payload.get("end_date")])
    summary.append(["Used feature columns", ", ".join(payload.get("used_feature_columns") or [])])
    summary.append(["Missing feature columns", ", ".join(payload.get("missing_feature_columns") or [])])
    for note in payload.get("notes") or []:
        summary.append(["Note", note])
    try:
        workbook.save(path)
        return path
    except PermissionError:
        stamped = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        workbook.save(stamped)
        return stamped
