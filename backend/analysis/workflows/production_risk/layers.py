"""Planner-facing layers built on the production-risk projection."""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk.crosswalk import (
    CurrentTechregimeStatus,
    DowntimeStats,
    EspSource,
    PlanData,
    map_model_field_from_well,
)
from analysis.workflows.production_risk.survival import (
    HazardLayer,
    StrataModel,
    WellState,
    compress_age_pmf,
    current_pump_p_fail,
    project_well,
    scenario_params,
    weighted_age_mean,
)


def _gtm_info(gtm: pd.DataFrame, forecast_start: date) -> dict[str, dict]:
    info: dict[str, dict] = {}
    if gtm is None or gtm.empty:
        return info
    horizon_90 = pd.Timestamp(forecast_start) + pd.Timedelta(days=90)
    for code, grp in gtm.dropna(subset=["well_code"]).groupby("well_code"):
        dates = pd.concat([grp["krs_date"], grp["startup_date"]], ignore_index=True).dropna().sort_values()
        first_date = dates.iloc[0] if not dates.empty else pd.NaT
        info[str(code)] = {
            "first_gtm_date": first_date if pd.notna(first_date) else None,
            "is_esp": bool(grp["is_esp"].fillna(False).any()),
            "event_90d_flag": bool(((dates >= pd.Timestamp(forecast_start)) & (dates <= horizon_90)).any()),
        }
    return info


def _advance_days(plan: PlanData, wid: str, source_cutoff: pd.Timestamp | None, forecast_start: date) -> int:
    if source_cutoff is None or wid not in plan.op_days.index:
        return 0
    start_ts = pd.Timestamp(source_cutoff) + pd.Timedelta(days=1)
    end_ts = pd.Timestamp(forecast_start)
    if start_ts >= end_ts:
        return 0
    total = 0.0
    cursor = pd.Timestamp(start_ts.year, start_ts.month, 1)
    while cursor < end_ts:
        month_id = cursor.strftime("%Y-%m")
        if month_id in plan.op_days.columns:
            next_month = cursor + pd.offsets.MonthBegin(1)
            overlap_start = max(start_ts, cursor)
            overlap_end = min(end_ts, next_month)
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days > 0:
                cal_days = float(plan.cal_days.get(month_id, pd.Period(month_id, freq="M").days_in_month))
                op_days = float(plan.op_days.at[wid, month_id])
                total += op_days * overlap_days / cal_days if cal_days > 0 else 0.0
        cursor = cursor + pd.offsets.MonthBegin(1)
    return int(round(total))


def _build_age_pools(
    plan: PlanData,
    esp_source: EspSource,
    model: StrataModel,
    forecast_start: date,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[float]]:
    by_stratum: dict[str, list[float]] = defaultdict(list)
    by_field: dict[str, list[float]] = defaultdict(list)
    global_ages: list[float] = []
    for wid in plan.producers:
        st = esp_source.states_by_well.get(wid)
        if st is None or not st.is_active or st.age_op is None:
            continue
        age = max(0.0, float(st.age_op) + _advance_days(plan, wid, esp_source.source_cutoff, forecast_start))
        model_field = map_model_field_from_well(wid)
        _, stratum = model.resolve(model_field, st.sour, st.ctr)
        by_stratum[stratum].append(age)
        field_key = None if stratum == "Global_Pooled" else stratum.split("_", 1)[0]
        if field_key:
            by_field[field_key].append(age)
        global_ages.append(age)
    return dict(by_stratum), dict(by_field), global_ages


def _techregime_age(
    tr_status: CurrentTechregimeStatus | None,
    plan: PlanData,
    wid: str,
    forecast_start: date,
) -> int | None:
    """Measured current-pump operating age from the techregime report, advanced by the
    plan's operating days between the report date and forecast start.  None when the
    report has no usable ННО for the well."""
    if tr_status is None or tr_status.age_op is None:
        return None
    age = float(tr_status.age_op)
    if not np.isfinite(age) or age < 0:
        return None
    if tr_status.report_date is not None:
        age += _advance_days(plan, wid, pd.Timestamp(tr_status.report_date), forecast_start)
    return int(round(age))


def _impute_age_pmf(
    stratum: str,
    model_field: str | None,
    ages_by_stratum: dict[str, list[float]],
    ages_by_field: dict[str, list[float]],
    global_ages: list[float],
) -> tuple[dict[int, float], str, str]:
    if stratum in ages_by_stratum and ages_by_stratum[stratum]:
        return compress_age_pmf(ages_by_stratum[stratum]), "imputed_stratum", "medium"
    if model_field and model_field in ages_by_field and ages_by_field[model_field]:
        return compress_age_pmf(ages_by_field[model_field]), "imputed_field", "medium"
    if global_ages:
        return compress_age_pmf(global_ages), "imputed_global", "low"
    return {0: 1.0}, "imputed_empty", "low"


def _well_arrays(plan: PlanData, wid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fwd = plan.fwd_months
    oil = plan.oil_volume.loc[wid, fwd].to_numpy(dtype=float) if wid in plan.oil_volume.index else np.zeros(len(fwd))
    liq = (
        plan.liquid_volume.loc[wid, fwd].to_numpy(dtype=float)
        if wid in plan.liquid_volume.index
        else np.zeros(len(fwd))
    )
    op = plan.op_days.loc[wid, fwd].to_numpy(dtype=float) if wid in plan.op_days.index else np.zeros(len(fwd))
    cal = plan.cal_days.loc[fwd].to_numpy(dtype=float)
    return oil, liq, op, cal


def _apply_time_map_idle_exposure(
    plan: PlanData,
    wid: str,
    model: StrataModel,
    model_field: str | None,
    reporting_field: str,
    age_start: float,
    op_days: np.ndarray,
    cal_days: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Add reduced hazard exposure in plan-idle months when the bundle map says so."""
    time_map = getattr(model, "time_map", None)
    if time_map is None or not getattr(time_map, "available", False):
        return op_days, 0.0, "none"
    idle_frac, idle_source = time_map.idle_fraction(reporting_field, model_field)
    if idle_frac <= 0.0:
        return op_days, 0.0, idle_source
    out = op_days.astype(float, copy=True)
    age = max(0.0, float(age_start))
    added = 0.0
    for idx, month in enumerate(plan.fwd_months):
        if idx >= len(out) or idx >= len(cal_days):
            break
        if out[idx] > 0.0:
            age += float(out[idx])
            continue
        uptime, _ = time_map.predict_uptime(
            model_field,
            age,
            pd.Period(month, freq="M").month,
            fallback=0.0,
        )
        extra = max(0.0, float(cal_days[idx]) * float(uptime) * float(idle_frac))
        out[idx] = extra
        added += extra
        age += extra
    return out, float(added), idle_source


def build_well_states(
    plan: PlanData,
    esp_source: EspSource,
    gtm: pd.DataFrame,
    model: StrataModel,
    cfg: C.RunConfig,
    current_status: dict[str, CurrentTechregimeStatus] | None = None,
) -> tuple[list[WellState], pd.DataFrame]:
    gtm_info = _gtm_info(gtm, cfg.forecast_start)
    ages_by_stratum, ages_by_field, global_ages = _build_age_pools(plan, esp_source, model, cfg.forecast_start)

    states: list[WellState] = []
    audit_rows: list[dict] = []
    current_status = current_status or {}
    for wid in plan.producers:
        meta = plan.producer_meta.loc[wid]
        oil, liq, op, cal = _well_arrays(plan, wid)
        gtm_row = gtm_info.get(wid, {})
        st = esp_source.states_by_well.get(wid)
        tr_status = current_status.get(wid)
        current_status_text = tr_status.status if tr_status is not None else None
        current_in_operation = tr_status.in_operation if tr_status is not None else None
        current_status_date = (
            tr_status.report_date.date().isoformat()
            if tr_status is not None and tr_status.report_date is not None
            else None
        )
        scope_label = "included"
        covariates: dict[str, float] = {}
        cov_source = "none"
        # Current techregime as a measured age source: «Время наработки (ННО)» is the
        # current pump's operating age (day scale validated against «Свод» actives).
        tr_age = _techregime_age(tr_status, plan, wid, cfg.forecast_start)
        tr_is_esp = bool(tr_status is not None and tr_status.lift and "ЭЦН" in str(tr_status.lift).upper())

        if st is not None:
            model_field = map_model_field_from_well(wid)
            params, stratum = model.resolve(model_field, st.sour, st.ctr)
            model_field = None if stratum == "Global_Pooled" else stratum.split("_", 1)[0]
            if st.is_active and st.age_op is not None:
                age = int(round(max(0.0, float(st.age_op) + _advance_days(plan, wid, esp_source.source_cutoff, cfg.forecast_start))))
                age_pmf = {age: 1.0}
                age_source = "history_running"
                confidence = "high"
                state_label = "exact_active"
                if current_in_operation is False:
                    state_label = "exact_active_off_today"
                covariates = dict(st.covariates)
                cov_source = st.cov_source
            elif tr_is_esp and tr_age is not None:
                # «Свод» record is stale (last run failed) but the techregime report
                # carries the replacement pump's measured operating age.
                age_pmf = {tr_age: 1.0}
                age_source = "techregime_nno"
                confidence = "high" if current_in_operation else "medium"
                state_label = "exact_tr_age"
            else:
                age_pmf, age_source, confidence = _impute_age_pmf(
                    stratum, model_field, ages_by_stratum, ages_by_field, global_ages
                )
                state_label = "exact_stale_imputed"
            contractor_group = st.ctr
            contractor_source = "exact_history"
            sour_class = st.sour
            sour_source = "exact_history"
            include_primary = True
        else:
            model_field = map_model_field_from_well(wid)
            params, stratum = model.resolve(model_field, "nonsour", "Pooled")
            if tr_is_esp and tr_age is not None:
                age_pmf = {tr_age: 1.0}
                age_source = "techregime_nno"
                confidence = "medium"
                state_label = "new_tr_esp"
            else:
                age_pmf = {0: 1.0}
                age_source = "new_start"
                confidence = "medium" if gtm_row.get("is_esp") else "low"
                state_label = "new_df_esp" if gtm_row.get("is_esp") else "new_plan_only"
            contractor_group = "Pooled"
            contractor_source = "fallback_pooled"
            sour_class = "nonsour"
            sour_source = "fallback_nonsour"
            # ГТМ ЭЦН flag or a techregime row on ЭЦН lift are both direct evidence
            # the well is ESP-operated.
            include_primary = (
                bool(gtm_row.get("is_esp")) or tr_is_esp or cfg.esp_scope_policy != C.ESP_SCOPE_CONSERVATIVE
            )
            if not include_primary:
                scope_label = "excluded_non_esp"

        age_mean = weighted_age_mean(age_pmf)
        op_for_hazard, idle_op_added, idle_op_source = _apply_time_map_idle_exposure(
            plan,
            wid,
            model,
            model_field,
            str(meta.get("plan_field", "")),
            age_mean,
            op,
            cal,
        )
        first_gtm_date = gtm_row.get("first_gtm_date")
        event_90d_flag = bool(gtm_row.get("event_90d_flag", False))
        states.append(
            WellState(
                code=wid,
                raw_id=str(meta.get("raw_id", wid)),
                plan_field=str(meta.get("plan_field", "")),
                model_field=model_field,
                stratum=stratum,
                params=params,
                age_pmf=age_pmf,
                age_source=age_source,
                age_mean=age_mean,
                state_label=state_label,
                scope_label=scope_label,
                confidence=confidence,
                contractor_group=contractor_group,
                contractor_source=contractor_source,
                sour_class=sour_class,
                sour_source=sour_source,
                covariates=covariates,
                cov_source=cov_source,
                current_status=current_status_text,
                current_in_operation=current_in_operation,
                current_status_date=current_status_date,
                include_primary=include_primary,
                oil_volume=oil,
                liquid_volume=liq,
                op_days=op_for_hazard,
                cal_days=cal,
                first_active_month=meta.get("first_active_month"),
                first_gtm_date=first_gtm_date,
                event_90d_flag=event_90d_flag,
            )
        )
        audit_rows.append(
            {
                "wid": wid,
                "raw_id": str(meta.get("raw_id", wid)),
                "plan_field": str(meta.get("plan_field", "")),
                "state_label": state_label,
                "scope_label": scope_label,
                "confidence": confidence,
                "model_field": model_field or "",
                "stratum": stratum,
                "planned_oil_t": float(meta.get("planned_oil_t", 0.0)),
                "planned_liquid_m3": float(meta.get("planned_liquid_m3", 0.0)),
                "first_active_month": meta.get("first_active_month"),
                "age_source": age_source,
                "age_mean": round(age_mean, 1),
                "contractor_group": contractor_group,
                "contractor_source": contractor_source,
                "sour_class": sour_class,
                "sour_source": sour_source,
                "cov_source": cov_source,
                "current_status": current_status_text,
                "current_in_operation": current_in_operation,
                "current_status_date": current_status_date,
                "first_gtm_date": first_gtm_date,
                "event_90d_flag": event_90d_flag,
                "runtime_anomaly_count": int(meta.get("runtime_anomaly_count", 0)),
                "registry_oil_positive_months": int(meta.get("registry_oil_positive_months", 0)),
                "registry_liquid_positive_months": int(meta.get("registry_liquid_positive_months", 0)),
                "summary_positive_months": int(meta.get("summary_positive_months", 0)),
                "time_map_idle_op_days_added": round(float(idle_op_added), 3),
                "time_map_idle_source": idle_op_source,
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(["scope_label", "planned_oil_t"], ascending=[True, False])
    return states, audit


def _downtime_days(
    model_field: str | None,
    global_stats: DowntimeStats,
    field_stats: dict[str, DowntimeStats],
    key: str,
    override_days: int | None = None,
) -> int:
    if override_days is not None:
        # floor 1: at 0 the renewal queue disappears and failed mass never returns
        return max(int(override_days), 1)
    stats = field_stats.get(model_field or "")
    if stats is not None and stats.n >= C.MIN_FIELD_DOWNTIME_N:
        return int(getattr(stats, key))
    return int(getattr(global_stats, key))


def _opdays_first_90_calendar(well: WellState) -> float:
    remaining = 90
    total = 0.0
    for op_days, cal_days in zip(well.op_days, well.cal_days):
        if remaining <= 0:
            break
        take = min(int(round(float(cal_days))), remaining)
        p = float(op_days / cal_days) if cal_days > 0 else 0.0
        total += p * take
        remaining -= take
    return total


def _planned_oil_first_90_calendar(well: WellState) -> float:
    remaining = 90
    total = 0.0
    for oil, cal_days in zip(well.oil_volume, well.cal_days):
        if remaining <= 0:
            break
        month_days = int(round(float(cal_days)))
        if month_days <= 0:
            continue
        take = min(month_days, remaining)
        total += float(oil) * (take / month_days)
        remaining -= take
    return total


def run_projection(
    states: list[WellState],
    scenarios: tuple[C.ScenarioSpec, ...],
    model: StrataModel,
    hazard: HazardLayer,
    global_downtime: DowntimeStats,
    field_downtime: dict[str, DowntimeStats],
    months: list[str],
    changeout_p90: float = C.CHANGEOUT_P90_THRESHOLD,
    downtime_override_days: int | None = None,
) -> pd.DataFrame:
    recs: list[dict] = []
    for well in states:
        if not well.include_primary:
            continue
        oil_per_op = np.divide(
            well.oil_volume,
            well.op_days,
            out=np.zeros_like(well.oil_volume, dtype=float),
            where=well.op_days > 0,
        )
        liq_per_op = np.divide(
            well.liquid_volume,
            well.op_days,
            out=np.zeros_like(well.liquid_volume, dtype=float),
            where=well.op_days > 0,
        )
        op_90 = _opdays_first_90_calendar(well)
        oil_90 = _planned_oil_first_90_calendar(well)
        for spec in scenarios:
            params, theta = scenario_params(well.params, well.covariates, hazard, spec.hazard_mode)
            downtime = _downtime_days(
                well.model_field,
                global_downtime,
                field_downtime,
                spec.downtime_key,
                override_days=downtime_override_days,
            )
            failures, lost_op_days = project_well(well, params, downtime, model)
            oil_loss = lost_op_days * oil_per_op
            liq_loss = lost_op_days * liq_per_op
            p90 = current_pump_p_fail(well.age_pmf, params, op_90, model)
            evar_90 = p90 * oil_90
            for mi, month in enumerate(months):
                recs.append(
                    {
                        "scenario": spec.scenario_id,
                        "scenario_label": spec.label_ru,
                        "scenario_primary": spec.primary,
                        "wid": well.code,
                        "raw_id": well.raw_id,
                        "plan_field": well.plan_field,
                        "model_field": well.model_field or "",
                        "stratum": well.stratum,
                        "state_label": well.state_label,
                        "scope_label": well.scope_label,
                        "confidence": well.confidence,
                        "age_source": well.age_source,
                        "age_mean": round(well.age_mean, 1),
                        "cov_source": well.cov_source,
                        "contractor_group": well.contractor_group,
                        "sour_class": well.sour_class,
                        "event_90d_flag": well.event_90d_flag,
                        "first_gtm_date": well.first_gtm_date,
                        "month": month,
                        "planned_oil_t": float(well.oil_volume[mi]),
                        "planned_liquid_m3": float(well.liquid_volume[mi]),
                        "planned_op_days": float(well.op_days[mi]),
                        "calendar_days": float(well.cal_days[mi]),
                        "expected_failures": float(failures[mi]),
                        "expected_lost_op_days": float(lost_op_days[mi]),
                        "oil_loss_t": float(oil_loss[mi]),
                        "liquid_loss_m3": float(liq_loss[mi]),
                        "p_fail_90d": float(p90),
                        "exp_value_at_risk_90d_t": float(evar_90),
                        "theta": float(theta),
                        "downtime_days": int(downtime),
                        "flag_changeout": "CHANGE-OUT" if (p90 >= changeout_p90 and oil_90 > 0) else "",
                    }
                )
    return pd.DataFrame(recs)


def _plan_monthly_totals(plan: PlanData) -> pd.DataFrame:
    rows = []
    for month in plan.fwd_months:
        rows.append(
            {
                "month": month,
                "total_planned_oil_t": float(plan.oil_volume.loc[plan.producers, month].sum()),
                "total_planned_liquid_m3": float(plan.liquid_volume.loc[plan.producers, month].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plan_field_monthly_totals(plan: PlanData) -> pd.DataFrame:
    rows: list[dict] = []
    for wid in plan.producers:
        field = str(plan.producer_meta.at[wid, "plan_field"])
        for month in plan.fwd_months:
            rows.append(
                {
                    "plan_field": field,
                    "month": month,
                    "total_planned_oil_t": float(plan.oil_volume.at[wid, month]),
                    "total_planned_liquid_m3": float(plan.liquid_volume.at[wid, month]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["plan_field", "month", "total_planned_oil_t", "total_planned_liquid_m3"])
    return pd.DataFrame(rows).groupby(["plan_field", "month"], as_index=False).sum()


def production_at_risk(proj: pd.DataFrame, plan: PlanData) -> dict[str, pd.DataFrame]:
    total_monthly = _plan_monthly_totals(plan)
    covered_monthly = (
        proj.groupby(["scenario", "scenario_label", "scenario_primary", "month"], as_index=False)
        .agg(
            covered_planned_oil_t=("planned_oil_t", "sum"),
            covered_planned_liquid_m3=("planned_liquid_m3", "sum"),
            expected_failures=("expected_failures", "sum"),
            expected_lost_op_days=("expected_lost_op_days", "sum"),
            oil_loss_t=("oil_loss_t", "sum"),
            liquid_loss_m3=("liquid_loss_m3", "sum"),
        )
        .merge(total_monthly, on="month", how="left")
    )
    covered_monthly["uncovered_planned_oil_t"] = (
        covered_monthly["total_planned_oil_t"] - covered_monthly["covered_planned_oil_t"]
    )
    covered_monthly["uncovered_planned_liquid_m3"] = (
        covered_monthly["total_planned_liquid_m3"] - covered_monthly["covered_planned_liquid_m3"]
    )
    covered_monthly["risk_adjusted_total_oil_t"] = covered_monthly["total_planned_oil_t"] - covered_monthly["oil_loss_t"]
    covered_monthly["risk_adjusted_total_liquid_m3"] = (
        covered_monthly["total_planned_liquid_m3"] - covered_monthly["liquid_loss_m3"]
    )
    covered_monthly["coverage_pct_oil"] = 100.0 * covered_monthly["covered_planned_oil_t"] / covered_monthly[
        "total_planned_oil_t"
    ].replace(0, np.nan)
    covered_monthly["risk_pct_total_oil"] = 100.0 * covered_monthly["oil_loss_t"] / covered_monthly[
        "total_planned_oil_t"
    ].replace(0, np.nan)

    by_well = (
        proj.groupby(
            [
                "scenario",
                "scenario_label",
                "scenario_primary",
                "wid",
                "raw_id",
                "plan_field",
                "model_field",
                "stratum",
                "state_label",
                "scope_label",
                "confidence",
                "age_source",
                "age_mean",
                "cov_source",
                "event_90d_flag",
                "first_gtm_date",
            ],
            as_index=False,
        )
        .agg(
            planned_oil_t=("planned_oil_t", "sum"),
            planned_liquid_m3=("planned_liquid_m3", "sum"),
            expected_failures=("expected_failures", "sum"),
            expected_lost_op_days=("expected_lost_op_days", "sum"),
            oil_loss_t=("oil_loss_t", "sum"),
            liquid_loss_m3=("liquid_loss_m3", "sum"),
            p_fail_90d=("p_fail_90d", "max"),
            exp_value_at_risk_90d_t=("exp_value_at_risk_90d_t", "max"),
            downtime_days=("downtime_days", "max"),
            theta=("theta", "max"),
            flag_changeout=("flag_changeout", "max"),
        )
        .sort_values(["scenario", "oil_loss_t"], ascending=[True, False])
    )
    by_well["risk_adjusted_oil_t"] = by_well["planned_oil_t"] - by_well["oil_loss_t"]
    by_well["risk_adjusted_liquid_m3"] = by_well["planned_liquid_m3"] - by_well["liquid_loss_m3"]

    total_field_monthly = _plan_field_monthly_totals(plan)
    by_field = (
        proj.groupby(["scenario", "scenario_label", "scenario_primary", "plan_field", "month"], as_index=False)
        .agg(
            covered_planned_oil_t=("planned_oil_t", "sum"),
            covered_planned_liquid_m3=("planned_liquid_m3", "sum"),
            expected_failures=("expected_failures", "sum"),
            expected_lost_op_days=("expected_lost_op_days", "sum"),
            oil_loss_t=("oil_loss_t", "sum"),
            liquid_loss_m3=("liquid_loss_m3", "sum"),
        )
        .merge(total_field_monthly, on=["plan_field", "month"], how="left")
        .fillna({"total_planned_oil_t": 0.0, "total_planned_liquid_m3": 0.0})
    )
    by_field["uncovered_planned_oil_t"] = by_field["total_planned_oil_t"] - by_field["covered_planned_oil_t"]
    by_field["uncovered_planned_liquid_m3"] = (
        by_field["total_planned_liquid_m3"] - by_field["covered_planned_liquid_m3"]
    )
    by_field["risk_adjusted_total_oil_t"] = by_field["total_planned_oil_t"] - by_field["oil_loss_t"]
    by_field["risk_adjusted_total_liquid_m3"] = by_field["total_planned_liquid_m3"] - by_field["liquid_loss_m3"]
    by_field["risk_pct_total_oil"] = 100.0 * by_field["oil_loss_t"] / by_field["total_planned_oil_t"].replace(0, np.nan)

    return {"monthly": covered_monthly, "by_well": by_well, "by_field": by_field}


def workover_load(proj: pd.DataFrame, gtm: pd.DataFrame, fwd_months: list[str]) -> pd.DataFrame:
    planned = pd.DataFrame({"month": fwd_months})
    if gtm is not None and not gtm.empty and gtm["krs_date"].notna().any():
        cnt = (
            gtm.dropna(subset=["krs_date"])
            .assign(month=lambda x: x["krs_date"].dt.strftime("%Y-%m"))
            .groupby("month")
            .size()
            .rename("planned_gtm_jobs")
            .reset_index()
        )
        planned = planned.merge(cnt, on="month", how="left")
    planned["planned_gtm_jobs"] = planned.get("planned_gtm_jobs", 0.0).fillna(0.0)
    reactive = proj.groupby(["scenario", "scenario_label", "scenario_primary", "month"], as_index=False).agg(
        reactive_esp_failures=("expected_failures", "sum")
    )
    out = reactive.merge(planned, on="month", how="left")
    out["total_workover_demand"] = out["planned_gtm_jobs"].fillna(0.0) + out["reactive_esp_failures"]
    return out.sort_values(["scenario", "month"])


def changeout(by_well: pd.DataFrame) -> pd.DataFrame:
    focus = by_well[
        by_well["scenario"].isin([C.PRIMARY_SCENARIO_ID, C.STRESS_SCENARIO_ID])
    ].copy()
    if focus.empty:
        return focus
    id_cols = [
        "wid",
        "raw_id",
        "plan_field",
        "model_field",
        "stratum",
        "state_label",
        "scope_label",
        "confidence",
        "age_source",
        "age_mean",
        "event_90d_flag",
        "first_gtm_date",
    ]
    pivot = focus.pivot_table(
        index=id_cols,
        columns="scenario",
        values=["p_fail_90d", "exp_value_at_risk_90d_t", "flag_changeout", "theta"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{scenario}" for metric, scenario in pivot.columns]
    out = pivot.reset_index()
    for scenario in (C.PRIMARY_SCENARIO_ID, C.STRESS_SCENARIO_ID):
        if f"p_fail_90d_{scenario}" not in out.columns:
            out[f"p_fail_90d_{scenario}"] = np.nan
            out[f"exp_value_at_risk_90d_t_{scenario}"] = np.nan
            out[f"flag_changeout_{scenario}"] = ""
            out[f"theta_{scenario}"] = np.nan
    return out.sort_values(
        [f"p_fail_90d_{C.PRIMARY_SCENARIO_ID}", f"exp_value_at_risk_90d_t_{C.PRIMARY_SCENARIO_ID}"],
        ascending=[False, False],
    ).reset_index(drop=True)
