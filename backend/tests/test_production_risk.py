from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import openpyxl

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk.crosswalk import (
    DowntimeStats,
    PlanData,
    load_plan,
    load_current_techregime_status,
)
from analysis.workflows.production_risk.layers import _downtime_days, _impute_age_pmf
from analysis.workflows.production_risk import repair_compat
from analysis.workflows.production_risk.survival import (
    HazardLayer,
    StrataModel,
    WellState,
    current_pump_p_fail,
    project_well,
)


def _bisect_b50(params: dict[str, float]) -> float:
    lo, hi = 0.0, 15.0 * max(params["eta1"], params["eta2"])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = (
            params["w1"] * math.exp(-((mid / params["eta1"]) ** params["beta1"]))
            + (1.0 - params["w1"]) * math.exp(-((mid / params["eta2"]) ** params["beta2"]))
        )
        if s > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _make_plan_workbook(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws_registry = wb.active
    ws_registry.title = "Реестр"
    ws_summary = wb.create_sheet("Сводные данные")

    headers = [
        "Месторождение",
        "УН",
        "Объект сбора",
        "Обьект подготовки",
        "Слой",
        "Дата ввода",
        "Резерв",
        "Тип нефти",
        "Показатель_2",
        "ЕИ",
        "Показатель",
        "Well_ID*",
        "№ куста*",
        datetime(2026, 1, 1),
    ]
    for ws in (ws_registry, ws_summary):
        for idx, value in enumerate(headers, start=1):
            ws.cell(row=3, column=idx, value=value)

    summary_rows = [
        ["Ярактинское НГКМ", "", "", "", "", "", "", "", "Добыча нефти", "т", "", "Ya_001", "1", 310.0],
        ["Ярактинское НГКМ", "", "", "", "", "", "", "", "Добыча жидкости", "м3", "", "Ya_001", "1", 620.0],
        ["Ярактинское НГКМ", "", "", "", "", "", "", "", "Отработанное время", "дн", "", "Ya_001", "1", 40.0],
    ]
    registry_rows = [
        ["Ярактинское НГКМ", "", "", "", "", "", "", "", "Добыча нефти", "т/сут", "", "Ya_001", "1", 10.0],
        ["Ярактинское НГКМ", "", "", "", "", "", "", "", "Добыча жидкости", "м3/сут", "", "Ya_001", "1", 20.0],
    ]
    for row_idx, row in enumerate(summary_rows, start=4):
        for col_idx, value in enumerate(row, start=1):
            ws_summary.cell(row=row_idx, column=col_idx, value=value)
    for row_idx, row in enumerate(registry_rows, start=4):
        for col_idx, value in enumerate(row, start=1):
            ws_registry.cell(row=row_idx, column=col_idx, value=value)
    wb.save(path)


def test_strata_resolution_matches_vba_fallback():
    model = StrataModel()
    _, key_exact = model.resolve("Ya", "nonsour", "brt")
    _, key_pooled = model.resolve("Vt", "sour", "missing")
    _, key_global = model.resolve("Unknown", "nonsour", "brt")
    assert key_exact == "Ya_nonsour_brt"
    assert key_pooled == "Vt_sour_Pooled"
    assert key_global == "Global_Pooled"


def test_b50_and_conditional_probability_from_registry():
    model = StrataModel()
    params, _ = model.resolve("Ya", "nonsour", "brt")
    b50 = _bisect_b50(params)
    assert abs(b50 - params["b50"]) / params["b50"] <= 0.01
    p30 = current_pump_p_fail({100: 1.0}, params, 30.0, model)
    p90 = current_pump_p_fail({100: 1.0}, params, 90.0, model)
    assert 0.0 <= p30 <= p90 <= 1.0


def test_theta_reference_and_missing_inputs_are_neutral():
    hazard = HazardLayer()
    coeffs = hazard.coeffs
    ref = {str(row["covariate"]): float(row["reference_value"]) for _, row in coeffs.iterrows()}
    assert abs(hazard.theta(ref) - 1.0) < 1e-12
    assert abs(hazard.theta({}) - 1.0) < 1e-12


def test_project_well_zero_uptime_zero_losses():
    model = StrataModel()
    params, _ = model.resolve("Ya", "nonsour", "brt")
    well = WellState(
        code="Ya_001",
        raw_id="Ya_001",
        plan_field="Ярактинское НГКМ",
        model_field="Ya",
        stratum="Ya_nonsour_brt",
        params=params,
        age_pmf={100: 1.0},
        age_source="history_running",
        age_mean=100.0,
        state_label="exact_active",
        scope_label="included",
        confidence="high",
        contractor_group="brt",
        contractor_source="exact_history",
        sour_class="nonsour",
        sour_source="exact_history",
        covariates={},
        cov_source="none",
        current_status=None,
        current_in_operation=None,
        current_status_date=None,
        include_primary=True,
        oil_volume=np.array([0.0, 0.0]),
        liquid_volume=np.array([0.0, 0.0]),
        op_days=np.array([0.0, 0.0]),
        cal_days=np.array([31.0, 28.0]),
        first_active_month=None,
        first_gtm_date=None,
        event_90d_flag=False,
    )
    failures, lost = project_well(well, params, 16, model)
    assert np.allclose(failures, 0.0)
    assert np.allclose(lost, 0.0)


def test_project_well_deterministic_outputs_are_bounded():
    model = StrataModel()
    params, _ = model.resolve("Ya", "nonsour", "brt")
    well = WellState(
        code="Ya_002",
        raw_id="Ya_002",
        plan_field="Ярактинское НГКМ",
        model_field="Ya",
        stratum="Ya_nonsour_brt",
        params=params,
        age_pmf={100: 1.0},
        age_source="history_running",
        age_mean=100.0,
        state_label="exact_active",
        scope_label="included",
        confidence="high",
        contractor_group="brt",
        contractor_source="exact_history",
        sour_class="nonsour",
        sour_source="exact_history",
        covariates={},
        cov_source="none",
        current_status=None,
        current_in_operation=None,
        current_status_date=None,
        include_primary=True,
        oil_volume=np.array([310.0]),
        liquid_volume=np.array([620.0]),
        op_days=np.array([31.0]),
        cal_days=np.array([31.0]),
        first_active_month="2026-07",
        first_gtm_date=None,
        event_90d_flag=False,
    )
    failures, lost = project_well(well, params, 16, model)
    assert failures.shape == (1,)
    assert lost.shape == (1,)
    assert 0.0 <= failures[0] <= 31.0
    assert 0.0 <= lost[0] <= 31.0


def test_age_imputation_prefers_stratum_then_field_then_global():
    pmf, source, conf = _impute_age_pmf("Ya_nonsour_brt", "Ya", {"Ya_nonsour_brt": [10, 20]}, {"Ya": [30]}, [40])
    assert source == "imputed_stratum"
    assert conf == "medium"
    assert abs(sum(pmf.values()) - 1.0) < 1e-12

    _, source, _ = _impute_age_pmf("Za_nonsour_brt", "Ya", {}, {"Ya": [30]}, [40])
    assert source == "imputed_field"

    _, source, conf = _impute_age_pmf("Za_nonsour_brt", "Za", {}, {}, [40])
    assert source == "imputed_global"
    assert conf == "low"


def test_downtime_days_use_field_when_n_is_enough_else_global():
    global_stats = DowntimeStats(n=100, p25=7, p50=16, p75=44, mean=22)
    field_stats = {
        "Ya": DowntimeStats(n=31, p25=8, p50=17, p75=47, mean=24),
        "Mc": DowntimeStats(n=10, p25=4, p50=10, p75=20, mean=12),
    }
    assert _downtime_days("Ya", global_stats, field_stats, "p50") == 17
    assert _downtime_days("Mc", global_stats, field_stats, "p50") == 16
    assert _downtime_days("None", global_stats, field_stats, "p75") == 44


def test_load_plan_uses_summary_sheet_and_clips_runtime(tmp_path: Path):
    path = tmp_path / "plan.xlsx"
    _make_plan_workbook(path)
    plan = load_plan(date(2026, 1, 1), date(2026, 1, 1), master_path=path)
    assert plan.oil_volume.at["YA_001", "2026-01"] == 310.0
    assert plan.registry_oil_rate.at["YA_001", "2026-01"] == 10.0
    assert plan.op_days.at["YA_001", "2026-01"] == 31.0
    assert not plan.anomalies.empty
    assert "runtime_gt_calendar" in set(plan.anomalies["reason"])


def test_load_current_techregime_status_marks_off_today(tmp_path: Path):
    path = tmp_path / "tr.xlsm"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ТР НЕФТЬ"
    ws.cell(row=3, column=4, value=datetime(2026, 7, 1))
    headers = {
        4: "ID скважины",
        46: "Способ эксплуатации",
        48: "Время наработки (ННО)",
        100: "Состояние по фонду",
    }
    for col, name in headers.items():
        ws.cell(row=8, column=col, value=name)
    ws.cell(row=9, column=4, value="Ya_001")
    ws.cell(row=9, column=46, value="ЭЦН")
    ws.cell(row=9, column=48, value=123)
    ws.cell(row=9, column=100, value="В бездействии прошлых лет")
    ws.cell(row=10, column=4, value="Ya_002")
    ws.cell(row=10, column=46, value="ЭЦН")
    ws.cell(row=10, column=48, value=45)
    ws.cell(row=10, column=100, value="В работе")
    wb.save(path)

    statuses = load_current_techregime_status(path)
    assert statuses["YA_001"].in_operation is False
    assert statuses["YA_001"].age_op == 123.0
    assert statuses["YA_002"].in_operation is True


def test_repair_compat_preserves_legacy_status_shape(tmp_path: Path):
    model = StrataModel()
    params, _ = model.resolve("Ya", "nonsour", "brt")
    state = WellState(
        code="YA_001",
        raw_id="Ya_001",
        plan_field="Ярактинское НГКМ",
        model_field="Ya",
        stratum="Ya_nonsour_brt",
        params=params,
        age_pmf={100: 1.0},
        age_source="history_running",
        age_mean=100.0,
        state_label="exact_active",
        scope_label="included",
        confidence="high",
        contractor_group="brt",
        contractor_source="exact_history",
        sour_class="nonsour",
        sour_source="exact_history",
        covariates={},
        cov_source="none",
        current_status="В работе",
        current_in_operation=True,
        current_status_date="2026-07-01",
        include_primary=True,
        oil_volume=np.array([310.0]),
        liquid_volume=np.array([620.0]),
        op_days=np.array([31.0]),
        cal_days=np.array([31.0]),
        first_active_month="2026-07",
        first_gtm_date=None,
        event_90d_flag=False,
    )
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "scenario": C.PRIMARY_SCENARIO_ID,
                "wid": "YA_001",
                "raw_id": "Ya_001",
                "plan_field": "Ярактинское НГКМ",
                "month": "2026-07",
                "planned_oil_t": 310.0,
                "planned_op_days": 31.0,
                "expected_failures": 1.2,
                "downtime_days": 3,
            }
        ]
    )
    cfg = C.RunConfig(forecast_start=date(2026, 7, 1), horizon_end=date(2026, 7, 1), write_excel=False)
    payload = repair_compat.build(df, [state], cfg)
    assert payload["rows"][0]["used_prediction_source"] == "survival"
    assert payload["rows"][0]["catboost_nno"] is None
    assert len(payload["rows"][0]["statuses"]) == len(payload["dates"])
    assert payload["rows"][0]["statuses"].count(0) == 3
    assert payload["rows"][0]["downtime_days"] == 3

    path = repair_compat.write_excel(tmp_path / "compat.xlsx", payload)
    assert path.exists()
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        assert "Repair forecast" in wb.sheetnames
        assert "Summary" in wb.sheetnames
    finally:
        wb.close()


def test_event_allocation_preserves_fleet_totals_and_month_profile():
    """Fleet event count must equal round(sum expected); months must not collapse
    into waves (the old floor-walk produced 0-event and 146-event months)."""
    import pandas as pd

    months = [f"2026-{m:02d}" for m in range(7, 13)] + [f"2027-{m:02d}" for m in range(1, 13)]
    rows = []
    n_wells = 200
    per_month = 0.06  # E_w = 1.08 per well; fleet expected = 216
    for i in range(n_wells):
        for month in months:
            rows.append(
                {
                    "scenario": C.PRIMARY_SCENARIO_ID,
                    "wid": f"YA_{i:03d}",
                    "raw_id": f"Ya_{i:03d}",
                    "plan_field": "Ярактинское НГКМ",
                    "month": month,
                    "planned_oil_t": 300.0,
                    "planned_op_days": 30.0,
                    "expected_failures": per_month,
                    "downtime_days": 1,
                }
            )
    df = pd.DataFrame(rows)
    cfg = C.RunConfig(forecast_start=date(2026, 7, 1), horizon_end=date(2027, 12, 1), write_excel=False)
    payload = repair_compat.build(df, [], cfg)

    total_expected = per_month * len(months) * n_wells
    total_allocated = sum(len(r["event_dates"]) for r in payload["rows"])
    assert total_allocated == round(total_expected)

    by_month: dict[str, int] = {}
    for row in payload["rows"]:
        for iso in row["event_dates"]:
            key = iso[:7]
            by_month[key] = by_month.get(key, 0) + 1
    expected_monthly = total_expected / len(months)  # 12 per month
    assert set(by_month) <= set(months)
    for month in months:
        got = by_month.get(month, 0)
        # uniform profile -> every month populated, no waves.  The old floor-walk
        # produced months with 0 and months with 146 events; hash-level fluctuation
        # around the mean (12) is expected and fine.
        assert got > 0, f"month {month} got zero events"
        assert got <= 3.0 * expected_monthly, (month, got)


def test_off_today_wells_get_downtime_block_not_single_day():
    import pandas as pd

    model = StrataModel()
    params, _ = model.resolve("Ya", "nonsour", "brt")
    state = WellState(
        code="YA_001",
        raw_id="Ya_001",
        plan_field="Ярактинское НГКМ",
        model_field="Ya",
        stratum="Ya_nonsour_brt",
        params=params,
        age_pmf={100: 1.0},
        age_source="history_running",
        age_mean=100.0,
        state_label="exact_active_off_today",
        scope_label="included",
        confidence="high",
        contractor_group="brt",
        contractor_source="exact_history",
        sour_class="nonsour",
        sour_source="exact_history",
        covariates={},
        cov_source="none",
        current_status="Остановлена",
        current_in_operation=False,
        current_status_date="2026-07-01",
        include_primary=True,
        oil_volume=np.array([310.0, 300.0]),
        liquid_volume=np.array([620.0, 600.0]),
        op_days=np.array([31.0, 31.0]),
        cal_days=np.array([31.0, 31.0]),
        first_active_month="2026-07",
        first_gtm_date=None,
        event_90d_flag=False,
    )
    df = pd.DataFrame(
        [
            {
                "scenario": C.PRIMARY_SCENARIO_ID,
                "wid": "YA_001",
                "raw_id": "Ya_001",
                "plan_field": "Ярактинское НГКМ",
                "month": "2026-07",
                "planned_oil_t": 310.0,
                "planned_op_days": 31.0,
                "expected_failures": 0.0,
                "downtime_days": 16,
            },
            {
                "scenario": C.PRIMARY_SCENARIO_ID,
                "wid": "YA_001",
                "raw_id": "Ya_001",
                "plan_field": "Ярактинское НГКМ",
                "month": "2026-08",
                "planned_oil_t": 300.0,
                "planned_op_days": 31.0,
                "expected_failures": 0.0,
                "downtime_days": 16,
            },
        ]
    )
    cfg = C.RunConfig(forecast_start=date(2026, 7, 1), horizon_end=date(2026, 8, 1), write_excel=False)
    payload = repair_compat.build(df, [state], cfg)
    statuses = payload["rows"][0]["statuses"]
    # down for the full scenario downtime block from forecast start, then back up
    assert statuses[:16] == [0] * 16
    assert statuses[16] == 1


def test_techregime_age_used_for_stale_and_new_wells():
    from analysis.workflows.production_risk.crosswalk import CurrentTechregimeStatus
    from analysis.workflows.production_risk.layers import _techregime_age

    class _PlanStub:
        op_days = __import__("pandas").DataFrame()
        cal_days = __import__("pandas").Series(dtype=float)

    tr = CurrentTechregimeStatus(
        well_code="YA_001",
        status="В работе",
        in_operation=True,
        lift="ЭЦН",
        age_op=234.0,
        report_date=datetime(2026, 7, 1),
    )
    assert _techregime_age(tr, _PlanStub(), "YA_001", date(2026, 7, 1)) == 234
    assert _techregime_age(None, _PlanStub(), "YA_001", date(2026, 7, 1)) is None
    tr_no_age = CurrentTechregimeStatus(
        well_code="YA_002", status="В работе", in_operation=True, lift="ЭЦН", age_op=None, report_date=None
    )
    assert _techregime_age(tr_no_age, _PlanStub(), "YA_002", date(2026, 7, 1)) is None


def test_lazy_analysis_import_keeps_scipy_out():
    """Guards the frozen-EXE size: importing the production-risk chain must not drag
    in the heavy scientific stack via analysis/__init__."""
    import subprocess
    import sys as _sys

    code = (
        "import sys; sys.path.insert(0, r'" + str(Path(__file__).resolve().parents[1]) + "'); "
        "import analysis; import analysis.workflows.production_risk.survival; "
        "assert 'scipy' not in sys.modules, 'scipy leaked into the lazy import chain'; "
        "from analysis import weibull_survival; import scipy; print('ok')"
    )
    proc = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_repair_compat_spreads_same_month_events_by_well():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "scenario": C.PRIMARY_SCENARIO_ID,
                "wid": "YA_001",
                "raw_id": "Ya_001",
                "plan_field": "Ярактинское НГКМ",
                "month": "2026-07",
                "planned_oil_t": 310.0,
                "planned_op_days": 31.0,
                "expected_failures": 1.0,
                "downtime_days": 1,
            },
            {
                "scenario": C.PRIMARY_SCENARIO_ID,
                "wid": "YA_002",
                "raw_id": "Ya_002",
                "plan_field": "Ярактинское НГКМ",
                "month": "2026-07",
                "planned_oil_t": 310.0,
                "planned_op_days": 31.0,
                "expected_failures": 1.0,
                "downtime_days": 1,
            },
        ]
    )
    cfg = C.RunConfig(forecast_start=date(2026, 7, 1), horizon_end=date(2026, 7, 1), write_excel=False)
    payload = repair_compat.build(df, [], cfg)
    dates = [row["event_dates"][0] for row in payload["rows"]]
    assert len(set(dates)) == len(dates)
