"""Russian-language companion workbook for production-risk outputs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from analysis.workflows.production_risk import config as C

_HDR_FILL = PatternFill("solid", fgColor="1F4E79")
_HDR_FONT = Font(bold=True, color="FFFFFF")
_TITLE_FONT = Font(bold=True, size=13)
_NOTE_FONT = Font(italic=True, size=9, color="666666")

_COLS_RU = {
    "scenario": "Сценарий",
    "scenario_label": "Сценарий (RU)",
    "scenario_primary": "Основной сценарий",
    "month": "Месяц",
    "plan_field": "Месторождение",
    "wid": "Скважина",
    "raw_id": "Well_ID",
    "model_field": "Поле модели",
    "stratum": "Страта модели",
    "state_label": "Состояние скважины",
    "scope_label": "Охват ESP-моделью",
    "confidence": "Доверие",
    "age_source": "Источник возраста",
    "age_mean": "Средний возраст, оп.сут",
    "cov_source": "Источник stress-ковариат",
    "current_status": "ТР: состояние по фонду",
    "current_in_operation": "ТР: в работе",
    "current_status_date": "ТР: дата статуса",
    "event_90d_flag": "ГТМ/КРС в 90 дней",
    "first_gtm_date": "Первая дата ГТМ/КРС",
    "planned_oil_t": "План нефти, т",
    "planned_liquid_m3": "План жидкости, м3",
    "planned_op_days": "План работы, сут",
    "calendar_days": "Календарных дней",
    "expected_failures": "Ожид. отказов УЭЦН",
    "expected_lost_op_days": "Ожид. потеря раб. дней",
    "oil_loss_t": "Потери нефти, т",
    "liquid_loss_m3": "Потери жидкости, м3",
    "p_fail_90d": "P(отказ 90д)",
    "exp_value_at_risk_90d_t": "Добыча под риском 90д, т",
    "theta": "Stress-множитель θ",
    "downtime_days": "Ремонт, дней",
    "flag_changeout": "Флаг замены",
    "covered_planned_oil_t": "Охваченный план нефти, т",
    "covered_planned_liquid_m3": "Охваченный план жидкости, м3",
    "total_planned_oil_t": "Общий план нефти, т",
    "total_planned_liquid_m3": "Общий план жидкости, м3",
    "uncovered_planned_oil_t": "Неохваченный план нефти, т",
    "uncovered_planned_liquid_m3": "Неохваченный план жидкости, м3",
    "risk_adjusted_total_oil_t": "Нефть с учетом риска, т",
    "risk_adjusted_total_liquid_m3": "Жидкость с учетом риска, м3",
    "coverage_pct_oil": "Охват модели, %",
    "risk_pct_total_oil": "Доля риска по нефти, %",
    "planned_gtm_jobs": "План. ГТМ/КРС, шт",
    "reactive_esp_failures": "Ожид. аварийные КРС, шт",
    "total_workover_demand": "Итого спрос КРС, шт",
    "contractor_group": "Группа подрядчика",
    "sour_class": "Кислый/некислый",
    "contractor_source": "Источник подрядчика",
    "sour_source": "Источник H2S-класса",
    "runtime_anomaly_count": "Аномалии наработки",
    "registry_oil_positive_months": "Реестр: нефть, мес",
    "registry_liquid_positive_months": "Реестр: жидкость, мес",
    "summary_positive_months": "Сводные данные: мес",
    f"p_fail_90d_{C.PRIMARY_SCENARIO_ID}": "P(отказ 90д), база/П50",
    f"exp_value_at_risk_90d_t_{C.PRIMARY_SCENARIO_ID}": "Добыча под риском 90д, база/П50, т",
    f"flag_changeout_{C.PRIMARY_SCENARIO_ID}": "Флаг замены, база/П50",
    f"theta_{C.PRIMARY_SCENARIO_ID}": "θ, база/П50",
    f"p_fail_90d_{C.STRESS_SCENARIO_ID}": "P(отказ 90д), stress/П75",
    f"exp_value_at_risk_90d_t_{C.STRESS_SCENARIO_ID}": "Добыча под риском 90д, stress/П75, т",
    f"flag_changeout_{C.STRESS_SCENARIO_ID}": "Флаг замены, stress/П75",
    f"theta_{C.STRESS_SCENARIO_ID}": "θ, stress/П75",
}


def _localize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_COLS_RU.get(col, col) for col in out.columns]
    return out


def _write_table(ws, df: pd.DataFrame, title: str, note: str = "") -> None:
    ws["A1"] = title
    ws["A1"].font = _TITLE_FONT
    if note:
        ws["A2"] = note
        ws["A2"].font = _NOTE_FONT
    df = _localize(df)
    start_row = 4
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=start_row, column=col_idx, value=col_name)
        cell.fill = _HDR_FILL
        cell.font = _HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        for col_idx, value in enumerate(row, start=1):
            if isinstance(value, float):
                value = round(value, 3)
            ws.cell(row=row_idx, column=col_idx, value=value)
    for col_idx, col_name in enumerate(df.columns, start=1):
        width = max(12, min(36, len(str(col_name)) + 4))
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = ws["A5"]


def _summary_sheet(ws, monthly: pd.DataFrame, audit: pd.DataFrame) -> None:
    ws["A1"] = "Интеграция прогноза отказов УЭЦН в план добычи 2026-2027"
    ws["A1"].font = _TITLE_FONT
    ws["A3"] = "Ключевые итоги"
    ws["A3"].font = Font(bold=True)

    primary = monthly[monthly["scenario"] == C.PRIMARY_SCENARIO_ID].copy()
    if primary.empty:
        primary = monthly.copy()
    annual = (
        primary.assign(year=primary["month"].str[:4])
        .groupby("year", as_index=False)
        .agg(
            total_planned_oil_t=("total_planned_oil_t", "sum"),
            uncovered_planned_oil_t=("uncovered_planned_oil_t", "sum"),
            oil_loss_t=("oil_loss_t", "sum"),
            expected_failures=("expected_failures", "sum"),
        )
    )
    header_row = 5
    headers = ["Год", "План нефти, т", "Неохвачено ESP-моделью, т", "Ожид. потери нефти, т", "Ожид. отказы УЭЦН"]
    for idx, value in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=idx, value=value)
        cell.fill = _HDR_FILL
        cell.font = _HDR_FONT
    for row_idx, (_, row) in enumerate(annual.iterrows(), start=header_row + 1):
        ws.cell(row=row_idx, column=1, value=row["year"])
        ws.cell(row=row_idx, column=2, value=round(float(row["total_planned_oil_t"]), 1))
        ws.cell(row=row_idx, column=3, value=round(float(row["uncovered_planned_oil_t"]), 1))
        ws.cell(row=row_idx, column=4, value=round(float(row["oil_loss_t"]), 1))
        ws.cell(row=row_idx, column=5, value=round(float(row["expected_failures"]), 2))

    included = int((audit["scope_label"] != "excluded_non_esp").sum()) if not audit.empty else 0
    excluded = int((audit["scope_label"] == "excluded_non_esp").sum()) if not audit.empty else 0
    ws["A10"] = f"Скважин в расчете: {included}"
    ws["A11"] = f"Скважин вне консервативного ESP-охвата: {excluded}"
    ws["A13"] = "Замечание"
    ws["A13"].font = Font(bold=True)
    ws["A14"] = (
        "Основной сценарий использует только валидированную стратифицированную baseline-модель. "
        "Stress-сценарий помечен как чувствительность и не является OOS-validated прогнозом."
    )
    ws["A14"].alignment = Alignment(wrap_text=True)
    ws.column_dimensions["A"].width = 32
    for col in ("B", "C", "D", "E"):
        ws.column_dimensions[col].width = 22


def _method_sheet(ws) -> None:
    ws["A1"] = "Методика"
    ws["A1"].font = _TITLE_FONT
    lines = [
        "1. База плана: только master-файл ПП; ДФ_04 используется как контекст ГТМ/КРС и ЭЦН-признак.",
        "2. Горизонт прогноза: июль 2026 — декабрь 2027.",
        "3. Основной риск: валидированная модель stratum + operating age из esp_models.csv.",
        "4. Stress-сценарий: direct hazard overlay из esp_cox_coeffs.csv, только как sensitivity.",
        "5. Часы модели: operating days (ttf_mix), а не календарные дни.",
        "6. Потери добычи считаются через ожидаемые потерянные operating days и плановые месячные объемы.",
        "7. Для скважин без подтвержденного ЭЦН-следа основной расчет консервативно не применяется.",
    ]
    for idx, line in enumerate(lines, start=3):
        ws.cell(row=idx, column=1, value=line)
    ws.column_dimensions["A"].width = 110


def _save_workbook_unlocked(wb: Workbook, path: Path) -> Path:
    try:
        wb.save(path)
        return path
    except PermissionError:
        stamped = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        wb.save(stamped)
        return stamped


def write(
    out_path: Path,
    production_tables: dict[str, pd.DataFrame],
    workover: pd.DataFrame,
    changeout: pd.DataFrame,
    audit: pd.DataFrame,
) -> Path:
    wb = Workbook()
    _summary_sheet(wb.active, production_tables["monthly"], audit)
    wb.active.title = "Итог"

    _write_table(
        wb.create_sheet("Помесячно"),
        production_tables["monthly"],
        "Помесячная оценка потерь добычи",
        "Total plan включает весь план, covered plan — только консервативно охваченные ESP-скважины.",
    )
    _write_table(
        wb.create_sheet("По_месторождениям"),
        production_tables["by_field"],
        "Помесячная оценка по месторождениям",
    )
    primary_well = production_tables["by_well"].copy()
    if "scenario" in primary_well.columns:
        primary_well = primary_well[primary_well["scenario"] == C.PRIMARY_SCENARIO_ID]
    _write_table(
        wb.create_sheet("Рейтинг_скважин"),
        primary_well.head(500),
        "Рейтинг скважин по ожидаемым потерям",
        "Показан основной сценарий база/П50. Для stress-сравнения см. лист changeout и CSV.",
    )
    _write_table(
        wb.create_sheet("ГТМ_КРС"),
        workover,
        "План ГТМ/КРС и ожидаемая аварийная нагрузка",
    )
    _write_table(
        wb.create_sheet("Аудит"),
        audit,
        "Аудит покрытия и мэппинга",
    )
    _write_table(
        wb.create_sheet("Changeout"),
        changeout.head(500),
        "Приоритет упреждающей замены",
    )
    _method_sheet(wb.create_sheet("Методика"))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _save_workbook_unlocked(wb, out_path)
