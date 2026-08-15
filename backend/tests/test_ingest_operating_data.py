"""Канонизация операционных источников: имена колонок и разбор значений.

Два инварианта, каждый из которых уже ронял данные молча:

* ничья при разрешении имени колонки должна быть ГРОМКОЙ. Выгрузка телеметрии от
  2026-08-15 добавила блок ОЗНА, «Дебит жидкости (ОЗНА)» стал вторым кандидатом
  на «дебитжидкости», резолвер требует единственного совпадения — и ``Qliq_m3d``
  вышел пустым во всех 1 414 192 строках при 1 168 492 в предыдущей сборке;
* векторный разбор чисел и дат обязан совпадать с поэлементным до ячейки.
  В частности голое число в колонке даты — это Excel-серийник с проверкой
  диапазона, а не наносекунды от эпохи.
"""
from __future__ import annotations

import math

import pandas as pd

from analysis.ingest.normalize import parse_date, parse_number
from analysis.ingest.operating_data import (
    CANONICAL_ALIASES,
    _coerce_dates,
    _coerce_numeric,
    ambiguous_canonical_columns,
    resolve_canonical_columns,
)

#: Заголовки выгрузки телеметрии от 2026-08-15 (сокращённо, с блоком ОЗНА).
TELEMETRY_2026_08_HEADERS = [
    "Месторождение", "ЛУ", "КП", "Скважина", "Дата",
    "Дебит нефти (шах), т/сут", "Дебит жидкости (шах), м3/сут",
    "Обводненность, %", "Газожидкостный фактор, м3/м3",
    "Загрузка, %", "Р на приеме насоса, атм", "Частота вращения двиг., Гц",
    "Расчетное забойное давление, атм",
    "Дебит жидкости (ОЗНА), м3/сут", "Дебит нефти (ОЗНА), т/сут",
    "Дебит воды (ОЗНА), м3/сут", "Расход газа (ОЗНА), м3/сут",
]


def _frame(headers):
    return pd.DataFrame({header: [1.0] for header in headers})


def test_liquid_rate_resolves_to_the_shah_column_not_the_ozna_device():
    """⚠⚠ Точный алиас снимает ничью в пользу товарной шахматки."""
    resolved = resolve_canonical_columns(_frame(TELEMETRY_2026_08_HEADERS))
    assert resolved["Qliq_m3d"] == "Дебит жидкости (шах), м3/сут"


def test_the_2026_08_telemetry_layout_has_no_silent_ties():
    assert ambiguous_canonical_columns(_frame(TELEMETRY_2026_08_HEADERS)) == {}


def test_an_ambiguous_layout_is_reported_rather_than_dropped():
    """Ничья без точного алиаса должна быть видна, а не превратиться в пустоту."""
    headers = ["Скважина", "Дата", "Дебит газа (шах), м3/сут", "Дебит газа (ОЗНА), м3/сут"]
    frame = _frame(headers)
    resolved = resolve_canonical_columns(frame)
    ties = ambiguous_canonical_columns(frame)

    assert "Qgas_m3d" not in resolved          # молча пусто — прежнее поведение
    assert ties["Qgas_m3d"] == sorted(headers[2:])  # но теперь названо вслух


def test_reservoir_pressure_is_not_bubble_point():
    """⚠ Рпл и Рнас — разные величины; в db_builder «рпл» стоял в алиасах Pbubble."""
    resolved = resolve_canonical_columns(_frame(["Скважина", "Дата", "Рпл.", "Давление насыщения"]))
    assert resolved["P_reservoir_atm"] == "Рпл."
    assert resolved["Pbubble_atm"] == "Давление насыщения"
    assert "рпл" not in [alias.lower() for alias in CANONICAL_ALIASES["Pbubble_atm"]]


# --------------------------------------------------------------------------- #
# Разбор значений: векторный == поэлементный
# --------------------------------------------------------------------------- #

NUMERIC_CASES = [1, 2.5, "3,5", " 4 000,25 ", "-", "н/д", "нет", "", None, float("nan"), "abc", "0", "  7  "]
DATE_CASES = ["01.04.2020", "2020-04-01", 44000, 5, None, "мусор", pd.Timestamp("2021-02-03")]


def _same_number(vectorized, per_cell) -> bool:
    left_missing = vectorized is None or (isinstance(vectorized, float) and math.isnan(vectorized))
    right_missing = per_cell is None or (isinstance(per_cell, float) and math.isnan(per_cell))
    if left_missing or right_missing:
        return left_missing and right_missing
    return abs(float(vectorized) - float(per_cell)) < 1e-9


def test_vectorized_numeric_parse_matches_per_cell():
    vectorized = _coerce_numeric(pd.Series(NUMERIC_CASES, dtype=object))
    assert vectorized.dtype == "float64"
    assert all(_same_number(got, parse_number(case)) for got, case in zip(vectorized, NUMERIC_CASES))


def test_numeric_columns_pass_through_unchanged():
    assert _coerce_numeric(pd.Series([1, 2, 3])).tolist() == [1.0, 2.0, 3.0]


def test_vectorized_date_parse_matches_per_cell_including_excel_serials():
    """⚠ 44000 — это 2020-06-18 (серийник), а 5 — не дата вовсе."""
    vectorized = _coerce_dates(pd.Series(DATE_CASES, dtype=object))
    for got, case in zip(vectorized, DATE_CASES):
        want = parse_date(case)
        if want is None:
            assert pd.isna(got), case
        else:
            assert got == want, case
    assert vectorized.iloc[2] == pd.Timestamp("2020-06-18")
    assert pd.isna(vectorized.iloc[3])


def test_already_typed_date_column_passes_through():
    series = pd.Series(pd.to_datetime(["2020-01-01", "2021-02-03"]))
    assert _coerce_dates(series).tolist() == series.tolist()
