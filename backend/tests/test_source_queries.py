"""Гигиена запросов к источникам: имена вместо номеров, покрытие до фильтра.

Три инварианта, каждый из которых уже стоил проекту дорого.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.data_utils import (
    SOURCE_SUFFIX,
    TECHREGIME_ABSENT_COLUMNS,
    TECHREGIME_SOURCE_NAMES,
    column_coverage_by_year,
    daily_merge_control,
    resolve_techregime_columns,
)


# --------------------------------------------------------------------------- #
# Позиционные имена колонок техрежима
# --------------------------------------------------------------------------- #

def test_techregime_columns_resolve_through_the_map():
    """⚠⚠ ``col_0067`` МОЛЧА прочитает другую величину, если выгрузка сдвинет столбцы.

    Тот же класс ошибки, что позиционный ``run`` вместо ключа (скважина, монтаж):
    запрос не падает, модель сходится, а данные другие.
    """
    resolved = resolve_techregime_columns()
    assert set(TECHREGIME_SOURCE_NAMES) <= set(resolved)
    assert all(str(value).startswith("col_") for key, value in resolved.items()
               if key not in TECHREGIME_ABSENT_COLUMNS)


def test_columns_absent_from_techregime_are_declared_not_guessed():
    """«Нет в карте» обязано остаться ошибкой, поэтому отсутствующее перечислено явно."""
    resolved = resolve_techregime_columns()
    for alias in TECHREGIME_ABSENT_COLUMNS:
        assert resolved[alias] == "NULL"


def test_a_renamed_source_column_fails_loudly(monkeypatch):
    """Молчаливый NULL хуже остановки: колонка окажется пустой, а слой «не подтвердится»."""
    import scripts.data_utils as U

    resolve_techregime_columns.cache_clear()
    monkeypatch.setitem(U.TECHREGIME_SOURCE_NAMES, "freq", "Такой колонки нет")
    with pytest.raises(KeyError, match="не найдены колонки"):
        resolve_techregime_columns()
    resolve_techregime_columns.cache_clear()


# --------------------------------------------------------------------------- #
# Покрытие ПЕРЕД фильтром
# --------------------------------------------------------------------------- #

def test_coverage_shows_the_column_that_would_empty_the_filter():
    """⚠⚠ «И» по разреженной колонке обнуляет выборку и выглядит содержательно.

    ``WHERE frequency_hz > 0 AND Qliq_m3d > 0`` даёт НОЛЬ строк до 2025 не потому,
    что фонд стоял, а потому что частоты в телеметрии там нет вовсе. На этом был
    построен и попал в отчёт вывод «слои опираются на 2021+».
    """
    frame = pd.DataFrame({
        "dt": pd.to_datetime(["2022-01-01"] * 3 + ["2025-01-01"] * 3),
        "qliq": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "freq": [None, None, None, 50.0, 51.0, None],
    })
    report = column_coverage_by_year(frame, ["qliq", "freq"]).set_index("_год")
    assert report.loc[2022, "qliq"] == 100.0
    assert report.loc[2022, "freq"] == 0.0      # условие «И» обнулило бы 2022 целиком
    assert report.loc[2025, "freq"] == pytest.approx(66.7, abs=0.1)


def test_provenance_columns_are_not_mistaken_for_values():
    frame = pd.DataFrame({
        "dt": pd.to_datetime(["2025-01-01"]),
        "freq": [50.0],
        f"freq{SOURCE_SUFFIX}": ["telemetry"],
    })
    assert f"freq{SOURCE_SUFFIX}" not in column_coverage_by_year(frame).columns


# --------------------------------------------------------------------------- #
# Контроль стыковки
# --------------------------------------------------------------------------- #

def test_merge_control_counts_matched_rows():
    telemetry = pd.DataFrame({"well_id": ["Ya_1", "Ya_2"], "dt": pd.to_datetime(["2025-01-01"] * 2)})
    techregime = pd.DataFrame({"well_id": ["ya_1"], "dt": pd.to_datetime(["2025-01-01"])})
    control = daily_merge_control(telemetry, techregime, None)
    # ⚠ Ключ нормализуется с ОБЕИХ сторон: Ya_1 и ya_1 — одна скважина.
    assert control["matched_rows"] == 1
    assert control["matched_share"] == 0.5


def test_zero_matches_means_a_broken_key_not_missing_data():
    """Ноль совпадений — сигнал о ключе или формате даты, и он обязан быть виден."""
    telemetry = pd.DataFrame({"well_id": ["Ya_1"], "dt": pd.to_datetime(["2025-01-01"])})
    techregime = pd.DataFrame({"well_id": ["Ya_1"], "dt": pd.to_datetime(["2019-01-01"])})
    assert daily_merge_control(telemetry, techregime, None)["matched_rows"] == 0
