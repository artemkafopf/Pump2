"""Горизонт наблюдения: регистр собирается НА ПОСЛЕДНЮЮ ДАТУ ПДК.

⚠⚠ Отказы приходят ТОЛЬКО из ПДК, а живые пуски — из паспорта оборудования,
который обновляется отдельно. Всё правее последней даты ПДК — экспозиция, в
которой событие наблюдать нечем:

* пуск, смонтированный после горизонта, разбавляет выборку запуском, чей исход
  принципиально не может быть записан;
* живой пуск, цензурированный сегодняшним днём, получает бесплатную выживаемость.

Оба смещения бьют в раннюю полосу — туда, где решается вопрос о детской
смертности. На реальной сборке это стоило 18 904 суток ненаблюдаемой экспозиции:
в паспорте ННО нет вовсе, и все 556 живых пусков получали наработку как
календарный возраст на СЕГОДНЯ, тогда как ПДК кончался 34 дня назад.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.ingest.svod.pdk_processor import (
    describe_horizon_completeness,
    pdk_observation_horizon,
)


def _pdk(dates):
    return pd.DataFrame({"failure_date": pd.to_datetime(dates)})


def test_horizon_is_the_last_pdk_stop_date():
    frame = _pdk(["2026-05-01", "2026-07-12", "2026-06-30"])
    assert pdk_observation_horizon(frame) == pd.Timestamp("2026-07-12")


def test_horizon_is_none_without_any_pdk_date():
    assert pdk_observation_horizon(pd.DataFrame()) is None
    assert pdk_observation_horizon(_pdk([None, None])) is None


def _six_full_months(per_month=60):
    dates = []
    for month in range(1, 7):
        dates += [f"2026-0{month}-15"] * per_month   # ~2.0 остановок/сут
    return dates


def test_a_half_month_at_the_normal_rate_is_not_flagged():
    """⚠⚠ Главный случай: месяц НЕПОЛНЫЙ, но темп обычный — тревоги быть не должно.

    Выгрузка почти всегда снимается в середине месяца, поэтому последний месяц
    всегда недобирает ПО СУММЕ. Сравнение сумм зажигало бы предупреждение на
    каждой сборке — а то, что горит всегда, перестают читать. Ровно так и вышло на
    выгрузке 2026-08: июль обвинили (24 остановки против 40), хотя его темп
    2.0/сут против июньских 2.1/сут.
    """
    dates = _six_full_months() + ["2026-07-06"] * 24     # 24 за 12 дней = 2.0/сут
    assert describe_horizon_completeness(_pdk(dates), pd.Timestamp("2026-07-12")) is None


def test_thinning_records_at_the_horizon_are_reported():
    """Настоящая беда: у горизонта записи редеют — отчёты ещё не доехали."""
    dates = _six_full_months() + ["2026-07-03"] * 8      # 8 за 20 дней = 0.4/сут
    warning = describe_horizon_completeness(_pdk(dates), pd.Timestamp("2026-07-20"))
    assert warning is not None
    assert "редеют" in warning and "--as-of" in warning


def test_a_full_final_month_is_not_flagged():
    dates = []
    for month in range(1, 8):
        dates += [f"2026-0{month}-15"] * 60
    assert describe_horizon_completeness(_pdk(dates), pd.Timestamp("2026-07-31")) is None


# --------------------------------------------------------------------------- #
# Полный прогон сборки с горизонтом
# --------------------------------------------------------------------------- #

from openpyxl import Workbook  # noqa: E402

from analysis.ingest.config import SVOD_SHEET_NAME  # noqa: E402
from analysis.ingest.svod.build_svod import build_svod_from_scratch  # noqa: E402

BIG_HEADER_ROW1 = [
    "Скважина", "Цель спуска", "Дата отказа", "Дата монтажа",
    "Дата запуска", "Дата демонтажа", "ННО",
]
BIG_HEADER_ROW2 = [None] * 7


def _write_pdk(path):
    """Отказ на 2026-03-01 задаёт горизонт; ничего позже ПДК не видит."""
    pd.DataFrame([
        {
            "Скв.": "Vt_100", "Тип скважины": "НФ",
            "Дата монтажа": "2026-01-01", "Дата остановки": "2026-03-01",
            "Наработка (сут)": 59, "Причина остановки": "Отказ",
            "Отказавший узел": "ЭЦН",
        },
    ]).to_excel(path, index=False)


def _write_big(path, rows):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Скважинное оборудование"
    for col, value in enumerate(BIG_HEADER_ROW1, start=1):
        worksheet.cell(row=1, column=col, value=value)
    for col, value in enumerate(BIG_HEADER_ROW2, start=1):
        worksheet.cell(row=2, column=col, value=value)
    for row_idx, values in enumerate(rows, start=3):
        for col, value in enumerate(values, start=1):
            worksheet.cell(row=row_idx, column=col, value=value)
    workbook.save(path)


@pytest.fixture
def built(tmp_path):
    pdk = tmp_path / "pdk.xlsx"
    big = tmp_path / "big.xlsx"
    output = tmp_path / "svod.xlsx"
    _write_pdk(pdk)
    _write_big(big, [
        # живой пуск, начатый ДО горизонта -> остаётся, наработка обрезается горизонтом
        ["Vt_200", "Мех. добыча", None, "2026-01-01", "2026-01-02", None, None],
        # ⚠ смонтирован ПОСЛЕ горизонта -> чистое разбавление, выбрасывается
        ["Vt_300", "Мех. добыча", None, "2026-05-01", "2026-05-02", None, None],
    ])
    workflow = build_svod_from_scratch(
        pdk_path=[str(pdk)],
        artificial_lift_path=[str(big)],
        include_techregime=False,
        include_telemetry=False,
        output_workbook=str(output),
    )
    return workflow, pd.read_excel(output, sheet_name=SVOD_SHEET_NAME)


def test_horizon_comes_from_pdk_not_from_today(built):
    workflow, _ = built
    assert workflow.observation_horizon == pd.Timestamp("2026-03-01")


def test_a_run_mounted_after_the_horizon_is_dropped(built):
    """Его исход записать нечем — в выборке он работает чистым разбавителем."""
    _, register = built
    assert "Vt_300" not in set(register["Скв."].dropna())


def test_an_open_run_is_censored_at_the_horizon_not_today(built):
    """⚠⚠ Иначе живой пуск получает бесплатную выживаемость до сегодняшнего дня."""
    _, register = built
    row = register[register["Скв."] == "Vt_200"].iloc[0]
    # запуск 2026-01-02 -> горизонт 2026-03-01 = 58 суток, а не возраст на сегодня
    assert row["Наработка (сут)"] == 58
    assert pd.isna(row["Дата остановки"])
    assert row["Работает"] == 1


def test_explicit_as_of_overrides_the_pdk_horizon(tmp_path):
    """Последний месяц ПДК бывает неполон — горизонт можно задать руками."""
    pdk = tmp_path / "pdk.xlsx"
    big = tmp_path / "big.xlsx"
    output = tmp_path / "svod.xlsx"
    _write_pdk(pdk)
    _write_big(big, [["Vt_200", "Мех. добыча", None, "2026-01-01", "2026-01-02", None, None]])

    workflow = build_svod_from_scratch(
        pdk_path=[str(pdk)],
        artificial_lift_path=[str(big)],
        include_techregime=False,
        include_telemetry=False,
        output_workbook=str(output),
        as_of="2026-02-01",
    )
    assert workflow.observation_horizon == pd.Timestamp("2026-02-01")
    register = pd.read_excel(output, sheet_name=SVOD_SHEET_NAME)
    row = register[register["Скв."] == "Vt_200"].iloc[0]
    assert row["Наработка (сут)"] == 30      # 2026-01-02 -> 2026-02-01


def test_a_run_closing_after_the_horizon_is_kept_as_censored(tmp_path):
    """⚠⚠ Вторая половина правила: не терять пуск, чей отказ ПДК ещё не видел.

    Паспорт обновляется независимо от ПДК и регулярно фиксирует закрытие в окне,
    до которого ПДК не дошёл. Такой пуск не отказ (в ПДК его нет) и — при простой
    проверке «дата закрытия пуста» — не живой. Он исчезал из регистра целиком,
    вместе со всей прожитой жизнью. Правильное поведение — цензурировать его
    горизонтом: жизнь сохраняется, об исходе не утверждается ничего.
    """
    pdk = tmp_path / "pdk.xlsx"
    big = tmp_path / "big.xlsx"
    output = tmp_path / "svod.xlsx"
    _write_pdk(pdk)                       # горизонт = 2026-03-01
    _write_big(big, [
        # отказ 2026-04-10 — ЗА горизонтом: ПДК его не видел
        ["Vt_400", "Мех. добыча", "2026-04-10", "2026-01-01", "2026-01-02", "2026-04-12", None],
    ])

    build_svod_from_scratch(
        pdk_path=[str(pdk)],
        artificial_lift_path=[str(big)],
        include_techregime=False,
        include_telemetry=False,
        output_workbook=str(output),
    )

    register = pd.read_excel(output, sheet_name=SVOD_SHEET_NAME)
    assert "Vt_400" in set(register["Скв."].dropna()), "пуск не должен пропадать из регистра"
    row = register[register["Скв."] == "Vt_400"].iloc[0]
    assert pd.isna(row["Дата остановки"]), "отказ за горизонтом не записывается"
    assert row["Работает"] == 1
    assert row["Наработка (сут)"] == 58     # 2026-01-02 -> горизонт 2026-03-01


def test_a_late_dismantling_on_an_observed_failure_is_kept(tmp_path):
    """⚠ Демонтаж — не событие. Событие — остановка.

    ПДК регулярно пишет подъём на несколько дней позже остановки, и такие даты
    выходят за горизонт просто потому, что горизонт задан ПОСЛЕДНЕЙ остановкой.
    Гасить их нельзя: остановка наблюдаема, отказ подтверждён, а дата подъёма
    записана самим ПДК. На реальной выгрузке безусловное гашение стирало 5 таких
    дат (остановки 2026-05-31…2026-07-12, подъёмы 13–14 июля).
    """
    pdk = tmp_path / "pdk.xlsx"
    big = tmp_path / "big.xlsx"
    output = tmp_path / "svod.xlsx"
    pd.DataFrame([
        # горизонт задаёт ЭТА строка
        {"Скв.": "Vt_100", "Тип скважины": "НФ", "Дата монтажа": "2026-01-01",
         "Дата остановки": "2026-03-01", "Дата демонтажа": "2026-03-05",
         "Наработка (сут)": 59, "Причина остановки": "Отказ", "Отказавший узел": "ЭЦН"},
        # остановка в пределах горизонта, подъём — на два дня позже него
        {"Скв.": "Vt_500", "Тип скважины": "НФ", "Дата монтажа": "2026-01-01",
         "Дата остановки": "2026-02-25", "Дата демонтажа": "2026-03-03",
         "Наработка (сут)": 55, "Причина остановки": "Отказ", "Отказавший узел": "ПЭД"},
    ]).to_excel(pdk, index=False)
    _write_big(big, [["Vt_200", "Мех. добыча", None, "2026-01-01", "2026-01-02", None, None]])

    build_svod_from_scratch(
        pdk_path=[str(pdk)],
        artificial_lift_path=[str(big)],
        include_techregime=False,
        include_telemetry=False,
        output_workbook=str(output),
    )

    register = pd.read_excel(output, sheet_name=SVOD_SHEET_NAME)
    row = register[register["Скв."] == "Vt_500"].iloc[0]
    assert pd.Timestamp(row["Дата остановки"]) == pd.Timestamp("2026-02-25")
    assert pd.Timestamp(row["Дата демонтажа"]) == pd.Timestamp("2026-03-03"), (
        "поздний демонтаж на наблюдаемом отказе стирать нельзя"
    )
    assert row["Работает"] == 0
