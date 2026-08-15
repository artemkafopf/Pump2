"""Реестр ГРП: выравнивание сдвинутого заголовка, стадии, сводка, признаки по пускам.

Главный инвариант здесь — сдвиг заголовка. Выгрузка «Свод ГРП» несёт лишний
ведущий ``№`` в строке заголовка, которого нет в строках данных, поэтому наивное
``header=0`` подписывает все 280 колонок на одну вправо: `Скважина` получает
значения `Месторождение`, `Дата ГРП` — номера стадий. Молча. Тесты держат
детектор сдвига, а не константу.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.ingest.frac.processor import (
    DUPLICATE_LABEL_SEPARATOR,
    build_frac_stages,
    build_frac_wells,
    deduplicate_labels,
    detect_column_shift,
    detect_layout,
    find_header_row,
    load_frac_register,
)
from analysis.ingest.frac.sqlite_store import build_frac_sqlite, frac_flags_for_runs, load_frac_stages

HEADER = [
    "№", "№ п/п", "Флаг проверки данных", "Недропользователь", "Скважина",
    "Месторождение", "Скважина", "Куст", "Пласт", "Насыщение", "Фонд",
    "Вид ГТМ", "Исполнитель", "Дата ГРП", "№ стадии",
    "Refrac: дата предыдущего ГРП",
]
NUMBERING = [None, *range(len(HEADER) - 1)]

#: Data rows, written one column to the LEFT of the header — the real export's shape.
SHIFTED_ROWS = [
    [1, None, 'ЗАО "ИНК-Запад"', 13180, "Большетирское месторождение", "Bt_2003",
     "КП № 020", "Os", "нефть", "новая", "МГРП", "AGS 2", "04.07.2026", 6, None, None],
    [2, None, 'ЗАО "ИНК-Запад"', 13180, "Большетирское месторождение", "Bt_2003",
     "КП № 020", "Os", "нефть", "новая", "МГРП", "AGS 2", "04.07.2026", 5, None, None],
    [3, None, 'ООО "ИНК"', 19206, "Ярактинское", "Ya_554",
     "КП № 084", "Ya", "нефть", "фонд", "ГРП", "SLB 1", "17.05.2021", 1, None, None],
]


def _write_frac_workbook(path, rows=SHIFTED_ROWS, header=HEADER, numbering=NUMBERING):
    frame = pd.DataFrame([header, numbering, *rows])
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="FracturingSummary", index=False, header=False)
    return path


def test_find_header_row_locates_the_labels_row(tmp_path):
    path = _write_frac_workbook(tmp_path / "frac.xlsx")
    raw = pd.read_excel(path, sheet_name="FracturingSummary", header=None)
    assert find_header_row(raw) == 0


def test_detect_column_shift_finds_the_off_by_one(tmp_path):
    """Сдвиг определяется по якорям, а не задан константой."""
    path = _write_frac_workbook(tmp_path / "frac.xlsx")
    raw = pd.read_excel(path, sheet_name="FracturingSummary", header=None)
    layout = detect_layout(raw)
    assert layout.header_row == 0
    assert layout.data_start_row == 2       # строка нумерации пропущена
    assert layout.column_shift == -1


def test_detect_column_shift_accepts_a_well_formed_sheet(tmp_path):
    """Без лишнего ``№`` сдвиг равен нулю — детектор не «чинит» здоровый лист."""
    aligned_header = HEADER[1:]
    aligned_rows = [row[:-1] for row in SHIFTED_ROWS]
    path = _write_frac_workbook(
        tmp_path / "aligned.xlsx",
        rows=aligned_rows,
        header=aligned_header,
        numbering=list(range(len(aligned_header))),
    )
    raw = pd.read_excel(path, sheet_name="FracturingSummary", header=None)
    assert detect_column_shift(raw, 0, 2) == 0


def test_load_frac_register_puts_values_under_the_right_labels(tmp_path):
    """Проверка того самого, что ломается молча: значение под своей подписью."""
    path = _write_frac_workbook(tmp_path / "frac.xlsx")
    frame = load_frac_register(path)

    assert len(frame) == 3
    # Вторая «Скважина» — код скважины; первая — внутренний числовой ид.
    assert frame[f"Скважина{DUPLICATE_LABEL_SEPARATOR}2"].tolist() == ["Bt_2003", "Bt_2003", "Ya_554"]
    assert frame["Скважина"].tolist() == [13180, 13180, 19206]
    assert frame["Месторождение"].iloc[0] == "Большетирское месторождение"
    assert frame["Дата ГРП"].tolist() == ["04.07.2026", "04.07.2026", "17.05.2021"]
    assert frame["Вид ГТМ"].tolist() == ["МГРП", "МГРП", "ГРП"]


def test_load_frac_register_refuses_a_shift_that_would_drop_data(tmp_path):
    """Изменился формат — процессор говорит вслух, а не подписывает заново."""
    broken_header = ["мусор"] * len(HEADER)
    path = _write_frac_workbook(tmp_path / "broken.xlsx", header=broken_header)
    with pytest.raises(ValueError, match="не найдена строка заголовка"):
        load_frac_register(path)


def test_deduplicate_labels_keeps_the_first_occurrence_addressable():
    assert deduplicate_labels(["a", "b", "a", "a"]) == [
        "a", "b", f"a{DUPLICATE_LABEL_SEPARATOR}2", f"a{DUPLICATE_LABEL_SEPARATOR}3"
    ]


def test_build_frac_stages_types_and_keys(tmp_path):
    frame = load_frac_register(_write_frac_workbook(tmp_path / "frac.xlsx"))
    stages = build_frac_stages(frame)

    assert len(stages) == 3
    assert stages["well_key"].tolist() == ["bt_2003", "bt_2003", "ya_554"]
    # Месторождение здесь — название лицензионного участка; код берётся из
    # префикса скважины, как и везде в проекте.
    assert stages["field"].tolist() == ["Bt", "Bt", "Ya"]
    assert stages["frac_date"].iloc[0] == pd.Timestamp("2026-07-04")
    assert stages["stage_no"].tolist() == [6.0, 5.0, 1.0]


def test_build_frac_wells_counts_treatments_not_stages(tmp_path):
    """МГРП из 15 стадий — ОДНА обработка, а не пятнадцать."""
    frame = load_frac_register(_write_frac_workbook(tmp_path / "frac.xlsx"))
    wells = build_frac_wells(build_frac_stages(frame))

    by_well = wells.set_index("well_key")
    assert by_well.loc["bt_2003", "stages"] == 2
    assert by_well.loc["bt_2003", "treatments"] == 1
    assert by_well.loc["bt_2003", "max_stages_per_treatment"] == 2
    assert by_well.loc["ya_554", "treatments"] == 1
    assert set(wells["has_frac"]) == {1}


def test_frac_flags_only_count_fracs_before_the_mount(tmp_path):
    """⚠ ГРП после спуска насоса о его наработке ничего сказать не может."""
    stages = pd.DataFrame(
        {
            "well_key": ["vt_100", "vt_100", "vt_200"],
            "frac_date": pd.to_datetime(["2023-01-10", "2025-06-01", "2024-03-03"]),
        }
    )
    runs = pd.DataFrame(
        [
            # Монтаж между двумя ГРП: засчитывается только первый.
            {"well": "Vt_100", "installation_date": pd.Timestamp("2024-05-01")},
            # ГРП позже монтажа — «до монтажа» нет, но скважина фраченая.
            {"well": "Vt_200", "installation_date": pd.Timestamp("2023-01-01")},
            # Скважина без ГРП вовсе.
            {"well": "Vt_300", "installation_date": pd.Timestamp("2024-01-01")},
        ]
    )

    flags = frac_flags_for_runs(runs, stages=stages)
    assert flags["ГРП"].tolist() == [1, 1, 0]
    assert flags["ГРП до монтажа"].tolist() == [1, 0, 0]
    assert flags["ГРП стадий"].tolist() == [1, 0, 0]
    assert flags["Дата последнего ГРП"].iloc[0] == pd.Timestamp("2023-01-10")
    assert pd.isna(flags["Дата последнего ГРП"].iloc[1])


def test_frac_flags_join_on_well_and_mount_not_row_position(tmp_path):
    """⚠⚠ Стыковка по (скважина, монтаж). Позиция строки не значит ничего."""
    stages = pd.DataFrame(
        {"well_key": ["vt_100"], "frac_date": pd.to_datetime(["2023-01-10"])}
    )
    runs = pd.DataFrame(
        [
            {"well": "Vt_999", "installation_date": pd.Timestamp("2024-05-01")},
            {"well": "Vt_100", "installation_date": pd.Timestamp("2024-05-01")},
        ],
        index=[7, 3],  # намеренно непозиционный индекс
    )
    flags = frac_flags_for_runs(runs, stages=stages)
    assert flags.loc[7, "ГРП"] == 0
    assert flags.loc[3, "ГРП"] == 1


def test_build_frac_sqlite_round_trip(tmp_path):
    source = _write_frac_workbook(tmp_path / "frac.xlsx")
    store = tmp_path / "frac.sqlite"
    result = build_frac_sqlite(source, sqlite_path=store)

    assert result.stage_rows == 3
    assert result.well_rows == 2
    stages = load_frac_stages(store)
    assert stages["frac_date"].max() == pd.Timestamp("2026-07-04")
    assert set(stages["well_key"]) == {"bt_2003", "ya_554"}
