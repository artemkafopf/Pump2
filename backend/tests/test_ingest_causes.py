"""Таксономия причин и категория узла — разметка, которая теперь живёт в Своде.

Держит три вещи, каждая из которых уже стоила разбирательства:

* «Механическое повреждение кабеля» — это МОНТАЖ, а не эксплуатация;
* «Необеспечен приток» и «влияние газа» — это ЭКСПЛУАТАЦИЯ по критерию
  заказчика, хотя физически это пласт;
* «узел_категория» без диагноза ПУСТА, а не «Износ РО».
"""
from __future__ import annotations

import pandas as pd

from analysis.ingest.svod.causes import (
    CAUSE_GROUP_COLUMN,
    DISPUTED_COLUMN,
    GROUP_BUILD,
    GROUP_OPERATION,
    GROUP_OTHER,
    GROUP_UNSPECIFIED,
    GROUP_WELL,
    NODE_CATEGORIES,
    NODE_CATEGORY_COLUMN,
    attach_cause_columns,
    classify_cause_group,
    classify_node_category,
    has_node_diagnosis,
    is_disputed_cause,
    unmatched_causes,
)
from analysis.workflows.vt_failure.config import FAILURE_CATEGORIES


def test_node_categories_match_the_modelling_config():
    """Один список категорий на два места — иначе они снова разойдутся."""
    assert list(NODE_CATEGORIES) == list(FAILURE_CATEGORIES)


# --------------------------------------------------------------------------- #
# Группы причин
# --------------------------------------------------------------------------- #

def test_cable_mechanical_damage_is_assembly_not_operation():
    """⚠ 145 строк ПДК, медиана 0 суток, 74 % в первом месяце — это монтаж."""
    assert classify_cause_group("Механическое повреждение кабеля") == GROUP_BUILD
    assert classify_cause_group("Механическое повреждение кабеля и элементов кабельной линии") == GROUP_BUILD


def test_inflow_and_gas_are_operation_by_the_customer_criterion():
    """⚠ Пласт, но повлиять на старте нельзя ⇒ эксплуатация.

    Без этого β газовой группы переворачивалась 1.085 → 0.931.
    """
    assert classify_cause_group("Необеспечен приток") == GROUP_OPERATION
    assert classify_cause_group("Не обеспечен приток") == GROUP_OPERATION
    assert classify_cause_group("Влияние газа") == GROUP_OPERATION


def test_well_and_organisational_causes_group_together():
    for reason in (
        "Брак подготовки скважины",
        "Конструкция скважины",
        "Работа в кривизне",
        "Смещение эксплуатационной колонны",
        "Негерметичность эксплуатационной колонны",
        "Организационные причины ИНК",
        "Организационные причины ТКРС",
        "Отсутствие необходимого оборудования",
        "Необоснованный подъем",
    ):
        assert classify_cause_group(reason) == GROUP_WELL, reason


def test_wear_causes_group_as_operation():
    for reason in (
        "Засорение механическими примесями",
        "Солеотложения (гипс)",
        "Сероводород",
        "Коррозия УЭЦН",
        "Выработка ресурса",
        "Старение изоляции кабеля",
        "Брак эксплуатации УЭЦН",
    ):
        assert classify_cause_group(reason) == GROUP_OPERATION, reason


def test_factory_and_repair_defects_group_as_assembly():
    for reason in (
        "Брак монтажа", "Брак подбора УЭЦН", "Брак комплектации УЭЦН",
        "Брак ПЭД", "Брак кабеля", "Брак гидрозащиты", "Брак ремонта НКТ",
        "Скрытый дефект", "Скрытый деффект", "Конструктивный недостаток оборудования",
    ):
        assert classify_cause_group(reason) == GROUP_BUILD, reason


def test_planned_and_blank_reasons_are_unspecified():
    for reason in ("ГТМ", "ГТМ ППР", "ППР", "Нет", "", None, float("nan"), "Дорасследование"):
        assert classify_cause_group(reason) == GROUP_UNSPECIFIED, reason


def test_free_text_is_read_only_when_the_reason_is_blank():
    """Редкое слово в примечании не должно перебивать записанную причину."""
    # Причина пуста -> текст решает.
    assert classify_cause_group("ГТМ", "Непроход при спуске") == GROUP_WELL
    assert classify_cause_group("", "Заводской брак секции") == GROUP_BUILD
    # Причина есть -> текст игнорируется.
    assert classify_cause_group("Засорение механическими примесями", "заводской брак") == GROUP_OPERATION


def test_unmatched_reasons_are_reported_not_swallowed():
    """Новая формулировка в ПДК должна быть видна, а не раствориться."""
    leftovers = unmatched_causes(["Засорение механическими примесями", "Нечто новое", "Нечто новое"])
    assert leftovers.to_dict() == {"Нечто новое": 2}
    assert classify_cause_group("Нечто новое") == GROUP_OTHER


# --------------------------------------------------------------------------- #
# Спорная зона
# --------------------------------------------------------------------------- #

def test_disputed_flag_marks_the_commercial_boundary():
    """⚠⚠ «брак <узел>» против «брак монтажа» решает, КТО ПЛАТИТ.

    Флаг ставится там, где отнесение спорно, а не на всей группе монтажа:
    расчёт гоняется вилкой по нему, и вилка двигает β на 0.07.
    """
    assert is_disputed_cause("Брак ПЭД") == 1
    assert is_disputed_cause("Брак кабеля") == 1
    assert is_disputed_cause("Скрытый дефект") == 1
    assert is_disputed_cause("Прочие по вине оборудования") == 1
    # Однозначно наши: монтаж, подбор, комплектация.
    assert is_disputed_cause("Брак монтажа") == 0
    assert is_disputed_cause("Брак подбора УЭЦН") == 0
    # Не причина вовсе.
    assert is_disputed_cause("ГТМ") == 0
    assert is_disputed_cause(None) == 0


# --------------------------------------------------------------------------- #
# Категория узла
# --------------------------------------------------------------------------- #

def test_node_category_is_empty_without_a_diagnosis():
    """⚠⚠ Дефолт «Износ РО» приписывал +109 выдуманных износов (278 против 169)."""
    assert classify_node_category("", "", "", "") is None
    assert classify_node_category("Нет", "нет", "", "") is None
    assert classify_node_category(None, None, None, None) is None
    assert not has_node_diagnosis("нет", "-")


def test_node_category_classifies_the_known_nodes():
    assert classify_node_category("Кабельная линия") == "КЛ (R-0)"
    assert classify_node_category("ПЭД", "статор с обмоткой") == "ПЭД (R-0)"
    assert classify_node_category("Гидрозащита") == "Износ/негермет.гидрозащиты"
    assert classify_node_category("ЭЦН", "вал", "слом") == "Слом вала"
    assert classify_node_category("НКТ") == "НКТ"
    assert classify_node_category("ЭЦН", "рабочие органы", "износ") == "Износ РО"
    assert classify_node_category("ЭЦН", "рабочие организмы", "засорение") == "Засорение РО"


def test_node_category_default_is_opt_in_only():
    """Дефолт можно поднять до строки, но только осознанно, параметром."""
    # Узел ЭЦН есть, но ни характер, ни причина ничего не говорят.
    assert classify_node_category("ЭЦН", "секция", "", "") is None
    assert classify_node_category("ЭЦН", "секция", "", "", default="Износ РО") == "Износ РО"


# --------------------------------------------------------------------------- #
# Применение к кадру
# --------------------------------------------------------------------------- #

def test_attach_cause_columns_adds_all_three():
    records = pd.DataFrame(
        [
            {
                "Причина отказа УЭЦН": "Механическое повреждение кабеля",
                "Отказавший узел": "Кабельная линия",
                "Отказавший элемент": "основная длина",
                "Характер неисправности": "механическое разрушение",
                "Примечание": "",
            },
            {
                "Причина отказа УЭЦН": "Брак ПЭД",
                "Отказавший узел": "ПЭД",
                "Отказавший элемент": "статор с обмоткой",
                "Характер неисправности": "замыкание",
                "Примечание": "",
            },
            {
                "Причина отказа УЭЦН": "ГТМ",
                "Отказавший узел": "нет",
                "Отказавший элемент": "нет",
                "Характер неисправности": "",
                "Примечание": "",
            },
        ]
    )
    result = attach_cause_columns(records, verbose=False)

    assert result[CAUSE_GROUP_COLUMN].tolist() == [GROUP_BUILD, GROUP_BUILD, GROUP_UNSPECIFIED]
    assert result[DISPUTED_COLUMN].tolist() == [0, 1, 0]
    assert result[NODE_CATEGORY_COLUMN].tolist() == ["КЛ (R-0)", "ПЭД (R-0)", None]


def test_attach_cause_columns_survives_an_empty_frame():
    result = attach_cause_columns(pd.DataFrame(), verbose=False)
    assert list(result.columns) == [CAUSE_GROUP_COLUMN, DISPUTED_COLUMN, NODE_CATEGORY_COLUMN]


# --------------------------------------------------------------------------- #
# Регистр приезжает размеченным — загрузчик его РАЗМЕТКУ ЧИТАЕТ, а не пересчитывает
# --------------------------------------------------------------------------- #

SVOD_COLUMNS = [
    "Скв.", "Дата монтажа", "Дата остановки",
    "Отказавший узел", "Отказавший элемент",
    "Характер неисправности", "Причина отказа УЭЦН", "Кислый/Некислый",
]

SVOD_ROWS = [
    ("Vt_100", "2024-01-01", "2024-06-01", "Кабельная линия", "основная длина",
     "механическое разрушение", "Механическое повреждение кабеля", "Некислый"),
    ("Vt_200", "2024-02-01", "2024-07-01", "нет", "нет", "", "ГТМ", "Некислый"),
]


def _write_svod(path, *, with_category):
    frame = pd.DataFrame(SVOD_ROWS, columns=SVOD_COLUMNS)
    if with_category:
        frame[NODE_CATEGORY_COLUMN] = ["КЛ (R-0)", None]
    frame.to_excel(path, sheet_name="Свод", index=False)
    return path


def test_load_failure_categories_reads_the_precomputed_column(tmp_path, monkeypatch):
    """Свод, собранный сборщиком, уже несёт «узел_категория» — её и надо брать.

    Пересчёт здесь снова завёл бы второе место, где живут те же правила, — ровно
    тот разрыв, ради устранения которого сборщик переехал в этот репозиторий.
    """
    from analysis.workflows.vt_failure import data as D

    path = _write_svod(tmp_path / "svod_with.xlsx", with_category=True)
    monkeypatch.setenv("PUMP2_SVOD_MAIN_PATH", str(path))
    result = D.load_failure_categories()

    assert len(result) == 2
    by_well = result.set_index("well_key")["failure_category"]
    assert by_well.loc["vt_100"] == "КЛ (R-0)"
    assert pd.isna(by_well.loc["vt_200"])


def test_precomputed_and_on_the_fly_classification_agree(tmp_path, monkeypatch):
    """Регистр без колонки классифицируется на лету — тем же классификатором."""
    from analysis.workflows.vt_failure import data as D

    monkeypatch.setenv("PUMP2_SVOD_MAIN_PATH", str(_write_svod(tmp_path / "with.xlsx", with_category=True)))
    with_column = D.load_failure_categories()
    monkeypatch.setenv("PUMP2_SVOD_MAIN_PATH", str(_write_svod(tmp_path / "without.xlsx", with_category=False)))
    without_column = D.load_failure_categories()

    pd.testing.assert_frame_equal(
        with_column.sort_values("well_key").reset_index(drop=True),
        without_column.sort_values("well_key").reset_index(drop=True),
        check_dtype=False,
    )
