"""Tests for pump-type parsing and the nominal-parameter database."""

# Перенесено из db_builder/tests/test_pump_specs.py вместе с кодом сборщика: тесты
# обязаны переезжать с тем, что они держат, иначе перенос молча теряет именно
# те инварианты, ради которых код и писался.

import pandas as pd

from analysis.ingest.svod.pump_specs import (
    NOMINAL_FILL_NOTE,
    append_comment,
    build_pump_nominal_db,
    fill_missing_pump_specs,
    parse_pump_type,
)


def test_parse_pump_type_covers_the_three_conventions():
    russian = parse_pump_type("5а-200-2350")
    assert (russian.gabarit, russian.flow_m3d, russian.head_m) == ("5A", 200.0, 2350.0)

    russian_prefixed = parse_pump_type("10.2ЭЦНДИК5а-80-1600")
    assert (russian_prefixed.gabarit, russian_prefixed.flow_m3d, russian_prefixed.head_m) == ("5A", 80.0, 1600.0)

    mt = parse_pump_type("MT5A-100DP")
    assert mt.gabarit == "5A" and mt.flow_m3d == 100.0

    reda = parse_pump_type("DN460")  # 460 bbl/d -> ~73 m³/d
    assert reda.flow_m3d == round(460 * 0.159, 1)

    assert parse_pump_type("WR").parsed is False
    assert parse_pump_type(None).parsed is False


def test_fill_missing_pump_specs_from_type_code():
    records = pd.DataFrame([
        {"Тип УЭЦН": "5а-200-2350", "Производительность": None, "Габарит": None,
         "Ном. Произв. м₃/сут": None, "Ном.напор (50Гц)": None},
    ])

    filled = fill_missing_pump_specs(records)
    row = filled.iloc[0]
    assert row["Производительность"] == 200.0
    assert row["Ном. Произв. м₃/сут"] == 200.0
    assert row["Ном.напор (50Гц)"] == 2350.0
    assert row["Габарит"] == "5A"


def test_fill_missing_pump_specs_records_provenance_comment():
    records = pd.DataFrame([
        # Blanks -> filled -> note recorded.
        {"Тип УЭЦН": "5а-200-2350", "Производительность": None, "Ном. Произв. м₃/сут": None,
         "Ном.напор (50Гц)": None, "Габарит": None, "Кол.ступеней": None},
        # All nominal fields already present -> nothing filled -> no note.
        {"Тип УЭЦН": "5а-200-2350", "Производительность": 195, "Ном. Произв. м₃/сут": 195,
         "Ном.напор (50Гц)": 2300, "Габарит": "5А", "Кол.ступеней": 250},
    ])
    filled = fill_missing_pump_specs(records, comment_column="Комментарий")
    assert NOMINAL_FILL_NOTE in str(filled.iloc[0]["Комментарий"])
    assert filled.iloc[1]["Комментарий"] in (None, "")


def test_append_comment_deduplicates():
    assert append_comment(None, "A") == "A"
    assert append_comment("A", "B") == "A; B"
    assert append_comment("A; B", "A") == "A; B"


def test_fill_missing_pump_specs_does_not_overwrite_present_values():
    records = pd.DataFrame([
        {"Тип УЭЦН": "5а-200-2350", "Производительность": 195, "Габарит": "5А-custom"},
    ])
    filled = fill_missing_pump_specs(records)
    assert filled.iloc[0]["Производительность"] == 195
    assert filled.iloc[0]["Габарит"] == "5А-custom"


def test_build_pump_nominal_db_uses_fleet_mode_for_missing_codes():
    # An unparseable type ("WR2") gets its nominal flow from fleet frequency.
    fleet = pd.DataFrame([
        {"Тип УЭЦН": "WR2", "Ном. Произв. м₃/сут": 500, "Ном.напор (50Гц)": 2000},
        {"Тип УЭЦН": "WR2", "Ном. Произв. м₃/сут": 500, "Ном.напор (50Гц)": 2000},
        {"Тип УЭЦН": "WR2", "Ном. Произв. м₃/сут": 480, "Ном.напор (50Гц)": 2000},
    ])
    database = build_pump_nominal_db(fleet)
    assert database["WR2"]["flow"] == 500.0  # mode, not mean

    target = pd.DataFrame([{"Тип УЭЦН": "WR2", "Производительность": None}])
    filled = fill_missing_pump_specs(target, database)
    assert filled.iloc[0]["Производительность"] == 500.0
