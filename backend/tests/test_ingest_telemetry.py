
# Перенесено из db_builder/tests/test_telemetry_utils.py вместе с кодом сборщика: тесты
# обязаны переезжать с тем, что они держат, иначе перенос молча теряет именно
# те инварианты, ради которых код и писался.
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from analysis.ingest.operating_data import build_operating_features
from analysis.ingest.telemetry import (
    build_telemetry_sqlite,
    load_telemetry_sqlite_frame,
    query_telemetry_sqlite,
)


def _write_workbook(path: Path, rows: list[list[object]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    for row in rows:
        worksheet.append(row)
    workbook.save(path)


def test_build_telemetry_sqlite_creates_raw_and_daily_tables(tmp_path: Path):
    telemetry_path = tmp_path / "telemetry_a.xlsx"
    sqlite_path = tmp_path / "telemetry.sqlite"
    _write_workbook(
        telemetry_path,
        [
            ["Скважина", "Дата", "Частота", "Дебит жидк.", "Дав. Нас"],
            ["Ic_101", "2023-02-08", 50, 10, 100],
            ["Ic_101", "2023-02-08", 54, 14, 104],
            ["Ic_101", "2023-02-09", 55, 12, 102],
            ["Ya_202", "2023-02-09", 48, 20, 120],
        ],
    )

    result = build_telemetry_sqlite(source=[telemetry_path], sqlite_path=sqlite_path)

    assert result.total_rows == 4
    assert result.raw_indexed_rows == 4
    assert result.daily_rows == 3
    assert sqlite_path.exists()

    daily = load_telemetry_sqlite_frame(sqlite_path, daily=True)
    assert len(daily) == 3
    ic_day = daily[(daily["well"] == "ic_101") & (daily["date"] == pd.Timestamp("2023-02-08"))]
    assert ic_day.iloc[0]["frequency_hz"] == 52.0
    assert ic_day.iloc[0]["Qliq_m3d"] == 12.0
    assert ic_day.iloc[0]["P_intake_atm"] == 102.0


def test_query_telemetry_sqlite_filters_by_well_and_date(tmp_path: Path):
    telemetry_path = tmp_path / "telemetry_a.xlsx"
    sqlite_path = tmp_path / "telemetry.sqlite"
    _write_workbook(
        telemetry_path,
        [
            ["Скважина", "Дата", "Частота"],
            ["Ic_101", "2023-02-08", 50],
            ["Ic_101", "2023-02-09", 55],
            ["Ya_202", "2023-02-09", 48],
        ],
    )

    build_telemetry_sqlite(source=[telemetry_path], sqlite_path=sqlite_path)
    queried = query_telemetry_sqlite(
        sqlite_path,
        well_value="Ic_101",
        start_date="2023-02-09",
        end_date="2023-02-09",
        daily=True,
    )

    assert len(queried) == 1
    assert queried.iloc[0]["well"] == "ic_101"
    assert queried.iloc[0]["frequency_hz"] == 55.0


def test_operating_pipeline_prefers_telemetry_sqlite_when_present(tmp_path: Path):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    telemetry_xlsx = telemetry_dir / "field_a.xlsx"
    sqlite_path = telemetry_dir / "telemetry.sqlite"
    _write_workbook(
        telemetry_xlsx,
        [
            ["Скважина", "Дата", "Частота", "Дебит жидк.", "Ном. Произв. м3/сут"],
            ["Ic_329", "2023-02-08", 50, 10, 20],
            ["Ic_329", "2023-02-09", 52, 12, 20],
            ["Ic_329", "2023-02-10", 54, 14, 20],
        ],
    )
    build_telemetry_sqlite(source=[telemetry_xlsx], sqlite_path=sqlite_path)

    target_runs = pd.DataFrame(
        {
            "ID скважины": ["Ic_329"],
            "Дата запуска": ["2023-01-01"],
            "Дата отказа": ["2023-02-10"],
            "Наработка (сут)": [40],
        }
    )
    techregime = pd.DataFrame(
        {
            "ID скважины": ["Ic_329"],
            "Дата": ["2023-02-10"],
            "Частота": [45],
        }
    )

    result = build_operating_features(
        target_runs,
        telemetry=[telemetry_dir],
        techregime=techregime,
    )

    features = result.features.iloc[0]
    assert features["frequency_hz_30d_mean"] == 52.0
    assert features["frequency_hz_30d_mean_source"] == "telemetry"
