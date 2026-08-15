"""Конвенция источников: у версии документа дата актуализации стоит в имени.

``WellsArtificialLiftBig 20260815.xlsx`` — и резолвер обязан брать САМУЮ СВЕЖУЮ
такую версию. Иначе датированный файл просто не находится, а сборка молча идёт на
позапрошлой выгрузке и об этом никто не узнает: имя-то прежнее лежит рядом.
"""
from __future__ import annotations

from datetime import date

import pytest

from analysis.paths import parse_version_date, resolve_latest_version


def _touch(directory, name, mtime=None):
    path = directory / name
    path.write_bytes(b"x")
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------------------- #
# Разбор даты из имени
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name, expected", [
    ("WellsArtificialLiftBig 20260815.xlsx", date(2026, 8, 15)),
    ("Свод ГРП 20260815.xlsx", date(2026, 8, 15)),
    ("отчёт_20240101.xlsx", date(2024, 1, 1)),
    ("отчёт-20240101.xlsx", date(2024, 1, 1)),
])
def test_version_date_is_read_from_the_name(name, expected):
    assert parse_version_date(name) == expected


@pytest.mark.parametrize("name", [
    "WellsArtificialLiftBig.xlsx",              # версии нет
    "Свод ПДК 2018-2022гг Архив.xlsx",          # цифры есть, восьмизначной даты нет
    "выгрузка 99999999.xlsx",                   # восемь цифр, но не дата
    "выгрузка 20261345.xlsx",                   # 13-й месяц
])
def test_non_versions_are_not_mistaken_for_dates(name):
    assert parse_version_date(name) is None


# --------------------------------------------------------------------------- #
# Выбор свежей версии
# --------------------------------------------------------------------------- #

def test_the_dated_version_beats_the_undated_one(tmp_path):
    """Главный случай: рядом лежат старое имя и свежая датированная версия."""
    _touch(tmp_path, "WellsArtificialLiftBig.xlsx")
    fresh = _touch(tmp_path, "WellsArtificialLiftBig 20260815.xlsx")
    assert resolve_latest_version(tmp_path, "WellsArtificialLiftBig.xlsx") == fresh


def test_the_newest_date_wins_regardless_of_file_time(tmp_path):
    """Дата в имени решает, а не время файла: перезалитый старый файл не должен побеждать."""
    old = _touch(tmp_path, "big 20260601.xlsx", mtime=2_000_000_000)   # свежее по mtime
    new = _touch(tmp_path, "big 20260815.xlsx", mtime=1_000_000_000)
    assert resolve_latest_version(tmp_path, "big.xlsx") == new
    assert old.exists()


def test_the_undated_file_is_used_when_it_is_the_only_one(tmp_path):
    plain = _touch(tmp_path, "big.xlsx")
    assert resolve_latest_version(tmp_path, "big.xlsx") == plain


def test_a_different_document_is_not_picked_up(tmp_path):
    """Совпадать должно ИМЯ документа, а не просто наличие даты в папке."""
    _touch(tmp_path, "Свод ГРП 20260815.xlsx")
    assert resolve_latest_version(tmp_path, "WellsArtificialLiftBig.xlsx") is None


def test_extension_must_match(tmp_path):
    _touch(tmp_path, "big 20260815.csv")
    assert resolve_latest_version(tmp_path, "big.xlsx") is None


def test_excel_lock_files_are_ignored(tmp_path):
    """``~$`` — временный файл открытой книги, а не версия."""
    real = _touch(tmp_path, "big 20260601.xlsx")
    _touch(tmp_path, "~$big 20260815.xlsx")
    assert resolve_latest_version(tmp_path, "big.xlsx") == real


def test_a_missing_directory_resolves_to_nothing(tmp_path):
    assert resolve_latest_version(tmp_path / "нет-такой-папки", "big.xlsx") is None


def test_the_dated_name_itself_can_be_asked_for(tmp_path):
    """Запрос по датированному имени находит более свежую версию того же документа."""
    _touch(tmp_path, "big 20260601.xlsx")
    newest = _touch(tmp_path, "big 20260815.xlsx")
    assert resolve_latest_version(tmp_path, "big 20260601.xlsx") == newest


# --------------------------------------------------------------------------- #
# Переименование регистра: «Отказы свод с анализом» -> «Свод ЭЦН»
# --------------------------------------------------------------------------- #

def test_the_new_register_name_wins_over_the_legacy_one(tmp_path, monkeypatch):
    """⚠⚠ Иначе регистр под новым именем просто не находится.

    Резолвер искал одно имя. Положи рядом «Свод ЭЦН 20260712.xlsx» — и расчёт
    молча продолжил бы читать прежний «Отказы свод с анализом.xlsx», то есть
    позапрошлый файл, ничем себя не выдав.
    """
    import analysis.paths as P

    monkeypatch.setattr(P, "LOCAL_INPUT_DIR", tmp_path)
    monkeypatch.delenv("PUMP2_SVOD_MAIN_PATH", raising=False)
    _touch(tmp_path, P.SVOD_LEGACY_NAME)
    assert P.resolve_svod_main_path().name == P.SVOD_LEGACY_NAME

    fresh = _touch(tmp_path, "Свод ЭЦН 20260712.xlsx")
    assert P.resolve_svod_main_path() == fresh
    # Тот же файл получает и прогнозный контур — источник у них один.
    assert P.resolve_prediction_workbook_path() == fresh


def test_the_newest_register_version_wins(tmp_path, monkeypatch):
    import analysis.paths as P

    monkeypatch.setattr(P, "LOCAL_INPUT_DIR", tmp_path)
    monkeypatch.delenv("PUMP2_SVOD_MAIN_PATH", raising=False)
    _touch(tmp_path, "Свод ЭЦН 20260612.xlsx")
    newest = _touch(tmp_path, "Свод ЭЦН 20260712.xlsx")
    assert P.resolve_svod_main_path() == newest


def test_an_explicit_env_var_still_overrides_everything(tmp_path, monkeypatch):
    import analysis.paths as P

    monkeypatch.setattr(P, "LOCAL_INPUT_DIR", tmp_path)
    _touch(tmp_path, "Свод ЭЦН 20260712.xlsx")
    chosen = _touch(tmp_path, "иной.xlsx")
    monkeypatch.setenv("PUMP2_SVOD_MAIN_PATH", str(chosen))
    assert P.resolve_svod_main_path() == chosen
