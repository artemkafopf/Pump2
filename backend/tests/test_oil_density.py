"""Плотность нефти и перевод между газовыми осями.

⚠⚠ Плотность берётся ИЗ СПРАВОЧНИКА, а не из отношения источников. Отношение
``ГЖФ/(ГФ×(1−ХВ))`` на данных действительно выходит константой по группе до третьего
знака, но выводить из него ρ значит проверять величину ею самой: согласие осей потом
гарантировано по построению и ничего не подтверждает. Здесь наоборот — справочник
даёт значение, отношение служит КОНТРОЛЕМ стыковки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.data.oil_density import (
    DENSITY_RANGE,
    DensityLookup,
    attach_density,
    gas_liquid_ratio_from_gor,
    gor_from_gas_liquid_ratio,
    load_density_lookup,
    validate_against_observed,
)


@pytest.fixture(scope="module")
def lookup() -> DensityLookup:
    return load_density_lookup()


# --------------------------------------------------------------------------- #
# Справочник
# --------------------------------------------------------------------------- #

def test_the_pair_is_the_key_not_the_field(lookup):
    """⚠ У Ya и Bt плотность РАЗНАЯ по участкам — ключом обязана быть пара."""
    assert lookup.get("Ya", "Ki") == 0.829
    assert lookup.get("Ya", "Au") == 0.833
    assert lookup.get("Bt", "Vt") == 0.811
    assert lookup.get("Bt", "Bt") == 0.821


def test_lookup_is_case_insensitive(lookup):
    """⚠⚠ Телеметрия даёт ``Ya_403``, техрежим ``ya_403``.

    С точным сравнением плотность молча не находилась у 70 530 строк одного лишь
    Ярактинского — при том что пара в справочнике есть.
    """
    assert lookup.get("ya", "ya") == lookup.get("Ya", "Ya") == 0.829
    assert lookup.get("BT", "vt") == 0.811


def test_a_missing_pair_returns_none_not_a_neighbour(lookup):
    """Пропуск остаётся пропуском: чужая плотность масштабирует ось молча."""
    assert lookup.get("Нечто", "Nowhere") is None
    assert lookup.get(None, "Ya") is None


def test_the_cascade_falls_back_to_the_well_prefix(lookup):
    """Справочник смешивает длинные имена и короткие коды, поэтому каскад нужен."""
    assert lookup.get("Ярактинское НГКМ", "Ya", "Ya") == 0.829


def test_every_reference_density_is_physically_possible(lookup):
    low, high = DENSITY_RANGE
    assert all(low <= value <= high for value in lookup.by_pair.values())


# --------------------------------------------------------------------------- #
# Перевод между осями
# --------------------------------------------------------------------------- #

def test_the_conversion_is_exact_both_ways():
    gor = pd.Series([250.0, 100.0])
    density = pd.Series([0.829, 0.812])
    watercut = pd.Series([60.0, 0.0])
    glf = gas_liquid_ratio_from_gor(gor, density, watercut)
    assert np.allclose(glf, [250.0 * 0.829 * 0.4, 100.0 * 0.812])
    assert np.allclose(gor_from_gas_liquid_ratio(glf, density, watercut), gor)


def test_full_water_is_undefined_not_zero():
    """⚠ При ХВ = 100 % нефти нет, и ГЖФ не определён. Ноль означал бы «газа нет»."""
    result = gas_liquid_ratio_from_gor(pd.Series([250.0]), pd.Series([0.829]), pd.Series([100.0]))
    assert result.isna().all()


def test_conversion_needs_density_and_yields_nothing_without_it():
    result = gas_liquid_ratio_from_gor(pd.Series([250.0]), pd.Series([np.nan]), pd.Series([50.0]))
    assert result.isna().all()


# --------------------------------------------------------------------------- #
# Контроль стыковки
# --------------------------------------------------------------------------- #

def _frame(density, observed_density):
    """Кадр, где ГЖФ построен по observed_density, а справочник даёт density."""
    gor = pd.Series([200.0] * 40)
    watercut = pd.Series([50.0] * 40)
    glf = gor * observed_density * 0.5
    return pd.DataFrame({
        "field": ["Ya"] * 40, "lu": ["Ya"] * 40,
        "gas_factor_m3t": gor, "gas_liquid_ratio_m3m3": glf,
        "watercut_percent": watercut, "oil_density_t_m3": density,
    })


def test_a_matching_reference_reports_ok():
    report = validate_against_observed(_frame(0.829, 0.829))
    assert report.loc[0, "статус"] == "ОК"
    assert report.loc[0, "расхождение"] == 0.0


def test_a_wrong_pair_is_reported_not_smoothed():
    """Расхождение значит, что пара подобрана неверно, — его надо УВИДЕТЬ.

    На реальных данных так вскрылось ``Tk_Zy``: справочник 0.816, наблюдение 0.864.
    """
    report = validate_against_observed(_frame(0.816, 0.864))
    assert report.loc[0, "статус"] == "РАСХОЖДЕНИЕ"
    assert report.loc[0, "расхождение"] > 0.05


def test_a_pair_absent_from_the_reference_is_named():
    report = validate_against_observed(_frame(np.nan, 0.833))
    assert report.loc[0, "статус"] == "нет в справочнике"
    assert report.loc[0, "справочная"] is None


def test_attach_density_leaves_unknown_pairs_empty():
    frame = pd.DataFrame({"field": ["Ya", "Нечто"], "lu": ["Ya", "Nowhere"], "well": ["Ya_403", "Xx_1"]})
    result = attach_density(frame)
    assert result["oil_density_t_m3"].tolist()[0] == 0.829
    assert pd.isna(result["oil_density_t_m3"].tolist()[1])
