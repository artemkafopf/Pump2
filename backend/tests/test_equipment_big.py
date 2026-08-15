"""Tests for analysis.data.equipment_big (normalizers pure; loader file-guarded)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.equipment_big import (
    is_esp_row,
    is_esp_strict,
    is_esp_type_positive,
    is_purpose_mech_production,
    norm_coating,
    norm_corrosion_class,
    norm_exec_group,
    norm_gabarit,
    parse_type_execution_flags,
)
from analysis.paths import resolve_equipment_big_path


class CorrosionClassTests(unittest.TestCase):
    def test_cyrillic_and_latin_k(self):
        self.assertEqual(norm_corrosion_class("К2"), 2.0)
        self.assertEqual(norm_corrosion_class("K0"), 0.0)
        self.assertEqual(norm_corrosion_class(" К3 "), 3.0)

    def test_non_class_values_are_none(self):
        for v in (None, float("nan"), "", "нет", "Кx", "5А"):
            self.assertIsNone(norm_corrosion_class(v))


class CoatingTests(unittest.TestCase):
    def test_monel_any_case(self):
        for v in ("монель", "Монель", "МОНЕЛЬ", "покрытие монель"):
            self.assertEqual(norm_coating(v), "monel")

    def test_yes_no_and_unknown(self):
        self.assertEqual(norm_coating("да"), "yes")
        self.assertEqual(norm_coating("нет"), "no")
        self.assertEqual(norm_coating("без покрытия"), "no")
        self.assertEqual(norm_coating("-"), "no")
        self.assertIsNone(norm_coating(None))
        self.assertIsNone(norm_coating("полимер?"))


class ExecGroupTests(unittest.TestCase):
    def test_cyrillic_latin_twins_collapse(self):
        for v in ("Н2-ЛЧ", "Н2", "N2", "2N"):
            self.assertEqual(norm_exec_group(v)[0], "H2", v)
        for v in ("Н3-ЛЧ", "Н3", "N3", "3N"):
            self.assertEqual(norm_exec_group(v)[0], "H3", v)

    def test_lch_flag(self):
        self.assertTrue(norm_exec_group("Н2-ЛЧ")[1])
        self.assertFalse(norm_exec_group("Н2")[1])

    def test_none(self):
        self.assertEqual(norm_exec_group(None), (None, False))


class TypeExecutionFlagTests(unittest.TestCase):
    def test_novomet_letters(self):
        self.assertEqual(parse_type_execution_flags("30.2 ЭЦНДИК Э"), (True, True))
        self.assertEqual(parse_type_execution_flags("ЭЦНМИКэ"), (True, True))
        self.assertEqual(parse_type_execution_flags("10.1ЭЦНД"), (False, False))
        self.assertEqual(parse_type_execution_flags("ЭЦНКИД"), (True, True))

    def test_non_ecn_names_no_flags(self):
        self.assertEqual(parse_type_execution_flags("GN6200"), (False, False))
        self.assertEqual(parse_type_execution_flags("MT5A-100DP"), (False, False))
        self.assertEqual(parse_type_execution_flags(None), (False, False))


class GabaritTests(unittest.TestCase):
    def test_russian_codes(self):
        self.assertEqual(norm_gabarit("5А"), ("GAB:5А", 103.0))
        self.assertEqual(norm_gabarit("5A"), ("GAB:5А", 103.0))   # Latin A
        self.assertEqual(norm_gabarit("6"), ("GAB:6", 114.0))

    def test_reda_series_and_540_exception(self):
        self.assertEqual(norm_gabarit("538"), ("SER:538", 136.7))
        self.assertEqual(norm_gabarit("540"), ("SER:540", 130.3))  # 5.13 in, not 5.40
        self.assertEqual(norm_gabarit("387"), ("SER:387", 98.3))

    def test_direct_mm(self):
        self.assertEqual(norm_gabarit("101ММ"), ("MM:101", 101.0))

    def test_garbage(self):
        self.assertEqual(norm_gabarit("что-то"), (None, None))
        self.assertEqual(norm_gabarit(None), (None, None))


class IsEspTests(unittest.TestCase):
    def test_non_esp_rows(self):
        for v in ("Воронка", "Воронка НКТ-73", "УГРП на НКТ-89", "Отсутствует", None):
            self.assertFalse(is_esp_row(v), v)

    def test_esp_rows(self):
        for v in ("30.2 ЭЦНДИК Э", "GN6200", "ВНН", "MT5A-100DP"):
            self.assertTrue(is_esp_row(v), v)


class EspStrictTests(unittest.TestCase):
    def test_positive_vocabulary_includes_reda_slb(self):
        for v in ("30.2 ЭЦНДИК Э", "MT5A-100DP", "MT5-125DP", "G6200N", "S8000N",
                  "S11000N", "D460N", "D3500N", "GN10000", "ESP 538-7000", "ESP 400-3000"):
            self.assertTrue(is_esp_type_positive(v), v)

    def test_screw_rod_excluded(self):
        for v in ("ВНН", "ВНН5", "УВНН", "ШГН", "ШВН", "УШГН"):
            self.assertFalse(is_esp_type_positive(v), v)

    def test_non_lift_placeholders_excluded(self):
        for v in ("Воронка НКТ-73", "УГРП на НКТ-89", "Пакер", None):
            self.assertFalse(is_esp_type_positive(v), v)

    def test_purpose_gate(self):
        self.assertTrue(is_purpose_mech_production("Мех. добыча"))
        self.assertTrue(is_purpose_mech_production(" мех. добыча "))
        for v in ("Водозаборная", "Нагнетательная", "Пьезометр", "ГРП", None):
            self.assertFalse(is_purpose_mech_production(v), v)

    def test_strict_needs_both_gates(self):
        # ESP type but wrong purpose (водозаборная) → not a survival unit
        self.assertFalse(is_esp_strict("Водозаборная", "30.2 ЭЦНДИК Э"))
        # мех.добыча but screw pump → excluded
        self.assertFalse(is_esp_strict("Мех. добыча", "ВНН"))
        # both gates satisfied
        self.assertTrue(is_esp_strict("Мех. добыча", "G6200N"))
        self.assertTrue(is_esp_strict("Мех. добыча", "30.2 ЭЦНДИК Э"))


@unittest.skipUnless(resolve_equipment_big_path().exists(),
                     "WellsArtificialLiftBig.xlsx not available")
class LoaderSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from analysis.data.equipment_big import load_equipment_big
        cls.df = load_equipment_big()

    def test_shape_and_keys(self):
        self.assertGreater(len(self.df), 8000)
        self.assertTrue(self.df["well_key"].notna().mean() > 0.99)
        self.assertTrue(self.df["row_id"].is_unique)

    def test_esp_filter_plausible(self):
        share = self.df["is_esp"].mean()
        self.assertGreater(share, 0.6)
        self.assertLess(share, 0.95)

    def test_corrosion_profile_present(self):
        self.assertGreater((self.df["pump_coating"] == "monel").sum(), 100)
        self.assertGreater(self.df["pump_exec_group"].notna().sum(), 4000)
        self.assertGreater(self.df["type_corr_resistant"].fillna(False).sum(), 500)

    def test_strict_filter_columns(self):
        self.assertIn("is_esp_strict", self.df.columns)
        self.assertIn("purpose", self.df.columns)
        self.assertIn("pump_serial", self.df.columns)
        # strict ESP is a subset of the old negative filter and is smaller
        n_strict = int(self.df["is_esp_strict"].sum())
        n_old = int(self.df["is_esp"].sum())
        self.assertGreater(n_strict, 3000)
        self.assertLess(n_strict, n_old)
        # every strict-ESP row is мех.добыча
        mech = self.df["purpose"].map(is_purpose_mech_production)
        self.assertTrue((~mech[self.df["is_esp_strict"]]).sum() == 0)

    def test_vnn_screw_pumps_excluded_from_strict(self):
        vnn = self.df["gno_type"].astype(str).str.upper().str.startswith("ВНН")
        self.assertGreater(int(vnn.sum()), 100)  # ВНН present in workbook
        self.assertEqual(int(self.df.loc[vnn, "is_esp_strict"].sum()), 0)

    def test_pull_fail_gap_median_small(self):
        med = self.df["pull_fail_gap_d"].dropna().median()
        self.assertGreaterEqual(med, 0)
        self.assertLess(med, 15)


if __name__ == "__main__":
    unittest.main()


def test_header_lookup_accepts_both_type_and_model_spellings():
    """⚠⚠ Выгрузка паспорта 2026-08-15 переименовала «Тип X» → «Модель X».

    Пять секций — газосепаратор, мультифазная, гидрозащита, ПЭД, ТМС — сменили
    ярлык при тех же позициях и том же содержимом. Загрузчик знал одно написание и
    падал целиком: `KeyError: header lookup failed`. Падал ГРОМКО, и это хорошо, но
    держать одно написание было нечем.
    """
    from analysis.data.equipment_big import _find_column

    header_map = {
        63: ("Газосепаратор", "Модель Газосепаратора"),
        92: ("ПЭД", "Тип ПЭД"),
    }
    # Оба написания находят свою колонку, в каком бы виде она ни пришла.
    assert _find_column(header_map, "Газосепаратор", ("Тип Газосепаратора", "Модель Газосепаратора")) == 63
    assert _find_column(header_map, "ПЭД", ("Тип ПЭД", "Модель ПЭД")) == 92
    # Отсутствующее написание по-прежнему даёт None, а не случайную колонку.
    assert _find_column(header_map, "Газосепаратор", ("Габарит ГС",)) is None
    # Строка как параметр по-прежнему работает — старый контракт не сломан.
    assert _find_column(header_map, "ПЭД", "Тип ПЭД") == 92
