"""Tests for analysis.data.static_covariates.

The normalizers are pure and always run.  The attach helpers are exercised on synthetic
frames — the traps they guard against (procherk read as zero, frac leaking from the future,
failure history leaking from the future, first-run history read as "clean") are exactly the
ones that cannot be caught by cross-validation.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data import static_covariates as SC


class NormKeyTests(unittest.TestCase):
    def test_case_and_space_folding(self):
        for v in ("Ya_149", " ya_149 ", "YA_149"):
            self.assertEqual(SC.norm_key(v), "ya_149")

    def test_empty_is_none(self):
        for v in (None, float("nan"), "", "   "):
            self.assertIsNone(SC.norm_key(v))


class CurvatureTests(unittest.TestCase):
    def test_procherk_is_a_value_not_a_gap(self):
        """⚠⚠ Прочерк — намеренная запись «не работает в кривизне», а НЕ пропуск."""
        for v in ("-", "—", " - "):
            state, num = SC.parse_curvature(v)
            self.assertEqual(state, "нет")
            self.assertIsNone(num)

    def test_missing_is_its_own_level(self):
        for v in (None, float("nan"), ""):
            state, num = SC.parse_curvature(v)
            self.assertEqual(state, "неизвестно")
            self.assertIsNone(num)

    def test_numbers_split_at_the_declared_threshold(self):
        self.assertEqual(SC.parse_curvature(0.33)[0], "слабая")
        self.assertEqual(SC.parse_curvature("0,90")[0], "сильная")
        self.assertEqual(SC.parse_curvature(SC.CURV_SPLIT)[0], "сильная")
        self.assertAlmostEqual(SC.parse_curvature("1.25")[1], 1.25)

    def test_nonpositive_and_garbage_are_unknown_not_zero(self):
        for v in ("0", "-1", "н/д", "прочее"):
            self.assertEqual(SC.parse_curvature(v)[0], "неизвестно")


class WellboreTests(unittest.TestCase):
    def test_known_levels(self):
        self.assertEqual(SC.parse_wellbore("Горизонтальная"), "гориз")
        self.assertEqual(SC.parse_wellbore("Многостовольная"), "гориз")
        self.assertEqual(SC.parse_wellbore("Наклонно-направленная"), "негориз")
        self.assertEqual(SC.parse_wellbore("Вертикальная"), "негориз")

    def test_missing_is_unknown(self):
        for v in (None, float("nan"), ""):
            self.assertEqual(SC.parse_wellbore(v), "неизвестно")


def _pop(rows):
    return pd.DataFrame(rows)


class FracTests(unittest.TestCase):
    def setUp(self):
        self.stages = pd.DataFrame({
            "well_key": ["ya_1", "ya_1", "ya_2"],
            "frac_date": pd.to_datetime(["2020-01-10", "2020-01-12", "2024-05-01"]),
            "stage_no": [1.0, 2.0, 1.0],
            "gtm_type": ["МГРП", "МГРП", "ГРП"],
        })

    def test_only_fracs_strictly_before_install_count(self):
        pop = _pop([
            {"code": "Ya_1", "install": pd.Timestamp("2019-06-01")},   # до ГРП
            {"code": "Ya_1", "install": pd.Timestamp("2021-06-01")},   # после двух стадий
            {"code": "Ya_2", "install": pd.Timestamp("2023-01-01")},   # ГРП будет позже
            {"code": "Ya_3", "install": pd.Timestamp("2023-01-01")},   # ГРП нет вовсе
        ])
        out = SC.attach_frac(pop, stages=self.stages)
        self.assertEqual(list(out["frac_before"]), [False, True, False, False])
        self.assertEqual(list(out["frac_stages_before"]), [0, 2, 0, 0])

    def test_future_frac_is_flagged_not_merged_into_no_frac(self):
        """«ГРП позже монтажа» — не ковариата; но опорную группу оно засоряет, и это видно."""
        pop = _pop([{"code": "Ya_2", "install": pd.Timestamp("2023-01-01")},
                    {"code": "Ya_3", "install": pd.Timestamp("2023-01-01")}])
        out = SC.attach_frac(pop, stages=self.stages)
        self.assertEqual(list(out["frac_later"]), [True, False])

    def test_frac_on_the_install_day_does_not_count_as_before(self):
        pop = _pop([{"code": "Ya_2", "install": pd.Timestamp("2024-05-01")}])
        out = SC.attach_frac(pop, stages=self.stages)
        self.assertFalse(bool(out["frac_before"].iloc[0]))

    def test_age_since_last_frac(self):
        pop = _pop([{"code": "Ya_1", "install": pd.Timestamp("2020-02-11")}])
        out = SC.attach_frac(pop, stages=self.stages)
        self.assertEqual(int(out["frac_age_days"].iloc[0]), 30)


class FailureHistoryTests(unittest.TestCase):
    def setUp(self):
        # одна скважина, три последовательных пуска; второй отказал засорением
        self.pop = _pop([
            {"code": "Ya_1", "install": pd.Timestamp("2020-01-01"), "t1": 100.0,
             "event": 1.0, "cause": "Износ РО"},
            {"code": "Ya_1", "install": pd.Timestamp("2020-06-01"), "t1": 100.0,
             "event": 1.0, "cause": "Засорение РО"},
            {"code": "Ya_1", "install": pd.Timestamp("2021-01-01"), "t1": 100.0,
             "event": 0.0, "cause": "не определено"},
            {"code": "Ya_9", "install": pd.Timestamp("2021-01-01"), "t1": 50.0,
             "event": 1.0, "cause": "НКТ"},
        ])

    def test_first_run_has_no_history_and_shares_stay_nan(self):
        """⚠ «Истории нет» и «история чистая» — разные состояния, ноль их бы слил."""
        out = SC.attach_failure_history(self.pop)
        self.assertEqual(int(out["hist_n"].iloc[0]), 0)
        self.assertTrue(np.isnan(out["hist_fail_share"].iloc[0]))
        self.assertTrue(np.isnan(out["hist_clog_share"].iloc[0]))
        self.assertEqual(int(out["hist_n"].iloc[3]), 0)

    def test_history_counts_only_runs_that_ended_before_this_install(self):
        out = SC.attach_failure_history(self.pop)
        self.assertEqual(int(out["hist_n"].iloc[1]), 1)
        self.assertEqual(int(out["hist_n"].iloc[2]), 2)
        self.assertAlmostEqual(float(out["hist_fail_share"].iloc[2]), 1.0)

    def test_node_history_is_per_node_not_aggregate(self):
        out = SC.attach_failure_history(self.pop)
        self.assertAlmostEqual(float(out["hist_clog_share"].iloc[1]), 0.0)
        self.assertAlmostEqual(float(out["hist_clog_share"].iloc[2]), 0.5)
        self.assertAlmostEqual(float(out["hist_wear_share"].iloc[2]), 0.5)

    def test_overlapping_run_does_not_enter_history(self):
        """Пуск, который ещё не кончился к монтажу следующего, историей не считается."""
        pop = self.pop.copy()
        pop.loc[0, "t1"] = 10_000.0        # первый пуск «длится» дольше монтажа второго
        out = SC.attach_failure_history(pop)
        self.assertEqual(int(out["hist_n"].iloc[1]), 0)


class PassportTests(unittest.TestCase):
    def setUp(self):
        self.svod = pd.DataFrame({
            "well_key": ["ya_1"],
            "install": [pd.Timestamp("2020-01-01")],
            "curvature_raw": ["0.33"],
            "wellbore_type_raw": ["Горизонтальная"],
            "stages": [300.0], "ped_power_kw": [90.0],
            "head_nom_m": [2000.0], "pump_depth_m": [2500.0],
        })
        self.big = pd.DataFrame({
            "well_key": ["ya_2", "ya_3"],
            "install": [pd.Timestamp("2015-03-05"), pd.Timestamp("2015-03-05")],
            "curvature_raw": [np.nan, 1.4],
            "wellbore_type_raw": [np.nan, np.nan],
            "stages": [250.0, 260.0], "ped_power_kw": [63.0, 70.0],
            "head_nom_m": [1800.0, 1900.0], "pump_depth_m": [2400.0, 2450.0],
        })

    def test_svod_matches_exactly_and_big_fills_the_gap_within_tolerance(self):
        pop = _pop([
            {"code": "Ya_1", "install": pd.Timestamp("2020-01-01")},
            {"code": "Ya_2", "install": pd.Timestamp("2015-03-08")},   # 3 суток — в допуске
            {"code": "Ya_3", "install": pd.Timestamp("2015-06-01")},   # далеко — не берётся
        ])
        out = SC.attach_passport(pop, svod=self.svod, big=self.big)
        self.assertEqual(list(out["passport_source"]), ["svod", "big", "нет"])
        self.assertEqual(out["curv_state"].iloc[0], "слабая")
        self.assertEqual(out["wellbore"].iloc[0], "гориз")
        self.assertEqual(out["curv_state"].iloc[1], "неизвестно")   # у big кривизны нет
        self.assertAlmostEqual(float(out["stages"].iloc[1]), 250.0)
        self.assertTrue(np.isnan(out["stages"].iloc[2]))

    def test_wellbore_from_big_is_unknown_not_horizontal(self):
        """У Big типа ствола НЕТ вовсе — это пропуск, а не «горизонтальная по умолчанию»."""
        pop = _pop([{"code": "Ya_2", "install": pd.Timestamp("2015-03-05")}])
        out = SC.attach_passport(pop, svod=self.svod, big=self.big)
        self.assertEqual(out["wellbore"].iloc[0], "неизвестно")


class CoverageTests(unittest.TestCase):
    def test_coverage_reports_fleet_row(self):
        d = pd.DataFrame({"stratum": ["Ya", "Ya", "Vt_sour"],
                          "x": [1.0, np.nan, 2.0]})
        t = SC.coverage(d, "x")
        self.assertIn("ФОНД", list(t["stratum"]))
        self.assertEqual(int(t.loc[t["stratum"] == "ФОНД", "непустых"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
