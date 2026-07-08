"""Tests for the pump_type parser (Phase A T7 — reused by run_covariates)."""
import unittest

import pandas as pd

from analysis.data.pump_type_parser import (
    BBL_TO_M3,
    parse_pump_type,
    parse_pump_type_series,
    parse_report,
)


class RussianEcnTests(unittest.TestCase):
    def test_basic_gabarit_flow_head(self):
        p = parse_pump_type("5а-800-2000")
        self.assertEqual(p.pump_series, "russian_ecn")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.q_design_m3d, 800.0)
        self.assertEqual(p.head_design_m, 2000.0)
        self.assertEqual(p.od_group_mm, 103.0)
        self.assertTrue(p.parsed)

    def test_latin_a_suffix_equivalent_to_cyrillic(self):
        cyr = parse_pump_type("5а-400-2250")
        lat = parse_pump_type("5A-400-2250")
        self.assertEqual(cyr.pump_gabarit, lat.pump_gabarit)
        self.assertEqual(cyr.pump_gabarit, "5A")

    def test_plain_gabarit_no_suffix(self):
        p = parse_pump_type("3-320-2400")
        self.assertEqual(p.pump_gabarit, "3")
        self.assertEqual(p.q_design_m3d, 320.0)
        self.assertEqual(p.head_design_m, 2400.0)

    def test_russian_series_prefix_detected(self):
        p = parse_pump_type("ЭЦНДИК5-200-2400")
        self.assertEqual(p.pump_series, "russian_ecn")
        self.assertEqual(p.pump_series_detail, "ЭЦНДИК")
        self.assertEqual(p.pump_gabarit, "5")
        self.assertEqual(p.q_design_m3d, 200.0)

    def test_vnn_prefix_with_gabarit(self):
        p = parse_pump_type("ВНН5а-500-1500")
        self.assertEqual(p.pump_series_detail, "ВНН")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.head_design_m, 1500.0)

    def test_leading_dimension_prefix(self):
        p = parse_pump_type("10.2ЭЦНДИК5а-80-1600")
        self.assertEqual(p.pump_series, "russian_ecn")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.q_design_m3d, 80.0)
        self.assertEqual(p.head_design_m, 1600.0)


class MtModularTests(unittest.TestCase):
    def test_latin_mt(self):
        p = parse_pump_type("MT5A-100DP")
        self.assertEqual(p.pump_series, "mt")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.q_design_m3d, 100.0)
        self.assertTrue(p.parsed)

    def test_cyrillic_mt(self):
        p = parse_pump_type("МТ5А-200DP")
        self.assertEqual(p.pump_series, "mt")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.q_design_m3d, 200.0)

    def test_mixed_latin_cyrillic_mt(self):
        p = parse_pump_type("MT5А-320DP")
        self.assertEqual(p.pump_gabarit, "5A")
        self.assertEqual(p.q_design_m3d, 320.0)


class RedaTests(unittest.TestCase):
    def test_gn_series_bbl_conversion(self):
        p = parse_pump_type("GN6200")
        self.assertEqual(p.pump_series, "reda")
        self.assertEqual(p.pump_series_detail, "GN")
        self.assertAlmostEqual(p.q_design_m3d, round(6200 * BBL_TO_M3, 1))
        self.assertEqual(p.head_design_m, None)
        self.assertTrue(p.parsed)

    def test_dn_series_od(self):
        # DN fleet is predominantly the 400-series slimline 3.87-in [98.3 mm] housing
        # (verified 2026-07-07 against recorded габарит in WellsArtificialLiftBig;
        # DN3500 itself: 29 joined runs, 86% series-387).
        p = parse_pump_type("DN3500")
        self.assertEqual(p.pump_series_detail, "DN")
        self.assertAlmostEqual(p.od_group_mm, round(3.87 * 25.4, 1))

    def test_single_letter_reda(self):
        p = parse_pump_type("DN460")
        self.assertEqual(p.pump_series, "reda")
        self.assertAlmostEqual(p.q_design_m3d, round(460 * BBL_TO_M3, 1))


class FallbackTests(unittest.TestCase):
    def test_empty_and_none(self):
        for val in (None, "", float("nan")):
            p = parse_pump_type(val)
            self.assertEqual(p.pump_series, "other")
            self.assertFalse(p.parsed)

    def test_garbage_string(self):
        p = parse_pump_type("???unknown###")
        self.assertEqual(p.pump_series, "other")
        self.assertFalse(p.parsed)

    def test_never_raises(self):
        for val in ["5а", "ЭЦН", "MT", "-", "12.3", "ESP-something"]:
            parse_pump_type(val)  # must not raise


class VectorisedTests(unittest.TestCase):
    def test_series_parse_aligns_index(self):
        s = pd.Series(["5а-800-2000", "GN6200", "junk"], index=[10, 20, 30])
        out = parse_pump_type_series(s)
        self.assertEqual(list(out.index), [10, 20, 30])
        self.assertEqual(out.loc[10, "pump_gabarit"], "5A")
        self.assertEqual(out.loc[20, "pump_series"], "reda")
        self.assertFalse(bool(out.loc[30, "parsed"]))

    def test_parse_report_shape(self):
        s = pd.Series(["5а-800-2000", "MT5A-100DP", "GN6200", "junk"])
        rep = parse_report(s)
        self.assertEqual(rep["n_total"], 4)
        self.assertEqual(rep["n_parsed"], 3)
        self.assertAlmostEqual(rep["parse_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
