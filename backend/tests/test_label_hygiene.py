"""Tests for label-hygiene helpers (Phase A T5)."""
import unittest

import pandas as pd

from analysis.data.label_hygiene import (
    trim_failed_node,
    normalize_field,
    field_alias_report,
    count_nonvt_sour,
)


class TrimFailedNodeTests(unittest.TestCase):
    def test_trailing_space_merges(self):
        self.assertEqual(trim_failed_node("НКТ "), trim_failed_node("НКТ"))
        self.assertEqual(trim_failed_node("НКТ "), "НКТ")

    def test_no_failure_sentinel_casefold(self):
        for v in ("нет", "Нет", "Нет ", " нет "):
            self.assertEqual(trim_failed_node(v), "нет")

    def test_none_and_blank(self):
        self.assertIsNone(trim_failed_node(None))
        self.assertIsNone(trim_failed_node("   "))
        self.assertIsNone(trim_failed_node(float("nan")))

    def test_real_label_case_preserved(self):
        self.assertEqual(trim_failed_node("ЭЦН"), "ЭЦН")


class NormalizeFieldTests(unittest.TestCase):
    def test_known_alias_maps(self):
        self.assertEqual(normalize_field("Ичёдинское нефтяное месторождение"), "Ic")
        self.assertEqual(normalize_field("АЗЛУ"), "Az")

    def test_canonical_codes_unchanged(self):
        for c in ("Ya", "Vt", "Az", "Mc", "Da"):
            self.assertEqual(normalize_field(c), c)

    def test_unmapped_returns_none(self):
        self.assertIsNone(normalize_field("ЯНГКМ"))
        self.assertIsNone(normalize_field("Гораздинское"))

    def test_report_splits_resolved_and_needs_review(self):
        s = pd.Series(["Ya", "Ya", "ЯНГКМ", "Ичёдинское нефтяное месторождение"])
        applied, needs = field_alias_report(s)
        self.assertEqual(len(applied), 3)
        self.assertEqual(len(needs), 1)
        self.assertEqual(needs.iloc[0]["field_alias"], "ЯНГКМ")


class NonVtSourTests(unittest.TestCase):
    def test_flip_count_excludes_vt(self):
        df = pd.DataFrame({
            "field": ["Vt", "Vt", "Za", "Za", "Ya"],
            "h2s_proxy_mg_l": [50, 5, 20, 5, 100],
        })
        out = count_nonvt_sour(df, threshold=10.0)
        za = out[out["field"] == "Za"].iloc[0]
        self.assertEqual(za["n_sour_if_threshold_applied"], 1)
        self.assertEqual(za["would_flip_to_sour"], 1)
        vt = out[out["field"] == "Vt"].iloc[0]
        self.assertEqual(vt["would_flip_to_sour"], 0)  # Vt already stratified


if __name__ == "__main__":
    unittest.main()
