"""Integration smoke test for the Phase B competing-risks loader (B0.3).

Skipped automatically when the warehouse DB is absent (clean CI).
"""
import unittest

from analysis.data.competing_risks_loader import (
    COX_MODE_GROUPS,
    WAREHOUSE_DB,
    build_competing_risks_df,
    event_column_for,
)
from analysis.data.failure_modes import MODE_GROUPS


@unittest.skipUnless(WAREHOUSE_DB.exists(), "warehouse pump2.db not available")
class CompetingRisksLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_competing_risks_df(tte_col="ttf_mix")

    def test_one_row_per_run(self):
        self.assertEqual(len(self.df), 2634)
        self.assertEqual(self.df["row_id"].nunique(), 2634)

    def test_event_indicators_partition_failures(self):
        # Each failure belongs to exactly one group; sum of per-group indicators
        # equals the all-cause event count (zero unlabelled failures).
        total_cause = sum(int(self.df[f"event_{g}"].sum()) for g in MODE_GROUPS)
        self.assertEqual(total_cause, int(self.df["event"].sum()))
        self.assertEqual(total_cause, 1527)

    def test_group_counts_match_confirmed_mapping(self):
        counts = {g: int(self.df[f"event_{g}"].sum()) for g in MODE_GROUPS}
        # §0.1 confirmed counts (warehouse post-hygiene 2026-07-06).
        self.assertEqual(counts["hydraulic"], 693)
        self.assertEqual(counts["electro-thermal"], 603)
        self.assertEqual(counts["protector"], 186)
        self.assertEqual(counts["other"], 45)

    def test_no_event_indicator_on_censored(self):
        censored = self.df[self.df["event"] == 0]
        for g in MODE_GROUPS:
            self.assertEqual(int(censored[f"event_{g}"].sum()), 0)

    def test_is_sour_flagged_binary_and_present_outside_vt(self):
        self.assertTrue(set(self.df["is_sour_flagged"].unique()) <= {0, 1})
        # Phase A found flagged-sour runs outside Vt (Az/Ya/Za); assert some exist.
        non_vt_sour = int(
            self.df[(self.df["field"] != "Vt") & (self.df["is_sour_flagged"] == 1)].shape[0]
        )
        self.assertGreater(non_vt_sour, 0)

    def test_event_column_for_helper(self):
        self.assertEqual(event_column_for("hydraulic"), "event_hydraulic")
        with self.assertRaises(KeyError):
            event_column_for("nonsense")

    def test_cox_groups_subset(self):
        for g in COX_MODE_GROUPS:
            self.assertIn(g, MODE_GROUPS)


if __name__ == "__main__":
    unittest.main()
