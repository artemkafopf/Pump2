"""Phase D — D4 alarm-metric tests (pure, no DB / lifelines)."""
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.alarm_metrics import (
    precision_recall_at_capacity, lead_time_distribution,
    false_alarms_per_pump_year, dynamic_auc, cluster_bootstrap_delta,
)


class PrecisionRecall(unittest.TestCase):
    def _df(self):
        # 10 landmarks, scores 0.1..1.0; the two highest scores are the positives
        return pd.DataFrame({
            "score": np.linspace(0.1, 1.0, 10),
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "well_key": [f"w{i}" for i in range(10)],
        })

    def test_top10pct_flags_one_catches_top_positive(self):
        pr = precision_recall_at_capacity(self._df(), "score", k_fracs=(0.1,))
        r = pr.iloc[0]
        self.assertEqual(r["n_flagged"], 1)
        self.assertEqual(r["tp"], 1)       # highest score is a positive
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 0.5)  # 1 of 2 positives

    def test_censored_excluded(self):
        d = self._df()
        d.loc[0, "y"] = np.nan     # censored-in-window landmark
        pr = precision_recall_at_capacity(d, "score", k_fracs=(0.5,))
        self.assertEqual(pr.iloc[0]["n_scored"], 9)


class DynamicAUC(unittest.TestCase):
    def test_perfect_separation_auc_1(self):
        d = pd.DataFrame({"score": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
        self.assertEqual(dynamic_auc(d, "score"), 1.0)

    def test_reversed_auc_0(self):
        d = pd.DataFrame({"score": [0.9, 0.8, 0.2, 0.1], "y": [0, 0, 1, 1]})
        self.assertEqual(dynamic_auc(d, "score"), 0.0)

    def test_ties_half(self):
        d = pd.DataFrame({"score": [0.5, 0.5, 0.5, 0.5], "y": [0, 0, 1, 1]})
        self.assertAlmostEqual(dynamic_auc(d, "score"), 0.5, places=3)


class LeadTime(unittest.TestCase):
    def test_lead_from_first_alarm(self):
        # one failing run, two landmarks; flagged at op_age 30, terminal 100 → lead 70
        d = pd.DataFrame({
            "row_id": [1, 1], "op_age": [30, 60], "terminal_op_age": [100, 100],
            "score": [0.99, 0.99], "y": [1, 1], "tte_land": [70, 40],
        })
        res = lead_time_distribution(d, "score", 1.0)
        self.assertEqual(res["n"], 1)
        self.assertEqual(res["median"], 70.0)


class FalseAlarms(unittest.TestCase):
    def test_false_alarm_rate(self):
        # 4 survivor landmarks flagged at top-50% → 2 false alarms
        d = pd.DataFrame({"score": [0.1, 0.2, 0.9, 0.95], "y": [0, 0, 0, 0],
                          "row_id": [1, 2, 3, 4]})
        rate = false_alarms_per_pump_year(d, "score", 0.5, cadence=30)
        # 2 fp over 4*30/365 = 0.3288 pump-years → ~6.08
        self.assertAlmostEqual(rate, 2 / (4 * 30 / 365.0), places=2)


class Bootstrap(unittest.TestCase):
    def test_delta_ci_brackets_point(self):
        rng = np.random.default_rng(0)
        d = pd.DataFrame({"well_key": [f"w{i%20}" for i in range(200)],
                          "a": rng.normal(1, 1, 200)})
        point, lo, hi = cluster_bootstrap_delta(d, lambda f: float(f["a"].mean()),
                                                n_boot=200)
        self.assertTrue(lo <= point <= hi)


if __name__ == "__main__":
    unittest.main()
