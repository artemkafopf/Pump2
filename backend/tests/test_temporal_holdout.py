"""Unit tests for the Phase C C5 temporal-holdout helpers (synthetic, no DB)."""
import math
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.temporal_holdout import (
    temporal_split, n_at_risk, ipcw_brier, cindex_at, _censoring_km,
)


def _frame():
    return pd.DataFrame({
        "install_dt": pd.to_datetime(
            ["2021-06-01", "2023-01-01", "2024-03-01", "2024-09-01", "2022-12-31"]),
        "tte": [100.0, 200.0, 50.0, 400.0, 300.0],
        "event": [1, 0, 1, 1, 0],
        "well_key": ["w1", "w2", "w3", "w4", "w5"],
        "stratum_key": ["s1"] * 5,
    })


class TemporalSplitTests(unittest.TestCase):
    def test_split_by_cutoff_inclusive_on_train(self):
        s = temporal_split(_frame(), "2023-12-31")
        # installs on/before 2023-12-31 → train (3), after → test (2)
        self.assertEqual(len(s.train), 3)
        self.assertEqual(len(s.test), 2)
        self.assertTrue((pd.to_datetime(s.test["install_dt"]) > pd.Timestamp("2023-12-31")).all())

    def test_sensitivity_cutoff_moves_boundary(self):
        s = temporal_split(_frame(), "2022-12-31")
        # 2022-12-31 install is inclusive → train; 2023-01-01 → test
        self.assertEqual(len(s.train), 2)
        self.assertEqual(len(s.test), 3)


class NAtRiskTests(unittest.TestCase):
    def test_counts_followed_or_failed(self):
        df = _frame()
        # horizon 150: informative = T>150 (200,300,400) + (T<=150 & event) (100,50) = 5
        self.assertEqual(n_at_risk(df, 150), 5)
        # horizon 250: T>250 → 300,400 (2); T<=250 & event → 100,50 (2) => 4
        # (200 is censored before 250 → not informative)
        self.assertEqual(n_at_risk(df, 250), 4)


class BrierAndCindexTests(unittest.TestCase):
    def test_ipcw_brier_no_censoring_matches_plain_brier(self):
        # no censoring → G(t)=1 everywhere → IPCW Brier == plain Brier
        test = pd.DataFrame({"tte": [50.0, 150.0, 250.0], "event": [1, 1, 1],
                             "well_key": ["a", "b", "c"], "stratum_key": ["s"] * 3})
        cens = _censoring_km(test)  # all events → censoring KM stays at 1
        surv = np.array([0.2, 0.6, 0.9])
        h = 100.0
        br, n = ipcw_brier(test, surv, h, cens)
        # subj1 failed by 100 → (0-0.2)^2; subj2,3 survive past 100 → (1-0.6)^2,(1-0.9)^2
        expected = np.mean([(0 - 0.2) ** 2, (1 - 0.6) ** 2, (1 - 0.9) ** 2])
        self.assertAlmostEqual(br, expected, places=4)
        self.assertEqual(n, 3)

    def test_cindex_perfect_ordering(self):
        # survival score monotone with event time → perfect concordance
        # (>=10 rows / >=5 events to clear the real-data robustness guard)
        n = 12
        times = np.arange(1, n + 1, dtype=float) * 10
        test = pd.DataFrame({"tte": times, "event": [1] * n,
                             "well_key": [f"w{i}" for i in range(n)],
                             "stratum_key": ["s"] * n})
        surv = times / (times.max() + 1)   # later failure ⇒ higher survival
        self.assertAlmostEqual(cindex_at(test, surv), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
