"""Tests for restart reconciliation (directive #4)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.features.restart_events import reconcile_restarts


def _frame(qliq, treg=None):
    dt = pd.date_range("2022-01-01", periods=len(qliq), freq="D")
    d = {"dt": dt, "qliq": qliq}
    if treg is not None:
        d["treg_in_operation"] = treg
    return pd.DataFrame(d)


class ReconcileRestartTests(unittest.TestCase):
    def test_clean_single_restart_confirmed_by_both(self):
        # down for 3 days then up; techregime agrees
        q = [10, 10, 0, 0, 0, 12, 12]
        t = [True, True, False, False, False, True, True]
        r = reconcile_restarts(_frame(q, t))
        self.assertEqual(r["n_restarts_raw"], 1)
        self.assertEqual(r["n_restarts_reconciled"], 1)
        self.assertEqual(r["n_outages_removed"], 0)
        self.assertEqual(r["n_restarts_both"], 1)

    def test_telemetry_outage_is_not_a_restart(self):
        # qliq drops to 0 (sensor outage) but techregime says "В работе" throughout
        q = [10, 10, 0, 0, 0, 12, 12]
        t = [True, True, True, True, True, True, True]
        r = reconcile_restarts(_frame(q, t))
        self.assertEqual(r["n_restarts_raw"], 1)          # naive telemetry sees a restart
        self.assertEqual(r["n_restarts_reconciled"], 0)   # techregime reveals it was running
        self.assertEqual(r["n_outages_removed"], 1)

    def test_genuine_shutdown_both_agree_down(self):
        q = [10, 0, 0, 8]
        t = [True, False, False, True]
        r = reconcile_restarts(_frame(q, t))
        self.assertEqual(r["n_restarts_reconciled"], 1)
        self.assertEqual(r["n_outages_removed"], 0)

    def test_two_restarts(self):
        q = [5, 0, 6, 0, 0, 7]
        t = [True, False, True, False, False, True]
        r = reconcile_restarts(_frame(q, t))
        self.assertEqual(r["n_restarts_reconciled"], 2)

    def test_no_techregime_falls_back_to_telemetry(self):
        q = [10, 0, 0, 9]
        r = reconcile_restarts(_frame(q))            # no treg column
        self.assertEqual(r["n_restarts_raw"], 1)
        self.assertEqual(r["n_restarts_reconciled"], 1)

    def test_empty(self):
        r = reconcile_restarts(pd.DataFrame(columns=["dt", "qliq"]))
        self.assertEqual(r["n_restarts_reconciled"], 0)
        self.assertFalse(r["restart_reliable"])

    def test_reliability_flag_on_agreement(self):
        n = 40
        q = [0 if i % 10 == 0 else 10 for i in range(n)]
        t = [False if i % 10 == 0 else True for i in range(n)]
        r = reconcile_restarts(_frame(q, t))
        self.assertTrue(r["restart_reliable"])

    def test_reliability_false_on_divergence(self):
        n = 40
        q = [10] * n                       # telemetry: always running
        t = [i % 2 == 0 for i in range(n)]  # techregime: running half the days
        r = reconcile_restarts(_frame(q, t))
        self.assertFalse(r["restart_reliable"])


if __name__ == "__main__":
    unittest.main()
