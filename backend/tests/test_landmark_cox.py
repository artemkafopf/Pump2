"""Phase D — landmark_cox helper tests (split, episode frame, baseline risk)."""
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.landmark_cox import (
    install_cohort_split, build_episode_frame, baseline_window_risk,
)


def _frame():
    # two runs, each with two landmarks (op_age 30, 60)
    rows = []
    specs = [
        # row_id, well, install, event, terminal, stratum
        (1, "w1", "2021-01-01", 1, 80, "s1"),
        (2, "w2", "2024-06-01", 0, 200, "s1"),
    ]
    for rid, well, inst, ev, term, strat in specs:
        for L in (30, 60):
            rows.append({
                "row_id": rid, "well_key": well, "stratum_key": strat,
                "install_dt": pd.Timestamp(inst), "op_age": L,
                "tte_land": term - L, "event_land": ev,
                "event_hydraulic": ev, "f1": 0.5 * L, "f2": 1.0,
            })
    return pd.DataFrame(rows)


class InstallCohortSplit(unittest.TestCase):
    def test_split_inclusive_on_train(self):
        s = install_cohort_split(_frame(), "2023-12-31")
        self.assertTrue((s.train["row_id"] == 1).all())   # 2021 install → train
        self.assertTrue((s.test["row_id"] == 2).all())     # 2024 install → test


class EpisodeFrame(unittest.TestCase):
    def test_intervals_and_terminal_event(self):
        epi = build_episode_frame(_frame(), ["f1", "f2"])
        r1 = epi[epi["row_id"] == 1].sort_values("start")
        # landmarks 30,60 → intervals [30,60),[60,80]; event only on last
        self.assertEqual(list(r1["start"]), [30.0, 60.0])
        self.assertEqual(list(r1["stop"]), [60.0, 80.0])
        self.assertEqual(list(r1["event"]), [0, 1])

    def test_censored_run_has_no_event(self):
        epi = build_episode_frame(_frame(), ["f1"])
        r2 = epi[epi["row_id"] == 2]
        self.assertEqual(int(r2["event"].sum()), 0)

    def test_cause_specific_event_source(self):
        # a run whose all-cause event is 1 but cause event is 0 → no episode event
        f = _frame()
        f["event_electro-thermal"] = 0
        epi = build_episode_frame(f, ["f1"], event_source="event_electro-thermal")
        self.assertEqual(int(epi["event"].sum()), 0)


class BaselineRisk(unittest.TestCase):
    def test_risk_in_unit_interval_and_monotone_in_horizon(self):
        # build a train frame with several failing runs so KM is estimable
        rng = np.random.default_rng(0)
        rows = []
        for i in range(60):
            term = int(rng.integers(40, 300)); ev = int(rng.random() < 0.6)
            for L in (30,):
                rows.append({"row_id": i, "well_key": f"w{i}", "stratum_key": "s1",
                             "install_dt": pd.Timestamp("2021-01-01"), "op_age": L,
                             "tte_land": term - L, "event_land": ev})
        train = pd.DataFrame(rows)
        test = train.head(5).copy()
        r_short = baseline_window_risk(train, test, g=7, H=30)
        r_long = baseline_window_risk(train, test, g=7, H=90)
        self.assertTrue(np.all((r_short >= 0) & (r_short <= 1)))
        # longer horizon ⇒ at least as much cumulative window risk
        self.assertTrue(np.all(r_long + 1e-9 >= r_short))


if __name__ == "__main__":
    unittest.main()
