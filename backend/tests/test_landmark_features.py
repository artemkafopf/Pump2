"""Phase D — D0 leakage & feature tests (synthetic daily frames, no DB).

The two named leakage rules the phase fails on if violated:

* **rule 1 (past-only):** recomputing a landmark's features after truncating all
  data beyond the landmark must be *bit-identical* — :class:`TruncationInvariance`.
* **rule 2 (guard gap):** events in ``(t, t+g]`` are excluded from the target window
  (neither positive nor at-risk) — :class:`GuardGapTargets`.

Plus per-feature correctness on hand-built windows.
"""
import unittest

import numpy as np
import pandas as pd

from analysis.data.landmark_features import (
    operating_day_index, theil_sen_slope, compute_window_features,
    build_landmark_targets, build_landmark_frame, FEATURE_STEMS, WINDOWS,
)


def _daily(row_id=1, n=120, seed=0, gap_days=None, well="w1"):
    """Synthetic daily frame for one run: mostly-operating with optional idle gap."""
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2022-01-01", periods=n, freq="D")
    qliq = np.full(n, 100.0) + rng.normal(0, 2, n)
    freq = np.full(n, 50.0) + rng.normal(0, 0.3, n)
    load = np.full(n, 70.0) + rng.normal(0, 1, n)
    kprod = np.full(n, 0.9) + rng.normal(0, 0.01, n)
    if gap_days is not None:
        qliq[gap_days] = 0.0            # idle days (pump off)
    return pd.DataFrame({
        "row_id": row_id, "well_key": well, "dt": dt, "qliq": qliq,
        "freq": freq, "load": load, "kprod": kprod,
        "rpump_intake": np.full(n, 80.0), "rzab": np.full(n, 90.0),
        "rpl": np.full(n, 150.0), "watercut": np.full(n, 40.0),
        "gas_factor": np.full(n, 30.0),
    })


class OperatingDayIndex(unittest.TestCase):
    def test_idle_days_do_not_advance_index(self):
        q = np.array([10, 0, 10, 10, 0, 10.0])
        idx = operating_day_index(q)
        # cumulative count of qliq>0: 1,1,2,3,3,4
        self.assertEqual(list(idx), [1, 1, 2, 3, 3, 4])

    def test_nan_treated_as_idle(self):
        q = np.array([10, np.nan, 10.0])
        self.assertEqual(list(operating_day_index(q)), [1, 1, 2])


class TheilSen(unittest.TestCase):
    def test_recovers_known_slope(self):
        x = np.arange(10.0)
        y = 3.0 * x + 5.0
        self.assertAlmostEqual(theil_sen_slope(y, x), 3.0, places=6)

    def test_nan_robust(self):
        y = np.array([0.0, np.nan, 2.0, 3.0])
        self.assertAlmostEqual(theil_sen_slope(y), 1.0, places=6)

    def test_too_few_points_is_nan(self):
        self.assertTrue(np.isnan(theil_sen_slope(np.array([1.0]))))


class TruncationInvariance(unittest.TestCase):
    """Leakage rule 1: features at landmark L are past-only → truncating data
    beyond L must not change them (bit-identical)."""

    def test_features_bit_identical_after_truncation(self):
        d = _daily(n=120, seed=3, gap_days=[40, 41, 80])
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        for landmark in (30, 60, 90):
            past = d[d["op_index"] <= landmark]
            truncated = d[d["dt"] <= past["dt"].max()]   # drop everything after L
            for w in WINDOWS:
                f_past = compute_window_features(past, w)
                f_trunc = compute_window_features(truncated, w)
                for stem in FEATURE_STEMS:
                    a, b = f_past[stem], f_trunc[stem]
                    if np.isnan(a) and np.isnan(b):
                        continue
                    self.assertEqual(a, b, msg=f"{stem} w{w} landmark {landmark}")

    def test_future_rows_never_leak_into_window(self):
        # A wild future spike after the landmark must not perturb the landmark feature.
        d = _daily(n=120, seed=1)
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        past = d[d["op_index"] <= 30]
        base = compute_window_features(past, 30)["load_mean"]
        d2 = d.copy()
        d2.loc[d2["op_index"] > 30, "load"] = 500.0    # future garbage
        d2 = d2[d2["op_index"] <= 30]
        self.assertEqual(base, compute_window_features(d2, 30)["load_mean"])


class GuardGapTargets(unittest.TestCase):
    """Leakage rule 2: guard-gap window (t, t+g] is excluded from the target."""

    def test_failure_in_guard_is_not_positive(self):
        # landmark 30, terminal 35 (fail), g=7 → failure at d=5 ≤ g → excluded
        t = build_landmark_targets(30, 35, 1, horizons=(60,), guards=(7,))
        self.assertTrue(np.isnan(t["y_H60_g7"]))
        self.assertEqual(t["eligible_H60_g7"], 0)

    def test_failure_inside_window_is_positive(self):
        # landmark 30, terminal 50 (fail), g=7, H=60 → d=20 in (7, 67] → positive
        t = build_landmark_targets(30, 50, 1, horizons=(60,), guards=(7,))
        self.assertEqual(t["y_H60_g7"], 1.0)
        self.assertEqual(t["eligible_H60_g7"], 1)

    def test_survivor_past_window_is_negative(self):
        # landmark 30, terminal 200 (censored), g=7, H=60 → survived (7,67] → 0
        t = build_landmark_targets(30, 200, 0, horizons=(60,), guards=(7,))
        self.assertEqual(t["y_H60_g7"], 0.0)
        self.assertEqual(t["eligible_H60_g7"], 1)

    def test_censored_inside_window_is_unknown(self):
        # censored at d=40, window (7,67] not fully observed → y NaN but eligible
        t = build_landmark_targets(30, 70, 0, horizons=(60,), guards=(7,))
        self.assertTrue(np.isnan(t["y_H60_g7"]))
        self.assertEqual(t["eligible_H60_g7"], 1)

    def test_guard_sensitivity_moves_boundary(self):
        # terminal at d=5: excluded for g=7, but a positive for g=3 (5 in (3,63])
        t = build_landmark_targets(30, 35, 1, horizons=(60,), guards=(3, 7))
        self.assertEqual(t["y_H60_g3"], 1.0)
        self.assertTrue(np.isnan(t["y_H60_g7"]))

    def test_tte_land_and_event_carried(self):
        t = build_landmark_targets(30, 90, 1, horizons=(60,), guards=(7,))
        self.assertEqual(t["tte_land"], 60.0)
        self.assertEqual(t["event_land"], 1)


class WindowFeatureCorrectness(unittest.TestCase):
    def test_restart_and_idle_counts(self):
        d = _daily(n=60, seed=5, gap_days=[20, 21, 22])
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        f = compute_window_features(d, 60)
        # one contiguous idle spell of 3 days → 1 restart, longest_idle 3
        self.assertEqual(f["restarts"], 1.0)
        self.assertEqual(f["longest_idle"], 3.0)

    def test_freq_step_count(self):
        d = _daily(n=40, seed=0)
        d.loc[10, "freq"] = 60.0   # a >1 Hz jump both into and out of day 10
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        f = compute_window_features(d, 40)
        self.assertGreaterEqual(f["freq_step"], 2.0)
        self.assertGreaterEqual(f["freq_days55"], 1.0)

    def test_kprod_pctdrop_positive_on_decline(self):
        d = _daily(n=60, seed=0)
        d["kprod"] = np.linspace(1.0, 0.5, 60)   # declining productivity
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        f = compute_window_features(d, 14)
        self.assertGreater(f["kprod_pctdrop"], 0.0)
        self.assertLess(f["kprod_slope"], 0.0)

    def test_missing_channel_yields_nan_not_crash(self):
        d = _daily(n=40, seed=0).drop(columns=["rpl"])
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        f = compute_window_features(d, 30)
        self.assertTrue(np.isnan(f["drawdown"]))
        self.assertFalse(np.isnan(f["load_mean"]))

    def test_load_excursion_count(self):
        d = _daily(n=40, seed=0)
        d.loc[5:9, "load"] = 95.0   # 5 excursion days > 90
        d["op_index"] = operating_day_index(d["qliq"].to_numpy())
        f = compute_window_features(d, 40)
        self.assertGreaterEqual(f["load_exc"], 5.0)


class FrameBuilder(unittest.TestCase):
    def _run_df(self, row_id=1, event=1, op_total=120, well="w1"):
        return pd.DataFrame([{
            "row_id": row_id, "well_key": well, "stratum_key": "s1", "field": "Vt",
            "install_dt": pd.Timestamp("2022-01-01"), "run_seq": 1, "log_run_seq": 0.69,
            "is_sour_flagged": 0, "has_telemetry": 1, "event": event,
            "mode_group": "hydraulic", "event_hydraulic": event,
            "event_electro-thermal": 0, "event_protector": 0,
        }])

    def test_landmarks_emitted_at_cadence(self):
        daily = _daily(row_id=1, n=125, seed=0)
        res = build_landmark_frame(self._run_df(), daily, cadence=30,
                                   horizons=(60,), guards=(7,))
        # 125 operating days → landmarks at 30,60,90,120
        self.assertEqual(sorted(res.frame["op_age"].tolist()), [30, 60, 90, 120])
        self.assertIn("kprod_mean_w30", res.frame.columns)
        self.assertIn("kprod_mean_w30_missing", res.frame.columns)

    def test_short_run_below_eligibility_skipped(self):
        daily = _daily(row_id=2, n=20, seed=0, well="w2")   # < 30 op days
        res = build_landmark_frame(self._run_df(row_id=2, well="w2"), daily,
                                   cadence=30, horizons=(60,), guards=(7,))
        self.assertTrue(res.frame.empty)

    def test_reconciliation_flags_discrepancy(self):
        daily = _daily(row_id=1, n=100, seed=0)
        ttf = pd.DataFrame({"row_id": [1], "ttf_true_best_days": [50.0]})  # vs ~100 op
        res = build_landmark_frame(self._run_df(op_total=100), daily, ttf_true=ttf,
                                   cadence=30, horizons=(60,), guards=(7,))
        self.assertTrue(bool(res.reconciliation["flag"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
