"""Unit tests for the Block-2 operational feature derivations (Phase C C0).

Exercises the two derivations the spec singles out — **kpod** (Ql/Qnominal) and
**restart counting** — on synthetic daily frames, so the maths is pinned
independently of the warehouse.
"""
import math
import unittest

import numpy as np

from analysis.features.operational import (
    compute_kpod_features,
    compute_cycling_features,
    std_from_moments,
)


class KpodDerivationTests(unittest.TestCase):
    def test_basic_kpod_mean_and_frac(self):
        # nominal 100 m³/d; days at 50, 70, 120 → kpod 0.5, 0.7, 1.2
        r = compute_kpod_features(np.array([50.0, 70.0, 120.0]), 100.0)
        self.assertAlmostEqual(r["kpod_mean"], (0.5 + 0.7 + 1.2) / 3, places=6)
        # frac below 0.7 is strict (< 0.7): only the 0.5 day counts
        self.assertAlmostEqual(r["frac_kpod_below_0p7"], 1 / 3, places=6)
        self.assertEqual(r["n_op_days"], 3)

    def test_freq_scaled_kpod(self):
        # at 25 Hz, affinity BEP halves → kpod_freq doubles vs kpod
        r = compute_kpod_features(np.array([50.0]), 100.0, freq=np.array([25.0]))
        self.assertAlmostEqual(r["kpod_mean"], 0.5, places=6)
        # qliq / (100 * 25/50) = 50 / 50 = 1.0
        self.assertAlmostEqual(r["kpod_freq_mean"], 1.0, places=6)

    def test_invalid_nominal_returns_nan(self):
        r = compute_kpod_features(np.array([50.0]), 0.0)
        self.assertTrue(math.isnan(r["kpod_mean"]))
        self.assertEqual(r["n_op_days"], 0)

    def test_nan_qliq_days_dropped(self):
        r = compute_kpod_features(np.array([50.0, np.nan, 150.0]), 100.0)
        self.assertEqual(r["n_op_days"], 2)
        self.assertAlmostEqual(r["kpod_mean"], (0.5 + 1.5) / 2, places=6)

    def test_freq_zero_excluded_from_freq_kpod(self):
        r = compute_kpod_features(
            np.array([50.0, 60.0]), 100.0, freq=np.array([0.0, 50.0]))
        # only the 50 Hz day contributes: 60 / (100*1) = 0.6
        self.assertAlmostEqual(r["kpod_freq_mean"], 0.6, places=6)


class RestartDerivationTests(unittest.TestCase):
    def test_counts_off_to_on_transitions(self):
        # on, off, on, on, off, on → two 0→>0 transitions
        q = np.array([10.0, 0.0, 10.0, 10.0, 0.0, 10.0])
        r = compute_cycling_features(q, span_days=len(q))
        self.assertEqual(r["n_restarts_raw"], 2)
        self.assertAlmostEqual(r["n_restarts_per_100d"], 100.0 * 2 / 6, places=3)

    def test_continuous_run_no_restarts(self):
        r = compute_cycling_features(np.array([5.0, 5.0, 5.0]), span_days=3)
        self.assertEqual(r["n_restarts_raw"], 0)

    def test_nan_qliq_treated_as_idle(self):
        # NaN is idle → on/off/on = one restart
        q = np.array([10.0, np.nan, 10.0])
        r = compute_cycling_features(q, span_days=3)
        self.assertEqual(r["n_restarts_raw"], 1)

    def test_leading_idle_then_on_counts_as_restart(self):
        # 0 then >0 is a 0→>0 transition (pump first comes on)
        r = compute_cycling_features(np.array([0.0, 10.0, 10.0]), span_days=3)
        self.assertEqual(r["n_restarts_raw"], 1)

    def test_freq_steps_and_std(self):
        # |Δf|: 0.5, 3.0, 0.2 → one step > 1 Hz
        f = np.array([50.0, 50.5, 53.5, 53.7])
        r = compute_cycling_features(np.array([1.0, 1.0, 1.0, 1.0]), f, span_days=4)
        self.assertEqual(r["n_steps_raw"], 1)
        self.assertAlmostEqual(r["freq_std"], float(np.std(f)), places=3)

    def test_freq_nan_dropped_from_std(self):
        f = np.array([50.0, np.nan, 52.0])
        r = compute_cycling_features(np.array([1.0, 1.0, 1.0]), f, span_days=3)
        # only 2 valid freq days (< MIN_STD_DAYS) → std NaN
        self.assertTrue(math.isnan(r["freq_std"]))


class StdFromMomentsTests(unittest.TestCase):
    def test_matches_numpy_population_std(self):
        x = np.array([50.0, 52.0, 48.0, 51.0, 49.0])
        s = std_from_moments(float(x.mean()), float((x * x).mean()), len(x))
        self.assertAlmostEqual(s, float(np.std(x)), places=6)

    def test_negative_variance_clamped(self):
        # floating point can make sq_mean - mean^2 slightly negative
        s = std_from_moments(50.0, 2500.0 - 1e-12, 10)
        self.assertEqual(s, 0.0)

    def test_too_few_days_returns_nan(self):
        self.assertTrue(math.isnan(std_from_moments(50.0, 2501.0, 2)))


if __name__ == "__main__":
    unittest.main()
