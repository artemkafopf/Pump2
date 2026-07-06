"""Tests for the Aalen–Johansen CIF helpers (Phase B B2).

The core correctness test cross-checks the manual AJ estimator against lifelines'
``AalenJohansenFitter`` on a synthetic competing-risks dataset, and asserts the
partition identity ``Σ_k CIF_k(t) = 1 − S_all(t)``.
"""
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.cif import (
    aalen_johansen_cif,
    all_cause_km,
    cif_by_cause,
    event_code_series,
    mode_mix_at,
    parametric_competing_cif,
    time_to_incidence,
)


def _synth(n=500, seed=0):
    rng = np.random.default_rng(seed)
    # cause 1 ~ Exp(1/80), cause 2 ~ Exp(1/120); first to occur wins; admin censor 400
    t1 = rng.exponential(80, n)
    t2 = rng.exponential(120, n)
    t = np.minimum(t1, t2)
    code = np.where(t1 <= t2, 1, 2)
    admin = 400.0
    censored = t > admin
    t = np.minimum(t, admin)
    code = np.where(censored, 0, code)
    return t, code


class AalenJohansenTests(unittest.TestCase):
    def test_partition_identity(self):
        t, code = _synth(seed=1)
        c1 = aalen_johansen_cif(t, code, 1)
        c2 = aalen_johansen_cif(t, code, 2)
        km = all_cause_km(t, (code > 0).astype(int))
        for tt in (50, 100, 200, 350):
            lhs = c1.at(tt) + c2.at(tt)
            rhs = km.at(tt)   # 1 − S_all
            self.assertAlmostEqual(lhs, rhs, places=6)

    def test_matches_lifelines(self):
        try:
            from lifelines import AalenJohansenFitter
        except Exception:
            self.skipTest("lifelines not available")
        t, code = _synth(seed=2)
        mine = aalen_johansen_cif(t, code, 1)
        ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
        ajf.fit(t, code, event_of_interest=1)
        for tt in (60, 150, 300):
            ll = float(ajf.predict(tt))
            self.assertAlmostEqual(mine.at(tt), ll, delta=0.02)

    def test_cif_is_monotone_nondecreasing(self):
        t, code = _synth(seed=3)
        c1 = aalen_johansen_cif(t, code, 1)
        self.assertTrue(np.all(np.diff(c1.cif) >= -1e-12))

    def test_cif_below_one_minus_km(self):
        # A single cause's CIF must never exceed all-cause incidence.
        t, code = _synth(seed=4)
        c1 = aalen_johansen_cif(t, code, 1)
        km = all_cause_km(t, (code > 0).astype(int))
        for tt in (50, 200, 400):
            self.assertLessEqual(c1.at(tt), km.at(tt) + 1e-9)


class ModeMixTests(unittest.TestCase):
    def test_event_code_series_censors_on_event_zero(self):
        mg = pd.Series(["hydraulic", "electro-thermal", "protector", "hydraulic"])
        ev = pd.Series([1, 1, 0, 0])
        code = event_code_series(mg, ev, ("hydraulic", "electro-thermal", "protector"))
        np.testing.assert_array_equal(code, np.array([1, 2, 0, 0]))

    def test_mode_mix_sums_to_one(self):
        t, code = _synth(seed=5)
        # relabel as two named groups
        df_code = code  # 0/1/2
        cifs = {
            "a": aalen_johansen_cif(t, df_code, 1),
            "b": aalen_johansen_cif(t, df_code, 2),
        }
        mix = mode_mix_at(cifs, 300)
        self.assertAlmostEqual(sum(mix.values()), 1.0, places=6)
        self.assertTrue(all(v >= 0 for v in mix.values()))


class ParametricCIFTests(unittest.TestCase):
    def test_partition_identity_and_bounded(self):
        # Two causes, one with β<1 (the case that broke the naive S·h trapezoid).
        betas, etas = [0.7, 1.4], [1200.0, 300.0]
        u, s_all, cifs = parametric_competing_cif(betas, etas, max_time=3000.0)
        total = cifs[0] + cifs[1]
        # Σ CIF_k(t) = 1 − S_all(t) everywhere.
        np.testing.assert_allclose(total, 1.0 - s_all, atol=2e-3)
        # Every CIF is a probability in [0, 1] and monotone.
        for cif in cifs:
            self.assertLessEqual(cif.max(), 1.0 + 1e-9)
            self.assertGreaterEqual(cif.min(), -1e-12)
            self.assertTrue(np.all(np.diff(cif) >= -1e-12))

    def test_beta_lt_one_does_not_explode(self):
        # Regression: the old competing_curve_frame gave CIF≈27 here.
        u, s_all, cifs = parametric_competing_cif([0.71], [1244.8], max_time=3000.0)
        self.assertLessEqual(cifs[0].max(), 1.0 + 1e-9)

    def test_time_to_incidence(self):
        u, s_all, cifs = parametric_competing_cif([1.5], [200.0], max_time=2000.0)
        # single cause: CIF = 1−S = 1−exp(−(t/200)^1.5); t@10% solves that.
        t10 = time_to_incidence(u, cifs[0], 0.10)
        expected = 200.0 * (-np.log(0.9)) ** (1 / 1.5)
        self.assertAlmostEqual(t10, expected, delta=5.0)

    def test_time_to_incidence_unreached_is_nan(self):
        u, s_all, cifs = parametric_competing_cif([1.2, 1.2], [300.0, 300.0], max_time=2000.0)
        # with a strong competing risk, one cause may never reach 0.9 incidence
        self.assertTrue(np.isnan(time_to_incidence(u, cifs[0], 0.9)))


if __name__ == "__main__":
    unittest.main()
