"""Regression tests for the corrected time-interaction Cox module (Phase B B0.2).

The headline test synthesises a dataset with a **known constant (flat) hazard
ratio** — proportional hazards, so the true log-time interaction γ is exactly 0 —
and asserts the fitted γ ≈ 0.  The buggy grid-split-at-episode-end estimator
returned γ ≈ −0.851 on such data; this test is the guard that would have caught it.
"""
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.time_interaction_cox import (
    episode_split,
    fit_time_interaction_cox,
)


def _synth_proportional_hazards(n=600, hr=2.0, seed=0, admin_censor=400.0):
    """Two groups, exponential lifetimes with a constant hazard ratio ``hr``.

    Constant HR ⇒ proportional hazards ⇒ true γ (log-time interaction) = 0 and
    true β = log(hr).  Exponential (β_weibull = 1) keeps the baseline flat so the
    only structure is the group offset.  Times are generated on a **day-like
    scale** (mean lifetime ~100 d) so the ``log(max(stop, 1))`` clamp in the
    episode split — designed for ttf in days — does not distort the time axis.
    """
    rng = np.random.default_rng(seed)
    cov = np.repeat([0, 1], n // 2)
    base_rate = 1.0 / 100.0
    rate = base_rate * np.where(cov == 1, hr, 1.0)
    t = rng.exponential(1.0 / rate)
    event = (t <= admin_censor).astype(int)
    t = np.minimum(t, admin_censor)
    return pd.DataFrame({
        "subject_id": np.arange(n),
        "well_key": np.arange(n),          # one run per well → no clustering
        "duration": t,
        "event": event,
        "covariate": cov.astype(float),
    })


class EpisodeSplitTests(unittest.TestCase):
    def setUp(self):
        self.subjects = pd.DataFrame({
            "subject_id": [0, 1, 2],
            "well_key": ["a", "b", "c"],
            "duration": [5.0, 3.0, 8.0],
            "event": [1, 1, 0],
            "covariate": [1.0, 0.0, 1.0],
        })

    def test_events_preserved(self):
        split = episode_split(self.subjects)
        # exactly the original number of events survives the split
        self.assertEqual(int(split["event"].sum()), 2)

    def test_terminal_stop_equals_duration(self):
        split = episode_split(self.subjects)
        for sid, dur in zip([0, 1, 2], [5.0, 3.0, 8.0]):
            rows = split[split["subject_id"] == sid]
            self.assertAlmostEqual(rows["stop"].max(), dur)

    def test_cov_logt_is_covariate_times_log_stop(self):
        split = episode_split(self.subjects)
        expected = split["cov"] * np.log(np.maximum(split["stop"], 1.0))
        np.testing.assert_allclose(split["cov_logt"].to_numpy(), expected.to_numpy())

    def test_cluster_column_carried(self):
        split = episode_split(self.subjects)
        self.assertIn("well_key", split.columns)


class GammaRecoveryTests(unittest.TestCase):
    def test_flat_hr_recovers_gamma_near_zero(self):
        subjects = _synth_proportional_hazards(n=800, hr=2.0, seed=7)
        res = fit_time_interaction_cox(subjects)
        # The bug produced γ ≈ −0.851; the corrected estimator must sit near 0.
        self.assertLess(abs(res.gamma), 0.15, f"gamma={res.gamma:.3f} should be ~0")
        # β trades off with γ (it is the HR extrapolated to t=1 day), so pin the
        # effect at a representative time (~mean lifetime 100 d) to true HR=2.
        self.assertAlmostEqual(res.hr_at(100.0), 2.0, delta=0.6)

    def test_hr_is_approximately_time_constant(self):
        subjects = _synth_proportional_hazards(n=800, hr=2.0, seed=11)
        res = fit_time_interaction_cox(subjects)
        # HR at early vs late time should be close under true proportional hazards.
        self.assertAlmostEqual(res.hr_at(30.0), res.hr_at(300.0), delta=0.7)


if __name__ == "__main__":
    unittest.main()
