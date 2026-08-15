"""The eta-gap link: a hard FLOOR at min_eta_ratio, and no ceiling.

Guards a bug where `eta2 = eta1 * (2 + softplus(phi))` with `phi <= 12` capped
eta2/eta1 at 2 + softplus(12) = 14.0 exactly.  Only a floor was ever intended, and the
cap bound 4 of 9 production strata — fixing their eta2, B50 and tail to a bound rather
than to the data.
"""
import numpy as np

from analysis.models.survival.weibull_em import (
    _LOG_ETA_GAP_BOUNDS,
    fit_latent_weibull_em,
)


def _two_component_sample(n=4000, seed=0, w1=0.3, b1=1.2, e1=30.0, b2=1.5, e2=1500.0):
    """eta2/eta1 = 50 — far past the old 14.0 ceiling."""
    rng = np.random.default_rng(seed)
    pick = rng.random(n) < w1
    t = np.where(pick, e1 * rng.weibull(b1, n), e2 * rng.weibull(b2, n))
    return t, np.ones(n, dtype=int)


def test_gap_bounds_allow_far_more_than_the_old_ceiling():
    lo, hi = _LOG_ETA_GAP_BOUNDS
    assert 2.0 + np.exp(lo) < 2.001          # floor essentially exactly min_eta_ratio
    assert 2.0 + np.exp(hi) > 1000.0         # no ceiling anywhere near 14


def test_recovers_an_eta_ratio_far_above_the_old_ceiling():
    t, e = _two_component_sample()
    r = fit_latent_weibull_em(t, e, num_starts=10, max_iter=300)
    ratio = r.model.component_2.eta / r.model.component_1.eta
    assert ratio > 20.0, f"eta2/eta1={ratio:.2f} — still capped near the old 14.0"


def test_min_eta_ratio_floor_is_still_hard():
    """A sample with NO scale separation must not be able to push the ratio below 2."""
    rng = np.random.default_rng(1)
    t = 100.0 * rng.weibull(1.3, 1500)
    r = fit_latent_weibull_em(t, np.ones(1500, dtype=int), num_starts=6, max_iter=200,
                              min_eta_ratio=2.0)
    ratio = r.model.component_2.eta / r.model.component_1.eta
    assert ratio >= 2.0 - 1e-6, f"eta2/eta1={ratio:.4f} broke the floor"
