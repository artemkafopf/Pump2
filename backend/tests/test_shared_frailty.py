"""Shared gamma-frailty Weibull — recovery, null behaviour and design-effect algebra."""

import numpy as np
import pandas as pd
import pytest

from analysis.models.survival.shared_frailty import (
    design_effect,
    effective_cluster_size,
    fit_shared_frailty,
)


def _simulate(theta_true, n_wells=480, mean_size=4.5, shape=1.3, scale=400.0,
              cens=500.0, seed=7):
    """Clustered Weibull with shared gamma frailty and administrative censoring."""
    rng = np.random.default_rng(seed)
    sizes = rng.poisson(mean_size - 1, n_wells) + 1
    rows = []
    for well, size in enumerate(sizes):
        z = 1.0 if theta_true <= 0 else rng.gamma(1.0 / theta_true, theta_true)
        for _ in range(size):
            t = scale * (rng.exponential() / z) ** (1.0 / shape)
            c = rng.uniform(0.3 * cens, 2.0 * cens)
            rows.append((well, min(t, c), float(t <= c)))
    return pd.DataFrame(rows, columns=["code", "t", "event"])


def _fit(df):
    return fit_shared_frailty(df, duration_col="t", event_col="event", cluster_col="code")


def test_recovers_frailty_variance():
    """theta is recovered to within the censoring-induced downward bias."""
    fit = _fit(_simulate(0.8))
    assert 0.55 < fit.theta < 1.05
    assert fit.lr_pvalue < 1e-10
    assert 0.35 < fit.rho < 0.55  # closed form at theta=0.8 is ~0.46


def test_null_when_runs_are_independent():
    """No well effect in the data means no well effect in the fit."""
    fit = _fit(_simulate(0.0))
    assert fit.theta < 0.05
    assert fit.lr_pvalue > 0.05
    assert fit.design_effect < 1.1


def test_unmodelled_frailty_attenuates_the_weibull_shape():
    """The independent fit reads a flatter hazard than the truth — the reason the shape
    has to be checked against a frailty fit before an infant-hazard claim is made."""
    fit = _fit(_simulate(0.8))
    assert fit.shape > fit.shape_indep
    assert fit.shape_indep < 1.3  # true shape, attenuated by ignoring the clustering


def test_rho_increases_with_theta():
    fits = [_fit(_simulate(t, seed=3)) for t in (0.1, 0.5, 1.2)]
    rhos = [f.rho for f in fits]
    assert rhos == sorted(rhos)
    assert all(0.0 <= r < 1.0 for r in rhos)


def test_kendall_tau_matches_closed_form():
    fit = _fit(_simulate(0.5))
    assert fit.kendall_tau == pytest.approx(fit.theta / (fit.theta + 2.0))


def test_effective_cluster_size_penalises_unequal_clusters():
    """sum(m^2)/sum(m) exceeds the arithmetic mean whenever cluster sizes vary."""
    equal = np.array([4, 4, 4, 4])
    uneven = np.array([1, 1, 1, 13])
    assert effective_cluster_size(equal) == pytest.approx(4.0)
    assert effective_cluster_size(uneven) > uneven.mean()


def test_design_effect_is_one_without_correlation():
    sizes = np.array([2, 5, 9])
    assert design_effect(sizes, 0.0) == pytest.approx(1.0)
    assert design_effect(sizes, 0.2) > 1.0


def test_covariates_are_estimated_alongside_the_frailty():
    """A covariate with a known sign comes back with that sign under the frailty fit."""
    df = _simulate(0.4, seed=5)
    rng = np.random.default_rng(1)
    df["x"] = rng.normal(size=len(df))
    df["t"] = df["t"] * np.exp(-0.4 * df["x"])  # higher x -> shorter life -> beta > 0
    fit = fit_shared_frailty(
        df, duration_col="t", event_col="event", cluster_col="code", covariates=["x"]
    )
    assert fit.coefs["x"] > 0
