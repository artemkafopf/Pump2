"""Tests for the pump-sizing landmark design.

The three defects the landmark exists to fix are pinned as tests, because each of them
silently produces a *plausible* wrong answer rather than an error:

* the fixed window must not vary with how long the run turned out to be
  (:func:`test_window_features_are_invariant_to_what_happens_after_the_landmark`);
* runs that never reach the landmark must be excluded and **counted**, not dropped
  quietly (:func:`test_short_run_returns_none`);
* the survival fit must be censoring-aware — dropping censored rows is the single
  easiest way to manufacture a result here (:func:`test_ph_fit_uses_censored_rows`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import pump_sizing_landmark as PSL

LM = 30  # small landmark keeps the synthetic frames readable


def _daily(n_days: int, *, q0=100.0, q_slope=0.0, p0=80.0, p_slope=0.0,
           gor=200.0, wct=30.0, start="2020-01-01"):
    """A synthetic run: `n_days` consecutive operating days with linear trends."""
    dts = pd.date_range(start, periods=n_days, freq="D").to_numpy()
    i = np.arange(n_days, dtype=float)
    return {
        "dts": dts,
        "qliq": np.full(n_days, q0) + q_slope * i,
        "freq": np.full(n_days, 50.0),
        "p_intake": np.full(n_days, p0) + p_slope * i,
        "rpl": np.full(n_days, 200.0),
        "rzab": np.full(n_days, 100.0),
        "kprod": np.full(n_days, 2.0),
        "watercut": np.full(n_days, wct),
        "gas_factor": np.full(n_days, gor),
    }


def _feats(d, **kw):
    kw.setdefault("qnom", 200.0)
    kw.setdefault("p_bubble", 250.0)
    kw.setdefault("landmark_op_days", LM)
    return PSL.landmark_window_features(**d, **kw)


# ---------------------------------------------------------------------------
# Eligibility and the fixed window
# ---------------------------------------------------------------------------

def test_short_run_returns_none():
    """A run that never reaches the landmark is not eligible — and says so."""
    assert _feats(_daily(LM - 1)) is None


def test_exactly_at_landmark_is_eligible():
    out = _feats(_daily(LM))
    assert out is not None
    assert out["n_op_days_window"] == LM


def test_idle_days_do_not_count_toward_the_landmark():
    """The clock is OPERATING days (qliq > 0), so idle days must not advance it."""
    d = _daily(LM + 10)
    d["qliq"][:10] = 0.0                       # ten idle days at the start
    out = _feats(d)
    assert out is not None
    assert out["n_op_days_window"] == LM
    # 10 idle + 30 operating = the landmark lands 39 calendar days after the first row.
    assert out["landmark_day_offset"] == pytest.approx(39.0)


def test_run_too_short_after_idle_days_is_ineligible():
    d = _daily(LM)
    d["qliq"][:5] = 0.0                        # only LM-5 operating days remain
    assert _feats(d) is None


def test_window_features_are_invariant_to_what_happens_after_the_landmark():
    """THE defect the landmark design exists to fix.

    A feature computed over a run's own length correlates with run length almost
    definitionally.  Here the two runs are identical up to the landmark and diverge wildly
    afterwards; every windowed feature must be bit-identical.
    """
    short = _daily(LM)
    long = _daily(LM * 6)
    long["qliq"][LM:] = 5.0                    # collapses after the landmark
    long["p_intake"][LM:] = 10.0

    a, b = _feats(short), _feats(long)
    keys = [k for k in a if isinstance(a[k], float) and np.isfinite(a[k])]
    assert keys
    for k in keys:
        assert a[k] == pytest.approx(b[k], rel=1e-12), f"{k} leaked post-landmark data"


def test_slope_is_measured_per_100_operating_days():
    """kpod slope units: a −1 m³/d per day decline on Qnom=200 is −0.5 Kpod/100 op-days."""
    out = _feats(_daily(LM, q0=200.0, q_slope=-1.0), qnom=200.0)
    assert out["kpod_slope"] == pytest.approx(-0.5, rel=1e-6)


def test_level_and_slope_are_independent_signals():
    """Same mean Kpod, opposite trajectories — the case mean-Kpod cannot see.

    This is the fleet result in miniature: ``corr(Qnom, Kpod_mean) = −0.04`` while the
    slope carries the signal.  If these two ever collapse to the same number the whole
    analysis is pointless.
    """
    rising = _feats(_daily(LM, q0=100.0, q_slope=+2.0))
    falling = _feats(_daily(LM, q0=100.0 + 2.0 * (LM - 1), q_slope=-2.0))
    assert rising["kpod_level"] == pytest.approx(falling["kpod_level"], rel=1e-9)
    assert rising["kpod_slope"] > 0 > falling["kpod_slope"]


def test_insufficient_points_gives_nan_slope_not_a_guess():
    d = _daily(LM)
    d["p_intake"][:] = np.nan
    d["p_intake"][:3] = 80.0                   # fewer than MIN_SLOPE_POINTS
    out = _feats(d)
    assert np.isnan(out["pint_slope"])


def test_bad_qnom_gives_nan_kpod_but_keeps_pressure_features():
    out = _feats(_daily(LM), qnom=float("nan"))
    assert out is not None
    assert np.isnan(out["kpod_level"])
    assert np.isfinite(out["pint_level"])


# ---------------------------------------------------------------------------
# Gas mediator wiring
# ---------------------------------------------------------------------------

def test_beta_is_anchored_when_bubble_point_is_supplied():
    out = _feats(_daily(LM))
    assert out["beta_anchored"] is True
    assert 0.0 <= out["beta_mean"] <= 1.0


def test_beta_unanchored_flag_when_bubble_point_missing():
    out = _feats(_daily(LM), p_bubble=float("nan"))
    assert out["beta_anchored"] is False


def test_falling_intake_pressure_raises_beta_slope():
    """The mediator must respond to the mechanism it is supposed to carry."""
    out = _feats(_daily(LM, p0=150.0, p_slope=-3.0))
    assert out["pint_slope"] < 0
    assert out["beta_slope"] > 0


def test_impossible_rzab_is_guarded_out_of_drawdown():
    """Warehouse rzab spans −125 973 to 1 249 937; an unguarded mean is meaningless."""
    d = _daily(LM)
    d["rzab"][:] = -125_973.3
    out = _feats(d)
    assert np.isnan(out["drawdown_level"])


# ---------------------------------------------------------------------------
# Weibull-PH fit
# ---------------------------------------------------------------------------

def _surv_frame(n=800, coef=0.7, seed=0):
    """Weibull data with a known PH coefficient on ``x`` and real right-censoring."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    beta, eta = 1.3, 400.0
    u = rng.uniform(size=n)
    t = eta * (-np.log(u) / np.exp(coef * x)) ** (1.0 / beta)
    c = rng.uniform(50, 1200, n)
    return pd.DataFrame({"t_resid": np.minimum(t, c),
                         PSL.EVENT_COL: (t <= c).astype(int), "x": x})


def test_ph_fit_recovers_a_known_coefficient():
    fit, _ = PSL.fit_weibull_ph(_surv_frame(coef=0.7), ("x",), standardize=False)
    assert fit.converged
    assert fit.coefs["x"] == pytest.approx(0.7, abs=0.12)
    assert fit.beta_shape == pytest.approx(1.3, rel=0.15)


def test_ph_fit_uses_censored_rows():
    """Dropping censored runs biases the fit — the failure mode ``project_field_sim`` pins.

    Fitting the same data failures-only must move the scale materially; if it did not,
    the censored rows were not contributing and the likelihood is wrong.
    """
    d = _surv_frame(seed=3)
    full, _ = PSL.fit_weibull_ph(d, ("x",), standardize=False)
    only, _ = PSL.fit_weibull_ph(d[d[PSL.EVENT_COL] == 1], ("x",), standardize=False)
    assert only.eta_scale < full.eta_scale * 0.9


def test_ph_fit_is_degenerate_safe_on_tiny_input():
    d = pd.DataFrame({"t_resid": [10.0, 20.0], PSL.EVENT_COL: [1, 0], "x": [0.0, 1.0]})
    fit, _ = PSL.fit_weibull_ph(d, ("x",))
    assert not fit.converged
    assert np.isnan(fit.beta_shape)


def test_rmst_matches_closed_form_for_exponential():
    """β = 1 → RMST(0,T) = η(1 − e^(−T/η)), a closed form the integrator must reproduce."""
    eta, T = 500.0, 730.0
    assert PSL.rmst(eta, 1.0, T) == pytest.approx(eta * (1 - np.exp(-T / eta)), rel=1e-4)


def test_rmst_is_bounded_by_the_horizon():
    assert PSL.rmst(1e9, 1.0, 730.0) == pytest.approx(730.0, rel=1e-3)


def test_rmst_invalid_parameters_are_nan():
    assert np.isnan(PSL.rmst(np.nan, 1.0))
    assert np.isnan(PSL.rmst(500.0, -1.0))


def test_lr_test_refuses_non_nested_comparison():
    """Different complete-case n means the models are not nested — must not report a p."""
    a = PSL.PHFit(terms=("x",), beta_shape=1.0, eta_scale=1.0, loglik=-100.0, n=100)
    b = PSL.PHFit(terms=("x", "y"), beta_shape=1.0, eta_scale=1.0, loglik=-90.0, n=80)
    r = PSL.lr_test(a, b)
    assert np.isnan(r["p"])
    assert "NOT NESTED" in r["note"]


def test_lr_test_on_nested_models():
    a = PSL.PHFit(terms=("x",), beta_shape=1.0, eta_scale=1.0, loglik=-100.0, n=100)
    b = PSL.PHFit(terms=("x", "y"), beta_shape=1.0, eta_scale=1.0, loglik=-95.0, n=100)
    r = PSL.lr_test(a, b)
    assert r["lr"] == pytest.approx(10.0)
    assert r["df"] == 1
    assert r["p"] < 0.01


def test_cv_loglik_prefers_the_informative_covariate():
    """Out-of-sample, not in-sample — the check that caught the v4 overfit."""
    d = _surv_frame(n=600, coef=0.9, seed=11)
    d["noise"] = np.random.default_rng(5).normal(0, 1, len(d))
    real = PSL.cv_loglik(d, ("x",), repeats=2)
    junk = PSL.cv_loglik(d, ("noise",), repeats=2)
    assert real["cv_ll_per_event"] > junk["cv_ll_per_event"]


def test_coverage_reports_the_events_it_discards():
    cov = PSL.LandmarkCoverage(
        landmark_op_days=90, n_runs_total=100, n_no_qnom=0, n_no_telemetry=0,
        n_short_of_landmark=50, n_eligible=50,
        n_events_dropped_short=40, n_events_eligible=40)
    assert cov.share_of_events_lost() == pytest.approx(0.5)
    f = cov.as_frame()
    assert f["share_events_lost_to_landmark"].iloc[0] == pytest.approx(0.5)
    # The short-run frame must never leak into the flat summary row.
    assert not any(isinstance(v, pd.DataFrame) for v in f.iloc[0].to_list())


def test_selection_audit_detects_size_skew_in_the_dropped_runs():
    """If the landmark preferentially drops big pumps, the audit must SAY so.

    This is the check that decides whether the estimand is usable at all.
    """
    kept = pd.DataFrame({"qnom": np.full(200, 200.0), "field": "Ya"})
    drop = pd.DataFrame({"qnom": np.full(200, 800.0), "field": "Ya",
                         PSL.EVENT_COL: 1, "t_total": 30.0})
    audit = PSL.landmark_selection_audit(kept, drop)
    assert (audit["flag"] == "DROPPED RUNS ARE LARGER").any()


def test_selection_audit_is_quiet_when_there_is_no_skew():
    rng = np.random.default_rng(2)
    kept = pd.DataFrame({"qnom": rng.normal(300, 50, 300), "field": "Ya"})
    drop = pd.DataFrame({"qnom": rng.normal(300, 50, 300), "field": "Ya",
                         PSL.EVENT_COL: 1, "t_total": 30.0})
    audit = PSL.landmark_selection_audit(kept, drop)
    assert (audit["flag"] == "no size difference").all()
