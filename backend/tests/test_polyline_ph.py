"""Structural guards for the shape-constrained polyline PH layers.

These test the *contract* — the shapes hold by construction, θ is pinned and clamped, the
guard behaves, and the fitter recovers a known effect while holding a no-signal arm flat —
plus an equivalence check against the inlined machinery in the shipped Vt v3.2 hybrid, which
is what licenses reusing this module instead of copying that one again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.models.survival import polyline_ph as P
from analysis.workflows.production_risk import vt_v32_hybrid as V32


# --- shape construction -----------------------------------------------------

def test_tent_is_pinned_at_reference_and_monotone_outward():
    spec = P.ArmSpec("k", "k", (0.4, 0.6, 0.8, 1.0, 1.2), 0.8, P.TENT)
    th = np.exp(P.log_theta_from_increments(spec, [0.3, 0.2, 0.5, 0.1]))
    pin = spec.pin_index
    assert th[pin] == pytest.approx(1.0)
    assert np.all(th >= 1.0 - 1e-12)
    assert np.all(np.diff(th[:pin + 1]) <= 1e-12)      # rises going left
    assert np.all(np.diff(th[pin:]) >= -1e-12)         # rises going right


def test_mono_is_non_decreasing_and_pinned_at_the_reference():
    spec = P.ArmSpec("q", "q", (47.0, 100.0, 250.0, 450.0), 250.0, P.MONO)
    th = np.exp(P.log_theta_from_increments(spec, [0.2, 0.3, 0.4]))
    assert np.all(np.diff(th) >= -1e-12)
    assert th[spec.pin_index] == pytest.approx(1.0)
    assert th[0] < 1.0 and th[-1] > 1.0                # below/above the pin


def test_negative_increments_are_clipped_so_the_shape_cannot_invert():
    tent = P.ArmSpec("k", "k", (0.4, 0.6, 0.8, 1.0, 1.2), 0.8, P.TENT)
    assert np.allclose(np.exp(P.log_theta_from_increments(tent, [-5.0, -1.0, -2.0, -3.0])), 1.0)
    mono = P.ArmSpec("q", "q", (47.0, 100.0, 250.0), 250.0, P.MONO)
    assert np.allclose(np.exp(P.log_theta_from_increments(mono, [-1.0, -2.0])), 1.0)


def test_polyline_is_clamped_outside_the_knot_range():
    knots, vals = (0.4, 0.8, 1.2), (0.5, 0.0, 0.7)
    assert P.eval_polyline([0.1], knots, vals)[0] == pytest.approx(0.5)
    assert P.eval_polyline([9.9], knots, vals)[0] == pytest.approx(0.7)
    assert P.eval_polyline([0.6], knots, vals)[0] == pytest.approx(0.25)   # linear, not splined


def test_guard_flattens_below_the_guard_and_repins():
    spec = P.ArmSpec("q", "q", (47.0, 100.0, 250.0, 450.0), 250.0, P.MONO, guard=100.0)
    th = P.apply_guard_and_pin(spec, np.array([0.4, 0.9, 1.0, 1.3]))
    assert th[0] == pytest.approx(th[1])                       # flat below the guard
    assert th[spec.pin_index] == pytest.approx(1.0)            # still pinned at the reference


def test_arm_spec_rejects_a_pin_that_is_not_a_knot_and_unsorted_knots():
    with pytest.raises(ValueError):
        P.ArmSpec("k", "k", (0.4, 0.6, 0.8), 0.7, P.TENT)
    with pytest.raises(ValueError):
        P.ArmSpec("k", "k", (0.8, 0.6, 0.4), 0.8, P.TENT)


# --- equivalence with the shipped Vt v3.2 machinery -------------------------

def test_matches_the_shipped_vt_v32_tent_and_mono_elementwise():
    """Same numbers as the inlined v3.2 implementation — the reuse licence."""
    kp = P.ArmSpec("Kpod", "kpod_run", V32.KPOD_KNOTS, V32.KPOD_PIN, P.TENT)
    inc_l, inc_r = [0.3, 0.2], [0.5, 0.1, 0.4]
    assert np.allclose(P.log_theta_from_increments(kp, inc_l + inc_r),
                       V32._tent_full(len(V32.KPOD_KNOTS), V32._KP_PIN_IX, inc_l, inc_r))
    ql = P.ArmSpec("Ql", "ql", V32.QL_KNOTS, V32.QL_PIN, P.MONO)
    inc = [0.15, 0.25, 0.35, 0.05]
    assert np.allclose(P.log_theta_from_increments(ql, inc), V32._mono_full(np.array(inc)))


# --- fitting ----------------------------------------------------------------

def _synthetic(rng, n_wells=260, *, beta_ql=0.5, oth_hr=2.5, over_hr=1.0):
    """Runs with a known monotone log-Ql hazard, a contractor effect, an optional Kpod
    overload effect, and NO frequency effect."""
    rows = []
    for w in range(n_wells):
        for _ in range(3):
            ql = float(np.exp(rng.uniform(np.log(50), np.log(800))))
            kpod = float(rng.uniform(0.3, 1.4))
            freq = float(rng.uniform(40, 60))
            cg = rng.choice(["brt", "slb", "oth"], p=[0.5, 0.3, 0.2])
            lp = (beta_ql * (np.log(ql) - np.log(250.0))
                  + (np.log(oth_hr) if cg == "oth" else 0.0)
                  + np.log(over_hr) * max(kpod - 0.8, 0.0))
            eta = 600.0 * np.exp(-lp)
            t = eta * (-np.log(rng.uniform(1e-9, 1.0)))
            c = rng.uniform(50, 1500)
            dur, ev = (t, 1.0) if t <= c else (c, 0.0)
            rows.append({"well_key": f"w{w}", "ql": ql, "kpod_run": kpod, "freq_dev": freq - 50.0,
                         "slb": float(cg == "slb"), "oth": float(cg == "oth"),
                         "t_cal": max(dur, 1.0), "event": ev})
    return pd.DataFrame(rows)


_ARMS = (
    P.ArmSpec("Ql", "ql", (47.0, 100.0, 250.0, 450.0, 823.0), 250.0, P.MONO, ridge=1.0),
    P.ArmSpec("Kpod", "kpod_run", (0.4, 0.6, 0.8, 1.0, 1.2), 0.8, P.TENT, ridge=2.0),
    P.ArmSpec("freq_dev", "freq_dev", (-10.0, -5.0, 0.0, 5.0, 10.0), 0.0, P.TENT, ridge=2.0),
)


def test_fit_recovers_a_monotone_ql_effect_and_the_contractor_hr():
    rng = np.random.default_rng(3)
    f = P.fit_polyline_ph(_synthetic(rng), clock="t_cal", event_col="event",
                          arms=_ARMS, linear_terms=("slb", "oth"))
    ql = f.theta_knots["Ql"]
    assert np.all(np.diff(ql) >= -1e-9)                 # monotone by construction
    assert ql[-1] > ql[0] * 1.5                         # and it actually moved
    assert f.hr("oth") > 1.6
    assert f.theta_at("Ql", np.array([250.0]))[0] == pytest.approx(1.0)


def test_a_one_sided_arm_rectifies_noise_upward_and_the_ridge_is_what_controls_it():
    """A tent arm can only rise, so sampling noise is **rectified upward**: on data with no
    true frequency effect the arm still lifts off θ=1.  That is why Kpod/freq carry a heavy
    ridge and why their intervals must come from the bootstrap — a mildly-rising tent arm is
    not evidence of an effect on its own.  The contract tested here is that raising the ridge
    monotonically shrinks a no-signal arm back toward θ≡1."""
    rng = np.random.default_rng(5)
    d = _synthetic(rng)
    peaks = []
    for lam in (0.5, 6.0, 300.0):
        arms = tuple(P.ArmSpec(a.name, a.column, a.knots, a.pin, a.shape,
                               ridge=(lam if a.name == "freq_dev" else a.ridge))
                     for a in _ARMS)
        f = P.fit_polyline_ph(d, clock="t_cal", event_col="event", arms=arms,
                              linear_terms=("slb", "oth"))
        th = f.theta_knots["freq_dev"]
        assert min(th) >= 1.0 - 1e-9                    # never dips below the pin
        peaks.append(max(th))
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[-1] < 1.05                             # heavy ridge ⇒ effectively flat


def test_fit_recovers_a_true_overload_arm_and_keeps_theta_ge_one():
    rng = np.random.default_rng(11)
    f = P.fit_polyline_ph(_synthetic(rng, n_wells=340, over_hr=4.0), clock="t_cal",
                          event_col="event", arms=_ARMS, linear_terms=("slb", "oth"))
    grid = np.linspace(0.4, 1.2, 200)
    th = f.theta_at("Kpod", grid)
    assert np.all(th >= 1.0 - 1e-6)
    assert np.all(np.diff(th[grid >= 0.8]) >= -1e-9)    # monotone rising above the pin
    assert f.theta_at("Kpod", np.array([1.2]))[0] > 1.2
