"""Structural guards for the Vt v3.2 hybrid θ-model.

These test the model's *contract* (shape constraints, pooling policy, clamping, composition
rule) using synthetic fits — no warehouse access, so they stay fast and deterministic.
"""
import numpy as np
import pytest

from analysis.workflows.production_risk import vt_v32_hybrid as M


def _fit(stratum, n, theta_overrides=None):
    theta = {a: np.ones(len(M._KNOTS[a])) for a in M._ARMS}
    for a, v in (theta_overrides or {}).items():
        theta[a] = np.asarray(v, float)
    return M.StratumFit(
        stratum=stratum, n=n, events=n // 2, cc_n=n, cc_events=n // 2,
        beta0=1.25, eta0=500.0, hr_slb=1.4, hr_oth=3.0,
        theta_knots=theta, loglik=-1.0, beta_marginal=0.9, eta_marginal=600.0,
    )


def test_kpod_grid_extends_to_1p5_and_freq_stops_at_60hz():
    """Kpod was extended to 1.5 (support exists); freq must NOT go past +10 Hz (=60 Hz)."""
    assert max(M.KPOD_KNOTS) == 1.5
    assert M.APPLICABILITY["kpod"][1] == 1.5
    assert max(M.FREQ_DEV_KNOTS) == 10.0
    assert M.APPLICABILITY["freq_hz"][1] == 60.0


def test_tent_is_pinned_at_reference_and_monotone_outward():
    """Kpod/freq tents: θ=1 at the pin and non-decreasing moving away on both arms."""
    full = M._tent_full(len(M.KPOD_KNOTS), M._KP_PIN_IX,
                        inc_l=[0.3, 0.2], inc_r=[0.5, 0.1, 0.4])
    th = np.exp(full)
    pin = M._KP_PIN_IX
    assert th[pin] == pytest.approx(1.0)
    assert np.all(th >= 1.0 - 1e-12)
    assert np.all(np.diff(th[:pin + 1]) <= 1e-12)   # rises going left (descending index)
    assert np.all(np.diff(th[pin:]) >= -1e-12)      # rises going right


def test_negative_increments_are_clipped_so_theta_never_drops_below_one():
    full = M._tent_full(len(M.KPOD_KNOTS), M._KP_PIN_IX, inc_l=[-5.0, -1.0], inc_r=[-2.0, -3.0, -1.0])
    assert np.allclose(np.exp(full), 1.0)


def test_theta_is_clamped_outside_the_knot_range():
    """Beyond the last knot θ is flat — an assumption, documented by APPLICABILITY."""
    m = M.build_model({"nonsour": _fit("nonsour", 400), "sour": _fit("sour", 100)})
    at_last = m.theta_at("Kpod", "nonsour", np.array([1.5]))[0]
    beyond = m.theta_at("Kpod", "nonsour", np.array([2.0]))[0]
    assert beyond == pytest.approx(at_last)


def test_ql_guard_flattens_below_the_guard_and_repins_at_reference():
    ql_theta = np.array([0.4, 0.9, 1.0, 1.3, 1.5])       # suspicious low-Ql bonus at knot 47
    m = M.build_model({"nonsour": _fit("nonsour", 400, {"Ql": ql_theta}),
                       "sour": _fit("sour", 100, {"Ql": ql_theta})})
    th = m.theta[("Ql", "nonsour")]
    i47, i100 = M.QL_KNOTS.index(47.0), M.QL_KNOTS.index(100.0)
    assert th[i47] == pytest.approx(th[i100])            # guarded flat below QL_GUARD
    assert th[M._QL_PIN_IX] == pytest.approx(1.0)        # re-pinned at Ql=250


def test_pooling_is_fleet_weighted_geometric_mean_for_ql_kpod_only():
    kp_ns = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    kp_s = np.array([1.0, 1.0, 1.0, 8.0, 8.0, 8.0])
    fr_ns = np.ones(len(M.FREQ_DEV_KNOTS))
    # must VARY across knots: build_model re-pins θ=1 at the reference, so a constant
    # freq θ would (correctly) flatten to all-ones and the strata would look identical.
    fr_s = np.array([1.3, 1.25, 1.2, 1.0, 1.0, 1.05])
    m = M.build_model({
        "nonsour": _fit("nonsour", 300, {"Kpod": kp_ns, "freq_dev": fr_ns}),
        "sour": _fit("sour", 100, {"Kpod": kp_s, "freq_dev": fr_s}),
    })
    # Kpod pooled -> identical across strata, = weighted geometric mean (0.75 / 0.25)
    assert np.allclose(m.theta[("Kpod", "nonsour")], m.theta[("Kpod", "sour")])
    assert m.theta[("Kpod", "nonsour")][-1] == pytest.approx(2.0 ** 0.75 * 8.0 ** 0.25)
    # freq NOT pooled -> strata keep their own values
    assert not np.allclose(m.theta[("freq_dev", "nonsour")], m.theta[("freq_dev", "sour")])


def test_compose_multiplies_thetas_and_beats_naive_rmst_chaining():
    """The headline trap: chaining RMST multipliers overstates life; compose() must not."""
    m = M.build_model({"nonsour": _fit("nonsour", 400), "sour": _fit("sour", 100)})
    s = "nonsour"
    b0, e0 = m.baseline(s)
    r0 = m.rmst_ref(s)
    res = m.compose(s, contractor="oth", ql=None, kpod=None, freq_dev=None)
    assert res["theta_total"] == pytest.approx(m.contractor_hr(s, "oth"))
    assert res["rmst"] == pytest.approx(M.rmst(e0 * res["theta_total"] ** (-1 / b0), b0))
    # two effects: correct composition is stricter than multiplying the two RMST multipliers
    th_a, th_b = 3.0, 2.0
    correct = M.rmst(e0 * (th_a * th_b) ** (-1 / b0), b0) / r0
    naive = (M.rmst(e0 * th_a ** (-1 / b0), b0) / r0) * (M.rmst(e0 * th_b ** (-1 / b0), b0) / r0)
    assert correct < naive
