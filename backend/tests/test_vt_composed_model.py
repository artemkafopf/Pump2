"""Contract tests for the Vt composed operating-life model (deployment closed forms)."""
import numpy as np
import pytest

from analysis.workflows.production_risk import vt_composed_model as C


def test_all_layers_are_one_at_the_reference_point():
    for s in C.STRATA:
        assert C.theta_ql(s, C.QL_REF) == pytest.approx(1.0, abs=2e-3)
    assert float(C.theta_freq(C.FREQ_REF)) == pytest.approx(1.0, abs=1e-6)
    # Kpod bathtub reference is the flat band 0.8–1.0; rounds to ~1.0 (accepted small dip)
    assert float(C.theta_kpod(C.KPOD_REF)) == pytest.approx(1.0, abs=1e-2)
    assert C.contractor_hr("nonsour", "brt") == 1.0


def test_ql_is_monotone_with_flat_floor_over_the_whole_window():
    """θ_Ql must be well-behaved over the whole documented window, 0 → 2000 m³/d."""
    lo, hi = C.QL_EVAL_RANGE
    q = np.linspace(lo, hi, 800)
    for s in C.STRATA:
        th = C.theta_ql(s, q)
        assert np.all(np.diff(th) >= -1e-9)                     # monotone up everywhere
        assert np.all(np.isfinite(th))                          # no pole anywhere in range
        assert C.theta_ql(s, 0) == pytest.approx(C.theta_ql(s, C.QL_LO))   # flat guard <100
        assert C.theta_ql(s, 40) == pytest.approx(C.theta_ql(s, C.QL_LO))
        c, A, b, u0 = C.QL_COEF[s]
        for x in (450.0, 823.0, 1250.0, 2000.0):
            u = np.log(x / C.QL_REF)
            assert float(C.theta_ql(s, x)) == pytest.approx(
                np.exp(c * u + A * (np.tanh(b * (u - u0)) + np.tanh(b * u0))), rel=1e-9)


#: The v3.2 hybrid's fitted θ_Ql polyline — the shape the deployed curve must reproduce.
V32_KNOTS = (100.0, 250.0, 450.0, 823.0)
V32_THETA = {"nonsour": (0.820, 1.000, 1.483, 1.644),
             "sour": (0.927, 1.000, 1.159, 1.159)}


def test_ql_reproduces_the_fitted_polyline_knots():
    """The deployed curve is a rendering of the v3.2 fit — it must pass through the knots.

    The pure power law that briefly replaced this missed θ(450) by 0.096 on nonsour, in the
    densest band of the data (250–823 holds 135 runs / 70 events).
    """
    for s, tol in (("nonsour", 3e-3), ("sour", 2.5e-2)):
        for knot, expected in zip(V32_KNOTS, V32_THETA[s]):
            assert float(C.theta_ql(s, knot)) == pytest.approx(expected, abs=tol)


def test_ql_slope_is_continuous_and_decays_to_the_tail():
    """No kink anywhere: the slope must fall smoothly to c, never jump.

    The spliced form this replaced jumped d(lnθ)/d(lnQl) from 0.171→0.30 (nonsour) and
    0.006→0.30 (sour) at Ql 823.
    """
    q = np.geomspace(C.QL_LO, C.QL_EVAL_RANGE[1], 3000)
    for s in C.STRATA:
        c_tail = C.QL_COEF[s][0]
        slope = np.diff(np.log(C.theta_ql(s, q))) / np.diff(np.log(q))
        assert slope.min() >= c_tail - 1e-6                 # never falls below the tail slope
        assert np.abs(np.diff(slope)).max() < 0.02          # smooth: no step in the slope
        assert slope[-1] == pytest.approx(c_tail, abs=2e-3)  # asymptotes to the tail slope


def test_ql_tail_slopes_are_the_agreed_values():
    """c is the one remaining prior; pin it, and the 2.23× stratum ratio it was scaled by."""
    c_ns, c_sr = C.QL_COEF["nonsour"][0], C.QL_COEF["sour"][0]
    assert c_ns == pytest.approx(0.15)
    assert c_sr == pytest.approx(0.067)
    assert c_ns / c_sr == pytest.approx(2.23, abs=0.05)
    assert C.QL_UNBACKED_ABOVE == pytest.approx(1433.0)      # fleet-wide max observed Ql
    assert float(C.theta_ql("nonsour", 2000)) == pytest.approx(1.878, abs=5e-3)
    assert float(C.theta_ql("sour", 2000)) == pytest.approx(1.255, abs=5e-3)


def test_no_spliced_extrapolation_knob_remains():
    """γ is retired — the tail is part of the fitted curve, not a bolt-on."""
    assert not hasattr(C, "QL_EXTRAP_GAMMA")


def test_freq_is_a_valley_min_at_nominal_and_clamped():
    f = np.linspace(35, 70, 400)
    th = C.theta_freq(f)
    assert np.argmin(np.abs(f - 50)) == np.argmin(np.abs(th - th.min())) or th.min() >= 0.99
    assert np.all(th >= 0.99)                                    # θ ≥ 1 valley (no protective dip)
    assert C.theta_freq(35) == pytest.approx(C.theta_freq(42))   # flat below 42
    assert C.theta_freq(70) == pytest.approx(C.theta_freq(65))   # flat above 65
    assert C.theta_freq(42) > 1.2 and C.theta_freq(65) > 1.4     # both arms rise


def test_kpod_is_a_bathtub_and_clamped():
    assert C.theta_kpod(0.2) == pytest.approx(1.15, abs=0.02)    # under-load
    assert float(C.theta_kpod(0.9)) < 1.01                       # flat optimal band
    assert C.theta_kpod(1.5) == pytest.approx(1.20, abs=0.02)    # over-load
    assert C.theta_kpod(0.1) == pytest.approx(C.theta_kpod(0.2))  # flat below 0.2
    assert C.theta_kpod(2.0) == pytest.approx(C.theta_kpod(1.5))  # flat above 1.5


def test_compose_multiplies_thetas_then_converts_once():
    r = C.compose("sour", contractor="oth", ql=450, freq=60, kpod=1.0)
    expected_theta = (C.contractor_hr("sour", "oth") * float(C.theta_ql("sour", 450))
                      * float(C.theta_freq(60)) * float(C.theta_kpod(1.0)))
    assert r.theta_total == pytest.approx(expected_theta)
    b = C.BASELINE["sour"]
    assert r.eta_eff == pytest.approx(b["eta0"] * r.theta_total ** (-1 / b["beta0"]))
    assert r.rmst == pytest.approx(C.rmst(r.eta_eff, b["beta0"]))


def test_rmst_multipliers_do_not_multiply():
    """The headline trap: chaining single-covariate RMST mults overstates life."""
    s = "nonsour"; b = C.BASELINE[s]; r0 = C.rmst_ref(s)
    correct = C.compose(s, contractor="oth", ql=823).rmst_mult
    m_oth = C.compose(s, contractor="oth").rmst_mult
    m_ql = C.compose(s, ql=823).rmst_mult
    assert correct < m_oth * m_ql                               # strictly stronger than the product


def test_reference_rmst_values():
    """From the v3.2 polyline fit — the same fit the deployed Ql curve renders."""
    assert C.rmst_ref("nonsour") == pytest.approx(479.2, abs=1.0)
    assert C.rmst_ref("sour") == pytest.approx(191.0, abs=1.0)
