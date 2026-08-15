"""Contract tests for the Vt v4 nameplate-rate model.

Structural only — no data fit is required, so these stay fast.  The fitted-value tests live
with the run outputs; what is pinned here is the model's *contract*: layer form, guards,
composition rule, and the decisions that must not be silently reverted.
"""
import numpy as np
import pytest

from analysis.workflows.production_risk import vt_v4 as V


def _model():
    """A V4Model with hand-set parameters — exercises the contract, not the fit."""
    fits = {
        "nonsour": V.StratumFit(
            stratum="nonsour", n=249, events=127, beta0=1.2134, eta0=686.4,
            level={"brt": 1.0, "slb": 1.1304, "oth": 1.8926},
            level_ci={"brt": (1.0, 1.0, np.nan), "slb": (0.689, 1.854, 0.6275),
                      "oth": (0.948, 3.780, 0.0707)},
            theta_qnom=np.array([0.7775, 1.0, 1.0, 1.3762, 1.8193, 1.8193, 1.9598, 2.5081]),
            loglik=-918.02, window_n=126, window_events=60, n_capped=1),
        "sour": V.StratumFit(
            stratum="sour", n=116, events=81, beta0=1.2766, eta0=202.1,
            level={"brt": 1.0, "slb": 1.1814, "oth": 1.44},
            level_ci={"brt": (1.0, 1.0, np.nan), "slb": (0.599, 2.329, 0.6302),
                      "oth": (0.757, 2.737, 0.2659)},
            theta_qnom=np.array([0.5173, 1.0, 1.0, 1.1602, 1.197, 1.197, 1.197, 1.197]),
            loglik=-487.08, window_n=60, window_events=45, n_capped=0),
    }
    support = {(s, c): (150.0, 900.0, 100, 50) for s in V.STRATA for c in V.CGS}
    return V.V4Model(fits=fits, support=support)


def test_reference_point_is_all_ones():
    m = _model()
    for s in V.STRATA:
        r = m.compose(s)                       # brt, Qnom 250, 50 Hz, Kpod 0.8
        assert r["theta_total"] == pytest.approx(1.0, abs=1e-2)
        assert r["rmst"] == pytest.approx(r["rmst_ref"], rel=1e-6)


def test_kpod_overlay_is_off_by_default_and_is_a_pure_null():
    """Kpod is the Ql residual after Qnom (0.99, p=0.98) — the default must not penalise."""
    assert V.KPOD_OVERLAY is False
    k = np.array([0.2, 0.6, 0.8, 1.0, 1.5])
    assert np.allclose(V.theta_kpod(k), 1.0)
    # switched on, it is the Ya bathtub: >= 1 everywhere, minimum in the 0.8-1.0 band
    on = V.theta_kpod(k, overlay=True)
    assert np.all(on >= 0.99) and on.max() > 1.1


def test_qnom_layer_is_monotone_and_guarded():
    m = _model()
    q = np.linspace(0, 3000, 800)
    for s in V.STRATA:
        th = m.theta_qnom(s, q)
        assert np.all(np.diff(th) >= -1e-12)                  # isotonic by construction
        assert np.all(np.isfinite(th))
        assert m.theta_qnom(s, 10) == pytest.approx(m.theta_qnom(s, V.QNOM_GUARD))
        assert m.theta_qnom(s, 5000) == pytest.approx(m.theta_qnom(s, V.QNOM_KNOTS[-1]))


def test_qnom_cap_and_knots_cover_the_observed_fleet():
    """52 runs / 30 events live above Qnom 1000 — the grid must not stop at 900."""
    assert V.QNOM_KNOTS[-1] >= 1600.0
    assert V.QNOM_CAP == pytest.approx(1600.0)   # the lone 7000 nameplate is implausible


def test_contractor_window_is_resolved_at_call_time():
    """A default-arg binding would silently ignore sensitivity runs — it did, once."""
    import inspect
    assert inspect.signature(V.fit_contractor_levels).parameters["window"].default is None


def test_compose_multiplies_thetas_then_converts_once():
    m = _model()
    r = m.compose("sour", contractor="oth", qnom=600, freq=60)
    expected = (m.contractor_level("sour", "oth")
                * float(m.theta_qnom("sour", np.array([600.0]))[0])
                * float(V.theta_freq(60)) * 1.0)
    assert r["theta_total"] == pytest.approx(expected)
    b0, e0 = m.baseline("sour")
    assert r["eta_eff"] == pytest.approx(e0 * r["theta_total"] ** (-1 / b0))
    assert r["rmst"] == pytest.approx(V.rmst(r["eta_eff"], b0))


def test_rmst_multipliers_do_not_multiply():
    """The standing trap: chaining single-covariate RMST multipliers overstates life."""
    m = _model()
    both = m.compose("nonsour", contractor="oth", qnom=600).rmst_mult if False else \
        m.compose("nonsour", contractor="oth", qnom=600)["rmst_mult"]
    only_c = m.compose("nonsour", contractor="oth")["rmst_mult"]
    only_q = m.compose("nonsour", qnom=600)["rmst_mult"]
    assert both < only_c * only_q


def test_freq_layer_is_the_ya_transfer():
    """Vt's own frequency response is inert; the deliverable imports Ya's U-valley."""
    assert float(V.theta_freq(50)) == pytest.approx(1.0, abs=1e-6)
    assert float(V.theta_freq(42)) > 1.2 and float(V.theta_freq(65)) > 1.4
    assert float(V.theta_freq(35)) == pytest.approx(float(V.theta_freq(42)))   # clamped
    assert float(V.theta_freq(70)) == pytest.approx(float(V.theta_freq(65)))


def test_in_support_flags_the_confident_window():
    m = _model()
    assert m.in_support("nonsour", "brt", 400) is True
    assert m.in_support("nonsour", "brt", 1400) is False
