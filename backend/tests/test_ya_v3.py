"""Contract tests for Ya v3 — the blended-rate model.

Structural only, so they stay fast: what is pinned here is the model's *contract* — the blend
form, the absent Kpod layer, the composition rule, and the honesty of the w interval.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import ya_v3 as V3


def _layers(w: float = 0.5) -> V3.LayerFit:
    """Hand-set layers — exercises the contract without refitting."""
    return V3.LayerFit(
        beta0=1.0847, eta0=1149.1, w=w,
        linear={"slb": float(np.log(1.194)), "oth": float(np.log(2.210))},
        theta_qnom=np.array([0.479, 0.646, 1.000, 1.315, 1.532, 2.331]),
        theta_ql=np.array([0.854, 0.854, 0.997, 1.000, 1.000, 1.644]),
        theta_freq=np.array([1.185, 1.185, 1.000, 1.000, 1.148, 1.148, 1.724]),
        loglik=-4966.270, n=1227, events=635, n_par=21)


def _model(w: float = 0.5) -> V3.YaV3Model:
    return V3.YaV3Model(layers=_layers(w), baseline=None)


def test_blend_is_arithmetic_not_geometric():
    """The owner specified w·θ_Qnom + (1−w)·θ_Ql.  The geometric form is a different model."""
    m = _model(0.5)
    qn, ql = 900.0, 900.0
    t_qn = float(np.exp(V3._sk(qn, V3._QN, np.log(m.layers.theta_qnom))))
    t_ql = float(np.exp(V3._sk(ql, V3._QL, np.log(m.layers.theta_ql))))
    got = float(np.atleast_1d(m.theta_rate(qn, ql))[0])
    assert got == pytest.approx(0.5 * t_qn + 0.5 * t_ql)          # arithmetic
    assert got != pytest.approx(t_qn ** 0.5 * t_ql ** 0.5)        # NOT geometric


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rate_layer_is_one_at_the_reference_for_any_w(w):
    """Both arms are pinned at their reference, so any blend of them is 1 there."""
    m = _model(w)
    assert float(np.atleast_1d(m.theta_rate(V3.QNOM_PIN, V3.QL_PIN))[0]) == pytest.approx(1.0, abs=5e-3)


def test_w_endpoints_recover_the_single_arm_models():
    """w = 1 must be pure Qnom (v2.1-like) and w = 0 pure Ql (v2-like)."""
    qn, ql = 500.0, 450.0
    t_qn = float(np.exp(V3._sk(qn, V3._QN, np.log(_layers().theta_qnom))))
    t_ql = float(np.exp(V3._sk(ql, V3._QL, np.log(_layers().theta_ql))))
    assert float(np.atleast_1d(_model(1.0).theta_rate(qn, ql))[0]) == pytest.approx(t_qn)
    assert float(np.atleast_1d(_model(0.0).theta_rate(qn, ql))[0]) == pytest.approx(t_ql)


def test_no_kpod_layer_anywhere():
    """Kpod is dropped by measurement (θ ≡ 1 in ~10 fits), not shipped flat."""
    m = _model()
    assert not hasattr(m, "theta_kpod")
    assert not any("kpod" in c.lower() for c in V3._spec_table(m)["component"].unique())
    assert "kpod" not in {k.lower() for k in m.layers.linear}


def test_w_interval_is_reported_and_wide():
    """w is NOT identified — the interval must travel with the model, not be quietly dropped."""
    assert V3.W_ML == pytest.approx(0.50)
    lo, hi = V3.W_PROFILE_CI
    assert (lo, hi) == pytest.approx((0.10, 1.00))
    assert hi - lo > 0.5, "if this ever narrows, the docstring's caveat must be revisited"
    assert V3.W_REJECTED == pytest.approx(0.0)      # pure Ql is the one blend ruled out


def test_profile_flags_the_ci_band():
    """profile_w must mark which w values sit inside the 1.92 log-likelihood drop."""
    p = pd.DataFrame({"w": [0.0, 0.5, 1.0], "loglik": [-4970.3, -4966.3, -4966.9]})
    top = p["loglik"].max()
    inci = p["loglik"] >= top - 1.92
    assert not bool(inci.iloc[0])         # w = 0 excluded
    assert bool(inci.iloc[1]) and bool(inci.iloc[2])


def test_monotone_arms_and_clamped_outside():
    m = _model()
    q = np.linspace(10, 3000, 400)
    for arm, knots, th in (("Qnom", V3._QN, m.layers.theta_qnom),
                           ("Ql", V3._QL, m.layers.theta_ql)):
        v = np.exp(V3._sk(q, knots, np.log(th)))
        assert np.all(np.diff(v) >= -1e-12), f"{arm} must be monotone"
        assert v[0] == pytest.approx(th[0])       # clamped flat below
        assert v[-1] == pytest.approx(th[-1])     # clamped flat above


def test_freq_tent_is_one_at_nominal_and_never_protective():
    m = _model()
    assert float(np.atleast_1d(m.theta_freq_at(0.0))[0]) == pytest.approx(1.0)
    assert np.all(m.theta_freq_at(np.linspace(-20, 20, 200)) >= 1.0 - 1e-9)


def test_compose_multiplies_thetas_then_converts_once():
    m = _model()
    r = m.compose(contractor="oth", qnom=900, ql=900, freq_dev=10)
    expected = (m.contractor_hr("oth")
                * float(np.atleast_1d(m.theta_rate(900, 900))[0])
                * float(np.atleast_1d(m.theta_freq_at(10))[0]))
    assert r["theta_total"] == pytest.approx(expected)
    assert r["eta_eff"] == pytest.approx(m.layers.eta0 * r["theta_total"] ** (-1 / m.layers.beta0))
    assert r["rmst"] == pytest.approx(V3.rmst(r["eta_eff"], m.layers.beta0))


def test_rmst_multipliers_do_not_multiply():
    """The standing trap across this workflow."""
    m = _model()
    both = m.compose(contractor="oth", qnom=900, ql=900)["rmst_mult"]
    only_c = m.compose(contractor="oth")["rmst_mult"]
    only_r = m.compose(qnom=900, ql=900)["rmst_mult"]
    assert both < only_c * only_r
