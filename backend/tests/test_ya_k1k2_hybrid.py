"""Structural guards for the Ya k1/k2 hybrid θ-model.

These test the model's *contract* — the user's shape requirement (tent minimum at Kpod 0.8
and at the nominal frequency), the knot grids against Ya's actual support, clamping, the
frame's nominal-frequency policy, and the composition rule — with synthetic fits, so no
warehouse access is needed and the suite stays fast and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.models.survival import mixture_baseline as MB
from analysis.models.survival import polyline_ph as P
from analysis.workflows.production_risk import ya_k1k2_hybrid as M


def _model(*, rate="Ql", theta_overrides=None, w1=0.0, beta=1.2, eta=600.0,
           slb=0.2, oth=0.8) -> M.YaModel:
    """A YaModel with hand-set θ — exercises the contract without fitting."""
    arms = M._arms_for(rate)
    theta = {a.name: np.ones(len(a.knots)) for a in arms}
    for k, v in (theta_overrides or {}).items():
        theta[k] = np.asarray(v, float)
    layers = P.PolylinePHFit(
        arms=arms, theta_knots=theta, linear={"slb": slb, "oth": oth},
        beta0=beta, eta0=eta, loglik=-1.0, n=1000, events=500, concordance=0.63)
    baseline = MB.BaselineFit(
        model_kind="k1" if w1 == 0.0 else "k2", w1=w1, beta1=beta, eta1=eta,
        beta2=beta, eta2=eta, n=2000, events=1000, loglik_k1=-1.0, aic_k1=2.0,
        loglik_k2=-1.0, aic_k2=4.0, delta_aic=2.0, k2_short_life=100.0, k2_long_life=800.0)
    return M.YaModel(baseline=baseline, layers=layers, boot={}, rate_name=rate, arms=arms,
                     version=M.VERSION_BY_RATE[rate], slug=M.SLUG_BY_RATE[rate],
                     applicability=M._applicability_for(rate))


# --- the shape requirement --------------------------------------------------

def test_kpod_and_freq_are_tents_pinned_at_the_required_references():
    """User requirement: hazard minimal (θ=1) at Kpod 0.8 and at the NOMINAL frequency,
    monotone rising on both arms."""
    kp, fr = M._ARM_BY_NAME["Kpod"], M._ARM_BY_NAME["freq_dev"]
    assert (kp.shape, kp.pin) == (P.TENT, 0.8)
    assert (fr.shape, fr.pin) == (P.TENT, 0.0)          # 0 = f − f_nom ⇒ the nominal frequency
    assert M._ARM_BY_NAME["Ql"].shape == P.MONO         # Ql has a direction, not an optimum


def test_tent_arms_are_monotone_away_from_the_reference_in_both_directions():
    kpod = np.exp(P.log_theta_from_increments(M._ARM_BY_NAME["Kpod"], [0.3, 0.1, 0.2, 0.4, 0.2, 0.1]))
    freq = np.exp(P.log_theta_from_increments(M._ARM_BY_NAME["freq_dev"], [0.2, 0.1, 0.05, 0.1, 0.3, 0.2]))
    for arm, th in (("Kpod", kpod), ("freq_dev", freq)):
        pin = M._ARM_BY_NAME[arm].pin_index
        assert th[pin] == pytest.approx(1.0)
        assert np.all(th >= 1.0 - 1e-12)
        assert np.all(np.diff(th[:pin + 1]) <= 1e-12)
        assert np.all(np.diff(th[pin:]) >= -1e-12)


# --- knot grids vs Ya's support --------------------------------------------

def test_grids_match_ya_support_not_vt_s():
    """Ya's grids are set from Ya's own support and deliberately differ from Vt v3.2."""
    assert min(M.KPOD_KNOTS) == 0.2 and max(M.KPOD_KNOTS) == 1.5   # real mass below 0.4
    assert max(M.FREQ_DEV_KNOTS) == 15.0                            # Ya runs above 60 Hz; Vt never does
    assert M.APPLICABILITY["freq_hz"] == (35.0, 65.0)
    assert M.QL_GUARD is None                                       # low-Ql band is well populated
    assert M.QL_PIN == 250.0 and M.KPOD_PIN == 0.8


def test_theta_is_clamped_flat_outside_the_knot_range():
    m = _model(theta_overrides={"Kpod": [1.4, 1.2, 1.1, 1.0, 1.1, 1.3, 1.6]})
    assert m.theta_at("Kpod", np.array([2.0]))[0] == pytest.approx(
        m.theta_at("Kpod", np.array([1.5]))[0])
    assert m.theta_at("Kpod", np.array([0.05]))[0] == pytest.approx(
        m.theta_at("Kpod", np.array([0.2]))[0])


def test_theta_freq_hz_converts_through_the_pump_nominal():
    m = _model(theta_overrides={"freq_dev": [1.5, 1.2, 1.1, 1.0, 1.1, 1.2, 1.7]})
    # 55 Hz on a 50 Hz pump and 65 Hz on a 60 Hz pump are the same +5 Hz deviation
    assert m.theta_freq_hz(55.0)[0] == pytest.approx(m.theta_freq_hz(65.0, f_nom=60.0)[0])
    assert m.theta_freq_hz(50.0)[0] == pytest.approx(1.0)


# --- frame policy -----------------------------------------------------------

def test_prepare_frame_falls_back_to_50hz_on_implausible_nominal_frequency():
    """Ya's ``nominal_freq_hz`` carries unit errors (220/240) — those must not become f_nom."""
    df = pd.DataFrame({
        "field": ["Ya"] * 4, "t_cal": [100.0, 200.0, 300.0, -5.0], "event": [1.0, 0.0, 1.0, 1.0],
        "nominal_freq_hz": [50.0, 240.0, 60.0, 50.0], "freq_run": [52.0, 52.0, 62.0, 50.0],
        "contractor_group": ["brt", "slb", "oth", "brt"], "code": ["a", "b", "c", "d"],
    })
    out = M.prepare_frame(cached=df)
    assert len(out) == 3                                     # the non-positive clock is dropped
    assert list(out["f_nom"]) == [50.0, 50.0, 60.0]          # 240 → 50 fallback
    assert list(out["freq_dev"]) == [2.0, 2.0, 2.0]          # all three are +2 Hz over nominal
    assert list(out["oth"]) == [0.0, 0.0, 1.0]


# --- composition ------------------------------------------------------------

def test_compose_multiplies_thetas_and_beats_naive_rmst_chaining():
    """The headline trap: chaining RMST multipliers overstates life; compose() must not."""
    m = _model(theta_overrides={"Ql": [0.6, 0.7, 0.8, 1.0, 1.2, 1.8]})
    r = m.compose(contractor="oth", ql=900.0)
    assert r["theta_parts"]["contractor"] == pytest.approx(m.contractor_hr("oth"))
    assert r["theta_total"] == pytest.approx(m.contractor_hr("oth") * 1.8)
    naive = m.life_mult(m.contractor_hr("oth")) * m.life_mult(1.8)
    assert r["rmst_mult"] < naive                            # composition is stricter


def test_compose_at_the_reference_point_is_the_baseline():
    m = _model()
    r = m.compose()                                          # brt, no covariates
    assert r["theta_total"] == pytest.approx(1.0)
    assert r["rmst"] == pytest.approx(r["rmst_ref"])
    assert r["rmst_mult"] == pytest.approx(1.0)


def test_higher_theta_always_shortens_life():
    m = _model()
    mults = [m.life_mult(th) for th in (0.5, 1.0, 2.0, 4.0)]
    assert mults == sorted(mults, reverse=True)
    assert mults[1] == pytest.approx(1.0)


def test_k1_baseline_uses_the_degenerate_esp_models_convention():
    """k1 ships as w1=0 with both components equal — the five-number esp_models form."""
    rng = np.random.default_rng(2)
    t = 500.0 * (-np.log(rng.uniform(1e-9, 1.0, 900))) ** (1 / 1.1)
    c = rng.uniform(50, 1600, 900)
    b = MB.fit_baseline(np.maximum(np.minimum(t, c), 1.0), (t <= c).astype(float), num_starts=15)
    if b.model_kind == "k1":
        assert b.w1 == 0.0 and b.beta1 == b.beta2 and b.eta1 == b.eta2
        # ...but the k2 fit is kept, so a "k2 overlay" cannot silently redraw the k1 curve
        assert b.k2_params and b.k2_params["eta2"] > b.k2_params["eta1"]
        assert not np.allclose(b.S_k2(np.array([300.0])), b.S(np.array([300.0])))
    assert MB.mixture_S(np.array([0.0]), **b.params)[0] == pytest.approx(1.0)


# --- v2.1: Qnom as the rate arm --------------------------------------------

def test_v21_swaps_the_rate_arm_to_qnom_and_keeps_kpod_and_freq():
    """v2.1 = v2 with Qnom (nameplate) in place of Ql; Kpod tent and freq tent unchanged."""
    arms = M._arms_for("Qnom")
    names = [a.name for a in arms]
    assert names == ["Qnom", "Kpod", "freq_dev"]              # rate arm swapped, rest identical
    qn = arms[0]
    assert qn.column == "qnom" and qn.shape == P.MONO and qn.pin == 200.0
    assert M.VERSION_BY_RATE["Qnom"] == "v2.1"
    # distinct result slug so v2.1 never overwrites v2
    assert M.SLUG_BY_RATE["Qnom"] != M.SLUG_BY_RATE["Ql"]
    # applicability carries a qnom window (not ql), plus the common kpod/freq windows
    ap = M._applicability_for("Qnom")
    assert "qnom" in ap and "ql" not in ap and ap["kpod"] == (0.2, 1.5)


def test_v21_rate_arm_is_monotone_and_pinned_at_qnom_200():
    """θ_Qnom is a monotone dose-response, θ=1 at the 200 m³/d reference (pump-size level)."""
    m = _model(rate="Qnom", theta_overrides={"Qnom": [0.66, 0.78, 1.0, 1.16, 1.27, 1.88]})
    qn = m.rate_arm
    th = m.theta_at("Qnom", np.array(qn.knots))
    assert np.all(np.diff(th) >= -1e-9)                       # monotone rising with pump size
    assert m.theta_at("Qnom", np.array([200.0]))[0] == pytest.approx(1.0)
    assert m.theta_at("Qnom", np.array([9999.0]))[0] == pytest.approx(th[-1])   # clamped


def test_v21_compose_takes_qnom_and_rejects_ql():
    """compose is arm-aware: v2.1 accepts qnom=, and refuses ql= (guards against a mix-up)."""
    m = _model(rate="Qnom", theta_overrides={"Qnom": [0.66, 0.78, 1.0, 1.16, 1.27, 1.88]})
    r = m.compose(contractor="oth", qnom=900.0)
    assert r["theta_parts"]["Qnom"] == pytest.approx(1.88)
    assert r["theta_total"] == pytest.approx(m.contractor_hr("oth") * 1.88)
    with pytest.raises(ValueError):
        m.compose(ql=900.0)                                   # Ql is not an arm of v2.1


def test_prepare_frame_exposes_qnom_from_nominal_flow():
    df = pd.DataFrame({
        "field": ["Ya"] * 3, "t_cal": [100.0, 200.0, 300.0], "event": [1.0, 0.0, 1.0],
        "nominal_freq_hz": [50.0, 50.0, 60.0], "freq_run": [52.0, 52.0, 62.0],
        "nominal_flow_m3d": [80.0, 320.0, np.nan],
        "contractor_group": ["brt", "slb", "oth"], "code": ["a", "b", "c"],
    })
    out = M.prepare_frame(cached=df)
    assert list(out["qnom"].fillna(-1)) == [80.0, 320.0, -1.0]   # nameplate carried, NaN kept
