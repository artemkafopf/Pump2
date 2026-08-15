"""Tests for the full-Vt survival model — baseline mixture-Weibull (k1/k2) + PH hazard
layers (``analysis.workflows.production_risk.vt_physics_model``).

Covers: the polyline tent basis + θ (pinned reference = 1, clamped outside); baseline k1/k2
AIC selection with synthetic recovery (single-Weibull ⇒ k1; two-mode mixture ⇒ k2 with the
modes recovered); and the Cox hazard-layer recovery of a known Ql effect with Kpod/freq held
minor by the ridge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import vt_physics_model as M


# --- polyline tent basis + theta -------------------------------------------

def test_poly_from_free_interpolates_and_pins_reference():
    # free coefs for the non-pinned Kpod knots (0.2,0.4,0.6,1.0,1.2); 0.8 pinned to 0
    free = [0.3, 0.2, 0.1, 0.5, 0.6]
    at_knots = M._poly_from_free(np.array(M.KPOD_KNOTS), M.KPOD_KNOTS, M.KPOD_PIN, free)
    assert at_knots[M.KPOD_KNOTS.index(0.8)] == pytest.approx(0.0)     # pinned
    assert at_knots[0] == pytest.approx(0.3)                          # knot 0.2
    assert at_knots[4] == pytest.approx(0.5)                          # knot 1.0
    # linear midpoint between 0.6 (0.1) and 0.8 (0.0)
    mid = M._poly_from_free(np.array([0.7]), M.KPOD_KNOTS, M.KPOD_PIN, free)[0]
    assert mid == pytest.approx(0.05)


def test_poly_theta_one_at_reference_and_clamped():
    coef = {0.2: 0.3, 0.4: 0.1, 0.6: 0.0, 1.0: -0.2, 1.2: -0.4}   # 0.8 pinned (absent)
    # at the pinned reference θ = 1
    assert M._poly_theta(np.array([M.KPOD_PIN]), M.KPOD_KNOTS, M.KPOD_PIN, coef)[0] == pytest.approx(1.0)
    # at a knot θ = exp(coef - coef_ref); coef_ref (0.8) = 0 → exp(coef)
    assert M._poly_theta(np.array([0.2]), M.KPOD_KNOTS, M.KPOD_PIN, coef)[0] == pytest.approx(np.exp(0.3))
    # clamped below/above the knot range
    below = M._poly_theta(np.array([0.05]), M.KPOD_KNOTS, M.KPOD_PIN, coef)[0]
    assert below == pytest.approx(np.exp(0.3))       # = value at min knot 0.2
    above = M._poly_theta(np.array([1.6]), M.KPOD_KNOTS, M.KPOD_PIN, coef)[0]
    assert above == pytest.approx(np.exp(-0.4))       # = value at max knot 1.2


# --- baseline k1 / k2 recovery ---------------------------------------------

def _weibull_draw(rng, n, beta, eta):
    return eta * (-np.log(rng.uniform(1e-9, 1.0, n))) ** (1.0 / beta)


def _censor(rng, t, horizon):
    c = rng.uniform(50.0, horizon, len(t))
    dur = np.minimum(t, c)
    ev = (t <= c).astype(float)
    return np.maximum(dur, 1.0), ev


def test_baseline_selects_k1_and_recovers_single_weibull():
    rng = np.random.default_rng(11)
    t = _weibull_draw(rng, 1200, beta=1.1, eta=500.0)
    dur, ev = _censor(rng, t, 1600.0)
    b = M.fit_baseline(dur, ev, num_starts=20)
    assert b.model_kind == "k1"                        # a genuine single Weibull
    assert b.beta1 == pytest.approx(1.1, rel=0.12)
    assert b.eta1 == pytest.approx(500.0, rel=0.12)


def test_baseline_selects_k2_on_two_mode_mixture():
    rng = np.random.default_rng(7)
    # 35% short-lived (eta 120) + 65% long-lived (eta 800) — the sour/nonsour analogue
    n = 1600
    short = rng.uniform(size=n) < 0.35
    t = np.where(short, _weibull_draw(rng, n, 1.0, 120.0), _weibull_draw(rng, n, 1.1, 800.0))
    dur, ev = _censor(rng, t, 2000.0)
    b = M.fit_baseline(dur, ev, num_starts=30)
    assert b.model_kind == "k2"
    assert b.delta_aic < -M.AIC_K2_MARGIN
    # the two recovered modes bracket the truth (short ~120, long ~800 life)
    assert 60 < b.k2_short_life < 260
    assert 550 < b.k2_long_life < 1100


# --- Cox hazard layers ------------------------------------------------------

def _make_cc(rng, n_wells=200, *, beta_clq=0.35, oth_hr=2.5, kpod_over_hr=1.0):
    """Complete-case frame with a known Ql (clq) hazard, a strong oth contractor effect, and
    an optional Kpod OVERLOAD hazard (log-HR per unit of Kpod above 0.8). freq has no effect."""
    rows = []
    for w in range(n_wells):
        code = f"W{w:04d}"
        for _ in range(3):
            ql = float(np.exp(rng.uniform(np.log(60), np.log(700))))
            clq = np.log(np.clip(ql, 47, 823)) - np.log(250)
            cg = rng.choice(["brt", "slb", "oth"], p=[0.45, 0.35, 0.20])
            kpod = float(rng.uniform(0.3, 1.2))
            freq = float(rng.uniform(38, 60))
            over = max(kpod - 0.8, 0.0)
            lp = (beta_clq * clq + (np.log(oth_hr) if cg == "oth" else 0.0)
                  + np.log(kpod_over_hr) * over)
            eta = 500.0 * np.exp(-lp)                 # higher hazard → shorter life
            t = eta * (-np.log(rng.uniform(1e-9, 1.0)))
            c = rng.uniform(50, 1400)
            dur, ev = (t, 1.0) if t <= c else (c, 0.0)
            rows.append({"code": code, "ql": ql, "clq": clq, "kpod_run": kpod,
                         "freq_run": freq, "freq_dev": freq - 50.0,
                         "slb": float(cg == "slb"), "oth": float(cg == "oth"),
                         "contractor_group": cg, M.CLOCK: max(dur, 1.0), M.EVENT_COL: ev})
    return pd.DataFrame(rows)


def test_hazard_theta_ge_one_everywhere_and_one_at_reference():
    """The core requirement: θ = 1 at the reference (Kpod 0.8 / 50 Hz) and θ ≥ 1 everywhere."""
    rng = np.random.default_rng(3)
    cc = _make_cc(rng, n_wells=260, beta_clq=0.35, oth_hr=2.5)
    layers = M.fit_hazard_layers(cc)
    assert layers.theta_kpod(np.array([M.KPOD_PIN]))[0] == pytest.approx(1.0)
    assert layers.theta_freq(np.array([M.FNOM_DEFAULT]))[0] == pytest.approx(1.0)
    kgrid = np.linspace(*M.APPLICABILITY["kpod"], 300)
    fgrid = np.linspace(*M.APPLICABILITY["freq"], 300)
    assert np.all(layers.theta_kpod(kgrid) >= 1.0 - 1e-6)
    assert np.all(layers.theta_freq(fgrid) >= 1.0 - 1e-6)
    # every reported knot θ ≥ 1 too
    assert min(r["theta"] for r in layers.kpod_knots) >= 1.0 - 1e-6
    assert min(r["theta"] for r in layers.freq_knots) >= 1.0 - 1e-6


def test_hazard_layers_recover_ql_and_hold_minor_when_no_effect():
    rng = np.random.default_rng(4)
    cc = _make_cc(rng, n_wells=260, beta_clq=0.35, oth_hr=2.5)   # no Kpod/freq effect
    layers = M.fit_hazard_layers(cc)
    prim = layers.primary.set_index("covariate")
    assert prim.loc["clq", "hr"] == pytest.approx(np.exp(0.35), rel=0.30) and prim.loc["clq", "hr"] > 1.0
    assert prim.loc["oth", "hr"] > 1.6
    # no true Kpod/freq signal ⇒ θ stays ≥1 but near 1 (minor)
    assert max(r["theta"] for r in layers.kpod_knots) < 1.6
    assert max(r["theta"] for r in layers.freq_knots) < 1.6


def test_constrained_is_monotone_unimodal():
    """The strict shape: θ minimal (=1) at the reference and MONOTONE on each arm — left arm
    non-increasing toward the reference, right arm non-decreasing away — for both Kpod and freq
    (⇔ life is unimodal with its peak at the reference)."""
    rng = np.random.default_rng(9)
    cc = _make_cc(rng, n_wells=340, beta_clq=0.2, oth_hr=1.5, kpod_over_hr=4.0)
    layers = M.fit_hazard_layers(cc, lam_poly=2.0)
    kg = np.linspace(*M.APPLICABILITY["kpod"], 200)
    tk = layers.theta_kpod(kg)
    assert np.all(np.diff(tk[kg <= M.KPOD_PIN]) <= 1e-9)     # left: non-increasing toward ref
    assert np.all(np.diff(tk[kg >= M.KPOD_PIN]) >= -1e-9)    # right: non-decreasing away
    fg = np.linspace(*M.APPLICABILITY["freq"], 200)
    tf = layers.theta_freq(fg)
    assert np.all(np.diff(tf[fg <= M.FNOM_DEFAULT]) <= 1e-9)
    assert np.all(np.diff(tf[fg >= M.FNOM_DEFAULT]) >= -1e-9)
    # the true overload is recovered: θ rises above the reference and 1.2 ≥ 1.0 ≥ 1
    assert layers.theta_kpod(np.array([1.2]))[0] >= layers.theta_kpod(np.array([1.0]))[0] >= 1.0
    assert layers.theta_kpod(np.array([1.2]))[0] > 1.2


def test_apply_theta_shortens_life_when_hazard_up():
    from analysis.workflows.production_risk.survival import HazardLayer
    params = {"w1": 0.0, "beta1": 1.0, "eta1": 500.0, "beta2": 1.0, "eta2": 500.0}
    hi = HazardLayer.apply_theta(params, 2.0)       # double the hazard
    assert hi["eta1"] < params["eta1"]              # shorter scale ⇒ shorter life
