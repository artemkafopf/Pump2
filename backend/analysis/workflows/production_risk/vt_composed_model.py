"""Vt composed operating-life model (v3.2 baseline/contractor/Ql + Ya-transferred freq/Kpod).

This is the **deployment** form: closed-form rational hazard layers, no fitting. It is the
single source of truth for the Excel/VBA port. Built 2026-07-24 as a constrained engineering
deliverable (physically-correct behaviour + visual confirmation + predictions not wildly wrong),
NOT a pure statistical fit — see ``results/production_risk_vt_freq_transfer/*/PROPOSED_MODEL.md``.

    θ_total = HR_contractor · θ_Ql(Ql) · θ_freq(f) · θ_Kpod(Kpod)
    η_eff   = η₀ · θ_total^(−1/β₀)
    RMST(0,730) = ∫₀⁷³⁰ exp(−(t/η_eff)^β₀) dt          [the headline TTF]
    median  = η_eff · (ln 2)^(1/β₀)

Reference point (all θ = 1): contractor **brt**, **Ql = 250 m³/d**, **freq = 50 Hz**, **Kpod = 0.8**.

Provenance of each layer:
* baseline β₀/η₀, contractor, θ_Ql — from the Vt v3.2 hybrid ([[project_vt_v32_hybrid_model]]),
  θ_Ql a stratum-specific **saturating shape plus a persistent tail slope**, flat below 100 —
  one smooth monotone curve over ``QL_EVAL_RANGE`` = 0–2000 m³/d, with no splice and no kink,
  that reproduces the v3.2 polyline knots.  Above ``QL_UNBACKED_ABOVE`` = 1433 (the fleet-wide
  maximum observed Ql) the curve rests on no observation and its slope is the chosen tail ``c``.
* θ_freq — transferred from Ya (Vt's own freq is inert); a U-valley at nominal, rational fit to
  the reconciled V3 layer, clamp flat outside [42, 65] Hz ([[project_ya_to_vt_freq_transfer]]).
* θ_Kpod — transferred from Ya; a bathtub / safe-operating-window (under-load 1.15 @0.2, flat
  0.8–1.0, over-load 1.20 @1.5), rational, clamp flat outside [0.2, 1.5]. Data-consistent PRIOR
  (Ya joint fit sees Kpod flat; partly a Qnom/pump-size confound), not a hard estimate.

**θ's multiply; RMST multipliers do NOT** — always go through :func:`compose`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma as _gamma

import numpy as np

RMST_HORIZON = 730.0
STRATA = ("nonsour", "sour")

# --- baseline (reference point: brt, Ql 250, freq 50, Kpod 0.8) -------------
# From the v3.2 polyline fit — the same fit the Ql layer below renders.  Baseline, contractor
# and Ql must always come from ONE fit: swapping the Ql layer's shape changes what the baseline
# has to absorb.  (A brief 2026-07-27 experiment with a pure power-law Ql arm required its own
# baseline — β₀ 1.2298/η₀ 656.0 and 1.2695/228.1 — and was reverted with it.)
BASELINE = {
    "nonsour": {"beta0": 1.23, "eta0": 705.8},      # RMST₇₃₀ ref 479.2 d
    "sour":    {"beta0": 1.31, "eta0": 207.7},      # RMST₇₃₀ ref 191.0 d
}
# --- contractor hazard multiplier (brt = 1.0) ------------------------------
CONTRACTOR = {
    "nonsour": {"brt": 1.0, "slb": 1.423, "oth": 3.017},
    "sour":    {"brt": 1.0, "slb": 1.064, "oth": 1.507},
}

# --- θ_Ql: persistent tail slope + saturating term -------------------------
#
#   θ_Ql(Ql) = exp( c·u + A·[ tanh(b(u − u₀)) + tanh(b·u₀) ] ),  u = ln(max(Ql, 100)/250)
#
# One smooth monotone curve over the whole 0–2000 window — **no splice and no kink** — that
# nevertheless **reproduces the v3.2 polyline knots** (nonsour exactly, RMSE 0.00000) and keeps
# rising at high rate.  The two jobs are cleanly separated by construction:
#   * the **tanh term** carries the fitted mid-range shape (steep 250→450, flattening after);
#   * the **c·u term** is the persistent high-rate slope, i.e. what used to be the γ prior.
# Because the tanh saturates, d(lnθ)/d(lnQl) falls smoothly from ~0.96 in the mid range to
# exactly c far out — the prior is confined to where the fit has nothing to say, without a
# corner marking the handover.
#
# **c is chosen as large as the fit allows for free.**  For nonsour every c ≤ 0.15 reproduces
# all four knots at RMSE 0.0000; 0.20 starts to cost (0.0097) and the pure power law c = 0.358
# costs 0.0793 (θ(450) 1.387 vs the fitted 1.483, in the densest band of the data — 135 runs /
# 70 events live in 250–823).  Sour's fitted arm is exactly flat 450→823 so any c > 0 costs a
# little; c is scaled by the same 2.23× stratum sensitivity ratio the rest of the model uses,
# 0.15/2.23 ≈ 0.067, for max knot deviation 0.024 (2.1%).
#
# History (2026-07-27, do not retry): a log-rational (2,2) **turned DOWN inside the fitted
# range**; a hard γ splice onto the saturating arm produced a 0.171→0.30 slope jump at 823; a
# **pure power law** removed the kink but under-fit the dense mid range; and free MLE of any
# 3–4 parameter sigmoid **degenerates to a step** (sour even flips to a decreasing curve).
QL_REF, QL_LO = 250.0, 100.0
QL_SMOOTH_B_CAP = 6.0        # only binds on sour; nonsour's free b is 4.27
QL_COEF = {  # (c_tail, A, b, u0)
    "nonsour": (0.1500, 0.1898, 4.2743, 0.1933),   # RMSE vs polyline knots 0.00000
    "sour":    (0.0670, 0.0514, 6.0000, 0.1452),   # RMSE 0.0168, max knot deviation 0.0243
}
#: Top knot of the v3.2 polyline — a support marker only.  :func:`theta_ql` does not change
#: behaviour there; the curve is one continuous function across it.
QL_HI = 823.0
#: Documented evaluation window, and the fleet-wide maximum observed Ql.  The curve does not
#: change form at QL_UNBACKED_ABOVE — beyond it the *same* fitted slope simply continues into
#: a region with no observations (fleet has ZERO runs above 1500 in any field).
QL_EVAL_RANGE = (0.0, 2000.0)
QL_UNBACKED_ABOVE = 1433.0
# --- θ_freq: rational, u = f − 50; clamp flat outside [42, 65] --------------
FREQ_REF, FREQ_LO, FREQ_HI = 50.0, 42.0, 65.0
FREQ_COEF = (0.03235, 0.00176, 0.03540, -0.00106)   # (a1, a2, b1, b2), num has leading 1
# --- θ_Kpod: rational bathtub in k; clamp flat outside [0.2, 1.5] -----------
KPOD_REF, KPOD_LO, KPOD_HI = 0.8, 0.2, 1.5
KPOD_COEF = (1.2936, 0.1597, -0.2982, 0.8460, -0.6911)  # (a0,a1,a2,b1,b2)


def theta_ql(stratum: str, ql) -> np.ndarray:
    """Ql layer: saturating shape + persistent tail slope, flat below QL_LO.  Smooth, monotone."""
    c, A, b, u0 = QL_COEF[stratum]
    u = np.log(np.maximum(np.asarray(ql, float), QL_LO) / QL_REF)   # flat guard below 100
    return np.exp(c * u + A * (np.tanh(b * (u - u0)) + np.tanh(b * u0)))


def theta_freq(f) -> np.ndarray:
    a1, a2, b1, b2 = FREQ_COEF
    u = np.clip(np.asarray(f, float), FREQ_LO, FREQ_HI) - FREQ_REF
    return (1 + a1 * u + a2 * u ** 2) / (1 + b1 * u + b2 * u ** 2)


def theta_kpod(k) -> np.ndarray:
    a0, a1, a2, b1, b2 = KPOD_COEF
    kc = np.clip(np.asarray(k, float), KPOD_LO, KPOD_HI)
    return (a0 + a1 * kc + a2 * kc ** 2) / (1 + b1 * kc + b2 * kc ** 2)


def contractor_hr(stratum: str, contractor: str) -> float:
    return CONTRACTOR[stratum][contractor]


def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 4000) -> float:
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


def rmst_ref(stratum: str, horizon: float = RMST_HORIZON) -> float:
    b = BASELINE[stratum]
    return rmst(b["eta0"], b["beta0"], horizon)


@dataclass
class TTF:
    stratum: str
    theta_total: float
    theta_parts: dict
    eta_eff: float
    rmst: float
    rmst_ref: float
    rmst_mult: float
    median: float
    mrl: float


def compose(stratum: str, *, contractor: str = "brt", ql: float = QL_REF,
            freq: float = FREQ_REF, kpod: float = KPOD_REF,
            horizon: float = RMST_HORIZON) -> TTF:
    """The ONE correct combination: multiply θ's, then convert once.

    RMST multipliers are NOT multiplicative — chaining them overstates life.
    """
    b = BASELINE[stratum]; b0, e0 = b["beta0"], b["eta0"]
    parts = {
        "contractor": float(contractor_hr(stratum, contractor)),
        "Ql": float(theta_ql(stratum, ql)),
        "freq": float(theta_freq(freq)),
        "Kpod": float(theta_kpod(kpod)),
    }
    theta = float(np.prod(list(parts.values())))
    eta_eff = e0 * theta ** (-1.0 / b0)
    r0 = rmst(e0, b0, horizon); r = rmst(eta_eff, b0, horizon)
    return TTF(stratum=stratum, theta_total=theta, theta_parts=parts, eta_eff=eta_eff,
               rmst=r, rmst_ref=r0, rmst_mult=r / r0,
               median=eta_eff * np.log(2.0) ** (1.0 / b0),
               mrl=eta_eff * _gamma(1.0 + 1.0 / b0))
