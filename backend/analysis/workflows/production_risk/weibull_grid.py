"""Weibull model grid for one survival stratum: (k1 | k2) x cut x clock.

Extracted from ``scripts/run/production_risk_mc_weibull_grid.py`` so any stratum
(Мирнинский, Vt sour/nonsour, contractor sub-strata) can be swept with the same
likelihood, the same scoring, and the same honesty checks:

* k1 = single Weibull; k2 = two-component mixture (``constrained`` = the EM
  geometry beta2 > max(1, beta1) AND eta2 > 2*eta1; ``free`` releases both).
* cut c = left truncation past day c — drop the early-failure mass, fit
  conditional on survival to c.  Identical likelihood for k1/k2 so the LR test
  against the nested k1 is valid.
* clock = whatever TTE array the caller passes (calendar «Наработка» vs op-days).

Every fit is scored by log-likelihood, max|dS| against the (left-truncated,
right-censored) Kaplan-Meier, and — the standing reporting rule — RMST(0->H)
and MRL(0), with the KM's own RMST as the control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize
from scipy.stats import chi2

LOG_BETA = (np.log(0.05), np.log(8.0))
LOG_ETA = (np.log(0.5), np.log(50_000.0))
LOG_GAP = (np.log(1e-4), np.log(1e4))
LOGIT_W1 = (-8.0, 4.0)
MIN_RATIO = 2.0
RMST_HORIZON = 1080.0


# ── shared likelihood pieces (log space; linear-space clipping gives fake optima) ──
def _logpdf(t, b, e):
    lt = np.log(np.clip(t, 1e-12, None)) - np.log(e)
    return np.log(b) - np.log(e) + (b - 1.0) * lt - np.exp(np.clip(b * lt, -700, 700))


def _logsurv(t, b, e):
    return -np.exp(np.clip(b * (np.log(np.clip(t, 1e-12, None)) - np.log(e)), -700, 700))


def _log_mix(la, lb, w1):
    x = np.log(max(w1, 1e-300)) + la
    y = np.log(max(1.0 - w1, 1e-300)) + lb
    hi = np.maximum(x, y)
    return hi + np.log(np.exp(x - hi) + np.exp(y - hi))


def surv_k2(t, w1, b1, e1, b2, e2):
    t = np.clip(np.asarray(t, float), 1e-12, None)
    return w1 * np.exp(-((t / e1) ** b1)) + (1.0 - w1) * np.exp(-((t / e2) ** b2))


def surv_params(t, p: dict[str, float]) -> np.ndarray:
    return surv_k2(t, p["w1"], p["beta1"], p["eta1"], p["beta2"], p["eta2"])


# ── k1: single truncated Weibull ──
def fit_k1(t, ev, cut, restarts=20):
    m = t > cut
    t, ev = t[m], ev[m]

    def nll(x):
        b, e = np.exp(np.clip(x, -20, 20))
        ll = np.sum(np.where(ev == 1, _logpdf(t, b, e), _logsurv(t, b, e)))
        if cut > 0:
            ll -= len(t) * float(_logsurv(np.array([cut]), b, e)[0])
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for s in range(restarts):
        rng = np.random.default_rng(s)
        r = minimize(nll, [np.log(rng.uniform(0.4, 2.5)), np.log(rng.uniform(max(cut, 1), 900))],
                     method="Nelder-Mead", options=dict(maxiter=12000, fatol=1e-10, xatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    b, e = np.exp(best.x)
    return {"w1": 0.0, "beta1": b, "eta1": e, "beta2": b, "eta2": e,
            "loglik": -float(best.fun), "n": int(len(t)), "events": int(ev.sum())}


# ── k2: mixture, truncated, constrained or free ──
def _unpack(x, constrained):
    w1 = float(1.0 / (1.0 + np.exp(-x[0])))
    b1 = float(np.exp(x[1]))
    e1 = float(np.exp(x[2]))
    b2 = max(1.0, b1) + float(np.exp(x[3])) if constrained else float(np.exp(x[3]))
    gap = float(np.exp(x[4]))
    e2 = e1 * (MIN_RATIO + gap) if constrained else e1 * (1.0 + gap)  # free: eta2 > eta1 only
    return w1, b1, e1, b2, e2


def fit_k2(t, ev, cut, constrained, restarts=50):
    m = t > cut
    t, ev = t[m], ev[m]
    bounds = [LOGIT_W1, LOG_BETA, LOG_ETA, LOG_BETA, LOG_GAP]

    def nll(x):
        w1, b1, e1, b2, e2 = _unpack(x, constrained)
        lf = _log_mix(_logpdf(t, b1, e1), _logpdf(t, b2, e2), w1)
        ls = _log_mix(_logsurv(t, b1, e1), _logsurv(t, b2, e2), w1)
        lsc = 0.0
        if cut > 0:
            lsc = float(_log_mix(_logsurv(np.array([cut]), b1, e1),
                                 _logsurv(np.array([cut]), b2, e2), w1)[0])
        ll = float(np.sum(np.where(ev == 1, lf, ls)) - len(t) * lsc)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for s in range(restarts):
        rng = np.random.default_rng(s)
        x0 = np.array([rng.uniform(-4, 1), np.log(rng.uniform(0.3, 4)),
                       np.log(rng.uniform(max(cut, 0.5), 300)),
                       np.log(rng.uniform(0.3, 3)), np.log(rng.uniform(0.05, 400))])
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    w1, b1, e1, b2, e2 = _unpack(best.x, constrained)
    return {"w1": w1, "beta1": b1, "eta1": e1, "beta2": b2, "eta2": e2,
            "loglik": -float(best.fun), "n": int(len(t)), "events": int(ev.sum())}


# ── scoring ──
def km_curve(t, ev, cut=0.0):
    m = t > cut
    km = KaplanMeierFitter().fit(
        t[m], ev[m], entry=np.full(int(m.sum()), cut) if cut > 0 else None)
    x = km.survival_function_.index.to_numpy(float)
    y = km.survival_function_.iloc[:, 0].to_numpy(float)
    return x, y


def max_ks(t, ev, p, cut):
    """max|dS| between the fitted (cut-conditional) S and the left-truncated KM."""
    x, y = km_curve(t, ev, cut)
    s = surv_params(x, p)
    sc = float(surv_params(np.array([max(cut, 1e-9)]), p)[0]) if cut > 0 else 1.0
    return float(np.max(np.abs(y - s / sc)))


def b50(p):
    grid = np.arange(0.5, 20000, 1.0)
    s = surv_params(grid, p)
    hit = np.nonzero(s <= 0.5)[0]
    return float(grid[hit[0]]) if hit.size else float("nan")


def rmst_model(p, horizon=RMST_HORIZON, cut=0.0):
    """RMST(0->horizon) of the cut-conditional fitted S, by trapezoid."""
    grid = np.linspace(max(cut, 0.0), max(cut, 0.0) + horizon, 4000)
    s = surv_params(grid, p)
    sc = float(surv_params(np.array([max(cut, 1e-9)]), p)[0]) if cut > 0 else 1.0
    return float(np.trapz(s / sc, grid))


def mrl0(p, cut=0.0, upper=20000.0):
    """MRL(0) = integral of the (cut-conditional) S — the honest mean life."""
    grid = np.linspace(max(cut, 0.0), upper, 40000)
    s = surv_params(grid, p)
    sc = float(surv_params(np.array([max(cut, 1e-9)]), p)[0]) if cut > 0 else 1.0
    return float(np.trapz(s / sc, grid))


def rmst_km(t, ev, cut=0.0, horizon=RMST_HORIZON):
    """KM control for RMST(0->horizon): step-function integral, capped at horizon."""
    x, y = km_curve(t, ev, cut)
    x = x - max(cut, 0.0)
    end = min(horizon, float(x.max())) if len(x) else 0.0
    xs = np.concatenate([[0.0], x[(x > 0) & (x <= end)], [end]])
    # step integral: S is right-continuous, carry previous value across each step
    s_prev = np.concatenate([[1.0], y[(x > 0) & (x <= end)]])
    return float(np.sum(np.diff(xs) * s_prev[: len(xs) - 1]))


def honesty_flags(p: dict[str, float], kind: str) -> str:
    """Boundary/degeneracy marks — a flagged k2 is a k1 in disguise or a bound-rider."""
    flags = []
    if kind == "k2":
        if p["w1"] < 0.005 or p["w1"] > 0.95:
            flags.append("w1_degen")
        if p["beta1"] >= 7.9 or p["beta2"] >= 7.9:
            flags.append("beta_bound")
        if p["eta2"] >= 45_000:
            flags.append("eta2_bound")
        if p["eta2"] / max(p["eta1"], 1e-9) <= MIN_RATIO * 1.01:
            flags.append("eta_ratio_floor")
    if p["beta1"] <= 0.055:
        flags.append("beta1_floor")
    return "+".join(flags)


def sweep(t, ev, clock_name: str, cuts=(0.0, 3.0, 7.0, 30.0),
          k1_restarts=20, k2_restarts=50) -> list[dict]:
    """The full grid for one stratum on one clock; k2 is LR-tested against its k1."""
    rows = []
    for cut in cuts:
        k1 = fit_k1(t, ev, cut, restarts=k1_restarts)
        k1.update(kind="k1", regime="-", clock=clock_name, cut=cut,
                  ks=max_ks(t, ev, k1, cut), b50=b50(k1),
                  rmst=rmst_model(k1, cut=cut), mrl0=mrl0(k1, cut=cut),
                  rmst_km=rmst_km(t, ev, cut), flags=honesty_flags(k1, "k1"))
        rows.append(k1)
        for constrained in (True, False):
            k2 = fit_k2(t, ev, cut, constrained, restarts=k2_restarts)
            lr = 2 * (k2["loglik"] - k1["loglik"])
            k2.update(kind="k2", regime="constrained" if constrained else "free",
                      clock=clock_name, cut=cut, ks=max_ks(t, ev, k2, cut), b50=b50(k2),
                      rmst=rmst_model(k2, cut=cut), mrl0=mrl0(k2, cut=cut),
                      rmst_km=rmst_km(t, ev, cut),
                      lr_vs_k1=round(lr, 2), p_lr=round(float(chi2.sf(max(lr, 0), 3)), 4),
                      flags=honesty_flags(k2, "k2"))
            rows.append(k2)
    return rows


GRID_COLUMNS = ["stratum", "clock", "cut", "kind", "regime", "n", "events",
                "w1", "beta1", "eta1", "beta2", "eta2",
                "b50", "rmst", "rmst_km", "mrl0", "loglik", "ks",
                "lr_vs_k1", "p_lr", "flags"]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[[c for c in GRID_COLUMNS if c in df.columns]]
    for c in ("w1", "beta1", "eta1", "beta2", "eta2", "b50", "rmst", "rmst_km",
              "mrl0", "loglik", "ks"):
        if c in df.columns:
            df[c] = df[c].round(3)
    return df
