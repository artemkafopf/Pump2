"""Spike + constant-hazard survival fit for ESP runs.

The empirical hazard (exposure-corrected, `Ya_nonsour_brt`, 440 events) is:

    0-30 d    0.067 /month    <- infant mortality
    30-600 d  0.023-0.028     <- flat
    600-800   0.028
    1100-1500 0.017           <- still flat; NO wear-out arm

A single Weibull cannot express "spike then flat", so its MLE compromises at
beta ~= 0.7 — a curve that decays forever.  That is what produces the unphysical
"RUL grows with age" and an unidentifiable B50 (which swings 40-60% at low event
counts; see the Ya truncation experiment).

This fitter imposes the shape the data actually shows:

    S(t) = w1 * exp(-(t/eta1)^beta1)  +  (1 - w1) * exp(-(t/eta2))
           \_____ early-life component ____/    \__ constant hazard __/

``beta2`` is pinned to 1 (memoryless plateau) and the early component is confined
to the infant window, so the two parts stay interpretable instead of trading off.
The output uses the shipped registry's five columns unchanged — StrataModel.S
evaluates it natively and nothing downstream changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# Early component stays inside the infant-mortality window; beta1 stays in a range
# a CSV/VBA consumer can carry (the unconstrained MLE otherwise runs to ~7e6,
# which is a point mass in disguise).
ETA1_MIN, ETA1_MAX = 0.2, 90.0
BETA1_MIN, BETA1_MAX = 0.3, 10.0
PLATEAU_BETA = 1.0


def _unpack(x: np.ndarray) -> tuple[float, float, float, float]:
    # expit, not 1/(1+exp(-v)): Nelder-Mead walks far into the tails and the naive
    # form overflows there.
    w1 = float(np.clip(expit(x[0]), 1e-6, 0.9))
    beta1 = float(BETA1_MIN + (BETA1_MAX - BETA1_MIN) * expit(x[1]))
    eta1 = float(ETA1_MIN + (ETA1_MAX - ETA1_MIN) * expit(x[2]))
    eta2 = float(np.exp(np.clip(x[3], -50.0, 50.0)))
    return w1, beta1, eta1, eta2


def survival(t, w1: float, beta1: float, eta1: float, eta2: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return w1 * np.exp(-((t / eta1) ** beta1)) + (1.0 - w1) * np.exp(-(t / eta2))


def _pdf(t: np.ndarray, w1: float, beta1: float, eta1: float, eta2: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    f1 = (beta1 / eta1) * ((t / eta1) ** (beta1 - 1.0)) * np.exp(-((t / eta1) ** beta1))
    f2 = (1.0 / eta2) * np.exp(-(t / eta2))
    return w1 * f1 + (1.0 - w1) * f2


def _nll(x: np.ndarray, t: np.ndarray, e: np.ndarray, entry: np.ndarray) -> float:
    w1, beta1, eta1, eta2 = _unpack(x)
    f = np.clip(_pdf(t[e == 1], w1, beta1, eta1, eta2), 1e-300, None)
    s = np.clip(survival(t[e == 0], w1, beta1, eta1, eta2), 1e-300, None)
    # Left truncation: a run observed only from age `entry` onward contributes
    # conditionally on having survived to it — S(entry) divides out.
    trunc = np.clip(survival(entry, w1, beta1, eta1, eta2), 1e-300, None)
    value = -(np.sum(np.log(f)) + np.sum(np.log(s)) - np.sum(np.log(trunc)))
    return value if np.isfinite(value) else 1e12


def fit(tte, event, entry=None, restarts: int = 40, seed: int = 0) -> dict[str, float]:
    """MLE of the spike + constant-hazard model under right-censoring.

    ``entry`` gives each run's age at the start of the observation window (0 when
    it was installed inside it).  Supplying it fits only a recent regime while
    keeping the exposure older pumps contributed to that window — the honest way
    to handle a field whose failure rate is non-stationary (Мирнинский: ~0 events
    2020-2023, then ~0.04/month from 2024).
    """
    t = np.asarray(tte, dtype=float)
    e = np.asarray(event, dtype=int)
    entry = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=float)
    ok = np.isfinite(t) & (t > 0) & (t > entry)
    t, e, entry = t[ok], e[ok], entry[ok]
    if e.sum() < 3:
        raise ValueError("need at least 3 events to fit")

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.array([
            rng.normal(-2.0, 1.5),                       # w1 small
            rng.normal(0.0, 1.5),                        # beta1
            rng.normal(0.0, 1.5),                        # eta1
            np.log(rng.uniform(200.0, 3000.0)),          # eta2
        ])
        res = minimize(_nll, x0, args=(t, e, entry), method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, fatol=1e-10, xatol=1e-8))
        if best is None or res.fun < best.fun:
            best = res

    w1, beta1, eta1, eta2 = _unpack(best.x)
    return {
        "w1": w1,
        "beta1": beta1,
        "eta1": eta1,
        "beta2": PLATEAU_BETA,
        "eta2": eta2,
        "loglik": -float(best.fun),
        "n_runs": int(len(t)),
        "n_failures": int(e.sum()),
    }


def registry_params(fitted: dict[str, float]) -> dict[str, float]:
    """The five columns esp_models.csv carries."""
    return {k: float(fitted[k]) for k in ("w1", "beta1", "eta1", "beta2", "eta2")}


def quantile(p: float, params: dict[str, float], t_max: float = 20000.0) -> float:
    grid = np.arange(0.0, t_max, 1.0)
    s = survival(grid, params["w1"], params["beta1"], params["eta1"], params["eta2"])
    hit = np.nonzero(s <= 1.0 - p)[0]
    return float(grid[hit[0]]) if hit.size else float("nan")


def monthly_hazard(age: float, params: dict[str, float], days: float = 30.4) -> float:
    """Conditional failure probability over `days` given survival to `age`."""
    s0 = survival(age, params["w1"], params["beta1"], params["eta1"], params["eta2"])
    s1 = survival(age + days, params["w1"], params["beta1"], params["eta1"], params["eta2"])
    return float(np.clip(1.0 - s1 / max(s0, 1e-12), 0.0, 1.0))
