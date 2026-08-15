"""Small, dependency-light Kaplan-Meier and Weibull helpers.

Kept local (rather than pulling lifelines) so the simulation core is cheap to
call thousands of times.  ``event == 1`` is a failure (a KM step down); anything
else (workover or still-running) is right-censored and never steps the curve.
"""
from __future__ import annotations

import numpy as np


def kaplan_meier(
    durations: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, survival, n_at_risk) for the KM estimator.

    ``times``/``survival`` start at (0, 1).  Ties at the same duration are
    collapsed into a single step; censored observations at a given time remain
    at risk for the event at that time (standard convention).
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    ok = np.isfinite(t) & (t > 0)
    t, e = t[ok], e[ok]
    if t.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([0.0])

    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    n = t.size

    # group by exact duration (vectorized): d_k events, n_k at risk per unique time
    uniq, first = np.unique(t, return_index=True)
    d = np.add.reduceat((e == 1).astype(float), first)
    n_risk = (n - first).astype(float)

    keep = d > 0  # only failure times step the curve
    dd, nn, tt = d[keep], n_risk[keep], uniq[keep]
    surv = np.cumprod(1.0 - dd / nn)

    times = np.concatenate([[0.0], tt])
    surv = np.concatenate([[1.0], surv])
    at_risk = np.concatenate([[float(n)], nn])
    return times, surv, at_risk


def rmst(times: np.ndarray, surv: np.ndarray, tau: float) -> float:
    """Restricted mean survival time on [0, tau] = area under the KM curve.

    The KM curve is a right-continuous step function; we integrate it up to
    ``tau`` (clamping the last observed level out to tau).
    """
    times = np.asarray(times, dtype=float)
    surv = np.asarray(surv, dtype=float)
    if times.size == 0:
        return 0.0
    # level surv[k] applies on [times[k], times[k+1]); carry the last level to tau
    edges = np.clip(np.concatenate([times, [tau]]), 0.0, tau)
    return float(np.sum(surv * np.diff(edges)))


def km_median(times: np.ndarray, surv: np.ndarray) -> float:
    """First time the KM curve drops to/through 0.5, or NaN if it never does."""
    surv = np.asarray(surv, dtype=float)
    idx = np.where(surv <= 0.5)[0]
    return float(np.asarray(times, dtype=float)[idx[0]]) if idx.size else float("nan")


def weibull_survival(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Weibull survival S(t) = exp(-(t/eta)^beta)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-np.power(np.clip(t, 0.0, None) / eta, beta))
