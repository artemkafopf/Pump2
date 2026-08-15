"""Fit Weibull parameters to a snapshot three ways, and build its KM curve.

- ``censored``: MLE that treats workovers + the still-running current run as
  right-censored (the correct, cause-specific estimator).
- ``failures_only``: drops every censored record and fits the failure durations
  as if complete (the biased estimator the experiment exposes).
- ``all-pulls``: every completed run (failure OR workover) counts as an event.

All three use :func:`_weibull_mle`, a fast profiled 1-D MLE: for a Weibull the
scale ``eta`` has a closed form given ``beta``, so the estimator profiles it out
and solves a single 1-D score equation for ``beta`` (unimodal, so no 2-D search
or multi-start).  This is the exact MLE, ~1-2 orders of magnitude cheaper than a
2-D L-BFGS-B optimisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .km import kaplan_meier, km_median, rmst

# A fit needs at least this many observed failures to be even weakly identified;
# below it, eta runs off to infinity (a flat likelihood in the low-hazard limit).
MIN_FAIL_FOR_FIT = 3
# Any eta beyond ~270 years on a 20-year sim is a degenerate / non-identified fit.
ETA_DEGENERATE_DAYS = 1.0e5


def _clean_fit(res: dict) -> tuple[float, float]:
    beta, eta = float(res["beta"]), float(res["eta"])
    if not res.get("success", True) or not np.isfinite(eta) or eta > ETA_DEGENERATE_DAYS:
        return np.nan, np.nan
    return beta, eta


def _weibull_mle(durations: np.ndarray, events: np.ndarray) -> dict:
    """Right-censored Weibull MLE via the profiled 1-D score equation.

    Given ``beta``, the scale MLE is ``eta**beta = sum_all(t**beta) / d`` where
    ``d`` is the event count.  Substituting leaves the score in ``beta``:

        1/beta + mean_event(log t) - sum(t**beta * log t) / sum(t**beta) = 0

    which is monotone decreasing in ``beta`` and has a unique root.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    keep = np.isfinite(t) & (t > 0)
    t, e = t[keep], e[keep]
    d = int(np.sum(e == 1))
    if d < 1 or t.size < 2:
        return {"beta": np.nan, "eta": np.nan, "success": False}

    logt = np.log(t)
    mean_event_logt = float(np.sum(logt[e == 1]) / d)
    lmax = float(logt.max())  # factor out to keep t**beta from overflowing

    def score(beta: float) -> float:
        w = np.exp(beta * (logt - lmax))          # ∝ t**beta, numerically stable
        return 1.0 / beta + mean_event_logt - float(np.sum(w * logt) / np.sum(w))

    lo, hi = 1e-3, 1.0e2
    g_lo, g_hi = score(lo), score(hi)
    tries = 0
    while g_lo * g_hi > 0 and tries < 12:  # widen the bracket if needed
        if g_hi > 0:
            hi *= 2.0
            g_hi = score(hi)
        else:
            lo *= 0.5
            g_lo = score(lo)
        tries += 1
    if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_lo * g_hi > 0:
        return {"beta": np.nan, "eta": np.nan, "success": False}

    beta = float(brentq(score, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=200))
    w = np.exp(beta * (logt - lmax))
    eta = float(np.exp(lmax) * (np.sum(w) / d) ** (1.0 / beta))
    return {"beta": beta, "eta": eta, "success": True}


def fit_snapshot_arrays(dur: np.ndarray, ended: np.ndarray, is_fail: np.ndarray,
                        rmst_tau: float = 730.0, fail_mode: np.ndarray | None = None,
                        n_modes: int = 1) -> dict:
    """Core of :func:`fit_snapshot` on plain arrays (the hot-loop fast path).

    ``ended`` marks runs that have been pulled by the snapshot; ``is_fail`` is the
    run's cause (only consulted where ``ended``).  Avoids any pandas / string ops.

    With competing failure modes, ``fail_mode`` carries which one ended each run
    and the fit is repeated **cause-specifically per mode** (``beta_m0``,
    ``eta_m0``, …): that mode's failures are events, and every other mode's
    failure is just another censoring.  That is the estimator that recovers each
    node's own β/η — pooling them instead returns the shape of the *minimum*,
    which belongs to no single node.
    """
    dur = np.asarray(dur, dtype=float)
    ended = np.asarray(ended, dtype=bool)
    is_fail = np.asarray(is_fail, dtype=bool)
    keep = np.isfinite(dur) & (dur > 0)
    dur, ended, is_fail = dur[keep], ended[keep], is_fail[keep]
    if fail_mode is not None:
        fail_mode = np.asarray(fail_mode, dtype=int)[keep]

    ev = ended & is_fail       # failure events
    ev_all = ended             # any pull (failure or workover)
    n_runs = int(dur.size)
    n_fail = int(np.sum(ev))
    n_all = int(np.sum(ev_all))

    out: dict = {
        "n_runs": n_runs,
        "n_fail": n_fail,
        "n_workover": int(np.sum(ended & ~is_fail)),
        "n_running": int(np.sum(~ended)),
        "beta_cens": np.nan, "eta_cens": np.nan,
        "beta_fo": np.nan, "eta_fo": np.nan,
        "beta_all": np.nan, "eta_all": np.nan,
        # naive observed mean time (Σt/N) — biased under censoring
        "obs_ttf_fail": float(np.mean(dur[ev])) if n_fail else np.nan,
        "obs_ttf_all": float(np.mean(dur[ev_all])) if n_all else np.nan,
        # censoring-corrected KM life summaries (of the failure distribution)
        "life_rmst": np.nan, "life_mrl0": np.nan, "life_median": np.nan,
    }
    if n_modes > 1:
        for m in range(int(n_modes)):
            out[f"n_fail_m{m}"] = 0
            out[f"beta_m{m}"], out[f"eta_m{m}"] = np.nan, np.nan
    if n_runs < 2 or n_fail < MIN_FAIL_FOR_FIT:
        return out

    ev_i = ev.astype(int)
    # (1) censoring-aware, cause-specific: only true failures are events.
    out["beta_cens"], out["eta_cens"] = _clean_fit(_weibull_mle(dur, ev_i))

    # (2) failures-only: drop every censored run, fit failures as complete.
    dfail = dur[ev]
    if dfail.size >= MIN_FAIL_FOR_FIT:
        out["beta_fo"], out["eta_fo"] = _clean_fit(_weibull_mle(dfail, np.ones_like(dfail, dtype=int)))

    # (3) all-pulls: every completed run (failure OR workover) is an event; only
    #     the still-running pump is censored.  The all-cause pull-time Weibull.
    if n_all >= MIN_FAIL_FOR_FIT:
        out["beta_all"], out["eta_all"] = _clean_fit(_weibull_mle(dur, ev_all.astype(int)))

    # KM life summaries from the cause-specific (failure) KM.
    km_t, km_s, _ = kaplan_meier(dur, ev_i)
    out["life_rmst"] = rmst(km_t, km_s, rmst_tau)           # RMST(0, tau)
    out["life_mrl0"] = rmst(km_t, km_s, float(km_t[-1]))    # MRL(0) = full KM area
    out["life_median"] = km_median(km_t, km_s)              # KM median

    # (4) one cause-specific fit per competing failure mode: only this mode's
    #     failures are events, every other pull (other modes, WO, running) is
    #     censored — the estimator that gives each node back its own β/η.
    if n_modes > 1 and fail_mode is not None:
        for m in range(int(n_modes)):
            ev_m = ev & (fail_mode == m)
            out[f"n_fail_m{m}"] = int(np.sum(ev_m))
            if int(np.sum(ev_m)) >= MIN_FAIL_FOR_FIT:
                out[f"beta_m{m}"], out[f"eta_m{m}"] = _clean_fit(
                    _weibull_mle(dur, ev_m.astype(int)))
    return out


def fit_snapshot(observation: pd.DataFrame, eta_hint: float | None = None,
                 rmst_tau: float = 730.0, n_modes: int = 1) -> dict:
    """Fit the three Weibulls + observed TTF + KM life summaries for a snapshot."""
    status = observation["status"].to_numpy()
    modes = observation["fail_mode"].to_numpy(dtype=int) if "fail_mode" in observation else None
    return fit_snapshot_arrays(
        observation["duration"].to_numpy(dtype=float),
        status != "running",
        status == "fail",
        rmst_tau=rmst_tau,
        fail_mode=modes,
        n_modes=n_modes,
    )


def bin_life_arrays(dur: np.ndarray, ended: np.ndarray, is_fail: np.ndarray,
                    bin_idx: np.ndarray, n_bins: int, rmst_tau: float = 730.0) -> list[dict]:
    """Per-covariate-bin life summaries at one snapshot.

    For each hazard-layer bin: how many runs sit in it, how many have failed,
    the **naive observed mean and median TTF** of those failures (over failures
    only — the quantities a per-bin pivot table in Excel would give you), and
    the censoring-corrected KM RMST(0, τ) and median of the same bin.

    The gap between the two families is the point: the naive per-bin summaries
    are pulled down by censoring and can flatten or even invert a real covariate
    effect, while RMST recovers the ordering the θ layer actually put in.  Mean
    and median are both reported because they are biased by *different* amounts
    — the median is the more robust of the two but still censoring-blind.
    """
    dur = np.asarray(dur, dtype=float)
    ended = np.asarray(ended, dtype=bool)
    is_fail = np.asarray(is_fail, dtype=bool)
    bin_idx = np.asarray(bin_idx, dtype=int)
    ev = ended & is_fail

    rows: list[dict] = []
    for b in range(int(n_bins)):
        m = bin_idx == b
        d_b, ev_b = dur[m], ev[m]
        n_fail = int(np.sum(ev_b))
        row = {
            "bin": b,
            "n_runs": int(d_b.size),
            "n_fail": n_fail,
            "obs_ttf_fail": float(np.mean(d_b[ev_b])) if n_fail else np.nan,
            "obs_ttf_median": float(np.median(d_b[ev_b])) if n_fail else np.nan,
            "km_rmst": np.nan,
            "km_median": np.nan,
        }
        if n_fail >= 1:
            km_t, km_s, _ = kaplan_meier(d_b, ev_b.astype(int))
            row["km_rmst"] = rmst(km_t, km_s, rmst_tau)
            row["km_median"] = km_median(km_t, km_s)
        rows.append(row)
    return rows


def km_curve_arrays(dur: np.ndarray, ev: np.ndarray) -> dict:
    """KM survival curve arrays from plain (duration, failure-event) arrays."""
    dur = np.asarray(dur, dtype=float)
    ev = np.asarray(ev, dtype=int)
    t, s, n = kaplan_meier(dur, ev)
    return {
        "time": t.tolist(),
        "surv": s.tolist(),
        "n_at_risk": n.tolist(),
        "n_runs": int(np.sum(np.isfinite(dur) & (dur > 0))),
        "n_fail": int(np.sum(ev == 1)),
    }


def km_curve_failures_only(dur: np.ndarray, ev: np.ndarray) -> dict:
    """Survival curve of the failure durations with every censored run dropped.

    With no censored records left there is nothing for the KM to correct, so it
    collapses to ``1 − ECDF`` of the observed failures.  That is the point of
    plotting it next to the censoring-aware curve: the estimator is not broken,
    the *sample* is — long-lived runs are still alive and never enter it, so the
    curve falls too fast and reads as a shorter-lived fleet than the truth.
    """
    dur = np.asarray(dur, dtype=float)[np.asarray(ev, dtype=int) == 1]
    return km_curve_arrays(dur, np.ones(dur.size, dtype=int))


def km_curve(observation: pd.DataFrame) -> dict:
    """KM survival curve arrays for one (possibly pooled) observation."""
    return km_curve_arrays(
        observation["duration"].to_numpy(dtype=float),
        observation["event"].to_numpy(dtype=int),
    )
