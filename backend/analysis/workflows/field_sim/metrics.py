"""Fleet-level operational metrics at a snapshot: failure/pull rate and oil loss.

There is no explicit production rate in the simulation, so oil loss is measured
as the **share of well-days the well is not producing** — i.e. downtime while a
pulled pump is being replaced.  Downtime is cause-dependent (unplanned failures
cost more than planned workovers), so a workover programme trades planned
downtime against the unplanned downtime of the failures it prevents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc

from .config import (
    DAYS_PER_YEAR,
    WORKOVER_DETERMINISTIC,
    WORKOVER_NONE,
    WORKOVER_STATISTICAL,
    SimConfig,
)
from .hazard import eta_effective, layers_from_config, split_layers, theta_mixture
from .simulate import commission_days, runs_to_arrays

# Life summaries the eta calculator can invert.
LIFE_RMST = "rmst"
LIFE_MEAN = "mean"      # = MRL(0) = E[T]
LIFE_MEDIAN = "median"
LIFE_KINDS = (LIFE_RMST, LIFE_MEAN, LIFE_MEDIAN)

# Smallest age fed to a hazard: with beta < 1 the hazard diverges at t = 0
# (integrably), and 0**negative would be an inf rather than a large number.
_T_FLOOR = 1e-9


def eta_components(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(probs, etas)`` of the failure-time mixture induced by the hazard layers.

    Every run draws its own covariate bins, so the marginal failure law is a
    finite mixture of Weibulls that share ``beta_fail`` but differ in scale.
    With no layers this is the single component ``([1.0], [eta_fail_days])``,
    which is why every formula below can integrate over it unconditionally.

    Note the mixture is **not** itself Weibull — that heterogeneity (frailty) is
    exactly what biases the pooled fit, and is the point of the layers.

    This is the **single-mode** view (there is no one ``eta`` once several modes
    compete); with ``modes_on`` use :func:`base_cumhaz` / :func:`life_given_theta`.
    """
    probs, theta = theta_mixture(layers_from_config(cfg))
    return probs, np.asarray(eta_effective(cfg.eta_fail_days, cfg.beta_fail, theta), dtype=float)


# ── the failure law: competing modes under a PH multiplier ───────────────────
# Modes are independent, so their cumulative hazards add.  A PH multiplier theta
# scales every mode's hazard by the same factor, which means it factors straight
# out of the sum:
#
#     S(t | theta) = prod_m exp(-(t / (eta_m * theta**(-1/beta_m)))**beta_m)
#                  = exp(-theta * sum_m (t/eta_m)**beta_m)
#                  = exp(-theta * H0(t))
#
# so every quantity below is an integral of exp(-theta*H0) and the one-mode case
# collapses back to the plain Weibull it always was.
def base_cumhaz(t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """``H₀(t) = Σ_m (t/η_m)^β_m`` — the combined cumulative hazard at θ = 1."""
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return sum(np.power(t / eta, beta) for _, beta, eta in cfg.modes())


def base_hazard(t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """``h₀(t) = Σ_m (β_m/η_m)(t/η_m)^(β_m−1)`` — the combined hazard at θ = 1."""
    t = np.maximum(np.asarray(t, dtype=float), _T_FLOOR)
    return sum((beta / eta) * np.power(t / eta, beta - 1.0) for _, beta, eta in cfg.modes())


def survival_given_theta(t: np.ndarray, cfg: SimConfig, theta: float | np.ndarray) -> np.ndarray:
    """``S(t | θ) = exp(−θ·H₀(t))`` of the combined failure law."""
    return np.exp(-np.asarray(theta, dtype=float) * base_cumhaz(t, cfg))


def mixture_survival(t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """S(t) of the marginal failure law (mixture over the hazard-layer bins)."""
    probs, theta = theta_mixture(layers_from_config(cfg))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    H = base_cumhaz(t, cfg)[:, None]
    return np.sum(probs * np.exp(-theta * H), axis=1)


def _component_rates(cfg: SimConfig, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(p_fail, e_run)`` per θ component: P(cycle ends in a failure) and E[run].

    One mode without a statistical workover is closed form (and stays exact for
    the tests that pin it); anything else is a single adaptive quadrature over
    the whole θ vector at once, so the cost does not grow with the number of
    hazard-layer bins.
    """
    theta = np.asarray(theta, dtype=float)
    modes = cfg.modes()
    wo = cfg.workover_mode

    if len(modes) == 1 and wo != WORKOVER_STATISTICAL:
        _, beta, eta0 = modes[0]
        eta = eta0 * theta ** (-1.0 / beta)
        mean_fail = eta * gamma_fn(1.0 + 1.0 / beta)          # E[T_fail]
        if wo == WORKOVER_NONE:
            return np.ones_like(theta), mean_fail
        a = cfg.pm_age_days()                                  # deterministic PM
        A = (a / eta) ** beta
        return 1.0 - np.exp(-A), mean_fail * gammainc(1.0 / beta, A)

    surv = lambda t: survival_given_theta(t, cfg, theta)
    if wo == WORKOVER_NONE:
        e_run = quad_vec(surv, 0.0, np.inf)[0]
        return np.ones_like(theta), e_run
    if wo == WORKOVER_DETERMINISTIC:
        a = cfg.pm_age_days()
        return 1.0 - surv(a), quad_vec(surv, 0.0, a)[0]

    beta_w, eta_w = cfg.beta_wo, cfg.wo_eta_days()
    surv_wo = lambda t: float(np.exp(-((max(t, 0.0) / eta_w) ** beta_w)))
    e_run = quad_vec(lambda t: surv(t) * surv_wo(t), 0.0, np.inf)[0]
    # P(fail) = ∫ f_fail·S_wo, with f_fail(t|θ) = θ·h₀(t)·S(t|θ)
    p_fail = quad_vec(lambda t: theta * base_hazard(t, cfg) * surv(t) * surv_wo(t),
                      0.0, np.inf)[0]
    return p_fail, e_run


def expected_rates(cfg: SimConfig) -> dict:
    """Closed-form steady-state rates implied by the input parameters.

    Renewal-reward theory: a pump *cycle* is one run of length
    ``L = min(T_fail, T_workover)`` followed by cause-dependent downtime ``D``.
    In the long run a well's failure rate = P(fail) / E[cycle], its workover
    rate = P(workover) / E[cycle], and its oil loss = E[D] / E[cycle].  Rates
    are reported per well-year (× 365); the simulated plateau converges to these.

    The two draw schedules have to be aggregated differently.  Per-run layers
    (Ql, frequency) redraw each cycle, so a well's cycles stay i.i.d. and the
    numerator and denominator can simply be averaged over those bins.  The
    per-well frailty does **not** redraw: each well runs its own renewal process
    at its own θ, so the fleet rate is the average of the per-well *ratios*, not
    the ratio of the averages — a bad well contributes more pulls per year
    precisely because its cycles are shorter.
    """
    well_layers, run_layers = split_layers(layers_from_config(cfg))
    p_well, th_well = theta_mixture(well_layers)
    p_run, th_run = theta_mixture(run_layers)

    # every (well, run) combination, then collapse the run axis inside each well
    p_fail_c, e_run_c = _component_rates(cfg, np.outer(th_well, th_run).ravel())
    shape = (th_well.size, th_run.size)
    p_fail_w = np.sum(p_run * np.clip(p_fail_c.reshape(shape), 0.0, 1.0), axis=1)
    e_run_w = np.sum(p_run * e_run_c.reshape(shape), axis=1)

    e_down_w = p_fail_w * cfg.downtime_fail_days + (1.0 - p_fail_w) * cfg.downtime_workover_days
    e_cycle_w = e_run_w + e_down_w

    pulls_w = p_well / e_cycle_w                      # pulls per well-day, per group
    pull_per_day = float(np.sum(pulls_w))
    # pull-weighted fleet averages, so that pull rate = 1 / mean_cycle exactly
    e_cycle = 1.0 / pull_per_day
    e_run = float(np.sum(pulls_w * e_run_w)) * e_cycle
    e_down = float(np.sum(pulls_w * e_down_w)) * e_cycle
    p_fail = float(np.clip(np.sum(pulls_w * p_fail_w) / pull_per_day, 0.0, 1.0))
    yr = DAYS_PER_YEAR
    return {
        "p_fail": p_fail,
        "mean_run_days": float(e_run),
        "mean_cycle_days": float(e_cycle),
        "fail_per_well_year": yr * pull_per_day * p_fail,
        "workover_per_well_year": yr * pull_per_day * (1.0 - p_fail),
        "pull_per_well_year": yr * pull_per_day,
        "oil_loss_pct": 100.0 * e_down / e_cycle,
    }


def weibull_life(beta: float, eta: float, tau: float) -> dict:
    """Life summaries of a single Weibull(beta, eta): mean, median, RMST(0, tau)."""
    mean = float(eta * gamma_fn(1.0 + 1.0 / beta))
    median = float(eta * (np.log(2.0)) ** (1.0 / beta))
    rmst = float(mean * gammainc(1.0 / beta, (tau / eta) ** beta))  # ∫₀^τ S dt
    return {"mean": mean, "mrl0": mean, "median": median, "rmst": rmst, "tau": float(tau)}


def truth_curve_label(cfg: SimConfig) -> str:
    """What the dashed ground-truth S(t) on a KM figure actually is.

    Worth spelling out on the figure: with modes it is a *product* of survivals
    and with layers a *mixture* over θ — neither is the single Weibull the eye
    assumes, and both are what the pooled fit is failing to be.
    """
    modes = cfg.modes()
    law = (f"β={modes[0][1]:g}, η={modes[0][2]:g}d" if len(modes) == 1
           else " × ".join(f"{n} (β={b:g}, η={e:g}d)" for n, b, e in modes))
    kind = "true Weibull" if len(modes) == 1 else "true competing modes"
    if layers_from_config(cfg):
        return f"true mixture — {law} × θ-layers"
    return f"{kind} ({law})"


def _solve_survival(surv, level: float, scale: float) -> float:
    """Age at which ``surv(t)`` first drops to ``level`` (expanding bracket)."""
    hi = max(float(scale), 1.0)
    while surv(hi) > level and hi < 1e12:
        hi *= 2.0
    return float(brentq(lambda t: float(surv(t)) - level, hi * 1e-9, hi))


def life_given_theta(cfg: SimConfig, theta: float = 1.0, tau: float | None = None) -> dict:
    """Life summaries of the failure law at one PH multiplier ``theta``.

    One mode is the closed-form Weibull with ``η·θ^(−1/β)``; competing modes
    have no closed form (a product of Weibull survivals is not Weibull), so the
    mean and RMST are quadratures of S and the median is solved on it.
    """
    tau = cfg.rmst_tau_days if tau is None else float(tau)
    modes = cfg.modes()
    if len(modes) == 1:
        _, beta, eta = modes[0]
        return weibull_life(beta, float(eta * theta ** (-1.0 / beta)), tau)

    surv = lambda t: float(survival_given_theta(t, cfg, theta))
    mean = float(quad(surv, 0.0, np.inf, limit=200)[0])
    rmst = float(quad(surv, 0.0, tau, limit=200)[0])
    median = _solve_survival(surv, 0.5, mean)
    return {"mean": mean, "mrl0": mean, "median": median, "rmst": rmst, "tau": float(tau)}


def eta_equivalent_days(cfg: SimConfig) -> float:
    """Age where the baseline (θ = 1) failure law has S(t) = 1/e.

    That is the defining property of the Weibull scale, so for one mode this is
    exactly ``eta_fail_days``; with competing modes it is the honest stand-in
    the "% of TTF" workover fractions can still be quoted against.
    """
    modes = cfg.modes()
    if len(modes) == 1:
        return float(modes[0][2])
    return _solve_survival(lambda t: float(survival_given_theta(t, cfg, 1.0)),
                           float(np.exp(-1.0)), min(e for _, _, e in modes))


def expected_life(cfg: SimConfig, tau: float | None = None) -> dict:
    """True life summaries of the *failure* law (the target the KM recovers).

    ``mean`` and ``mrl0`` coincide (both are E[T]); ``rmst`` is the restricted
    mean on [0, tau]; ``median`` is where S(t) crosses 0.5.  With hazard layers
    on, the marginal law is a mixture: mean and RMST average linearly over the
    components, but the median has to be solved on the mixture S(t) (the median
    of a mixture is not the mixture of the medians).
    """
    tau = cfg.rmst_tau_days if tau is None else float(tau)
    probs, thetas = theta_mixture(layers_from_config(cfg))
    if probs.size == 1:
        return life_given_theta(cfg, float(thetas[0]), tau)

    parts = [life_given_theta(cfg, float(th), tau) for th in thetas]
    mean = float(np.sum(probs * np.array([p["mean"] for p in parts])))
    rmst = float(np.sum(probs * np.array([p["rmst"] for p in parts])))
    median = _solve_survival(lambda t: float(mixture_survival(np.atleast_1d(t), cfg)[0]),
                             0.5, mean)
    return {"mean": mean, "mrl0": mean, "median": median, "rmst": rmst, "tau": float(tau)}


def eta_from_life(value: float, kind: str, beta: float, tau: float | None = None) -> float:
    """Invert a life summary of a *single* Weibull back to its scale ``eta``.

    ``mean``/``median`` are closed form (both are ``eta`` times a constant that
    depends only on ``beta``).  ``rmst`` is monotone increasing in ``eta`` but
    has no closed inverse, so it is bracketed and solved.  This is the "what
    eta do I type to get a 400-day RMST?" helper behind the app's calculator; it
    describes the *baseline* law, before any hazard layer redistributes it.
    """
    if value <= 0:
        raise ValueError("life summary must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    g = float(gamma_fn(1.0 + 1.0 / beta))
    if kind == LIFE_MEAN:
        return value / g
    if kind == LIFE_MEDIAN:
        return value / float(np.log(2.0) ** (1.0 / beta))
    if kind != LIFE_RMST:
        raise ValueError(f"kind must be one of {LIFE_KINDS}")
    if tau is None or tau <= 0:
        raise ValueError("rmst inversion needs a positive tau")
    if value >= tau:
        raise ValueError(f"RMST(0, τ) < τ always — {value:g} is not reachable with τ = {tau:g}")

    f = lambda eta: weibull_life(beta, eta, tau)["rmst"] - value
    # RMST(0,τ) ≤ E[T] = eta·g, so the mean-matched eta is always a lower bound;
    # RMST → τ > value as eta → ∞, so growing from there brackets the root.
    lo, hi = value / g, (value / g) * 1.6
    while f(hi) < 0 and hi < 1e12:
        lo, hi = hi, hi * 1.6
    return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200))


def snapshot_metrics_arrays(arr: dict, cdays: np.ndarray, snap_days: float) -> dict:
    """Numpy core of :func:`snapshot_metrics` (hot-loop fast path).

    ``arr`` comes from :func:`runs_to_arrays`; ``cdays`` from
    :func:`commission_days` (computed once per experiment, not per snapshot).
    """
    start, end = arr["start"], arr["end"]
    started = start <= snap_days

    # calendar well-days: from each commissioned slot's start to the snapshot
    well_days = float(np.sum(np.clip(snap_days - cdays[cdays <= snap_days], 0.0, None)))

    # producing pump-days (time pumps were actually running, up to the snapshot)
    end_clip = np.minimum(end[started], snap_days)
    exposure_days = float(np.sum(np.clip(end_clip - start[started], 0.0, None)))

    # downtime after every already-pulled run, clipped to the snapshot
    completed = started & (end <= snap_days)
    dt = np.clip(np.minimum(arr["downtime_after"][completed], snap_days - end[completed]), 0.0, None)
    cf = arr["is_fail"][completed]
    downtime_days = float(dt.sum())

    n_fail = int(np.sum(cf))
    n_workover = int(np.sum(~cf))
    well_years = well_days / DAYS_PER_YEAR
    nan = float("nan")
    return {
        "well_days": well_days,
        "exposure_days": exposure_days,
        "downtime_days": downtime_days,
        "downtime_fail_days": float(dt[cf].sum()),
        "downtime_wo_days": float(dt[~cf].sum()),
        "oil_loss_pct": 100.0 * downtime_days / well_days if well_days > 0 else nan,
        "fail_rate_well_yr": n_fail / well_years if well_years > 0 else nan,
        "pull_rate_well_yr": (n_fail + n_workover) / well_years if well_years > 0 else nan,
    }


def snapshot_metrics(runs: pd.DataFrame, cfg: SimConfig, snap_days: float) -> dict:
    """Exposure, downtime, oil-loss %, and failure/pull rates at ``snap_days``."""
    return snapshot_metrics_arrays(runs_to_arrays(runs), commission_days(cfg), snap_days)
