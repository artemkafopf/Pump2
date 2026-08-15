"""Shared gamma-frailty Weibull-PH — how much of the spread in life is *the well*?

Every run-level survival fit in this repo treats runs as independent.  They are not: a
well contributes many runs (Ya averages ~4.5), and if wells differ systematically then the
nominal sample size overstates the information by the design effect

    deff = 1 + (m_eff - 1) * rho

and every standard error is understated by ``sqrt(deff)``.  This module measures ``rho``
so that inflation can be quantified rather than assumed.

Model — cluster ``i`` (a well), run ``j``::

    h(t | x, Z_i) = Z_i * (b/eta) * (t/eta)^(b-1) * exp(x'beta)
    Z_i ~ Gamma(mean 1, variance theta)   shared by every run in the well

The gamma is conjugate to the cumulative hazard, so the frailty integrates out in closed
form — no quadrature, no Monte Carlo.  With ``a = 1/theta`` and ``D_i`` events in well
``i``::

    log L_i = sum_j d_ij * (log h0(t_ij) + x_ij'beta)
              + a*log(a) + lgamma(a + D_i) - lgamma(a) - (a + D_i)*log(a + H_i)

where ``H_i = sum_j H0(t_ij) * exp(x_ij'beta)`` is the well's total cumulative hazard.

Reported dependence measures:

* ``theta``   — frailty variance (0 = runs independent).
* ``rho``     — intra-well correlation of **log** lifetime.  For a Weibull AFT,
  ``log T = log eta + (log E - log Z)/b`` with ``E ~ Exp(1)``, so the ``1/b^2`` cancels and
  ``rho = trigamma(1/theta) / (trigamma(1/theta) + pi^2/6)`` — a function of ``theta``
  alone.  This is the ``rho`` that belongs in the design-effect formula.
* ``kendall_tau`` — ``theta/(theta + 2)``, the concordance between two runs in one well.

``theta = 0`` sits on the boundary of the parameter space, so the likelihood-ratio test
against the independent fit uses the ``0.5*chi2_0 + 0.5*chi2_1`` mixture null.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, polygamma
from scipy.stats import chi2

__all__ = ["FrailtyFit", "fit_shared_frailty", "design_effect", "effective_cluster_size"]

# theta below this is numerically indistinguishable from independence (a = 1/theta
# overflows lgamma's useful range long before the likelihood moves).
_THETA_FLOOR = 1e-6
_LOG_THETA_BOUNDS = (np.log(_THETA_FLOOR), np.log(50.0))


def effective_cluster_size(sizes: np.ndarray) -> float:
    """Mean cluster size that gives the right design effect when clusters are unequal.

    ``sum(m^2)/sum(m)``, not ``mean(m)`` — a few very large wells inflate the variance more
    than the arithmetic mean implies.
    """
    m = np.asarray(sizes, dtype=float)
    if m.sum() <= 0:
        return 0.0
    return float((m ** 2).sum() / m.sum())


def design_effect(sizes: np.ndarray, rho: float) -> float:
    """``1 + (m_eff - 1) * rho`` — the factor by which the naive variance is too small."""
    return float(1.0 + (effective_cluster_size(sizes) - 1.0) * rho)


@dataclass
class FrailtyFit:
    """Result of one shared-frailty fit, with the independent fit kept for comparison."""

    theta: float
    rho: float
    kendall_tau: float
    shape: float
    scale: float
    coefs: dict[str, float] = _field(default_factory=dict)
    loglik: float = np.nan
    loglik_indep: float = np.nan
    coefs_indep: dict[str, float] = _field(default_factory=dict)
    shape_indep: float = np.nan
    n_runs: int = 0
    n_events: int = 0
    n_clusters: int = 0
    cluster_sizes: np.ndarray = _field(default_factory=lambda: np.array([]))
    converged: bool = False

    @property
    def lr_stat(self) -> float:
        return float(2.0 * (self.loglik - self.loglik_indep))

    @property
    def lr_pvalue(self) -> float:
        """Boundary-corrected: theta = 0 lies on the edge, so the null is a 50:50 mixture."""
        stat = self.lr_stat
        if stat <= 0:
            return 1.0
        return float(0.5 * chi2.sf(stat, 1))

    @property
    def m_eff(self) -> float:
        return effective_cluster_size(self.cluster_sizes)

    @property
    def design_effect(self) -> float:
        return design_effect(self.cluster_sizes, self.rho)

    @property
    def se_inflation(self) -> float:
        """Multiply a naive standard error by this to get the cluster-aware one."""
        return float(np.sqrt(self.design_effect))

    @property
    def n_effective(self) -> float:
        return float(self.n_runs / self.design_effect) if self.design_effect > 0 else np.nan

    def summary(self) -> pd.DataFrame:
        rows = [
            ("n_runs", self.n_runs),
            ("n_events", self.n_events),
            ("n_clusters", self.n_clusters),
            ("mean_cluster_size", self.n_runs / max(self.n_clusters, 1)),
            ("m_eff", self.m_eff),
            ("theta_frailty_var", self.theta),
            ("rho_icc_logT", self.rho),
            ("kendall_tau", self.kendall_tau),
            ("design_effect", self.design_effect),
            ("se_inflation", self.se_inflation),
            ("n_effective", self.n_effective),
            ("weibull_shape", self.shape),
            ("weibull_shape_indep", self.shape_indep),
            ("weibull_scale", self.scale),
            ("loglik", self.loglik),
            ("loglik_indep", self.loglik_indep),
            ("lr_stat", self.lr_stat),
            ("lr_pvalue", self.lr_pvalue),
            ("converged", self.converged),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])


def _rho_from_theta(theta: float) -> float:
    """ICC of log lifetime.  ``trigamma(1/theta)`` is the between-well variance of log Z."""
    if theta <= _THETA_FLOOR:
        return 0.0
    var_log_z = float(polygamma(1, 1.0 / theta))
    return var_log_z / (var_log_z + np.pi ** 2 / 6.0)


def _neg_loglik(params, t, d, X, cluster_idx, n_clusters, with_frailty: bool) -> float:
    log_shape, log_scale = params[0], params[1]
    if with_frailty:
        log_theta = params[2]
        betas = params[3:]
    else:
        log_theta = None
        betas = params[2:]

    shape = np.exp(log_shape)
    scale = np.exp(log_scale)

    lp = X @ betas if X.shape[1] else np.zeros(len(t))
    lp = np.clip(lp, -30.0, 30.0)

    # Weibull baseline on the run's own clock.
    z = t / scale
    log_h0 = np.log(shape) - log_scale + (shape - 1.0) * np.log(z)
    H = (z ** shape) * np.exp(lp)

    event_part = float(np.sum(d * (log_h0 + lp)))

    H_i = np.bincount(cluster_idx, weights=H, minlength=n_clusters)
    D_i = np.bincount(cluster_idx, weights=d, minlength=n_clusters)

    if not with_frailty:
        return -(event_part - float(H_i.sum()))

    theta = np.exp(log_theta)
    a = 1.0 / theta
    # a*log(a) + lgamma(a + D) - lgamma(a) - (a + D)*log(a + H)
    cluster_part = (
        n_clusters * a * np.log(a)
        + np.sum(gammaln(a + D_i))
        - n_clusters * gammaln(a)
        - np.sum((a + D_i) * np.log(a + H_i))
    )
    val = event_part + float(cluster_part)
    return -val if np.isfinite(val) else 1e12


def fit_shared_frailty(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    cluster_col: str,
    covariates: list[str] | None = None,
    min_time: float = 1.0,
) -> FrailtyFit:
    """Fit the shared-frailty Weibull and its independent counterpart on the same rows.

    ``covariates`` are entered linearly on the log-hazard; pass ``None`` for the pure
    variance-components question (how much does the well alone explain?).
    """
    covariates = list(covariates or [])
    cols = [duration_col, event_col, cluster_col] + covariates
    d = df[cols].dropna().copy()
    d = d[d[duration_col] > 0].copy()
    d[duration_col] = d[duration_col].clip(lower=min_time)

    t = d[duration_col].to_numpy(dtype=float)
    ev = d[event_col].to_numpy(dtype=float)
    codes, uniques = pd.factorize(d[cluster_col])
    cluster_idx = codes.astype(int)
    n_clusters = len(uniques)

    X = d[covariates].to_numpy(dtype=float) if covariates else np.zeros((len(d), 0))
    if X.shape[1]:  # standardise so one optimiser scale fits every covariate
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

    sizes = np.bincount(cluster_idx, minlength=n_clusters).astype(float)

    x0_base = [np.log(1.0), np.log(max(np.median(t), 1.0))] + [0.0] * X.shape[1]
    bounds_base = [(np.log(0.05), np.log(10.0)), (np.log(1.0), np.log(1e6))]
    bounds_cov = [(-10.0, 10.0)] * X.shape[1]

    r0 = minimize(
        _neg_loglik,
        np.array(x0_base),
        args=(t, ev, X, cluster_idx, n_clusters, False),
        method="L-BFGS-B",
        bounds=bounds_base + bounds_cov,
    )
    ll_indep = -float(r0.fun)
    shape_indep = float(np.exp(r0.x[0]))
    coefs_indep = dict(zip(covariates, r0.x[2:])) if covariates else {}

    # Multi-start over theta: the frailty variance is the parameter the likelihood is
    # flattest in, and a single start lands on the wrong side often enough to matter.
    best = None
    for theta0 in (0.05, 0.25, 0.75, 1.5, 3.0):
        x0 = [r0.x[0], r0.x[1], np.log(theta0)] + list(r0.x[2:])
        r = minimize(
            _neg_loglik,
            np.array(x0),
            args=(t, ev, X, cluster_idx, n_clusters, True),
            method="L-BFGS-B",
            bounds=bounds_base[:2] + [_LOG_THETA_BOUNDS] + bounds_cov,
        )
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r

    theta = float(np.exp(best.x[2]))
    return FrailtyFit(
        theta=theta,
        rho=_rho_from_theta(theta),
        kendall_tau=theta / (theta + 2.0),
        shape=float(np.exp(best.x[0])),
        scale=float(np.exp(best.x[1])),
        coefs=dict(zip(covariates, best.x[3:])) if covariates else {},
        loglik=-float(best.fun),
        loglik_indep=ll_indep,
        coefs_indep=coefs_indep,
        shape_indep=shape_indep,
        n_runs=len(d),
        n_events=int(ev.sum()),
        n_clusters=n_clusters,
        cluster_sizes=sizes,
        converged=bool(best.success),
    )
