"""Extended Cox model with time-varying coefficient via log-log scale regression.

Model specification (single binary covariate X):
    log h(t|X) = log h₀(t) + β·X + γ·X·log(t)

For Weibull baseline h₀(t) = (β₀/η₀)(t/η₀)^(β₀−1):

    log H(t|X=1) − log H₀(t) = β* + γ·log(t)

where β* = β + log[β₀/(β₀+γ)] is the corrected intercept.

Estimation: OLS on the observable difference y(t) = log(−log S(t|X=1)) − log(−log S₀(t))
regressed on log(t).  The slope is γ; the intercept is β*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExtendedCoxResult:
    gamma: float
    beta_star: float
    beta: float
    beta0: float
    se_gamma: float
    se_beta_star: float
    cov_matrix: np.ndarray
    p_gamma: float
    p_beta_star: float
    r_squared: float
    n_points: int
    times: np.ndarray = field(repr=False)
    y_observed: np.ndarray = field(repr=False)
    y_fitted: np.ndarray = field(repr=False)

    @property
    def hr_at_t1(self) -> float:
        """Instantaneous HR at t=1 day (exp(β))."""
        return float(np.exp(self.beta))

    def hr_profile(self, times: np.ndarray) -> np.ndarray:
        """Time-varying HR(t) = exp(β + γ·log(t)) = exp(β*)·t^γ / correction."""
        return np.exp(self.beta + self.gamma * np.log(np.clip(times, 1e-9, None)))

    def hr_ci(self, times: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        95 % CI for log HR(t) via delta method on the regression parameters.

        log HR(t) ≈ β* + γ·log(t)  (ignoring the small Weibull correction)
        Var[β* + γ·log(t)] = Var[β*] + log(t)²·Var[γ] + 2·log(t)·Cov[β*, γ]
        """
        log_t = np.log(np.clip(times, 1e-9, None))
        z = stats.norm.ppf(1 - alpha / 2)
        se_log_hr = np.sqrt(
            self.cov_matrix[0, 0]
            + log_t ** 2 * self.cov_matrix[1, 1]
            + 2 * log_t * self.cov_matrix[0, 1]
        )
        log_hr_centre = self.beta_star + self.gamma * log_t
        return (
            np.exp(log_hr_centre - z * se_log_hr),
            np.exp(log_hr_centre + z * se_log_hr),
        )

    def summary_dict(self) -> dict:
        return {
            "gamma": round(self.gamma, 4),
            "se_gamma": round(self.se_gamma, 4),
            "p_gamma": round(self.p_gamma, 4),
            "beta_star": round(self.beta_star, 4),
            "se_beta_star": round(self.se_beta_star, 4),
            "p_beta_star": round(self.p_beta_star, 4),
            "beta": round(self.beta, 4),
            "hr_t1": round(self.hr_at_t1, 4),
            "beta0_used": round(self.beta0, 4),
            "r_squared": round(self.r_squared, 4),
            "n_points": self.n_points,
        }


def _km_loglog_at(km_sf: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """
    Step-interpolate log(−log S(t)) from a KM survival function DataFrame
    (index = timeline, single column = survival probability) at requested times.
    Returns NaN where S=0 or S=1.
    """
    timeline = km_sf.index.values
    surv = km_sf.iloc[:, 0].values
    result = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        mask = timeline <= t
        if not mask.any():
            continue
        s = float(surv[mask][-1])
        if s <= 0.0 or s >= 1.0:
            result[i] = np.nan
        else:
            result[i] = np.log(-np.log(s))
    return result


def fit_extended_cox_log_log(
    km_baseline_sf: pd.DataFrame,
    km_treatment_sf: pd.DataFrame,
    baseline_weibull_beta: float | None = None,
    min_time: float = 1.0,
    trim_quantile: float = 0.95,
) -> ExtendedCoxResult:
    """
    Fit Extended Cox model via log-log OLS regression.

    Parameters
    ----------
    km_baseline_sf : KM survival function DataFrame for the X=0 group
                     (lifelines km.survival_function_)
    km_treatment_sf : KM survival function DataFrame for the X=1 group
    baseline_weibull_beta : fitted Weibull shape β₀ for the baseline group.
                            Used only for the correction β = β* − log[β₀/(β₀+γ)].
                            If None, correction is skipped (β = β*).
    min_time : drop event times below this value (default 1 day)
    trim_quantile : trim extreme time quantile from treatment timeline to
                    avoid instability where KM has few at risk (default 0.95)
    """
    # --- collect event times from treatment KM ---
    tl = km_treatment_sf.index.values
    tl = tl[tl >= min_time]
    if len(tl) > 1:
        t_max = float(np.quantile(tl, trim_quantile))
        tl = tl[tl <= t_max]

    # --- compute y(t) at each treatment event time ---
    ll1 = _km_loglog_at(km_treatment_sf, tl)
    ll0 = _km_loglog_at(km_baseline_sf, tl)
    y = ll1 - ll0
    log_t = np.log(tl)

    valid = np.isfinite(y) & np.isfinite(log_t)
    y_v = y[valid]
    lt_v = log_t[valid]
    t_v = tl[valid]

    if len(y_v) < 4:
        raise ValueError(
            f"Too few valid time-points for regression (n={len(y_v)}). "
            "Check that both groups have enough failures."
        )

    # --- OLS: y = β* + γ·log(t) ---
    X = np.column_stack([np.ones(len(y_v)), lt_v])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
    beta_star = float(coeffs[0])
    gamma = float(coeffs[1])

    y_fit = X @ coeffs
    residuals = y_v - y_fit
    n = len(y_v)
    sigma2 = float(np.sum(residuals ** 2) / (n - 2))
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * XtX_inv
    se_beta_star = float(np.sqrt(cov[0, 0]))
    se_gamma = float(np.sqrt(cov[1, 1]))

    t_bs = beta_star / se_beta_star if se_beta_star > 0 else 0.0
    t_g = gamma / se_gamma if se_gamma > 0 else 0.0
    p_beta_star = float(2 * (1 - stats.t.cdf(abs(t_bs), df=n - 2)))
    p_gamma = float(2 * (1 - stats.t.cdf(abs(t_g), df=n - 2)))

    ss_tot = float(np.sum((y_v - y_v.mean()) ** 2))
    r_squared = float(1 - np.sum(residuals ** 2) / ss_tot) if ss_tot > 1e-12 else 0.0

    # --- Weibull correction ---
    beta0 = float(baseline_weibull_beta) if baseline_weibull_beta is not None else 1.0
    denom = beta0 + gamma
    correction = float(np.log(beta0 / denom)) if denom > 0 else 0.0
    beta = beta_star - correction

    return ExtendedCoxResult(
        gamma=gamma,
        beta_star=beta_star,
        beta=beta,
        beta0=beta0,
        se_gamma=se_gamma,
        se_beta_star=se_beta_star,
        cov_matrix=cov,
        p_gamma=p_gamma,
        p_beta_star=p_beta_star,
        r_squared=r_squared,
        n_points=n,
        times=t_v,
        y_observed=y_v,
        y_fitted=y_fit,
    )


def fit_extended_cox_on_survival_grid(
    times: np.ndarray,
    s_lo: np.ndarray,
    s_hi: np.ndarray,
    min_time: float = 5.0,
    trim_quantile: float = 0.97,
) -> ExtendedCoxResult:
    """
    Fit Extended Cox from pre-evaluated survival arrays on a common time grid.

    y(t) = log(−log S_hi(t)) − log(−log S_lo(t)) = β* + γ·log(t)

    Use when both survival functions are already evaluated on a fine grid
    (e.g. from a latent Weibull mixture) rather than from KM step functions.
    Avoids the pseudo-KM DataFrame indirection of fit_extended_cox_log_log.

    Parameters
    ----------
    times       : 1-D float array of time points (days)
    s_lo        : survival probabilities for the reference (low) group at each time
    s_hi        : survival probabilities for the treatment (high) group at each time
    min_time    : exclude time points below this value
    trim_quantile : exclude time points above this quantile of `times`
    """
    t = np.asarray(times, dtype=float)
    sl = np.asarray(s_lo, dtype=float)
    sh = np.asarray(s_hi, dtype=float)

    mask_t = t >= min_time
    if mask_t.sum() > 1:
        t_max = float(np.quantile(t[mask_t], trim_quantile))
        mask_t &= t <= t_max
    t, sl, sh = t[mask_t], sl[mask_t], sh[mask_t]

    valid = (sl > 1e-9) & (sl < 1.0 - 1e-9) & (sh > 1e-9) & (sh < 1.0 - 1e-9)
    t_v, sl_v, sh_v = t[valid], sl[valid], sh[valid]

    y = np.log(-np.log(sh_v)) - np.log(-np.log(sl_v))
    log_t = np.log(t_v)

    if len(y) < 4:
        raise ValueError(f"Too few valid grid points (n={len(y)}) for regression.")

    X = np.column_stack([np.ones(len(y)), log_t])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta_star = float(coeffs[0])
    gamma = float(coeffs[1])

    y_fit = X @ coeffs
    residuals = y - y_fit
    n = len(y)
    sigma2 = float(np.sum(residuals ** 2) / max(n - 2, 1))
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * XtX_inv
    se_bs = float(np.sqrt(cov[0, 0]))
    se_g = float(np.sqrt(cov[1, 1]))

    t_bs = beta_star / se_bs if se_bs > 0 else 0.0
    t_g = gamma / se_g if se_g > 0 else 0.0
    p_bs = float(2 * (1 - stats.t.cdf(abs(t_bs), df=n - 2)))
    p_g = float(2 * (1 - stats.t.cdf(abs(t_g), df=n - 2)))

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(residuals ** 2) / ss_tot) if ss_tot > 1e-12 else 0.0

    return ExtendedCoxResult(
        gamma=gamma,
        beta_star=beta_star,
        beta=beta_star,   # mixture baseline — no Weibull correction
        beta0=1.0,
        se_gamma=se_g,
        se_beta_star=se_bs,
        cov_matrix=cov,
        p_gamma=p_g,
        p_beta_star=p_bs,
        r_squared=r2,
        n_points=n,
        times=t_v,
        y_observed=y,
        y_fitted=y_fit,
    )
