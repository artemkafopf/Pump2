"""EM algorithm for K=2 Weibull mixture on raw (durations, events) data.

Constraints enforced by reparameterization:
  - β₁ > 0, ≤ 8  (free upper-bounded; no β₁ < 1 requirement)
  - β₂ > max(1, β₁)  (hard; β₂ = max(1, β₁) + exp(φ₂_excess) — prevents label swap)
  - η₂ ≥ min_eta_ratio · η₁  (hard FLOOR, no ceiling; η₂ = η₁ · (min_ratio + exp(φ_gap)))
  - w₁: soft only — flagged in message if > degenerate_w1_threshold

Returns LatentWeibullCurveFitResult for drop-in compatibility with Phase 3–6.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .latent_weibull_competing_risks import (
    EPSILON,
    LatentWeibullCurveFitResult,
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_survival,
)

# ── Public defaults ───────────────────────────────────────────────────────────

DEFAULT_MIN_ETA_RATIO: float = 2.0
DEFAULT_NUM_STARTS: int = 12
DEFAULT_MAX_ITER: int = 500
DEFAULT_TOL: float = 1e-7
DEFAULT_DEGENERATE_W1: float = 0.75

# M-step parameter bounds
_LOG_BETA1_BOUNDS = (np.log(0.05), np.log(8.0))      # β₁ ∈ [0.05, 8]
_LOG_ETA1_BOUNDS = (np.log(0.5), np.log(50_000.0))   # η₁ ∈ [0.5, 50 000]
_LOG_B2_EXCESS_BOUNDS = (np.log(0.01), np.log(50.0)) # β₂_excess = β₂ − max(1, β₁) ∈ [0.01, 50]
# log(η₂/η₁ − min_ratio).  η₂ = η₁·(min_ratio + exp(p)), so the floor η₂ ≥ min_ratio·η₁
# holds for every p while the ratio can reach min_ratio + 1e4 — effectively no ceiling.
#
# This was `(-12, 12)` fed through a softplus, which capped the ratio at
# 2 + softplus(12) = 14.0 exactly — an ACCIDENTAL CEILING that bound 4 of 9 strata
# (Da/Ic/Mc/Other), fixing their η₂, B50 and whole tail to a bound rather than the data.
# Only a floor was ever intended.  The softplus was also inconsistent with its own
# initialiser and starts, which both work in exp space (`log(sp_val)` inverts exp, not
# softplus), so a warm start at ratio 14 came back as 4.6.
_LOG_ETA_GAP_BOUNDS = (float(np.log(1e-4)), float(np.log(1e4)))


# ── Halton low-discrepancy sequence ──────────────────────────────────────────

def _van_der_corput(index: int, base: int) -> float:
    value, denominator, current = 0.0, 1.0, int(index)
    while current > 0:
        current, remainder = divmod(current, base)
        denominator *= float(base)
        value += float(remainder) / denominator
    return value


def _halton_points(num_points: int, dimension: int) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if dimension > len(primes):
        raise ValueError(f"Halton: max {len(primes)} dimensions.")
    return np.array(
        [[_van_der_corput(i, primes[d]) for d in range(dimension)]
         for i in range(1, num_points + 1)],
        dtype=float,
    )


# ── Numerical helpers ─────────────────────────────────────────────────────────

def _softplus(x: float) -> float:
    return float(x) if x > 50.0 else float(np.log1p(np.exp(x)))


def _log_weibull_pdf(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    log_t_eta = np.log(np.clip(t, EPSILON, None)) - np.log(eta)
    z = np.exp(np.clip(beta * log_t_eta, -700.0, 700.0))
    return np.log(beta) - np.log(eta) + (beta - 1.0) * log_t_eta - z


def _log_weibull_surv(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    return -np.power(np.clip(t, EPSILON, None) / eta, beta)


# ── Minimal KM for post-hoc RMSE ─────────────────────────────────────────────

def _simple_km(t: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival at each unique observed time."""
    order = np.argsort(t, kind="stable")
    t_s, e_s = t[order], events[order]
    n = len(t_s)
    survival = 1.0
    times_out: list[float] = []
    surv_out: list[float] = []
    i = 0
    while i < n:
        j = i
        while j < n and t_s[j] == t_s[i]:
            j += 1
        at_risk = n - i
        d = int(e_s[i:j].sum())
        if d > 0:
            survival *= 1.0 - d / at_risk
        times_out.append(float(t_s[i]))
        surv_out.append(float(survival))
        i = j
    return np.array(times_out), np.array(surv_out)


# ── E-step ────────────────────────────────────────────────────────────────────

def _e_step(
    t: np.ndarray,
    events: np.ndarray,
    w1: float,
    beta1: float, eta1: float,
    beta2: float, eta2: float,
) -> tuple[np.ndarray, np.ndarray]:
    log_w1 = np.log(max(w1, EPSILON))
    log_w2 = np.log(max(1.0 - w1, EPSILON))
    fail = events == 1
    log_c1 = np.where(fail, _log_weibull_pdf(t, beta1, eta1), _log_weibull_surv(t, beta1, eta1))
    log_c2 = np.where(fail, _log_weibull_pdf(t, beta2, eta2), _log_weibull_surv(t, beta2, eta2))
    log_n1 = log_w1 + log_c1
    log_n2 = log_w2 + log_c2
    log_max = np.maximum(log_n1, log_n2)
    log_denom = log_max + np.log(np.exp(log_n1 - log_max) + np.exp(log_n2 - log_max))
    r1 = np.exp(np.clip(log_n1 - log_denom, -500.0, 0.0))
    return r1, 1.0 - r1


def _compute_nll(
    t: np.ndarray,
    events: np.ndarray,
    w1: float,
    beta1: float, eta1: float,
    beta2: float, eta2: float,
) -> float:
    log_w1 = np.log(max(w1, EPSILON))
    log_w2 = np.log(max(1.0 - w1, EPSILON))
    fail = events == 1
    log_c1 = np.where(fail, _log_weibull_pdf(t, beta1, eta1), _log_weibull_surv(t, beta1, eta1))
    log_c2 = np.where(fail, _log_weibull_pdf(t, beta2, eta2), _log_weibull_surv(t, beta2, eta2))
    log_l1 = log_w1 + log_c1
    log_l2 = log_w2 + log_c2
    log_max = np.maximum(log_l1, log_l2)
    log_mix = log_max + np.log(np.exp(log_l1 - log_max) + np.exp(log_l2 - log_max))
    return float(-np.sum(log_mix))


# ── M-step ────────────────────────────────────────────────────────────────────

def _m_step(
    t: np.ndarray,
    events: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    beta1_prev: float, eta1_prev: float,
    beta2_prev: float, eta2_prev: float,
    min_eta_ratio: float,
    fix_beta1: float | None,
    fix_beta2: float | None,
) -> tuple[float, float, float, float]:
    """Joint M-step: update Weibull shape/scale parameters via weighted NLL.

    Parameterization:
      β₁ = exp(φ₁)                           (free, ≤ 8)
      β₂ = max(1, β₁) + exp(φ₂_excess)      (enforces β₂ > max(1, β₁), β₂ > 1)
      η₁ = exp(φ_η₁)
      η₂ = η₁ · (min_ratio + softplus(φ_gap)) (enforces η₂ ≥ min_ratio · η₁)
    """
    fail = events == 1

    p0_parts: list[float] = []
    bounds_parts: list[tuple[float, float]] = []

    if fix_beta1 is None:
        p0_parts.append(float(np.clip(np.log(max(beta1_prev, 0.05)), *_LOG_BETA1_BOUNDS)))
        bounds_parts.append(_LOG_BETA1_BOUNDS)

    p0_parts.append(float(np.clip(np.log(max(eta1_prev, 0.5)), *_LOG_ETA1_BOUNDS)))
    bounds_parts.append(_LOG_ETA1_BOUNDS)

    if fix_beta2 is None:
        # β₂_excess = β₂ - max(1, β₁); initialize from previous values
        b2_excess_init = max(beta2_prev - max(1.0, beta1_prev), 0.01)
        p0_parts.append(float(np.clip(np.log(b2_excess_init), *_LOG_B2_EXCESS_BOUNDS)))
        bounds_parts.append(_LOG_B2_EXCESS_BOUNDS)

    # η-gap: p = log(η₂/η₁ − min_ratio) — the exact inverse of the exp link below, so a
    # warm start round-trips (it did not under the old softplus link).
    gap_val = max(eta2_prev / max(eta1_prev, EPSILON) - min_eta_ratio, 1e-4)
    p0_parts.append(float(np.clip(np.log(gap_val), *_LOG_ETA_GAP_BOUNDS)))
    bounds_parts.append(_LOG_ETA_GAP_BOUNDS)

    p0 = np.array(p0_parts)

    def objective(params: np.ndarray) -> float:
        idx = 0
        b1 = fix_beta1 if fix_beta1 is not None else float(np.exp(params[idx]))
        if fix_beta1 is None:
            idx += 1
        e1 = float(np.exp(params[idx])); idx += 1
        if fix_beta2 is None:
            b2_excess = float(np.exp(params[idx])); idx += 1
            b2 = max(1.0, b1) + b2_excess
        else:
            b2 = fix_beta2
        e2 = e1 * (min_eta_ratio + float(np.exp(params[idx])))
        ll1 = np.where(fail, _log_weibull_pdf(t, b1, e1), _log_weibull_surv(t, b1, e1))
        ll2 = np.where(fail, _log_weibull_pdf(t, b2, e2), _log_weibull_surv(t, b2, e2))
        total = float(np.dot(r1, ll1) + np.dot(r2, ll2))
        return -total if np.isfinite(total) else 1e15

    res = minimize(
        objective, p0, method="L-BFGS-B", bounds=bounds_parts,
        options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-9},
    )

    idx = 0
    b1_new = fix_beta1 if fix_beta1 is not None else float(np.exp(res.x[idx]))
    if fix_beta1 is None:
        idx += 1
    e1_new = float(np.exp(res.x[idx])); idx += 1
    if fix_beta2 is None:
        b2_excess = float(np.exp(res.x[idx])); idx += 1
        b2_new = max(1.0, b1_new) + b2_excess
    else:
        b2_new = fix_beta2
    e2_new = e1_new * (min_eta_ratio + float(np.exp(res.x[idx])))

    return b1_new, e1_new, b2_new, e2_new


# ── Single EM run ─────────────────────────────────────────────────────────────

def _run_em(
    t: np.ndarray,
    events: np.ndarray,
    w1_0: float, beta1_0: float, eta1_0: float,
    beta2_0: float, eta2_0: float,
    max_iter: int,
    tol: float,
    min_eta_ratio: float,
    fix_beta1: float | None,
    fix_beta2: float | None,
) -> tuple[float, float, float, float, float, float, int, bool]:
    """EM from one start. Returns (w1, β₁, η₁, β₂, η₂, nll, n_iter, converged)."""
    w1, b1, e1, b2, e2 = w1_0, beta1_0, eta1_0, beta2_0, eta2_0
    nll_prev = _compute_nll(t, events, w1, b1, e1, b2, e2)

    for iteration in range(1, max_iter + 1):
        r1, r2 = _e_step(t, events, w1, b1, e1, b2, e2)

        w1_new = float(np.clip(float(r1.mean()), EPSILON, 1.0 - EPSILON))

        try:
            b1_new, e1_new, b2_new, e2_new = _m_step(
                t, events, r1, r2,
                b1, e1, b2, e2,
                min_eta_ratio, fix_beta1, fix_beta2,
            )
        except Exception:
            break

        w1, b1, e1, b2, e2 = w1_new, b1_new, e1_new, b2_new, e2_new
        nll_new = _compute_nll(t, events, w1, b1, e1, b2, e2)

        if not np.isfinite(nll_new):
            break

        improvement = abs(nll_prev - nll_new)
        if improvement < tol * max(abs(nll_prev), 1.0) or improvement < 1e-6:
            return w1, b1, e1, b2, e2, nll_new, iteration, True

        nll_prev = nll_new

    nll_final = _compute_nll(t, events, w1, b1, e1, b2, e2)
    return w1, b1, e1, b2, e2, nll_final, max_iter, False


# ── Public API ────────────────────────────────────────────────────────────────

def fit_latent_weibull_em(
    durations: np.ndarray,
    events: np.ndarray,
    *,
    initial_model: TwoComponentLatentWeibullModel | None = None,
    num_starts: int = DEFAULT_NUM_STARTS,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    min_eta_ratio: float = DEFAULT_MIN_ETA_RATIO,
    degenerate_w1_threshold: float = DEFAULT_DEGENERATE_W1,
    fix_beta1: float | None = None,
    fix_beta2: float | None = None,
) -> LatentWeibullCurveFitResult:
    """Fit K=2 latent Weibull mixture via EM on raw (durations, events) data.

    Parameterization enforces β₂ > max(1, β₁) (no label swap) and
    η₂ ≥ min_eta_ratio · η₁ (hard scale separation).

    Parameters
    ----------
    fix_beta1, fix_beta2 : float | None
        Two-stage mode: fix shape(s) from a global fit; only η₁, η₂, w₁ free.
    min_eta_ratio : float
        Minimum η₂/η₁ ratio (hard constraint; default 2.0).
    degenerate_w1_threshold : float
        w₁ above this is flagged as degenerate in result.message (default 0.75).
    """
    t = np.asarray(durations, dtype=float)
    ev = np.asarray(events, dtype=int)
    valid = np.isfinite(t) & (t > 0)
    t, ev = t[valid], ev[valid]

    if len(t) < 10:
        raise ValueError(f"Need ≥ 10 observations after filtering; got {len(t)}.")
    n_fail = int(ev.sum())
    if n_fail < 5:
        raise ValueError(f"Need ≥ 5 failures; got {n_fail}.")

    med_t = float(np.median(t))
    q25_t = float(np.percentile(t[ev == 1], 25)) if n_fail >= 4 else med_t * 0.25
    q75_t = float(np.percentile(t[ev == 1], 75)) if n_fail >= 4 else med_t * 1.5

    # Default starting point (from initial_model or heuristic)
    if initial_model is not None:
        w1_def = float(initial_model.weight_1)
        b1_def = float(initial_model.component_1.beta)
        e1_def = float(initial_model.component_1.eta)
        b2_def = float(initial_model.component_2.beta)
        e2_def = float(initial_model.component_2.eta)
    else:
        w1_def = 0.35
        b1_def = fix_beta1 if fix_beta1 is not None else 0.75
        e1_def = max(q25_t, 1.0)
        b2_def = fix_beta2 if fix_beta2 is not None else 2.0
        e2_def = max(min_eta_ratio * e1_def, q75_t)

    # Enforce hard constraints on the default start
    if fix_beta2 is not None:
        b2_def = fix_beta2
    else:
        b2_def = max(b2_def, max(1.0, b1_def) + 0.05)
    e2_def = max(e2_def, min_eta_ratio * e1_def + EPSILON)
    b1_def = float(np.clip(b1_def, 0.05, 8.0))

    # Build starts: default + Halton-randomised
    starts: list[tuple[float, float, float, float, float]] = [
        (w1_def, b1_def, e1_def, b2_def, e2_def)
    ]
    n_extra = max(0, num_starts - 1)
    if n_extra > 0:
        h = _halton_points(n_extra, 5)
        log_med = np.log(max(med_t, 1.0))
        for row in h:
            hw1 = 0.10 + 0.60 * row[0]
            hb1 = (fix_beta1 if fix_beta1 is not None
                   else float(np.exp(np.log(0.20) + row[1] * (np.log(8.0) - np.log(0.20)))))
            # η₁ log-uniform over [~0.04·med, ~3·med]
            he1 = float(np.exp(log_med - 3.2 + row[2] * 4.3))
            if fix_beta2 is not None:
                hb2 = fix_beta2
            else:
                # β₂_excess log-uniform over [0.05, 10]; β₂ = max(1, β₁) + excess
                hb2_excess = float(np.exp(np.log(0.05) + row[3] * (np.log(10.0) - np.log(0.05))))
                hb2 = max(1.0, hb1) + hb2_excess
            # η-gap log-uniform over [0.05, 400] — the old [0.05, 20] never proposed a
            # ratio past ~22, which reinforced the accidental 14.0 ceiling.
            hgap = float(np.exp(np.log(0.05) + row[4] * (np.log(400.0) - np.log(0.05))))
            he2 = he1 * (min_eta_ratio + hgap)
            starts.append((hw1, hb1, he1, hb2, he2))

    # Run EM from each start; keep global best NLL
    best_params: tuple | None = None
    best_nll = np.inf
    best_idx = 0
    best_n_iter = 0
    best_converged = False

    for s_idx, (w1_0, b1_0, e1_0, b2_0, e2_0) in enumerate(starts):
        # Ensure constraints before entering EM
        e2_0 = max(e2_0, min_eta_ratio * e1_0 + EPSILON)
        if fix_beta2 is None:
            b2_0 = max(b2_0, max(1.0, b1_0) + 0.01)
        try:
            result = _run_em(
                t, ev, w1_0, b1_0, e1_0, b2_0, e2_0,
                max_iter=max_iter, tol=tol,
                min_eta_ratio=min_eta_ratio,
                fix_beta1=fix_beta1, fix_beta2=fix_beta2,
            )
        except Exception:
            continue
        w1, b1, e1, b2, e2, nll, n_iter, converged = result
        if np.isfinite(nll) and nll < best_nll:
            best_params = (w1, b1, e1, b2, e2)
            best_nll = nll
            best_idx = s_idx
            best_n_iter = n_iter
            best_converged = converged

    if best_params is None:
        raise RuntimeError("All EM starts failed — check input data.")

    w1_fit, b1_fit, e1_fit, b2_fit, e2_fit = best_params

    # Post-convergence diagnostics
    is_degenerate = w1_fit > degenerate_w1_threshold
    msgs: list[str] = []
    if is_degenerate:
        msgs.append(f"DEGENERATE w1={w1_fit:.3f}>{degenerate_w1_threshold}")
    if not best_converged:
        msgs.append(f"NOT_CONVERGED after {max_iter} iters")
    message = "; ".join(msgs) if msgs else "OK"

    fitted_model = TwoComponentLatentWeibullModel(
        weight_1=w1_fit,
        component_1=WeibullParameters(beta=b1_fit, eta=e1_fit, label="C1_early"),
        component_2=WeibullParameters(beta=b2_fit, eta=e2_fit, label="C2_wearout"),
    )

    # RMSE vs KM for compatibility with Phase 3 diagnostics
    km_t, km_s = _simple_km(t, ev)
    mix_s = np.asarray(latent_survival(km_t, fitted_model), dtype=float)
    residuals = mix_s - km_s
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    n = len(t)
    n_risk = np.maximum(n - np.searchsorted(np.sort(t), km_t), 1).astype(float)
    w_km = n_risk / n_risk.max()
    weighted_rmse = float(np.sqrt(np.mean(w_km * residuals ** 2)))

    return LatentWeibullCurveFitResult(
        model=fitted_model,
        success=best_converged,
        message=message,
        objective_value=best_nll,
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        n_iter=best_n_iter,
        nfev=0,
        n_starts=len(starts),
        best_start_index=best_idx,
        method="em_raw_data",
    )
