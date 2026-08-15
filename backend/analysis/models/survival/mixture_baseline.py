"""Baseline survival curve in the repo's **k1 / k2** form, AIC-selected.

``esp_models.csv`` carries every deployed baseline as five numbers —
``w1, beta1, eta1, beta2, eta2`` — read as the two-component Weibull mixture::

    S(t) = w1·exp(−(t/η1)^β1) + (1−w1)·exp(−(t/η2)^β2)

with the **degenerate convention** ``w1 = 0`` meaning "single Weibull (k1), use component 2"
— which is why a k1 fit is stored with both components set to the same (β, η).

This module fits both forms on censored ``(durations, events)`` and selects between them:

* **k1** — a single Weibull via ``lifelines.WeibullFitter`` (2 parameters).
* **k2** — the constrained mixture EM ``weibull_em.fit_latent_weibull_em`` (5 parameters),
  which enforces ``β2 > max(1, β1)`` and ``η2 ≥ 2·η1`` so the components cannot swap labels.

k2 is only *selected* when it clears k1 by ``AIC_K2_MARGIN``; both are always reported, since
the k2 modes are usually the interpretable part (they name the heterogeneity the pooled fit
is absorbing — e.g. on Vt a short-life ≈sour and a long-life ≈nonsour component).

Extracted from ``workflows.production_risk.vt_physics_model`` (which re-exports these names
for backward compatibility) because the form is field-agnostic and Ya needs the same baseline.

Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline, MRL(0) alongside,
median reference only — :func:`life_from_S` returns all three.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from math import gamma as _gamma

import numpy as np

#: k2 is deployed only when it beats k1 by at least this AIC margin (else k1).
AIC_K2_MARGIN = 2.0

#: Default RMST horizon (days) — the repo reporting standard.
RMST_HORIZON = 730.0


# ---------------------------------------------------------------------------
# Mixture algebra
# ---------------------------------------------------------------------------
def weibull_pdf(t, beta: float, eta: float):
    z = (t / eta) ** beta
    return (beta / eta) * (t / eta) ** (beta - 1.0) * np.exp(-z)


def mixture_S(t, w1: float, beta1: float, eta1: float, beta2: float, eta2: float):
    """S(t) of the k1/k2 form (``w1=0`` ⇒ the single Weibull (b2, e2))."""
    t = np.maximum(np.asarray(t, float), 0.0)
    return (w1 * np.exp(-((t / eta1) ** beta1))
            + (1 - w1) * np.exp(-((t / eta2) ** beta2)))


def mixture_pdf(t, w1: float, beta1: float, eta1: float, beta2: float, eta2: float):
    return w1 * weibull_pdf(t, beta1, eta1) + (1 - w1) * weibull_pdf(t, beta2, eta2)


def censored_loglik(t, e, S, f) -> float:
    """Right-censored log-likelihood: log f at events, log S at censorings."""
    ll = np.where(e == 1, np.log(np.clip(f, 1e-300, None)), np.log(np.clip(S, 1e-300, None)))
    return float(np.sum(ll))


def life_from_S(S_fn, horizon: float = RMST_HORIZON, tail: float = 20000.0) -> dict:
    """RMST(0,horizon), MRL(0)=mean (∫S to a long tail), median — the reporting triple.

    MRL(0) integrates the *fitted* S far past the data, so it is a model extrapolation; RMST
    over an observed horizon is the headline number (``feedback_report_rmst_mrl``)."""
    t = np.linspace(0.0, horizon, 4000)
    rmst = float(np.trapezoid(S_fn(t), t))
    tt = np.linspace(0.0, tail, 20000)
    mrl = float(np.trapezoid(S_fn(tt), tt))
    Sv = S_fn(tt)
    below = np.where(Sv <= 0.5)[0]
    median = float(tt[below[0]]) if len(below) else float("nan")
    return {"rmst": round(rmst, 1), "mrl": round(mrl, 1), "median": round(median, 1)}


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
@dataclass
class BaselineFit:
    """A fitted baseline in ``esp_models.csv`` form plus the k1-vs-k2 evidence."""
    model_kind: str          # "k1" | "k2"
    w1: float
    beta1: float
    eta1: float
    beta2: float
    eta2: float
    n: int
    events: int
    loglik_k1: float
    aic_k1: float
    loglik_k2: float
    aic_k2: float
    delta_aic: float         # aic_k2 − aic_k1 (negative ⇒ k2 preferred)
    k2_short_life: float
    k2_long_life: float
    ci: dict = dc_field(default_factory=dict)
    #: The k2 EM parameters as fitted, kept **even when k1 is selected** — otherwise a
    #: "k2 overlay" on a k1 selection silently redraws the k1 curve (``params`` collapses to
    #: the degenerate form).  Use :meth:`S_k2` for that overlay, never ``S``.
    k2_params: dict = dc_field(default_factory=dict)

    @property
    def params(self) -> dict:
        return {"w1": self.w1, "beta1": self.beta1, "eta1": self.eta1,
                "beta2": self.beta2, "eta2": self.eta2}

    def S(self, t):
        """S(t) of the **selected** baseline (k1 degenerate or the k2 mixture)."""
        return mixture_S(t, self.w1, self.beta1, self.eta1, self.beta2, self.eta2)

    def S_k2(self, t):
        """S(t) of the k2 mixture as fitted, whichever model was selected (for the overlay)."""
        return mixture_S(t, **self.k2_params) if self.k2_params else self.S(t)


def fit_baseline(t: np.ndarray, e: np.ndarray, *, num_starts: int = 60) -> BaselineFit:
    """Fit k1 (single Weibull) and k2 (constrained mixture EM) and select by AIC.

    The EM likelihood is **multimodal** (``project_esp_survival_em``) — ``num_starts`` is the
    defence, not a formality; a single-seed EM is a sample, not a fit."""
    from lifelines import WeibullFitter
    from analysis.models.survival.weibull_em import fit_latent_weibull_em

    t = np.asarray(t, float)
    e = np.asarray(e, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = WeibullFitter().fit(t, e)
    b1s, e1s = float(wf.rho_), float(wf.lambda_)
    ll1 = censored_loglik(t, e, np.exp(-((t / e1s) ** b1s)), weibull_pdf(t, b1s, e1s))
    aic1 = 2 * 2 - 2 * ll1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit_latent_weibull_em(t, e, num_starts=num_starts)
    m = r.model
    w1 = float(m.weight_1)
    kb1, ke1 = float(m.component_1.beta), float(m.component_1.eta)
    kb2, ke2 = float(m.component_2.beta), float(m.component_2.eta)
    ll2 = censored_loglik(t, e, mixture_S(t, w1, kb1, ke1, kb2, ke2),
                          mixture_pdf(t, w1, kb1, ke1, kb2, ke2))
    aic2 = 2 * 5 - 2 * ll2
    delta = aic2 - aic1
    selected = "k2" if delta < -AIC_K2_MARGIN else "k1"

    if selected == "k1":
        w1o, b1o, e1o, b2o, e2o = 0.0, b1s, e1s, b1s, e1s   # esp_models degenerate convention
    else:
        w1o, b1o, e1o, b2o, e2o = w1, kb1, ke1, kb2, ke2
    return BaselineFit(
        model_kind=selected, w1=w1o, beta1=b1o, eta1=e1o, beta2=b2o, eta2=e2o,
        n=len(t), events=int(e.sum()), loglik_k1=round(ll1, 2), aic_k1=round(aic1, 2),
        loglik_k2=round(ll2, 2), aic_k2=round(aic2, 2), delta_aic=round(delta, 2),
        k2_short_life=round(min(ke1, ke2) * _gamma(1 + 1 / (kb1 if ke1 <= ke2 else kb2)), 1),
        k2_long_life=round(max(ke1, ke2) * _gamma(1 + 1 / (kb2 if ke2 >= ke1 else kb1)), 1),
        k2_params={"w1": w1, "beta1": kb1, "eta1": ke1, "beta2": kb2, "eta2": ke2},
    )
