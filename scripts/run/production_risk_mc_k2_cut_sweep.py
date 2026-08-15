"""K=2 Weibull mixtures for Мирнинский at left-truncation cuts c = 0/3/7/30.

Why this exists
---------------
`two_layer.fit(cut_days=c)` fits a **single** Weibull past `c` and books the early
failures as a separate per-install probability — so it has no `w1/beta1/eta1` to
report.  `fit_latent_weibull_em` fits a real k2 but has **no left truncation**, and
`esp_hazard_fit` pins `beta2 = 1`.  This fills the gap: a k2 fitted **conditional on
survival past c**, so the same 5 registry columns can be read at each cut.

What the cut does, and why it matters here
------------------------------------------
Mc's early mass sits at day 0 (5 failures at tte <= 3 of 44 total).  Cutting at c
removes it from the fit, which is exactly what `two_layer` does deliberately.  The
question this answers: once the day-0 mass is gone, **does the second component still
have anything to do** — or does `w1` collapse and the mixture degenerate to the single
Weibull `two_layer` already fits?

Two constraint regimes, so the constraint's contribution is visible rather than assumed:
  * `constrained` — the EM's geometry: `beta2 = max(1, beta1) + exp(.)`, `eta2 >= 2*eta1`;
  * `free`        — same bounds, `beta2` released (may fall below 1).
Both keep `beta1 <= 8`: a mixture likelihood is unbounded, and without that bound a
component collapses onto a point mass and the likelihood diverges.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402
from analysis.workflows.production_risk import two_layer as TL  # noqa: E402

CUTS = (0.0, 3.0, 7.0, 30.0)
LOG_BETA = (np.log(0.05), np.log(8.0))
LOG_ETA1 = (np.log(0.5), np.log(50_000.0))
LOG_GAP = (np.log(1e-4), np.log(1e4))
LOGIT_W1 = (-8.0, 4.0)
MIN_ETA_RATIO = 2.0


def unpack(x: np.ndarray, constrained: bool) -> tuple[float, float, float, float, float]:
    w1 = float(1.0 / (1.0 + np.exp(-x[0])))
    b1 = float(np.exp(x[1]))
    e1 = float(np.exp(x[2]))
    b2 = max(1.0, b1) + float(np.exp(x[3])) if constrained else float(np.exp(x[3]))
    e2 = e1 * (MIN_ETA_RATIO + float(np.exp(x[4])))
    return w1, b1, e1, b2, e2


def surv(t, w1, b1, e1, b2, e2) -> np.ndarray:
    t = np.clip(np.asarray(t, float), 1e-12, None)
    return w1 * np.exp(-((t / e1) ** b1)) + (1.0 - w1) * np.exp(-((t / e2) ** b2))


def _logpdf(t, b, e):
    lt = np.log(np.clip(t, 1e-12, None)) - np.log(e)
    return np.log(b) - np.log(e) + (b - 1.0) * lt - np.exp(np.clip(b * lt, -700, 700))


def _logsurv(t, b, e):
    return -np.exp(np.clip(b * (np.log(np.clip(t, 1e-12, None)) - np.log(e)), -700, 700))


def _log_mix(la: np.ndarray, lb: np.ndarray, w1: float) -> np.ndarray:
    """log(w1*exp(la) + (1-w1)*exp(lb)) — stable, no clipping to a floor.

    Everything here is in log space on purpose.  Computing the mixture in linear
    space and clipping at 1e-300 hands the optimiser a fake optimum: with left
    truncation both the numerator and S(cut) underflow to the SAME floor, so
    log(1e-300) - log(1e-300) = 0 scores better than any real fit (every genuine
    solution has ll < 0).  That produced ll = -0.0000 at every cut > 0.
    """
    x = np.log(max(w1, 1e-300)) + la
    y = np.log(max(1.0 - w1, 1e-300)) + lb
    hi = np.maximum(x, y)
    return hi + np.log(np.exp(x - hi) + np.exp(y - hi))


def nll(x: np.ndarray, t: np.ndarray, ev: np.ndarray, cut: float, constrained: bool) -> float:
    w1, b1, e1, b2, e2 = unpack(x, constrained)
    log_f = _log_mix(_logpdf(t, b1, e1), _logpdf(t, b2, e2), w1)
    log_s = _log_mix(_logsurv(t, b1, e1), _logsurv(t, b2, e2), w1)
    # Left truncation: every run is observed only conditional on surviving `cut`,
    # so each contributes /S(cut).  Without this the cut silently biases the fit.
    log_sc = 0.0
    if cut > 0:
        log_sc = float(_log_mix(_logsurv(np.array([cut]), b1, e1),
                                _logsurv(np.array([cut]), b2, e2), w1)[0])
    ll = float(np.sum(np.where(ev == 1, log_f, log_s)) - len(t) * log_sc)
    return -ll if np.isfinite(ll) else 1e12


def k1_loglik(t: np.ndarray, ev: np.ndarray, beta: float, eta: float, cut: float) -> float:
    """Single Weibull, same left-truncated likelihood as the k2 — the nested baseline."""
    ll = float(np.sum(np.where(ev == 1, _logpdf(t, beta, eta), _logsurv(t, beta, eta))))
    if cut > 0:
        ll -= len(t) * float(_logsurv(np.array([cut]), beta, eta)[0])
    return ll


def fit_k2(t, ev, cut: float, constrained: bool, restarts: int = 60, seed: int = 0) -> dict:
    m = t > cut
    t, ev = t[m], ev[m]
    rng = np.random.default_rng(seed)
    bounds = [LOGIT_W1, LOG_BETA, LOG_ETA1, LOG_BETA, LOG_GAP]
    best = None
    for _ in range(restarts):
        x0 = np.array([rng.uniform(-4, 1), np.log(rng.uniform(0.3, 4.0)),
                       np.log(rng.uniform(max(cut, 0.5), 300.0)),
                       np.log(rng.uniform(0.3, 3.0)), np.log(rng.uniform(0.05, 400.0))])
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(nll, x0, args=(t, ev, cut, constrained), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    w1, b1, e1, b2, e2 = unpack(best.x, constrained)
    return {"cut": cut, "regime": "constrained" if constrained else "free",
            "n": int(len(t)), "events": int(ev.sum()),
            "w1": w1, "beta1": b1, "eta1": e1, "beta2": b2, "eta2": e2,
            "eta_ratio": e2 / e1, "loglik": -float(best.fun),
            "ks": max_ks(t, ev, (w1, b1, e1, b2, e2), cut)}


def max_ks(t, ev, p, cut: float) -> float:
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(t, ev, entry=np.full(len(t), cut) if cut > 0 else None)
    x = km.survival_function_.index.to_numpy(float)
    y = km.survival_function_.iloc[:, 0].to_numpy(float)
    s = surv(x, *p)
    sc = float(surv(np.array([max(cut, 1e-9)]), *p)[0]) if cut > 0 else 1.0
    return float(np.max(np.abs(y - s / sc)))


def main() -> None:
    out = results_dir("production_risk_mc_k2_cut_sweep")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pop = P.build("2026-07-01")
    mc = P.select(pop, "Mc")           # installs-2024+ cohort
    t = mc["tte"].to_numpy(float)
    ev = mc["event"].to_numpy(int)
    print("Mc installs 2024+: %d runs, %d failures (ГТМ censored), clock=«Наработка»"
          % (len(t), int(ev.sum())))
    print("early mass: %d failures at tte<=3, %d at tte<=7, %d at tte<=30"
          % (((t <= 3) & (ev == 1)).sum(), ((t <= 7) & (ev == 1)).sum(),
             ((t <= 30) & (ev == 1)).sum()))

    rows = []
    for cut in CUTS:
        for constrained in (True, False):
            rows.append(fit_k2(t, ev, cut, constrained))
        # The k1 that `two_layer` already fits, scored on the SAME truncated likelihood
        # so the k2's 3 extra parameters can be tested rather than assumed.
        # Scored DIRECTLY, not through the mixture parameterisation: forcing w1->0 there
        # leaves eta2 = eta1*(2 + exp(gap)), so it scores a different (near-immortal)
        # curve and the LR came back at p~1e-125 on 3 df.
        L = TL.fit(mc, cut_days=cut if cut > 0 else 0.0)
        k1_ll = k1_loglik(t[t > cut], ev[t > cut], L.beta, L.eta, cut)
        rows.append({"cut": cut, "regime": "two_layer(k1)", "n": int((t > cut).sum()),
                     "events": int(ev[t > cut].sum()), "w1": 0.0, "beta1": L.beta,
                     "eta1": L.eta, "beta2": L.beta, "eta2": L.eta,
                     "eta_ratio": 1.0, "loglik": k1_ll,
                     "ks": max_ks(t[t > cut], ev[t > cut], (0.0, L.beta, L.eta, L.beta, L.eta), cut)})
    df = pd.DataFrame(rows)

    from scipy.stats import chi2
    print()
    print("=== does the k2 earn its 3 extra params over the k1? (LR, 3 df) ===")
    for cut in CUTS:
        k2 = df[(df.cut == cut) & (df.regime == "constrained")].iloc[0]
        k1 = df[(df.cut == cut) & (df.regime == "two_layer(k1)")].iloc[0]
        stat = 2 * (k2.loglik - k1.loglik)
        print("  cut=%-4.0f k2 ll=%9.2f  k1 ll=%9.2f  LR=%7.2f  p=%.4g%s"
              % (cut, k2.loglik, k1.loglik, stat, chi2.sf(max(stat, 0), 3),
                 "   <- k2 NOT justified" if stat < 7.81 else ""))
    df.to_csv(tables / "mc_k2_cut_sweep.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.round(4).to_string(index=False))
    print(f"\nwrote {tables}")


if __name__ == "__main__":
    main()
