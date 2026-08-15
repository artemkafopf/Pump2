"""Every Weibull we know how to fit, for Мирнинский 2024+ — one grid.

    (k1 | k2) x cut in {0,3,7,30,60,90} x (op-days | calendar) x (constrained | free)

* k1 = single Weibull; k2 = two-component mixture.  Constraints apply only to k2's
  second component, so k1 has one regime, k2 has two -> 12 + 24 = 36 fits.
* cut c = left truncation past day c (drop the early-failure mass, fit conditional on
  survival to c).  Same left-truncated likelihood for k1 and k2 so they are comparable.
* clock: op-days (`esp_optime.tte_op`, telemetry-measured, ~ttf_mix) vs calendar
  («Наработка»).
* constrained = the EM geometry: beta2 > max(1, beta1) AND eta2 > 2*eta1.
  free = same bounds, both released (beta2 may fall below 1, eta2 down to eta1).
  Both keep beta1 <= 8: a mixture likelihood is unbounded without it.

Estimand: cause-specific — event = genuine failure; ГТМ pulls and still-running pumps
are censored.  Мирнинский = installs 2024+ (config.MC_INSTALL_COHORT_START).

Each fit is scored by log-likelihood and by max|dS| vs the (left-truncated) KM.  The k2
is LR-tested against the nested k1 at the same cut/clock, so its 3 extra parameters are
earned, not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_optime as O  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402

CUTS = (0.0, 3.0, 7.0, 30.0, 60.0, 90.0)
LOG_BETA = (np.log(0.05), np.log(8.0))
LOG_ETA = (np.log(0.5), np.log(50_000.0))
LOG_GAP = (np.log(1e-4), np.log(1e4))
LOGIT_W1 = (-8.0, 4.0)
MIN_RATIO = 2.0


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


def max_ks(t, ev, p, cut):
    m = t > cut
    km = KaplanMeierFitter().fit(t[m], ev[m], entry=np.full(int(m.sum()), cut) if cut > 0 else None)
    x = km.survival_function_.index.to_numpy(float)
    y = km.survival_function_.iloc[:, 0].to_numpy(float)
    s = surv_k2(x, p["w1"], p["beta1"], p["eta1"], p["beta2"], p["eta2"])
    sc = float(surv_k2(np.array([max(cut, 1e-9)]), p["w1"], p["beta1"], p["eta1"],
                       p["beta2"], p["eta2"])[0]) if cut > 0 else 1.0
    return float(np.max(np.abs(y - s / sc)))


def b50(p):
    grid = np.arange(0.5, 20000, 1.0)
    s = surv_k2(grid, p["w1"], p["beta1"], p["eta1"], p["beta2"], p["eta2"])
    hit = np.nonzero(s <= 0.5)[0]
    return float(grid[hit[0]]) if hit.size else float("nan")


def main():
    out = results_dir("production_risk_mc_weibull_grid")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pop = P.apply_mc_cohort(P.build("2026-07-01"))
    mc = pop[pop["field"] == "Mc"].copy()
    mc = O.measure(mc, as_of=pd.Timestamp("2026-07-01"))
    clocks = {"op_days": mc["tte_op"].to_numpy(float), "calendar": mc["tte"].to_numpy(float)}
    ev = mc["event"].to_numpy(int)
    print("Mc 2024+: %d runs, %d failures (ГТМ+live censored)" % (len(mc), int(ev.sum())))

    rows = []
    for clock, t in clocks.items():
        for cut in CUTS:
            k1 = fit_k1(t, ev, cut)
            k1.update(kind="k1", regime="-", clock=clock, cut=cut, ks=max_ks(t, ev, k1, cut), b50=b50(k1))
            rows.append(k1)
            for constrained in (True, False):
                k2 = fit_k2(t, ev, cut, constrained)
                lr = 2 * (k2["loglik"] - k1["loglik"])
                k2.update(kind="k2", regime="constrained" if constrained else "free",
                          clock=clock, cut=cut, ks=max_ks(t, ev, k2, cut), b50=b50(k2),
                          lr_vs_k1=round(lr, 2), p_lr=round(float(chi2.sf(max(lr, 0), 3)), 4))
                rows.append(k2)

    df = pd.DataFrame(rows)
    cols = ["clock", "cut", "kind", "regime", "n", "events", "w1", "beta1", "eta1",
            "beta2", "eta2", "b50", "loglik", "ks", "lr_vs_k1", "p_lr"]
    df = df[[c for c in cols if c in df.columns]]
    for c in ("w1", "beta1", "eta1", "beta2", "eta2", "b50", "loglik", "ks"):
        df[c] = df[c].round(3)
    df.to_csv(tables / "mc_weibull_grid.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_rows", 80):
        print(df.to_string(index=False))
    print("\nwrote", tables)


if __name__ == "__main__":
    main()
