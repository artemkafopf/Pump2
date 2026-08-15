"""Is the EM's `beta2 > max(1, beta1)` constraint supported by the data?

Refits the Phase-2 K=2 Weibull mixture on the full mart population on the `ttf_mix`
clock (genuinely operating days), **including the 42% still-running wells as
right-censored** (`event = 0`).  Мирнинский is the installs-2024+ cohort, applied by
`load_mart_df` (see `config.MC_INSTALL_COHORT_START`).  Compares:

  * **constrained** — `fit_latent_weibull_em`, which reparameterises
    `beta2 = max(1, beta1) + exp(phi)` with `exp(phi) >= 0.01`, so beta2 can never
    fall below ~1.01.  The constraint exists to prevent label swap (component 2 must
    be the "wear-out" arm), not because wear-out was measured;
  * **free** — the SAME likelihood (`weibull_em._compute_nll`, so the numbers are
    comparable) with both betas free.

The question the comparison answers is whether the constraint BINDS: if the free
optimum puts beta2 below 1, the constraint is not encoding physics, it is imposing a
shape the data rejects — and `beta2` pinned near 1 then reads as a "flat plateau"
that is an artefact of the parameterisation rather than a finding.

Reports per stratum: both fits, the log-likelihood the constraint costs, whether
beta2_excess sits on its 0.01 floor, and max|dS| against Kaplan-Meier.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from analysis.models.survival import weibull_em as W  # noqa: E402
from analysis.models.survival.weibull_em import (  # noqa: E402
    _compute_nll,
    _simple_km,
    fit_latent_weibull_em,
)
from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.esp_survival.data_mart import load_mart_df  # noqa: E402

B2_EXCESS_FLOOR = 0.01  # the constrained parameterisation's lower bound


def mixture_survival(t, w1, b1, e1, b2, e2) -> np.ndarray:
    t = np.clip(np.asarray(t, float), 1e-12, None)
    return w1 * np.exp(-((t / e1) ** b1)) + (1.0 - w1) * np.exp(-((t / e2) ** b2))


# The free fit must sit in the SAME feasible set as the EM except for the one
# constraint under test.  A finite-mixture likelihood is UNBOUNDED — let beta1 run and a
# component collapses onto a point mass, sending the density (and the log-likelihood) to
# infinity while the curve fit gets worse.  The EM's beta1 <= 8 bound is what regularises
# that, so it is kept here; only beta2's `> max(1, beta1)` floor is released.
#
# Every bound and the eta link are READ FROM THE MODULE, never copied: a local copy went
# stale the moment the module's eta link changed, and the free arm silently became the
# handicapped one (a free fit scoring WORSE than the constrained one is the tell).
_LOG_BETA_BOUNDS = W._LOG_BETA1_BOUNDS
_LOG_ETA1_BOUNDS = W._LOG_ETA1_BOUNDS
_LOG_ETA_GAP_BOUNDS = W._LOG_ETA_GAP_BOUNDS
_MIN_ETA_RATIO = W.DEFAULT_MIN_ETA_RATIO


def _free_unpack(x: np.ndarray) -> tuple[float, float, float, float, float]:
    w1 = float(np.clip(1.0 / (1.0 + np.exp(-x[0])), 1e-6, 1 - 1e-6))
    b1 = float(np.exp(x[1]))
    e1 = float(np.exp(x[2]))
    b2 = float(np.exp(x[3]))          # <- free: NOT tied to max(1, beta1)
    e2 = e1 * (_MIN_ETA_RATIO + float(np.exp(x[4])))   # same exp link as the module
    return w1, b1, e1, b2, e2


def _free_nll(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    w1, b1, e1, b2, e2 = _free_unpack(x)
    try:
        v = _compute_nll(t, e, w1, b1, e1, b2, e2)
    except Exception:
        return 1e12
    return v if np.isfinite(v) else 1e12


def fit_free(t: np.ndarray, e: np.ndarray, restarts: int = 24, seed: int = 0) -> dict:
    """Same likelihood and same bounds as the EM, with ONLY beta2's floor released."""
    rng = np.random.default_rng(seed)
    bounds = [(-6.0, 6.0), _LOG_BETA_BOUNDS, _LOG_ETA1_BOUNDS, _LOG_BETA_BOUNDS,
              _LOG_ETA_GAP_BOUNDS]
    best = None
    for _ in range(restarts):
        x0 = np.array([
            rng.normal(-1.0, 1.0),
            np.log(rng.uniform(0.3, 3.0)),
            np.log(rng.uniform(5.0, 300.0)),
            np.log(rng.uniform(0.3, 3.0)),
            np.log(rng.uniform(0.05, 400.0)),
        ])
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(_free_nll, x0, args=(t, e), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    w1, b1, e1, b2, e2 = _free_unpack(best.x)
    return {"w1": w1, "beta1": b1, "eta1": e1, "beta2": b2, "eta2": e2,
            "loglik": -float(best.fun)}


def max_ks(t: np.ndarray, e: np.ndarray, params: dict) -> float:
    times, surv = _simple_km(t, e)
    fitted = mixture_survival(times, params["w1"], params["beta1"], params["eta1"],
                              params["beta2"], params["eta2"])
    return float(np.max(np.abs(surv - fitted)))


def main() -> None:
    out = results_dir("esp_survival_em_constraint_check")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = load_mart_df(tte_col="ttf_mix")
    rows = []
    for stratum, g in df.groupby("stratum"):
        t = g["tte"].to_numpy(float)
        e = g["event"].to_numpy(int)
        if e.sum() < 10:
            continue

        r = fit_latent_weibull_em(t, e, num_starts=12, max_iter=500)
        m = r.model
        con = {"w1": float(m.weight_1),
               "beta1": float(m.component_1.beta), "eta1": float(m.component_1.eta),
               "beta2": float(m.component_2.beta), "eta2": float(m.component_2.eta)}
        con["loglik"] = -float(_compute_nll(t, e, con["w1"], con["beta1"], con["eta1"],
                                            con["beta2"], con["eta2"]))
        free = fit_free(t, e)

        excess = con["beta2"] - max(1.0, con["beta1"])
        rows.append({
            "stratum": stratum,
            "n": len(g), "events": int(e.sum()), "cens": int((e == 0).sum()),
            "con_w1": round(con["w1"], 3), "con_beta1": round(con["beta1"], 3),
            "con_eta1": round(con["eta1"], 1), "con_beta2": round(con["beta2"], 3),
            "con_eta2": round(con["eta2"], 1),
            "b2_excess": round(excess, 4),
            "b2_on_floor": bool(excess <= B2_EXCESS_FLOOR * 1.5),
            "free_w1": round(free["w1"], 3), "free_beta1": round(free["beta1"], 3),
            "free_eta1": round(free["eta1"], 1), "free_beta2": round(free["beta2"], 3),
            "free_eta2": round(free["eta2"], 1),
            "free_beta2_lt_1": bool(free["beta2"] < 1.0),
            "loglik_con": round(con["loglik"], 2),
            "loglik_free": round(free["loglik"], 2),
            "loglik_cost": round(free["loglik"] - con["loglik"], 2),
            "ks_con": round(max_ks(t, e, con), 4),
            "ks_free": round(max_ks(t, e, free), 4),
        })

    res = pd.DataFrame(rows)
    res.to_csv(tables / "em_constraint_check.csv", index=False, encoding="utf-8-sig")

    print("=== constrained (beta2 > max(1, beta1)) vs free, ttf_mix, all runs incl. censored ===")
    print(res[["stratum", "n", "events", "cens", "con_w1", "con_beta1", "con_beta2",
               "b2_excess", "b2_on_floor", "free_beta1", "free_beta2",
               "loglik_cost", "ks_con", "ks_free"]].to_string(index=False))
    print("\nbeta2 pinned to the 0.01 floor: %d of %d strata" % (res["b2_on_floor"].sum(), len(res)))
    print("free beta2 < 1 (constraint contradicts the MLE): %d of %d"
          % (res["free_beta2_lt_1"].sum(), len(res)))
    print(f"\nwrote {tables}")


if __name__ == "__main__":
    main()
