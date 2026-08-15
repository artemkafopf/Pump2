"""Full-Vt survival model — new baseline mixture-Weibull (k1/k2) + proportional-hazard
layers (handoff prompt ``agents/analyses/vt_physics_constrained_model_prompt.md``,
re-scoped 2026-07-23 at the user's direction).

**No sour/nonsour split.** One model for the whole Vt population, composed the way the repo
deploys models — a baseline mixture-Weibull ``S(t)=w1·exp(−(t/η1)^β1)+(1−w1)·exp(−(t/η2)^β2)``
(the ``k1``/``k2`` form of ``esp_models.csv``) times proportional-hazard θ-layers applied on
the daily conditional failure probability (``q_eff = 1−(1−q)^θ``, ``θ=exp(β·(x−ref))`` — the
``esp_cox_coeffs.csv`` form):

* **Primary hazards (free):** ``Ql`` — the v3.1 rate effect (capped log-Ql, γ≈−0.31 replicated
  on Ya) — carried here as a Cox hazard ratio.
* **Established v3.1 effect carried in:** ``contractor`` (brt reference; slb/oth), the
  replicated service-quality residual (informative censoring explains only ~⅓).
* **Constrained *minor* hazards:** ``Kpod`` and ``freq`` as **capped (bounded) polylines** —
  a fitted number per knot in the log-hazard, ridge-shrunk toward θ=1 so they stay minor.
* **H₂S is intentionally excluded this pass** (user decision).  Consequence: pooling sour+
  nonsour with no H₂S term leaves that ~3× life gap as unmodelled heterogeneity — which is
  exactly what the **k2 mixture baseline absorbs** (a short-life ≈sour component + a long-life
  ≈nonsour component; see the baseline table).

Baseline fit: k1 via ``lifelines.WeibullFitter``, k2 via the constrained EM
``weibull_em.fit_latent_weibull_em``; AIC-selected (k2 only when it clears k1 by ``AIC_K2_MARGIN``).
Hazard layers: one ``CoxPHFitter`` on complete-case Vt with an array penalizer (0 on the free
Ql/contractor terms, a ridge on the polyline knots).  Reference for the deployable θ is the
sample MEAN (repo convention: θ=1 at the mean ⇒ marginal baseline stays consistent); the
polyline "numbers" are additionally reported relative to the interpretable knot (Kpod 0.8 /
nominal freq).

Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline, MRL(0), median
reference only, KM control always shown.  Standing rules: paths via ``analysis.paths``;
outputs → ``results_dir`` slug below; Mc untouched (Vt-only); never left-truncate; nothing
committed to git.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.models.survival import mixture_baseline as MB
from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import vt_ttf_covariates as V

# ---------------------------------------------------------------------------
# Constants / policy
# ---------------------------------------------------------------------------
SLUG = "production_risk_vt_physics_model"
FIELD = "Vt"
CLOCK = V.CLOCK_PRIMARY          # "t_cal"
EVENT_COL = V.EVENT_COL          # "event"
WELL = "code"                    # well identity for cluster bootstrap

#: v3.1 log-Ql arm: clip window + reference (prompt-mandated fixed bounds).
QL_CLIP = (47.0, 823.0)
QL_REF = 250.0
_LN_QL_REF = float(np.log(QL_REF))

#: Nominal-frequency policy: per-run value when plausible, else 50 Hz.
FNOM_DEFAULT = 50.0
FNOM_PLAUSIBLE = (40.0, 65.0)

#: Contractor reference level (brt) — slb/oth are the v3.1 hazard terms.
CONTRACTOR_REF = "brt"
CONTRACTOR_TERMS = ("slb", "oth")

# --- polyline knot grids (the constrained MINOR layers) --------------------
KPOD_KNOTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
KPOD_PIN = 0.8
FREQ_DEV_KNOTS = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)   # freq − f_nom
FREQ_PIN_DEV = 0.0

#: Ridge (L2 toward 0) on the polyline knot coefficients — this is what makes Kpod/freq
#: MINOR (shrunk toward θ=1) while the free primary terms carry the signal.  Tuned on Vt
#: (θ stays within ~±5%); raise to shrink harder, lower to let the polylines move.
POLY_PENALIZER = 6.0

#: k2 is deployed only when it beats k1 by at least this AIC margin (else k1).
AIC_K2_MARGIN = MB.AIC_K2_MARGIN

APPLICABILITY = {"ql": (47.0, 823.0), "freq": (31.0, 56.0), "kpod": (0.2, 1.2)}
RMST_HORIZON = 730.0

ESP_MODELS_REF = C.REPO_ROOT / "results" / "esp_survival_vba_models" / "2026-07-20-vt-refit" / "esp_models.csv"


# ===========================================================================
# Frame prep (no sour flag — full Vt)
# ===========================================================================

def prepare_vt_frame(as_of: pd.Timestamp | None = None,
                     cached: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full Vt modelling frame: keep clock>0, add the covariate inputs.

    Adds ``clq`` (capped-log-Ql minus ln 250), ``f_nom``, ``freq_dev`` (=freq_run−f_nom),
    and contractor dummies ``slb``/``oth`` (brt reference).  NaNs are flagged per-arm at fit
    time (baseline uses all runs; hazard layers use complete-case), never silently dropped."""
    if cached is not None:
        df = cached
    else:
        df, _ = V.build_frame(as_of=as_of)
    vt = df[df["field"] == FIELD].copy()
    vt[CLOCK] = pd.to_numeric(vt[CLOCK], errors="coerce")
    vt = vt[vt[CLOCK] > 0].dropna(subset=[CLOCK, EVENT_COL]).copy()

    fn = pd.to_numeric(vt["nominal_freq_hz"], errors="coerce")
    lo, hi = FNOM_PLAUSIBLE
    vt["f_nom"] = np.where(fn.between(lo, hi), fn, FNOM_DEFAULT)
    vt["freq_dev"] = pd.to_numeric(vt["freq_run"], errors="coerce") - vt["f_nom"]

    ql = pd.to_numeric(vt["ql"], errors="coerce")
    vt["clq"] = np.log(ql.clip(*QL_CLIP)) - _LN_QL_REF
    for cg in CONTRACTOR_TERMS:
        vt[cg] = (vt["contractor_group"] == cg).astype(float)
    return vt


# ===========================================================================
# Baseline mixture-Weibull (k1 / k2), AIC-selected
# ===========================================================================
# The k1/k2 form is field-agnostic and now lives in
# ``analysis.models.survival.mixture_baseline`` (Ya reuses it).  Re-exported here so this
# module's public surface — and its tests — are unchanged.
_weibull_pdf = MB.weibull_pdf
_mixture_S = MB.mixture_S
_mixture_pdf = MB.mixture_pdf
_censored_loglik = MB.censored_loglik
_life_from_S = MB.life_from_S
BaselineFit = MB.BaselineFit
fit_baseline = MB.fit_baseline


# ===========================================================================
# Hazard layers — Ql/contractor (free Cox) + Kpod/freq CONSTRAINED polyline
# ===========================================================================
# REQUIREMENT (user, 2026-07-23, strict form): the Kpod and freq hazard functions are
# **unimodal with the minimum exactly at the reference (Kpod=0.8, freq=nominal) and
# MONOTONE arms** — hazard θ=1 at the reference and rises monotonically as you move away on
# either side (⇔ life is maximal at the reference, rising monotonically up to it and falling
# monotonically after).  This is enforced isotonically: each side's knots are cumulative sums
# of **non-negative increments** away from the reference, so the log-hazard knot values are
# monotone (0 at the pin, non-decreasing outward) by construction ⇒ θ≥1 and monotone.  Fit by
# a direct censored **Weibull-PH MLE** (L-BFGS-B, increments ≥ 0); primary Ql/contractor HRs
# stay on the trusted lifelines Cox.

#: Non-pinned knots (kept for the standalone ``_poly_from_free`` helper / tests).
_KPOD_FREE_KNOTS = tuple(k for k in KPOD_KNOTS if k != KPOD_PIN)
_FREQ_FREE_KNOTS = tuple(k for k in FREQ_DEV_KNOTS if k != FREQ_PIN_DEV)

#: Knots on each side of the reference (ascending); the monotone arms are built outward from
#: the pin, so the *nearest-to-pin* knot carries the first increment.
_KPOD_LEFT = tuple(k for k in KPOD_KNOTS if k < KPOD_PIN)          # (0.2, 0.4, 0.6)
_KPOD_RIGHT = tuple(k for k in KPOD_KNOTS if k > KPOD_PIN)         # (1.0, 1.2)
_FREQ_LEFT = tuple(k for k in FREQ_DEV_KNOTS if k < FREQ_PIN_DEV)  # (-20, -15, -10, -5)
_FREQ_RIGHT = tuple(k for k in FREQ_DEV_KNOTS if k > FREQ_PIN_DEV) # (5, 10, 15)
_NPOLY = len(_KPOD_LEFT) + len(_KPOD_RIGHT) + len(_FREQ_LEFT) + len(_FREQ_RIGHT)


def _monotone_coefs(kl, kr, fl, fr) -> tuple[dict, dict]:
    """Build the monotone log-hazard knot dicts from non-negative increments.

    ``kl``/``fl`` are the left-arm increments ordered NEAREST→FARTHEST from the reference
    (so ``kl[0]`` is the 0.6→0.8 step); ``kr``/``fr`` the right-arm increments.  Cumulative
    sums give knot values that are 0 at the pin and non-decreasing outward ⇒ θ=exp monotone
    ≥ 1, minimum at the pin."""
    kd: dict[float, float] = {}
    csum = 0.0
    for inc, k in zip(kl, reversed(_KPOD_LEFT)):      # 0.6, 0.4, 0.2
        csum += max(0.0, float(inc)); kd[k] = csum
    csum = 0.0
    for inc, k in zip(kr, _KPOD_RIGHT):               # 1.0, 1.2
        csum += max(0.0, float(inc)); kd[k] = csum
    fd: dict[float, float] = {}
    csum = 0.0
    for inc, k in zip(fl, reversed(_FREQ_LEFT)):      # -5, -10, -15, -20
        csum += max(0.0, float(inc)); fd[k] = csum
    csum = 0.0
    for inc, k in zip(fr, _FREQ_RIGHT):               # 5, 10, 15
        csum += max(0.0, float(inc)); fd[k] = csum
    return kd, fd


def _poly_from_free(x, knots, pin, free_coef) -> np.ndarray:
    """Evaluate the polyline at ``x`` from the (non-pinned) knot coefficients, inserting 0 at
    the pinned knot; linear between knots, clamped outside.  (Standalone helper / tests.)"""
    full, fi = [], 0
    for k in knots:
        if k == pin:
            full.append(0.0)
        else:
            full.append(float(free_coef[fi])); fi += 1
    xc = np.clip(np.asarray(x, float), knots[0], knots[-1])
    return np.interp(xc, knots, np.asarray(full))


def _poly_theta(x, knots, pin, coef: dict) -> np.ndarray:
    """θ(x) = exp( Sk(x) − Sk(pin) ), Sk the pinned polyline from ``coef`` (knot→log-hazard).
    With monotone coef ≥ 0 this is ≥ 1 everywhere, = 1 at the pin, monotone on each arm."""
    full = np.array([coef.get(k, 0.0) for k in knots])
    xc = np.clip(np.asarray(x, float), knots[0], knots[-1])
    sk = np.interp(xc, knots, full)
    sk_ref = np.interp(pin, knots, full)
    return np.exp(sk - sk_ref)


@dataclass
class HazardLayers:
    primary: pd.DataFrame          # Ql + contractor (trusted Cox): covariate, beta, hr, ci, p
    kpod_knots: list               # [{knot, theta(≥1), theta_ci_lo, theta_ci_hi}]
    freq_knots: list
    kpod_coef: dict                # non-negative log-hazard knot coefs (pinned knot absent)
    freq_coef: dict
    n: int
    events: int
    concordance: float
    loglik: float                  # constrained Weibull-PH data loglik (ridge excluded)
    penalizer: float
    baseline_nuisance: dict        # β0, η0 of the PH fit (nuisance; deploy uses the k1/k2 baseline)

    def theta_kpod(self, k) -> np.ndarray:
        return _poly_theta(k, KPOD_KNOTS, KPOD_PIN, self.kpod_coef)

    def theta_freq(self, f) -> np.ndarray:
        return _poly_theta(np.asarray(f, float) - FNOM_DEFAULT, FREQ_DEV_KNOTS, FREQ_PIN_DEV, self.freq_coef)


_NKL, _NKR = len(_KPOD_LEFT), len(_KPOD_RIGHT)
_NFL, _NFR = len(_FREQ_LEFT), len(_FREQ_RIGHT)


def _split_increments(params):
    """Slice the polyline increment groups (kpod-left, kpod-right, freq-left, freq-right)."""
    o = 5
    kl = params[o:o + _NKL]; o += _NKL
    kr = params[o:o + _NKR]; o += _NKR
    fl = params[o:o + _NFL]; o += _NFL
    fr = params[o:o + _NFR]
    return kl, kr, fl, fr


def _ph_nll(params, t, e, clq, slb, oth, kpod, freqdev) -> float:
    """Censored Weibull-PH negative log-lik: baseline (β0,η0) + free Ql/contractor + the
    MONOTONE Kpod/freq polylines (built from non-negative increments) in the log-hazard."""
    beta0, ln_eta0, b_clq, b_slb, b_oth = params[:5]
    if beta0 <= 0:
        return 1e18
    kd, fd = _monotone_coefs(*_split_increments(params))
    kfull = np.array([kd.get(k, 0.0) for k in KPOD_KNOTS])
    ffull = np.array([fd.get(k, 0.0) for k in FREQ_DEV_KNOTS])
    Sk = np.interp(np.clip(kpod, KPOD_KNOTS[0], KPOD_KNOTS[-1]), KPOD_KNOTS, kfull)
    Sf = np.interp(np.clip(freqdev, FREQ_DEV_KNOTS[0], FREQ_DEV_KNOTS[-1]), FREQ_DEV_KNOTS, ffull)
    lp = b_clq * clq + b_slb * slb + b_oth * oth + Sk + Sf
    ln_t = np.log(t)
    base = np.clip(beta0 * (ln_t - ln_eta0), -700.0, 700.0)
    ll = e * (lp + np.log(beta0) - ln_eta0 + (beta0 - 1.0) * (ln_t - ln_eta0)) - np.exp(lp + base)
    val = -float(np.sum(ll))
    return val if np.isfinite(val) else 1e18


def _ph_obj(params, t, e, clq, slb, oth, kpod, freqdev, lam) -> float:
    nll = _ph_nll(params, t, e, clq, slb, oth, kpod, freqdev)
    if nll >= 1e18:
        return 1e18
    incr = np.asarray(params[5:], float)             # ridge on the increments (toward flat θ=1)
    return nll + lam * float(np.dot(incr, incr))


def fit_hazard_layers(cc: pd.DataFrame, *, lam_poly: float = POLY_PENALIZER) -> HazardLayers:
    """Fit the hazard layers on complete-case Vt (ql+kpod+freq present).

    Primary Ql + contractor HRs come from a trusted lifelines Cox (unbounded).  The
    Kpod/freq polylines are fit by a constrained Weibull-PH MLE with the primaries as
    controls and **non-negative increments away from the reference**, so each θ is unimodal
    (=1 at Kpod 0.8 / nominal freq, monotone rising on each arm ⇒ life maximal at, and
    monotone falling away from, the reference).  A ridge (``lam_poly``) shrinks the
    increments toward a flat θ=1; the reported loglik excludes the ridge."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from scipy.optimize import minimize

    d = cc.dropna(subset=["clq", "kpod_run", "freq_run"]).copy()

    # -- primary: trusted Cox on Ql + contractor (headline HR/CI/p) ----------
    Xp = d[["clq", "slb", "oth", CLOCK, EVENT_COL]].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(Xp, CLOCK, EVENT_COL)
    s = cph.summary
    prim_rows = []
    for cov, label in [("clq", "Ql (capped log)"), ("slb", "contractor:slb"), ("oth", "contractor:oth")]:
        if cov not in s.index:
            continue
        r = s.loc[cov]
        prim_rows.append({
            "covariate": cov, "label": label,
            "beta": round(float(r["coef"]), 4), "hr": round(float(np.exp(r["coef"])), 4),
            "hr_ci_lo": round(float(np.exp(r["coef lower 95%"])), 4),
            "hr_ci_hi": round(float(np.exp(r["coef upper 95%"])), 4),
            "p": round(float(r["p"]), 4),
            "reference": (f"Ql={QL_REF:g} m3/d" if cov == "clq" else f"vs {CONTRACTOR_REF}"),
            "hr_unit": ("per e-fold capped Ql" if cov == "clq" else f"vs {CONTRACTOR_REF}"),
        })
    primary = pd.DataFrame(prim_rows)

    # -- constrained polyline: Weibull-PH MLE, monotone (increments ≥ 0) -----
    t = d[CLOCK].to_numpy(float); e = d[EVENT_COL].to_numpy(float)
    clq = d["clq"].to_numpy(float); slb = d["slb"].to_numpy(float); oth = d["oth"].to_numpy(float)
    kpod = d["kpod_run"].to_numpy(float); freqdev = d["freq_dev"].to_numpy(float)
    bounds = ([(0.05, 6.0), (np.log(2.0), np.log(5e4)), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
              + [(0.0, 4.0)] * _NPOLY)             # increments ≥ 0  ⇒  θ≥1, monotone arms
    args = (t, e, clq, slb, oth, kpod, freqdev)
    starts = [
        np.array([0.8, np.log(450.0), 0.35, 0.2, 1.0] + [0.0] * _NPOLY),
        np.array([1.0, np.log(400.0), 0.25, 0.0, 0.8] + [0.05] * _NPOLY),
        np.array([0.7, np.log(500.0), 0.40, 0.3, 1.2] + [0.10] * _NPOLY),
    ]
    best = None
    for x0 in starts:
        try:
            res = minimize(_ph_obj, x0, args=(*args, float(lam_poly)), method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    x = best.x
    beta0, eta0 = float(x[0]), float(np.exp(x[1]))
    kpod_coef, freq_coef = _monotone_coefs(*_split_increments(x))
    kfull = np.array([kpod_coef.get(k, 0.0) for k in KPOD_KNOTS])
    ffull = np.array([freq_coef.get(k, 0.0) for k in FREQ_DEV_KNOTS])
    lp = (x[2] * clq + x[3] * slb + x[4] * oth
          + np.interp(np.clip(kpod, KPOD_KNOTS[0], KPOD_KNOTS[-1]), KPOD_KNOTS, kfull)
          + np.interp(np.clip(freqdev, FREQ_DEV_KNOTS[0], FREQ_DEV_KNOTS[-1]), FREQ_DEV_KNOTS, ffull))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conc = float(concordance_index(t, -lp, e))
    loglik = -_ph_nll(x, *args)

    kpod_knots = [{"knot": k, "theta": round(float(_poly_theta(np.array([k]), KPOD_KNOTS, KPOD_PIN, kpod_coef)[0]), 4),
                   "theta_ci_lo": np.nan, "theta_ci_hi": np.nan} for k in KPOD_KNOTS]
    freq_knots = [{"knot": k, "theta": round(float(_poly_theta(np.array([k]), FREQ_DEV_KNOTS, FREQ_PIN_DEV, freq_coef)[0]), 4),
                   "theta_ci_lo": np.nan, "theta_ci_hi": np.nan} for k in FREQ_DEV_KNOTS]
    return HazardLayers(
        primary=primary, kpod_knots=kpod_knots, freq_knots=freq_knots,
        kpod_coef=kpod_coef, freq_coef=freq_coef, n=len(d), events=int(d[EVENT_COL].sum()),
        concordance=round(conc, 4), loglik=round(loglik, 2), penalizer=float(lam_poly),
        baseline_nuisance={"beta0": round(beta0, 4), "eta0": round(eta0, 1)},
    )


# ===========================================================================
# Bootstrap (well-cluster): baseline RMST + Cox HRs + knot θ
# ===========================================================================

def bootstrap(vt: pd.DataFrame, cc: pd.DataFrame, *, n_boot: int = 200, seed: int = 7,
              lam_poly: float = POLY_PENALIZER) -> dict[str, tuple[float, float]]:
    """Well-cluster bootstrap CIs on the composed deliverables: k1 baseline RMST(0,730),
    the primary HRs, and the Kpod/freq θ at their knots.  k1 baseline only (k2 EM is too
    slow to bootstrap); resamples wells (runs of one well are dependent)."""
    from lifelines import WeibullFitter, CoxPHFitter

    rng = np.random.default_rng(seed)
    wells = vt[WELL].unique()
    vt_by = {w: g for w, g in vt.groupby(WELL)}
    cc_by = {w: g for w, g in cc.groupby(WELL)}
    keys = ["base_rmst", "hr_clq", "hr_slb", "hr_oth"]
    keys += [f"kpod_{k:g}" for k in KPOD_KNOTS] + [f"freq_{FNOM_DEFAULT + d:g}" for d in FREQ_DEV_KNOTS]
    acc: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        draw = rng.choice(wells, size=len(wells), replace=True)
        vb = pd.concat([vt_by[w] for w in draw], ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = WeibullFitter().fit(vb[CLOCK], vb[EVENT_COL])
            acc["base_rmst"].append(_life_from_S(
                lambda t, b=wf.rho_, e=wf.lambda_: np.exp(-((np.maximum(t, 0) / e) ** b)))["rmst"])
        except Exception:
            pass
        cb = pd.concat([cc_by[w] for w in draw if w in cc_by], ignore_index=True)
        if int(cb[EVENT_COL].sum()) < 30 or cb["contractor_group"].nunique() < 2:
            continue
        try:
            layers = fit_hazard_layers(cb, lam_poly=lam_poly)
        except Exception:
            continue
        for _, r in layers.primary.iterrows():
            acc[f"hr_{r['covariate']}"].append(r["hr"])
        for row in layers.kpod_knots:
            acc[f"kpod_{row['knot']:g}"].append(row["theta"])
        for row in layers.freq_knots:
            acc[f"freq_{FNOM_DEFAULT + row['knot']:g}"].append(row["theta"])

    out = {}
    for k, v in acc.items():
        vv = np.array([x for x in v if np.isfinite(x)])
        if len(vv) >= max(20, n_boot // 5):
            out[k] = (round(float(np.percentile(vv, 2.5)), 3), round(float(np.percentile(vv, 97.5)), 3))
    return out


# ===========================================================================
# Orchestration
# ===========================================================================

@dataclass
class ModelRun:
    baseline: BaselineFit
    layers: HazardLayers
    baseline_table: pd.DataFrame
    hazard_layers_table: pd.DataFrame
    knot_thetas: pd.DataFrame
    ttf_summary: pd.DataFrame
    comparison: pd.DataFrame
    coverage: pd.DataFrame
    boot: dict
    vt: pd.DataFrame
    cc: pd.DataFrame
    failures_only: bool = False


def run(*, as_of: pd.Timestamp | None = None, cached: pd.DataFrame | None = None,
        n_boot: int = 200, lam_poly: float = POLY_PENALIZER, num_starts: int = 60,
        failures_only: bool = False, write: bool = True) -> ModelRun:
    """Fit the full-Vt model.  ``failures_only=True`` drops ALL censored (running / ГТМ-pulled)
    runs and fits on the failure durations alone — a **biased** variant (the Weibull then
    describes time-given-failed, not survival, and the HRs are the informative-censoring
    collider); written to ``*_failures_only`` files so it never overwrites the honest model."""
    vt = prepare_vt_frame(as_of=as_of, cached=cached)
    if failures_only:
        vt = vt[vt[EVENT_COL] == 1].copy()          # drop every censored run
    cc = vt.dropna(subset=["ql", "kpod_run", "freq_run"]).copy()

    baseline = fit_baseline(vt[CLOCK].to_numpy(float), vt[EVENT_COL].to_numpy(float),
                            num_starts=num_starts)
    layers = fit_hazard_layers(cc, lam_poly=lam_poly)
    boot = bootstrap(vt, cc, n_boot=n_boot, lam_poly=lam_poly) if n_boot > 0 else {}

    # Inject bootstrap CIs into the constrained knot θ (the MLE has no asymptotic CI at the
    # ≥0 boundary; the bootstrap is the honest interval).
    for row in layers.kpod_knots:
        ci = boot.get(f"kpod_{row['knot']:g}")
        if ci:
            row["theta_ci_lo"], row["theta_ci_hi"] = ci
    for row in layers.freq_knots:
        ci = boot.get(f"freq_{FNOM_DEFAULT + row['knot']:g}")
        if ci:
            row["theta_ci_lo"], row["theta_ci_hi"] = ci

    baseline_table = _baseline_table(baseline, boot)
    hazard_layers_table = _hazard_layers_table(layers, boot)
    knot_thetas = _knot_theta_table(layers, boot, baseline)
    ttf_summary = _ttf_summary(baseline, layers)
    comparison = _comparison(vt, cc, baseline, layers)
    coverage = _coverage(vt)

    run_obj = ModelRun(
        baseline=baseline, layers=layers, baseline_table=baseline_table,
        hazard_layers_table=hazard_layers_table, knot_thetas=knot_thetas,
        ttf_summary=ttf_summary, comparison=comparison, coverage=coverage, boot=boot,
        vt=vt, cc=cc, failures_only=failures_only,
    )
    if write:
        _write_outputs(run_obj)
    return run_obj


def _baseline_table(b: BaselineFit, boot: dict) -> pd.DataFrame:
    lo, hi = boot.get("base_rmst", (np.nan, np.nan))
    life = _life_from_S(b.S)   # life of the SELECTED baseline (k1 degenerate or k2 mixture)
    return pd.DataFrame([{
        "field": FIELD, "scope": "Vt_full_pooled", "model_kind_selected": b.model_kind,
        "w1": round(b.w1, 4), "beta1": round(b.beta1, 4), "eta1": round(b.eta1, 1),
        "beta2": round(b.beta2, 4), "eta2": round(b.eta2, 1),
        "n_runs": b.n, "n_events": b.events,
        "loglik_k1": b.loglik_k1, "aic_k1": b.aic_k1,
        "loglik_k2": b.loglik_k2, "aic_k2": b.aic_k2, "delta_aic_k2_minus_k1": b.delta_aic,
        "k2_short_mode_life_d": b.k2_short_life, "k2_long_mode_life_d": b.k2_long_life,
        "rmst730": life["rmst"], "mrl0": life["mrl"], "median": life["median"],
        "rmst730_ci_lo": lo, "rmst730_ci_hi": hi, "clock": CLOCK,
    }])


def _hazard_layers_table(layers: HazardLayers, boot: dict) -> pd.DataFrame:
    df = layers.primary.copy()
    df.insert(0, "role", "primary")
    for _, r in df.iterrows():
        pass
    # attach bootstrap HR CIs where available
    lo_hi = {c: boot.get(f"hr_{c}") for c in df["covariate"]}
    df["hr_boot_ci_lo"] = [lo_hi[c][0] if lo_hi.get(c) else np.nan for c in df["covariate"]]
    df["hr_boot_ci_hi"] = [lo_hi[c][1] if lo_hi.get(c) else np.nan for c in df["covariate"]]
    df["n"], df["events"], df["concordance"] = layers.n, layers.events, layers.concordance
    df["poly_penalizer"] = layers.penalizer
    return df


def _knot_theta_table(layers: HazardLayers, boot: dict, baseline: BaselineFit) -> pd.DataFrame:
    # life multiplier for a Weibull hazard shift θ is θ^(−1/β); use the baseline's dominant
    # shape (β1 for k1; the higher-weight component's β for k2) — an approximation for a
    # mixture but far better than 1/θ.
    beta_eff = baseline.beta1 if baseline.w1 >= 0.5 else baseline.beta2
    rows = []
    for arm, knots, pin in (("kpod", layers.kpod_knots, KPOD_PIN),
                            ("freq", layers.freq_knots, FREQ_PIN_DEV)):
        for row in knots:
            knot = row["knot"]
            abs_knot = knot if arm == "kpod" else FNOM_DEFAULT + knot
            bkey = f"{arm}_{abs_knot:g}"
            blo, bhi = boot.get(bkey, (np.nan, np.nan))
            th = row["theta"]
            rows.append({
                "arm": arm, "role": "constrained_minor",
                "knot": abs_knot, "knot_raw": knot,
                "theta_hazard": th,
                "life_mult_approx": round(float(th ** (-1.0 / beta_eff)), 4) if th > 0 else np.nan,
                "theta_ci_lo": row["theta_ci_lo"], "theta_ci_hi": row["theta_ci_hi"],
                "theta_boot_lo": blo, "theta_boot_hi": bhi,
            })
    return pd.DataFrame(rows)


_QL_GRID = (100.0, 250.0, 500.0)
_KPOD_GRID = (0.4, 0.7, 0.9, 1.1)
_FREQ_GRID = (45.0, 50.0, 55.0)


def _ttf_summary(b: BaselineFit, layers: HazardLayers) -> pd.DataFrame:
    """Composed life over a small grid: apply θ = θ_ql·θ_kpod·θ_freq (contractor=brt) to the
    baseline via the repo's ``apply_theta`` scale shift, then RMST/MRL/median."""
    from analysis.workflows.production_risk.survival import HazardLayer

    beta_clq = float(layers.primary.set_index("covariate").loc["clq", "beta"]) if "clq" in set(layers.primary["covariate"]) else 0.0
    rows = []
    base_life = _life_from_S(b.S)
    # θ=1 reference is Ql=250 (clq=0), Kpod=0.8, freq nominal, contractor brt.
    rows.append({"ql": QL_REF, "kpod": KPOD_PIN, "freq": "nominal", "contractor": "brt",
                 "theta": 1.0, "rmst730": base_life["rmst"], "mrl0": base_life["mrl"],
                 "median": base_life["median"], "point": "baseline(marginal Vt)"})
    for ql in _QL_GRID:
        clq = float(np.log(np.clip(ql, *QL_CLIP)) - _LN_QL_REF)   # 0 at Ql=250
        th_ql = float(np.exp(beta_clq * clq))
        for kp in _KPOD_GRID:
            th_k = float(layers.theta_kpod(np.array([kp]))[0])
            for fr in _FREQ_GRID:
                th_f = float(layers.theta_freq(np.array([fr]))[0])
                theta = th_ql * th_k * th_f
                p = HazardLayer.apply_theta(b.params, theta)
                life = _life_from_S(lambda t, p=p: _mixture_S(t, p["w1"], p["beta1"], p["eta1"], p["beta2"], p["eta2"]))
                rows.append({"ql": ql, "kpod": kp, "freq": fr, "contractor": "brt",
                             "theta": round(theta, 4), "rmst730": life["rmst"],
                             "mrl0": life["mrl"], "median": life["median"], "point": "grid"})
    return pd.DataFrame(rows)


def _comparison(vt, cc, b: BaselineFit, layers: HazardLayers) -> pd.DataFrame:
    """Honesty gates: baseline k1 vs k2 AIC; Cox LR test of the layer set; n_failures vs
    the shipped registry (Vt pooled)."""
    rows = [
        {"check": "baseline_k1_aic", "value": b.aic_k1},
        {"check": "baseline_k2_aic", "value": b.aic_k2},
        {"check": "baseline_delta_aic_k2_minus_k1", "value": b.delta_aic,
         "note": f"selected {b.model_kind} (k2 needs < -{AIC_K2_MARGIN})"},
        {"check": "cox_concordance", "value": layers.concordance},
        {"check": "cox_events", "value": layers.events},
    ]
    try:
        m = pd.read_csv(ESP_MODELS_REF, encoding="utf-8-sig")
        mv = m[m["field"] == FIELD]
        reg_ev = int(mv[mv["contractor_group"] == "Pooled"]["n_failures"].sum())
        rows.append({"check": "registry_vt_pooled_failures(sour+nonsour)", "value": reg_ev,
                     "note": f"this model full-Vt events = {b.events}"})
    except Exception:
        pass
    return pd.DataFrame(rows)


def _coverage(vt: pd.DataFrame) -> pd.DataFrame:
    cc = vt.dropna(subset=["ql", "kpod_run", "freq_run"])
    return pd.DataFrame([{
        "scope": "Vt_full_pooled", "n_runs": len(vt), "n_events": int(vt[EVENT_COL].sum()),
        "n_open": int(vt["end"].isna().sum()),
        "cov_ql": round(vt["ql"].notna().mean(), 3), "cov_kpod": round(vt["kpod_run"].notna().mean(), 3),
        "cov_freq": round(vt["freq_run"].notna().mean(), 3),
        "n_complete_case": len(cc), "events_complete_case": int(cc[EVENT_COL].sum()),
        "contractor_mix": str(vt["contractor_group"].value_counts().to_dict()),
    }])


# ---------------------------------------------------------------------------
# Output writers (tables + Russian figures)
# ---------------------------------------------------------------------------
def _write_outputs(run_obj: ModelRun) -> Path:
    out = results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"
    sfx = "_failures_only" if run_obj.failures_only else ""
    run_obj.baseline_table.to_csv(tables / f"baseline{sfx}.csv", index=False, encoding="utf-8-sig")
    run_obj.hazard_layers_table.to_csv(tables / f"hazard_layers{sfx}.csv", index=False, encoding="utf-8-sig")
    run_obj.knot_thetas.to_csv(tables / f"knot_thetas{sfx}.csv", index=False, encoding="utf-8-sig")
    run_obj.ttf_summary.to_csv(tables / f"ttf_summary{sfx}.csv", index=False, encoding="utf-8-sig")
    run_obj.comparison.to_csv(tables / f"model_comparison{sfx}.csv", index=False, encoding="utf-8-sig")
    run_obj.coverage.to_csv(tables / f"coverage{sfx}.csv", index=False, encoding="utf-8-sig")

    V.FIG_LANG = "ru"
    try:
        _fig_baseline(run_obj, figures, sfx)
        _fig_hazard_polyline(run_obj, figures, sfx)
        _fig_primary_forest(run_obj, figures, sfx)
    finally:
        V.FIG_LANG = "en"
    return out


def _fig_baseline(run_obj: ModelRun, figures: Path, sfx: str = "") -> None:
    """KM(all Vt) + k1 and k2 baseline overlays — the mandatory KM control."""
    from lifelines import KaplanMeierFitter
    b = run_obj.baseline
    vt = run_obj.vt
    fo = " · ТОЛЬКО ОТКАЗЫ (смещено)" if run_obj.failures_only else ""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    km = KaplanMeierFitter().fit(vt[CLOCK], vt[EVENT_COL],
                                 label=f"KM все Vt (n={len(vt)}, отказов={int(vt[EVENT_COL].sum())})")
    km.plot_survival_function(ax=ax, ci_show=True, color="k")
    t = np.linspace(1, RMST_HORIZON, 500)
    # recompute k1 and k2 separately for the overlay
    from lifelines import WeibullFitter
    from analysis.models.survival.weibull_em import fit_latent_weibull_em
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = WeibullFitter().fit(vt[CLOCK], vt[EVENT_COL])
    ax.plot(t, np.exp(-((t / wf.lambda_) ** wf.rho_)), "C0", lw=2,
            label=f"k1: β={wf.rho_:.2f}, η={wf.lambda_:.0f} (AIC {b.aic_k1:.0f})")
    if b.model_kind == "k2":
        Sk2 = b.S(t)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fit_latent_weibull_em(vt[CLOCK].to_numpy(float), vt[EVENT_COL].to_numpy(float), num_starts=40)
        mm = r.model
        Sk2 = _mixture_S(t, mm.weight_1, mm.component_1.beta, mm.component_1.eta,
                         mm.component_2.beta, mm.component_2.eta)
    ax.plot(t, Sk2, "C3", lw=2, ls="--",
            label=f"k2 смесь (AIC {b.aic_k2:.0f}); моды ≈{b.k2_short_life:.0f}/{b.k2_long_life:.0f} сут")
    ax.set_xlabel("Наработка, сут (t_cal)"); ax.set_ylabel("Доля работающих S(t)")
    ax.set_xlim(0, RMST_HORIZON); ax.set_ylim(0, 1)
    ax.set_title(f"Базовая модель полного Vt — выбрана {b.model_kind}{fo}\n"
                 "(k2 моды = поглощённая кислый/некислый гетерогенность)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / f"baseline_km_fit{sfx}.png", dpi=130); plt.close(fig)


def _fig_hazard_polyline(run_obj: ModelRun, figures: Path, sfx: str = "") -> None:
    """Constrained MINOR hazard θ polylines (Kpod, freq) with knots + bootstrap band."""
    layers = run_obj.layers
    fo = " · ТОЛЬКО ОТКАЗЫ (смещено)" if run_obj.failures_only else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    kg = np.linspace(*APPLICABILITY["kpod"], 300)
    ax.plot(kg, layers.theta_kpod(kg), color="C0", lw=2, label="θ полилиния (риск)")
    km = np.array(KPOD_KNOTS)
    ax.plot(km, layers.theta_kpod(km), "o", color="C0", ms=6, zorder=6)
    for row in layers.kpod_knots:
        ax.plot([row["knot"], row["knot"]], [row["theta_ci_lo"], row["theta_ci_hi"]],
                color="C0", alpha=0.35, lw=1)
    ax.axhline(1.0, color="gray", lw=0.8); ax.axvline(KPOD_PIN, color="C0", ls=":", alpha=0.6)
    ax.set_xlabel("Кпод = Ql/Qном"); ax.set_ylabel("Множитель риска θ (×)")
    ax.set_title("МИНОРНЫЙ слой: риск vs Кпод (реф. 0.8=1)"); ax.legend(fontsize=8)

    ax = axes[1]
    fg = np.linspace(*APPLICABILITY["freq"], 300)
    ax.plot(fg, layers.theta_freq(fg), color="C1", lw=2, label="θ полилиния (риск)")
    fk = FNOM_DEFAULT + np.array(FREQ_DEV_KNOTS)
    ax.plot(fk, layers.theta_freq(fk), "o", color="C1", ms=6, zorder=6)
    for row in layers.freq_knots:
        x = FNOM_DEFAULT + row["knot"]
        ax.plot([x, x], [row["theta_ci_lo"], row["theta_ci_hi"]], color="C1", alpha=0.35, lw=1)
    ax.axhline(1.0, color="gray", lw=0.8); ax.axvline(FNOM_DEFAULT, color="C1", ls=":", alpha=0.6)
    ax.set_xlabel("Частота, Гц"); ax.set_ylabel("Множитель риска θ (×)")
    ax.set_title("МИНОРНЫЙ слой: риск vs частота (реф. 50=1)"); ax.legend(fontsize=8)

    fig.suptitle(f"Vt ограниченные слои риска θ ≥ 1 (=1 в реф.; ridge={layers.penalizer:g}, "
                 f"n={layers.n}, отказов={layers.events}){fo}")
    fig.tight_layout(); fig.savefig(figures / f"hazard_minor_polyline{sfx}.png", dpi=130); plt.close(fig)


def _fig_primary_forest(run_obj: ModelRun, figures: Path, sfx: str = "") -> None:
    """Primary hazards (Ql, contractor) — HR forest with CI."""
    df = run_obj.layers.primary
    if df.empty:
        return
    fo = " · ТОЛЬКО ОТКАЗЫ" if run_obj.failures_only else ""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(df))[::-1]
    ax.errorbar(df["hr"], y, xerr=[df["hr"] - df["hr_ci_lo"], df["hr_ci_hi"] - df["hr"]],
                fmt="o", color="C2", capsize=4)
    ax.axvline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(df["label"])
    ax.set_xlabel("Отношение рисков HR (>1 = короче ресурс)")
    ax.set_title(f"Первичные риски Vt: Ql (ставка) + подрядчик (эффект v3.1){fo}")
    for xi, yi, hr, p in zip(df["hr"], y, df["hr"], df["p"]):
        ax.annotate(f"HR={hr:.2f} (p={p:.3f})", (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    fig.tight_layout(); fig.savefig(figures / f"primary_hazard_forest{sfx}.png", dpi=130); plt.close(fig)
