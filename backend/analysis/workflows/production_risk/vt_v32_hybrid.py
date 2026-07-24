"""Vt **v3.2 hybrid θ-model** — the deployable survival model on the v3.2 relabeled population.

Population: :mod:`esp_population` with ``config.SOUR_WELL_LEVEL_ALL_RUNS`` forced True (the
v3.2 well-level sour relabel — recovers the 30 running pumps the Свод flag never stamped,
because that flag is written only from the failure/workover DB).  Vt only, split sour/nonsour,
clock ``t_cal``, cause-specific failure.

Form (proportional hazards on a Weibull baseline)::

    h(t | x) = h0(t; β0, η0) · HR_contractor · θ_Ql(Ql) · θ_Kpod(Kpod) · θ_freq(f − f_nom)

* **baseline** and **contractor** are per stratum (brt = reference).
* **θ_Ql** — MONOTONE hazard polyline (data-driven, no prior shape), θ=1 at Ql=250.
  *Guarded*: held flat below ``QL_GUARD`` — the sour ×1.8 RMST bonus at Ql<100 rested on a
  handful of runs and is not trusted, so no extrapolated benefit outside supported range.
* **θ_Kpod** — TENT: θ minimum = 1 at Kpod 0.8, hazard monotone non-decreasing on both arms
  (⇔ life maximal at 0.8).  Empirically the left arm sits at the θ=1 constraint boundary — the
  data has no valley at 0.8 (see ``project_vt_kpod_freq_reconciliation``), so this behaves as
  an **overload-only dial**.
* **θ_freq** — same TENT form at the nominal frequency.

**Pooling (hybrid).**  ``Ql`` and ``Kpod`` are POOLED across strata (weighted geometric mean of
θ at the knots, weights = stratum fleet size) because the strata agree there and the signal is
weak; ``freq`` is kept STRATUM-SPECIFIC because pooling dilutes sour's real under-speed penalty
into a fleet-wide value matching neither stratum.  Baseline/contractor are never pooled.

**Applicability.**  θ is *clamped* outside the knot range (flat extrapolation).  ``APPLICABILITY``
records the supported window; beyond it the returned θ is an assumption, not an estimate.
Frequency is bounded at 60 Hz because the fleet has **zero** runs above 63 Hz — do not extend.
Kpod runs to 1.5 (25 runs / 15 events live in 1.2–1.5+ and the crude hazard keeps rising there);
above 1.5 it is clamped flat, which *under*-penalises extreme overload.

**Composition.**  θ's multiply; RMST multipliers DO NOT.  Always go through :func:`compose`.

Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline; median secondary.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import vt_ttf_covariates as V

# ---------------------------------------------------------------------------
# Policy / constants
# ---------------------------------------------------------------------------
SLUG = "production_risk_vt_v32_hybrid"
FIELD = "Vt"
CLOCK = V.CLOCK_PRIMARY            # "t_cal"
EVENT_COL = V.EVENT_COL            # "event"
STRATA = ("nonsour", "sour")
CONTRACTOR_REF = "brt"
CONTRACTOR_TERMS = ("slb", "oth")
RMST_HORIZON = 730.0

#: Ql: monotone arm, θ=1 at the pin; held flat below ``QL_GUARD`` (untrusted low-Ql bonus).
QL_KNOTS = (47.0, 100.0, 250.0, 450.0, 823.0)
QL_PIN = 250.0
QL_GUARD = 100.0

#: Kpod: tent pinned at 0.8.  Extended to 1.5 (2026-07-24): 25 runs / 15 events sit above 1.2
#: and the crude hazard keeps climbing (1.19× then 1.35× vs the 0.8–1.0 band), so stopping at
#: 1.2 under-penalised overload.  NO knot at 2.0 — only ~5 runs live there.
KPOD_KNOTS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
KPOD_PIN = 0.8

#: freq deviation (f − f_nom): tent pinned at nominal.  Last knot +10 Hz (=60 Hz); the fleet's
#: max observed frequency is 63 Hz and there are ZERO runs above 66 Hz — never extend to 70.
FREQ_DEV_KNOTS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0)
FREQ_PIN = 0.0

#: Arms pooled across strata (weighted geometric mean of θ); freq stays stratum-specific.
POOLED_ARMS = ("Ql", "Kpod")

#: Ridge on the polyline increments — keeps Kpod/freq MINOR, lets the primary Ql move.
LAM_QL, LAM_KPOD, LAM_FREQ = 1.0, 6.0, 6.0

#: Supported covariate window.  Outside it θ is clamped (flat) — an assumption, not a fit.
APPLICABILITY = {
    "ql": (47.0, 823.0),
    "kpod": (0.4, 1.5),          # clamped flat 1.5→2.0; under-penalises extreme overload
    "freq_hz": (35.0, 60.0),     # zero fleet observations above 63 Hz
}

_ARMS = ("Ql", "Kpod", "freq_dev")
_KNOTS = {"Ql": QL_KNOTS, "Kpod": KPOD_KNOTS, "freq_dev": FREQ_DEV_KNOTS}
_PIN = {"Ql": QL_PIN, "Kpod": KPOD_PIN, "freq_dev": FREQ_PIN}
_COL = {"Ql": "ql", "Kpod": "kpod_run", "freq_dev": "freq_dev"}

_N_QL = len(QL_KNOTS) - 1
_KP_L = sum(k < KPOD_PIN for k in KPOD_KNOTS)
_KP_R = sum(k > KPOD_PIN for k in KPOD_KNOTS)
_FR_L = sum(k < FREQ_PIN for k in FREQ_DEV_KNOTS)
_FR_R = sum(k > FREQ_PIN for k in FREQ_DEV_KNOTS)
_N_INC = _N_QL + _KP_L + _KP_R + _FR_L + _FR_R
_QL_PIN_IX = QL_KNOTS.index(QL_PIN)
_KP_PIN_IX = KPOD_KNOTS.index(KPOD_PIN)
_FR_PIN_IX = FREQ_DEV_KNOTS.index(FREQ_PIN)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def prepare_frame(as_of: pd.Timestamp | None = None,
                  cached: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vt modelling frame on the **v3.2 relabeled** population (flag forced, then restored)."""
    if cached is not None:
        df = cached
    else:
        prev = C.SOUR_WELL_LEVEL_ALL_RUNS
        C.SOUR_WELL_LEVEL_ALL_RUNS = True          # v3.2 relabel
        try:
            df, _ = V.build_frame(as_of=as_of)
        finally:
            C.SOUR_WELL_LEVEL_ALL_RUNS = prev
    vt = df[df["field"] == FIELD].copy()
    vt[CLOCK] = pd.to_numeric(vt[CLOCK], errors="coerce")
    vt = vt[vt[CLOCK] > 0].dropna(subset=[CLOCK, EVENT_COL]).copy()
    fn = pd.to_numeric(vt.get("nominal_freq_hz"), errors="coerce")
    vt["f_nom"] = np.where(fn.between(40.0, 65.0), fn, 50.0)
    vt["freq_dev"] = pd.to_numeric(vt["freq_run"], errors="coerce") - vt["f_nom"]
    for cg in CONTRACTOR_TERMS:
        vt[cg] = (vt["contractor_group"] == cg).astype(float)
    return vt


# ---------------------------------------------------------------------------
# Constrained polyline machinery (isotonic: non-negative increments)
# ---------------------------------------------------------------------------
def _mono_full(inc: np.ndarray) -> np.ndarray:
    """Monotone arm (Ql): cumulative non-negative increments, pinned to 0 at the reference."""
    full = np.concatenate([[0.0], np.cumsum(np.maximum(inc, 0.0))])
    return full - full[_QL_PIN_IX]


def _tent_full(n: int, pin_ix: int, inc_l, inc_r) -> np.ndarray:
    """Tent arm: 0 at the pin, cumulative non-negative increments outward on each side.

    ⇒ θ = exp(·) ≥ 1 everywhere, = 1 at the pin, monotone on each arm (life maximal at pin).
    """
    full = np.zeros(n)
    c = 0.0
    for j, i in enumerate(range(pin_ix - 1, -1, -1)):
        c += max(0.0, inc_l[j]); full[i] = c
    c = 0.0
    for j, i in enumerate(range(pin_ix + 1, n)):
        c += max(0.0, inc_r[j]); full[i] = c
    return full


def _split(p: np.ndarray):
    o = 4
    ql = p[o:o + _N_QL]; o += _N_QL
    kl = p[o:o + _KP_L]; o += _KP_L
    kr = p[o:o + _KP_R]; o += _KP_R
    fl = p[o:o + _FR_L]; o += _FR_L
    fr = p[o:o + _FR_R]
    return ql, kl, kr, fl, fr


def _fulls(p: np.ndarray) -> dict[str, np.ndarray]:
    ql, kl, kr, fl, fr = _split(p)
    return {"Ql": _mono_full(ql),
            "Kpod": _tent_full(len(KPOD_KNOTS), _KP_PIN_IX, kl, kr),
            "freq_dev": _tent_full(len(FREQ_DEV_KNOTS), _FR_PIN_IX, fl, fr)}


def _sk(x, knots, full) -> np.ndarray:
    """Polyline in LOG-hazard, linear between knots, **clamped** outside (flat extrapolation)."""
    knots = np.asarray(knots, float)
    return np.interp(np.clip(np.asarray(x, float), knots[0], knots[-1]), knots, full)


def _nll(p, t, e, ql, slb, oth, kpod, fdev) -> float:
    beta0, ln_eta0, b_slb, b_oth = p[:4]
    if beta0 <= 0:
        return 1e18
    f = _fulls(p)
    lp = (b_slb * slb + b_oth * oth
          + _sk(ql, QL_KNOTS, f["Ql"])
          + _sk(kpod, KPOD_KNOTS, f["Kpod"])
          + _sk(fdev, FREQ_DEV_KNOTS, f["freq_dev"]))
    ln_t = np.log(t)
    base = np.clip(beta0 * (ln_t - ln_eta0), -700.0, 700.0)
    ll = e * (lp + np.log(beta0) - ln_eta0 + (beta0 - 1.0) * (ln_t - ln_eta0)) - np.exp(lp) * np.exp(base)
    v = -float(np.sum(ll))
    return v if np.isfinite(v) else 1e18


def _obj(p, *args) -> float:
    nll = _nll(p, *args)
    if nll >= 1e18:
        return 1e18
    ql, kl, kr, fl, fr = _split(p)
    ridge = (LAM_QL * float(np.dot(ql, ql))
             + LAM_KPOD * (float(np.dot(kl, kl)) + float(np.dot(kr, kr)))
             + LAM_FREQ * (float(np.dot(fl, fl)) + float(np.dot(fr, fr))))
    return nll + ridge


@dataclass
class StratumFit:
    stratum: str
    n: int
    events: int
    cc_n: int
    cc_events: int
    beta0: float
    eta0: float
    hr_slb: float
    hr_oth: float
    theta_knots: dict          # arm -> np.ndarray of θ at that arm's knots
    loglik: float
    beta_marginal: float
    eta_marginal: float


def fit_stratum(g: pd.DataFrame) -> StratumFit:
    """Constrained Weibull-PH MLE on complete-case rows of one stratum."""
    from lifelines import WeibullFitter

    cc = g.dropna(subset=["ql", "kpod_run", "freq_run"]).copy()
    args = (cc[CLOCK].to_numpy(float), cc[EVENT_COL].to_numpy(float),
            cc["ql"].to_numpy(float), cc["slb"].to_numpy(float), cc["oth"].to_numpy(float),
            cc["kpod_run"].to_numpy(float), cc["freq_dev"].to_numpy(float))
    bounds = ([(0.05, 6.0), (np.log(5.0), np.log(5e4)), (-4.0, 4.0), (-4.0, 4.0)]
              + [(0.0, 4.0)] * _N_INC)
    starts = [np.array([0.9, np.log(400.0), 0.0, 0.0] + [0.0] * _N_INC),
              np.array([1.1, np.log(300.0), 0.2, 0.6] + [0.05] * _N_INC),
              np.array([0.7, np.log(500.0), -0.2, 0.3] + [0.10] * _N_INC)]
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            r = minimize(_obj, x0, args=args, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        wf = WeibullFitter().fit(g[CLOCK], g[EVENT_COL])
    p = best.x
    fulls = _fulls(p)
    return StratumFit(
        stratum=str(g["h2s_class"].iloc[0]), n=len(g), events=int(g[EVENT_COL].sum()),
        cc_n=len(cc), cc_events=int(args[1].sum()),
        beta0=float(p[0]), eta0=float(np.exp(p[1])),
        hr_slb=float(np.exp(p[2])), hr_oth=float(np.exp(p[3])),
        theta_knots={a: np.exp(fulls[a]) for a in _ARMS},
        loglik=-_nll(p, *args),
        beta_marginal=float(wf.rho_), eta_marginal=float(wf.lambda_),
    )


# ---------------------------------------------------------------------------
# Hybrid model (pooling + guard) and the composition rule
# ---------------------------------------------------------------------------
def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 3000) -> float:
    """RMST(0, horizon) of Weibull(beta, eta)."""
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


@dataclass
class HybridModel:
    """Fitted hybrid model: pooled Ql/Kpod θ, stratum-specific freq θ, per-stratum baselines."""
    fits: dict                     # stratum -> StratumFit
    weights: dict                  # stratum -> pooling weight (fleet share)
    theta: dict                    # (arm, stratum) -> np.ndarray θ at knots (post pool+guard)

    # -- θ evaluation --------------------------------------------------------
    def theta_at(self, arm: str, stratum: str, x) -> np.ndarray:
        """θ(x) for one arm, clamped outside the knot range (see APPLICABILITY)."""
        th = self.theta[(arm, stratum)]
        return np.exp(_sk(x, _KNOTS[arm], np.log(th)))

    def baseline(self, stratum: str) -> tuple[float, float]:
        f = self.fits[stratum]
        return f.beta0, f.eta0

    def rmst_ref(self, stratum: str, horizon: float = RMST_HORIZON) -> float:
        b0, e0 = self.baseline(stratum)
        return rmst(e0, b0, horizon)

    def contractor_hr(self, stratum: str, contractor: str) -> float:
        f = self.fits[stratum]
        return {"brt": 1.0, "slb": f.hr_slb, "oth": f.hr_oth}[contractor]

    # -- the ONE correct way to combine effects ------------------------------
    def compose(self, stratum: str, *, contractor: str = CONTRACTOR_REF,
                ql: float | None = None, kpod: float | None = None,
                freq_dev: float | None = None, horizon: float = RMST_HORIZON) -> dict:
        """Combine covariates properly: multiply θ's, then convert ONCE.

        RMST multipliers are **not** multiplicative (RMST is a saturating functional of the
        hazard) — chaining them overstates life.  Always compose here.
        """
        b0, e0 = self.baseline(stratum)
        theta = self.contractor_hr(stratum, contractor)
        parts = {"contractor": self.contractor_hr(stratum, contractor)}
        for arm, val in (("Ql", ql), ("Kpod", kpod), ("freq_dev", freq_dev)):
            if val is None:
                continue
            th = float(self.theta_at(arm, stratum, np.array([float(val)]))[0])
            parts[arm] = th
            theta *= th
        eta_eff = e0 * theta ** (-1.0 / b0)
        r0 = rmst(e0, b0, horizon)
        r = rmst(eta_eff, b0, horizon)
        return {"stratum": stratum, "theta_total": theta, "theta_parts": parts,
                "beta0": b0, "eta0": e0, "eta_eff": eta_eff,
                "rmst_ref": r0, "rmst": r, "rmst_mult": r / r0,
                "median": eta_eff * np.log(2.0) ** (1.0 / b0)}


def _pool(fits: dict, weights: dict, arm: str) -> np.ndarray:
    """Weighted geometric mean of θ across strata (pooling is linear in LOG-hazard)."""
    lg = sum(weights[s] * np.log(fits[s].theta_knots[arm]) for s in STRATA)
    return np.exp(lg)


def build_model(fits: dict) -> HybridModel:
    """Apply the hybrid policy: pool Ql/Kpod by fleet size, guard Ql, keep freq per stratum."""
    tot = sum(fits[s].n for s in STRATA)
    weights = {s: fits[s].n / tot for s in STRATA}
    theta: dict = {}
    for arm in _ARMS:
        pooled = _pool(fits, weights, arm) if arm in POOLED_ARMS else None
        for s in STRATA:
            th = (pooled if pooled is not None else fits[s].theta_knots[arm]).copy()
            if arm == "Ql":                       # guard: no extrapolated benefit below QL_GUARD
                k = np.asarray(QL_KNOTS, float)
                th[k < QL_GUARD] = float(np.interp(QL_GUARD, k, th))
            th = th / float(np.interp(_PIN[arm], _KNOTS[arm], th))   # re-pin θ=1 at reference
            theta[(arm, s)] = th
    return HybridModel(fits=fits, weights=weights, theta=theta)


# ---------------------------------------------------------------------------
# Orchestration + outputs
# ---------------------------------------------------------------------------
@dataclass
class ModelRun:
    model: HybridModel
    spec: pd.DataFrame
    baseline_table: pd.DataFrame
    vt: pd.DataFrame


def run(*, as_of: pd.Timestamp | None = None, cached: pd.DataFrame | None = None,
        write: bool = True) -> ModelRun:
    """Fit the v3.2 hybrid model end-to-end and (optionally) write tables + figures."""
    vt = prepare_frame(as_of=as_of, cached=cached)
    fits = {s: fit_stratum(vt[vt["h2s_class"] == s]) for s in STRATA}
    model = build_model(fits)
    baseline_table = _baseline_table(model)
    spec = _spec_table(model)
    run_obj = ModelRun(model=model, spec=spec, baseline_table=baseline_table, vt=vt)
    if write:
        _write_outputs(run_obj)
    return run_obj


def _baseline_table(m: HybridModel) -> pd.DataFrame:
    rows = []
    for s in STRATA:
        f = m.fits[s]
        rows.append({
            "stratum": f"Vt_{s}", "n": f.n, "events": f.events,
            "cc_n": f.cc_n, "cc_events": f.cc_events,
            "beta_marginal": round(f.beta_marginal, 3), "eta_marginal": round(f.eta_marginal, 1),
            "rmst730_marginal": round(rmst(f.eta_marginal, f.beta_marginal)),
            "beta0_ref": round(f.beta0, 3), "eta0_ref": round(f.eta0, 1),
            "rmst730_ref": round(m.rmst_ref(s), 1),
            "median_ref": round(f.eta0 * np.log(2) ** (1 / f.beta0)),
            "pool_weight": round(m.weights[s], 3), "clock": CLOCK,
        })
    return pd.DataFrame(rows)


def _spec_table(m: HybridModel) -> pd.DataFrame:
    rows = []
    for s in STRATA:
        b0, e0 = m.baseline(s)
        r0 = m.rmst_ref(s)
        for cg in CONTRACTOR_TERMS:
            hr = m.contractor_hr(s, cg)
            rows.append({"stratum": f"Vt_{s}", "component": "contractor", "scope": "stratum",
                         "term": f"contractor:{cg}", "knot": np.nan, "theta": round(hr, 3),
                         "rmst730_mult": round(rmst(e0 * hr ** (-1 / b0), b0) / r0, 3)})
        for arm in _ARMS:
            scope = "POOLED(fleet)" if arm in POOLED_ARMS else "stratum-specific"
            for k, th in zip(_KNOTS[arm], m.theta[(arm, s)]):
                rows.append({"stratum": f"Vt_{s}", "component": f"theta_{arm}", "scope": scope,
                             "term": arm, "knot": k, "theta": round(float(th), 3),
                             "rmst730_mult": round(rmst(e0 * th ** (-1 / b0), b0) / r0, 3)})
    return pd.DataFrame(rows)


def _write_outputs(run_obj: ModelRun) -> Path:
    out = results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    run_obj.baseline_table.to_csv(tables / "baseline.csv", index=False, encoding="utf-8-sig")
    run_obj.spec.to_csv(tables / "model_spec.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"covariate": k, "lo": v[0], "hi": v[1],
                   "note": "theta clamped flat outside this window (assumption, not a fit)"}
                  for k, v in APPLICABILITY.items()]).to_csv(
        tables / "applicability.csv", index=False, encoding="utf-8-sig")
    _fig_multipliers(run_obj.model, figures)
    return out


def _fig_multipliers(m: HybridModel, figures: Path) -> None:
    """θ and RMST(0,730) multiplier charts — piecewise-linear in log-θ (the model's own form).

    Deliberately NOT splined: a smooth interpolant invents curvature the fit never estimated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"Ql": "Ql, м3/сут", "Kpod": "Kpod = Ql/Qном", "freq_dev": "частота − номинал, Гц"}
    cs = {"nonsour": "#2471a3", "sour": "#c0392b"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
    for j, arm in enumerate(_ARMS):
        kn = np.asarray(_KNOTS[arm], float)
        xs = np.linspace(kn[0], kn[-1], 400)
        pooled = arm in POOLED_ARMS
        if pooled:
            th = m.theta_at(arm, STRATA[0], xs)
            ax[0, j].plot(xs, th, color="#111111", lw=2.6, label="объединённая (обе страты)")
            ax[0, j].plot(kn, m.theta_at(arm, STRATA[0], kn), "o", color="#111111", ms=5.5)
        for s in STRATA:
            b0, e0 = m.baseline(s); r0 = m.rmst_ref(s)
            th = m.theta_at(arm, s, xs)
            if not pooled:
                ax[0, j].plot(xs, th, color=cs[s], lw=2.4, label=s)
                ax[0, j].plot(kn, m.theta_at(arm, s, kn), "o", color=cs[s], ms=5.5)
            rm = np.array([rmst(e0 * t ** (-1 / b0), b0) / r0 for t in th])
            ax[1, j].plot(xs, rm, color=cs[s], lw=2.4, label=f"{s} (ref {r0:.0f} д)")
            thk = m.theta_at(arm, s, kn)
            ax[1, j].plot(kn, [rmst(e0 * t ** (-1 / b0), b0) / r0 for t in thk], "o",
                          color=cs[s], ms=5.5)
        if arm == "Ql":
            for r in (0, 1):
                ax[r, j].axvspan(kn[0], QL_GUARD, color="orange", alpha=.13)
        tag = "ОБЪЕДИНЁННАЯ" if pooled else "ПО СТРАТАМ"
        ax[0, j].set_title(f"θ хазард — {arm}  [{tag}]", fontsize=11)
        ax[1, j].set_title(f"RMST(730) множитель — {arm}", fontsize=11)
        for r in (0, 1):
            a = ax[r, j]
            a.axhline(1.0, color="gray", lw=.8)
            a.axvline(_PIN[arm], ls=":", color="k", alpha=.45)
            a.set_xlabel(labels[arm]); a.grid(alpha=.25); a.legend(fontsize=8)
    ax[0, 0].set_ylabel("θ (множитель риска)")
    ax[1, 0].set_ylabel("множитель RMST(0,730)")
    fig.suptitle("Vt v3.2 гибрид: Ql и Kpod объединены по флоту; частота — по стратам; "
                 "базовая линия своя у страты", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figures / "v32_hybrid_multipliers.png", dpi=140)
    plt.close(fig)
