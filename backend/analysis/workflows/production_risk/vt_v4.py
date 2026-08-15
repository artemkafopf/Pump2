"""**Vt v4** — the nameplate-rate operating-life model.

    h(t | x) = h₀(t; β₀, η₀) · level_contractor · θ_Qnom(Qnom) · θ_freq(f) · θ_Kpod(Kpod)

Strata ``sour`` / ``nonsour``; clock ``t_cal``; cause-specific failure; population = the v3.2
well-level sour relabel (see :mod:`vt_v32_hybrid`).

**What changed from v3.2 / the composed model, and why.**

*Rate is nameplate, not realised.*  ``Qnom`` (``nominal_flow_m3d``) replaces ``Ql``.  On Vt,
Ql dies inside Qnom (HR 0.99 [0.65,1.52], p=0.98 nonsour; 1.07, p=0.66 pooled) while Qnom
survives inside Ql (1.53, p=0.031), Qnom-only wins AIC in every stratum, and 5-fold CV prefers
it in **100 %** of repeats on nonsour and pooled.  This replicates Ya v2.1.  It is also the
causally cleaner variable: the pump is chosen at install, so its nameplate cannot be
contaminated by how the run ended, unlike a run-average liquid rate.

⚠ **RETRACTED (2026-07-27): "Kpod is the Ql residual, measured at 0.99, p = 0.98."**  Two
errors compounded there and the conclusion — ``KPOD_OVERLAY = False`` — is wrong.

1. ``kpod_run`` is telemetry-derived and is **not** ``ql / qnom``.  The two coincide in the
   Свод panel, which is where the identity came from; in this modelling frame they do not
   (corr 0.82 on Vt, max discrepancy 2.6×).  Kpod is a free covariate, near-orthogonal to
   nameplate rate (corr(log Kpod, log Qnom) = −0.14).
2. Every null came from a **tent** pinned at 0.8 with θ ≥ 1 on both arms, a shape that cannot
   represent a monotone effect and therefore returns θ ≡ 1 whenever the truth is monotone.

On a free polyline the effect is large and clean: Vt HR 1.90 [1.47, 2.47], θ rising 0.58 →
1.27 across Kpod 0.2 → 1.6, worth **+8.4 out-of-sample log-likelihood units**.  See
:mod:`unified_v4`, which supersedes this layer choice.  ``KPOD_OVERLAY`` is left as-is only
so the shipped Excel calculator keeps reproducing; new work should use ``unified_v4``.

*Contractor is identified on overlap.*  brt and slb are badly imbalanced on rate, so a
contractor coefficient fitted jointly with the rate layer absorbs part of it.  Levels are
estimated once, on runs with **Ql in [200, 500]** (:data:`CONTRACTOR_WINDOW`) where the
contractors' rate distributions overlap, then held FIXED.  Estimates are stable across
candidate windows (nonsour slb 1.13–1.30, oth 1.89–2.72) and **slb is not significantly
different from brt in any of them** — treat the slb level as a level, not a finding.

*Layer order.*  Contractor levels → impose the Ya-transferred freq (and Kpod overlay if on) as
fixed offsets → fit θ_Qnom and the baseline against them.  v3.2 fitted Vt's own frequency and
then swapped in Ya's, which left the baseline matched to a layer the deliverable never used.

**Composition.**  θ's multiply; RMST multipliers DO NOT — always go through :meth:`V4Model.compose`.
Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline, median secondary.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.paths import results_dir
from analysis.workflows.production_risk import vt_composed_model as CM
from analysis.workflows.production_risk import vt_v32_hybrid as H

SLUG = "production_risk_vt_v4"
STRATA = ("nonsour", "sour")
CONTRACTOR_REF = "brt"
CONTRACTOR_TERMS = ("slb", "oth")
CGS = (CONTRACTOR_REF,) + CONTRACTOR_TERMS
CLOCK, EVENT_COL = H.CLOCK, H.EVENT_COL
RMST_HORIZON = 730.0

#: Contractor levels are estimated on this **Ql** window — where brt/slb rate distributions
#: overlap.  Measured translation: these runs have Kpod median 0.716, i.e. Qnom ≈ 320–500 (IQR).
CONTRACTOR_WINDOW = (200.0, 500.0)

#: Nameplate knots.  Extended past 900 on 2026-07-27: 52 runs / 30 events live above Qnom 1000,
#: so an earlier "saturation at 600" was a knot artefact, not the data.
QNOM_KNOTS = (125.0, 200.0, 250.0, 400.0, 600.0, 900.0, 1300.0, 1600.0)
QNOM_REF = 250.0
QNOM_GUARD = 125.0
#: Backstop only.  The lone 7000 m³/d nameplate that used to hit this cap was an unconverted
#: bbl/day figure (`ESP 538-7000`), now repaired at source by
#: :func:`analysis.data.pump_nominal_catalog.canonical_qnom` — with the catalogue on,
#: ``n_qnom_capped`` is 0 and real sizes top out at 1600.  Kept so a future bad row is clamped
#: and *counted* rather than silently steering the top knot.
QNOM_CAP = 1600.0

#: Reference point (all θ = 1): brt, Qnom 250, 50 Hz, Kpod 0.8.
FREQ_REF, KPOD_REF = CM.FREQ_REF, CM.KPOD_REF

#: Kpod overlay.  Default OFF — θ_Kpod ≡ 1.0, the measured null.  When True the Ya-transferred
#: bathtub from :mod:`vt_composed_model` is applied and the model then REQUIRES a realised rate.
KPOD_OVERLAY = False

#: Use the operator's own nameplate class where the designation supplies one
#: (:mod:`analysis.data.pump_nominal_catalog`).  Only MT designations are rewritten — REDA keeps
#: its published 50 Hz value and Russian names carry no nominal.  Measured impact on Vt is
#: negligible (107 runs move by ±10 % in both directions; β₀/η₀ shift <0.1 %, θ_Qnom ≤0.011,
#: contractor levels identical because they are estimated on the Ql window).  On by default
#: because it makes Qnom the number the operator plans with, not because it changes the fit.
USE_CANONICAL_QNOM = True

#: Tolerance for matching a run to its equipment-passport row; mirrors ``kpod_features``.
QNOM_JOIN_TOL_DAYS = 30

_QN_PIN = QNOM_KNOTS.index(QNOM_REF)
_N_QN = len(QNOM_KNOTS) - 1


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def attach_pump_designation(vt: pd.DataFrame, *, tol_days: int | None = None) -> pd.DataFrame:
    """Attach ``gno_type`` from the equipment passport, reusing the pipeline's own join.

    Deliberately calls :mod:`kpod_features`' ``_big_nominals``/``_asof_nearest`` rather than
    re-implementing a (well, install) match: a hand-rolled nearest-date join missed 23 runs
    whose ``kpod_qnom_source`` is already ``big``, i.e. the pipeline had matched them and the
    replica had not.
    """
    from analysis.workflows.production_risk import kpod_features as K

    tol = QNOM_JOIN_TOL_DAYS if tol_days is None else tol_days
    base = vt[["code", "install"]].copy()
    try:
        joined = K._asof_nearest(base, K._big_nominals(), tol)
    except Exception:                       # passport unavailable → keep recorded nameplates
        vt["gno_type"] = None
        return vt
    vt = vt.copy()
    vt["gno_type"] = joined["gno_type"].to_numpy()
    return vt


def prepare_frame(as_of: pd.Timestamp | None = None,
                  cached: pd.DataFrame | None = None,
                  *, canonical: bool | None = None) -> pd.DataFrame:
    """v3.2 relabeled Vt population with a capped, non-null nameplate rate.

    With ``canonical`` (default :data:`USE_CANONICAL_QNOM`) the operator's nameplate class
    replaces the recorded BEP flow wherever the designation supplies one; ``qnom_source``
    records which rule fired and ``qnom_recorded`` keeps the original for audit.
    """
    use_canon = USE_CANONICAL_QNOM if canonical is None else canonical
    vt = H.prepare_frame(as_of=as_of, cached=cached)
    vt["qnom_raw"] = pd.to_numeric(vt["nominal_flow_m3d"], errors="coerce")
    vt["ql"] = pd.to_numeric(vt["ql"], errors="coerce")
    vt = vt.dropna(subset=["qnom_raw"]).copy()
    vt["qnom_recorded"] = vt["qnom_raw"]

    if use_canon:
        from analysis.data.pump_nominal_catalog import canonical_qnom

        vt = attach_pump_designation(vt)
        res = [canonical_qnom(g, q) for g, q in zip(vt["gno_type"], vt["qnom_raw"])]
        vt["qnom_raw"] = [r[0] if r[0] is not None else np.nan for r in res]
        vt["pump_family"] = [r[1] for r in res]
        vt["qnom_source"] = [r[2] for r in res]
        vt = vt.dropna(subset=["qnom_raw"]).copy()
    else:
        vt["pump_family"] = "n/a"
        vt["qnom_source"] = "recorded"

    vt["qnom_capped"] = vt["qnom_raw"] > QNOM_CAP
    vt["qnom"] = vt["qnom_raw"].clip(upper=QNOM_CAP)
    return vt


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def theta_freq(f) -> np.ndarray:
    """Ya-transferred frequency layer (U-valley at 50 Hz, clamped [42, 65])."""
    return CM.theta_freq(f)


def theta_kpod(k, overlay: bool | None = None) -> np.ndarray:
    """Kpod overlay.  Default ≡ 1.0 (measured null); the Ya bathtub when switched on."""
    on = KPOD_OVERLAY if overlay is None else overlay
    if not on:
        return np.ones_like(np.asarray(k, float))
    return CM.theta_kpod(k)


def _iso(inc: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) log-θ, pinned to 0 at the reference knot."""
    full = np.concatenate([[0.0], np.cumsum(np.maximum(inc, 0.0))])
    return full - full[_QN_PIN]


def theta_qnom_from(knot_theta: np.ndarray, qnom) -> np.ndarray:
    """Evaluate the nameplate layer; flat below the guard and above the top knot."""
    q = np.clip(np.asarray(qnom, float), QNOM_GUARD, QNOM_KNOTS[-1])
    return np.exp(np.interp(q, np.asarray(QNOM_KNOTS, float), np.log(knot_theta)))


# ---------------------------------------------------------------------------
# Stage 1 — contractor levels on the overlap window
# ---------------------------------------------------------------------------
#: Adjust the level for the size layers it would otherwise absorb — see
#: :data:`unified_v4.CONTRACTOR_SIZE_ADJUST` for the measurement and the reasoning.
#: **Default False here on purpose**: v4 is the published fit of record (RMST ref 471.3 / 186.7,
#: slb 1.130 / oth 1.893) and must keep reproducing its own tables.  The corrected levels ship
#: from ``unified_v4``, which is what the calculators run.
CONTRACTOR_SIZE_ADJUST = False


def fit_contractor_levels(g: pd.DataFrame, window: tuple | None = None,
                          *, size_adjust: bool | None = None) -> dict:
    """Contractor hazard levels from the Ql overlap window (brt = 1.0), held fixed after.

    ``window`` is resolved at CALL time, not at definition time, so sensitivity runs that
    reassign :data:`CONTRACTOR_WINDOW` actually take effect.  ``size_adjust`` adds
    ``log(Qnom/250)`` and ``log(Kpod/0.8)`` so the level is a *reference-point* level rather
    than one averaged over each contractor's own pump mix.
    """
    from lifelines import CoxPHFitter

    adjust = CONTRACTOR_SIZE_ADJUST if size_adjust is None else size_adjust
    lo, hi = CONTRACTOR_WINDOW if window is None else window
    ov = g[(g["ql"] >= lo) & (g["ql"] <= hi)].copy()
    for cg in CONTRACTOR_TERMS:
        ov[cg] = (ov["contractor_group"] == cg).astype(float)
    cols = [CLOCK, EVENT_COL] + list(CONTRACTOR_TERMS)
    if adjust:
        ov["lq"] = np.log(pd.to_numeric(ov["qnom"], errors="coerce") / QNOM_REF)
        ov["lk"] = np.log(pd.to_numeric(ov["kpod_run"], errors="coerce") / KPOD_REF)
        ov = ov.dropna(subset=["lq", "lk"]).copy()
        for c in ("lq", "lk"):                       # standardised — see unified_v4
            ov[c] = ov[c] / max(float(ov[c].std()), 1e-9)
        cols += ["lq", "lk"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=0.1).fit(ov[cols], CLOCK, EVENT_COL)
    out = {CONTRACTOR_REF: 1.0}
    ci = {CONTRACTOR_REF: (1.0, 1.0, np.nan)}
    for cg in CONTRACTOR_TERMS:
        s = cph.summary.loc[cg]
        out[cg] = float(s["exp(coef)"])
        ci[cg] = (float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"]))
    return {"level": out, "ci": ci, "n": len(ov), "events": int(ov[EVENT_COL].sum())}


# ---------------------------------------------------------------------------
# Stage 2 — θ_Qnom and the baseline, with every other layer fixed
# ---------------------------------------------------------------------------
@dataclass
class StratumFit:
    stratum: str
    n: int
    events: int
    beta0: float
    eta0: float
    level: dict
    level_ci: dict
    theta_qnom: np.ndarray
    loglik: float
    window_n: int
    window_events: int
    n_capped: int


def fit_stratum(g: pd.DataFrame, *, overlay: bool | None = None) -> StratumFit:
    """Fit θ_Qnom + baseline with contractor, freq and Kpod imposed as fixed offsets."""
    lev = fit_contractor_levels(g)
    cc = g.dropna(subset=["qnom", "freq_run"]).copy()
    t = cc[CLOCK].to_numpy(float)
    e = cc[EVENT_COL].to_numpy(float)
    qn = cc["qnom"].to_numpy(float)
    off = (np.log(np.array([lev["level"][c] for c in cc["contractor_group"]]))
           + np.log(theta_freq(cc["freq_run"].to_numpy(float)))
           + np.log(theta_kpod(cc["kpod_run"].to_numpy(float), overlay=overlay)))

    def nll(p: np.ndarray) -> float:
        b0, le0 = p[0], p[1]
        if b0 <= 0:
            return 1e18
        lp = off + np.interp(np.clip(qn, QNOM_GUARD, QNOM_KNOTS[-1]),
                             np.asarray(QNOM_KNOTS, float), _iso(p[2:]))
        lt = np.log(t)
        base = np.clip(b0 * (lt - le0), -700.0, 700.0)
        ll = e * (lp + np.log(b0) - le0 + (b0 - 1.0) * (lt - le0)) - np.exp(lp) * np.exp(base)
        v = -float(np.sum(ll))
        return v if np.isfinite(v) else 1e18

    bounds = [(0.05, 6.0), (np.log(5.0), np.log(5e4))] + [(0.0, 4.0)] * _N_QN
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s0 in (0.0, 0.05, 0.15):
            r = minimize(lambda p: nll(p) + H.LAM_QL * float(np.dot(p[2:], p[2:])),
                         np.array([1.1, np.log(400.0)] + [s0] * _N_QN),
                         method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
    p = best.x
    return StratumFit(
        stratum=str(g["h2s_class"].iloc[0]), n=len(cc), events=int(e.sum()),
        beta0=float(p[0]), eta0=float(np.exp(p[1])),
        level=lev["level"], level_ci=lev["ci"],
        theta_qnom=np.exp(_iso(p[2:])), loglik=-nll(p),
        window_n=lev["n"], window_events=lev["events"],
        n_capped=int(g["qnom_capped"].sum()),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 4000) -> float:
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


@dataclass
class V4Model:
    fits: dict
    support: dict = dc_field(default_factory=dict)   # (stratum, contractor) -> (qnom_lo, hi, n, ev)

    def baseline(self, stratum: str) -> tuple[float, float]:
        f = self.fits[stratum]
        return f.beta0, f.eta0

    def rmst_ref(self, stratum: str, horizon: float = RMST_HORIZON) -> float:
        b0, e0 = self.baseline(stratum)
        return rmst(e0, b0, horizon)

    def theta_qnom(self, stratum: str, qnom) -> np.ndarray:
        return theta_qnom_from(self.fits[stratum].theta_qnom, qnom)

    def contractor_level(self, stratum: str, contractor: str) -> float:
        return self.fits[stratum].level[contractor]

    def in_support(self, stratum: str, contractor: str, qnom: float) -> bool:
        """Is this nameplate inside the contractor's own observed range?"""
        lo, hi, *_ = self.support[(stratum, contractor)]
        return bool(lo <= float(qnom) <= hi)

    def compose(self, stratum: str, *, contractor: str = CONTRACTOR_REF,
                qnom: float = QNOM_REF, freq: float = FREQ_REF,
                kpod: float = KPOD_REF, overlay: bool | None = None,
                horizon: float = RMST_HORIZON) -> dict:
        """Multiply the θ's, then convert ONCE.  RMST multipliers are not multiplicative."""
        b0, e0 = self.baseline(stratum)
        parts = {
            "contractor": float(self.contractor_level(stratum, contractor)),
            "Qnom": float(self.theta_qnom(stratum, np.array([float(qnom)]))[0]),
            "freq": float(theta_freq(float(freq))),
            "Kpod": float(theta_kpod(np.array([float(kpod)]), overlay=overlay)[0]),
        }
        theta = float(np.prod(list(parts.values())))
        eta_eff = e0 * theta ** (-1.0 / b0)
        r0 = rmst(e0, b0, horizon)
        r = rmst(eta_eff, b0, horizon)
        return {"stratum": stratum, "theta_total": theta, "theta_parts": parts,
                "beta0": b0, "eta0": e0, "eta_eff": eta_eff,
                "rmst_ref": r0, "rmst": r, "rmst_mult": r / r0,
                "median": eta_eff * np.log(2.0) ** (1.0 / b0),
                "in_support": self.in_support(stratum, contractor, qnom)
                if self.support else None}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class ModelRun:
    model: V4Model
    spec: pd.DataFrame
    baseline_table: pd.DataFrame
    support_table: pd.DataFrame
    vt: pd.DataFrame


def run(*, as_of: pd.Timestamp | None = None, cached: pd.DataFrame | None = None,
        write: bool = True, overlay: bool | None = None) -> ModelRun:
    vt = prepare_frame(as_of=as_of, cached=cached)
    fits = {s: fit_stratum(vt[vt["h2s_class"] == s], overlay=overlay) for s in STRATA}
    support = {}
    for s in STRATA:
        g = vt[vt["h2s_class"] == s]
        for cg in CGS:
            b = g[g["contractor_group"] == cg]
            # p10–p90, NOT min–max: the extremes are single runs (nonsour brt spans 35–1600),
            # so a min–max window would restrict nothing.  Bootstrap put the CI on θ under
            # ~2.5× inside p10–p90 and up to 12× outside it.
            support[(s, cg)] = ((float(b["qnom"].quantile(0.10)),
                                 float(b["qnom"].quantile(0.90)),
                                 len(b), int(b[EVENT_COL].sum())) if len(b)
                                else (np.nan, np.nan, 0, 0))
    model = V4Model(fits=fits, support=support)
    obj = ModelRun(model=model, spec=_spec_table(model),
                   baseline_table=_baseline_table(model),
                   support_table=_support_table(model), vt=vt)
    if write:
        _write_outputs(obj)
    return obj


def _baseline_table(m: V4Model) -> pd.DataFrame:
    rows = []
    for s in STRATA:
        f = m.fits[s]
        rows.append({"stratum": f"Vt_{s}", "n": f.n, "events": f.events,
                     "beta0": round(f.beta0, 4), "eta0": round(f.eta0, 1),
                     "rmst730_ref": round(m.rmst_ref(s), 1),
                     "median_ref": round(f.eta0 * np.log(2) ** (1 / f.beta0)),
                     "contractor_window_n": f.window_n,
                     "contractor_window_events": f.window_events,
                     "n_qnom_capped": f.n_capped, "loglik": round(f.loglik, 3),
                     "clock": CLOCK, "kpod_overlay": bool(KPOD_OVERLAY)})
    return pd.DataFrame(rows)


def _spec_table(m: V4Model) -> pd.DataFrame:
    rows = []
    for s in STRATA:
        b0, e0 = m.baseline(s)
        r0 = m.rmst_ref(s)
        f = m.fits[s]
        for cg in CGS:
            lo, hi, p = f.level_ci[cg]
            th = f.level[cg]
            rows.append({"stratum": f"Vt_{s}", "component": "contractor",
                         "term": cg, "knot": np.nan, "theta": round(th, 4),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "p": round(p, 4),
                         "rmst730_mult": round(rmst(e0 * th ** (-1 / b0), b0) / r0, 4)})
        for k, th in zip(QNOM_KNOTS, f.theta_qnom):
            rows.append({"stratum": f"Vt_{s}", "component": "theta_Qnom", "term": "Qnom",
                         "knot": k, "theta": round(float(th), 4),
                         "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                         "rmst730_mult": round(rmst(e0 * th ** (-1 / b0), b0) / r0, 4)})
        for fq in (42, 45, 50, 55, 60, 65):
            th = float(theta_freq(fq))
            rows.append({"stratum": f"Vt_{s}", "component": "theta_freq", "term": "freq_hz",
                         "knot": fq, "theta": round(th, 4),
                         "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                         "rmst730_mult": round(rmst(e0 * th ** (-1 / b0), b0) / r0, 4)})
    return pd.DataFrame(rows)


def _support_table(m: V4Model) -> pd.DataFrame:
    rows = []
    for (s, cg), (lo, hi, n, ev) in m.support.items():
        rows.append({"stratum": f"Vt_{s}", "contractor": cg,
                     "qnom_p10": round(lo), "qnom_p90": round(hi), "n": n, "events": ev,
                     "note": "confident window; theta is clamped flat outside the knot grid "
                             "and outside this window it is an extrapolation across "
                             "contractors"})
    return pd.DataFrame(rows).sort_values(["stratum", "contractor"]).reset_index(drop=True)


def _write_outputs(obj: ModelRun) -> Path:
    out = results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    obj.baseline_table.to_csv(tables / "baseline.csv", index=False, encoding="utf-8-sig")
    obj.spec.to_csv(tables / "model_spec.csv", index=False, encoding="utf-8-sig")
    obj.support_table.to_csv(tables / "support.csv", index=False, encoding="utf-8-sig")
    _fig_layers(obj.model, figures)
    return out


def _fig_layers(m: V4Model, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ccol = {"brt": "#2471a3", "slb": "#e67e22", "oth": "#7d3c98"}
    slab = {"nonsour": "несернистые (Vt_nonsour)", "sour": "сернистые (Vt_sour)"}
    q = np.linspace(QNOM_GUARD, QNOM_KNOTS[-1], 700)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))
    for j, s in enumerate(STRATA):
        b0, e0 = m.baseline(s)
        r0 = m.rmst_ref(s)
        ax_t, ax_r = axes[0, j], axes[1, j]
        base = m.theta_qnom(s, q)
        for cg in CGS:
            lo, hi, n, ev = m.support[(s, cg)]
            th = m.contractor_level(s, cg) * base
            rm = np.array([rmst(e0 * v ** (-1 / b0), b0) / r0 for v in th])
            inside = (q >= lo) & (q <= hi)
            for ax, y in ((ax_t, th), (ax_r, rm)):
                ax.plot(q, y, color=ccol[cg], lw=1.2, ls=(0, (4, 3)), alpha=.5, zorder=3)
                ax.plot(np.where(inside, q, np.nan), np.where(inside, y, np.nan),
                        color=ccol[cg], lw=2.7, zorder=4,
                        label=f"{cg} ×{m.contractor_level(s, cg):.2f} (n={n}, отк. {ev})")
        for ax in (ax_t, ax_r):
            ax.axvspan(320, 500, color="#27ae60", alpha=.10, lw=0)
            ax.axvline(QNOM_REF, ls=":", color="#555555", lw=1.0)
            ax.axhline(1.0, color="#888888", lw=.9)
            ax.grid(alpha=.2); ax.set_axisbelow(True)
            ax.set_xlabel("Qном, м³/сут"); ax.set_xlim(QNOM_GUARD, QNOM_KNOTS[-1])
        ax_t.set_ylabel("θ = подрядчик × Qном")
        ax_r.set_ylabel(f"множитель RMST(0,730)  [база {r0:.0f} сут]")
        ax_t.set_title(f"{slab[s]} — θ", fontsize=11)
        ax_r.set_title(f"{slab[s]} — множитель наработки", fontsize=11)
        ax_t.legend(fontsize=8, loc="upper left")
        ax_r.legend(fontsize=8, loc="lower left")
    fig.suptitle("Vt v4 — номинальный дебит × подрядчик (частота из Ya, Кпод — опциональная "
                 "накладка, по умолчанию 1.0)\n"
                 "зелёное — окно оценки уровня подрядчика (Ql 200–500 ≈ Qном 320–500) · "
                 "штрих — вне наблюдённого диапазона подрядчика", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(figures / "v4_qnom_contractor_layers.png", dpi=140)
    plt.close(fig)
