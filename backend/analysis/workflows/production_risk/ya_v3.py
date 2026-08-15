"""**Ya v3** — blended-rate model: ``w·θ_Qnom + (1−w)·θ_Ql``, frequency, contractor.  No Kpod.

    h(t | x) = h₀(t; β₀, η₀) · HR_contractor · [w·θ_Qnom(Qnom) + (1−w)·θ_Ql(Ql)] · θ_freq(f−f_nom)

Sits alongside :mod:`ya_k1k2_hybrid` (v2 = Ql arm, v2.1 = Qnom arm); all three coexist and write
to separate slugs.

**What v3 asks that v2/v2.1 could not.**  The two earlier versions each commit to one rate
variable, so "does Qnom subsume Ql?" had to be answered by comparing separate fits.  v3 puts
both inside ONE likelihood with a mixing weight, so the question becomes a parameter.

**The blend is ARITHMETIC on θ, not geometric** — ``w·θ_Qnom + (1−w)·θ_Ql``, per the model
owner's specification.  This is deliberate and must not be "simplified" to ``θ_Qnom^w ·
θ_Ql^(1−w)``: the geometric form is the natural one in log-hazard and gives different numbers.
Both preserve θ = 1 at the reference (each arm is pinned there), so that is not a discriminator.
Because the blend is a sum of θ's rather than a product, this model cannot use
``fit_polyline_ph`` — the likelihood is written out directly in :func:`fit_layers`.

⚠ **RETRACTED (2026-07-27): "Kpod is absent by measurement."**  The ten fits behind that claim
all used a **tent** pinned at 0.8 with θ ≥ 1 on both arms — a shape structurally unable to
represent a monotone effect, which therefore returns θ ≡ 1 whenever the truth is monotone.
The supporting argument was also wrong: ``kpod_run`` is telemetry-derived and is **not**
``ql / qnom`` (that identity holds only in the Свод panel), so Kpod is not the Ql residual and
carrying Qnom + Kpod is not over-parameterisation — corr(log Kpod, log Qnom) = 0.011 on Ya.

On a free polyline Ya's Kpod effect is real but small: HR 1.28 [1.11, 1.46], concentrated at
low loading (θ 0.78 [0.66, 0.92] at Kpod 0.2, flat above the reference), worth +2.0
out-of-sample log-likelihood units against +8.4 on Vt.  The tent scored **−1.4** on Ya, i.e.
it was actively worse than no layer.  :mod:`unified_v4` supersedes this layer choice.

What survives unchanged: ``Ql`` itself has no place alongside Qnom and Kpod, since
``log Ql = log Qnom + log Kpod`` makes it their product.

⚠ **``w`` IS NOT IDENTIFIED.**  Profile likelihood over w (2026-07-27, n=1227 / 635 events):
ML w = 0.50, but the 95 % interval is **[0.10, 1.00]** and the profile varies by only 0.33
log-likelihood units across w ∈ [0.3, 1.0].  What the data *does* reject is **w = 0, pure Ql**
(Δloglik 4.06 against the maximum).  So v3 supports "the rate effect is mostly or entirely
nameplate" and refutes "it is purely realised flow", but cannot place the mixture.  Read
:data:`W_PROFILE_CI` before quoting :data:`W_ML` as if it were estimated precisely.

**vs v2.1**, same 1227 runs and — coincidentally — the same 21 parameters: AIC 9974.5 vs 9975.5.
A wash.  v3 is preferred for what it *says*, not for how it fits.

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
from analysis.models.survival import mixture_baseline as MB
from analysis.workflows.production_risk import ya_k1k2_hybrid as YA

SLUG = "production_risk_ya_v3"
VERSION = "v3"
CLOCK, EVENT_COL = YA.CLOCK, YA.EVENT_COL
CONTRACTOR_TERMS = YA.CONTRACTOR_TERMS
CONTRACTOR_REF = "brt"
RMST_HORIZON = 730.0

#: Arm grids are inherited from v2/v2.1 unchanged, so the layers stay comparable across versions.
QNOM_KNOTS, QNOM_PIN = YA.QNOM_KNOTS, YA.QNOM_PIN
QL_KNOTS, QL_PIN = YA.QL_KNOTS, YA.QL_PIN
FREQ_DEV_KNOTS, FREQ_PIN = YA.FREQ_DEV_KNOTS, YA.FREQ_PIN
LAM_QNOM, LAM_QL, LAM_FREQ = YA.LAM_QNOM, YA.LAM_QL, YA.LAM_FREQ

#: Maximum-likelihood mixing weight, and its profile-likelihood interval.  The interval is the
#: honest object here — see the module docstring.
W_ML = 0.50
W_PROFILE_CI = (0.10, 1.00)
#: The one blend the data rules out.
W_REJECTED = 0.0

_QN = np.asarray(QNOM_KNOTS, float)
_QL = np.asarray(QL_KNOTS, float)
_FD = np.asarray(FREQ_DEV_KNOTS, float)
_QN_PIN = list(QNOM_KNOTS).index(QNOM_PIN)
_QL_PIN = list(QL_KNOTS).index(QL_PIN)
_FD_PIN = list(FREQ_DEV_KNOTS).index(FREQ_PIN)
_N_QN, _N_QL = len(_QN) - 1, len(_QL) - 1
_N_FL, _N_FR = _FD_PIN, len(_FD) - _FD_PIN - 1
_N_INC = _N_QN + _N_QL + _N_FL + _N_FR


# ---------------------------------------------------------------------------
# Shape helpers (isotonic monotone arms; tent for frequency)
# ---------------------------------------------------------------------------
def _mono(inc: np.ndarray, pin: int) -> np.ndarray:
    full = np.concatenate([[0.0], np.cumsum(np.maximum(inc, 0.0))])
    return full - full[pin]


def _tent(inc_l: np.ndarray, inc_r: np.ndarray) -> np.ndarray:
    full = np.zeros(len(_FD))
    c = 0.0
    for j, i in enumerate(range(_FD_PIN - 1, -1, -1)):
        c += max(0.0, inc_l[j]); full[i] = c
    c = 0.0
    for j, i in enumerate(range(_FD_PIN + 1, len(_FD))):
        c += max(0.0, inc_r[j]); full[i] = c
    return full


def _sk(x, knots: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Polyline in LOG θ, linear between knots, clamped flat outside."""
    return np.interp(np.clip(np.asarray(x, float), knots[0], knots[-1]), knots, full)


def _split(p: np.ndarray):
    o = 4
    a = p[o:o + _N_QN]; o += _N_QN
    b = p[o:o + _N_QL]; o += _N_QL
    fl = p[o:o + _N_FL]; o += _N_FL
    return p[0], p[1], p[2], p[3], a, b, fl, p[o:]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
@dataclass
class LayerFit:
    beta0: float
    eta0: float
    w: float
    linear: dict                     # contractor log-HRs
    theta_qnom: np.ndarray
    theta_ql: np.ndarray
    theta_freq: np.ndarray
    loglik: float
    n: int
    events: int
    n_par: int


def _nll(p: np.ndarray, w: float, t, e, qn, ql, fd, slb, oth) -> float:
    b0, le0, cs, co, a, b, fl, fr = _split(p)
    if b0 <= 0:
        return 1e18
    th_qn = np.exp(_sk(qn, _QN, _mono(a, _QN_PIN)))
    th_ql = np.exp(_sk(ql, _QL, _mono(b, _QL_PIN)))
    rate = np.maximum(w * th_qn + (1.0 - w) * th_ql, 1e-9)   # ARITHMETIC blend
    lp = cs * slb + co * oth + np.log(rate) + _sk(fd, _FD, _tent(fl, fr))
    lt = np.log(t)
    base = np.clip(b0 * (lt - le0), -700.0, 700.0)
    ll = e * (lp + np.log(b0) - le0 + (b0 - 1.0) * (lt - le0)) - np.exp(lp) * np.exp(base)
    v = -float(np.sum(ll))
    return v if np.isfinite(v) else 1e18


def fit_layers(cc: pd.DataFrame, w: float) -> LayerFit:
    """Constrained Weibull-PH MLE of the blended-rate layers at a FIXED ``w``."""
    t = cc[CLOCK].to_numpy(float); e = cc[EVENT_COL].to_numpy(float)
    qn = cc["qnom"].to_numpy(float); ql = cc["ql"].to_numpy(float)
    fd = cc["freq_dev"].to_numpy(float)
    slb = (cc["contractor_group"] == "slb").to_numpy(float)
    oth = (cc["contractor_group"] == "oth").to_numpy(float)
    args = (w, t, e, qn, ql, fd, slb, oth)

    def obj(p):
        n_ = _nll(p, *args)
        if n_ >= 1e18:
            return 1e18
        _, _, _, _, a, b, fl, fr = _split(p)
        return (n_ + LAM_QNOM * float(a @ a) + LAM_QL * float(b @ b)
                + LAM_FREQ * (float(fl @ fl) + float(fr @ fr)))

    bounds = ([(0.05, 6.0), (np.log(5.0), np.log(5e4)), (-4.0, 4.0), (-4.0, 4.0)]
              + [(0.0, 4.0)] * _N_INC)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s0 in (0.0, 0.05, 0.15):
            x0 = np.array([1.0, np.log(900.0), 0.2, 0.8] + [s0] * _N_INC)
            r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
    p = best.x
    b0, le0, cs, co, a, b, fl, fr = _split(p)
    return LayerFit(
        beta0=float(b0), eta0=float(np.exp(le0)), w=float(w),
        linear={"slb": float(cs), "oth": float(co)},
        theta_qnom=np.exp(_mono(a, _QN_PIN)),
        theta_ql=np.exp(_mono(b, _QL_PIN)),
        theta_freq=np.exp(_tent(fl, fr)),
        loglik=-_nll(p, *args), n=len(cc), events=int(e.sum()),
        n_par=4 + _N_INC + 1,          # +1 for w
    )


def profile_w(cc: pd.DataFrame, grid=None) -> pd.DataFrame:
    """Refit every layer at each ``w`` — the only honest way to report the mixing weight.

    With both arm shapes free, ``w`` trades off against them; a single optimiser run would
    return a point estimate with no indication that the surface is flat.
    """
    grid = np.round(np.arange(0.0, 1.01, 0.1), 2) if grid is None else np.asarray(grid, float)
    rows = []
    for w in grid:
        f = fit_layers(cc, float(w))
        rows.append({"w": float(w), "loglik": f.loglik,
                     "aic": -2.0 * f.loglik + 2.0 * f.n_par})
    out = pd.DataFrame(rows)
    top = out["loglik"].max()
    out["in_ci95"] = out["loglik"] >= top - 1.92      # χ²₁/2 drop
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 4000) -> float:
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


@dataclass
class YaV3Model:
    layers: LayerFit
    baseline: object                      # k1/k2 mixture, for the reported baseline
    w_ci: tuple = W_PROFILE_CI
    profile: pd.DataFrame | None = None
    version: str = VERSION
    slug: str = SLUG

    def theta_rate(self, qnom, ql) -> np.ndarray:
        """The blended rate layer.  ARITHMETIC in θ — see the module docstring."""
        th_qn = np.exp(_sk(qnom, _QN, np.log(self.layers.theta_qnom)))
        th_ql = np.exp(_sk(ql, _QL, np.log(self.layers.theta_ql)))
        return self.layers.w * th_qn + (1.0 - self.layers.w) * th_ql

    def theta_freq_at(self, freq_dev) -> np.ndarray:
        return np.exp(_sk(freq_dev, _FD, np.log(self.layers.theta_freq)))

    def implied_kpod(self, kpod, qnom: float = QNOM_PIN, kpod_ref: float = 0.8) -> np.ndarray:
        """The Kpod response v3 **implies**, without carrying a Kpod arm.

        ``Ql = Qnom · Kpod``, so at fixed nameplate a change in loading is a change in realised
        flow.  Slicing the blended rate layer along that line and re-pinning at ``kpod_ref``
        gives a Kpod curve::

            θ_Kpod(k | Qnom) = θ_rate(Qnom, Qnom·k) / θ_rate(Qnom, Qnom·k_ref)

        Two properties worth stating, because they are the whole point:

        * at ``w = 1`` (pure nameplate) the slice is **identically 1** — the model says loading
          does not matter once pump size is known, which is what every direct Kpod fit in this
          workflow also concluded;
        * it is **not separable**: the curve depends on ``Qnom``, so there is no single Kpod
          layer to extract.  A well at Qnom 100 and one at Qnom 900 get different Kpod
          responses, because the same ratio puts them at very different absolute flows.

        This is a derived diagnostic, not a fitted layer — do not ship it as one.
        """
        k = np.asarray(kpod, float)
        num = self.theta_rate(np.full_like(k, float(qnom)), float(qnom) * k)
        den = float(np.atleast_1d(self.theta_rate(qnom, float(qnom) * kpod_ref))[0])
        return np.asarray(num, float) / den

    def contractor_hr(self, contractor: str) -> float:
        if contractor == CONTRACTOR_REF:
            return 1.0
        return float(np.exp(self.layers.linear.get(contractor, 0.0)))

    def rmst_ref(self, horizon: float = RMST_HORIZON) -> float:
        return rmst(self.layers.eta0, self.layers.beta0, horizon)

    def compose(self, *, contractor: str = CONTRACTOR_REF, qnom: float = QNOM_PIN,
                ql: float = QL_PIN, freq_dev: float = 0.0,
                horizon: float = RMST_HORIZON) -> dict:
        """Multiply the θ's, then convert ONCE.  RMST multipliers are not multiplicative."""
        b0, e0 = self.layers.beta0, self.layers.eta0
        parts = {
            "contractor": self.contractor_hr(contractor),
            "rate": float(np.atleast_1d(self.theta_rate(qnom, ql))[0]),
            "freq": float(np.atleast_1d(self.theta_freq_at(freq_dev))[0]),
        }
        theta = float(np.prod(list(parts.values())))
        eta_eff = e0 * theta ** (-1.0 / b0)
        r0 = rmst(e0, b0, horizon)
        r = rmst(eta_eff, b0, horizon)
        return {"theta_total": theta, "theta_parts": parts, "beta0": b0, "eta0": e0,
                "eta_eff": eta_eff, "rmst_ref": r0, "rmst": r, "rmst_mult": r / r0,
                "median": eta_eff * np.log(2.0) ** (1.0 / b0)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class ModelRun:
    model: YaV3Model
    spec: pd.DataFrame
    baseline_table: pd.DataFrame
    profile: pd.DataFrame
    ya: pd.DataFrame
    cc: pd.DataFrame


def prepare_frame(as_of=None, cached: pd.DataFrame | None = None) -> pd.DataFrame:
    return YA.prepare_frame(as_of=as_of, cached=cached)


def run(*, as_of=None, cached: pd.DataFrame | None = None, w: float | None = None,
        write: bool = True, num_starts: int = 60, profile: bool = True) -> ModelRun:
    """Fit Ya v3.  ``w`` defaults to the profile maximum; pass a value to pin it."""
    ya = prepare_frame(as_of=as_of, cached=cached)
    cc = ya.dropna(subset=["qnom", "ql", "freq_dev"]).copy()
    prof = profile_w(cc) if profile else pd.DataFrame(columns=["w", "loglik", "aic", "in_ci95"])
    if w is None:
        w = float(prof.loc[prof["loglik"].idxmax(), "w"]) if len(prof) else W_ML
    layers = fit_layers(cc, w)
    baseline = MB.fit_baseline(ya[CLOCK].to_numpy(float), ya[EVENT_COL].to_numpy(float),
                               num_starts=num_starts)
    ci = ((float(prof.loc[prof.in_ci95, "w"].min()), float(prof.loc[prof.in_ci95, "w"].max()))
          if len(prof) and prof.in_ci95.any() else W_PROFILE_CI)
    model = YaV3Model(layers=layers, baseline=baseline, w_ci=ci, profile=prof)
    obj = ModelRun(model=model, spec=_spec_table(model),
                   baseline_table=_baseline_table(model), profile=prof, ya=ya, cc=cc)
    if write:
        _write_outputs(obj)
    return obj


def _baseline_table(m: YaV3Model) -> pd.DataFrame:
    L = m.layers
    return pd.DataFrame([{
        "version": m.version, "n": L.n, "events": L.events,
        "w": round(L.w, 3), "w_ci_lo": m.w_ci[0], "w_ci_hi": m.w_ci[1],
        "beta0": round(L.beta0, 4), "eta0": round(L.eta0, 1),
        "rmst730_ref": round(m.rmst_ref(), 1),
        "median_ref": round(L.eta0 * np.log(2) ** (1 / L.beta0)),
        "loglik": round(L.loglik, 3), "n_par": L.n_par,
        "aic": round(-2 * L.loglik + 2 * L.n_par, 1),
        "baseline_beta1": round(float(m.baseline.beta1), 4),
        "baseline_eta1": round(float(m.baseline.eta1), 1),
        "clock": CLOCK, "kpod_layer": "absent (measured null)",
        "blend": "arithmetic: w*theta_Qnom + (1-w)*theta_Ql",
    }])


def _spec_table(m: YaV3Model) -> pd.DataFrame:
    L = m.layers
    b0, e0 = L.beta0, L.eta0
    r0 = m.rmst_ref()
    rows = []
    for cg in (CONTRACTOR_REF,) + tuple(CONTRACTOR_TERMS):
        hr = m.contractor_hr(cg)
        rows.append({"component": "contractor", "term": cg, "knot": np.nan,
                     "theta": round(hr, 4),
                     "rmst730_mult": round(rmst(e0 * hr ** (-1 / b0), b0) / r0, 4)})
    for name, knots, th in (("theta_Qnom", QNOM_KNOTS, L.theta_qnom),
                            ("theta_Ql", QL_KNOTS, L.theta_ql),
                            ("theta_freq_dev", FREQ_DEV_KNOTS, L.theta_freq)):
        for k, v in zip(knots, th):
            rows.append({"component": name, "term": name.replace("theta_", ""), "knot": k,
                         "theta": round(float(v), 4),
                         "rmst730_mult": round(rmst(e0 * float(v) ** (-1 / b0), b0) / r0, 4)})
    return pd.DataFrame(rows)


def _write_outputs(obj: ModelRun) -> Path:
    out = results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    obj.baseline_table.to_csv(tables / "baseline.csv", index=False, encoding="utf-8-sig")
    obj.spec.to_csv(tables / "model_spec.csv", index=False, encoding="utf-8-sig")
    obj.profile.to_csv(tables / "w_profile.csv", index=False, encoding="utf-8-sig")
    _fig_profile(obj, figures)
    return out


def _fig_profile(obj: ModelRun, figures: Path) -> None:
    """The w profile — drawn because a point estimate would hide how flat it is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = obj.profile
    if not len(p):
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    top = p["loglik"].max()
    ax.plot(p["w"], p["loglik"], "o-", color="#2471a3", lw=2.4, ms=7)
    ax.axhline(top - 1.92, ls="--", color="#c0392b", lw=1.3,
               label="порог 95% (Δloglik = 1.92)")
    inci = p[p.in_ci95]
    ax.axvspan(inci["w"].min(), inci["w"].max(), color="#2471a3", alpha=.10, lw=0,
               label=f"95% интервал w = [{inci['w'].min():.2f}, {inci['w'].max():.2f}]")
    ax.plot([obj.model.layers.w], [top], "*", color="#c0392b", ms=18, zorder=6,
            label=f"ML w = {obj.model.layers.w:.2f}")
    ax.set_xlabel("w  (доля θ_Qном в смеси)")
    ax.set_ylabel("log-правдоподобие")
    ax.set_title("Ya v3: профиль правдоподобия по w — вес смеси НЕ идентифицирован;\n"
                 "данные отвергают только w = 0 (чистый Ql)", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "w_profile.png", dpi=140)
    plt.close(fig)
