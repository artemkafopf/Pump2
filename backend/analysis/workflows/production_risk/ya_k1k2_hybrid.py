"""Ya **k1/k2 hybrid θ-model** — mixture-Weibull baseline × shape-constrained hazard layers.

The Ya counterpart of the Vt v3.1/v3.2 line, built to the same contract but on Ya's own
support.  Form::

    S(t | x) = S_k1k2( t ; apply_theta(params, Θ) ),
    Θ = HR_contractor · θ_rate(rate) · θ_Kpod(Kpod) · θ_freq(f − f_nom)

**Two versions, one code path** — ``run(rate=...)`` selects the rate arm; everything else
(baseline, Kpod tent, freq tent, contractor, shape constraints, bootstrap) is identical:

* **v2** (``rate="Ql"``, the original) — ``θ_rate = θ_Ql``, MONOTONE, θ=1 at Ql=250.  Ya's
  crude rate climbs 0.70→2.10 per 1000 op-days across Ql *within every* Kpod and freq band.
* **v2.1** (``rate="Qnom"``) — ``θ_rate = θ_Qnom``, MONOTONE, θ=1 at Qnom=200 m³/d.  Uses the
  pump's **nameplate BEP capacity** (``nominal_flow_m3d``) instead of measured flow, on the
  finding that Qnom *subsumes* Ql: with both in a Cox, Qnom survives within Ql-strata
  (HR 1.28, p=0.003) while Ql dies within Qnom-strata (p=0.79); Qnom's crude rate is a clean
  monotone dose-response with Kpod flat across its bands, and it edges Ql on CV concordance.
  Kpod (= Ql/Qnom) then carries the pure ratio and goes fully flat.  Caveats: Ql/Qnom are
  0.87-correlated (part of "Ql dies" is the cleaner catalog value beating a noisy windowed
  measurement) and it is observational (bigger pumps are *specified* for demanding wells).

Common to both:

* **Baseline — k1/k2, AIC-selected** (:mod:`analysis.models.survival.mixture_baseline`).
  The deployable curve is the ``esp_models.csv`` five-number form
  ``(w1, β1, η1, β2, η2)``; ``w1=0`` is the degenerate "single Weibull" convention.  θ is
  applied as the repo deploys it — a per-component scale shift ``η ← η·Θ^(−1/β)``
  (``survival.HazardLayer.apply_theta``).  The baseline is rate-arm-independent (same RMST).
* **θ_Kpod** — TENT pinned at **Kpod 0.8**: θ=1 there and monotone non-decreasing on both
  arms ⇔ life maximal at 0.8 (the user's shape requirement, identical to Vt v3.2).
* **θ_freq** — TENT pinned at the **nominal frequency** (f − f_nom = 0), same shape rule.
* **contractor** — free log-HR terms, ``brt`` reference (Ya crude rates: brt 1.33, slb 1.63,
  oth 4.47 per 1000 op-days).

**No stratification and no pooling.**  Unlike Vt, Ya is a single H₂S class — every one of its
2145 runs carries ``h2s_class == "nonsour"`` — so there is no sour/nonsour split to fit, and
therefore nothing to pool across.  Contractor is carried as a proportional-hazard term rather
than as a stratum (it is a rate effect, checked in ``contractor_check.csv``).

**Shape constraints are structural, not penalties**: each arm is built from non-negative
increments (:mod:`analysis.models.survival.polyline_ph`), so θ cannot violate its shape at any
point of the optimisation.  An arm whose data has no signal in the constrained direction
therefore lands *on the boundary* (θ≡1, flat) rather than bending the wrong way — read a flat
arm as "the constraint is binding here", not as "the effect was estimated to be zero".

**Applicability.**  θ is clamped flat outside the knot range.  Ya's knots were set from Ya's
own support, which differs from Vt's: Ya has real mass at low Ql (254 runs / 111 failures below
47 m³/d, so no low-Ql guard is needed) and at low Kpod (256 runs / 128 failures below 0.4), and
— unlike Vt, where the fleet never runs above 63 Hz — Ya has 34 runs / 21 failures above
+10 Hz, which is why the frequency grid reaches +15 Hz (65 Hz).  Above 65 Hz (9 runs) and above
Kpod 1.5 (27 runs) θ is clamped, which *under*-penalises those extremes.

**Composition.**  θ's multiply; RMST multipliers DO NOT (RMST saturates).  Always go through
:meth:`YaModel.compose`.

Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline, MRL(0) alongside,
median reference only, KM control always plotted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival import mixture_baseline as MB
from analysis.models.survival.polyline_ph import (
    MONO, TENT, ArmSpec, bootstrap_theta, fit_polyline_ph,
)
from analysis.paths import results_dir
from analysis.workflows.production_risk import vt_ttf_covariates as V
from analysis.workflows.production_risk.survival import HazardLayer

# ---------------------------------------------------------------------------
# Policy / constants
# ---------------------------------------------------------------------------
SLUG = "production_risk_ya_k1k2_hybrid"
FIELD = "Ya"
CLOCK = V.CLOCK_PRIMARY            # "t_cal"
EVENT_COL = V.EVENT_COL            # "event"
CLUSTER = "well_key"               # bootstrap cluster: runs of one well are dependent
CONTRACTOR_REF = "brt"
CONTRACTOR_TERMS = ("slb", "oth")
RMST_HORIZON = MB.RMST_HORIZON     # 730 d

#: Nominal-frequency policy: the per-run value when plausible, else 50 Hz.  Ya's
#: ``nominal_freq_hz`` column carries obvious unit errors (220/240 on 30 runs) — those fall
#: back to 50, which is the value on 2089 of 2145 runs anyway.
FNOM_DEFAULT = 50.0
FNOM_PLAUSIBLE = (40.0, 65.0)

#: Ql — monotone arm, θ=1 at 250 m³/d.  Knots span Ya's q02–q97; **no guard**: unlike Vt sour
#: (where the low-Ql life bonus rested on a handful of runs and was held flat), Ya's low band
#: is well populated (254 runs / 111 failures below 47) and its crude rate is monotone there.
QL_KNOTS = (20.0, 47.0, 100.0, 250.0, 450.0, 900.0)
QL_PIN = 250.0
QL_GUARD = None

#: Kpod — tent pinned at 0.8.  Extends down to 0.2 (Ya has 256 runs / 128 failures below 0.4,
#: far more under-loading than Vt) and up to 1.5; above that only 27 runs, so clamp.
KPOD_KNOTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
KPOD_PIN = 0.8

#: freq deviation (f − f_nom) — tent pinned at nominal.  Reaches ±15 Hz: Ya genuinely runs
#: over-speed (34 runs / 21 failures above +10 Hz, the highest crude rate in every Ql band),
#: which is NOT true of Vt (zero runs above 63 Hz — its grid stops at +10 and must stay there).
FREQ_DEV_KNOTS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
FREQ_PIN = 0.0

#: Ridge on the polyline increments — small on Ql (the primary arm, free to move), larger on
#: Kpod/freq so they stay MINOR layers.  Same values as Vt v3.2 for comparability.
LAM_QL, LAM_KPOD, LAM_FREQ = 1.0, 6.0, 6.0

#: Qnom (nameplate `nominal_flow_m3d`, the pump's BEP design delivery) — monotone arm, θ=1 at
#: 200 m³/d (the fleet-median pump size).  v2.1 uses this as the rate arm INSTEAD of Ql, on the
#: finding that nameplate Qnom subsumes measured Ql: Qnom survives within Ql-strata (HR1.28
#: p=0.003) while Ql dies within Qnom-strata (p=0.79); crude rate is a clean monotone
#: dose-response (0.62→1.73/1000d) with Kpod flat across Qnom bands.  Knots span Ya's
#: nameplate q05–q95 (37–1069); a genuine ordinal dose (76 catalog sizes), not a jumpy label.
QNOM_KNOTS = (50.0, 100.0, 200.0, 320.0, 500.0, 900.0)
QNOM_PIN = 200.0
LAM_QNOM = 1.0

QL_ARM = ArmSpec(name="Ql", column="ql", knots=QL_KNOTS, pin=QL_PIN,
                 shape=MONO, ridge=LAM_QL, guard=QL_GUARD)
QNOM_ARM = ArmSpec(name="Qnom", column="qnom", knots=QNOM_KNOTS, pin=QNOM_PIN,
                   shape=MONO, ridge=LAM_QNOM, guard=None)
KPOD_ARM = ArmSpec(name="Kpod", column="kpod_run", knots=KPOD_KNOTS, pin=KPOD_PIN,
                   shape=TENT, ridge=LAM_KPOD)
FREQ_ARM = ArmSpec(name="freq_dev", column="freq_dev", knots=FREQ_DEV_KNOTS, pin=FREQ_PIN,
                   shape=TENT, ridge=LAM_FREQ)

#: The rate arm is selectable: "Ql" → v2 (the original), "Qnom" → v2.1.  Everything else
#: (baseline, Kpod tent, freq tent, contractor, shapes, bootstrap) is identical between them.
RATE_ARMS = {"Ql": QL_ARM, "Qnom": QNOM_ARM}
VERSION_BY_RATE = {"Ql": "v2", "Qnom": "v2.1"}
SLUG_BY_RATE = {"Ql": "production_risk_ya_k1k2_hybrid", "Qnom": "production_risk_ya_v21_qnom"}

ARMS = (QL_ARM, KPOD_ARM, FREQ_ARM)            # module default = v2 (Ql); v2.1 built in run()
_ARM_BY_NAME = {a.name: a for a in ARMS}

#: Supported covariate window.  Outside it θ is clamped flat — an assumption, not a fit.
#: The rate-arm entry ("ql" / "qnom") is swapped per version; kpod/freq are common.
_COMMON_APPLICABILITY = {
    "kpod": (0.2, 1.5),           # clamped above 1.5 ⇒ under-penalises extreme overload
    "freq_hz": (35.0, 65.0),      # 26 runs below 35 Hz, 9 above 65 Hz
}
_RATE_APPLICABILITY = {
    "Ql": ("ql", (20.0, 900.0)),      # 73 runs below 20, 70 above 900: clamped
    "Qnom": ("qnom", (50.0, 900.0)),  # nameplate q05≈37 / q95≈1069: clamp outside
}
#: arm name → applicability-dict key (and figure/table label routing).
_APPLIC_KEY = {"Ql": "ql", "Qnom": "qnom", "Kpod": "kpod", "freq_dev": "freq_hz"}


def _applicability_for(rate: str) -> dict:
    key, win = _RATE_APPLICABILITY[rate]
    return {key: win, **_COMMON_APPLICABILITY}


def _arms_for(rate: str) -> tuple:
    """The arm tuple for a rate choice: (rate arm, Kpod tent, freq tent)."""
    if rate not in RATE_ARMS:
        raise ValueError(f"rate must be one of {tuple(RATE_ARMS)}, got {rate!r}")
    return (RATE_ARMS[rate], KPOD_ARM, FREQ_ARM)


APPLICABILITY = _applicability_for("Ql")       # module default = v2


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def prepare_frame(as_of: pd.Timestamp | None = None,
                  cached: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ya modelling frame: clock > 0, nominal-frequency deviation, contractor dummies.

    NaN covariates are left in place — the baseline uses **all** runs, the θ-layers only the
    complete cases; neither silently drops rows the other keeps."""
    df = cached if cached is not None else V.build_frame(as_of=as_of)[0]
    ya = df[df["field"] == FIELD].copy()
    ya[CLOCK] = pd.to_numeric(ya[CLOCK], errors="coerce")
    ya = ya[ya[CLOCK] > 0].dropna(subset=[CLOCK, EVENT_COL]).copy()
    fn = pd.to_numeric(ya.get("nominal_freq_hz"), errors="coerce")
    ya["f_nom"] = np.where(fn.between(*FNOM_PLAUSIBLE), fn, FNOM_DEFAULT)
    ya["freq_dev"] = pd.to_numeric(ya["freq_run"], errors="coerce") - ya["f_nom"]
    #: Qnom = nameplate BEP delivery (the pump's design capacity) — the v2.1 rate arm.
    ya["qnom"] = pd.to_numeric(ya.get("nominal_flow_m3d"), errors="coerce")
    for cg in CONTRACTOR_TERMS:
        ya[cg] = (ya["contractor_group"] == cg).astype(float)
    if CLUSTER not in ya.columns:
        ya[CLUSTER] = ya["code"].astype(str).str.casefold()
    return ya


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
#: compose() keyword aliases → arm name (so callers may pass ``ql=``/``qnom=``/``kpod=``).
_COMPOSE_ALIAS = {"ql": "Ql", "qnom": "Qnom", "kpod": "Kpod", "freq_dev": "freq_dev"}


@dataclass
class YaModel:
    """Fitted Ya model: k1/k2 baseline + constrained θ-layers, with the composition rule.

    The rate arm is whatever ``run(rate=...)`` chose — ``rate_name`` is ``"Ql"`` (v2) or
    ``"Qnom"`` (v2.1); ``arms`` carries the actual :class:`ArmSpec` tuple, so every table,
    figure and the composition rule are agnostic to which rate covariate is in play."""
    baseline: MB.BaselineFit
    layers: object                     # polyline_ph.PolylinePHFit
    boot: dict
    rate_name: str = "Ql"
    arms: tuple = ARMS
    version: str = "v2"
    slug: str = SLUG
    applicability: dict = dc_field(default_factory=lambda: dict(APPLICABILITY))

    @property
    def rate_arm(self):
        return next(a for a in self.arms if a.name == self.rate_name)

    # -- θ evaluation --------------------------------------------------------
    def theta_at(self, arm: str, x) -> np.ndarray:
        """θ(x) for one arm, clamped outside the knot range (see ``APPLICABILITY``)."""
        return self.layers.theta_at(arm, x)

    def theta_freq_hz(self, freq_hz, f_nom: float = FNOM_DEFAULT) -> np.ndarray:
        """θ at an absolute running frequency, given the pump's nominal frequency."""
        return self.theta_at("freq_dev", np.asarray(freq_hz, float) - f_nom)

    def contractor_hr(self, contractor: str) -> float:
        if contractor == CONTRACTOR_REF:
            return 1.0
        return self.layers.hr(contractor)

    # -- life of the deployed baseline --------------------------------------
    def life_ref(self, horizon: float = RMST_HORIZON) -> dict:
        return MB.life_from_S(self.baseline.S, horizon)

    # -- the ONE correct way to combine effects ------------------------------
    def compose(self, *, contractor: str = CONTRACTOR_REF,
                horizon: float = RMST_HORIZON, **covariates) -> dict:
        """Combine covariates properly: multiply θ's, then convert to life ONCE.

        Pass covariate values by arm name or its alias (``ql=``/``qnom=`` for the rate arm,
        ``kpod=``, ``freq_dev=``) — an unknown key that is not one of this model's arms is an
        error, so you cannot accidentally compose Ql into a v2.1 (Qnom) model.

        RMST multipliers are **not** multiplicative (RMST is a saturating functional of the
        hazard) — chaining them overstates life.  Always compose here."""
        arm_names = {a.name for a in self.arms}
        vals = {}
        for k, v in covariates.items():
            name = _COMPOSE_ALIAS.get(k, k)
            if name not in arm_names:
                raise ValueError(f"{name!r} is not an arm of this {self.version} model "
                                 f"(arms: {sorted(arm_names)})")
            vals[name] = v
        theta = self.contractor_hr(contractor)
        parts = {"contractor": theta}
        for a in self.arms:
            val = vals.get(a.name)
            if val is None:
                continue
            th = float(self.theta_at(a.name, np.array([float(val)]))[0])
            parts[a.name] = th
            theta *= th
        p0 = self.baseline.params
        p = HazardLayer.apply_theta(p0, theta)
        life = MB.life_from_S(lambda t: MB.mixture_S(t, **p), horizon)
        ref = MB.life_from_S(lambda t: MB.mixture_S(t, **p0), horizon)
        return {"theta_total": theta, "theta_parts": parts, "params": p,
                "rmst": life["rmst"], "mrl": life["mrl"], "median": life["median"],
                "rmst_ref": ref["rmst"], "mrl_ref": ref["mrl"],
                "rmst_mult": round(life["rmst"] / ref["rmst"], 4) if ref["rmst"] else np.nan,
                "mrl_mult": round(life["mrl"] / ref["mrl"], 4) if ref["mrl"] else np.nan}

    def life_mult(self, theta: float, horizon: float = RMST_HORIZON) -> float:
        """RMST(0,horizon) multiplier of a single total θ (for tables/figures)."""
        p = HazardLayer.apply_theta(self.baseline.params, float(theta))
        r = MB.life_from_S(lambda t: MB.mixture_S(t, **p), horizon)["rmst"]
        r0 = MB.life_from_S(self.baseline.S, horizon)["rmst"]
        return r / r0 if r0 else float("nan")


# ---------------------------------------------------------------------------
# Orchestration + outputs
# ---------------------------------------------------------------------------
@dataclass
class ModelRun:
    model: YaModel
    baseline_table: pd.DataFrame
    spec: pd.DataFrame
    knot_thetas: pd.DataFrame
    ttf_summary: pd.DataFrame
    coverage: pd.DataFrame
    contractor_check: pd.DataFrame
    ya: pd.DataFrame
    cc: pd.DataFrame


def run(*, rate: str = "Ql", as_of: pd.Timestamp | None = None,
        cached: pd.DataFrame | None = None, n_boot: int = 200, num_starts: int = 60,
        write: bool = True) -> ModelRun:
    """Fit the Ya k1/k2 hybrid end-to-end and (optionally) write tables + figures.

    ``rate`` selects the rate arm: ``"Ql"`` → **v2** (the original), ``"Qnom"`` → **v2.1**
    (nameplate pump capacity instead of measured flow — see ``QNOM_ARM``).  Everything else
    is identical; the two write to different result slugs so neither overwrites the other."""
    arms = _arms_for(rate)
    ya = prepare_frame(as_of=as_of, cached=cached)
    cc = ya.dropna(subset=[a.column for a in arms]).copy()

    baseline = MB.fit_baseline(ya[CLOCK].to_numpy(float), ya[EVENT_COL].to_numpy(float),
                               num_starts=num_starts)
    layers = fit_polyline_ph(cc, clock=CLOCK, event_col=EVENT_COL, arms=arms,
                             linear_terms=CONTRACTOR_TERMS)
    boot = (bootstrap_theta(cc, clock=CLOCK, event_col=EVENT_COL, arms=arms,
                            linear_terms=CONTRACTOR_TERMS, cluster=CLUSTER, n_boot=n_boot)
            if n_boot > 0 else {})
    model = YaModel(baseline=baseline, layers=layers, boot=boot, rate_name=rate, arms=arms,
                    version=VERSION_BY_RATE[rate], slug=SLUG_BY_RATE[rate],
                    applicability=_applicability_for(rate))

    run_obj = ModelRun(
        model=model,
        baseline_table=_baseline_table(model, ya),
        spec=_spec_table(model),
        knot_thetas=_knot_theta_table(model),
        ttf_summary=_ttf_summary(model),
        coverage=_coverage(ya, cc),
        contractor_check=_contractor_check(ya),
        ya=ya, cc=cc,
    )
    if write:
        _write_outputs(run_obj)
    return run_obj


def _km_rmst(ya: pd.DataFrame, horizon: float = RMST_HORIZON) -> float:
    """KM RMST(0,horizon) — the mandatory non-parametric control on the baseline."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KaplanMeierFitter().fit(ya[CLOCK], ya[EVENT_COL])
        return round(float(restricted_mean_survival_time(km, t=horizon)), 1)


def _baseline_table(m: YaModel, ya: pd.DataFrame) -> pd.DataFrame:
    b, life = m.baseline, m.life_ref()
    return pd.DataFrame([{
        "field": FIELD, "scope": "Ya_pooled(nonsour only)", "model_kind_selected": b.model_kind,
        "w1": round(b.w1, 4), "beta1": round(b.beta1, 4), "eta1": round(b.eta1, 1),
        "beta2": round(b.beta2, 4), "eta2": round(b.eta2, 1),
        "n_runs": b.n, "n_events": b.events,
        "loglik_k1": b.loglik_k1, "aic_k1": b.aic_k1,
        "loglik_k2": b.loglik_k2, "aic_k2": b.aic_k2, "delta_aic_k2_minus_k1": b.delta_aic,
        "k2_short_mode_life_d": b.k2_short_life, "k2_long_mode_life_d": b.k2_long_life,
        "rmst730_ref": life["rmst"], "mrl0_ref": life["mrl"], "median_ref": life["median"],
        "km_rmst730_control": _km_rmst(ya),
        "ph_nuisance_beta0": round(m.layers.beta0, 4), "ph_nuisance_eta0": round(m.layers.eta0, 1),
        "concordance": m.layers.concordance, "clock": CLOCK, "model_version": m.version,
        "rate_arm": m.rate_name,
        "reference_point": f"{CONTRACTOR_REF}, {m.rate_name}={m.rate_arm.pin:g}, "
                           f"Kpod={KPOD_PIN:g}, f=f_nom",
    }])


def _spec_table(m: YaModel) -> pd.DataFrame:
    """The deliverable: every θ the model applies, with its RMST(0,730) life multiplier."""
    rows = []
    for cg in CONTRACTOR_TERMS:
        hr = m.contractor_hr(cg)
        lo, hi = m.boot.get(f"hr_{cg}", (np.nan, np.nan))
        rows.append({"component": "contractor", "term": f"contractor:{cg}", "shape": "free",
                     "knot": np.nan, "theta": round(hr, 4),
                     "theta_ci_lo": lo, "theta_ci_hi": hi,
                     "rmst730_mult": round(m.life_mult(hr), 4)})
    for a in m.arms:
        for kn, th in zip(a.knots, m.layers.theta_knots[a.name]):
            lo, hi = m.boot.get(f"{a.name}@{kn:g}", (np.nan, np.nan))
            rows.append({"component": f"theta_{a.name}", "term": a.name, "shape": a.shape,
                         "knot": kn, "theta": round(float(th), 4),
                         "theta_ci_lo": lo, "theta_ci_hi": hi,
                         "rmst730_mult": round(m.life_mult(float(th)), 4)})
    return pd.DataFrame(rows)


_ARM_UNIT = {"Ql": "m3/d", "Qnom": "m3/d (nameplate BEP)", "Kpod": "Ql/Qnom",
             "freq_dev": "Hz (at f_nom=50)"}


def _knot_theta_table(m: YaModel) -> pd.DataFrame:
    """Knot θ in interpretable units (frequency knots as absolute Hz at f_nom = 50)."""
    rows = []
    for a in m.arms:
        for kn, th in zip(a.knots, m.layers.theta_knots[a.name]):
            lo, hi = m.boot.get(f"{a.name}@{kn:g}", (np.nan, np.nan))
            rows.append({
                "arm": a.name, "shape": a.shape, "ridge": a.ridge,
                "knot": kn,
                "knot_natural": FNOM_DEFAULT + kn if a.name == "freq_dev" else kn,
                "unit": _ARM_UNIT[a.name],
                "theta_hazard": round(float(th), 4),
                "theta_boot_lo": lo, "theta_boot_hi": hi,
                "rmst730_mult": round(m.life_mult(float(th)), 4),
                "on_constraint_boundary": bool(abs(float(th) - 1.0) < 1e-6),
            })
    return pd.DataFrame(rows)


_KPOD_GRID = (0.4, 0.8, 1.2)
_FREQ_GRID = (45.0, 50.0, 55.0, 60.0, 65.0)   # 65 Hz carries the one real frequency effect
#: rate-arm grid for the deployment view (natural units of the rate covariate).
_RATE_GRID = {"Ql": (47.0, 100.0, 250.0, 450.0, 900.0),
              "Qnom": (100.0, 200.0, 320.0, 500.0, 900.0)}


def _ttf_summary(m: YaModel) -> pd.DataFrame:
    """Composed life over a small operating grid (contractor = brt), the deployment view.

    The rate column is named after the model's rate arm (``ql`` for v2, ``qnom`` for v2.1)."""
    rate = m.rate_name
    rcol = rate.lower()
    rows = []
    ref = m.compose()
    rows.append({rcol: m.rate_arm.pin, "kpod": KPOD_PIN, "freq_hz": FNOM_DEFAULT,
                 "contractor": CONTRACTOR_REF, "theta": 1.0, "rmst730": ref["rmst_ref"],
                 "mrl0": ref["mrl_ref"], "median": ref["median"], "point": "reference"})
    for rv in _RATE_GRID[rate]:
        for kp in _KPOD_GRID:
            for f in _FREQ_GRID:
                r = m.compose(**{rate: rv, "kpod": kp, "freq_dev": f - FNOM_DEFAULT})
                rows.append({rcol: rv, "kpod": kp, "freq_hz": f, "contractor": CONTRACTOR_REF,
                             "theta": round(r["theta_total"], 4), "rmst730": r["rmst"],
                             "mrl0": r["mrl"], "median": r["median"], "point": "grid"})
    return pd.DataFrame(rows)


def _coverage(ya: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "scope": "Ya_pooled", "n_runs": len(ya), "n_events": int(ya[EVENT_COL].sum()),
        "n_censored": int((ya[EVENT_COL] == 0).sum()), "n_open": int(ya["end"].isna().sum()),
        "h2s_classes": str(ya["h2s_class"].value_counts().to_dict()),
        "cov_ql": round(ya["ql"].notna().mean(), 3),
        "cov_qnom": round(ya["qnom"].notna().mean(), 3),
        "cov_kpod": round(ya["kpod_run"].notna().mean(), 3),
        "cov_freq": round(ya["freq_run"].notna().mean(), 3),
        "n_complete_case": len(cc), "events_complete_case": int(cc[EVENT_COL].sum()),
        "contractor_mix": str(ya["contractor_group"].value_counts().to_dict()),
        "contractor_mix_cc": str(cc["contractor_group"].value_counts().to_dict()),
    }])


def _contractor_check(ya: pd.DataFrame) -> pd.DataFrame:
    """Is contractor a RATE effect (θ term) or a separate baseline SHAPE (stratum)?

    Per-contractor marginal Weibull: if the β's agree and only η moves, a proportional-hazard
    term is the right carrier and stratifying would only cost events."""
    from lifelines import WeibullFitter
    rows = []
    for cg, g in ya.groupby("contractor_group"):
        if int(g[EVENT_COL].sum()) < 20:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter().fit(g[CLOCK], g[EVENT_COL])
        rows.append({"contractor_group": cg, "n": len(g), "events": int(g[EVENT_COL].sum()),
                     "beta_marginal": round(float(wf.rho_), 3),
                     "eta_marginal": round(float(wf.lambda_), 1),
                     "km_rmst730": _km_rmst(g),
                     "rate_per_1000_opd": round(1000 * g[EVENT_COL].sum() / g[CLOCK].sum(), 3)})
    return pd.DataFrame(rows)


def _write_outputs(run_obj: ModelRun) -> Path:
    out = results_dir(run_obj.model.slug)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, df in (("baseline", run_obj.baseline_table), ("model_spec", run_obj.spec),
                     ("knot_thetas", run_obj.knot_thetas), ("ttf_summary", run_obj.ttf_summary),
                     ("coverage", run_obj.coverage),
                     ("contractor_check", run_obj.contractor_check)):
        df.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"covariate": k, "lo": v[0], "hi": v[1],
                   "note": "theta clamped flat outside this window (assumption, not a fit)"}
                  for k, v in run_obj.model.applicability.items()]).to_csv(
        tables / "applicability.csv", index=False, encoding="utf-8-sig")
    _fig_baseline(run_obj, figures)
    _fig_multipliers(run_obj.model, figures)
    _fig_hazard_polylines(run_obj, figures)
    return out


# ---------------------------------------------------------------------------
# Figures (Russian labels — these go into the deck)
# ---------------------------------------------------------------------------
def _fig_baseline(run_obj: ModelRun, figures: Path) -> None:
    """KM control + the k1 and k2 baselines — never ship a parametric fit without the KM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter, WeibullFitter

    b, ya = run_obj.model.baseline, run_obj.ya
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KaplanMeierFitter().fit(
            ya[CLOCK], ya[EVENT_COL],
            label=f"Каплан–Мейер, весь Ya (n={len(ya)}, отказов={int(ya[EVENT_COL].sum())})")
        km.plot_survival_function(ax=ax, ci_show=True, color="k")
        wf = WeibullFitter().fit(ya[CLOCK], ya[EVENT_COL])
    t = np.linspace(1, RMST_HORIZON, 500)
    ax.plot(t, np.exp(-((t / wf.lambda_) ** wf.rho_)), "C0", lw=2,
            label=f"k1: β={wf.rho_:.2f}, η={wf.lambda_:.0f} (AIC {b.aic_k1:.0f})")
    # S_k2, not S: on a k1 selection ``S`` is the degenerate single Weibull, so plotting it
    # here would redraw the k1 curve under a "k2" label.
    ax.plot(t, b.S_k2(t), "C3", lw=2, ls="--" if b.model_kind == "k2" else ":",
            label=(f"k2 смесь (AIC {b.aic_k2:.0f}); моды ≈{b.k2_short_life:.0f}/"
                   f"{b.k2_long_life:.0f} сут"))
    ax.set_xlabel("Наработка, сут (t_cal)"); ax.set_ylabel("Доля работающих S(t)")
    ax.set_xlim(0, RMST_HORIZON); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    ax.set_title(f"Ya базовая модель — выбрана {b.model_kind}\n"
                 f"RMST(0,730) = {run_obj.model.life_ref()['rmst']:.0f} сут "
                 f"(контроль КМ {_km_rmst(ya):.0f} сут)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "baseline_km_fit.png", dpi=140); plt.close(fig)


def _crude_rate(ya: pd.DataFrame, col: str, edges) -> list[dict]:
    """Per-bin crude failure rate (events per 1000 op-days) — the honest, model-free picture
    the polyline is drawn against.  Returns [{x_lo, x_hi, x_mid, n, ev, rate}]."""
    d = ya[ya[col].notna() & ya[CLOCK].notna()]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = d[(d[col] >= lo) & (d[col] < hi)]
        exp = float(s[CLOCK].sum())
        out.append({"x_lo": lo, "x_hi": hi, "x_mid": (lo + hi) / 2, "n": len(s),
                    "ev": int(s[EVENT_COL].sum()),
                    "rate": 1000.0 * s[EVENT_COL].sum() / exp if exp else np.nan})
    return out


#: Crude-rate bin edges per arm (for the model-free overlay on the polyline figure).
_CRUDE_EDGES = {
    "Ql": [20, 47, 100, 175, 250, 350, 450, 650, 900],
    "Qnom": [50, 80, 125, 200, 320, 500, 900],
    "Kpod": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    "freq_dev": [-15, -10, -5, -2, 2, 5, 10, 15],
}

#: Per-arm figure styling (works for both v2 and v2.1 rate arms).
_ARM_COLOR = {"Ql": "#2471a3", "Qnom": "#1e8449", "Kpod": "#7d3c98", "freq_dev": "#c0392b"}
_ARM_XLABEL = {"Ql": "Ql, м3/сут", "Qnom": "Qном (номинал насоса), м3/сут",
               "Kpod": "Kpod = Ql/Qном", "freq_dev": "рабочая частота, Гц"}
_ARM_XLABEL_DEV = dict(_ARM_XLABEL, freq_dev="частота − номинал, Гц")


def _shape_ru(a) -> str:
    if a.shape == MONO:
        word = "θ растёт с дебитом" if a.name == "Ql" else \
               ("θ растёт с размером насоса" if a.name == "Qnom" else "монотонная")
        return f"МОНОТОННАЯ ({word})"
    return f"ТЕНТ, минимум θ=1 в {'Kpod 0.8' if a.name == 'Kpod' else 'номинале'}"


def _fig_hazard_polylines(run_obj: ModelRun, figures: Path) -> None:
    """The deliverable: the three Ya hazard polylines (Ql, Kpod, frequency) on one row.

    Each panel shows the fitted θ hazard polyline (piecewise-linear in log-θ — the model's own
    form, deliberately NOT splined), the knot markers, the well-cluster bootstrap 95% band, the
    θ=1 reference and the pin, plus the model-free crude failure rate (right axis, rescaled to
    θ=1 at the pin) so the reader sees what the constrained fit is tracking.  The frequency
    panel is drawn in **absolute Hz** (at nominal 50) because that is what an operator sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, ya = run_obj.model, run_obj.ya
    labels, colors = _ARM_XLABEL, _ARM_COLOR

    def _to_axis(arm, v):
        return FNOM_DEFAULT + np.asarray(v, float) if arm == "freq_dev" else np.asarray(v, float)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
    for ax, a in zip(axes, m.arms):
        c = colors[a.name]
        kn = np.asarray(a.knots, float)
        knx = _to_axis(a.name, kn)
        xs = np.linspace(kn[0], kn[-1], 500)
        th = m.theta_at(a.name, xs)
        thk = m.theta_at(a.name, kn)

        # bootstrap band: interpolate the per-knot 2.5/97.5 θ (linear in θ across the grid)
        lo_k = np.array([m.boot.get(f"{a.name}@{k:g}", (np.nan, np.nan))[0] for k in kn])
        hi_k = np.array([m.boot.get(f"{a.name}@{k:g}", (np.nan, np.nan))[1] for k in kn])
        if np.isfinite(lo_k).all():
            lo = np.interp(xs, kn, lo_k)
            hi = np.interp(xs, kn, hi_k)
            ax.fill_between(_to_axis(a.name, xs), lo, hi, color=c, alpha=.13,
                            label="бутстрап 95% (по скважинам)")

        ax.plot(_to_axis(a.name, xs), th, color=c, lw=2.6, label="θ хазард (полилиния)")
        ax.plot(knx, thk, "o", color=c, ms=6.5, zorder=6)
        ax.axhline(1.0, color="gray", lw=.9)
        ax.axvline(_to_axis(a.name, a.pin), ls=":", color="k", alpha=.5,
                   label=f"референс (θ=1) в {'50 Гц' if a.name == 'freq_dev' else f'{a.pin:g}'}")

        # crude model-free rate, right axis, rescaled to 1 at the pin band
        crude = _crude_rate(ya, a.column, _CRUDE_EDGES[a.name])
        pin_bin = min(crude, key=lambda r: abs(r["x_mid"] - a.pin))
        base = pin_bin["rate"] if np.isfinite(pin_bin["rate"]) and pin_bin["rate"] > 0 else np.nan
        ax2 = ax.twinx()
        if np.isfinite(base):
            xm = _to_axis(a.name, [r["x_mid"] for r in crude])
            ratio = [r["rate"] / base for r in crude]
            ax2.plot(xm, ratio, "s--", color="#555555", ms=4.5, lw=1.1, alpha=.75,
                     label="сырой темп отказов /1000 сут (норм. к референсу)")
            ax2.set_ylim(*ax.get_ylim())
            ax2.set_yticks([])
        # constrained flat arm: name it, don't let a 1% wobble read as a finding
        if float(np.max(thk)) < 1.05:
            ax.text(0.5, 0.5, f"θ на границе ограничения:\nэффекта {a.name} в данных нет",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color="#555555", style="italic")

        lo_a, hi_a = m.applicability[_APPLIC_KEY[a.name]]
        ax.set_title(f"{a.name} — {_shape_ru(a)}\nприменимость {lo_a:g}–{hi_a:g} "
                     "(вне — θ фиксирована)", fontsize=10.5)
        ax.set_xlabel(labels[a.name]); ax.grid(alpha=.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper left")
    axes[0].set_ylabel("θ — множитель риска (хазард)")
    fig.suptitle(
        f"Ya {m.version}: полилинии риска θ (хазард) — рейт-слой {m.rate_name}, база смесь "
        f"{m.baseline.model_kind}, RMST(0,730) {m.life_ref()['rmst']:.0f} сут; "
        f"θ перемножаются, ресурс считается один раз (compose)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(figures / "hazard_polylines.png", dpi=145)
    plt.close(fig)


def _fig_multipliers(m: YaModel, figures: Path) -> None:
    """θ and RMST(0,730) multiplier charts — piecewise-linear in log-θ (the model's own form).

    Deliberately NOT splined: a smooth interpolant invents curvature the fit never estimated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, colors = _ARM_XLABEL_DEV, _ARM_COLOR
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
    curves = {}
    for a in m.arms:
        kn = np.asarray(a.knots, float)
        xs = np.linspace(kn[0], kn[-1], 400)
        curves[a.name] = (kn, xs, m.theta_at(a.name, xs), m.theta_at(a.name, kn))
    # Shared y-limits per row: on its own axis a null arm's 1% wobble looks like a finding.
    all_theta = np.concatenate([c[2] for c in curves.values()])
    all_mult = np.array([m.life_mult(float(x)) for x in all_theta])
    ylim_theta = (min(0.98, all_theta.min() * 0.97), all_theta.max() * 1.06)
    ylim_mult = (all_mult.min() * 0.96, max(1.02, all_mult.max() * 1.03))

    for j, a in enumerate(m.arms):
        kn, xs, th, thk = curves[a.name]
        ax[0, j].plot(xs, th, color=colors[a.name], lw=2.4)
        ax[0, j].plot(kn, thk, "o", color=colors[a.name], ms=5.5)
        for kk, tt in zip(kn, thk):
            lo, hi = m.boot.get(f"{a.name}@{kk:g}", (np.nan, np.nan))
            if np.isfinite(lo):
                ax[0, j].plot([kk, kk], [lo, hi], color=colors[a.name], alpha=.35, lw=1.4)
        ax[1, j].plot(xs, [m.life_mult(float(x)) for x in th], color=colors[a.name], lw=2.4)
        ax[1, j].plot(kn, [m.life_mult(float(x)) for x in thk], "o", color=colors[a.name], ms=5.5)
        shape = "МОНОТОННАЯ" if a.shape == MONO else f"ТЕНТ (мин. в {a.pin:g})"
        ax[0, j].set_title(f"θ хазард — {a.name}  [{shape}]", fontsize=11)
        ax[1, j].set_title(f"RMST(730) множитель — {a.name}", fontsize=11)
        if float(np.max(thk)) < 1.05:          # arm sitting on the constraint: say so
            ax[0, j].text(0.5, 0.86, "на границе ограничения:\nэффекта в данных нет",
                          transform=ax[0, j].transAxes, ha="center", fontsize=9.5,
                          color="#555555")
        for r in (0, 1):
            axx = ax[r, j]
            axx.axhline(1.0, color="gray", lw=.8)
            axx.axvline(a.pin, ls=":", color="k", alpha=.45)
            axx.set_xlabel(labels[a.name]); axx.grid(alpha=.25)
            axx.set_ylim(*(ylim_theta if r == 0 else ylim_mult))
    ax[0, 0].set_ylabel("θ (множитель риска)")
    ax[1, 0].set_ylabel(f"множитель RMST(0,730), база {m.life_ref()['rmst']:.0f} сут")
    fig.suptitle(
        f"Ya {m.version} k1/k2 гибрид (рейт-слой {m.rate_name}): базовая смесь "
        f"{m.baseline.model_kind} × слои риска; Kpod и частота — тент с минимумом в "
        f"Kpod {KPOD_PIN:g} / номинальной частоте; "
        f"подрядчик: slb ×{m.contractor_hr('slb'):.2f}, oth ×{m.contractor_hr('oth'):.2f}",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figures / "ya_hybrid_multipliers.png", dpi=140)
    plt.close(fig)
