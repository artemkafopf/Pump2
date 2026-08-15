"""Unified v4 — one model structure for **Ya, Vt_nonsour and Vt_sour**, every layer a polyline.

Motivation
----------
The field models had drifted apart: Vt v4 carried a nameplate polyline plus an imported
tent for frequency and θ_Kpod ≡ 1; Ya v3 blended two rate arms and dropped Kpod entirely.
Worse, the two "Kpod is null" verdicts that justified dropping it were **artifacts of the
tent**: a shape pinned at 0.8 with θ ≥ 1 on both arms cannot represent a monotone effect, so
it returns θ ≡ 1 whenever the truth is monotone.  On an unconstrained form the effect is
large and clean — Vt HR 1.90 [1.47, 2.47], Ya HR 1.28 [1.11, 1.46], both essentially
orthogonal to nameplate rate (corr(log Kpod, log Qnom) = −0.14 Vt, +0.01 Ya).

So every layer here is a ``FREE`` polyline: piecewise-linear in log-θ, pinned at an
interpretable reference, clamped outside the knots, and **no assumed direction or optimum**.
Whether Qnom comes out monotone, whether Kpod has a valley and whether frequency has one are
answers the fit gives, not premises it is handed.

Structure (identical in all three strata)::

    h(t | x) = (β₀/η₀)(t/η₀)^(β₀−1) · HR_contractor · θ_Qnom(q) · θ_Kpod(k) · θ_freq(Δf)

Estimation is three-stage, so a layer measured on the population that can identify it is
held fixed for the strata that cannot:

1. **Contractor** levels from the Ql overlap window :data:`CONTRACTOR_WINDOW`, per field.
   Comparing contractors only where their rate ranges overlap is what stops the contractor
   term from quietly absorbing the rate effect (the low-rate slb wells were being penalised
   for their rate).
2. **Kpod and frequency** from the whole field at once (Vt: both strata, sour entering as a
   linear dummy), contractor carried as a fixed offset.
3. **Baseline (β₀, η₀) and θ_Qnom per stratum**, with contractor + Kpod + freq as offset.

Stage 2 pools the shape across Vt's strata deliberately: Vt_sour has 81 events, which cannot
carry ~20 free polyline parameters.  :func:`stratum_shape_lr` reports the likelihood-ratio
test for whether sour actually wants its own Kpod/freq — pooling is a decision that is
checked, not assumed.  θ_Qnom stays stratum-specific because the ≈2× difference in the rate
slope between sour and nonsour is an established finding, not a sample-size artifact.

Not in the model
----------------
**Ql.**  With Qnom and Kpod both present Ql is redundant by construction — log Ql = log Qnom
+ log Kpod — and forcing all three in blows the fit apart (Kpod 5.17 / Qnom 5.53 / Ql 0.25,
mutually cancelling).  Qnom and Kpod are the identified pair; Ql is their product.

**A Ya sour stratum.**  All 2145 Ya runs carry a single H₂S class, so there is nothing to
split.  Ya is one stratum by measurement, not by choice.

⚠ ``kpod_run`` is telemetry-derived and is *not* ``ql / qnom``.  In the Свод panel the two
coincide, which is where the "Kpod is just the Ql residual" claim came from; in the modelling
frame they do not (corr 0.82 Vt, 0.90 Ya, max discrepancy 2.6×).  Kpod is a free covariate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival.polyline_ph import (
    FREE, ArmSpec, PolylinePHFit, bootstrap_theta, eval_polyline, fit_polyline_ph)

SLUG = "production_risk_unified_v4"

CLOCK = "t_cal"
EVENT_COL = "event"
CLUSTER = "code"

#: The two fields fitted at full depth.  The replication fields in :data:`REPLICATION_FIELDS`
#: use the same structure with a form scaled to their evidence — see :mod:`field_v4`.
STRATA = ("Ya", "Vt_nonsour", "Vt_sour")

#: Fields carried for the calculators, in descending event count.  Da is fitted but not offered
#: in the workbooks (it is not on the КРС_ЭПУ roster).
REPLICATION_FIELDS = ("Az", "Ic", "Za", "Mc", "Da")

#: The pooled fallback stratum, fitted over every field at once — see :mod:`field_v4`.
FLEET = "Fleet"

FIELD_OF = {"Ya": "Ya", "Vt_nonsour": "Vt", "Vt_sour": "Vt", FLEET: FLEET,
            **{f: f for f in REPLICATION_FIELDS}}

CONTRACTOR_REF = "brt"
CONTRACTOR_TERMS = ("slb", "oth")

RMST_HORIZON = 730.0

#: Contractors are compared only inside this Ql window, where all three actually overlap.
CONTRACTOR_WINDOW = (200.0, 500.0)

# --- knots -----------------------------------------------------------------
# Chosen from the pooled support (roughly the 2.5–97.5 percentile span of every stratum) and
# log-spaced for rate, even-spaced for the two ratios.  Outside the range θ is clamped flat —
# an assumption, and the one place this model still extrapolates.
QNOM_KNOTS = (60.0, 100.0, 160.0, 250.0, 400.0, 640.0, 1000.0, 1600.0)
QNOM_REF = 250.0
QNOM_CAP = 1600.0

KPOD_KNOTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6)
KPOD_REF = 0.8

FREQ_KNOTS = (-22.0, -15.0, -9.0, -4.0, 0.0, 4.0, 9.0)
FREQ_REF = 0.0

#: L2 on the polyline increments, per field, chosen by 5-fold well-clustered CV of the
#: held-out Weibull log-likelihood (``scripts/run/unified_v4.py --cv``).  Every arm is
#: two-sided, so the ridge shrinks toward θ ≡ 1 — the honest null — rather than rectifying
#: noise upward the way a one-sided arm does.  Ya wants markedly heavier shrinkage than Vt,
#: which is the same message as :data:`LAYER_CV_DELTA`: its layers carry less.
#:
#: The replication fields carry 150–165 events each (Mc: 42 in its 2024+ cohort) against Vt's
#: 208 and Ya's 635, so they start at Ya-level shrinkage or heavier.  These are **provisional**
#: until ``--cv`` is run per field; `field_v4` records the CV that justified whatever ships.
RIDGE = {
    "Ya": {"qnom": 12.0, "kpod": 32.0, "freq": 32.0},
    "Vt": {"qnom": 6.0, "kpod": 16.0, "freq": 16.0},
    "Az": {"qnom": 16.0, "kpod": 32.0, "freq": 32.0},
    "Ic": {"qnom": 16.0, "kpod": 32.0, "freq": 32.0},
    "Za": {"qnom": 16.0, "kpod": 32.0, "freq": 32.0},
    "Da": {"qnom": 32.0, "kpod": 64.0, "freq": 64.0},
    "Mc": {"qnom": 32.0, "kpod": 64.0, "freq": 64.0},
    # Pooled over heterogeneous fields, so shrink like Ya rather than like Vt: the layers
    # have to describe seven baselines at once and a wiggle that suits one is noise for the
    # others.
    "Fleet": {"qnom": 12.0, "kpod": 32.0, "freq": 32.0},
}
RIDGE_DEFAULT = RIDGE["Vt"]

#: Out-of-sample CV gain of each layer over ``Qnom + contractor`` alone, in log-likelihood
#: units (5 folds, clustered on well).  **Read this before deploying a layer.**
#:
#: * Kpod is the real find: **+8.4 on Vt**, +2.0 on Ya.  The tent form recovers only +4.1 on
#:   Vt and is *negative* (−1.4) on Ya — the shape constraint, not the data, produced every
#:   previous "Kpod is null" verdict.
#: * Frequency **does not earn its place in either field** (Ya +1.1, Vt −0.8, both inside
#:   noise on 635/208 events).  This corroborates the earlier finding that the Ya frequency
#:   U-valley is a failures-only collider: it halves and dies once censored runs enter.
#:   The layer is still fitted and plotted so the evidence is visible, and it is retained as
#:   a *tunable* deliverable for the prognosis model — but it is not a measured effect.
LAYER_CV_DELTA = {
    ("Ya", "kpod"): 2.03, ("Ya", "freq"): 1.08,
    ("Vt", "kpod"): 8.35, ("Vt", "freq"): -0.76,
}

#: Every arm is ``FREE``: piecewise-linear in log-θ, pinned, clamped, no assumed direction.
#: Fitted free, Qnom comes out monotone by itself and Kpod comes out monotone-rising in both
#: fields — so the monotone constraint is *confirmed* here rather than imposed, and it costs
#: ≤ 0.2 CV units to leave it off.  That is the whole point of asking with a free shape.
SHAPES = {"qnom": FREE, "kpod": FREE, "freq": FREE}


def qnom_fold_per_decade(theta, knots=QNOM_KNOTS) -> float:
    """The curve's average slope, read as a **θ fold-change per decade of Qnom**.

    A readable summary of the whole nameplate layer in one number: 1.0 is flat, 1.5 means the
    failure rate rises by half for every tenfold increase in nameplate flow.
    """
    k = np.asarray(knots, float)
    th = np.asarray(theta, float)
    decades = np.log10(k[-1] / k[0])
    if decades <= 0 or th[0] <= 0 or th[-1] <= 0:
        return 1.0
    return float((th[-1] / th[0]) ** (1.0 / decades))


def deploy_theta_qnom(theta, knots=QNOM_KNOTS, ref: float = QNOM_REF) -> np.ndarray:
    """Deployment guard: never let θ_Qnom **fall** above the reference.

    The fit is left exactly as fitted; this is applied only when parameters are exported to a
    calculator.  Vt_sour is why: fitted free, θ goes 1.3668 at Qnom 1000 → **1.2034 at 1600**,
    i.e. the shipped VBA models a 1600 pump as longer-lived than a 1000 one.  That is 20 runs /
    14 events above Qnom 640 talking, not physics — and an operator reading the curve as
    "buy bigger" would be acting on noise.

    Below the reference nothing is touched: the low end is where the data is thickest and a
    genuinely protective small-pump segment is a finding, not an artefact.
    """
    th = np.array(theta, dtype=float, copy=True)
    k = np.asarray(knots, float)
    above = np.flatnonzero(k >= float(ref))
    for i, j in zip(above[:-1], above[1:]):
        if th[j] < th[i]:
            th[j] = th[i]
    return th


def rotate_qnom(theta, fold_per_decade: float, knots=QNOM_KNOTS,
                pivot: float = QNOM_REF) -> np.ndarray:
    """Rotate the whole nameplate curve about ``pivot`` to a target average slope.

    One number per stratum.  The polyline is linear in log θ against log Qnom, so "steepen it"
    is literally adding a constant slope in that space — a rotation::

        g = (ln F − ln F₀) / ln 10                       F = target fold per decade
        θ'(k) = θ(k) · exp(g · (ln k − ln pivot))

    Two properties make this the right control rather than moving individual knots:

    * **θ(pivot) stays exactly 1**, so the reference point — and with it the baseline η₀ and
      every RMST the model reports at reference — is untouched.  Nothing is silently
      recalibrated underneath.
    * **Every wiggle in the fitted shape is preserved.**  Rotation changes the trend and
      nothing else, so the local structure the data actually paid for survives; what the
      operator is overriding is only the overall steepness, which is the part where the
      bootstrap band is widest (Vt_nonsour θ@1600 CI [1.00, 3.69]).

    Below the pivot the curve moves the *opposite* way to above it — that is what a rotation
    is, and it is the honest consequence of holding the reference fixed.
    """
    th = np.array(theta, dtype=float, copy=True)
    k = np.asarray(knots, float)
    f0 = qnom_fold_per_decade(th, k)
    if fold_per_decade <= 0 or f0 <= 0:
        return th
    g = (np.log(float(fold_per_decade)) - np.log(f0)) / np.log(10.0)
    return th * np.exp(g * (np.log(k) - np.log(float(pivot))))


def arms(names=("qnom", "kpod", "freq"), *, field: str | None = None) -> tuple:
    """The polyline arms, in fit order, with this field's CV-chosen ridge."""
    r = RIDGE.get(field, RIDGE_DEFAULT)
    spec = {
        "qnom": ArmSpec("qnom", "qnom", QNOM_KNOTS, QNOM_REF, SHAPES["qnom"], r["qnom"]),
        "kpod": ArmSpec("kpod", "kpod_run", KPOD_KNOTS, KPOD_REF, SHAPES["kpod"], r["kpod"]),
        "freq": ArmSpec("freq", "freq_dev", FREQ_KNOTS, FREQ_REF, SHAPES["freq"], r["freq"]),
    }
    return tuple(spec[n] for n in names)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
#: Calculator field code → repo field label.  The workbooks' «УН» / «Участок недр» list is
#: Au, Az, Ic, Mc, Mr, Vt, Ya; the modelling frame labels two of those differently because
#: :data:`config.FIELD_PREFIX_MAP` folds the pads in at population build time — ``AU*`` →
#: ``Za`` and ``MR`` → ``Mc`` (131 MC + 63 MR runs sit in the one ``Mc`` frame).
FIELD_ALIAS = {"Au": "Za", "Mr": "Mc"}

#: Fields whose population is a **cohort**, filtered on install date.  Мирнинский only:
#: pre-2024 Mc is the anomaly (2 failures vs 11.8 expected on 21 runs), and left-truncating
#: instead keeps the old runs' later exposure and hides that recent runs are shorter.
FIELD_COHORT_START = {"Mc": "MC_INSTALL_COHORT_START"}

#: Vt is the only field with two H₂S classes.  Ya carries a single class by measurement, and
#: the replication fields have no usable H₂S at all — never invent a split.
SPLIT_H2S = ("Vt",)

FNOM_DEFAULT = 50.0
FNOM_PLAUSIBLE = (40.0, 65.0)


def build_cached_frame(as_of=None) -> pd.DataFrame:
    """The shared covariate frame, built **once**, with the v3.2 sour relabel forced on.

    ⚠ Do not hand :func:`prepare_field` a frame built any other way.  ``vt_v32_hybrid``'s
    ``prepare_frame`` only forces ``config.SOUR_WELL_LEVEL_ALL_RUNS`` when it builds the frame
    itself; passing a plain ``cached=`` silently skips the relabel, 30 Vt running pumps stay in
    ``Vt_nonsour``, and every Vt number moves (contractor slb 1.200 → 1.172, oth 1.910 → 1.992)
    with nothing in the output saying why.  The relabel is a no-op for the other fields — they
    carry a single H₂S class — so one frame serves them all.
    """
    from analysis.workflows.production_risk import config as C
    from analysis.workflows.production_risk import vt_ttf_covariates as V

    prev = C.SOUR_WELL_LEVEL_ALL_RUNS
    C.SOUR_WELL_LEVEL_ALL_RUNS = True
    try:
        return V.build_frame(as_of=as_of)[0]
    finally:
        C.SOUR_WELL_LEVEL_ALL_RUNS = prev


def _attach_canonical_qnom(d: pd.DataFrame) -> pd.DataFrame:
    """Vendor-catalogue nameplate + cap, so ``qnom`` means the same thing in every field."""
    from analysis.data.pump_nominal_catalog import canonical_qnom
    from analysis.workflows.production_risk import vt_v4 as V

    d = d.copy()
    d["qnom_raw"] = pd.to_numeric(d["nominal_flow_m3d"], errors="coerce")
    d["qnom_recorded"] = d["qnom_raw"]
    d = V.attach_pump_designation(d)
    res = [canonical_qnom(g, q) for g, q in zip(d["gno_type"], d["qnom_raw"])]
    d["qnom_raw"] = [r[0] if r[0] is not None else np.nan for r in res]
    d["pump_family"] = [r[1] for r in res]
    d["qnom_source"] = [r[2] for r in res]
    d = d.dropna(subset=["qnom_raw"]).copy()
    d["qnom_capped"] = d["qnom_raw"] > QNOM_CAP
    d["qnom"] = d["qnom_raw"].clip(upper=QNOM_CAP)
    return d


def _prepare_generic(field: str, *, as_of=None, cached: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """One replication field off the shared covariate frame (Az, Ic, Za/Au, Mc+Mr, Da).

    Mirrors what :mod:`ya_k1k2_hybrid` does for Ya — clock > 0, nominal-frequency deviation,
    contractor dummies — rather than inventing a second frame convention.
    """
    from analysis.workflows.production_risk import config as C
    from analysis.workflows.production_risk import vt_ttf_covariates as V

    df = cached if cached is not None else V.build_frame(as_of=as_of)[0]
    d = df[df["field"] == field].copy()
    d[CLOCK] = pd.to_numeric(d[CLOCK], errors="coerce")
    d = d[d[CLOCK] > 0].dropna(subset=[CLOCK, EVENT_COL]).copy()

    cohort_attr = FIELD_COHORT_START.get(field)
    if cohort_attr is not None:
        start = pd.Timestamp(getattr(C, cohort_attr))
        d = d[pd.to_datetime(d["install"], errors="coerce") >= start].copy()

    fn = pd.to_numeric(d.get("nominal_freq_hz"), errors="coerce")
    d["f_nom"] = np.where(fn.between(*FNOM_PLAUSIBLE), fn, FNOM_DEFAULT)
    d["freq_dev"] = pd.to_numeric(d["freq_run"], errors="coerce") - d["f_nom"]
    d = _attach_canonical_qnom(d)
    for cg in CONTRACTOR_TERMS:
        d[cg] = (d["contractor_group"] == cg).astype(float)
    d["stratum"] = field
    return d


def prepare_field(field: str, *, as_of=None, cached: pd.DataFrame | None = None) -> pd.DataFrame:
    """Modelling frame for one field with the canonical nameplate rate attached.

    Vt reuses :mod:`vt_v4`'s frame (v3.2 relabeling + vendor catalogue); Ya reuses
    :mod:`ya_k1k2_hybrid`'s; every other field comes off the shared covariate frame.  All three
    paths end in the same catalogue rules, so ``qnom`` means one thing across fields.

    ``field`` accepts either a repo label or a calculator code (:data:`FIELD_ALIAS`).
    """
    from analysis.workflows.production_risk import vt_v4 as V

    field = FIELD_ALIAS.get(field, field)
    if field == "Vt":
        d = V.prepare_frame(as_of=as_of, cached=cached)
        d["stratum"] = "Vt_" + d["h2s_class"].astype(str)
    elif field == "Ya":
        from analysis.workflows.production_risk import ya_k1k2_hybrid as YA

        d = _attach_canonical_qnom(YA.prepare_frame(as_of=as_of, cached=cached))
        d["stratum"] = "Ya"
    else:
        d = _prepare_generic(field, as_of=as_of, cached=cached)

    d["ql"] = pd.to_numeric(d["ql"], errors="coerce")
    for c in ("kpod_run", "freq_dev"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def complete_case(d: pd.DataFrame, names=("qnom", "kpod", "freq")) -> pd.DataFrame:
    cols = [a.column for a in arms(names)] + [CLOCK, EVENT_COL, "contractor_group"]
    return d.dropna(subset=cols).copy()


# ---------------------------------------------------------------------------
# Stage 1 — contractor on the rate-overlap window
# ---------------------------------------------------------------------------
#: Adjust the contractor level for the size layers it would otherwise absorb.
#:
#: Restricting to a **Ql** window does not equalise the pumps: ``log Ql = log Qnom + log Kpod``,
#: so at fixed Ql a contractor that runs bigger pumps simply runs them at a lower Kpod.  Measured
#: on Vt inside the window (n=154, 88 events): slb geomean Qnom **465** vs brt **354**, median
#: Kpod **0.569** vs **0.712**.  With no size term the level absorbs part of that, and θ_Qnom —
#: fitted afterwards against a frozen level — then applies it a second time.
#:
#: Adding ``log(Qnom/ref)`` and ``log(Kpod/ref)`` makes the returned number a **reference-point**
#: level: the contractor effect at Qnom 250 / Kpod 0.8, which is exactly where the θ layers are
#: pinned.  Effect on Vt: slb 1.200 → 1.196, oth 1.910 → 1.750 (the size term is 1.85/e-fold and
#: Kpod 2.18/e-fold, and slb's two biases largely cancel — oth's do not).
#:
#: ⚠ The cancellation for slb holds against the **fitted** Kpod layer.  The shipped calculators
#: use the operator bathtub instead, which has the opposite sign at low Kpod, so a typical slb
#: pump still composes ~34 % higher hazard there.  That is a stated deployment assumption, not a
#: fit defect — see the note written onto ``МодельОтказов``.
CONTRACTOR_SIZE_ADJUST = True

#: A contractor term is only estimable where it actually appears inside the window.  Ic has 8
#: ``oth`` runs in the whole field (2 in the window) and Mc has **no** ``slb`` at all; fitting a
#: near-empty dummy returns a number with no information in it.  Below this support the term is
#: dropped and that contractor falls back to the reference level, which is recorded, not hidden.
CONTRACTOR_MIN_RUNS = 10
CONTRACTOR_MIN_EVENTS = 5

#: Ridge on every term in the stage-1 fit.
#:
#: The two size adjusters are **standardised inside the window** before fitting, so one ridge
#: constant means the same thing in every field.  It matters: inside a narrow Ql window
#: ``log Qnom`` and ``log Kpod`` are near-collinear by construction — their sum is ``log Ql``,
#: which the window has just pinned — and their spread differs field to field, so a ridge on the
#: raw log scale would shrink Az five times harder than Vt for no stated reason.  Coefficients
#: are reported both ways (``size_terms`` per e-fold, ``size_terms_sd`` per SD).
CONTRACTOR_PENALTY = 0.1


def fit_contractor_levels(d: pd.DataFrame, window: tuple | None = None,
                          *, size_adjust: bool | None = None) -> dict:
    """Contractor hazard levels (brt = 1) from the Ql overlap window.

    ``window`` resolves at CALL time so sensitivity runs that reassign the module constant
    actually take effect.  ``size_adjust`` (default :data:`CONTRACTOR_SIZE_ADJUST`) controls the
    Qnom/Kpod adjusters; pass ``False`` to reproduce the pre-2026-08 levels.
    """
    from lifelines import CoxPHFitter

    adjust = CONTRACTOR_SIZE_ADJUST if size_adjust is None else size_adjust
    lo, hi = CONTRACTOR_WINDOW if window is None else window
    ov = d[(d["ql"] >= lo) & (d["ql"] <= hi)].copy()
    for cg in CONTRACTOR_TERMS:
        ov[cg] = (ov["contractor_group"] == cg).astype(float)
    sd = {}
    if adjust:
        ov["lq"] = np.log(pd.to_numeric(ov["qnom"], errors="coerce") / QNOM_REF)
        ov["lk"] = np.log(pd.to_numeric(ov["kpod_run"], errors="coerce") / KPOD_REF)
        ov = ov.dropna(subset=["lq", "lk"]).copy()
        for c in ("lq", "lk"):
            sd[c] = max(float(ov[c].std()), 1e-9)
            ov[c] = ov[c] / sd[c]

    terms, unsupported = [], {}
    for cg in CONTRACTOR_TERMS:
        m = ov["contractor_group"] == cg
        n_ev = int(ov.loc[m, EVENT_COL].sum())
        if int(m.sum()) >= CONTRACTOR_MIN_RUNS and n_ev >= CONTRACTOR_MIN_EVENTS:
            terms.append(cg)
        else:
            unsupported[cg] = {"n": int(m.sum()), "events": n_ev}

    # Stratum fixed effects: comparing contractors across strata that differ in baseline
    # (Vt sour vs nonsour, or one field vs another in the pooled fleet fit) would let the
    # contractor term absorb the stratum mix.  All-but-the-first stratum gets a dummy —
    # for Vt that is exactly the single `Vt_sour` indicator this has always used.
    cols = [CLOCK, EVENT_COL, *terms]
    strata = sorted(ov["stratum"].unique()) if "stratum" in ov else []
    for s in strata[1:]:
        name = f"s_{s}"
        ov[name] = (ov["stratum"] == s).astype(float)
        cols.append(name)
    if adjust:
        cols += ["lq", "lk"]

    level = {CONTRACTOR_REF: 1.0}
    ci = {CONTRACTOR_REF: (1.0, 1.0, float("nan"))}
    size_terms: dict = {}
    size_terms_sd: dict = {}
    if terms:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=CONTRACTOR_PENALTY).fit(ov[cols], CLOCK, EVENT_COL)
        for cg in terms:
            s = cph.summary.loc[cg]
            level[cg] = float(s["exp(coef)"])
            ci[cg] = (float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]),
                      float(s["p"]))
        for t in ("lq", "lk"):
            if adjust and t in cph.summary.index:
                coef = float(cph.summary.loc[t, "coef"])
                size_terms_sd[t] = float(np.exp(coef))            # per SD in the window
                size_terms[t] = float(np.exp(coef / sd[t]))       # per e-fold
    for cg in unsupported:
        level[cg] = 1.0
        ci[cg] = (float("nan"), float("nan"), float("nan"))
    return {"level": level, "ci": ci, "n": len(ov), "events": int(ov[EVENT_COL].sum()),
            "window": (lo, hi), "size_adjust": bool(adjust), "size_terms": size_terms,
            "size_terms_sd": size_terms_sd, "unsupported": unsupported}


def contractor_offset(d: pd.DataFrame, level: dict) -> np.ndarray:
    return np.log([level.get(c, 1.0) for c in d["contractor_group"]])


# ---------------------------------------------------------------------------
# Out-of-sample check — the gate every layer has to pass before it ships
# ---------------------------------------------------------------------------
def held_out_loglik(fit, d: pd.DataFrame, offset: np.ndarray) -> float:
    """Weibull log-likelihood of held-out rows under a fitted model."""
    lp = np.asarray(offset, float).copy()
    for a in fit.arms:
        lp = lp + eval_polyline(d[a.column], a.knots, np.log(fit.theta_knots[a.name]))
    for term, c in fit.linear.items():
        lp = lp + d[term].to_numpy(float) * c
    t = d[CLOCK].to_numpy(float)
    e = d[EVENT_COL].to_numpy(float)
    lt, b, le = np.log(t), fit.beta0, np.log(fit.eta0)
    return float(np.sum(e * (lp + np.log(b) - le + (b - 1) * (lt - le))
                        - np.exp(lp) * np.exp(b * (lt - le))))


def cross_validate(cc: pd.DataFrame, arm_specs, k: int = 5, seed: int = 11,
                   *, offset: str = "off_contractor") -> float:
    """K-fold CV **clustered on well** — runs of one well are dependent, so folds split wells.

    Splitting on runs instead would leak a well's own history into its test fold and make every
    layer look better than it is.
    """
    lin = ("sour",) if cc["stratum"].nunique() > 1 else ()
    wells = cc[CLUSTER].unique()
    fold = pd.Series(np.random.default_rng(seed).integers(0, k, len(wells)), index=wells)
    total = 0.0
    for i in range(k):
        te = cc[cc[CLUSTER].map(fold) == i]
        tr = cc[cc[CLUSTER].map(fold) != i]
        if len(te) == 0 or te[EVENT_COL].sum() == 0:
            continue
        f = fit_polyline_ph(tr, clock=CLOCK, event_col=EVENT_COL, arms=arm_specs,
                            linear_terms=lin, offset=offset,
                            eta0_start=float(tr[CLOCK].median()))
        total += held_out_loglik(f, te, te[offset].to_numpy(float))
    return total


# ---------------------------------------------------------------------------
# Stages 2–3
# ---------------------------------------------------------------------------
@dataclass
class StratumFit:
    """One stratum's deployable parameters."""
    stratum: str
    n: int
    events: int
    beta0: float
    eta0: float
    theta: dict                      # arm -> θ at that arm's knots
    contractor: dict                 # group -> HR
    loglik: float
    concordance: float
    shared_shape: tuple = ()         # arms taken from the pooled field fit
    boot: dict = dc_field(default_factory=dict)

    # -- evaluation ---------------------------------------------------------
    def theta_at(self, name: str, x) -> np.ndarray:
        knots = {"qnom": QNOM_KNOTS, "kpod": KPOD_KNOTS, "freq": FREQ_KNOTS}[name]
        return np.exp(eval_polyline(x, knots, np.log(self.theta[name])))

    def theta_total(self, *, contractor: str = CONTRACTOR_REF, qnom=QNOM_REF,
                    kpod=KPOD_REF, freq_dev=FREQ_REF) -> np.ndarray:
        return (self.contractor.get(contractor, 1.0)
                * self.theta_at("qnom", qnom)
                * self.theta_at("kpod", kpod)
                * self.theta_at("freq", freq_dev))

    def compose(self, **kw) -> dict:
        """θ's multiply; **RMST multipliers do not** — convert once, at the end.

        The standing trap in this workflow: composing per-layer RMST multipliers instead of
        composing θ and converting once overstates the effect by ~30 %.
        """
        th = float(np.atleast_1d(self.theta_total(**kw))[0])
        eta = self.eta0 * th ** (-1.0 / self.beta0)
        r = rmst(eta, self.beta0)
        return {"theta_total": th, "eta_eff": eta, "rmst": r,
                "rmst_mult": r / rmst(self.eta0, self.beta0)}

    def rmst_mult_curve(self, name: str, x) -> np.ndarray:
        """RMST(0, 730) relative to the reference, moving one layer only."""
        th = self.theta_at(name, x)
        base = rmst(self.eta0, self.beta0)
        eta = self.eta0 * th ** (-1.0 / self.beta0)
        return np.array([rmst(float(e), self.beta0) for e in np.atleast_1d(eta)]) / base


def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 4000) -> float:
    """∫₀^h S(t) dt for a Weibull — the reported life quantity throughout this workflow.

    Not the Weibull mean ηΓ(1+1/β): with β < 1 that is dominated by a tail nobody observes.
    """
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


def fit_field(d: pd.DataFrame, *, field: str | None = None, level: dict | None = None,
              n_boot: int = 0, seed: int = 7) -> dict:
    """Stages 1–3 for one field.  Returns ``{stratum: StratumFit}`` plus the diagnostics."""
    cc = complete_case(d)
    if field is None:
        field = FIELD_OF[cc["stratum"].iloc[0]]
    lvl = fit_contractor_levels(cc) if level is None else {"level": level}
    cc["off_contractor"] = contractor_offset(cc, lvl["level"])

    strata = sorted(cc["stratum"].unique())
    lin = ()
    if len(strata) > 1:
        cc["sour"] = (cc["stratum"] == "Vt_sour").astype(float)
        lin = ("sour",)

    # stage 2 — Kpod & freq shapes from the whole field
    pooled = fit_polyline_ph(cc, clock=CLOCK, event_col=EVENT_COL, arms=arms(field=field),
                             linear_terms=lin, offset="off_contractor",
                             eta0_start=float(cc[CLOCK].median()))

    # eval_polyline over log-θ already RETURNS log-θ — the offset is a log-hazard, so these
    # terms go in as they come out.  (Wrapping them in another log silently NaNs every row.)
    cc["off_all"] = (cc["off_contractor"]
                     + eval_polyline(cc["kpod_run"], KPOD_KNOTS,
                                     np.log(pooled.theta_knots["kpod"]))
                     + eval_polyline(cc["freq_dev"], FREQ_KNOTS,
                                     np.log(pooled.theta_knots["freq"])))

    # stage 3 — baseline + θ_Qnom per stratum
    out, boots = {}, {}
    for s in strata:
        g = cc[cc["stratum"] == s]
        f = fit_polyline_ph(g, clock=CLOCK, event_col=EVENT_COL,
                            arms=arms(("qnom",), field=field),
                            offset="off_all", eta0_start=float(g[CLOCK].median()))
        if n_boot:
            boots[s] = bootstrap_theta(g, clock=CLOCK, event_col=EVENT_COL,
                                       arms=arms(("qnom",), field=field), cluster=CLUSTER,
                                       n_boot=n_boot, seed=seed, offset="off_all",
                                       min_events=max(10, int(0.5 * f.events)))
        out[s] = StratumFit(
            stratum=s, n=f.n, events=f.events, beta0=f.beta0, eta0=f.eta0,
            theta={"qnom": f.theta_knots["qnom"],
                   "kpod": pooled.theta_knots["kpod"],
                   "freq": pooled.theta_knots["freq"]},
            contractor=dict(lvl["level"]), loglik=f.loglik, concordance=f.concordance,
            shared_shape=("kpod", "freq") if len(strata) > 1 else (),
            boot=boots.get(s, {}))

    shape_boot = {}
    if n_boot:
        shape_boot = bootstrap_theta(cc, clock=CLOCK, event_col=EVENT_COL,
                                     arms=arms(field=field), linear_terms=lin,
                                     cluster=CLUSTER, n_boot=n_boot, seed=seed,
                                     offset="off_contractor")
    return {"fits": out, "pooled": pooled, "contractor": lvl, "shape_boot": shape_boot,
            "frame": cc}


def stratum_shape_lr(cc: pd.DataFrame) -> dict:
    """LR test: does Vt_sour want its own Kpod/freq polylines, or does pooling hold?

    Pooling those two shapes is what makes an 81-event stratum estimable at all, so the test
    is reported with the model rather than left implicit.
    """
    field = FIELD_OF[cc["stratum"].iloc[0]]
    A = arms(field=field)
    lin = ("sour",) if cc["stratum"].nunique() > 1 else ()
    if lin:
        cc = cc.copy()
        cc["sour"] = (cc["stratum"] == "Vt_sour").astype(float)
    shared = fit_polyline_ph(cc, clock=CLOCK, event_col=EVENT_COL, arms=A,
                             linear_terms=lin, offset="off_contractor")
    ll, npar = 0.0, 0
    for s in sorted(cc["stratum"].unique()):
        f = fit_polyline_ph(cc[cc["stratum"] == s], clock=CLOCK, event_col=EVENT_COL,
                            arms=A, offset="off_contractor")
        ll += f.loglik
        npar += 2 + sum(a.n_increments for a in A)
    from scipy.stats import chi2

    npar_shared = 2 + len(lin) + sum(a.n_increments for a in A)
    dof = max(npar - npar_shared, 1)
    stat = 2.0 * (ll - shared.loglik)
    return {"loglik_shared": shared.loglik, "loglik_split": ll, "dof": dof,
            "lr": stat, "p": float(chi2.sf(max(stat, 0.0), dof)),
            "aic_shared": 2 * npar_shared - 2 * shared.loglik,
            "aic_split": 2 * npar - 2 * ll}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
@dataclass
class ModelRun:
    fits: dict
    contractor: dict
    pooled: dict
    shape_boot: dict
    frames: dict
    lr: dict


def run(*, as_of=None, cached: pd.DataFrame | None = None, n_boot: int = 0,
        write: bool = True, seed: int = 7) -> ModelRun:
    fits, contr, pooled, sboot, frames = {}, {}, {}, {}, {}
    for field in ("Ya", "Vt"):
        r = fit_field(prepare_field(field, as_of=as_of, cached=cached),
                      field=field, n_boot=n_boot, seed=seed)
        fits.update(r["fits"])
        contr[field] = r["contractor"]
        pooled[field] = r["pooled"]
        sboot[field] = r["shape_boot"]
        frames[field] = r["frame"]
    obj = ModelRun(fits=fits, contractor=contr, pooled=pooled, shape_boot=sboot,
                   frames=frames, lr={"Vt": stratum_shape_lr(frames["Vt"])})
    if write:
        _write_outputs(obj)
    return obj


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
KNOTS = {"qnom": QNOM_KNOTS, "kpod": KPOD_KNOTS, "freq": FREQ_KNOTS}


def layer_ci(obj: ModelRun, stratum: str, layer: str, knot: float) -> tuple:
    """Bootstrap interval for one knot — from the stratum's own resample for θ_Qnom, from the
    field-level resample for the two shapes that are fitted pooled."""
    key = f"{layer}@{knot:g}"
    src = (obj.fits[stratum].boot if layer == "qnom"
           else obj.shape_boot.get(FIELD_OF[stratum], {}))
    return src.get(key, (np.nan, np.nan))


def layer_table(obj: ModelRun) -> pd.DataFrame:
    rows = []
    for s, f in obj.fits.items():
        for name, knots in KNOTS.items():
            mult = f.rmst_mult_curve(name, np.asarray(knots, float))
            for k, th, m in zip(knots, f.theta[name], mult):
                lo, hi = layer_ci(obj, s, name, k)
                rows.append({"stratum": s, "layer": name, "x": k,
                             "theta": round(float(th), 4),
                             "theta_lo": lo, "theta_hi": hi,
                             "rmst_mult": round(float(m), 4),
                             "shared_shape": name in f.shared_shape,
                             "cv_delta": LAYER_CV_DELTA.get((FIELD_OF[s], name), np.nan)})
    return pd.DataFrame(rows)


def baseline_table(obj: ModelRun) -> pd.DataFrame:
    return pd.DataFrame([{
        "stratum": s, "n": f.n, "events": f.events,
        "beta0": round(f.beta0, 4), "eta0": round(f.eta0, 1),
        "rmst730_ref": round(rmst(f.eta0, f.beta0), 1),
        "concordance": f.concordance,
        **{f"hr_{c}": round(v, 4) for c, v in f.contractor.items()},
    } for s, f in obj.fits.items()])


LAYER_LABEL = {
    "qnom": ("Номинальная подача Qном, м³/сут", "log"),
    "kpod": ("Коэффициент подачи Kпод", "linear"),
    "freq": ("Отклонение частоты от номинала, Гц", "linear"),
}
STRATUM_COLOR = {"Ya": "#3B7EA1", "Vt_nonsour": "#C4622D", "Vt_sour": "#7A4E9E"}


def fig_layers(obj: ModelRun, figures: Path) -> Path:
    """3 layers × 2 rows — hazard θ on top, RMST(0, 730) multiplier beneath.

    Two panels answer different questions and must not be read off one curve: θ is what
    multiplies in the model, the RMST multiplier is what the owner feels in days.  They are
    *not* proportional, and RMST multipliers do not compose.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4), sharex="col")
    for j, layer in enumerate(("qnom", "kpod", "freq")):
        knots = np.asarray(KNOTS[layer], float)
        grid = np.linspace(knots[0], knots[-1], 400)
        label, scale = LAYER_LABEL[layer]
        ax_t, ax_r = axes[0, j], axes[1, j]
        for s, f in obj.fits.items():
            c = STRATUM_COLOR[s]
            if layer in f.shared_shape and s == "Vt_sour":
                continue                        # identical to Vt_nonsour by construction
            nm = "Vt (общая форма)" if layer in f.shared_shape else s
            ax_t.plot(grid, f.theta_at(layer, grid), color=c, lw=2, label=nm)
            ax_t.plot(knots, f.theta[layer], "o", color=c, ms=5)
            lo = np.array([layer_ci(obj, s, layer, k)[0] for k in knots], float)
            hi = np.array([layer_ci(obj, s, layer, k)[1] for k in knots], float)
            if np.isfinite(lo).any():
                ax_t.fill_between(knots, lo, hi, color=c, alpha=0.13, lw=0)
            ax_r.plot(grid, f.rmst_mult_curve(layer, grid), color=c, lw=2, label=nm)
            ax_r.plot(knots, f.rmst_mult_curve(layer, knots), "o", color=c, ms=5)
        for ax in (ax_t, ax_r):
            ax.axhline(1.0, color="0.4", lw=0.9, ls=":")
            ax.set_xscale(scale)
            ax.grid(alpha=0.25, lw=0.6)
        cvd = {k[0]: v for k, v in LAYER_CV_DELTA.items() if k[1] == layer}
        note = ("  ".join(f"{f} CV {v:+.1f}" for f, v in cvd.items())) if cvd else "опорный слой"
        ax_t.set_title(f"θ_{layer}   ({note})", fontsize=11)
        ax_t.legend(fontsize=8.5, frameon=False, loc="best")
        ax_r.set_xlabel(label, fontsize=10)
        # How many knots actually clear 1?  A busy line with 1/14 clearing knots is noise;
        # stating the count keeps the reader from reading either verdict off the shape alone.
        spans = [layer_ci(obj, s, layer, k) for s in obj.fits
                 for k in knots if not (layer in obj.fits[s].shared_shape and s == "Vt_sour")]
        spans = [(lo, hi) for lo, hi in spans if np.isfinite(lo)]
        clear = sum(1 for lo, hi in spans if not (lo <= 1.0 <= hi))
        if spans:
            ax_r.text(0.5, 0.055, f"95 % ДИ не накрывает 1: {clear} из {len(spans)} узлов",
                      transform=ax_r.transAxes, ha="center", fontsize=9,
                      color="#8A2B2B" if clear <= 2 else "#2F6B3A")
    axes[0, 0].set_ylabel("θ — множитель интенсивности отказов", fontsize=10)
    axes[1, 0].set_ylabel("Множитель RMST(0, 730), доли", fontsize=10)
    fig.suptitle("Unified v4 — свободные полилинии (форма не задана априори).  "
                 "Полоса = кластерный бутстрэп по скважинам, 95 %", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = figures / "unified_v4_layers.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _write_outputs(obj: ModelRun) -> Path:
    from analysis.paths import results_dir

    out = results_dir(SLUG)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    layer_table(obj).to_csv(out / "tables" / "layers.csv", index=False, encoding="utf-8-sig")
    baseline_table(obj).to_csv(out / "tables" / "baselines.csv", index=False,
                               encoding="utf-8-sig")
    pd.DataFrame([{"field": k, **v} for k, v in obj.lr.items()]).to_csv(
        out / "tables" / "shape_pooling_lr.csv", index=False, encoding="utf-8-sig")
    fig_layers(obj, out / "figures")
    return out
