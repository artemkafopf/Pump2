"""Ya **v0** — the base Ya model: nothing but the clock, the cohort and the competing risk.

Ya v0 sits *below* :mod:`ya_k1k2_hybrid` (Ya v2/v2.1) in the model stack.  Where v2 layers
shape-constrained θ-polylines for rate / Kpod / frequency on top of a mixture baseline, v0
adds **no covariate at all**.  It answers one question — *how long does a Ya pump last, and
what takes it out* — with the two things that were being papered over by fitting a single
"failure" curve:

1. **A workover is not censoring.**  754 of Ya's 1971 closed runs end in ГТМ/ППР.  The
   standard fit censors them, which silently assumes a pump pulled for a workover would have
   gone on to fail on the same schedule as one left in the well.  v0 gives ГТМ its **own
   k1/k2 latent-Weibull survival model** and reads the pair as competing risks, so the
   question "how many pulls will the workshop see next year, and what share is failure" has
   an answer that adds up.
2. **The clock and the cohort are choices, not defaults.**  Both are swept: ``t_cal`` vs
   ``t_mix``, full Ya vs Ya-modern.

Form
----
Each cause gets the repo's five-number baseline (``w1, β1, η1, β2, η2``, AIC-selected k1 vs k2
— :mod:`analysis.models.survival.mixture_baseline`), fitted **cause-specifically**: the other
cause is treated as censoring.  The two are then composed as latent failure times::

    S_all(t)  = S_fail(t) · S_gtm(t)                      (H_all = H_fail + H_gtm)
    CIF_k(t)  = ∫₀ᵗ S_all(u) dH_k(u),   H_k = −log S_k

which reproduces ``S_all + Σ CIF_k ≡ 1``.  The composition assumes the two causes are
**conditionally independent** given age — that assumption is not free, so it is *tested*, not
asserted: the parametric CIF is checked against the non-parametric Aalen–Johansen CIF, and
``S_all`` against the all-pulls KM.  Both checks are written out (``cif_check.csv``); a
parametric curve that tracks AJ is the evidence the composition is usable.

Which clock is honest — the measured answer, not the argument
-------------------------------------------------------------
The three candidates, audited on Ya in ``clock_audit.csv``:

* ``t_cal`` (pull − install) — available for **100 %** of runs, events and censorings alike,
  never imputed.  **This is the primary clock.**
* ``t_nno`` («Наработка», the mart's ``ttf_mix_svod_plus_big_nno``) — **disqualified.**  It is
  reported only for closed runs; ``load_svod_runs`` back-fills every *open* run with its
  calendar age, so ННО times the events and the calendar times the censorings.  That is the
  legacy ``tte`` defect exactly (``project_time_scales``), and since ННО < calendar it biases
  survival **up**.  61 Ya runs additionally report ННО exceeding their own calendar span.
* ``t_op`` / ``t_mix`` (telemetry operating days; ``t_mix`` = ``t_op`` measured else
  ``t_cal × Кэкспл``) — honest *only on the modern cohort*.  Measured coverage on full Ya is
  **30.9 %**, and it is **differential in the worst possible direction**: 69 % of running
  (censored) runs are measured against 30.6 % of failures and 22.5 % of ГТМ, because coverage
  is 0 % before 2020 and ~87 % from 2021.  On full Ya, ``t_mix`` is therefore ~69 %
  ``t_cal × 0.892`` — a rescaling of the calendar clock wearing an op-clock's name.  On
  **Ya-modern it is ~87 % genuinely measured**, and there it is a real second clock.

⇒ **Full Ya: fit on ``t_cal``, report ``t_mix`` as a sensitivity.  Ya-modern: both are honest,
and their gap is the first measurement of what op-time actually buys.**  ``t_nno`` is carried
for the business (ННО is what gets reported) but is never a fit clock.

Cohorts
-------
* ``full``   — every Ya run (2145 runs / 1217 failures / 754 ГТМ / 174 running).
* ``modern`` — installs on/after :data:`MODERN_INSTALL_START` (2022-01-01; 681 / 351 / 169 /
  161).  A **cohort filter on the install date**, the same construction as the Мирнинский
  2024+ rule and for the same reason — left-truncating instead would keep the old runs' later
  exposure and hide that recent runs are shorter (``feedback_mc_2024_cohort``).  The cutoff is
  swept in ``cohort_scan.csv`` so the choice is visible rather than assumed.

Reporting standard (``feedback_report_rmst_mrl``): RMST(0,730) headline, MRL(0) alongside,
median for reference, KM control always computed.  Uncertainty is a **well-cluster** bootstrap
(runs of one well are not independent).

Outputs → ``results_dir("production_risk_ya_v0")``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.models.survival import cif as CIF
from analysis.models.survival import mixture_baseline as MB
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
)
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_optime
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk.competing_risks_core import (
    CENSORED,
    FAILURE,
    GTM,
    classify_cause,
)

# ---------------------------------------------------------------------------
# Policy / constants
# ---------------------------------------------------------------------------
SLUG = "production_risk_ya_v0"
FIELD = "Ya"

#: Calendar clock — primary.  Always available, never imputed, one clock for events and
#: censorings alike.  ``t_mix`` is the op-time sensitivity (honest on ``modern`` only).
CLOCK_PRIMARY = "t_cal"
CLOCK_SENS = "t_mix"
CLOCKS = (CLOCK_PRIMARY, CLOCK_SENS)

#: Reported but never fitted — mixed-clock by construction (see the module docstring).
CLOCK_REPORTING_ONLY = "t_nno"

#: Ya-modern = installs on/after this date.  A cohort filter on the install date, NOT left
#: truncation.  2022 keeps 681 runs / 351 failures / 169 ГТМ — thick enough for a 5-parameter
#: mixture per cause, recent enough that telemetry op-time is ~87 % measured.
MODERN_INSTALL_START = date(2022, 1, 1)
COHORTS = ("full", "modern")

#: Cutoffs swept in ``cohort_scan.csv`` so the ``modern`` choice is visible, not assumed.
COHORT_SCAN_CUTS = (
    date(2019, 1, 1), date(2020, 1, 1), date(2021, 1, 1),
    date(2022, 1, 1), date(2023, 1, 1), date(2024, 1, 1),
)

CAUSES = ("failure", "gtm")
CAUSE_CODE = {"failure": FAILURE, "gtm": GTM}
CAUSE_RU = {"failure": "отказ", "gtm": "ГТМ/ППР"}

RMST_HORIZON = MB.RMST_HORIZON          # 730 d
CIF_EVAL_TIMES = (90.0, 180.0, 365.0, 730.0)

#: Bootstrap cluster — 60 %+ of runs repeat on the same well, so runs are not independent.
CLUSTER = "well_key"

#: EM multi-start counts.  The latent-Weibull likelihood is **multimodal**
#: (``project_esp_survival_em``) — a single-seed EM is a sample, not a fit — so the headline
#: fits pay for 60 starts.  The cohort sweep is a sensitivity, not a deliverable, and the
#: bootstrap warm-starts from the point estimate, so both run cheaper.
EM_STARTS = 60
SCAN_EM_STARTS = 12
BOOT_EM_STARTS = 1

N_BOOT = 200
SEED = 42


# ===========================================================================
# Population
# ===========================================================================
def build_population(
    as_of: date | pd.Timestamp | None = None,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ya runs, one row each, carrying every clock and the competing-risks cause code.

    Columns added on top of :func:`esp_population.build`:
    ``cause_code`` (0 running / 1 failure / 2 ГТМ), ``cause``, ``is_open``, ``current_age``,
    ``well_key``, and the clocks ``t_cal`` / ``t_nno`` / ``t_op`` / ``t_mix`` with
    ``t_mix_source``.

    Guards, all raising rather than warning: the open runs must sit on the calendar clock
    (the mixed-clock ``tte`` trap hit 166 of 167 open Ya runs), ``t_cal`` must be finite and
    positive, and Ya must remain a single H₂S class (it is what licenses "no stratification").
    """
    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    pop = P.add_time_scales(P.build(asof, gtm_is_failure=False), asof)
    pop = esp_optime.measure(pop, daily=daily, as_of=asof)

    ya = pop[pop["field"] == FIELD].copy()
    ended = ya["end"].notna() & (ya["end"] <= asof)
    ya["is_open"] = ~ended
    ya["cause_code"] = [
        classify_cause(r, u, e)
        for r, u, e in zip(ya["pull_reason"], ya["has_failed_unit"], ended)
    ]
    ya["cause"] = ya["cause_code"].map({CENSORED: "running", FAILURE: "failure", GTM: "gtm"})
    ya["current_age"] = ya["t_cal"].astype(float)
    ya["well_key"] = ya["code"]

    # ``t_nno`` as carried looks 100 % complete, but that completeness is manufactured:
    # ``load_svod_runs`` back-fills every OPEN run's «Наработка» with its calendar age at
    # ``SVOD_OPEN_ASOF``.  ``t_nno_reported`` strips the substitution back out, which is what
    # makes the mixed-clock defect visible in the audit instead of hiding behind a full column.
    substituted = ya["is_open"] & np.isclose(
        pd.to_numeric(ya["t_nno"], errors="coerce"),
        (asof - ya["install"]).dt.days.clip(lower=0).astype(float),
    )
    ya["t_nno_substituted"] = substituted
    ya["t_nno_reported"] = pd.to_numeric(ya["t_nno"], errors="coerce").where(~substituted)

    op = ya[ya["is_open"]]
    if len(op):
        cal_age = (asof - op["install"]).dt.days.clip(lower=P.DAY_ZERO_TTE).astype(float)
        mismatch = int((~np.isclose(op["t_cal"], cal_age)).sum())
        if mismatch:
            raise AssertionError(
                f"{mismatch} open Ya runs are not on the calendar clock (t_cal != as_of−install) "
                "— refusing to fit on a mixed clock."
            )
    if not np.isfinite(ya["t_cal"]).all() or (ya["t_cal"] <= 0).any():
        raise AssertionError("non-finite or non-positive t_cal in the Ya population.")
    classes = set(ya["h2s_class"].unique())
    if classes != {"nonsour"}:
        raise AssertionError(
            f"Ya is expected to be a single H₂S class; found {sorted(classes)} — the "
            "no-stratification design no longer holds."
        )
    return ya.reset_index(drop=True)


def cohort(pop: pd.DataFrame, name: str) -> pd.DataFrame:
    """``"full"`` → every run; ``"modern"`` → installs on/after :data:`MODERN_INSTALL_START`."""
    if name == "full":
        return pop.copy()
    if name == "modern":
        return pop[pop["install"] >= pd.Timestamp(MODERN_INSTALL_START)].copy()
    raise ValueError(f"unknown cohort {name!r}; expected one of {COHORTS}")


# ===========================================================================
# Clock audit — the "which clock is honest" evidence
# ===========================================================================
#: Clocks audited.  ``t_nno_reported`` — not ``t_nno`` — is the honest ННО column: the raw one
#: is back-filled with calendar age on every open run and so looks deceptively complete.
AUDIT_CLOCKS = ("t_cal", "t_nno_reported", "t_op", "t_mix")

#: A clock whose *measured* (non-imputed) share differs across outcomes by more than this is
#: not a common clock — it times some outcomes and estimates others.
MAX_MEASURED_SPREAD = 0.10
#: Between the two thresholds the clock is usable as a **sensitivity**, not as a primary: the
#: imputation is still differential, but mildly enough that the fit is not driven by it.
BORDERLINE_MEASURED_SPREAD = 0.25


def clock_audit(pop: pd.DataFrame) -> pd.DataFrame:
    """Per cohort × clock × outcome: availability, measured share, median, ratio to ``t_cal``.

    The disqualifying column is **``available_share`` read down the outcome rows**: a clock
    that exists for events but not for censorings (or vice versa) cannot time a survival fit,
    however good its physics.  ``t_nno_reported`` fails that test by construction (no running
    pump has an ННО) and ``t_op`` fails it empirically on full Ya; only ``t_cal`` passes
    everywhere.  Cohort is a dimension because the answer *changes* with it — telemetry
    coverage is ~0 before 2020 and ~87 % after, so op-time is unusable on full Ya and usable
    on Ya-modern.
    """
    rows = []
    for cohort_name in COHORTS:
        cg = cohort(pop, cohort_name)
        scopes = [("all", cg)] + [
            (name, cg[cg["cause"] == name]) for name in ("failure", "gtm", "running")
        ]
        for clock in AUDIT_CLOCKS:
            if clock not in pop.columns:
                continue
            for scope, g in scopes:
                if not len(g):
                    continue
                s = pd.to_numeric(g[clock], errors="coerce")
                avail = s.notna() & np.isfinite(s)
                base = pd.to_numeric(g["t_cal"], errors="coerce")
                ratio = (s / base).replace([np.inf, -np.inf], np.nan)
                if clock in ("t_op", "t_mix") and "t_mix_source" in g.columns:
                    measured = float((g["t_mix_source"] == "measured").mean())
                elif clock == "t_cal":
                    measured = 1.0
                elif clock == "t_nno_reported":
                    measured = float(avail.mean())
                else:
                    measured = np.nan
                rows.append({
                    "cohort": cohort_name,
                    "clock": clock,
                    "scope": scope,
                    "n_runs": len(g),
                    "n_available": int(avail.sum()),
                    "available_share": round(float(avail.mean()), 3),
                    "measured_share": round(measured, 3) if np.isfinite(measured) else np.nan,
                    "median_days": round(float(s.median()), 1) if avail.any() else np.nan,
                    "median_ratio_to_t_cal": round(float(ratio.median()), 3) if avail.any() else np.nan,
                })
    out = pd.DataFrame(rows)
    if "t_nno_valid" in pop.columns:
        # Reported operating time cannot exceed elapsed calendar time; these runs say it does.
        out.attrs["t_nno_impossible_runs"] = int((~pop["t_nno_valid"]).sum())
        out.attrs["t_nno_substituted_runs"] = int(pop["t_nno_substituted"].sum())
    return out


def clock_verdict(audit: pd.DataFrame) -> pd.DataFrame:
    """Per cohort × clock: the spread across outcomes and the resulting eligibility.

    ``availability_spread`` = max − min of ``available_share`` over failure / ГТМ / running.
    A non-zero spread means the clock is present for some outcomes and absent for others, so
    fitting on it applies **different clocks to events and censorings** — the defect that
    biases survival upward.  ``measured_share_spread`` catches the subtler version: a clock
    that is *always available* only because the missing part was imputed, with the imputation
    landing differentially across outcomes.  A clock must pass both to be eligible.
    """
    rows = []
    keys = ["cohort", "clock"] if "cohort" in audit.columns else ["clock"]
    sub = audit[audit["scope"].isin(("failure", "gtm", "running"))]
    for key, g in sub.groupby(keys):
        shares = g["available_share"].to_numpy(float)
        spread = float(shares.max() - shares.min())
        meas = g["measured_share"].to_numpy(float)
        meas_spread = (
            float(np.nanmax(meas) - np.nanmin(meas)) if np.isfinite(meas).any() else np.nan
        )
        meas_ok = (not np.isfinite(meas_spread)) or meas_spread < MAX_MEASURED_SPREAD
        if spread >= 1e-9:
            severity = "disqualified_structural"   # the clock does not exist for some outcomes
        elif meas_ok:
            severity = "ok"
        elif meas_spread < BORDERLINE_MEASURED_SPREAD:
            severity = "borderline_sensitivity_only"
        else:
            severity = "disqualified_differential_imputation"
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "min_available_share": round(float(shares.min()), 3),
            "max_available_share": round(float(shares.max()), 3),
            "availability_spread": round(spread, 3),
            "min_measured_share": round(float(np.nanmin(meas)), 3) if np.isfinite(meas).any() else np.nan,
            "measured_share_spread": round(meas_spread, 3) if np.isfinite(meas_spread) else np.nan,
            "severity": severity,
            "eligible_as_fit_clock": bool(spread < 1e-9 and meas_ok),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys + ["availability_spread"]).reset_index(drop=True)


# ===========================================================================
# Cause-specific k1/k2 fits
# ===========================================================================
def _clean(df: pd.DataFrame, clock: str, cause_code: int) -> tuple[np.ndarray, np.ndarray]:
    """(durations, cause-specific event indicator) with non-finite clock rows dropped."""
    t = pd.to_numeric(df[clock], errors="coerce").to_numpy(float)
    e = (df["cause_code"].to_numpy(int) == cause_code).astype(int)
    ok = np.isfinite(t) & (t > 0)
    return t[ok], e[ok]


def fit_cause(
    df: pd.DataFrame,
    clock: str,
    cause: str,
    *,
    num_starts: int = EM_STARTS,
) -> MB.BaselineFit:
    """Cause-specific k1/k2 latent-Weibull fit — **the other cause is censoring**.

    This is the *net* (latent) survival for the cause, which is what the latent-failure-time
    composition in :class:`YaV0Fit` needs.  It is not the observable probability of dying of
    that cause — that is the CIF, and it is smaller, because the competing cause removes
    pumps before they can get there.
    """
    t, e = _clean(df, clock, CAUSE_CODE[cause])
    return MB.fit_baseline(t, e, num_starts=num_starts)


#: A k2 component this short-lived, or this lightly weighted, is a spike the EM parked on a
#: handful of day-0 pulls rather than a second population.  Flagged, never silently shipped.
DEGENERATE_ETA_DAYS = 5.0
DEGENERATE_WEIGHT = 0.02


def k2_degenerate(fit: MB.BaselineFit) -> bool:
    """True when the selected k2 has a spike component instead of a second life mode.

    Ya keeps day-0 startup failures (clipped to 0.5 d rather than dropped), so a mixture is
    free to buy likelihood with a near-zero-η component carrying almost no weight.  That fits
    the data and means nothing physical, and it wrecks any RMST read off the short mode — so
    it is surfaced as a column, and a flagged cell should be read off k1 instead.
    """
    if fit.model_kind != "k2":
        return False
    return (
        min(fit.eta1, fit.eta2) < DEGENERATE_ETA_DAYS
        or min(fit.w1, 1.0 - fit.w1) < DEGENERATE_WEIGHT
    )


def _params(fit: MB.BaselineFit) -> tuple[float, float, float, float, float]:
    return fit.w1, fit.beta1, fit.eta1, fit.beta2, fit.eta2


def _S_of(fit: MB.BaselineFit):
    p = _params(fit)
    return lambda t: MB.mixture_S(t, *p)


def _rmst_of(fit: MB.BaselineFit, horizon: float = RMST_HORIZON, n: int = 2000) -> float:
    u = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(MB.mixture_S(u, *_params(fit)), u))


@dataclass
class YaV0Fit:
    """The v0 model for one cohort × clock: two cause-specific baselines, composed.

    ``S_all = S_fail · S_gtm`` is the latent-failure-time composition (equivalently
    ``H_all = H_fail + H_gtm``); it holds under **conditional independence of the two causes
    given age**.  That assumption is checked, not assumed — :func:`cif_check` compares the
    resulting CIF against Aalen–Johansen and ``S_all`` against the all-pulls KM.
    """
    cohort: str
    clock: str
    fits: dict[str, MB.BaselineFit]
    n_runs: int
    n_wells: int

    # -- per-cause net survival -------------------------------------------------
    def S_cause(self, cause: str, t):
        return MB.mixture_S(t, *_params(self.fits[cause]))

    def H_cause(self, cause: str, t):
        """Cumulative hazard of one cause, ``−log S_k`` (finite at 0 for any β)."""
        return -np.log(np.clip(self.S_cause(cause, t), 1e-300, None))

    def hazard_cause(self, cause: str, t):
        p = _params(self.fits[cause])
        return MB.mixture_pdf(t, *p) / np.clip(MB.mixture_S(t, *p), 1e-300, None)

    # -- composition ------------------------------------------------------------
    def S_all(self, t):
        """All-cause (any pull) survival — the product of the cause-specific curves."""
        out = np.ones_like(np.asarray(t, dtype=float))
        for cause in self.fits:
            out = out * np.asarray(self.S_cause(cause, t), dtype=float)
        return out

    def curve_frame(self, max_time: float = 1460.0, n_grid: int = 4000) -> pd.DataFrame:
        """Grid of ``S_all``, per-cause hazard and CIF, plus the identity residual.

        ``ΔCIF_k = S̄_all·ΔH_k`` uses the **analytic cumulative hazard** ``H_k = −log S_k``,
        which is finite at t=0 even when β<1; a naive ``∫S·h`` trapezoid diverges there (Ya's
        failure component has β≈0.79, so this is not hypothetical).  ``S̄_all`` is the interval
        midpoint rather than the left endpoint — same scheme as
        ``competing_risks_core.competing_curve_frame``, and it halves ``identity_error`` at no
        cost, which matters because that residual is the only signal that the grid is too
        coarse for the horizon asked of it.
        """
        u = np.linspace(0.0, float(max_time), int(n_grid))
        s_all = self.S_all(u)
        s_mid = 0.5 * (s_all[:-1] + s_all[1:])
        frame = {"time": u, "s_all": s_all, "all_cause_failure": 1.0 - s_all}
        total = np.zeros_like(u)
        for cause in self.fits:
            H = self.H_cause(cause, u)
            cif = np.concatenate([[0.0], np.cumsum(s_mid * np.diff(H))])
            total += cif
            frame[f"hazard_{cause}"] = self.hazard_cause(cause, u)
            frame[f"cif_{cause}"] = cif
        frame["identity_error"] = s_all + total - 1.0
        return pd.DataFrame(frame)

    def life(self, cause: str) -> dict:
        """RMST(0,730) / MRL(0) / median of the cause's **net** curve."""
        return MB.life_from_S(_S_of(self.fits[cause]))

    def life_all(self) -> dict:
        """The same triple for the composed all-cause (any-pull) curve."""
        return MB.life_from_S(self.S_all)

    def window_probability(self, age: float, horizon: float, *, max_time: float | None = None) -> dict:
        """P(cause k takes this pump out within ``horizon``) given survival to ``age``.

        The deliverable form: conditional cumulative incidence, ``(CIF_k(age+h) − CIF_k(age))
        / S_all(age)``.  Shares sum to the all-cause pull probability, so the workshop load
        and its failure/ГТМ split come from one object.
        """
        top = float(max_time if max_time is not None else (age + horizon) * 1.2 + 1.0)
        f = self.curve_frame(max_time=top)
        s_now = float(np.interp(age, f["time"], f["s_all"]))
        out = {}
        for cause in self.fits:
            c0 = float(np.interp(age, f["time"], f[f"cif_{cause}"]))
            c1 = float(np.interp(age + horizon, f["time"], f[f"cif_{cause}"]))
            out[cause] = float(np.clip((c1 - c0) / max(s_now, 1e-12), 0.0, 1.0))
        s_fut = float(np.interp(age + horizon, f["time"], f["s_all"]))
        out["any_pull"] = float(np.clip(1.0 - s_fut / max(s_now, 1e-12), 0.0, 1.0))
        return out


def fit_v0(
    df: pd.DataFrame,
    clock: str,
    cohort_name: str,
    *,
    num_starts: int = EM_STARTS,
) -> YaV0Fit:
    fits = {c: fit_cause(df, clock, c, num_starts=num_starts) for c in CAUSES}
    return YaV0Fit(
        cohort=cohort_name,
        clock=clock,
        fits=fits,
        n_runs=len(df),
        n_wells=int(df[CLUSTER].nunique()),
    )


# ===========================================================================
# Non-parametric controls
# ===========================================================================
def km_cause(df: pd.DataFrame, clock: str, cause: str | None,
             horizon: float = RMST_HORIZON) -> dict:
    """KM control for one cause (other cause censored), or all-pulls when ``cause is None``.

    Mandatory alongside every fit (``feedback_report_rmst_mrl``).  ``mrl`` here is the KM
    integral to the last observed time — a **lower bound** on MRL(0) when the tail is
    censored, unlike the model's MRL which extrapolates the fitted curve.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    t = pd.to_numeric(df[clock], errors="coerce").to_numpy(float)
    if cause is None:
        e = (df["cause_code"].to_numpy(int) != CENSORED).astype(int)
    else:
        e = (df["cause_code"].to_numpy(int) == CAUSE_CODE[cause]).astype(int)
    ok = np.isfinite(t) & (t > 0)
    t, e = t[ok], e[ok]
    if len(t) < 5 or e.sum() < 1:
        return {"rmst": np.nan, "mrl": np.nan, "median": np.nan, "n": len(t), "events": int(e.sum())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KaplanMeierFitter().fit(t, e)
        rmst = float(restricted_mean_survival_time(km, t=horizon))
        mrl = float(restricted_mean_survival_time(km, t=float(t.max())))
        med = float(km.median_survival_time_)
    return {"rmst": round(rmst, 1), "mrl": round(mrl, 1), "median": round(med, 1),
            "n": len(t), "events": int(e.sum())}


def cif_check(df: pd.DataFrame, clock: str, model: YaV0Fit,
              times: tuple[float, ...] = CIF_EVAL_TIMES) -> pd.DataFrame:
    """The independence check: parametric CIF vs Aalen–Johansen, and the naive 1−KM gap.

    Three numbers per cause per horizon:

    * ``cif_aj``        — non-parametric truth (no independence assumption);
    * ``cif_parametric`` — what the composed v0 model says;
    * ``naive_1_km``    — what you get treating the competing cause as free censoring.

    ``cif_parametric ≈ cif_aj`` is the licence to use the model.  ``naive_1_km − cif_aj`` is
    the cost of *not* modelling the competing risk — the overstatement that a failure-only
    fit ships into a forecast.
    """
    t = pd.to_numeric(df[clock], errors="coerce").to_numpy(float)
    ec = df["cause_code"].to_numpy(int)
    ok = np.isfinite(t) & (t > 0)
    t, ec = t[ok], ec[ok]

    aj = {c: CIF.aalen_johansen_cif(t, ec, CAUSE_CODE[c]) for c in CAUSES}
    naive = {c: CIF.all_cause_km(t, (ec == CAUSE_CODE[c]).astype(int)) for c in CAUSES}
    all_km = CIF.all_cause_km(t, (ec != CENSORED).astype(int))
    frame = model.curve_frame(max_time=max(float(max(times)) * 1.5, float(t.max()) + 1.0))

    rows = []
    for tt in times:
        s_all_model = float(np.interp(tt, frame["time"], frame["s_all"]))
        for c in CAUSES:
            cif_p = float(np.interp(tt, frame["time"], frame[f"cif_{c}"]))
            cif_a = float(aj[c].at(tt))
            nk = float(naive[c].at(tt))
            rows.append({
                "cohort": model.cohort, "clock": clock, "cause": c, "time_d": tt,
                "cif_aj": round(cif_a, 4),
                "cif_parametric": round(cif_p, 4),
                "param_minus_aj": round(cif_p - cif_a, 4),
                "naive_1_km": round(nk, 4),
                "naive_overstatement_abs": round(nk - cif_a, 4),
                "naive_overstatement_pct": round(100.0 * (nk - cif_a) / cif_a, 1) if cif_a > 0 else np.nan,
                "s_all_model": round(s_all_model, 4),
                "s_all_km": round(1.0 - float(all_km.at(tt)), 4),
                "s_all_model_minus_km": round(s_all_model - (1.0 - float(all_km.at(tt))), 4),
                "identity_error": round(float(np.interp(tt, frame["time"], frame["identity_error"])), 5),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Well-cluster bootstrap
# ===========================================================================
def _refit_like(t: np.ndarray, e: np.ndarray, fit: MB.BaselineFit,
                starts: int) -> tuple[float, float, float, float, float] | None:
    """Refit **the selected model kind**, warm-started from the point estimate.

    Re-running the k1-vs-k2 AIC selection inside every bootstrap replicate would mix two
    different models into one interval; the interval reported is for the model that was
    selected.  Warm starting also makes the EM ~40× cheaper than a cold 60-start fit, which
    is what makes a 200-replicate cluster bootstrap affordable at all.
    """
    from lifelines import WeibullFitter

    try:
        if fit.model_kind == "k1":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = WeibullFitter().fit(t, e)
            b, eta = float(wf.rho_), float(wf.lambda_)
            if not (1e-2 < b < 1e2 and 0 < eta < 1e6):
                return None
            return 0.0, b, eta, b, eta
        init = TwoComponentLatentWeibullModel(
            weight_1=float(np.clip(fit.w1, 1e-4, 1 - 1e-4)),
            component_1=WeibullParameters(beta=fit.beta1, eta=fit.eta1),
            component_2=WeibullParameters(beta=fit.beta2, eta=fit.eta2),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fit_latent_weibull_em(t, e, initial_model=init, num_starts=starts)
        m = r.model
        p = (float(m.weight_1), float(m.component_1.beta), float(m.component_1.eta),
             float(m.component_2.beta), float(m.component_2.eta))
        if not all(np.isfinite(p)) or p[2] <= 0 or p[4] <= 0:
            return None
        return p
    except Exception:
        return None


def bootstrap_cause(
    df: pd.DataFrame,
    clock: str,
    cause: str,
    fit: MB.BaselineFit,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    horizon: float = RMST_HORIZON,
) -> dict:
    """Percentile CIs for β₁/η₁/RMST(0,730) by resampling **wells**, not runs.

    Runs of one well share its fluid, its completion and its operator, so a run-level
    bootstrap understates the spread.  Replicates that fail to fit are dropped and counted
    (``n_boot_ok``) rather than silently backfilled.
    """
    code = CAUSE_CODE[cause]
    wells = df[CLUSTER].dropna().unique()
    by_well = {w: g for w, g in df.groupby(CLUSTER)}
    rng = np.random.default_rng(seed)

    b1s, e1s, rmsts = [], [], []
    for _ in range(int(n_boot)):
        drawn = rng.choice(wells, size=len(wells), replace=True)
        boot = pd.concat([by_well[w] for w in drawn], ignore_index=True)
        t = pd.to_numeric(boot[clock], errors="coerce").to_numpy(float)
        e = (boot["cause_code"].to_numpy(int) == code).astype(int)
        ok = np.isfinite(t) & (t > 0)
        t, e = t[ok], e[ok]
        if e.sum() < 5 or len(t) < 10:
            continue
        p = _refit_like(t, e, fit, BOOT_EM_STARTS)
        if p is None:
            continue
        u = np.linspace(0.0, horizon, 1500)
        rmsts.append(float(np.trapezoid(MB.mixture_S(u, *p), u)))
        b1s.append(p[1])
        e1s.append(p[2])

    def _ci(a: list[float]) -> tuple[float, float]:
        if len(a) < max(10, int(0.1 * n_boot)):
            return (np.nan, np.nan)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    b_lo, b_hi = _ci(b1s)
    e_lo, e_hi = _ci(e1s)
    r_lo, r_hi = _ci(rmsts)
    return {
        "n_boot_ok": len(rmsts),
        "beta1_lo": round(b_lo, 4) if np.isfinite(b_lo) else np.nan,
        "beta1_hi": round(b_hi, 4) if np.isfinite(b_hi) else np.nan,
        "eta1_lo": round(e_lo, 1) if np.isfinite(e_lo) else np.nan,
        "eta1_hi": round(e_hi, 1) if np.isfinite(e_hi) else np.nan,
        "rmst_lo": round(r_lo, 1) if np.isfinite(r_lo) else np.nan,
        "rmst_hi": round(r_hi, 1) if np.isfinite(r_hi) else np.nan,
    }


# ===========================================================================
# Cohort sweep
# ===========================================================================
def cohort_scan(
    pop: pd.DataFrame,
    clock: str = CLOCK_PRIMARY,
    cuts: tuple[date, ...] = COHORT_SCAN_CUTS,
    *,
    num_starts: int = SCAN_EM_STARTS,
) -> pd.DataFrame:
    """Refit both causes at each install-date cutoff — makes the ``modern`` choice visible.

    Two things move together as the cutoff advances and must be read together: the cohort
    genuinely gets younger (recent runs are shorter), and the **administrative censoring gets
    heavier** (a 2024 install cannot show a 900-day life by 2026).  A drop in RMST across the
    sweep is therefore only evidence of a real trend if it survives the censoring, which the
    censored fit handles but the raw ``events/runs`` column does not — both are reported.
    """
    rows = []
    for cut in cuts:
        g = pop[pop["install"] >= pd.Timestamp(cut)]
        row = {
            "install_cut": str(cut),
            "n_runs": len(g),
            "n_wells": int(g[CLUSTER].nunique()),
            "n_failure": int((g["cause_code"] == FAILURE).sum()),
            "n_gtm": int((g["cause_code"] == GTM).sum()),
            "n_running": int((g["cause_code"] == CENSORED).sum()),
            "censored_share": round(float((g["cause_code"] == CENSORED).mean()), 3),
            "max_t_cal": round(float(g["t_cal"].max()), 0) if len(g) else np.nan,
            "measured_op_share": round(float((g["t_mix_source"] == "measured").mean()), 3),
        }
        for cause in CAUSES:
            n_ev = int((g["cause_code"] == CAUSE_CODE[cause]).sum())
            if n_ev < 20:
                row.update({f"{cause}_model": "too_thin", f"{cause}_rmst730": np.nan,
                            f"{cause}_km_rmst730": np.nan})
                continue
            f = fit_cause(g, clock, cause, num_starts=num_starts)
            row[f"{cause}_model"] = f.model_kind
            row[f"{cause}_beta1"] = round(f.beta1, 3)
            row[f"{cause}_eta1"] = round(f.eta1, 1)
            row[f"{cause}_rmst730"] = round(_rmst_of(f), 1)
            row[f"{cause}_km_rmst730"] = km_cause(g, clock, cause)["rmst"]
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Diagnostics
# ===========================================================================
def population_table(pop: pd.DataFrame) -> pd.DataFrame:
    """Per cohort: runs, wells, causes, censored share, and op-time measured share."""
    rows = []
    for name in COHORTS:
        g = cohort(pop, name)
        rows.append({
            "cohort": name,
            "install_from": "all" if name == "full" else str(MODERN_INSTALL_START),
            "n_runs": len(g),
            "n_wells": int(g[CLUSTER].nunique()),
            "n_failure": int((g["cause_code"] == FAILURE).sum()),
            "n_gtm": int((g["cause_code"] == GTM).sum()),
            "n_running": int((g["cause_code"] == CENSORED).sum()),
            "gtm_share_of_pulls": round(
                float((g["cause_code"] == GTM).sum() / max((g["cause_code"] != CENSORED).sum(), 1)), 3),
            "censored_share": round(float((g["cause_code"] == CENSORED).mean()), 3),
            "measured_op_share": round(float((g["t_mix_source"] == "measured").mean()), 3),
            "median_t_cal": round(float(g["t_cal"].median()), 1),
            "median_t_mix": round(float(g["t_mix"].median()), 1),
            # Day-0/near-day-0 pulls are KEPT (clipped to 0.5 d), not filtered away.  They are
            # real startup failures, but they are also what a mixture can park a spike
            # component on — see ``k2_degenerate``.
            "runs_under_2d": int((g["t_cal"] <= 2.0).sum()),
            "failures_under_2d": int(((g["t_cal"] <= 2.0) & (g["cause_code"] == FAILURE)).sum()),
            "t_nno_substituted": int(g["t_nno_substituted"].sum()),
            "t_nno_impossible": int((~g["t_nno_valid"]).sum()) if "t_nno_valid" in g else 0,
        })
    return pd.DataFrame(rows)


def cause_assignment_audit(pop: pd.DataFrame) -> pd.DataFrame:
    """How every closed run got its cause — the classifier's own coverage.

    Kept visible because one branch is a judgement call: a closed run whose «Причина
    остановки» is blank falls through to ``has_failed_unit``, i.e. the presence of a named
    failed component decides.  On Ya that branch carries ~94 runs, all of which land on
    *failure*; if that inheritance is wrong the failure count is overstated by that much and
    the ГТМ count is untouched.
    """
    from analysis.workflows.production_risk.esp_population import (
        GENUINE_FAILURE_REASONS, WORKOVER_REASONS, _norm,
    )
    closed = pop[~pop["is_open"]].copy()
    reason = closed["pull_reason"].map(_norm).str.casefold()
    branch = np.where(
        reason.isin(WORKOVER_REASONS), "reason_workover",
        np.where(reason.isin(GENUINE_FAILURE_REASONS), "reason_failure",
                 np.where(closed["has_failed_unit"], "fallback_failed_unit", "fallback_censored")),
    )
    closed["branch"] = branch
    out = (closed.groupby(["branch", "cause"]).size().rename("n_runs").reset_index())
    out["share_of_closed"] = (out["n_runs"] / len(closed)).round(4)
    return out.sort_values("n_runs", ascending=False).reset_index(drop=True)


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class YaV0Run:
    as_of: pd.Timestamp
    pop: pd.DataFrame
    models: dict[tuple[str, str], YaV0Fit]      # (cohort, clock) -> fit
    tables: dict[str, pd.DataFrame] = dc_field(default_factory=dict)
    out_dir: Path | None = None

    def model(self, cohort_name: str = "full", clock: str = CLOCK_PRIMARY) -> YaV0Fit:
        return self.models[(cohort_name, clock)]

    @property
    def headline(self) -> YaV0Fit:
        """Full Ya on the calendar clock — the model to quote unless a reason says otherwise."""
        return self.models[("full", CLOCK_PRIMARY)]


def run(
    as_of: date | pd.Timestamp | None = None,
    *,
    num_starts: int = EM_STARTS,
    n_boot: int = N_BOOT,
    scan_starts: int = SCAN_EM_STARTS,
    write: bool = True,
    pop: pd.DataFrame | None = None,
) -> YaV0Run:
    """Fit Ya v0 over cohort × clock × cause and write the tables and figures."""
    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    pop = build_population(asof) if pop is None else pop

    audit = clock_audit(pop)
    tables: dict[str, pd.DataFrame] = {
        "clock_audit": audit,
        "clock_verdict": clock_verdict(audit),
        "population": population_table(pop),
        "cause_assignment_audit": cause_assignment_audit(pop),
    }

    models: dict[tuple[str, str], YaV0Fit] = {}
    fit_rows, cif_rows = [], []
    for cohort_name in COHORTS:
        g = cohort(pop, cohort_name)
        for clock in CLOCKS:
            m = fit_v0(g, clock, cohort_name, num_starts=num_starts)
            models[(cohort_name, clock)] = m

            for cause in CAUSES:
                f = m.fits[cause]
                life = m.life(cause)
                km = km_cause(g, clock, cause)
                boot = bootstrap_cause(g, clock, cause, f, n_boot=n_boot)
                fit_rows.append({
                    "cohort": cohort_name, "clock": clock, "cause": cause,
                    "n_runs": f.n, "n_events": f.events,
                    "model_kind": f.model_kind,
                    "w1": round(f.w1, 4), "beta1": round(f.beta1, 4), "eta1": round(f.eta1, 1),
                    "beta2": round(f.beta2, 4), "eta2": round(f.eta2, 1),
                    "delta_aic_k2_minus_k1": f.delta_aic,
                    "k2_degenerate_spike": k2_degenerate(f),
                    "k2_short_life": f.k2_short_life, "k2_long_life": f.k2_long_life,
                    "rmst_0_730": life["rmst"], "mrl_0": life["mrl"], "median": life["median"],
                    "km_rmst_0_730": km["rmst"], "km_mrl_lower_bound": km["mrl"],
                    "km_median": km["median"],
                    "rmst_model_minus_km": round(life["rmst"] - km["rmst"], 1)
                    if np.isfinite(km["rmst"]) else np.nan,
                    **boot,
                })
            # all-cause row: the composed curve against the all-pulls KM
            life_all = m.life_all()
            km_all = km_cause(g, clock, None)
            fit_rows.append({
                "cohort": cohort_name, "clock": clock, "cause": "all_pulls(composed)",
                "n_runs": len(g), "n_events": km_all["events"], "model_kind": "composed",
                "rmst_0_730": life_all["rmst"], "mrl_0": life_all["mrl"],
                "median": life_all["median"],
                "km_rmst_0_730": km_all["rmst"], "km_mrl_lower_bound": km_all["mrl"],
                "km_median": km_all["median"],
                "rmst_model_minus_km": round(life_all["rmst"] - km_all["rmst"], 1)
                if np.isfinite(km_all["rmst"]) else np.nan,
            })
            cif_rows.append(cif_check(g, clock, m))

    tables["cause_fits"] = pd.DataFrame(fit_rows)
    tables["cif_check"] = pd.concat(cif_rows, ignore_index=True)
    tables["cohort_scan"] = cohort_scan(pop, CLOCK_PRIMARY, num_starts=scan_starts)
    tables["window_probability"] = _window_table(models)

    obj = YaV0Run(as_of=asof, pop=pop, models=models, tables=tables)
    if write:
        _write_outputs(obj)
    return obj


def _window_table(models: dict[tuple[str, str], YaV0Fit],
                  ages: tuple[float, ...] = (0.0, 90.0, 180.0, 365.0, 730.0),
                  horizon: float = 365.0) -> pd.DataFrame:
    """P(pull within one year | alive at age a), split failure vs ГТМ — the deliverable form."""
    rows = []
    for (cohort_name, clock), m in models.items():
        for a in ages:
            p = m.window_probability(a, horizon)
            rows.append({
                "cohort": cohort_name, "clock": clock, "age_d": a, "horizon_d": horizon,
                "p_failure": round(p["failure"], 4),
                "p_gtm": round(p["gtm"], 4),
                "p_any_pull": round(p["any_pull"], 4),
                "failure_share_of_pulls": round(p["failure"] / max(p["failure"] + p["gtm"], 1e-12), 3),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================
_CAUSE_COLOR = {"failure": "#c0392b", "gtm": "#2471a3"}


def _km_step(t: np.ndarray, e: np.ndarray):
    from lifelines import KaplanMeierFitter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KaplanMeierFitter().fit(t, e)
    sf = km.survival_function_
    return sf.index.to_numpy(float), sf.iloc[:, 0].to_numpy(float)


def fig_cause_km_fit(run_obj: YaV0Run, path: Path, clock: str = CLOCK_PRIMARY) -> None:
    """Cause-specific KM vs the fitted k1/k2 curve, per cohort × cause — the fit control."""
    fig, axes = plt.subplots(len(COHORTS), len(CAUSES), figsize=(11, 8), sharex=True, sharey=True)
    for i, cohort_name in enumerate(COHORTS):
        g = cohort(run_obj.pop, cohort_name)
        m = run_obj.model(cohort_name, clock)
        for j, cause in enumerate(CAUSES):
            ax = axes[i][j]
            t, e = _clean(g, clock, CAUSE_CODE[cause])
            kt, ks = _km_step(t, e)
            ax.step(kt, ks, where="post", color="0.35", lw=1.3, label="KM (cause-specific)")
            u = np.linspace(0.0, 1460.0, 800)
            ax.plot(u, m.S_cause(cause, u), color=_CAUSE_COLOR[cause], lw=2.0,
                    label=f"k1/k2 fit ({m.fits[cause].model_kind})")
            life = m.life(cause)
            km = km_cause(g, clock, cause)
            ax.set_title(
                f"{cohort_name} · {cause} ({CAUSE_RU[cause]})\n"
                f"n={m.fits[cause].n}, events={m.fits[cause].events} · "
                f"RMST₇₃₀ fit {life['rmst']:.0f} d / KM {km['rmst']:.0f} d",
                fontsize=9,
            )
            ax.set_xlim(0, 1460)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if i == len(COHORTS) - 1:
                ax.set_xlabel(f"{clock} (days)")
            if j == 0:
                ax.set_ylabel("net survival S(t)")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Ya v0 — cause-specific latent survival, clock={clock}\n"
        "(net curves: the other cause treated as censoring)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cif(run_obj: YaV0Run, path: Path, clock: str = CLOCK_PRIMARY) -> None:
    """Parametric CIF vs Aalen–Johansen vs the naive 1−KM — the independence check, drawn."""
    fig, axes = plt.subplots(1, len(COHORTS), figsize=(12, 5), sharey=True)
    for i, cohort_name in enumerate(COHORTS):
        ax = axes[i]
        g = cohort(run_obj.pop, cohort_name)
        m = run_obj.model(cohort_name, clock)
        t = pd.to_numeric(g[clock], errors="coerce").to_numpy(float)
        ec = g["cause_code"].to_numpy(int)
        ok = np.isfinite(t) & (t > 0)
        t, ec = t[ok], ec[ok]
        frame = m.curve_frame(max_time=1460.0)
        u = np.linspace(0.0, 1460.0, 500)
        for cause in CAUSES:
            aj = CIF.aalen_johansen_cif(t, ec, CAUSE_CODE[cause])
            ax.step(u, aj.at(u), where="post", color=_CAUSE_COLOR[cause], lw=1.2, alpha=0.55,
                    label=f"AJ CIF {cause}")
            ax.plot(frame["time"], frame[f"cif_{cause}"], color=_CAUSE_COLOR[cause], lw=2.0,
                    ls="--", label=f"v0 parametric CIF {cause}")
            naive = CIF.all_cause_km(t, (ec == CAUSE_CODE[cause]).astype(int))
            ax.plot(u, naive.at(u), color=_CAUSE_COLOR[cause], lw=1.0, ls=":", alpha=0.8,
                    label=f"naive 1−KM {cause}")
        ax.set_title(f"{cohort_name} (n={len(t)})", fontsize=10)
        ax.set_xlabel(f"{clock} (days)")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1460)
        if i == 0:
            ax.set_ylabel("cumulative incidence")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Ya v0 — competing-risks incidence: dashed = model, solid = Aalen–Johansen, "
        "dotted = naive 1−KM\n(the dotted/solid gap is what censoring ГТМ as if it were "
        "random costs you)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_hazards(run_obj: YaV0Run, path: Path, clock: str = CLOCK_PRIMARY) -> None:
    """Cause-specific hazard shapes — where the infant spike is and whether ГТМ wears out."""
    fig, axes = plt.subplots(1, len(COHORTS), figsize=(12, 4.6), sharey=True)
    u = np.linspace(1.0, 1460.0, 800)
    for i, cohort_name in enumerate(COHORTS):
        ax = axes[i]
        m = run_obj.model(cohort_name, clock)
        for cause in CAUSES:
            ax.plot(u, m.hazard_cause(cause, u) * 1000.0, color=_CAUSE_COLOR[cause], lw=2.0,
                    label=f"{cause} (β₁={m.fits[cause].beta1:.2f})")
        ax.set_yscale("log")
        ax.set_title(f"{cohort_name}", fontsize=10)
        ax.set_xlabel(f"{clock} (days)")
        ax.grid(alpha=0.25, which="both")
        if i == 0:
            ax.set_ylabel("hazard per 1000 days")
            ax.legend(fontsize=8)
    fig.suptitle("Ya v0 — cause-specific hazards (log scale)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_clock(run_obj: YaV0Run, path: Path) -> None:
    """The clock question, drawn: availability by outcome, and what the clock does to S(t)."""
    audit = run_obj.tables["clock_audit"]
    outcomes = ("failure", "gtm", "running")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    def _cell(cohort_name: str, clock: str, scope: str, col: str) -> float:
        s = audit[(audit["cohort"] == cohort_name) & (audit["clock"] == clock)
                  & (audit["scope"] == scope)][col]
        return float(s.iloc[0]) if len(s) else np.nan

    ax = axes[0]
    clocks = [c for c in AUDIT_CLOCKS if (audit["clock"] == c).any()]
    x = np.arange(len(clocks))
    width = 0.25
    for k, scope in enumerate(outcomes):
        ax.bar(x + (k - 1) * width,
               [_cell("full", c, scope, "available_share") for c in clocks],
               width, label=scope)
    ax.set_xticks(x)
    ax.set_xticklabels(clocks, fontsize=8, rotation=15)
    ax.set_ylabel("share of runs with the clock available")
    ax.set_title("full Ya — availability by outcome\n"
                 "(unequal bars ⇒ mixed clock ⇒ disqualified)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    x = np.arange(len(outcomes))
    for k, cohort_name in enumerate(COHORTS):
        ax.bar(x + (k - 0.5) * 0.36,
               [_cell(cohort_name, "t_mix", s, "measured_share") for s in outcomes],
               0.36, label=cohort_name)
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share MEASURED (not imputed)")
    ax.set_title("t_mix: measured share by outcome\n"
                 "(flat on modern ⇒ usable; skewed on full ⇒ not)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    for cohort_name, ls in zip(COHORTS, ("-", "--")):
        for clock, col in zip(CLOCKS, ("#c0392b", "#2471a3")):
            m = run_obj.model(cohort_name, clock)
            u = np.linspace(0.0, 1460.0, 600)
            ax.plot(u, m.S_cause("failure", u), ls=ls, color=col, lw=1.7,
                    label=f"{cohort_name} · {clock}")
    ax.set_xlabel("days")
    ax.set_ylabel("failure-cause net S(t)")
    ax.set_title("What the clock choice does to the fit", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1460)

    fig.suptitle("Ya v0 — clock audit", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_outputs(run_obj: YaV0Run) -> Path:
    out = results_dir(SLUG)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for name, df in run_obj.tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    fig_cause_km_fit(run_obj, out / "figures" / "ya_v0_cause_km_fit.png")
    fig_cif(run_obj, out / "figures" / "ya_v0_cif.png")
    fig_hazards(run_obj, out / "figures" / "ya_v0_hazards.png")
    fig_clock(run_obj, out / "figures" / "ya_v0_clock.png")
    run_obj.out_dir = out
    return out


__all__ = [
    "SLUG", "FIELD", "CLOCK_PRIMARY", "CLOCK_SENS", "CLOCKS", "COHORTS",
    "MODERN_INSTALL_START", "CAUSES", "CAUSE_CODE", "RMST_HORIZON",
    "build_population", "cohort", "clock_audit", "clock_verdict",
    "fit_cause", "fit_v0", "YaV0Fit", "km_cause", "cif_check", "bootstrap_cause",
    "cohort_scan", "population_table", "cause_assignment_audit",
    "run", "YaV0Run",
]
