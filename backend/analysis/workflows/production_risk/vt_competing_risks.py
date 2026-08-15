"""Вахитовское (Vt) failure/ГТМ competing-risks accounting, H2S from LAB only.

The Vt sibling of :mod:`mc_competing_risks`.  It reuses the shared engine in
:mod:`competing_risks_core` (cause-specific Weibull + well-cluster bootstrap, band
hazards, Aalen–Johansen CIF vs naive 1−KM, the two-level informative-censoring
test, the renewal engine and the five-column decomposition) and answers the two
standing Vt questions, where the priors are the OPPOSITE of Mc's:

* **slb vs brt (collider resolution).**  The known "brt/slb look equal on
  failure-only data" is suspected to be an informative-censoring collider.  We redo
  it with cause-specific hazards and AJ CIF (ГТМ a competing risk, not free
  censoring) and ask whether a real contractor difference emerges — separately for
  sour and nonsour.
* **ГТМ informativeness (Vt_pulled).**  On Vt ГТМ is expected rate-correlated
  (informative), unlike Mc.  We quantify it (association + telemetry precursors)
  and state what it does to the latent no-ГТМ counterfactual: on Vt the latent
  column is expected to be UNDERSTATED, so it is reported as a lower bound.

**H2S class comes from lab.sqlite ONLY** (``svb_h2s_indicator`` in ``lab_samples``,
``'+'``-or-null, qualitative).  The Свод «Кислый/Некислый» flag and the V03
``h2s_mg_l`` column are both rejected here (the flag misfiled 30 running pumps; the
continuous column has a 754-vs-<1 mixed-units trap and informative missingness).
The lab class is **well-level** and three-way:

* ``sour_lab``    — any sample of the well ever carries ``'+'``;
* ``nonsour_lab`` — the well has lab samples but never ``'+'`` (a WEAK negative:
  ``'+'`` is the only value ever recorded, so this is "not detected", not
  "measured zero");
* ``no_lab``      — the well has no lab samples.  **Never** defaulted to nonsour;
  reported as its own column, fitted only if events allow.

Vt uses **all history** (the installs-2024+ cohort rule is Mc-only).  Sour and
nonsour are **never pooled** (~3× apart in life and different in shape).

Outputs → ``results_dir("production_risk_vt_competing_risks")``.  Do NOT commit.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival import cif as CIF
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_life_quantile,
)
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.paths import resolve_lab_db_path
from analysis.workflows.production_risk import competing_risks_core as core
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk.competing_risks_core import (
    CENSORED,
    FAILURE,
    GTM,
    HAZARD_BANDS,
    band_hazards_from_pop,
    cif_vs_km,
    classify_cause,
    combined_decomposition,
    fit_cause_specific,
    observed_window_prediction,
    piecewise_hazard,
)

VT_TELEMETRY_PREFIXES = ("VT",)

# Lab-based H2S classes.  ``no_lab`` is a real third class, never merged into nonsour.
LAB_CLASSES = ("sour_lab", "nonsour_lab", "no_lab")
FIT_CLASSES = ("sour_lab", "nonsour_lab")  # no_lab reported separately, fitted only if events allow

# Contractor cells for the per-stratum fits and the collider comparison.
CONTRACTOR_CELLS = ("all", "slb", "brt", "slb_brt", "oth")

# A cell is fitted as its own stratum only with at least this many failure events;
# thinner cells are pooled into the class-level ("all") parent and marked.
MIN_STRATUM_FAILURES = 15

CIF_EVAL_TIMES = (90.0, 180.0, 365.0)


# --------------------------------------------------------------------------- #
# Lab-based H2S classifier (well-level, three-way)                              #
# --------------------------------------------------------------------------- #
def lab_h2s_class_map(lab_db_path: Path | None = None) -> pd.Series:
    """Well-level lab H2S class, keyed by normalized (upper-case) well ``code``.

    ``sour_lab`` if the well ever has a ``svb_h2s_indicator == '+'`` sample, else
    ``nonsour_lab`` (the well has samples but none positive).  Wells absent from the
    result have no lab samples → ``no_lab`` (assigned by the caller, never defaulted
    to nonsour).  Join key: ``lab_samples.well_key`` (lowercase ``vt_XXXX``) is
    case-normalized through :func:`crosswalk.norm_well` to the population's
    upper-case ``VT_XXXX`` codes.
    """
    con = sqlite3.connect(str(lab_db_path or resolve_lab_db_path()))
    try:
        lab = pd.read_sql("SELECT well_key, svb_h2s_indicator FROM lab_samples", con)
    finally:
        con.close()
    lab["code"] = lab["well_key"].map(crosswalk.norm_well)
    lab = lab.dropna(subset=["code"])
    # '+' is the only non-null value; absence of a '+' is treated as "not detected".
    return lab.groupby("code")["svb_h2s_indicator"].agg(
        lambda s: "sour_lab" if (s.astype(str) == "+").any() else "nonsour_lab"
    )


# --------------------------------------------------------------------------- #
# Population (single t_cal clock, lab class attached) — Vt-specific              #
# --------------------------------------------------------------------------- #
def build_population(
    as_of: date | pd.Timestamp | None = None,
    lab_db_path: Path | None = None,
) -> pd.DataFrame:
    """Vt population (all history), one row per run, on the single ``t_cal`` clock.

    Adds a well-level ``lab_class`` (sour_lab / nonsour_lab / no_lab) and keeps the
    Свод flag ``h2s_class`` for the reclassification cross-tab.  Columns mirror the
    Mc build: ``code``, ``install``, ``end``, ``t_cal``, ``cause_code``, ``cause``,
    ``is_open``, ``current_age`` (= t_cal for open runs), ``contractor_group``,
    ``well_key``.
    """
    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    pop = P.add_time_scales(P.build(asof, gtm_is_failure=False), asof)
    vt = pop[pop["field"] == "Vt"].copy()

    ended = vt["end"].notna() & (vt["end"] <= asof)
    vt["is_open"] = ~ended
    vt["cause_code"] = [
        classify_cause(r, u, e)
        for r, u, e in zip(vt["pull_reason"], vt["has_failed_unit"], ended)
    ]
    vt["cause"] = vt["cause_code"].map({CENSORED: "running", FAILURE: "failure", GTM: "gtm"})
    vt["current_age"] = vt["t_cal"].astype(float)
    vt["well_key"] = vt["code"]

    # Lab class is a well property, so mapping a well-level series onto every run of
    # the well makes all runs (open included) share one class BY CONSTRUCTION — the
    # fix for defect #4 (open rows on sour wells misfiled nonsour by the Свод flag).
    labmap = lab_h2s_class_map(lab_db_path)
    vt["lab_class"] = vt["code"].map(labmap).fillna("no_lab")

    # Single-clock invariant (the mixed-clock tte trap hit 138/138 Vt open runs).
    op = vt[vt["is_open"]]
    if len(op):
        cal_age = (asof - op["install"]).dt.days.clip(lower=P.DAY_ZERO_TTE).astype(float)
        mismatch = int((~np.isclose(op["t_cal"], cal_age)).sum())
        if mismatch:
            raise AssertionError(
                f"{mismatch} open Vt runs not on the calendar clock (t_cal != as_of-install) "
                "— refusing to fit on a mixed clock."
            )
    if not np.isfinite(vt["t_cal"]).all() or (vt["t_cal"] <= 0).any():
        raise AssertionError("non-finite or non-positive t_cal in the Vt population.")

    # Well-level inheritance: every run of a well shares one lab class (open too).
    per_well = vt.groupby("code")["lab_class"].nunique()
    if (per_well > 1).any():
        bad = per_well[per_well > 1].index.tolist()
        raise AssertionError(f"lab_class not well-constant for {bad} — inheritance broken.")

    return vt.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Diagnostics — lab coverage + reclassification cross-tab                        #
# --------------------------------------------------------------------------- #
def lab_coverage(pop: pd.DataFrame) -> pd.DataFrame:
    """Per lab class: wells, runs, failures, ГТМ, open runs, and event rate.

    ``no_lab`` sits next to the classified groups so the reader can judge the
    plausibly-MNAR missingness (labs sample where trouble is suspected).
    """
    rows = []
    for cls in LAB_CLASSES:
        g = pop[pop["lab_class"] == cls]
        n_runs = len(g)
        n_fail = int((g["cause_code"] == FAILURE).sum())
        n_gtm = int((g["cause_code"] == GTM).sum())
        n_open = int(g["is_open"].sum())
        rows.append(
            {
                "lab_class": cls,
                "wells": int(g["code"].nunique()),
                "runs": n_runs,
                "failures": n_fail,
                "gtm": n_gtm,
                "open_runs": n_open,
                "closed_runs": n_runs - n_open,
                "failure_rate_per_closed": round(n_fail / (n_runs - n_open), 3) if (n_runs - n_open) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def reclassification_crosstab(pop: pd.DataFrame) -> pd.DataFrame:
    """Lab class × Свод «Кислый/Некислый» flag class — how many runs move where.

    The direct successor to defect #4: the flag misfiled runs (esp. open ones); the
    lab class re-files them.  ``open_runs`` counts how many of each cell are running
    pumps (the ones the flag inheritance bug got wrong).
    """
    rows = []
    for cls in LAB_CLASSES:
        for flag in ("sour", "nonsour"):
            g = pop[(pop["lab_class"] == cls) & (pop["h2s_class"] == flag)]
            rows.append(
                {
                    "lab_class": cls,
                    "svod_flag_class": flag,
                    "runs": len(g),
                    "open_runs": int(g["is_open"].sum()),
                    "failures": int((g["cause_code"] == FAILURE).sum()),
                    "moved": cls.replace("_lab", "") != flag and cls != "no_lab",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stratum helpers                                                               #
# --------------------------------------------------------------------------- #
def _contractor_mask(sub: pd.DataFrame, cell: str) -> pd.Series:
    if cell == "all":
        return pd.Series(True, index=sub.index)
    if cell == "slb_brt":
        return sub["contractor_group"].isin(["slb", "brt"])
    return sub["contractor_group"] == cell


def _stratum_frame(pop: pd.DataFrame, cls: str, cell: str = "all") -> pd.DataFrame:
    sub = pop[pop["lab_class"] == cls]
    return sub[_contractor_mask(sub, cell)].copy()


# --------------------------------------------------------------------------- #
# Task A — lab-based strata and cause-specific fits                              #
# --------------------------------------------------------------------------- #
def cause_specific_params(pop: pd.DataFrame, *, n_boot: int = 300) -> pd.DataFrame:
    """Cause-specific Weibull (λ_fail with ГТМ censored, λ_gtm with failure censored)
    for every {sour_lab, nonsour_lab, no_lab} × {all, slb, brt, slb+brt, oth} cell.

    Cells with < ``MIN_STRATUM_FAILURES`` failure events are NOT fitted as their own
    stratum: the row is emitted with ``fitted=False`` / ``pooled_into=<class>_all``
    so a thin cell is visible but not presented as an independent Weibull.
    """
    rows = []
    for cls in LAB_CLASSES:
        for cell in CONTRACTOR_CELLS:
            g = _stratum_frame(pop, cls, cell)
            n_fail = int((g["cause_code"] == FAILURE).sum())
            n_gtm = int((g["cause_code"] == GTM).sum())
            base = {
                "lab_class": cls,
                "contractor_cell": cell,
                "n_runs": len(g),
                "n_wells": int(g["code"].nunique()),
                "n_failures": n_fail,
                "n_gtm": n_gtm,
                "n_open": int(g["is_open"].sum()),
            }
            if n_fail < MIN_STRATUM_FAILURES:
                rows.append({**base, "cause": "failure", "fitted": False,
                             "pooled_into": f"{cls}_all", "beta": np.nan, "eta": np.nan,
                             "beta_ci_lo": np.nan, "beta_ci_hi": np.nan,
                             "rmst_0_730": np.nan, "mrl_0": np.nan, "n_boot_ok": 0})
                if n_gtm >= MIN_STRATUM_FAILURES:
                    fg = fit_cause_specific(g, GTM, n_boot=n_boot)
                    rows.append(_fit_row(base, fg, fitted=True, pooled_into=""))
                else:
                    rows.append({**base, "cause": "gtm", "fitted": False,
                                 "pooled_into": f"{cls}_all", "beta": np.nan, "eta": np.nan,
                                 "beta_ci_lo": np.nan, "beta_ci_hi": np.nan,
                                 "rmst_0_730": np.nan, "mrl_0": np.nan, "n_boot_ok": 0})
                continue
            ff = fit_cause_specific(g, FAILURE, n_boot=n_boot)
            rows.append(_fit_row(base, ff, fitted=True, pooled_into=""))
            if n_gtm >= 3:
                fg = fit_cause_specific(g, GTM, n_boot=n_boot if n_gtm >= MIN_STRATUM_FAILURES else 0)
                rows.append(_fit_row(base, fg, fitted=n_gtm >= MIN_STRATUM_FAILURES,
                                     pooled_into="" if n_gtm >= MIN_STRATUM_FAILURES else f"{cls}_all"))
    return pd.DataFrame(rows)


def _fit_row(base: dict, f: core.CauseFit, *, fitted: bool, pooled_into: str) -> dict:
    return {
        **base,
        "cause": f.cause,
        "fitted": fitted,
        "pooled_into": pooled_into,
        "beta": round(f.beta, 4),
        "eta": round(f.eta, 1),
        "beta_ci_lo": round(f.beta_lo, 3),
        "beta_ci_hi": round(f.beta_hi, 3),
        "rmst_0_730": round(f.rmst_0_730, 1),
        "mrl_0": round(f.mrl_0, 1),
        "n_boot_ok": f.n_boot_ok,
    }


def hazard_bands_by_class(pop: pd.DataFrame) -> pd.DataFrame:
    """Piecewise band hazards for λ_fail and λ_gtm per fit class (the shape evidence)."""
    frames = []
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        for code in (FAILURE, GTM):
            hb = piecewise_hazard(g, code)
            hb.insert(0, "lab_class", cls)
            frames.append(hb)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# k1/k2 mixture-Weibull survival on the lab split (user request 2026-07-23)      #
# --------------------------------------------------------------------------- #
# Recalculate the shipped-registry k1/k2 survival (mixture-Weibull EM + AIC
# k1-vs-k2 selection, `phase2_mixture`/`vba_bundle` machinery) but with the new
# LAB H2S split, on the single `t_cal` clock, event = genuine FAILURE (ГТМ and
# running censored — user decision), no_lab OMITTED as non-informative.  Two
# populations: `slb+brt` (contractor axis, oth dropped) and `pulled` (the whole
# fleet subject to ГТМ pulls, all contractors).  Well-level sour/nonsour
# inheritance already boosts N (a well with any '+' sample is sour on every run).
#
# Init models and EM settings mirror phase2_mixture exactly so the fit is the same
# estimator as the registry — only the population/clock/split differ.
_NUM_STARTS = 12
_MIN_ETA_RATIO = 2.0
_DEFAULT_INIT = TwoComponentLatentWeibullModel(
    weight_1=0.5,
    component_1=WeibullParameters(beta=0.75, eta=60.0, label="C1_early"),
    component_2=WeibullParameters(beta=2.00, eta=300.0, label="C2_wearout"),
)
_SOUR_INIT = TwoComponentLatentWeibullModel(
    weight_1=0.5,
    component_1=WeibullParameters(beta=0.90, eta=15.0, label="C1_early"),
    component_2=WeibullParameters(beta=1.80, eta=120.0, label="C2_wearout"),
)

# The two populations to recalculate, keyed by stratum-name suffix.
K1K2_STRATA = ("slb+brt", "pulled")


def _k1k2_population(pop: pd.DataFrame, cls: str, stratum: str) -> pd.DataFrame:
    """Population for a lab-split k1/k2 stratum.

    ``slb+brt`` — slb+brt contractors pooled (oth dropped as non-informative);
    ``pulled``  — the whole fleet (all contractors) subject to ГТМ pulls.
    Both restricted to one lab H2S class (no_lab already excluded by the caller).
    """
    g = pop[pop["lab_class"] == cls]
    if stratum == "slb+brt":
        g = g[g["contractor_group"].isin(["slb", "brt"])]
    return g.copy()


def _mixture_survival(model: TwoComponentLatentWeibullModel, t: np.ndarray) -> np.ndarray:
    """S(t) for the two-component latent mixture (w1·S1 + w2·S2)."""
    c1, c2 = model.component_1, model.component_2
    s1 = np.exp(-np.power(np.clip(t, 0.0, None) / c1.eta, c1.beta))
    s2 = np.exp(-np.power(np.clip(t, 0.0, None) / c2.eta, c2.beta))
    return model.weight_1 * s1 + model.weight_2 * s2


def _rmst_mrl(model: TwoComponentLatentWeibullModel) -> tuple[float, float]:
    """RMST(0,730) and MRL(0)=∫₀^∞S from the mixture survival."""
    u1 = np.linspace(0.0, 730.0, 4000)
    rmst = float(np.trapezoid(_mixture_survival(model, u1), u1))
    u2 = np.linspace(0.0, 20000.0, 40000)
    mrl = float(np.trapezoid(_mixture_survival(model, u2), u2))
    return rmst, mrl


def _single_b50_ci(g: pd.DataFrame, *, n_boot: int, seed: int) -> tuple[float, float]:
    """Well-cluster bootstrap CI for the single-Weibull B50 (registry convention)."""
    wells = g["well_key"].dropna().unique()
    if len(wells) == 0:
        return (float("nan"), float("nan"))
    by = {w: g[g["well_key"] == w] for w in wells}
    rng = np.random.default_rng(seed)
    b50s: list[float] = []
    for _ in range(n_boot):
        boot = pd.concat([by[w] for w in rng.choice(wells, len(wells), replace=True)], ignore_index=True)
        ev = (boot["cause_code"].to_numpy(int) == FAILURE).astype(int)
        if ev.sum() < 3:
            continue
        try:
            r = fit_basic_weibull(boot["t_cal"].to_numpy(float), ev)
            b50s.append(float(r["eta"]) * (-np.log(0.5)) ** (1.0 / float(r["beta"])))
        except Exception:
            continue
    if len(b50s) < max(10, int(0.1 * n_boot)):
        return (float("nan"), float("nan"))
    return (round(float(np.percentile(b50s, 2.5)), 1), round(float(np.percentile(b50s, 97.5)), 1))


def k1k2_survival(pop: pd.DataFrame, *, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Registry-style k1/k2 mixture-Weibull survival on the lab split.

    One row per (stratum ∈ {slb+brt, pulled}) × (lab_class ∈ {sour_lab,
    nonsour_lab}) × (k ∈ {k1, k2}).  ``selected`` marks the AIC-chosen model
    (``model_kind`` in {k1_aic, k1_degenerate, k2}, decided exactly as
    ``vba_bundle._registry_row``: degenerate→k1_degenerate; mixture does not win
    AIC→k1_aic; else k2).  Reports w1/β1/η1/β2/η2, B20/B50/B80, RMST(0,730), MRL(0),
    ΔAIC (k1−k2, >0 ⇒ mixture) and a well-cluster single-Weibull B50 CI.
    """
    rows = []
    for stratum in K1K2_STRATA:
        for cls in FIT_CLASSES:
            g = _k1k2_population(pop, cls, stratum)
            t = g["t_cal"].to_numpy(float)
            ev = (g["cause_code"].to_numpy(int) == FAILURE).astype(int)
            n_fail = int(ev.sum())
            key = f"Vt_{cls}_{stratum}"
            if n_fail < 3:
                continue

            # k1 — single Weibull
            k1 = fit_basic_weibull(t, ev)
            b1_k1, e1_k1 = float(k1["beta"]), float(k1["eta"])
            aic_k1 = float(k1["aic"])
            m_k1 = TwoComponentLatentWeibullModel(
                weight_1=0.0,
                component_1=WeibullParameters(beta=b1_k1, eta=e1_k1, label="C1"),
                component_2=WeibullParameters(beta=b1_k1, eta=e1_k1, label="C2"),
            )

            # k2 — two-component latent mixture (same EM as the registry)
            init = _SOUR_INIT if "sour" in cls else _DEFAULT_INIT
            res = fit_latent_weibull_em(
                t, ev, initial_model=init, num_starts=_NUM_STARTS, min_eta_ratio=_MIN_ETA_RATIO,
            )
            m_k2 = res.model
            nll_k2 = float(res.objective_value)
            aic_k2 = 2.0 * 5 + 2.0 * nll_k2  # 5 free params (independent fit)
            delta_aic = aic_k1 - aic_k2       # >0 ⇒ mixture preferred
            degenerate = "DEGENERATE" in res.message
            mixture_wins = bool(delta_aic > 0) if np.isfinite(delta_aic) else False
            model_kind = "k1_degenerate" if degenerate else ("k2" if mixture_wins else "k1_aic")
            b50_lo, b50_hi = _single_b50_ci(g, n_boot=n_boot, seed=seed)

            for k, model, aic, sel in (
                ("k1", m_k1, aic_k1, model_kind != "k2"),
                ("k2", m_k2, aic_k2, model_kind == "k2"),
            ):
                rmst, mrl = _rmst_mrl(model)
                rows.append({
                    "stratum": key, "lab_class": cls, "population": stratum, "k": k,
                    "selected": sel, "model_kind": model_kind,
                    "n_runs": len(g), "n_wells": int(g["code"].nunique()),
                    "n_failures": n_fail, "n_gtm": int((g["cause_code"] == GTM).sum()),
                    "n_open": int(g["is_open"].sum()),
                    "w1": round(float(model.weight_1), 4),
                    "beta1": round(float(model.component_1.beta), 4),
                    "eta1": round(float(model.component_1.eta), 1),
                    "beta2": round(float(model.component_2.beta), 4),
                    "eta2": round(float(model.component_2.eta), 1),
                    "b20": round(float(latent_life_quantile(0.20, model)), 1),
                    "b50": round(float(latent_life_quantile(0.50, model)), 1),
                    "b80": round(float(latent_life_quantile(0.80, model)), 1),
                    "rmst_0_730": round(rmst, 1), "mrl_0": round(mrl, 1),
                    "aic": round(aic, 1),
                    "delta_aic_k1_minus_k2": round(delta_aic, 1) if np.isfinite(delta_aic) else np.nan,
                    "b50_ci_lo": b50_lo if k == "k1" else np.nan,
                    "b50_ci_hi": b50_hi if k == "k1" else np.nan,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Task B — the collider resolution (slb vs brt)                                 #
# --------------------------------------------------------------------------- #
def collider_slb_brt(pop: pd.DataFrame, *, n_boot: int = 400) -> pd.DataFrame:
    """slb vs brt compared three ways within each H2S class:

    (i)   failure-only naive 1−KM (reproduces the old "equal" result),
    (ii)  cause-specific λ_fail (ГТМ censored),
    (iii) AJ CIF of failure with ГТМ competing, plus the bootstrapped CIF
          difference at 90/180/365 d (Gray-style, well-cluster resampled).

    One tidy long frame with a ``panel`` discriminator.
    """
    rows = []
    for cls in FIT_CLASSES:
        sub = _stratum_frame(pop, cls, "slb_brt")
        for contractor in ("slb", "brt"):
            g = sub[sub["contractor_group"] == contractor]
            t = g["t_cal"].to_numpy(float)
            ec = g["cause_code"].to_numpy(int)
            # (i) naive 1-KM of failure (ГТМ + running censored)
            naive = CIF.all_cause_km(t, (ec == FAILURE).astype(int))
            for tt in CIF_EVAL_TIMES:
                rows.append({"h2s_class": cls, "panel": "naive_1_minus_km_failure",
                             "contractor": contractor, "time_d": tt,
                             "value": round(float(naive.at(tt)), 4),
                             "n_failures": int((ec == FAILURE).sum())})
            # (ii) cause-specific λ_fail
            if int((ec == FAILURE).sum()) >= 3:
                ff = fit_cause_specific(g, FAILURE, n_boot=n_boot // 2)
                rows.append({"h2s_class": cls, "panel": "cause_specific_lambda_fail",
                             "contractor": contractor, "beta": round(ff.beta, 4),
                             "eta": round(ff.eta, 1), "rmst_0_730": round(ff.rmst_0_730, 1),
                             "mrl_0": round(ff.mrl_0, 1), "n_failures": ff.n_events})
        # (iii) AJ CIF failure with ГТМ competing + bootstrap difference (slb - brt)
        if sub["contractor_group"].nunique() == 2 and len(sub):
            diff = CIF.bootstrap_cif_difference(
                sub, group_col="contractor_group", group_a="slb", group_b="brt",
                duration_col="t_cal", event_code_col="cause_code", cause_code=FAILURE,
                at_times=CIF_EVAL_TIMES, cluster_col="well_key", n_boot=n_boot,
            )
            for _, r in diff.iterrows():
                rows.append({"h2s_class": cls, "panel": "aj_cif_failure_diff_slb_minus_brt",
                             "time_d": float(r["time"]), "cif_slb": r["cif_slb"],
                             "cif_brt": r["cif_brt"], "diff_point": r["diff_point"],
                             "diff_ci_lo": r["diff_ci_lo"], "diff_ci_hi": r["diff_ci_hi"],
                             "boot_p_two_sided": r["boot_p_two_sided"]})
    return pd.DataFrame(rows)


def collider_support_balance(pop: pd.DataFrame) -> pd.DataFrame:
    """Balance check: is a slb-vs-brt gap a contractor effect or a rate/age proxy?

    Contractor≈rate was a *Mc* finding; on Vt we check mean Qliq, GLF-free calendar
    age balance between the two arms per class before attributing any gap to the
    contractor.
    """
    qliq = core.well_mean_qliq(VT_TELEMETRY_PREFIXES)
    rows = []
    for cls in FIT_CLASSES:
        sub = _stratum_frame(pop, cls, "slb_brt")
        sub = sub.assign(mean_qliq=sub["code"].str.lower().map(qliq))
        for contractor in ("slb", "brt"):
            g = sub[sub["contractor_group"] == contractor]
            rows.append(
                {
                    "h2s_class": cls,
                    "contractor": contractor,
                    "n_runs": len(g),
                    "median_age_d": round(float(g["t_cal"].median()), 1) if len(g) else np.nan,
                    "median_mean_qliq": round(float(g["mean_qliq"].median()), 1) if g["mean_qliq"].notna().any() else np.nan,
                    "qliq_coverage": round(float(g["mean_qliq"].notna().mean()), 2) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Task C — CIF vs naive KM per class                                            #
# --------------------------------------------------------------------------- #
def cif_vs_km_by_class(pop: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        c = cif_vs_km(g)
        c.insert(0, "lab_class", cls)
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Task D — informative-censoring / preventive-ГТМ test + latent bound            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatentBound:
    lab_class: str
    lambda_fail_rmst_baseline: float
    lambda_fail_rmst_upper: float
    lambda_fail_beta_baseline: float
    lambda_fail_beta_upper: float
    n_gtm_recoded: int
    n_gtm_total: int
    note: str


def informative_censoring_by_class(pop: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        l1 = core.informative_censoring_level1(g, VT_TELEMETRY_PREFIXES)
        l1.insert(0, "lab_class", cls)
        frames.append(l1)
    return pd.concat(frames, ignore_index=True)


def telemetry_level2_by_class(pop: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Level-2 precursor summary per class + the per-run frames (for the bound)."""
    summaries, runs_by_class = [], {}
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        summ, runs = core.informative_censoring_level2(g, VT_TELEMETRY_PREFIXES, return_runs=True)
        runs_by_class[cls] = runs
        if len(summ):
            summ.insert(0, "lab_class", cls)
            summaries.append(summ)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return summary, runs_by_class


def _precursor_positive_gtm(runs: pd.DataFrame) -> pd.Index:
    """ГТМ runs that look like a failure was coming: final-window frequency
    instability at/above the failure median, or подача declining at/below it.

    Used to build the latent upper bound (recode these ГТМ as failures).  Returns
    the run index labels within ``runs``.
    """
    if runs.empty or "cause" not in runs.columns:
        return pd.Index([])
    fail = runs[runs["cause"] == "failure"]
    gtm = runs[runs["cause"] == "gtm"]
    if fail.empty or gtm.empty:
        return pd.Index([])
    fs_med = float(fail["freq_std_final"].median()) if fail["freq_std_final"].notna().any() else np.inf
    sl_med = float(fail["qliq_slope_final"].median()) if fail["qliq_slope_final"].notna().any() else -np.inf
    unstable = gtm["freq_std_final"] >= fs_med
    declining = gtm["qliq_slope_final"] <= sl_med
    return gtm[unstable.fillna(False) | declining.fillna(False)].index


def latent_bounds(pop: pd.DataFrame, runs_by_class: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Upper bound on the latent no-ГТМ failure hazard per class.

    Recode the precursor-positive ГТМ runs as failures and re-fit λ_fail: if ГТМ is
    preventive, this hazard is higher (shorter RMST) than the observed λ_fail, and
    the true latent hazard lies between them.  The decomposition's latent column is
    therefore a **lower bound** on failures; this quantifies the gap.
    """
    rows = []
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all").copy()
        base = fit_cause_specific(g, FAILURE, n_boot=0)
        runs = runs_by_class.get(cls, pd.DataFrame())
        pos = _precursor_positive_gtm(runs)
        n_gtm_total = int((g["cause_code"] == GTM).sum())
        if len(pos):
            recode_codes = set(zip(runs.loc[pos, "code"], runs.loc[pos, "end"]))
            key = list(zip(g["code"], g["end"]))
            mask = np.array([k in recode_codes for k in key])
            up = g.copy()
            up.loc[mask, "cause_code"] = FAILURE
            up_fit = fit_cause_specific(up, FAILURE, n_boot=0)
            n_recoded = int(mask.sum())
        else:
            up_fit = base
            n_recoded = 0
        rows.append(
            {
                "lab_class": cls,
                "lambda_fail_rmst_baseline": round(base.rmst_0_730, 1),
                "lambda_fail_rmst_upper_bound": round(up_fit.rmst_0_730, 1),
                "lambda_fail_beta_baseline": round(base.beta, 3),
                "lambda_fail_beta_upper_bound": round(up_fit.beta, 3),
                "n_gtm_recoded_as_failure": n_recoded,
                "n_gtm_total": n_gtm_total,
                "note": "latent no-ГТМ column is a LOWER bound; upper recodes precursor-positive ГТМ as failures",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Task E — decomposition per H2S class                                          #
# --------------------------------------------------------------------------- #
def decomposition_by_class(
    pop: pd.DataFrame, *, as_of: pd.Timestamp, horizon_months: int = 12
) -> pd.DataFrame:
    frames = []
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        haz = band_hazards_from_pop(g)
        obs = observed_window_prediction(g, haz)
        dec = combined_decomposition(g, haz, obs, as_of=as_of, horizon_months=horizon_months)
        dec.insert(0, "lab_class", cls)
        frames.append(dec)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Task F — reconcile                                                            #
# --------------------------------------------------------------------------- #
def reconciliation(pop: pd.DataFrame, lat: pd.DataFrame) -> pd.DataFrame:
    """Fitted event counts + the sour RMST headline, lab class vs Свод flag.

    The registry Vt strata (esp_models.csv) are FLAG-based; ours are LAB-based, so
    counts differ by construction — the reclassification cross-tab is the
    explanation.  Also re-measures the headline the flag produced: sour RMST(0,730)
    under the lab class vs under the flag, and compares the sour λ_fail shape to the
    shipped 2026-07-20 sour refit (β≈1 memoryless, Vt_sour_Pooled in the registry).
    """
    rows = []
    # registry Vt failure counts (flag-based)
    try:
        reg = pd.read_csv(C.model_registry_path(), encoding="utf-8-sig")
        vt = reg[reg["field"] == "Vt"]
        # The *_Pooled rows are contractor aggregates of the brt/oth/slb rows — summing
        # them in would double-count, so restrict to the per-contractor strata.
        vt_cells = vt[vt["contractor_group"] != "Pooled"]
        reg_sour = int(vt_cells[vt_cells["h2s_class"] == "sour"]["n_failures"].sum())
        reg_nonsour = int(vt_cells[vt_cells["h2s_class"] == "nonsour"]["n_failures"].sum())
        sour_pooled = vt[vt["stratum"] == "Vt_sour_Pooled"]
        reg_sour_beta = float(sour_pooled["beta1"].iloc[0]) if len(sour_pooled) else np.nan
        reg_sour_eta = float(sour_pooled["eta1"].iloc[0]) if len(sour_pooled) else np.nan
    except Exception:
        reg_sour = reg_nonsour = -1
        reg_sour_beta = reg_sour_eta = np.nan

    lab_sour_fail = int((pop[pop["lab_class"] == "sour_lab"]["cause_code"] == FAILURE).sum())
    lab_nonsour_fail = int((pop[pop["lab_class"] == "nonsour_lab"]["cause_code"] == FAILURE).sum())
    # Apples-to-apples registry check: our population's FLAG-sour failures should track
    # the registry's flag strata (both flag-based) — confirms no population defect
    # before we attribute anything to the lab reclassification.
    our_flag_sour = int((pop[pop["h2s_class"] == "sour"]["cause_code"] == FAILURE).sum())
    our_flag_nonsour = int((pop[pop["h2s_class"] == "nonsour"]["cause_code"] == FAILURE).sum())
    rows.append({"item": "failure_counts_flag_check", "lab_sour": our_flag_sour, "lab_nonsour": our_flag_nonsour,
                 "flag_registry_sour": reg_sour, "flag_registry_nonsour": reg_nonsour,
                 "note": "our population's FLAG-sour/nonsour failures vs registry flag strata (both flag-based) — should track; confirms no population defect"})
    rows.append({"item": "failure_counts_lab", "lab_sour": lab_sour_fail, "lab_nonsour": lab_nonsour_fail,
                 "flag_registry_sour": reg_sour, "flag_registry_nonsour": reg_nonsour,
                 "note": "LAB-based failure counts (this analysis) next to the FLAG registry — differ by construction, see reclassification cross-tab"})

    # sour RMST headline: lab class vs Свод flag class
    lab_sour = pop[pop["lab_class"] == "sour_lab"]
    flag_sour = pop[pop["h2s_class"] == "sour"]
    lab_fit = fit_cause_specific(lab_sour, FAILURE, n_boot=0)
    flag_fit = fit_cause_specific(flag_sour, FAILURE, n_boot=0)
    rows.append({"item": "sour_rmst0_730_headline",
                 "lab_sour": round(lab_fit.rmst_0_730, 1), "lab_nonsour": np.nan,
                 "flag_registry_sour": round(flag_fit.rmst_0_730, 1), "flag_registry_nonsour": np.nan,
                 "note": f"λ_fail RMST(0,730): lab_sour β={lab_fit.beta:.2f} vs flag_sour β={flag_fit.beta:.2f}; "
                         "does the flag-era +30% sour story survive the lab split?"})

    # shape vs shipped 2026-07-20 sour refit (β≈1 memoryless)
    reg_sour_rmst = core._rmst(reg_sour_beta, reg_sour_eta) if np.isfinite(reg_sour_beta) else np.nan
    rows.append({"item": "sour_shape_vs_shipped_refit",
                 "lab_sour": round(lab_fit.beta, 3), "lab_nonsour": np.nan,
                 "flag_registry_sour": round(reg_sour_beta, 3), "flag_registry_nonsour": np.nan,
                 "note": f"lab_sour λ_fail β={lab_fit.beta:.2f} vs shipped Vt_sour_Pooled β={reg_sour_beta:.2f} "
                         f"(RMST {reg_sour_rmst:.0f}); 2026-07-20 refit read β≈1 memoryless"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures (Russian labels)                                                      #
# --------------------------------------------------------------------------- #
def _cif_curves(g: pd.DataFrame, tmax: float):
    t = g["t_cal"].to_numpy(float)
    ec = g["cause_code"].to_numpy(int)
    grid = np.linspace(0.0, tmax, 400)
    cf = CIF.aalen_johansen_cif(t, ec, FAILURE)
    cg = CIF.aalen_johansen_cif(t, ec, GTM)
    naive = CIF.all_cause_km(t, (ec == FAILURE).astype(int))
    return grid, cf.at(grid), cg.at(grid), naive.at(grid)


def figure_cif_stack_by_class(pop: pd.DataFrame, path: Path, tmax: float = 730.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    titles = {"sour_lab": "Кислый (лаб.)", "nonsour_lab": "Некислый (лаб.)"}
    for ax, cls in zip(axes, FIT_CLASSES):
        g = _stratum_frame(pop, cls, "all")
        grid, cif_f, cif_g, naive = _cif_curves(g, tmax)
        running = 1.0 - cif_f - cif_g
        ax.stackplot(grid, cif_f, cif_g, running,
                     labels=["Отказ (AJ CIF)", "ГТМ (AJ CIF)", "В работе"],
                     colors=["#c0392b", "#e67e22", "#d9e2ec"], alpha=0.9)
        ax.plot(grid, naive, "k--", lw=1.8, label="Наивная 1−KM (отказ)")
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Наработка (кал. дни, t_cal)")
        ax.set_title(f"{titles[cls]}: n={len(g)}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Кумулятивная доля")
    fig.suptitle("Вахитовское: конкурирующие риски отказ/ГТМ по H2S-классу (лаб.)\n"
                 "наивная 1−KM переоценивает частоту отказов (ГТМ как свободное цензурирование)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_hazard_shapes_by_class(pop: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = [(lo + (hi if np.isfinite(hi) else lo + 365)) / 2 for lo, hi in HAZARD_BANDS]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    styles = {"sour_lab": ("#c0392b", "-"), "nonsour_lab": ("#2e86c1", "--")}
    for cls in FIT_CLASSES:
        g = _stratum_frame(pop, cls, "all")
        hf = piecewise_hazard(g, FAILURE)
        color, ls = styles[cls]
        ax.step(centers, hf["hazard_per_1000d"], where="mid", color=color, ls=ls, lw=2,
                label=f"λ_отказ {cls} (n_отк={int(hf['n_events'].sum())})")
    ax.set_xlabel("Возраст насоса (кал. дни)")
    ax.set_ylabel("Опасность отказа на 1000 дней")
    ax.set_title("Вахитовское: форма λ_отказ по H2S-классу (лаб.)\n"
                 "кислый — инфант-тяжёлый (β<1), некислый площе")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_collider(collider: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = collider[collider["panel"] == "aj_cif_failure_diff_slb_minus_brt"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {"sour_lab": "#c0392b", "nonsour_lab": "#2e86c1"}
    for cls in FIT_CLASSES:
        d = diff[diff["h2s_class"] == cls].sort_values("time_d")
        if d.empty:
            continue
        ax.errorbar(d["time_d"], d["diff_point"],
                    yerr=[d["diff_point"] - d["diff_ci_lo"], d["diff_ci_hi"] - d["diff_point"]],
                    marker="o", capsize=4, color=colors[cls], label=f"{cls} (SLB−BRT)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Время, дни")
    ax.set_ylabel("Разница CIF отказа (SLB − BRT)")
    ax.set_title("Вахитовское: коллайдер slb vs brt — разница CIF отказа с ГТМ как конкур. риском\n"
                 "0 в ДИ ⇒ равность выдерживает; иначе — была порождена ГТМ-цензурированием")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_decomposition_by_class(decomp: pd.DataFrame, lat: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    titles = {"sour_lab": "Кислый (лаб.)", "nonsour_lab": "Некислый (лаб.)"}
    for ax, cls in zip(axes, FIT_CLASSES):
        d = decomp[(decomp["lab_class"] == cls) & (decomp["segment"] == "forecast")]
        x = np.arange(len(d))
        ax.bar(x, d["m1_obs_failures"], color="#c0392b", alpha=0.9, label="Прогноз отказов (кол.1)")
        ax.plot(x, d["m4_latent_nogtm_failures"], "o-", color="#117864", lw=1.8,
                label="Латентный без ГТМ (кол.4, нижняя оценка)")
        ax.set_xticks(x)
        ax.set_xticklabels(d["month"], rotation=90, fontsize=7)
        ax.set_title(f"{titles[cls]}: 12-мес отказы={d['m1_obs_failures'].sum():.1f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Отказы в месяц")
    fig.suptitle("Вахитовское: помесячная декомпозиция по H2S-классу — латентная колонка = НИЖНЯЯ оценка\n"
                 "(ГТМ информативно ⇒ истинная λ без ГТМ выше наблюдаемой)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                  #
# --------------------------------------------------------------------------- #
def run_analysis(
    as_of: date | pd.Timestamp | None = None,
    *,
    horizon_months: int = 12,
    n_boot: int = 300,
    write_figures: bool = True,
) -> dict:
    """End-to-end Vt competing-risks: tasks A–F, tables and Russian figures."""
    from analysis.paths import results_dir

    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    out = results_dir("production_risk_vt_competing_risks")
    tables, figures = out / "tables", out / "figures"

    pop = build_population(asof)

    # Diagnostics first (acceptance-critical)
    cov = lab_coverage(pop)
    xtab = reclassification_crosstab(pop)
    cov.to_csv(tables / "h2s_lab_coverage.csv", index=False, encoding="utf-8-sig")
    xtab.to_csv(tables / "h2s_lab_reclassification_crosstab.csv", index=False, encoding="utf-8-sig")

    # A — cause-specific fits + band shapes
    params = cause_specific_params(pop, n_boot=n_boot)
    bands = hazard_bands_by_class(pop)
    params.to_csv(tables / "cause_specific_params.csv", index=False, encoding="utf-8-sig")
    bands.to_csv(tables / "cause_specific_hazard_bands.csv", index=False, encoding="utf-8-sig")

    # k1/k2 mixture-Weibull survival on the lab split (Vt_slb+brt & Vt_pulled)
    k1k2 = k1k2_survival(pop, n_boot=max(200, n_boot))
    k1k2.to_csv(tables / "k1k2_survival_lab_split.csv", index=False, encoding="utf-8-sig")

    # B — collider
    collider = collider_slb_brt(pop, n_boot=max(200, n_boot))
    balance = collider_support_balance(pop)
    collider.to_csv(tables / "collider_slb_brt.csv", index=False, encoding="utf-8-sig")
    balance.to_csv(tables / "collider_support_balance.csv", index=False, encoding="utf-8-sig")

    # C — CIF vs naive KM
    cifkm = cif_vs_km_by_class(pop)
    cifkm.to_csv(tables / "cif_vs_km.csv", index=False, encoding="utf-8-sig")

    # D — informative censoring + latent bound
    l1 = informative_censoring_by_class(pop)
    l2, runs_by_class = telemetry_level2_by_class(pop)
    lat = latent_bounds(pop, runs_by_class)
    l1.to_csv(tables / "informative_censoring.csv", index=False, encoding="utf-8-sig")
    if len(l2):
        l2.to_csv(tables / "informative_censoring_telemetry_level2.csv", index=False, encoding="utf-8-sig")
    lat.to_csv(tables / "latent_bounds.csv", index=False, encoding="utf-8-sig")

    # E — decomposition per class
    decomp = decomposition_by_class(pop, as_of=asof, horizon_months=horizon_months)
    decomp.to_csv(tables / "monthly_decomposition.csv", index=False, encoding="utf-8-sig")

    # F — reconcile
    recon = reconciliation(pop, lat)
    recon.to_csv(tables / "reconciliation.csv", index=False, encoding="utf-8-sig")

    if write_figures:
        figure_cif_stack_by_class(pop, figures / "cif_stack_vs_naive_km.png")
        figure_hazard_shapes_by_class(pop, figures / "cause_specific_hazard_shapes.png")
        figure_collider(collider, figures / "collider_slb_brt_cif.png")
        figure_decomposition_by_class(decomp, lat, figures / "monthly_decomposition.png")

    return {
        "out": out, "pop": pop, "coverage": cov, "crosstab": xtab, "params": params,
        "bands": bands, "k1k2": k1k2, "collider": collider, "balance": balance, "cif": cifkm,
        "l1": l1, "l2": l2, "latent": lat, "decomp": decomp, "reconciliation": recon,
    }


__all__ = [
    "run_analysis",
    "build_population",
    "lab_h2s_class_map",
    "lab_coverage",
    "reclassification_crosstab",
    "cause_specific_params",
    "hazard_bands_by_class",
    "k1k2_survival",
    "collider_slb_brt",
    "collider_support_balance",
    "cif_vs_km_by_class",
    "informative_censoring_by_class",
    "telemetry_level2_by_class",
    "latent_bounds",
    "decomposition_by_class",
    "reconciliation",
    "LAB_CLASSES",
    "FIT_CLASSES",
]
