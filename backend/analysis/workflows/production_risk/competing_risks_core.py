"""Field-agnostic competing-risks engine (failure vs ГТМ) shared by fields.

Extracted verbatim from ``mc_competing_risks`` so the Mc accounting and the Vt
accounting run the SAME estimators — cause-specific Weibull with well-cluster
bootstrap, piecewise band hazards, Aalen–Johansen CIF vs naive 1−KM, the two-level
informative-censoring test, and the deterministic daily renewal engine with the
five-column decomposition.  Nothing here references a specific field; the field
enters only through the population a caller passes in and, for the telemetry
helpers, through an explicit list of normalized-well prefixes.

Mc behaviour is unchanged: ``mc_competing_risks`` imports and re-exports these and
its telemetry wrappers call the generic helpers with the ``("MC", "MR")`` prefixes,
which build the identical SQL the Mc module used before the extraction.

One clock for everything: the caller's population must be timed on ``t_cal``
(calendar: ``pull-install`` or ``as_of-install`` while running) for events, pulls
AND open runs — the single clock that is always available and never imputed.  The
mixed-clock ``tte`` trap (events on Наработка, censorings on calendar) is what this
convention avoids.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival import cif as CIF
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.paths import resolve_telemetry_db_path
from analysis.workflows.production_risk.esp_population import (
    GENUINE_FAILURE_REASONS,
    WORKOVER_REASONS,
    _norm,
)

# Competing-risks event codes (the AJ CIF convention in models.survival.cif).
CENSORED = 0   # still running (open) or administrative censor
FAILURE = 1    # genuine ESP failure
GTM = 2        # planned workover / ГТМ / ППР pull

CAUSE_ORDER = ("failure", "gtm")

# Age bands (days) for the piecewise-constant cause-specific hazards used by the
# renewal engine.  Same partition as time_map.AGE_BANDS.
HAZARD_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 30.0),
    (30.0, 90.0),
    (90.0, 180.0),
    (180.0, 365.0),
    (365.0, 730.0),
    (730.0, np.inf),
)
# Representative age for the "flat" (post-infant) hazard region — where a ГТМ
# renewal lands in the column-5 variant that suppresses re-entry into infant risk.
FLAT_RENEWAL_AGE = 365.0

CIF_EVAL_TIMES = (90.0, 180.0, 365.0)


# --------------------------------------------------------------------------- #
# Cause classification (single t_cal clock)                                     #
# --------------------------------------------------------------------------- #
def classify_cause(pull_reason: object, has_failed_unit: bool, ended: bool) -> int:
    """Competing-risks cause code from a run's outcome.

    Mirrors :func:`esp_population.classify` but keeps ГТМ as its own code (2)
    instead of collapsing it to a censor, so the two cause-specific hazards can be
    estimated from one population.
    """
    if not ended:
        return CENSORED
    reason = _norm(pull_reason).casefold()
    if reason in WORKOVER_REASONS:
        return GTM
    if reason in GENUINE_FAILURE_REASONS:
        return FAILURE
    return FAILURE if has_failed_unit else CENSORED


# --------------------------------------------------------------------------- #
# Cause-specific Weibull fits with well-cluster bootstrap CIs                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CauseFit:
    cause: str
    n_events: int
    n_at_risk: int
    beta: float
    eta: float
    beta_lo: float
    beta_hi: float
    eta_lo: float
    eta_hi: float
    rmst_0_730: float
    mrl_0: float
    n_boot_ok: int


def _weibull_survival(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    return np.exp(-np.power(np.clip(t, 0.0, None) / eta, beta))


def _rmst(beta: float, eta: float, horizon: float = 730.0, n: int = 4000) -> float:
    u = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(_weibull_survival(u, beta, eta), u))


def _mrl0(beta: float, eta: float, horizon: float = 20000.0, n: int = 40000) -> float:
    """Mean residual life at age 0 = ∫₀^∞ S(u) du (RMST to a far horizon)."""
    u = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(_weibull_survival(u, beta, eta), u))


def fit_cause_specific(
    pop: pd.DataFrame,
    cause_code: int,
    *,
    n_boot: int = 300,
    seed: int = 42,
) -> CauseFit:
    """Cause-specific Weibull (the other cause treated as censoring).

    Bootstrap CIs resample **wells** (clusters), not runs — 60%+ of runs repeat on
    the same well, so run-resampling understates uncertainty.  Thin strata make
    these CIs wide by construction; they are reported, not hidden.
    """
    t = pop["t_cal"].to_numpy(float)
    ev = (pop["cause_code"].to_numpy(int) == cause_code).astype(int)
    point = fit_basic_weibull(t, ev)
    beta, eta = float(point["beta"]), float(point["eta"])

    wells = pop["well_key"].dropna().unique()
    rows_by_well = {w: pop[pop["well_key"] == w] for w in wells}
    rng = np.random.default_rng(seed)
    betas, etas = [], []
    for _ in range(n_boot):
        drawn = rng.choice(wells, size=len(wells), replace=True)
        boot = pd.concat([rows_by_well[w] for w in drawn], ignore_index=True)
        bt = boot["t_cal"].to_numpy(float)
        be = (boot["cause_code"].to_numpy(int) == cause_code).astype(int)
        if be.sum() < 3:
            continue
        try:
            r = fit_basic_weibull(bt, be)
        except Exception:
            continue
        if r["success"] and 1e-2 < r["beta"] < 1e2 and r["eta"] < 1e6:
            betas.append(float(r["beta"]))
            etas.append(float(r["eta"]))

    def _ci(a: list[float]) -> tuple[float, float]:
        if len(a) < max(10, int(0.1 * n_boot)):
            return (float("nan"), float("nan"))
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    beta_lo, beta_hi = _ci(betas)
    eta_lo, eta_hi = _ci(etas)
    return CauseFit(
        cause={FAILURE: "failure", GTM: "gtm"}[cause_code],
        n_events=int(ev.sum()),
        n_at_risk=int(len(pop)),
        beta=beta,
        eta=eta,
        beta_lo=beta_lo,
        beta_hi=beta_hi,
        eta_lo=eta_lo,
        eta_hi=eta_hi,
        rmst_0_730=_rmst(beta, eta),
        mrl_0=_mrl0(beta, eta),
        n_boot_ok=len(betas),
    )


def piecewise_hazard(pop: pd.DataFrame, cause_code: int) -> pd.DataFrame:
    """Piecewise-constant cause-specific hazard by age band (per op-day).

    ``hazard = cause events in band / person-days at risk in band`` on ``t_cal``.
    This is the shape the renewal engine consumes and the honest read on "does the
    failure hazard rise with age": a flat profile means no wear-out.
    """
    t = pop["t_cal"].to_numpy(float)
    ec = pop["cause_code"].to_numpy(int)
    rows = []
    for lo, hi in HAZARD_BANDS:
        hi_eff = hi if np.isfinite(hi) else float(t.max() + 1.0) if len(t) else lo + 1.0
        person_days = float(np.clip(np.minimum(t, hi_eff) - lo, 0.0, None).sum())
        n_ev = int(((ec == cause_code) & (t > lo) & (t <= hi_eff)).sum())
        hazard = (n_ev / person_days) if person_days > 0 else 0.0
        rows.append(
            {
                "cause": {FAILURE: "failure", GTM: "gtm"}[cause_code],
                "band": f"{int(lo)}_{'inf' if not np.isfinite(hi) else int(hi)}",
                "age_lo": lo,
                "age_hi": hi,
                "person_days": person_days,
                "n_events": n_ev,
                "hazard_per_day": hazard,
                "hazard_per_1000d": hazard * 1000.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aalen–Johansen CIF vs naive 1-KM                                              #
# --------------------------------------------------------------------------- #
def cif_vs_km(pop: pd.DataFrame, at_times: tuple[float, ...] = CIF_EVAL_TIMES) -> pd.DataFrame:
    """AJ cumulative incidence of failure/ГТМ vs the naive (ГТМ-censored) 1-KM.

    The naive curve overstates failure incidence because it ignores the competing
    ГТМ risk that removes pumps before they can fail.
    """
    t = pop["t_cal"].to_numpy(float)
    ec = pop["cause_code"].to_numpy(int)
    cif_fail = CIF.aalen_johansen_cif(t, ec, FAILURE)
    cif_gtm = CIF.aalen_johansen_cif(t, ec, GTM)
    # Naive: treat ГТМ (and running) as plain censoring, KM of failure only.
    naive = CIF.all_cause_km(t, (ec == FAILURE).astype(int))

    rows = []
    for tt in at_times:
        cf = float(cif_fail.at(tt))
        cg = float(cif_gtm.at(tt))
        nk = float(naive.at(tt))
        rows.append(
            {
                "time_d": tt,
                "cif_failure_aj": round(cf, 4),
                "cif_gtm_aj": round(cg, 4),
                "cif_all_aj": round(cf + cg, 4),
                "naive_1_minus_km_failure": round(nk, 4),
                "naive_overstatement_abs": round(nk - cf, 4),
                "naive_overstatement_pct": round(100.0 * (nk - cf) / cf, 1) if cf > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Informative-censoring / preventive-ГТМ test (telemetry-backed)                #
# --------------------------------------------------------------------------- #
def _prefix_clause(column: str, prefixes: tuple[str, ...]) -> str:
    """SQL OR-clause matching a normalized-well column against upper-case prefixes."""
    ors = " OR ".join(f"upper({column}) LIKE '{p.upper()}%'" for p in prefixes)
    return f"({ors})"


def well_mean_qliq(prefixes: tuple[str, ...]) -> pd.Series:
    """Well-level mean liquid rate (a rate proxy for the association test).

    Keyed by lowercase code (telemetry convention).  ``prefixes`` selects the
    field's telemetry rows (e.g. ``("MC", "MR")`` for Мирнинский, ``("VT",)`` Vt).
    """
    try:
        con = sqlite3.connect(resolve_telemetry_db_path())
    except Exception:
        return pd.Series(dtype=float)
    try:
        df = pd.read_sql(
            "SELECT _meta_normalized_well AS well, Qliq_m3d FROM telemetry_daily "
            f"WHERE {_prefix_clause('_meta_normalized_well', prefixes)} "
            "AND Qliq_m3d IS NOT NULL",
            con,
        )
    except Exception:
        return pd.Series(dtype=float)
    finally:
        con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["code"] = df["well"].astype(str).str.lower()
    df["Qliq_m3d"] = pd.to_numeric(df["Qliq_m3d"], errors="coerce")
    return df.groupby("code")["Qliq_m3d"].mean()


def informative_censoring_level1(pop: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Level 1 — does the ГТМ hazard correlate with failure-risk factors?

    A point-biserial association among **closed** runs (ГТМ pull = 1 vs genuine
    failure = 0) on standardized predictors: mean liquid rate (telemetry), calendar
    duration, and contractor group.  A ГТМ hazard that tracks rate/age like the
    failure hazard is the association-level signature of informative censoring.
    """
    closed = pop[pop["cause_code"].isin([FAILURE, GTM])].copy()
    qliq = well_mean_qliq(prefixes)  # keyed by lowercase code (telemetry convention)
    closed["mean_qliq"] = closed["code"].str.lower().map(qliq)
    closed["gtm"] = (closed["cause_code"] == GTM).astype(int)
    closed["brt"] = (closed["contractor_group"] == "brt").astype(int)
    closed["slb"] = (closed["contractor_group"] == "slb").astype(int)

    preds = ["t_cal", "mean_qliq", "brt", "slb"]
    rows = []
    y = closed["gtm"].to_numpy(float)
    for p in preds:
        x = pd.to_numeric(closed[p], errors="coerce").to_numpy(float)
        mask = np.isfinite(x)
        n = int(mask.sum())
        if n < 10 or len(np.unique(y[mask])) < 2:
            rows.append({"predictor": p, "n": n, "corr_gtm": float("nan"), "note": "insufficient"})
            continue
        xs = x[mask]
        r = float(np.corrcoef(xs, y[mask])[0, 1]) if np.std(xs) > 0 else float("nan")
        gtm_mean = float(xs[y[mask] == 1].mean()) if (y[mask] == 1).any() else float("nan")
        fail_mean = float(xs[y[mask] == 0].mean()) if (y[mask] == 0).any() else float("nan")
        rows.append(
            {
                "predictor": p,
                "n": n,
                "corr_gtm": round(r, 3),
                "mean_if_gtm": round(gtm_mean, 2),
                "mean_if_failure": round(fail_mean, 2),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def informative_censoring_level2(
    pop: pd.DataFrame,
    prefixes: tuple[str, ...],
    *,
    window_days: int = 21,
    min_runs: int = 6,
    return_runs: bool = False,
):
    """Level 2 — do ГТМ runs show failure precursors in their final weeks?

    For each closed run, pull that well's telemetry over its final ``window_days``
    and summarise frequency instability (std of frequency_hz) and подача
    degradation (Qliq slope, m³/d per day).  Compare ГТМ-ending runs against
    failure-ending runs at comparable ages.  If ГТМ runs look like failures were
    coming, the latent hazard is understated.  Returns an empty frame (caller stops
    at level 1) when telemetry coverage is too thin.

    ``return_runs=True`` additionally returns the per-run frame (with the run
    ``code``/``end``) so a caller can recode precursor-positive ГТМ runs as failures
    for the latent upper bound.
    """
    empty = (pd.DataFrame(), pd.DataFrame()) if return_runs else pd.DataFrame()
    try:
        con = sqlite3.connect(resolve_telemetry_db_path())
        tel = pd.read_sql(
            "SELECT _meta_normalized_well AS well, date, frequency_hz, Qliq_m3d "
            f"FROM telemetry_daily WHERE {_prefix_clause('_meta_normalized_well', prefixes)}",
            con,
            parse_dates=["date"],
        )
    except Exception:
        return empty
    finally:
        try:
            con.close()
        except Exception:
            pass
    if tel.empty:
        return empty
    tel["code"] = tel["well"].astype(str).str.lower()
    tel["frequency_hz"] = pd.to_numeric(tel["frequency_hz"], errors="coerce")
    tel["Qliq_m3d"] = pd.to_numeric(tel["Qliq_m3d"], errors="coerce")
    tel_by = {c: g.sort_values("date") for c, g in tel.groupby("code")}

    recs = []
    closed = pop[pop["cause_code"].isin([FAILURE, GTM]) & pop["end"].notna()]
    for _, r in closed.iterrows():
        g = tel_by.get(str(r["code"]).lower())
        if g is None:
            continue
        end = pd.Timestamp(r["end"])
        w = g[(g["date"] > end - pd.Timedelta(days=window_days)) & (g["date"] <= end)]
        freq = w["frequency_hz"].dropna()
        ql = w[["date", "Qliq_m3d"]].dropna()
        if len(freq) < 3 and len(ql) < 3:
            continue
        freq_std = float(freq.std()) if len(freq) >= 3 else float("nan")
        ql_slope = float("nan")
        if len(ql) >= 3:
            x = (ql["date"] - ql["date"].min()).dt.days.to_numpy(float)
            if np.std(x) > 0:
                ql_slope = float(np.polyfit(x, ql["Qliq_m3d"].to_numpy(float), 1)[0])
        recs.append(
            {
                "code": r["code"],
                "end": end,
                "cause": r["cause"],
                "age_d": float(r["t_cal"]),
                "freq_std_final": freq_std,
                "qliq_slope_final": ql_slope,
            }
        )
    runs = pd.DataFrame(recs)
    if runs.empty or runs.groupby("cause").size().min() < min_runs:
        # Too thin for a group comparison — caller reports level-1 only.  With
        # return_runs the per-run frame is still handed back (may be empty) so a
        # caller can attempt the latent bound and report its own coverage.
        return (pd.DataFrame(), runs) if return_runs else pd.DataFrame()

    rows = []
    for metric in ("freq_std_final", "qliq_slope_final"):
        g = runs.groupby("cause")[metric]
        rows.append(
            {
                "metric": metric,
                "gtm_median": round(float(g.get_group("gtm").median()), 3) if "gtm" in runs["cause"].values else float("nan"),
                "failure_median": round(float(g.get_group("failure").median()), 3) if "failure" in runs["cause"].values else float("nan"),
                "n_gtm": int((runs["cause"] == "gtm").sum()),
                "n_failure": int((runs["cause"] == "failure").sum()),
            }
        )
    summary = pd.DataFrame(rows)
    return (summary, runs) if return_runs else summary


# --------------------------------------------------------------------------- #
# Deterministic monthly renewal + decomposition                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BandHazards:
    """Piecewise-constant per-day cause hazards, indexed by age band."""

    fail: np.ndarray  # per-band failure hazard (per op-day)
    gtm: np.ndarray   # per-band ГТМ hazard (per op-day)

    def as_age_arrays(self, max_age: int) -> tuple[np.ndarray, np.ndarray]:
        ages = np.arange(max_age + 2)
        hf = np.zeros(max_age + 2)
        hg = np.zeros(max_age + 2)
        for i, (lo, hi) in enumerate(HAZARD_BANDS):
            hi_eff = hi if np.isfinite(hi) else max_age + 2
            m = (ages >= lo) & (ages < hi_eff)
            hf[m] = self.fail[i]
            hg[m] = self.gtm[i]
        return hf, hg


def band_hazards_from_pop(pop: pd.DataFrame) -> BandHazards:
    hf = piecewise_hazard(pop, FAILURE)["hazard_per_day"].to_numpy(float)
    hg = piecewise_hazard(pop, GTM)["hazard_per_day"].to_numpy(float)
    return BandHazards(fail=hf, gtm=hg)


def simulate_renewal(
    seeds: list[tuple[int, int, float]],
    hazards: BandHazards,
    n_days: int,
    *,
    gtm_active: bool = True,
    gtm_reset_age: float = 0.0,
    max_age: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic expected-count competing-risks renewal on a daily grid.

    ``seeds`` = ``(offset_day, start_age, mass)`` injections.  Each day, mass at
    age ``a`` fails / is-pulled with the competing split of the band hazards;
    survivors advance one day; failed mass renews at age 0 (re-enters infant
    exposure) and ГТМ mass renews at ``gtm_reset_age`` (0 normally; a mature age in
    the column-5 variant that suppresses infant re-entry).  Returns daily failure
    and ГТМ event flows.
    """
    hf, hg = hazards.as_age_arrays(max_age)
    if not gtm_active:
        hg = np.zeros_like(hg)
    H = hf + hg
    p_ev = 1.0 - np.exp(-H)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_f = np.where(H > 0, hf / H, 0.0)
    frac_g = np.where(H > 0, 1.0 - frac_f, 0.0)

    seeds_by_day: dict[int, list[tuple[int, float]]] = {}
    for off, age0, mass in seeds:
        seeds_by_day.setdefault(int(off), []).append((int(age0), float(mass)))

    gi = int(round(gtm_reset_age))
    M = np.zeros(max_age + 2)
    daily_fail = np.zeros(n_days)
    daily_gtm = np.zeros(n_days)
    for d in range(n_days):
        for age0, mass in seeds_by_day.get(d, ()):  # inject new/current pumps
            M[min(age0, max_age)] += mass
        fail_flow = M * p_ev * frac_f
        gtm_flow = M * p_ev * frac_g
        daily_fail[d] = fail_flow.sum()
        daily_gtm[d] = gtm_flow.sum()
        surv = M - fail_flow - gtm_flow
        newM = np.zeros_like(M)
        newM[1:] += surv[:-1]           # survivors age one day
        newM[max_age + 1] += surv[max_age + 1]  # keep mass at the cap
        newM[0] += float(fail_flow.sum())        # failure renewal -> age 0
        newM[gi] += float(gtm_flow.sum())        # ГТМ renewal -> age 0 (or flat)
        M = newM
    return daily_fail, daily_gtm


def _months_from(start: pd.Timestamp, n_months: int) -> list[str]:
    return [(start.to_period("M") + i).strftime("%Y-%m") for i in range(n_months)]


def _bucket_daily_to_months(daily: np.ndarray, start: pd.Timestamp) -> dict[str, float]:
    idx = pd.date_range(start, periods=len(daily), freq="D")
    s = pd.Series(daily, index=idx)
    return {p.strftime("%Y-%m"): float(v) for p, v in s.groupby(s.index.to_period("M")).sum().items()}


def observed_window_prediction(pop: pd.DataFrame, hazards: BandHazards) -> pd.DataFrame:
    """No-renewal observed-failure/ГТМ prediction over each run's real lifetime.

    Over its OBSERVED exposure a run is known to be alive, so the expected events
    on a given day are the cause hazard at that age (conditional replay: hazard ×
    observed exposure), matching the repo's history-replay convention.  Summed
    across runs → predicted monthly failure/ГТМ counts.  Because the band hazards
    are the empirical rates, the grand totals reproduce the observed counts by
    construction; the discriminating check is the month-by-month allocation.  Also
    carries the ACTUAL monthly failure/ГТМ counts (month of the run's ``end``).
    """
    max_age = 3000
    hf, hg = hazards.as_age_arrays(max_age)

    pred: dict[str, dict[str, float]] = {}
    actual: dict[str, dict[str, float]] = {}

    def _add(store, month, key, val):
        store.setdefault(month, {"failure": 0.0, "gtm": 0.0})[key] += val

    for _, r in pop.iterrows():
        install = pd.Timestamp(r["install"])
        t_cal = float(r["t_cal"])
        end_eff = install + pd.Timedelta(days=t_cal)
        n = int(np.ceil(t_cal))
        if n <= 0:
            continue
        ages = np.arange(n)
        a = np.clip(ages, 0, max_age)
        inc_f = hf[a]
        inc_g = hg[a]
        dates = install + pd.to_timedelta(ages, unit="D")
        per = pd.PeriodIndex(dates, freq="M")
        for month, gf, gg in zip(pd.Series(per).astype(str), inc_f, inc_g):
            _add(pred, month, "failure", float(gf))
            _add(pred, month, "gtm", float(gg))
        m = end_eff.strftime("%Y-%m")
        if r["cause_code"] == FAILURE:
            _add(actual, m, "failure", 1.0)
        elif r["cause_code"] == GTM:
            _add(actual, m, "gtm", 1.0)

    months = sorted(set(pred) | set(actual))
    rows = []
    for m in months:
        rows.append(
            {
                "month": m,
                "pred_failures": round(pred.get(m, {}).get("failure", 0.0), 4),
                "pred_gtm": round(pred.get(m, {}).get("gtm", 0.0), 4),
                "actual_failures": actual.get(m, {}).get("failure", 0.0),
                "actual_gtm": actual.get(m, {}).get("gtm", 0.0),
            }
        )
    return pd.DataFrame(rows)


def monthly_decomposition(
    pop: pd.DataFrame,
    hazards: BandHazards,
    *,
    as_of: pd.Timestamp,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """The five-column forward decomposition, seeded from the running fleet.

    Each of the ``is_open`` runs enters at its exact current age (conditional on
    survival to that age — deterministic mass 1).  Columns:
      1 obs_failures  — both processes active, renew on both (age 0)
      2 gtm           — ГТМ events from λ_gtm
      3 total_pulls   — failures + ГТМ (workshop load)
      4 latent_nogtm  — λ_gtm off, renew on failure only (counterfactual)
      5 infant_effect — obs_failures minus a variant where ГТМ renews to a mature
                        (flat-hazard) age, isolating infant exposure manufactured
                        by workovers.
    """
    forecast_start = (as_of.to_period("M") + 1).to_timestamp()
    age_gap = int((forecast_start - as_of).days)
    open_runs = pop[pop["is_open"]]
    seeds = [(0, int(round(a)) + age_gap, 1.0) for a in open_runs["current_age"].to_numpy(float)]
    n_days = int(((forecast_start.to_period("M") + horizon_months).to_timestamp() - forecast_start).days)

    fail1, gtm1 = simulate_renewal(seeds, hazards, n_days, gtm_active=True, gtm_reset_age=0.0)
    fail4, _ = simulate_renewal(seeds, hazards, n_days, gtm_active=False)
    fail5, _ = simulate_renewal(seeds, hazards, n_days, gtm_active=True, gtm_reset_age=FLAT_RENEWAL_AGE)

    b_f1 = _bucket_daily_to_months(fail1, forecast_start)
    b_g1 = _bucket_daily_to_months(gtm1, forecast_start)
    b_f4 = _bucket_daily_to_months(fail4, forecast_start)
    b_f5 = _bucket_daily_to_months(fail5, forecast_start)

    months = _months_from(forecast_start, horizon_months)
    fleet = len(open_runs)
    rows = []
    for m in months:
        f1 = b_f1.get(m, 0.0)
        g1 = b_g1.get(m, 0.0)
        f4 = b_f4.get(m, 0.0)
        f5 = b_f5.get(m, 0.0)
        rows.append(
            {
                "month": m,
                "segment": "forecast",
                "fleet_running": fleet,
                "m1_obs_failures": round(f1, 4),
                "m2_gtm": round(g1, 4),
                "m3_total_pulls": round(f1 + g1, 4),
                "m4_latent_nogtm_failures": round(f4, 4),
                "m5_infant_renewal_effect": round(f1 - f5, 4),
                "actual_failures": np.nan,
                "actual_gtm": np.nan,
            }
        )
    return pd.DataFrame(rows)


def combined_decomposition(
    pop: pd.DataFrame,
    hazards: BandHazards,
    obs_window: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    horizon_months: int = 12,
    obs_tail_months: int = 12,
) -> pd.DataFrame:
    """One table: the observed-window calibration rows (model vs actual) followed
    by the forecast rows (five-column decomposition)."""
    fc = monthly_decomposition(pop, hazards, as_of=as_of, horizon_months=horizon_months)
    asof_month = as_of.to_period("M")
    tail = obs_window[obs_window["month"] < asof_month.strftime("%Y-%m")].copy()
    tail = tail.sort_values("month").tail(obs_tail_months)
    obs_rows = []
    for _, r in tail.iterrows():
        obs_rows.append(
            {
                "month": r["month"],
                "segment": "observed",
                "fleet_running": np.nan,
                "m1_obs_failures": round(float(r["pred_failures"]), 4),
                "m2_gtm": round(float(r["pred_gtm"]), 4),
                "m3_total_pulls": round(float(r["pred_failures"] + r["pred_gtm"]), 4),
                "m4_latent_nogtm_failures": np.nan,
                "m5_infant_renewal_effect": np.nan,
                "actual_failures": float(r["actual_failures"]),
                "actual_gtm": float(r["actual_gtm"]),
            }
        )
    return pd.concat([pd.DataFrame(obs_rows), fc], ignore_index=True)


__all__ = [
    "CENSORED", "FAILURE", "GTM", "CAUSE_ORDER", "HAZARD_BANDS", "FLAT_RENEWAL_AGE",
    "CIF_EVAL_TIMES", "classify_cause", "CauseFit", "fit_cause_specific",
    "piecewise_hazard", "cif_vs_km", "well_mean_qliq", "informative_censoring_level1",
    "informative_censoring_level2", "BandHazards", "band_hazards_from_pop",
    "simulate_renewal", "observed_window_prediction", "monthly_decomposition",
    "combined_decomposition",
]
