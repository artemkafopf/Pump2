"""Pump sizing in the **infant band** — the half of the failures the landmark cannot see.

Why this exists
---------------
:mod:`pump_sizing_landmark` measured what oversizing costs a pump that has already
survived its first 90 operating days: within-well HR 2.01, ~101 d of residual RMST per
doubling of nameplate.  It reached that by excluding every run that failed earlier —
**50.8 % of all failures**.  So the headline number is a cost conditional on surviving
infancy, and the obvious question it leaves open is whether an oversized pump also *dies
young*.  If it does, the true cost of oversizing is larger than 101 d and the sizing rule
needs re-pricing; if it does not, the landmark number is the whole story.

Why the landmark design cannot be reused
----------------------------------------
A fixed 90-op-day feature window is impossible for a run that fails at day 12.  Any
window short enough to fit infant runs is contaminated: a pump on its way to failure has
collapsing ``qliq`` and drifting pressure, so a "first 14 days" mean partly *is* the
outcome.  That is the differential-tail-guard problem ``kpod_features`` was built to
avoid, and shortening the window brings it straight back.

The way out is to give up telemetry features entirely and use only what is known **at
install**:

* ``log_qnom`` — nameplate flow.  Chosen by an engineer before the run starts, so it
  cannot be contaminated by how the run ended.  This is the same argument that made Qnom
  the preferred rate variable in ``vt_v4``/``unified_v4``, and it is what makes the
  infant band analysable at all.
* contractor, field, sour class, ``run_seq``.
* **prior-run deliverability** — ``kprod``/``rpl`` averaged over the well's last operating
  days *strictly before this run's install*.  Leakage-free by construction, and it
  substitutes for the own-run ``kprod`` the landmark design used to break the
  ``corr(Qnom, kprod) = +0.64`` confound.

The payoff for dropping telemetry is coverage: **4 139 runs / 2 136 events**, against
1 496 / 770 for the landmark — 2.8× the events, and no conditioning at all.

The estimand
------------
Rather than pick an infant/mature boundary and defend it, the size effect is allowed to
**vary over follow-up**: a Cox model in counting-process form with a heaviside
``log_qnom`` coefficient per interval (:data:`BAND_CUTS`).  This directly answers "is the
size effect present in the infant band, and is it the same size as in the mature band?"
on one common scale, and the mature interval should approximately reproduce the landmark
result as an internal consistency check.

⚠ This is a *plain* piecewise coefficient, not an ``X·log t`` interaction — the latter is
collinear with X and manufactures nulls (``project_cox_glf_null``).

Clock is ``t_cal`` throughout (calendar), because operating-day clocks require the daily
join that infant runs frequently lack.  Note this makes the band boundaries **not**
directly comparable to the landmark's 90 *operating* days: 34.4 % of failures fall inside
90 calendar days versus 50.8 % inside 90 operating days.

Mechanism check
---------------
If oversizing kills young pumps, it should leave a fingerprint in *how* they die.
:func:`failure_node_by_size` tests whether the failed-node composition of infant failures
shifts with pump size — an installation/commissioning defect population should be
size-indifferent, whereas overload should concentrate in specific nodes.

What it measured (2026-07-28)
------------------------------
Frame: **4 139 runs / 2 136 events** (vs 1 496 / 770 for the landmark), 2 197 runs with
prior-run deliverability.  Events by band: 366 / 366 / 760 / 644.

**An oversized pump does NOT die young.  The size effect grows with age.**
Within-well (``strata='well_key'``, reservoir quality cancelled), HR per e-fold of Qnom:

===============  ========  ==================  ========
band (cal. days) events    HR [95 % CI]        p
===============  ========  ==================  ========
[0, 30)          372       **1.13 [0.93, 1.37]**  **0.23**
[30, 90)         361       1.38 [1.12, 1.69]   0.0026
[90, 365)        762       1.66 [1.41, 1.95]   <1e-9
[365, ∞)         641       1.87 [1.49, 2.36]   <1e-7
===============  ========  ==================  ========

Monotone, and the bands are genuinely different — :func:`band_homogeneity_test` gives
χ² = 14.02, df = 3, **p = 0.0029** (Ya p = 0.022), so a single fleet-wide HR is the wrong
summary.  The mature band's 1.87 is consistent with the landmark's within-well 2.01,
which is a useful cross-design check: two different estimands, two different clocks, same
answer where they overlap.

Read plainly: **oversizing is a duty/wear effect that accumulates with running time, not
an installation effect.**  Infant mortality on this fleet is size-indifferent — consistent
with commissioning and installation defects, which is what the infant band is usually
made of.

*Consequence for the landmark result.*  The excluded half of the failures carries no size
signal, so the landmark's ~101 d cost per doubling is **not an underestimate** — the
conditioning on surviving to day 90 turns out to be benign for this particular exposure.
That was not knowable in advance and is the main reason this module exists.

*The confound, visible directly.*  Between-well (unstratified) the infant band shows a
"significant" effect — HR 1.25 [1.08, 1.45], p = 0.003 — which **vanishes within-well**
(1.13, p = 0.23).  That gap is the ``corr(Qnom, kprod) = +0.64`` selection showing up as
a spurious infant effect.  Anyone running this without well stratification would conclude
oversizing kills young pumps.  It does not.

*Failure-mode check is a non-replication.*  Node composition of infant failures shifts
with size pooled (χ² p = 0.018) but not on Ya (p = 0.25) or Vt (p = 0.90); the pooled
significance is field-mix.  No mechanistic support either way.

⚠ **Do not read this as "small pumps are safe".**  Unadjusted survival by size tercile
still separates from day 30 (S(30) = 0.927 small vs 0.881 large) — but that is the
confound, not causation, and the within-well fit is what removes it.
"""
from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import WAREHOUSE_DIR, results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as popmod
from analysis.workflows.production_risk import kpod_features as K
from analysis.workflows.production_risk import pump_sizing_landmark as PSL

SLUG = "production_risk_pump_sizing_infant"

CLOCK = "t_cal"
EVENT_COL = "event"

#: Interval boundaries (calendar days) for the piecewise size effect.  Chosen to isolate
#: startup/commissioning (<30 d), the rest of the infant spike (30-90), early-mature
#: (90-365) and mature (>365).  The hazard is known to be infant-spike-then-flat with no
#: wear-out (``project_mc_weibull_shape``), so these track a real regime change.
BAND_CUTS = (30.0, 90.0, 365.0)

#: Operating days of pre-install history averaged for the leakage-free deliverability
#: control.  Taken strictly before ``install``.
PRIOR_WINDOW_OPDAYS = 90
#: Look no further back than this for prior history (a two-year-old measurement is not
#: this well's current deliverability).
PRIOR_LOOKBACK_DAYS = 730

RMST_HORIZON = 730.0


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass
class InfantCoverage:
    n_runs_total: int
    n_with_qnom: int
    n_events: int
    n_with_prior_deliverability: int
    n_events_within_first_cut: int
    n_events_by_band: dict = dc_field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        d = {k: v for k, v in self.__dict__.items() if not isinstance(v, dict)}
        d.update({f"events_band_{k}": v for k, v in self.n_events_by_band.items()})
        return pd.DataFrame([d])


def _prior_deliverability(
    pop: pd.DataFrame, *, db_path: Path | None = None
) -> pd.DataFrame:
    """Well deliverability measured **strictly before** each run's install.

    For every run, averages ``kprod`` / ``rpl`` / ``qliq`` over the last
    :data:`PRIOR_WINDOW_OPDAYS` operating days occurring before ``install`` (and no more
    than :data:`PRIOR_LOOKBACK_DAYS` earlier).  A run that is the well's first has no
    prior history and gets NaN — that is honest, not imputable, and is counted.

    This is the infant-band substitute for the landmark design's own-run ``kprod``.  It
    cannot leak the outcome because every row used pre-dates the run.
    """
    keys = sorted(pop["well_key"].unique().tolist())
    if not keys:
        return pd.DataFrame()
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        ph = ",".join("?" * len(keys))
        d = pd.read_sql(
            "SELECT well_key, dt, qliq, kprod, rpl FROM proc__daily_merged "
            f"WHERE well_key IN ({ph})", con, params=keys)
    finally:
        con.close()
    d["dt"] = pd.to_datetime(d["dt"], errors="coerce")
    for c in ("qliq", "kprod", "rpl"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["qliq"] > 0) & d["dt"].notna()]
    d["kprod"] = PSL._guard(d["kprod"].to_numpy(), *PSL.KPROD_RANGE)
    from analysis.features.free_gas import P_RES_RANGE
    d["rpl"] = PSL._guard(d["rpl"].to_numpy(), *P_RES_RANGE)
    by = {k: g.sort_values("dt") for k, g in d.groupby("well_key", sort=False)}

    rows = []
    for idx, run in pop.iterrows():
        rec = {"_idx": idx, "prior_kprod": np.nan, "prior_rpl": np.nan,
               "prior_ql": np.nan, "prior_n_days": 0}
        g = by.get(run["well_key"])
        if g is not None and pd.notna(run["install"]):
            lo = pd.Timestamp(run["install"]) - pd.Timedelta(days=PRIOR_LOOKBACK_DAYS)
            w = g[(g["dt"] < run["install"]) & (g["dt"] >= lo)].tail(PRIOR_WINDOW_OPDAYS)
            if len(w):
                rec.update({
                    "prior_kprod": float(np.nanmean(w["kprod"])) if w["kprod"].notna().any() else np.nan,
                    "prior_rpl": float(np.nanmean(w["rpl"])) if w["rpl"].notna().any() else np.nan,
                    "prior_ql": float(np.nanmean(w["qliq"])),
                    "prior_n_days": int(len(w)),
                })
        rows.append(rec)
    return pd.DataFrame(rows).set_index("_idx")


def build_infant_frame(
    as_of: pd.Timestamp | None = None,
    *,
    use_canonical_qnom: bool = True,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, InfantCoverage]:
    """Full population with install-time covariates — **no landmark, no exclusion**."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(C.SVOD_OPEN_ASOF)

    pop = popmod.build(as_of=as_of)
    pop = popmod.add_time_scales(pop, as_of)
    pop = popmod.apply_mc_cohort(pop)
    pop = K.resolve_qnom(pop, db_path=db_path)
    if use_canonical_qnom:
        pop = PSL._apply_canonical_qnom(pop)
    pop["well_key"] = pop["code"].astype(str).str.casefold()
    pop = pop.sort_values(["well_key", "install"]).reset_index(drop=True)
    pop["run_seq"] = pop.groupby("well_key").cumcount() + 1

    n_total = len(pop)
    pop = pop[pop["nominal_flow_m3d"] > 0].copy()
    pop = pop[pop[CLOCK] > 0].copy()

    pop = pop.join(_prior_deliverability(pop, db_path=db_path))
    pop = _attach_failed_node(pop, db_path=db_path)

    pop["log_qnom"] = np.log(pop["nominal_flow_m3d"])
    pop["log_prior_kprod"] = np.log(pop["prior_kprod"].where(pop["prior_kprod"] > 0))
    pop["has_prior"] = pop["prior_kprod"].notna()

    bands = _band_labels(pop[CLOCK])
    ev_by_band = (pop[pop[EVENT_COL] == 1].assign(b=_band_labels(pop.loc[pop[EVENT_COL] == 1, CLOCK]))
                  .groupby("b", observed=True).size().to_dict())
    cov = InfantCoverage(
        n_runs_total=n_total, n_with_qnom=int(len(pop)),
        n_events=int(pop[EVENT_COL].sum()),
        n_with_prior_deliverability=int(pop["has_prior"].sum()),
        n_events_within_first_cut=int(((pop[EVENT_COL] == 1) & (pop[CLOCK] <= BAND_CUTS[0])).sum()),
        n_events_by_band={str(k): int(v) for k, v in ev_by_band.items()},
    )
    pop["band"] = bands
    return pop.reset_index(drop=True), cov


def _band_labels(t: pd.Series) -> pd.Series:
    edges = [0.0, *BAND_CUTS, np.inf]
    labels = [f"[{int(a)},{int(b) if np.isfinite(b) else 'inf'})"
              for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(t, bins=edges, labels=labels, right=False, include_lowest=True)


def _attach_failed_node(pop: pd.DataFrame, *, db_path: Path | None = None) -> pd.DataFrame:
    """Attach ``failed_node`` from the v03 run passport by (well, install) nearest match."""
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        r = pd.read_sql("SELECT well_key, install_date, failed_node FROM raw__v03_runs", con)
    finally:
        con.close()
    r["code"] = r["well_key"].astype(str).str.casefold()
    r["install"] = pd.to_datetime(r["install_date"], errors="coerce")
    r = r.dropna(subset=["install"])[["code", "install", "failed_node"]]

    left = pop.copy()
    left["_row"] = np.arange(len(left))
    left["code"] = left["well_key"]
    merged = pd.merge_asof(
        left.sort_values("install"), r.sort_values("install"),
        on="install", by="code", tolerance=pd.Timedelta(days=30), direction="nearest",
    )
    return merged.sort_values("_row").drop(columns=["_row", "code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The piecewise size effect — the core test
# ---------------------------------------------------------------------------

def to_counting_process(
    df: pd.DataFrame,
    *,
    cuts: tuple[float, ...] = BAND_CUTS,
    terms: tuple[str, ...] = ("log_qnom",),
    extra: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Split each run into follow-up intervals, one row per (run, interval).

    Produces ``start``/``stop``/``event`` plus a **separate copy of each term per
    interval** (``log_qnom__b0``, ``log_qnom__b1``, …), all zero outside their own
    interval.  Fitting a Cox model on this frame yields one coefficient per interval —
    the heaviside time-varying effect — with no ``X·log t`` collinearity.
    """
    edges = [0.0, *cuts, np.inf]
    labels = [f"b{i}" for i in range(len(edges) - 1)]
    rows = []
    for i, run in df.reset_index(drop=True).iterrows():
        t, ev = float(run[CLOCK]), int(run[EVENT_COL])
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if t <= a:
                break
            stop = min(t, b)
            rec = {"run_id": i, "start": a, "stop": stop,
                   EVENT_COL: int(ev == 1 and stop == t), "band": labels[j]}
            for term in terms:
                v = run[term]
                for k, lab in enumerate(labels):
                    rec[f"{term}__{lab}"] = float(v) if (k == j and pd.notna(v)) else 0.0
                if pd.isna(v):
                    rec[f"{term}__{labels[j]}"] = np.nan
            for c in extra:
                rec[c] = run[c]
            rows.append(rec)
            if stop == t:
                break
    return pd.DataFrame(rows)


def piecewise_size_effect(
    df: pd.DataFrame,
    *,
    cuts: tuple[float, ...] = BAND_CUTS,
    controls: tuple[str, ...] = (),
    strata: str | None = None,
    label: str = "",
    penalizer: float = 0.0,
) -> pd.DataFrame:
    """Cox with a per-interval ``log_qnom`` coefficient.

    ``strata='well_key'`` gives the within-well version, where reservoir quality cancels
    — the same identification the landmark analysis relied on, now applied across the
    whole of follow-up including the infant band.

    ⚠ ``penalizer`` defaults to **0**, and that matters more than it looks.  With 795
    well strata and one ``log_qnom`` column per band — each mostly zeros — even a ridge
    of 0.01 crushes the estimates toward the null: measured, the four band HRs collapse
    from 1.13 / 1.38 / 1.66 / 1.87 to 1.02 / 1.04 / 1.12 / 1.15, turning a clean monotone
    gradient into a flat non-result.  The validity check that catches this is
    :func:`_check_reproduces_plain_cox` — an unpenalised counting-process fit with the
    band columns summed back into one must reproduce a plain stratified Cox exactly.
    A penalty is applied only as a convergence fallback, and is reported when used.
    """
    from lifelines import CoxTimeVaryingFitter

    need = [CLOCK, EVENT_COL, "log_qnom", *controls] + ([strata] if strata else [])
    d = df.dropna(subset=need).copy()
    extra = tuple(controls) + ((strata,) if strata else ())
    long = to_counting_process(d, cuts=cuts, terms=("log_qnom",), extra=extra)
    long = long.dropna()
    if long.empty or long[EVENT_COL].sum() < 10:
        return pd.DataFrame()

    labels = [f"b{i}" for i in range(len(cuts) + 1)]
    cols = ["run_id", "start", "stop", EVENT_COL] + \
           [f"log_qnom__{l}" for l in labels] + list(controls)
    if strata:
        cols.append(strata)
        # A stratum with no event contributes nothing to the partial likelihood.
        ev = long.groupby(strata)[EVENT_COL].sum()
        long = long[long[strata].isin(ev[ev > 0].index)]
    # Drop all-zero interval columns (no exposure in that band at all).
    keep = [c for c in cols if not (c.startswith("log_qnom__") and long[c].abs().sum() == 0)]

    s, used_pen, note = None, penalizer, ""
    for pen in (penalizer, 0.01, 0.1):
        try:
            ctv = CoxTimeVaryingFitter(penalizer=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctv.fit(long[keep], id_col="run_id", start_col="start", stop_col="stop",
                        event_col=EVENT_COL, strata=[strata] if strata else None,
                        show_progress=False)
            s, used_pen = ctv.summary, pen
            if pen != penalizer:
                note = f"PENALISED at {pen} to converge — coefficients are shrunk toward 1"
            break
        except Exception as exc:
            last = exc
    if s is None:
        return pd.DataFrame([{"label": label, "strata": strata or "none",
                              "note": f"fit failed: {last}"}])

    edges = [0.0, *cuts, np.inf]
    out = []
    for i, l in enumerate(labels):
        term = f"log_qnom__{l}"
        if term not in s.index:
            continue
        n_ev = int(long.loc[long["band"] == l, EVENT_COL].sum()) if "band" in long else -1
        out.append({
            "label": label, "strata": strata or "none",
            "band": f"[{int(edges[i])},{int(edges[i+1]) if np.isfinite(edges[i+1]) else 'inf'})",
            "n_events_in_band": n_ev,
            "HR_per_e_fold": float(np.exp(s.loc[term, "coef"])),
            "HR_lo": float(np.exp(s.loc[term, "coef lower 95%"])),
            "HR_hi": float(np.exp(s.loc[term, "coef upper 95%"])),
            "p": float(s.loc[term, "p"]),
            "HR_per_doubling": float(np.exp(s.loc[term, "coef"] * np.log(2))),
            "penalizer": used_pen,
            "note": note,
        })
    o = pd.DataFrame(out)
    if not o.empty:
        # A stratified band with few events can return an "estimate" like HR 14.8
        # [0.39, 557].  That is not a number; mark it so it never reaches a headline.
        width = o["HR_hi"] / o["HR_lo"].clip(lower=1e-9)
        o["unstable"] = width > 20.0
        o.loc[o["unstable"], "note"] = (
            o.loc[o["unstable"], "note"].str.rstrip("; ") + "; CI spans >20x — NOT INTERPRETABLE"
        ).str.lstrip("; ")
    return o


def _check_reproduces_plain_cox(
    df: pd.DataFrame, *, strata: str | None = None, tol: float = 1e-3
) -> dict:
    """Validity gate on the counting-process construction.

    Collapsing the per-band columns back into one must reproduce a plain (stratified) Cox
    on the same rows.  If it does not, the interval split, the risk sets, or the penalty
    is wrong — and every band estimate is untrustworthy.  Measured clean: 1.4832 vs
    1.4832 stratified, 1.3291 vs 1.3291 unstratified.
    """
    from lifelines import CoxPHFitter, CoxTimeVaryingFitter

    need = [CLOCK, EVENT_COL, "log_qnom"] + ([strata] if strata else [])
    d = df.dropna(subset=need).copy()
    if strata:
        ev = d.groupby(strata)[EVENT_COL].sum()
        d = d[d[strata].isin(ev[ev > 0].index)]
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d[need], duration_col=CLOCK, event_col=EVENT_COL,
                strata=[strata] if strata else None)
    ref = float(cph.summary.loc["log_qnom", "coef"])

    long = to_counting_process(d, terms=("log_qnom",), extra=(strata,) if strata else ())
    bands = [c for c in long.columns if c.startswith("log_qnom__")]
    long["lq_all"] = long[bands].sum(axis=1)
    cols = ["run_id", "start", "stop", EVENT_COL, "lq_all"] + ([strata] if strata else [])
    ctv = CoxTimeVaryingFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctv.fit(long[cols], id_col="run_id", start_col="start", stop_col="stop",
                event_col=EVENT_COL, strata=[strata] if strata else None, show_progress=False)
    got = float(ctv.summary.loc["lq_all", "coef"])
    return {"strata": strata or "none", "plain_cox_coef": ref, "counting_process_coef": got,
            "abs_diff": abs(ref - got), "n_events": int(d[EVENT_COL].sum()),
            "flag": "OK" if abs(ref - got) < tol else "COUNTING-PROCESS CONSTRUCTION BROKEN"}


def band_homogeneity_test(pw: pd.DataFrame) -> dict:
    """Is the size effect the SAME in every band?

    A Wald-type test on the spread of the per-band log-HRs, using their CIs for standard
    errors.  Rejecting means the infant and mature bands price oversizing differently and
    a single fleet-wide HR is the wrong summary.
    """
    from scipy.stats import chi2

    d = pw.dropna(subset=["HR_per_e_fold", "HR_lo", "HR_hi"])
    if "unstable" in d.columns:
        d = d[~d["unstable"]]        # an uninterpretable band must not drive the test
    if len(d) < 2:
        return {"chi2": np.nan, "df": 0, "p": np.nan, "pooled_HR": np.nan}
    b = np.log(d["HR_per_e_fold"].to_numpy())
    se = (np.log(d["HR_hi"].to_numpy()) - np.log(d["HR_lo"].to_numpy())) / (2 * 1.96)
    w = 1.0 / np.clip(se, 1e-9, None) ** 2
    bbar = float(np.sum(w * b) / np.sum(w))
    q = float(np.sum(w * (b - bbar) ** 2))
    return {"chi2": q, "df": len(b) - 1, "p": float(chi2.sf(q, len(b) - 1)),
            "pooled_HR": float(np.exp(bbar))}


# ---------------------------------------------------------------------------
# Mechanism: does an oversized pump die of something different?
# ---------------------------------------------------------------------------

def _size_terciles(q: pd.Series) -> pd.Series:
    """Tercile labels for pump size, robust to a degenerate size distribution.

    ``pd.qcut(..., labels=[...], duplicates='drop')`` raises when tied quantiles collapse
    an edge, because the label list no longer matches the surviving bins.  Label after
    binning instead, so a two-valued size column yields two groups rather than an error.
    """
    codes = pd.qcut(q, 3, labels=False, duplicates="drop")
    n = int(pd.Series(codes).nunique(dropna=True))
    names = {0: "small", 1: "mid", 2: "large"} if n != 2 else {0: "small", 1: "large"}
    order = [names[k] for k in sorted(names)]
    # Ordered categorical, so groupby/plot order follows pump size rather than the
    # alphabet ("large" < "mid" < "small").
    return pd.Categorical(pd.Series(codes, index=q.index).map(names),
                          categories=order, ordered=True)


def failure_node_by_size(
    df: pd.DataFrame, *, band_max: float = 90.0, min_count: int = 15
) -> tuple[pd.DataFrame, dict]:
    """Failed-node composition of infant failures, by pump-size tercile.

    An infant population made of installation/commissioning defects should be broadly
    size-indifferent.  A shift toward specific nodes with size is mechanistic evidence
    that oversizing itself is doing the killing.
    """
    from scipy.stats import chi2_contingency

    d = df[(df[EVENT_COL] == 1) & (df[CLOCK] <= band_max)].dropna(
        subset=["failed_node", "nominal_flow_m3d"]).copy()
    if len(d) < 3 * min_count:
        return pd.DataFrame(), {"note": "too few infant failures with a node"}
    d["size_tercile"] = _size_terciles(d["nominal_flow_m3d"])
    keep = d["failed_node"].value_counts()
    keep = keep[keep >= min_count].index
    d = d[d["failed_node"].isin(keep)]
    ct = pd.crosstab(d["failed_node"], d["size_tercile"])
    # crosstab keeps the categorical dtype on the columns index, which makes `.join`
    # raise InvalidIndexError — flatten to plain strings before combining.
    ct.columns = [str(c) for c in ct.columns]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, {"note": "degenerate table"}
    chi2v, p, dof, _ = chi2_contingency(ct)
    share = (ct / ct.sum(axis=0)).round(3)
    share.columns = [f"share_{c}" for c in share.columns]
    return ct.join(share), {"chi2": float(chi2v), "dof": int(dof), "p": float(p),
                            "n": int(ct.to_numpy().sum()),
                            "note": "composition differs by size" if p < 0.05
                                    else "no size-dependent mode shift"}


def hazard_by_size_tercile(df: pd.DataFrame, *, horizon: float = RMST_HORIZON) -> pd.DataFrame:
    """Non-parametric survival by pump-size tercile — the shape, with no model imposed."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    d = df.dropna(subset=["nominal_flow_m3d", CLOCK, EVENT_COL]).copy()
    d["size_tercile"] = _size_terciles(d["nominal_flow_m3d"])
    rows = []
    for k, g in d.groupby("size_tercile", observed=True):
        kmf = KaplanMeierFitter().fit(g[CLOCK], g[EVENT_COL])
        h = min(horizon, float(g[CLOCK].max()))
        rows.append({
            "size_tercile": str(k), "n": len(g), "n_events": int(g[EVENT_COL].sum()),
            "qnom_median": float(g["nominal_flow_m3d"].median()),
            "surv_30d": float(kmf.predict(30.0)), "surv_90d": float(kmf.predict(90.0)),
            "surv_365d": float(kmf.predict(365.0)),
            "rmst": float(restricted_mean_survival_time(kmf, t=h)), "horizon": h,
            "median_life": float(kmf.median_survival_time_),
        })
    return pd.DataFrame(rows)


def total_cost_of_oversizing(
    df: pd.DataFrame,
    *,
    controls: tuple[str, ...] = ("log_prior_kprod",),
    label: str = "",
    size_multiple: float = 2.0,
    horizon: float = RMST_HORIZON,
) -> pd.DataFrame:
    """RMST cost of upsizing measured **from install**, i.e. including the infant band.

    The landmark analysis priced the same trade from day 90 onward.  The difference
    between the two is precisely the infant-band contribution, which is what this whole
    module exists to quantify.
    """
    terms = ("log_qnom",) + tuple(controls)
    fit, d = PSL.fit_weibull_ph(df, terms, time_col=CLOCK, standardize=False)
    if not np.isfinite(fit.loglik):
        return pd.DataFrame()
    X = d[list(terms)].to_numpy(float)
    lin_ref = float(X.mean(axis=0) @ np.array([fit.coefs[t] for t in terms]))
    eta_ref = fit.eta_scale * float(np.exp(-lin_ref / fit.beta_shape))
    base = PSL.rmst(eta_ref, fit.beta_shape, horizon)
    dlin = fit.coefs["log_qnom"] * np.log(size_multiple)
    up = PSL.rmst(eta_ref * float(np.exp(-dlin / fit.beta_shape)), fit.beta_shape, horizon)
    return pd.DataFrame([{
        "label": label, "scope": "from install (all bands)",
        "n": fit.n, "n_events": fit.n_events,
        "beta_shape": fit.beta_shape, "size_multiple": size_multiple,
        "HR_per_size_multiple": float(np.exp(dlin)),
        "rmst_ref_days": base, "rmst_upsized_days": up,
        "rmst_lost_days": base - up,
        "rmst_lost_pct": (base - up) / base if base else np.nan,
        "mrl_ref_days": PSL.mean_residual_life(eta_ref, fit.beta_shape),
    }])


def run(
    as_of: pd.Timestamp | None = None,
    *,
    fields: tuple[str, ...] = ("Ya", "Vt", "Mc"),
    out_dir: Path | None = None,
) -> dict:
    """End-to-end infant-band analysis; writes tables and figures."""
    out = out_dir or results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"

    df, cov = build_infant_frame(as_of=as_of)
    df.drop(columns=[c for c in ("band",) if c in df.columns], errors="ignore") \
      .to_csv(tables / "infant_frame.csv", index=False, encoding="utf-8-sig")
    cov.as_frame().to_csv(tables / "coverage.csv", index=False, encoding="utf-8-sig")

    # Validity gate first — a broken split invalidates every band estimate below.
    gate = pd.DataFrame([_check_reproduces_plain_cox(df, strata=s) for s in (None, "well_key")])
    gate.to_csv(tables / "counting_process_gate.csv", index=False, encoding="utf-8-sig")
    if (gate["flag"] != "OK").any():
        warnings.warn("counting-process gate FAILED — band estimates are not trustworthy")

    pws, homos, costs, terciles, nodes, node_tests = [], [], [], [], [], []
    scopes = [("pooled", df)] + [(f, df[df["field"] == f]) for f in fields]
    for name, g in scopes:
        if len(g) < 100 or g[EVENT_COL].sum() < 40:
            continue
        # Between-well (deliverability-controlled) and within-well versions.
        for strata, ctrl in [(None, ("log_prior_kprod",)), ("well_key", ())]:
            pw = piecewise_size_effect(g, controls=ctrl, strata=strata, label=name)
            if pw.empty:
                continue
            pws.append(pw)
            if "HR_per_e_fold" in pw.columns:
                homos.append({"label": name, "strata": strata or "none",
                              **band_homogeneity_test(pw)})
        costs.append(total_cost_of_oversizing(g, label=name))
        t = hazard_by_size_tercile(g)
        t.insert(0, "label", name)
        terciles.append(t)
        ct, test = failure_node_by_size(g)
        if not ct.empty:
            ct = ct.reset_index()
            ct.insert(0, "label", name)
            nodes.append(ct)
        node_tests.append({"label": name, **test})

    _w(tables / "piecewise_size_effect.csv", pws)
    pd.DataFrame(homos).to_csv(tables / "band_homogeneity.csv", index=False, encoding="utf-8-sig")
    _w(tables / "total_cost.csv", costs)
    _w(tables / "size_terciles.csv", terciles)
    _w(tables / "failure_node_by_size.csv", nodes)
    pd.DataFrame(node_tests).to_csv(tables / "failure_node_test.csv",
                                    index=False, encoding="utf-8-sig")
    _figures(df, pd.concat(pws) if pws else pd.DataFrame(), figures)

    return {"frame": df, "coverage": cov, "gate": gate,
            "piecewise": pd.concat(pws, ignore_index=True) if pws else pd.DataFrame(),
            "homogeneity": pd.DataFrame(homos),
            "cost": pd.concat(costs, ignore_index=True) if costs else pd.DataFrame(),
            "terciles": pd.concat(terciles, ignore_index=True) if terciles else pd.DataFrame(),
            "nodes": pd.concat(nodes, ignore_index=True) if nodes else pd.DataFrame(),
            "node_test": pd.DataFrame(node_tests), "out_dir": out}


def _w(path: Path, frames: list) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def _figures(df: pd.DataFrame, pw: pd.DataFrame, figdir: Path) -> None:
    """Survival by size tercile (the shape) and the size effect across follow-up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    d = df.dropna(subset=["nominal_flow_m3d", CLOCK]).copy()
    d["tercile"] = _size_terciles(d["nominal_flow_m3d"])
    for k, g in d.groupby("tercile", observed=True):
        KaplanMeierFitter(label=f"{k} (n={len(g)}, Qnom~{g['nominal_flow_m3d'].median():.0f})") \
            .fit(g[CLOCK], g[EVENT_COL]).plot_survival_function(ax=axes[0], ci_show=False)
    axes[0].axvline(BAND_CUTS[1], ls=":", c="k", lw=.8)
    axes[0].set(xlim=(0, 730), xlabel="calendar days from install",
                ylabel="S(t)", title="Survival by pump-size tercile")

    p = pw[(pw.get("strata") == "well_key") & (pw.get("label") == "pooled")] \
        if not pw.empty else pd.DataFrame()
    if not p.empty:
        x = np.arange(len(p))
        axes[1].errorbar(x, p["HR_per_e_fold"],
                         yerr=[p["HR_per_e_fold"] - p["HR_lo"], p["HR_hi"] - p["HR_per_e_fold"]],
                         fmt="o", capsize=4, color="C3")
        axes[1].axhline(1.0, lw=.8, color="k")
        axes[1].set_xticks(x, p["band"], rotation=20)
        axes[1].set_yscale("log")
        axes[1].set(ylabel="HR per e-fold Qnom (within-well)",
                    title="Does the size effect exist in the infant band?")
    fig.tight_layout()
    fig.savefig(figdir / "infant_size_effect.png", dpi=150)
    plt.close(fig)


__all__ = [
    "SLUG", "BAND_CUTS", "CLOCK", "EVENT_COL", "RMST_HORIZON",
    "InfantCoverage", "build_infant_frame", "to_counting_process",
    "piecewise_size_effect", "band_homogeneity_test", "failure_node_by_size",
    "hazard_by_size_tercile", "total_cost_of_oversizing", "run",
]
