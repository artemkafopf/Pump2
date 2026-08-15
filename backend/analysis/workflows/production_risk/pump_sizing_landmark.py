"""Pump sizing vs operating life — the day-90 landmark design.

The question
------------
Two operating strategies, and the fleet contains both:

* **A — oversize.**  Run a large pump, take the rate early, let reservoir pressure and
  liquid rate fall away; Kpod drifts down over the run and its *average* ends up low.
* **B — right-size.**  Run a smaller pump, hold the rate, produce at a roughly constant
  Kpod for longer.

Does A cost pump life, and if so how much?  Mean Kpod cannot answer this — measured
across 984 runs, ``corr(Qnom, Kpod_mean) = −0.04``.  The strategies are invisible in the
level and live entirely in the **trajectory**.

Why a landmark, and what it fixes
---------------------------------
The naive version of this analysis — fit a Kpod slope over each run's own guarded window
and correlate with run length — reproduces a strong result (ρ = +0.35) that is largely an
artefact.  Three defects, all fixed here:

1. **Window-length artefact.**  A slope fitted over a run's own length is noisier and
   steeper on short runs, so it correlates with run length semi-definitionally.  Fixed by
   computing every feature on a **fixed window of the first ``LANDMARK_OP_DAYS`` operating
   days** — identical for every run, regardless of how long the run turned out to be.
2. **Survivor selection.**  Only 41 % of runs reach 90 guarded op-days, and the ones that
   do not are disproportionately the infant failures.  Fixed by making the conditioning
   *explicit*: the estimand is residual life **given survival to the landmark**, and
   :attr:`LandmarkCoverage` reports exactly who was dropped and how they died.  Nothing
   is silently truncated (``project_field_sim``: never drop censored or young runs
   without saying so).
3. **Outcome leakage.**  Features stop at the landmark; the outcome starts at it.  There
   is no guard-gap fudge because the two windows do not overlap by construction.

Confounding — the part the data cannot fully close
--------------------------------------------------
``corr(Qnom, kprod) = +0.64``: big pumps are *chosen* for productive wells.  A naive
Qnom → life regression is therefore confounded by well quality, and will over-state the
cost of oversizing.  Two defences, both implemented:

* **Deliverability controls** — ``kprod``, ``rpl``, drawdown enter the fit, so Qnom is
  read at fixed well capability.
* **Within-well identification** (:func:`within_well_contrast`) — a Cox model stratified
  on ``well_key`` uses only wells that ran *different* pump sizes at different times, so
  reservoir quality cancels.  On the landmark-eligible frame this is 224 wells / 586 runs
  at a ≥1.25× size change (182 / 487 at ≥1.5×).

⚠ Even the within-well design does not close it: successive runs on a well are ordered in
time, so pump size is entangled with cumulative depletion.  ``run_seq`` is carried as a
control, and the direction of the size change (up vs down) is reported separately, but
"the big pump caused the decline" and "the well was declining anyway" are not fully
separable from observational data alone.  This is why β matters — see below.

Why free gas is in this module
------------------------------
Sizing and gas are one causal chain, not two questions:

    Qnom → drawdown → P_intake below P_bubble → free gas at intake → pump damage

β from :mod:`analysis.features.free_gas` is the **mediator**.  A large pump that drives
intake pressure down into heavy free gas has a mechanism; one that merely sits on a
declining well does not.  Testing whether β carries the Qnom effect is the closest this
data gets to distinguishing cause from correlation.

Reporting standard (``feedback_report_rmst_mrl``): RMST over the horizon measured **from
the landmark**, with a KM control; median secondary; never B50.

What it measured (2026-07-28, landmark 90 unless noted)
-------------------------------------------------------
*Frame.*  1 496 eligible runs / 770 events out of 4 175; Ya 696, Vt 240, Mc 75.  Weibull
RMST agrees with KM to 0.5-1.9 % in every scope, so the parametric form holds.

**1. Oversizing costs life, and the confound was HIDING it, not creating it.**
Between-well with deliverability controls, HR = **1.46 per e-fold of Qnom** (Ya 1.58,
Vt 1.41); within-well, where reservoir quality cancels, HR = **2.01 [1.59, 2.54],
p = 4e-9** (351 wells, 1 079 runs, 637 events).  The within-well estimate is *larger*.
That is the expected direction given ``corr(Qnom, kprod) = +0.64``: big pumps go on good
wells, so the naive between-well comparison is biased toward "sizing is harmless".
Stable across landmarks 60/90/120 (within-well 2.06 / 2.01 / 1.98) and across the
≥1.25× and ≥1.5× size-change thresholds.  Priced out: doubling nameplate costs ~52 d of
residual RMST between-well, ~101 d within-well — 11 % and 22 % of a 465 d baseline —
and buys ~161 m³/d of liquid.

**2. The trajectory hypothesis is only half-confirmed — and it does not replicate.**
Kpod slope enters with the hypothesised sign (flatter Kpod → lower hazard, HR 0.89/SD
pooled, 0.84 Ya) and the trajectory block is significant pooled (LR p = 0.038) and on Ya
(p = 0.017).  But on **Vt it reverses sign (HR 1.12) and LOSES out-of-sample**
(CV −6.9073 → −6.9378), and Mc is worse still.  Per the standing replication gate
(``project_hazard_catboost``: covariate transfer that dies on replication is dead), the
trajectory layer is **not shippable** — it is a Ya result, not a fleet result.  Sizing,
by contrast, replicates on both Ya and Vt and in the within-well design.

**3. Free gas at intake is an honest NULL as the mediator.**  This was the mechanism the
analysis was built to test, and it fails: adding β attenuates the Qnom coefficient by
**0.4 %** (Ya 0.8 %, Vt 2.0 %), and the LR test for β is non-significant everywhere
(pooled p = 0.36, Ya p = 0.26, Vt p = 0.065).  The attenuation is ~0 at landmarks 60, 90
and 120 alike.  β is well built — anchored, PVT-rank-invariant (Spearman 1.0000 across
the assumption grid), responsive to pressure — so this is a real null, not a broken
feature.  **The size effect does not run through gas interference at the intake.**
Whatever mechanism connects a large pump to a shorter life, this data says it is not that
one.  ⚠ Caveat that keeps the null honest: β carries no separation credit and is an
upper bound; if separator fitting correlates with pump size, a true mediation could be
masked.  The warehouse records no separator equipment, so this is untestable here.

⚠ **What this design cannot see.**  Half of all failures (50.8 % at landmark 90; 41.6 %
at 60, 57.2 % at 120) occur *before* the landmark and are excluded by construction.  The
estimand is residual life given survival to day 90 and it is silent about infant
mortality.  :func:`landmark_selection_audit` confirms the exclusion is not size-selective
(pooled p = 0.078 at L90; at L60 the dropped runs are *smaller*, p = 0.005, which biases
against finding a size effect) — so the sizing result is conservative, not inflated.  But
"does an oversized pump die young?" remains an open and separate question.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.data.landmark_features import theil_sen_slope
from analysis.features import free_gas as FG
from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as popmod
from analysis.workflows.production_risk import kpod_features as K

SLUG = "production_risk_pump_sizing_landmark"

#: Landmark, in **operating** days (qliq > 0).  90 is the primary: it keeps 984 runs
#: fleet-wide (Ya 477, Vt 131) while giving a long enough arm to estimate a slope.
#: 60 is the sensitivity copy — more runs, noisier slopes.
LANDMARK_OP_DAYS = 90
LANDMARK_SENSITIVITY = (60, 120)

#: Outcome horizon for RMST, in calendar days **from the landmark**.
RMST_HORIZON = 730.0

#: Minimum finite points in the landmark window before a slope is honest.
MIN_SLOPE_POINTS = 30

#: Telemetry guards.  Kept aligned with ``free_gas`` and ``landmark_features``; rzab in
#: particular carries catastrophic outliers in the warehouse (min −125 973, max 1 249 937).
RZAB_RANGE = (0.0, 600.0)
KPROD_RANGE = (0.0, 500.0)

#: Covariate blocks.  ``DELIVERABILITY`` is the confounder control, ``SIZING`` the
#: exposure, ``TRAJECTORY`` the hypothesis, ``GAS`` the mediator.
SIZING = ("log_qnom",)
DELIVERABILITY = ("log_kprod", "rpl_level", "drawdown_level")
TRAJECTORY = ("kpod_level", "kpod_slope", "pint_slope")
GAS = ("beta_mean",)

CLOCK = "t_cal"
EVENT_COL = "event"


# ---------------------------------------------------------------------------
# Landmark frame
# ---------------------------------------------------------------------------

@dataclass
class LandmarkCoverage:
    """Who entered the landmark frame and — more importantly — who did not."""

    landmark_op_days: int
    n_runs_total: int
    n_no_qnom: int
    n_no_telemetry: int
    n_short_of_landmark: int
    n_eligible: int
    #: Events among the runs dropped for not reaching the landmark.  This is the
    #: selection the design accepts; it must be reported, never assumed benign.
    n_events_dropped_short: int
    n_events_eligible: int
    #: The runs that failed to reach the landmark, kept so
    #: :func:`landmark_selection_audit` can check whether the design throws away the
    #: effect it is measuring.  Carried here rather than in ``df.attrs`` because pandas
    #: compares ``attrs`` on ``concat`` and propagates them into ``copy()`` — a
    #: DataFrame-valued attr then makes both ``concat`` and lifelines raise.
    short_runs: pd.DataFrame = dc_field(default_factory=pd.DataFrame, repr=False)

    def share_of_events_lost(self) -> float:
        """Share of all failures that occur **before** the landmark and are excluded.

        At the day-90 landmark this is ~0.51 fleet-wide.  It is the single most important
        number for interpreting anything downstream: the estimand is residual life given
        survival to the landmark, and it is silent about infant mortality.
        """
        tot = self.n_events_dropped_short + self.n_events_eligible
        return self.n_events_dropped_short / tot if tot else float("nan")

    def as_frame(self) -> pd.DataFrame:
        d = {k: v for k, v in self.__dict__.items() if not isinstance(v, pd.DataFrame)}
        d["share_eligible"] = self.n_eligible / self.n_runs_total if self.n_runs_total else np.nan
        d["share_events_lost_to_landmark"] = self.share_of_events_lost()
        return pd.DataFrame([d])


def _guard(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.asarray(a, dtype=float).copy()
    with np.errstate(invalid="ignore"):
        out[(out < lo) | (out > hi)] = np.nan
    return out


def _slope_per_100d(y: np.ndarray, x: np.ndarray) -> float:
    """Robust Theil-Sen slope per 100 operating days, or NaN if under-supported."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < MIN_SLOPE_POINTS:
        return float("nan")
    return theil_sen_slope(y[m], x[m]) * 100.0


def _mean(a: np.ndarray) -> float:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def landmark_window_features(
    dts: np.ndarray,
    qliq: np.ndarray,
    freq: np.ndarray,
    p_intake: np.ndarray,
    rpl: np.ndarray,
    rzab: np.ndarray,
    kprod: np.ndarray,
    watercut: np.ndarray,
    gas_factor: np.ndarray,
    *,
    qnom: float,
    p_bubble: float,
    landmark_op_days: int = LANDMARK_OP_DAYS,
    pvt: FG.PVT = FG.DEFAULT_PVT,
) -> dict | None:
    """Features over the first ``landmark_op_days`` **operating** days of one run.

    Pure (no DB): all arrays are daily rows already restricted to the run and sorted or
    not.  Returns ``None`` when the run never reaches the landmark — that is the
    eligibility rule, and the caller counts it.

    Every returned feature uses the fixed window only.  ``landmark_day_offset`` is the
    calendar age at which the landmark was reached, which the caller needs to convert the
    run's total life into residual life.
    """
    order = np.argsort(pd.to_datetime(pd.Series(dts)).to_numpy())
    dts = pd.to_datetime(pd.Series(dts)).to_numpy()[order]
    q = np.asarray(qliq, dtype=float)[order]

    op = np.isfinite(q) & (q > 0)
    if op.sum() < landmark_op_days:
        return None

    # The window is the first L operating days; `cut` is the positional index of the L-th.
    op_idx = np.flatnonzero(op)[:landmark_op_days]
    sel = op_idx
    x = np.arange(1.0, landmark_op_days + 1.0)      # operating-day clock inside the window

    def g(a, lo=None, hi=None):
        v = np.asarray(a, dtype=float)[order][sel]
        return v if lo is None else _guard(v, lo, hi)

    p = g(p_intake, *FG.P_INTAKE_RANGE)
    wct = g(watercut, *FG.WATERCUT_RANGE)
    gor = g(gas_factor, *FG.GOR_RANGE)
    pres = g(rpl, *FG.P_RES_RANGE)
    pzab = g(rzab, *RZAB_RANGE)
    kpr = g(kprod, *KPROD_RANGE)
    qw = q[sel]

    kpod = qw / qnom if np.isfinite(qnom) and qnom > 0 else np.full(len(qw), np.nan)

    out: dict = {
        "landmark_day_offset": float((dts[op_idx[-1]] - dts[0]) / np.timedelta64(1, "D")),
        "n_op_days_window": int(len(sel)),
        # -- sizing (fixed at install, cannot be contaminated by the outcome)
        "qnom": float(qnom),
        # -- trajectory: level AND slope, because they carry opposite-signed information
        "kpod_level": _mean(kpod),
        "kpod_slope": _slope_per_100d(kpod, x),
        "pint_level": _mean(p),
        "pint_slope": _slope_per_100d(p, x),
        "ql_level": _mean(qw),
        "ql_slope": _slope_per_100d(qw, x),
        "freq_level": _mean(g(freq, 20.0, 75.0)),
        # -- deliverability (the confounder block)
        "kprod_level": _mean(kpr),
        "rpl_level": _mean(pres),
        "rpl_slope": _slope_per_100d(pres, x),
        "drawdown_level": _mean(pres - pzab),
        # -- fluid
        "wct_level": _mean(wct),
        "glf_level": _mean(gor),
    }
    # -- gas mediator ---------------------------------------------------------
    pb = p_bubble if (np.isfinite(p_bubble) and p_bubble > 0) else None
    out.update(FG.free_gas_window(p, gor, wct, p_bubble_atm=pb, pvt=pvt))
    out["beta_slope"] = _slope_per_100d(
        FG.free_gas_fraction(p, gor, wct, p_bubble_atm=pb, pvt=pvt), x
    )
    out["beta_anchored"] = bool(pb is not None)
    return out


def _apply_canonical_qnom(pop: pd.DataFrame) -> pd.DataFrame:
    """Repair nameplate flow via the vendor catalogue before it becomes the exposure.

    Without this a handful of REDA rows carry the raw bbl/day model number (``ESP 538-7000``
    recorded as 7000 m³/d against a true ~1200), and since ``log_qnom`` is *the* exposure
    here a 6× error on the largest pumps is not survivable — it would sit at the extreme of
    the regressor and drag the slope.  Mirrors ``vt_v4.USE_CANONICAL_QNOM``.
    """
    from analysis.data.pump_nominal_catalog import canonical_qnom
    from analysis.workflows.production_risk import vt_v4

    out = vt_v4.attach_pump_designation(pop)
    vals, srcs = [], []
    for des, q in zip(out.get("gno_type", pd.Series(index=out.index, dtype=object)),
                      out["nominal_flow_m3d"]):
        try:
            qc, _fam, src = canonical_qnom(des, None if pd.isna(q) else float(q))
        except Exception:
            qc, src = (None if pd.isna(q) else float(q)), "recorded"
        vals.append(np.nan if qc is None else float(qc))
        srcs.append(src)
    out["nominal_flow_m3d"] = vals
    out["qnom_catalog_source"] = srcs
    return out


def build_landmark_frame(
    as_of: pd.Timestamp | None = None,
    *,
    landmark_op_days: int = LANDMARK_OP_DAYS,
    pvt: FG.PVT = FG.DEFAULT_PVT,
    use_canonical_qnom: bool = True,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, LandmarkCoverage]:
    """Build the run × landmark frame for all fields.

    One row per **eligible** run (reached ``landmark_op_days`` operating days), carrying
    fixed-window features and the residual-life outcome measured from the landmark.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(C.SVOD_OPEN_ASOF)

    pop = popmod.build(as_of=as_of)
    pop = popmod.add_time_scales(pop, as_of)
    pop = popmod.apply_mc_cohort(pop)          # Mc = installs 2024+ (standing rule)
    pop = K.resolve_qnom(pop, db_path=db_path)
    if use_canonical_qnom:
        pop = _apply_canonical_qnom(pop)
    pop["well_key"] = pop["code"].astype(str).str.casefold()

    # Run sequence per well — the depletion-order control for the within-well design.
    pop = pop.sort_values(["well_key", "install"]).reset_index(drop=True)
    pop["run_seq"] = pop.groupby("well_key").cumcount() + 1

    dailies = _load_dailies(sorted(pop["well_key"].unique().tolist()), db_path=db_path)
    by_well = {k: g.sort_values("dt") for k, g in dailies.groupby("well_key", sort=False)}
    pbub = _bubble_point_by_well(db_path=db_path)

    rows: list[dict] = []
    short_rows: list[dict] = []
    n_no_qnom = n_no_tel = n_short = 0
    ev_short = 0
    for _, run in pop.iterrows():
        if not np.isfinite(run["nominal_flow_m3d"]) or run["nominal_flow_m3d"] <= 0:
            n_no_qnom += 1
            continue
        g = by_well.get(run["well_key"])
        if g is None or pd.isna(run["install"]):
            n_no_tel += 1
            continue
        stop = pd.Timestamp(run["end"]) if pd.notna(run["end"]) else as_of
        w = g[(g["dt"] >= run["install"]) & (g["dt"] <= stop)]
        if w.empty:
            n_no_tel += 1
            continue
        feats = landmark_window_features(
            w["dt"].to_numpy(), w["qliq"].to_numpy(), w["freq"].to_numpy(),
            w["rpump_intake"].to_numpy(), w["rpl"].to_numpy(), w["rzab"].to_numpy(),
            w["kprod"].to_numpy(), w["watercut"].to_numpy(), w["gas_factor"].to_numpy(),
            qnom=float(run["nominal_flow_m3d"]),
            p_bubble=float(pbub.get(run["well_key"], np.nan)),
            landmark_op_days=landmark_op_days, pvt=pvt,
        )
        if feats is None:
            n_short += 1
            ev_short += int(run[EVENT_COL] == 1)
            # Keep the minimum needed to audit the selection this design accepts.
            short_rows.append({"well_key": run["well_key"], "field": run["field"],
                               "qnom": float(run["nominal_flow_m3d"]),
                               EVENT_COL: int(run[EVENT_COL]), "t_total": float(run[CLOCK])})
            continue
        rec = {
            "row_key": run.name, "well_key": run["well_key"], "code": run["code"],
            "field": run["field"], "contractor_group": run["contractor_group"],
            "h2s_class": run["h2s_class"], "install": run["install"], "end": run["end"],
            "run_seq": int(run["run_seq"]),
            "t_total": float(run[CLOCK]), EVENT_COL: int(run[EVENT_COL]),
            "kpod_qnom_source": run["kpod_qnom_source"],
            "p_bubble": float(pbub.get(run["well_key"], np.nan)),
        }
        rec.update(feats)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("landmark frame is empty — check the telemetry join")

    # -- outcome: residual life measured FROM the landmark --------------------
    df["t_resid"] = df["t_total"] - df["landmark_day_offset"]
    # A run whose landmark lands within a day of its stop carries no residual exposure.
    n_zero = int((df["t_resid"] <= 0.5).sum())
    df = df[df["t_resid"] > 0.5].copy()

    # -- model transforms -----------------------------------------------------
    df["log_qnom"] = np.log(df["qnom"].where(df["qnom"] > 0))
    df["log_kprod"] = np.log(df["kprod_level"].where(df["kprod_level"] > 0))
    df["log_t_resid"] = np.log(df["t_resid"])

    cov = LandmarkCoverage(
        landmark_op_days=landmark_op_days,
        n_runs_total=int(len(pop)),
        n_no_qnom=n_no_qnom, n_no_telemetry=n_no_tel, n_short_of_landmark=n_short,
        n_eligible=int(len(df)),
        n_events_dropped_short=ev_short,
        n_events_eligible=int((df[EVENT_COL] == 1).sum()),
        short_runs=pd.DataFrame(short_rows),
    )
    if n_zero:
        warnings.warn(f"{n_zero} runs reached the landmark within 0.5 d of stopping — dropped")
    return df, cov


def landmark_selection_audit(df: pd.DataFrame, short_runs: pd.DataFrame) -> pd.DataFrame:
    """Compare the runs the landmark KEPT against the ones it DROPPED, on pump size.

    This is the check that decides whether the whole design is fit for purpose.  Measured
    on the day-90 landmark, **50.8 % of all failures occur before the landmark** and are
    excluded by construction.  That is tolerable only if the excluded runs are not
    systematically the large pumps — because if oversized pumps preferentially die in
    infancy, the landmark removes exactly the effect being estimated and the surviving
    contrast is biased toward "sizing does not matter".

    Returns per-field median/mean Qnom for kept vs dropped runs plus a rank-sum p-value.
    A significant *upward* shift in the dropped group is a red flag on the estimand, and
    must be reported next to any headline, not buried.
    """
    from scipy.stats import mannwhitneyu

    short = short_runs
    if short is None or short.empty:
        return pd.DataFrame()
    # A handful of runs carry a null field (unmapped well code) — keep them in the pooled
    # row but do not try to sort them into a per-field scope.
    fields = sorted(f for f in df["field"].dropna().unique() if isinstance(f, str))
    rows = []
    for scope, kept, drop in [("pooled", df, short)] + [
        (f, df[df["field"] == f], short[short["field"] == f]) for f in fields
    ]:
        k = kept["qnom"].dropna()
        s = drop["qnom"].dropna()
        if len(k) < 10 or len(s) < 10:
            continue
        try:
            p = float(mannwhitneyu(k, s, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        rows.append({
            "scope": scope, "n_kept": len(k), "n_dropped": len(s),
            "qnom_median_kept": float(k.median()), "qnom_median_dropped": float(s.median()),
            "qnom_mean_kept": float(k.mean()), "qnom_mean_dropped": float(s.mean()),
            "events_dropped": int(drop[EVENT_COL].sum()),
            "p_ranksum": p,
            "flag": "DROPPED RUNS ARE LARGER" if (s.median() > k.median() and p < 0.05)
                    else ("dropped runs are smaller" if (s.median() < k.median() and p < 0.05)
                          else "no size difference"),
        })
    return pd.DataFrame(rows)


def _load_dailies(well_keys: list[str], *, db_path: Path | None = None) -> pd.DataFrame:
    """Daily telemetry for the landmark features (one query, all channels)."""
    import sqlite3

    from analysis.paths import WAREHOUSE_DIR

    if not well_keys:
        return pd.DataFrame()
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        ph = ",".join("?" * len(well_keys))
        df = pd.read_sql(
            "SELECT well_key, dt, freq, qliq, rpump_intake, rpl, rzab, kprod, "
            f"watercut, gas_factor FROM proc__daily_merged WHERE well_key IN ({ph})",
            con, params=list(well_keys),
        )
    finally:
        con.close()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for c in ["freq", "qliq", "rpump_intake", "rpl", "rzab", "kprod", "watercut", "gas_factor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["dt"])


def _bubble_point_by_well(*, db_path: Path | None = None) -> dict:
    """Per-well bubble point (atm) from the run passport, for the β anchoring."""
    import sqlite3

    from analysis.paths import WAREHOUSE_DIR

    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        r = pd.read_sql(
            "SELECT well_key, AVG(pbubble_atm) pb FROM raw__v03_runs "
            "WHERE pbubble_atm > 0 GROUP BY well_key", con)
    finally:
        con.close()
    r["well_key"] = r["well_key"].astype(str).str.casefold()
    return dict(zip(r["well_key"], r["pb"]))


# ---------------------------------------------------------------------------
# Weibull-PH on residual life
# ---------------------------------------------------------------------------

@dataclass
class PHFit:
    terms: tuple[str, ...]
    beta_shape: float
    eta_scale: float
    coefs: dict = dc_field(default_factory=dict)
    se: dict = dc_field(default_factory=dict)
    loglik: float = np.nan
    n: int = 0
    n_events: int = 0
    converged: bool = False

    @property
    def n_params(self) -> int:
        return 2 + len(self.terms)

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik

    def hazard_ratio(self, term: str, delta: float = 1.0) -> float:
        return float(np.exp(self.coefs.get(term, 0.0) * delta))

    def summary(self) -> pd.DataFrame:
        rows = []
        for t in self.terms:
            b, s = self.coefs[t], self.se.get(t, np.nan)
            z = b / s if np.isfinite(s) and s > 0 else np.nan
            rows.append({
                "term": t, "coef": b, "se": s,
                "HR": float(np.exp(b)),
                "HR_lo": float(np.exp(b - 1.96 * s)) if np.isfinite(s) else np.nan,
                "HR_hi": float(np.exp(b + 1.96 * s)) if np.isfinite(s) else np.nan,
                "z": z,
                "p": float(2 * (1 - _norm_cdf(abs(z)))) if np.isfinite(z) else np.nan,
            })
        return pd.DataFrame(rows)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _ph_nll(params, t, e, X):
    lb, le = params[0], params[1]
    b, eta = np.exp(lb), np.exp(le)
    lin = X @ params[2:] if X.shape[1] else np.zeros(len(t))
    z = (t / eta) ** b
    # loglik = Σ e·[log(b/eta) + (b−1)·log(t/eta) + lin] − Σ z·exp(lin)
    ll = np.sum(e * (np.log(b / eta) + (b - 1.0) * np.log(t / eta) + lin)) - np.sum(z * np.exp(lin))
    return -ll if np.isfinite(ll) else 1e12


def fit_weibull_ph(
    df: pd.DataFrame,
    terms: tuple[str, ...],
    *,
    time_col: str = "t_resid",
    event_col: str = EVENT_COL,
    standardize: bool = True,
) -> tuple[PHFit, pd.DataFrame]:
    """Weibull proportional-hazards MLE on ``time_col``, right-censored by ``event_col``.

    Covariates are z-scored by default so coefficients are per-SD and comparable across
    blocks — and so the optimiser is not handed a design matrix spanning 6 orders of
    magnitude.  Returns the fit and the complete-case frame it was fitted on.

    Plain PH: no ``X·log t`` interaction (``project_cox_glf_null`` — the interaction is
    collinear with X and manufactures nulls).
    """
    need = [time_col, event_col, *terms]
    d = df.dropna(subset=need).copy()
    if len(d) < 10 or d[event_col].sum() < 3:
        return PHFit(terms=terms, beta_shape=np.nan, eta_scale=np.nan, n=len(d),
                     n_events=int(d[event_col].sum())), d

    t = d[time_col].to_numpy(float)
    e = d[event_col].to_numpy(float)
    X = d[list(terms)].to_numpy(float) if terms else np.zeros((len(d), 0))
    mu = X.mean(axis=0) if (standardize and X.shape[1]) else np.zeros(X.shape[1])
    sd = X.std(axis=0) if (standardize and X.shape[1]) else np.ones(X.shape[1])
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd if X.shape[1] else X

    p0 = np.concatenate([[0.0, np.log(max(np.median(t), 1.0))], np.zeros(X.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(_ph_nll, p0, args=(t, e, Xs), method="BFGS",
                       options={"maxiter": 2000, "gtol": 1e-7})

    coefs = {term: float(res.x[2 + i]) for i, term in enumerate(terms)}
    se: dict = {}
    try:
        H = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else res.hess_inv.todense()
        d2 = np.sqrt(np.clip(np.diag(H), 0, None))
        se = {term: float(d2[2 + i]) for i, term in enumerate(terms)}
    except Exception:
        se = {term: float("nan") for term in terms}

    fit = PHFit(
        terms=terms, beta_shape=float(np.exp(res.x[0])), eta_scale=float(np.exp(res.x[1])),
        coefs=coefs, se=se, loglik=float(-res.fun), n=int(len(d)),
        n_events=int(e.sum()), converged=bool(res.success),
    )
    # Keep the standardisation so RMST can be evaluated at real covariate values.
    fit_scaling = pd.DataFrame({"term": list(terms), "mu": mu, "sd": sd})
    d.attrs["scaling"] = fit_scaling
    return fit, d


def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON, n: int = 4000) -> float:
    """RMST(0, horizon) for a Weibull by numeric integration of S(t)."""
    if not (np.isfinite(eta) and np.isfinite(beta)) or eta <= 0 or beta <= 0:
        return float("nan")
    t = np.linspace(0.0, horizon, n)
    return float(np.trapezoid(np.exp(-((t / eta) ** beta)), t))


def mean_residual_life(eta: float, beta: float, horizon: float = 5000.0, n: int = 20000) -> float:
    """MRL(0) = ∫S(t)dt to a long horizon — reported alongside RMST per the standing rule."""
    return rmst(eta, beta, horizon=horizon, n=n)


def rmst_at(fit: PHFit, lin_pred: float, horizon: float = RMST_HORIZON) -> float:
    """RMST for a subject whose linear predictor is ``lin_pred`` (PH scales η)."""
    eta_eff = fit.eta_scale * float(np.exp(-lin_pred / fit.beta_shape))
    return rmst(eta_eff, fit.beta_shape, horizon)


# ---------------------------------------------------------------------------
# The tests that matter
# ---------------------------------------------------------------------------

def lr_test(small: PHFit, big: PHFit) -> dict:
    """Likelihood-ratio test of nested Weibull-PH fits (``small`` ⊂ ``big``)."""
    from scipy.stats import chi2

    if small.n != big.n:
        return {"lr": np.nan, "df": np.nan, "p": np.nan,
                "note": f"NOT NESTED — different complete-case n ({small.n} vs {big.n})"}
    df = big.n_params - small.n_params
    lr = 2.0 * (big.loglik - small.loglik)
    return {"lr": float(lr), "df": int(df),
            "p": float(chi2.sf(lr, df)) if df > 0 else np.nan,
            "d_aic": float(big.aic - small.aic), "note": ""}


def cv_loglik(
    df: pd.DataFrame,
    terms: tuple[str, ...],
    *,
    k: int = 5,
    repeats: int = 5,
    seed: int = 20260728,
    time_col: str = "t_resid",
) -> dict:
    """Out-of-sample log-likelihood per held-out event, k-fold, repeated.

    In-sample fit lies about added covariates on this data (``project_vt_v4_model``:
    per-contractor Ql slopes won in-sample MAE and were pure overfitting, caught only by
    CV).  Any "the slope adds information" claim goes through this.
    """
    need = [time_col, EVENT_COL, *terms]
    d = df.dropna(subset=need).reset_index(drop=True)
    if len(d) < 5 * k:
        return {"cv_ll_per_event": np.nan, "n": len(d), "repeats": 0}

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = rng.permutation(len(d))
        folds = np.array_split(idx, k)
        tot_ll, tot_ev = 0.0, 0
        for f in folds:
            te = d.iloc[f]
            tr = d.drop(index=d.index[f])
            fit, _ = fit_weibull_ph(tr, terms, time_col=time_col, standardize=False)
            if not np.isfinite(fit.loglik):
                continue
            t = te[time_col].to_numpy(float)
            e = te[EVENT_COL].to_numpy(float)
            X = te[list(terms)].to_numpy(float) if terms else np.zeros((len(te), 0))
            lin = X @ np.array([fit.coefs[c] for c in terms]) if terms else np.zeros(len(te))
            b, eta = fit.beta_shape, fit.eta_scale
            ll = np.sum(e * (np.log(b / eta) + (b - 1) * np.log(t / eta) + lin)) \
                - np.sum((t / eta) ** b * np.exp(lin))
            tot_ll += float(ll)
            tot_ev += int(e.sum())
        if tot_ev:
            scores.append(tot_ll / tot_ev)
    return {"cv_ll_per_event": float(np.mean(scores)) if scores else np.nan,
            "cv_ll_sd": float(np.std(scores)) if scores else np.nan,
            "n": len(d), "repeats": len(scores)}


def within_well_contrast(
    df: pd.DataFrame,
    terms: tuple[str, ...] = ("log_qnom",),
    *,
    min_ratio: float = 1.25,
    time_col: str = "t_resid",
) -> dict:
    """Cox stratified on ``well_key`` — pump size read *within* well.

    Only wells that ran materially different pump sizes (``max/min Qnom ≥ min_ratio``)
    across ≥2 landmark-eligible runs contribute.  Reservoir quality, which drives the
    ``corr(Qnom, kprod) = +0.64`` confound, cancels inside the stratum.

    ``run_seq`` is added as a covariate so the estimate is not simply reading cumulative
    depletion, which is ordered with run number.
    """
    from lifelines import CoxPHFitter

    d = df.dropna(subset=[time_col, EVENT_COL, "well_key", "run_seq", *terms]).copy()
    agg = d.groupby("well_key")["qnom"].agg(["count", "min", "max"])
    keep = agg[(agg["count"] >= 2) & (agg["max"] / agg["min"] >= min_ratio)].index
    d = d[d["well_key"].isin(keep)].copy()
    out = {"n_wells": int(d["well_key"].nunique()), "n_runs": int(len(d)),
           "n_events": int(d[EVENT_COL].sum()), "min_ratio": min_ratio}
    # A stratum with no event contributes nothing to the partial likelihood.
    ev_by_well = d.groupby("well_key")[EVENT_COL].sum()
    d = d[d["well_key"].isin(ev_by_well[ev_by_well > 0].index)]
    out["n_informative_wells"] = int(d["well_key"].nunique())
    out["n_informative_runs"] = int(len(d))
    if d["well_key"].nunique() < 5 or d[EVENT_COL].sum() < 10:
        out["note"] = "insufficient within-well events"
        return out
    cols = [time_col, EVENT_COL, "well_key", "run_seq", *terms]
    try:
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(d[cols], duration_col=time_col, event_col=EVENT_COL, strata=["well_key"])
        s = cph.summary
        for t in (*terms, "run_seq"):
            if t in s.index:
                out[f"HR_{t}"] = float(np.exp(s.loc[t, "coef"]))
                out[f"HR_{t}_lo"] = float(np.exp(s.loc[t, "coef lower 95%"]))
                out[f"HR_{t}_hi"] = float(np.exp(s.loc[t, "coef upper 95%"]))
                out[f"p_{t}"] = float(s.loc[t, "p"])
        out["note"] = ""
    except Exception as exc:                       # convergence / separation
        out["note"] = f"cox failed: {exc}"
    return out


def km_control(df: pd.DataFrame, *, time_col: str = "t_resid") -> dict:
    """Kaplan-Meier RMST on the same rows — the mandatory non-parametric control.

    A Weibull RMST that disagrees with KM by more than a few percent means the parametric
    form is wrong, not that the covariates are interesting.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    d = df.dropna(subset=[time_col, EVENT_COL])
    kmf = KaplanMeierFitter().fit(d[time_col], d[EVENT_COL])
    horizon = min(RMST_HORIZON, float(d[time_col].max()))
    return {
        "km_rmst": float(restricted_mean_survival_time(kmf, t=horizon)),
        "horizon": horizon, "n": int(len(d)), "n_events": int(d[EVENT_COL].sum()),
        "km_median": float(kmf.median_survival_time_),
    }


def sizing_quartile_table(df: pd.DataFrame, *, by: str = "field") -> pd.DataFrame:
    """Descriptive: within-field Qnom quartiles against the fixed-window features.

    This is the honest version of the exploratory table that motivated the analysis — the
    features now come from a window identical across runs, so a steeper slope in the big-
    pump quartile can no longer be an artefact of those runs being shorter.
    """
    out = []
    for key, g in df.groupby(by):
        if len(g) < 40 or g["qnom"].nunique() < 4:
            continue
        gg = g.copy()
        gg["quartile"] = pd.qcut(gg["qnom"], 4,
                                 labels=["Q1 small", "Q2", "Q3", "Q4 big"], duplicates="drop")
        t = gg.groupby("quartile", observed=True).agg(
            n=("qnom", "size"), qnom_med=("qnom", "median"),
            kpod_level=("kpod_level", "median"), kpod_slope=("kpod_slope", "median"),
            pint_level=("pint_level", "median"), pint_slope=("pint_slope", "median"),
            beta_mean=("beta_mean", "median"), drawdown=("drawdown_level", "median"),
            kprod=("kprod_level", "median"),
            t_resid_med=("t_resid", "median"), event_rate=(EVENT_COL, "mean"),
        ).reset_index()
        t.insert(0, by, key)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def mediation_sequence(
    df: pd.DataFrame, *, label: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The nested model ladder — the core result.

    ``M0`` deliverability only → ``M1`` + sizing → ``M2`` + trajectory → ``M3`` + gas.

    The mediation read is on ``log_qnom``: if the size effect is *caused* through gas, its
    coefficient attenuates when β enters (M1 → M3).  If it barely moves, gas is not the
    channel and the size effect — whatever survives the deliverability controls — is
    running through something else.

    All models are fitted on the SAME complete-case rows so the LR tests are valid; that
    is enforced here rather than left to chance.

    Returns ``(ladder, lr_tests, mediation)`` as three frames.  They are returned rather
    than attached to ``ladder.attrs`` because pandas compares ``attrs`` on ``concat`` and
    a DataFrame-valued attr makes that comparison raise.
    """
    blocks = {
        "M0 deliverability": DELIVERABILITY,
        "M1 +sizing": DELIVERABILITY + SIZING,
        "M2 +trajectory": DELIVERABILITY + SIZING + TRAJECTORY,
        "M3 +gas": DELIVERABILITY + SIZING + TRAJECTORY + GAS,
        "M1g deliverability+sizing+gas": DELIVERABILITY + SIZING + GAS,
    }
    all_terms = sorted({t for ts in blocks.values() for t in ts})
    d = df.dropna(subset=["t_resid", EVENT_COL, *all_terms]).copy()

    rows, fits = [], {}
    for name, terms in blocks.items():
        fit, _ = fit_weibull_ph(d, tuple(terms))
        fits[name] = fit
        rows.append({
            "label": label, "model": name, "n": fit.n, "n_events": fit.n_events,
            "n_params": fit.n_params, "loglik": fit.loglik, "aic": fit.aic,
            "beta_shape": fit.beta_shape, "eta_scale": fit.eta_scale,
            "converged": fit.converged,
            **{f"coef_{t}": fit.coefs.get(t, np.nan) for t in all_terms},
            **{f"HR_{t}": fit.hazard_ratio(t) for t in all_terms},
        })
    tab = pd.DataFrame(rows)

    # Nested LR tests + the mediation attenuation.
    tests = []
    for small, big in [("M0 deliverability", "M1 +sizing"),
                       ("M1 +sizing", "M2 +trajectory"),
                       ("M2 +trajectory", "M3 +gas"),
                       ("M1 +sizing", "M1g deliverability+sizing+gas")]:
        r = lr_test(fits[small], fits[big])
        tests.append({"label": label, "comparison": f"{small} -> {big}", **r})

    c1 = fits["M1 +sizing"].coefs.get("log_qnom", np.nan)
    c3 = fits["M1g deliverability+sizing+gas"].coefs.get("log_qnom", np.nan)
    med = pd.DataFrame([{
        "label": label,
        "n": fits["M1 +sizing"].n, "n_events": fits["M1 +sizing"].n_events,
        "coef_log_qnom_before_gas": c1,
        "coef_log_qnom_after_gas": c3,
        "attenuation_share": float(1.0 - c3 / c1) if (np.isfinite(c1) and c1 != 0) else np.nan,
        "HR_per_e_fold_before": float(np.exp(c1)) if np.isfinite(c1) else np.nan,
        "HR_per_e_fold_after": float(np.exp(c3)) if np.isfinite(c3) else np.nan,
    }])
    return tab, pd.DataFrame(tests), med


def sizing_rmst_cost(
    df: pd.DataFrame,
    *,
    label: str = "",
    terms: tuple[str, ...] = DELIVERABILITY + SIZING,
    size_multiple: float = 2.0,
    horizon: float = RMST_HORIZON,
    within_well_hr: float | None = None,
) -> pd.DataFrame:
    """What does upsizing the pump cost in operating life — and buy in rate?

    This is the deliverable.  RMST is measured **from the landmark**, so it is residual
    life for a run that has already survived its first 90 operating days; the infant
    band is a separate question this design cannot speak to.

    Converts the fitted ``log_qnom`` coefficient into days of lost RMST for a
    ``size_multiple``× larger pump, holding the deliverability block at its mean, and
    pairs it with the **observed** liquid-rate difference between comparable runs so the
    trade is expressed the way the decision is actually made: *days of pump life per
    m³/d of extra liquid*.

    ``within_well_hr`` optionally re-prices the same trade using the within-well hazard
    ratio instead of the between-well one.  Those differ materially here (2.01 vs 1.46),
    and the within-well number is the less confounded of the two.
    """
    fit, d = fit_weibull_ph(df, terms, standardize=False)
    if not np.isfinite(fit.loglik):
        return pd.DataFrame()

    # Reference = the AVERAGE run in the frame.  Evaluating at the zero vector instead
    # would put the reference at Qnom = 1 m³/d and inflate baseline RMST by ~50 %.
    X = d[list(terms)].to_numpy(float)
    lin_ref = float(X.mean(axis=0) @ np.array([fit.coefs[t] for t in terms]))
    eta_ref = fit.eta_scale * float(np.exp(-lin_ref / fit.beta_shape))

    # Rate gain from upsizing: elasticity of realised liquid rate w.r.t. nameplate,
    # fitted on the same rows.  A quartile difference would overstate it — the top
    # quartile is far more than `size_multiple` times the bottom.
    e = d.dropna(subset=["ql_level", "qnom"])
    e = e[(e["ql_level"] > 0) & (e["qnom"] > 0)]
    dq = np.nan
    if len(e) >= 30:
        el = float(np.polyfit(np.log(e["qnom"]), np.log(e["ql_level"]), 1)[0])
        ql_ref = float(e["ql_level"].median())
        dq = ql_ref * (size_multiple ** el - 1.0)

    rows = []
    for src, coef in [("between-well (fitted)", fit.coefs.get("log_qnom", np.nan)),
                      ("within-well (stratified Cox)",
                       np.log(within_well_hr) if within_well_hr else np.nan)]:
        if not np.isfinite(coef):
            continue
        # PH shifts the scale: eta_eff = eta · exp(−Δlin/β).
        base = rmst(eta_ref, fit.beta_shape, horizon)
        dlin = coef * np.log(size_multiple)
        up = rmst(eta_ref * float(np.exp(-dlin / fit.beta_shape)), fit.beta_shape, horizon)
        rows.append({
            "label": label, "source": src, "n": fit.n, "n_events": fit.n_events,
            "size_multiple": size_multiple,
            "HR_per_size_multiple": float(np.exp(dlin)),
            "rmst_ref_days": base, "rmst_upsized_days": up,
            "rmst_lost_days": base - up,
            "rmst_lost_pct": (base - up) / base if base else np.nan,
            "ql_elasticity": el if len(e) >= 30 else np.nan,
            "observed_dql_m3d": dq,
            "days_lost_per_m3d": (base - up) / dq if (np.isfinite(dq) and dq > 0) else np.nan,
            "extra_m3_liquid_over_rmst": dq * up if np.isfinite(dq) else np.nan,
            "mrl_ref_days": mean_residual_life(eta_ref, fit.beta_shape),
        })
    return pd.DataFrame(rows)


def rmst_vs_km(df: pd.DataFrame, *, label: str = "") -> pd.DataFrame:
    """Parametric RMST against the Kaplan-Meier control — the mandatory sanity gate.

    A Weibull RMST that disagrees with KM by more than a few percent means the
    parametric form is wrong; the covariate story is then not worth reading.
    """
    fit, d = fit_weibull_ph(df, (), standardize=False)     # intercept-only baseline
    km = km_control(df)
    w = rmst(fit.eta_scale, fit.beta_shape, km["horizon"])
    return pd.DataFrame([{
        "label": label, "n": fit.n, "n_events": fit.n_events,
        "beta_shape": fit.beta_shape, "eta_scale": fit.eta_scale,
        "weibull_rmst": w, "km_rmst": km["km_rmst"], "horizon": km["horizon"],
        "abs_gap_days": abs(w - km["km_rmst"]),
        "rel_gap": abs(w - km["km_rmst"]) / km["km_rmst"] if km["km_rmst"] else np.nan,
        "weibull_mrl": mean_residual_life(fit.eta_scale, fit.beta_shape),
        "km_median": km["km_median"],
        "flag": "OK" if abs(w - km["km_rmst"]) / max(km["km_rmst"], 1e-9) < 0.05
                else "PARAMETRIC FORM SUSPECT",
    }])


def run(
    as_of: pd.Timestamp | None = None,
    *,
    landmark_op_days: int = LANDMARK_OP_DAYS,
    fields: tuple[str, ...] = ("Ya", "Vt", "Mc"),
    out_dir: Path | None = None,
) -> dict:
    """End-to-end: build the frame, run the ladder, write tables and figures."""
    out = out_dir or results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"

    df, cov = build_landmark_frame(as_of=as_of, landmark_op_days=landmark_op_days)
    df.to_csv(tables / "landmark_frame.csv", index=False, encoding="utf-8-sig")
    cov.as_frame().to_csv(tables / "coverage.csv", index=False, encoding="utf-8-sig")

    # -- does the landmark throw away the effect we are trying to measure? -----
    audit = landmark_selection_audit(df, cov.short_runs)
    audit.to_csv(tables / "landmark_selection_audit.csv", index=False, encoding="utf-8-sig")

    # -- honesty gates on β ---------------------------------------------------
    # (a) is the PVT set consistent with the fleet's own (P_b, GOR)?  Reported per field
    #     because it fails differently by field (Ya 0.76 vs Za 0.24) — which is *why* β
    #     is anchored rather than raw Standing.
    gate_rows = [{"scope": "pooled",
                  **FG.validate_pvt_against_bubble_point(df["p_bubble"], df["glf_level"])}]
    for f, g in df.groupby("field"):
        if len(g) >= 20:
            gate_rows.append({"scope": f, **FG.validate_pvt_against_bubble_point(
                g["p_bubble"], g["glf_level"])})
    pd.DataFrame(gate_rows).to_csv(tables / "pvt_gate.csv", index=False, encoding="utf-8-sig")

    # (b) does any β-based conclusion survive the PVT sweep?  β is consumed as a ranking
    #     covariate, so rank stability against the default set is the property that must
    #     hold; anything below ~0.99 invalidates the "PVT-free" claim.
    sens = _pvt_rank_sensitivity(df)
    sens.to_csv(tables / "pvt_sensitivity.csv", index=False, encoding="utf-8-sig")

    # descriptive
    q = sizing_quartile_table(df)
    q.to_csv(tables / "sizing_quartiles.csv", index=False, encoding="utf-8-sig")

    # -- the ladder, pooled and per field -------------------------------------
    ladders, lrs, meds, cvs, wells, kms, costs = [], [], [], [], [], [], []
    scopes = [("pooled", df)] + [(f, df[df["field"] == f]) for f in fields]
    for name, g in scopes:
        if len(g) < 60 or g[EVENT_COL].sum() < 15:
            continue
        tab, lr, med = mediation_sequence(g, label=name)
        ladders.append(tab)
        lrs.append(lr)
        meds.append(med)

        # CV: does trajectory beat level, out of sample?
        for cname, terms in [
            ("deliverability", DELIVERABILITY),
            ("+sizing", DELIVERABILITY + SIZING),
            ("+kpod level only", DELIVERABILITY + SIZING + ("kpod_level",)),
            ("+kpod level & slope", DELIVERABILITY + SIZING + ("kpod_level", "kpod_slope")),
            ("+trajectory", DELIVERABILITY + SIZING + TRAJECTORY),
            ("+trajectory+gas", DELIVERABILITY + SIZING + TRAJECTORY + GAS),
        ]:
            cvs.append({"label": name, "model": cname, **cv_loglik(g, tuple(terms))})

        ww = [within_well_contrast(g, min_ratio=r) for r in (1.25, 1.5)]
        for w in ww:
            wells.append({"label": name, **w})
        kms.append(rmst_vs_km(g, label=name))
        # Price the trade with both the between-well and the (less confounded) within-well
        # coefficient.  Only take the within-well HR when its CI is informative — the Mc
        # stratified fit returns HR 27 [0.68, 1112] on 17 events, which is not a number.
        hr = ww[0].get("HR_log_qnom")
        lo, hi = ww[0].get("HR_log_qnom_lo"), ww[0].get("HR_log_qnom_hi")
        usable = (hr is not None and np.isfinite(hr) and lo and hi and (hi / lo) < 20.0)
        costs.append(sizing_rmst_cost(g, label=name, within_well_hr=hr if usable else None))

    _write(tables / "model_ladder.csv", ladders)
    _write(tables / "lr_tests.csv", lrs)
    _write(tables / "mediation.csv", meds)
    pd.DataFrame(cvs).to_csv(tables / "cv_loglik.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wells).to_csv(tables / "within_well.csv", index=False, encoding="utf-8-sig")
    _write(tables / "km_control.csv", kms)
    _write(tables / "sizing_rmst_cost.csv", costs)

    _figures(df, figures)
    return {"frame": df, "coverage": cov, "quartiles": q, "audit": audit,
            "pvt_gate": pd.DataFrame(gate_rows), "pvt_sensitivity": sens,
            "ladder": pd.concat(ladders) if ladders else pd.DataFrame(),
            "lr_tests": pd.concat(lrs) if lrs else pd.DataFrame(),
            "mediation": pd.concat(meds) if meds else pd.DataFrame(),
            "cv": pd.DataFrame(cvs), "within_well": pd.DataFrame(wells),
            "km": pd.concat(kms) if kms else pd.DataFrame(),
            "cost": pd.concat(costs) if costs else pd.DataFrame(), "out_dir": out}


def _pvt_rank_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Rank stability of β across the PVT assumption grid.

    Evaluated at each run's **window-mean** conditions rather than by re-integrating the
    daily series under every PVT set — 27 grid points × 600k daily rows is not worth it,
    and the daily-level check has already been run once directly (Spearman 1.0000 against
    the default set over 594 861 rows).  This is the cheap standing regression test: if a
    future data or code change breaks the anchoring, ``rank_corr_vs_default`` drops below
    1 and the "β is PVT-free" claim in the docstring stops being true.
    """
    d = df.dropna(subset=["pint_level", "glf_level", "wct_level"])
    if d.empty:
        return pd.DataFrame()
    pb = d["p_bubble"].to_numpy()
    ref = FG.free_gas_fraction(d["pint_level"], d["glf_level"], d["wct_level"],
                               p_bubble_atm=pb)
    rows = []
    for p in FG.pvt_sensitivity_grid():
        b = FG.free_gas_fraction(d["pint_level"], d["glf_level"], d["wct_level"],
                                 p_bubble_atm=pb, pvt=p)
        ok = np.isfinite(ref) & np.isfinite(b)
        rows.append({
            "api": p.api_gravity, "temp_c": p.temp_c, "gas_gravity": p.gas_gravity,
            "beta_median": float(np.nanmedian(b)),
            "rank_corr_vs_default": float(pd.Series(ref[ok]).corr(
                pd.Series(b[ok]), method="spearman")),
        })
    return pd.DataFrame(rows)


def _write(path: Path, frames: list) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def _figures(df: pd.DataFrame, figdir: Path) -> None:
    """Kpod level vs slope against pump size, and β against intake pressure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    d = df.dropna(subset=["qnom", "kpod_level", "kpod_slope"])
    axes[0].scatter(d["qnom"], d["kpod_level"], s=9, alpha=.45)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Qnom (m³/d, log)", ylabel="Kpod level (fixed window)",
                title="Level carries no size signal")
    axes[1].scatter(d["qnom"], d["kpod_slope"], s=9, alpha=.45, color="C3")
    axes[1].set_xscale("log")
    axes[1].axhline(0, lw=.8, color="k")
    axes[1].set(xlabel="Qnom (m³/d, log)", ylabel="Kpod slope (per 100 op-days)",
                title="Trajectory is where sizing shows up")
    g = df.dropna(subset=["pint_level", "beta_mean"])
    sc = axes[2].scatter(g["pint_level"], g["beta_mean"], s=9, alpha=.45,
                         c=np.log10(g["qnom"].clip(lower=1)), cmap="viridis")
    axes[2].set(xlabel="P_intake (atm)", ylabel="β free gas at intake",
                title="The mediator")
    plt.colorbar(sc, ax=axes[2], label="log₁₀ Qnom")
    fig.tight_layout()
    fig.savefig(figdir / "sizing_level_vs_slope.png", dpi=150)
    plt.close(fig)


__all__ = [
    "SLUG", "LANDMARK_OP_DAYS", "LANDMARK_SENSITIVITY", "RMST_HORIZON",
    "SIZING", "DELIVERABILITY", "TRAJECTORY", "GAS",
    "LandmarkCoverage", "PHFit",
    "landmark_window_features", "build_landmark_frame",
    "fit_weibull_ph", "rmst", "mean_residual_life", "rmst_at",
    "lr_test", "cv_loglik", "within_well_contrast", "km_control",
    "sizing_quartile_table", "mediation_sequence", "run",
]
