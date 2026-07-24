"""Vt TTF ~ ESP operating-covariate relations (workflow plan
``agents/analyses/vt_ttf_covariates_plan.md``).

Goal: **simple mathematical relations for how time-to-failure changes with ESP
operating parameters** on Vt (sour + nonsour), per h2s_class × contractor_group,
delivered as a formula table of *life multipliers* — not a forecast layer.

This runs on the **reworked Свод** (2026-07-22: running/censored runs added, brine
phantoms filtered — see ``svod_build_running_wells_prompt.md`` /
``svod_build_wellidentity_prompt.md``).  The censored exposure this analysis leans
on now comes from Свод natively, so the Big open-row patch is gone
(``esp_population.load_big_runs`` keeps closed-only).

Design decisions baked in from the plan and prior memory:

* **contractor / h2s flag are strata, never β** (service-quality label; 3× sour/nonsour
  life gap — ``project_vt_weibull_scan``).
* **h2s-continuous is dropped**: the mart carries no ``h2s_proxy_source='measured'``
  rows, so the "sour-only measured-h2s" covariate is empty (resolved §8.1).
* **Kpod** carried as plain + freq-adjusted, each in t0 / guarded-run / frac-days
  form (``kpod_features``); tail guard 30 d, non-negotiable and differential.
* **Clock** ``t_cal`` primary (Vt refit decision, ``project_time_scales``), ``t_mix``
  sensitivity.  Never the legacy mixed ``tte``.
* **Plain PH for β** — no X·log t interaction (``project_cox_glf_null``).
* honesty gates: replication on Ya + Mc(2024+), quantile support overlap, n_failures
  asserted, complete-case KM covered-vs-uncovered.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_optime, esp_population as popmod
from analysis.workflows.production_risk import kpod_features as K
from analysis.workflows.production_risk import t0_covariates
from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox

FIELD = "Vt"
CLOCK_PRIMARY = "t_cal"
CLOCK_SENS = "t_mix"
EVENT_COL = "event"
CLUSTER = "well_key"

#: Figure language: "en" (default, used by the pipeline) or "ru" (slide deck).
#: Set ``vt_ttf_covariates.FIG_LANG = "ru"`` around a render pass; ``_tt(en, ru)`` picks.
FIG_LANG = "en"


def _tt(en: str, ru: str) -> str:
    """Return the ``ru`` string when ``FIG_LANG=='ru'``, else ``en`` — for figure labels."""
    return ru if FIG_LANG == "ru" else en


#: Kpod descriptive bins (plan §3.1).
KPOD_BINS = [0.0, 0.5, 0.7, 0.85, 1.0, 1.15, np.inf]
KPOD_BIN_LABELS = ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0", "1.0-1.15", ">1.15"]

#: Running-frequency descriptive bins (Hz); 50 Hz is the design point, >55 over-speed.
FREQ_BINS = [0.0, 45.0, 48.0, 50.0, 52.5, 55.0, np.inf]
FREQ_BIN_LABELS = ["<45", "45-48", "48-50", "50-52.5", "52.5-55", ">55"]

#: Hinge knots (plan §3.4 priors).
HINGE_LEFT_KNOT = 0.85
HINGE_RIGHT_KNOT = 1.0

#: Shipped Vt Weibull baselines for the PH route (plan §5.1).
ESP_MODELS_REF = "results/esp_survival_vba_models/2026-07-20-vt-refit/esp_models.csv"

#: A stratum needs ~10 events per parameter to carry its own β; below this it gets
#: descriptive treatment only (plan §4 diagnostics).
MIN_EVENTS_PER_PARAM = 10

#: Effective sour boundary from the DATA, not the legacy 10 mg/l.  Vt_sour H2S proxy
#: (well/pad medians) runs 122–1272 mg/l (q05 122); Vt_nonsour is ≤1.5 mg/l at q90.
#: The Свод «Кислый/Некислый» label (which actually sets the strata) matches proxy>~120,
#: so 10 mg/l is far below any sour value and does not drive this analysis' strata.
SOUR_PROXY_THRESHOLD_MG_L = 120.0
#: h2s_proxy sources that carry real within-stratum concentration signal (NOT the
#: uninformative field-fallback the plan meant to exclude).
H2S_INFORMATIVE_SOURCES = ("well_median", "pad_median")


# ===========================================================================
# Frame assembly
# ===========================================================================

@dataclass
class FrameCoverage:
    t0: pd.DataFrame
    kpod: pd.DataFrame
    kpod_g60: pd.DataFrame
    kvch: pd.DataFrame
    by_stratum: pd.DataFrame
    notes: list[str] = dc_field(default_factory=list)


def _load_kvch(pop: pd.DataFrame, db_path: Path | None = None) -> pd.Series:
    """Per-run mean КВЧ (mechanical impurities, mg/l): in-run, else 180 d pre-install.

    Low-coverage covariate (plan §1/§8.1) — kept as descriptive/sensitivity only.
    Joined on the casefolded well_key over the run window; ``NaN`` where absent.
    """
    import sqlite3
    from analysis.paths import WAREHOUSE_DIR, resolve_lab_db_path

    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        con.execute(f"ATTACH DATABASE '{resolve_lab_db_path()}' AS lab_db")
        ls = pd.read_sql(
            "SELECT well_key, sample_date, mechanical_impurities_mg_l AS kvch "
            "FROM lab_db.lab_samples WHERE mechanical_impurities_mg_l IS NOT NULL",
            con,
        )
    finally:
        con.close()
    ls["sample_date"] = pd.to_datetime(ls["sample_date"], errors="coerce")
    ls = ls.dropna(subset=["sample_date"])
    by_well = {k: g.sort_values("sample_date") for k, g in ls.groupby("well_key", sort=False)}

    out = pd.Series(np.nan, index=pop.index, dtype=float)
    for idx, run in pop.iterrows():
        g = by_well.get(str(run["code"]).casefold())
        if g is None or pd.isna(run["install"]):
            continue
        install = pd.Timestamp(run["install"])
        stop = pd.Timestamp(run["end"]) if pd.notna(run["end"]) else pd.Timestamp(C.SVOD_OPEN_ASOF)
        inrun = g[(g["sample_date"] >= install) & (g["sample_date"] <= stop)]
        if not inrun.empty:
            out.loc[idx] = float(inrun["kvch"].mean())
            continue
        pre = g[(g["sample_date"] >= install - pd.Timedelta(days=180)) & (g["sample_date"] < install)]
        if not pre.empty:
            out.loc[idx] = float(pre["kvch"].mean())
    return out


def _attach_h2s_proxy(pop: pd.DataFrame, db_path: Path | None = None) -> pd.DataFrame:
    """Join per-run H2S concentration proxy (mg/l) + source from the mart.

    Kept as a continuous covariate for the **within-sour** test only (the sour/nonsour
    flag already carries the threshold effect; concentration adds a within-stratum
    gradient).  Only ``well_median``/``pad_median`` rows carry real within-stratum
    signal — ``missing`` (field-fallback) is left NaN, not zero."""
    import sqlite3
    from analysis.paths import WAREHOUSE_DIR

    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        m = pd.read_sql(
            "SELECT well_key, install_date, h2s_proxy_mg_l, h2s_proxy_source "
            "FROM mart__weibull_input", con)
    finally:
        con.close()
    from analysis.workflows.production_risk import crosswalk
    m["code"] = m["well_key"].map(crosswalk.norm_well)
    m["install"] = pd.to_datetime(m["install_date"], errors="coerce")
    m = m.dropna(subset=["code", "install"]).sort_values("install")
    out = pop.sort_values("install")
    out = pd.merge_asof(
        out, m[["code", "install", "h2s_proxy_mg_l", "h2s_proxy_source"]],
        on="install", by="code", tolerance=pd.Timedelta(days=30), direction="nearest")
    informative = out["h2s_proxy_source"].isin(H2S_INFORMATIVE_SOURCES)
    h2s = pd.to_numeric(out["h2s_proxy_mg_l"], errors="coerce").where(informative)
    out["h2s_mg_l"] = h2s
    out["log_h2s"] = np.log(h2s.clip(lower=1))
    return out.reset_index(drop=True)


def _attach_cumulative_volume(pop: pd.DataFrame, as_of: pd.Timestamp,
                              db_path: Path | None = None) -> pd.Series:
    """Cumulative liquid volume pumped over each run to its observation time (kt = 10³ m³).

    ``t_vol`` is the **cumulative-throughput clock**: if failures are driven by total
    volume pumped (mechanical wear / work) rather than exposure time, survival is
    tightest on this axis and the rate covariates go null on it (they are what fill the
    volume budget).  Summed over ALL run days to the event/censor (not tail-guarded — a
    clock must run to the observed endpoint)."""
    import sqlite3
    from analysis.paths import WAREHOUSE_DIR
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        dm = pd.read_sql("SELECT well_key, dt, qliq FROM proc__daily_merged WHERE qliq > 0",
                         con, parse_dates=["dt"])
    finally:
        con.close()
    by = {k: g for k, g in dm.groupby("well_key")}
    out = pd.Series(np.nan, index=pop.index, dtype=float)
    for idx, r in pop.iterrows():
        g = by.get(str(r["code"]).casefold())
        if g is None or pd.isna(r["install"]):
            continue
        end = pd.Timestamp(r["end"]) if pd.notna(r["end"]) else as_of
        w = g[(g["dt"] >= pd.Timestamp(r["install"])) & (g["dt"] <= end)]
        if len(w):
            out.loc[idx] = float(w["qliq"].sum()) / 1000.0
    return out


def build_frame(
    as_of: pd.Timestamp | None = None,
    *,
    with_kvch: bool = True,
) -> tuple[pd.DataFrame, FrameCoverage]:
    """Assemble the full covariate frame for ALL fields (Vt + replication targets).

    Returns ``(df, coverage)``.  Covariates: t0 rate window (ql/qg/glf/qo/wcut),
    Kpod window family (plain + freq, t0 / guarded-run / frac-days) at guard 30 and
    a 60-day sensitivity copy, and (optionally) КВЧ.  Clocks ``t_cal`` + ``t_mix``.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(C.SVOD_OPEN_ASOF)
    notes: list[str] = []

    pop = popmod.build(as_of=as_of)
    pop = popmod.add_time_scales(pop, as_of)
    pop = esp_optime.measure(pop, as_of=as_of)          # attaches t_op / t_mix
    pop["well_key"] = pop["code"].astype(str).str.casefold()

    # -- t0 rate covariates (Ql, Qg, GLF, Qo, wcut) --------------------------
    pop, t0cov = t0_covariates.build(pop)
    pop["log_ql"] = np.log(pop["ql"].where(pop["ql"] > 0))
    pop["log1p_qg"] = np.log1p(pop["qg"].clip(lower=0))
    t0_by = _coverage_by_stratum(pop, "ql", as_of)

    # -- Kpod family: guard 30 (primary) + guard 60 (sensitivity) ------------
    pop = K.resolve_qnom(pop)
    pop, kcov = K.build_kpod_family(pop, as_of=as_of, guard_days=30)
    pop, kcov60 = K.build_kpod_family(pop, as_of=as_of, guard_days=60, suffix="_g60")
    kpod_by = _coverage_by_stratum(pop, "kpod_run", as_of)
    kpod60_by = _coverage_by_stratum(pop, "kpod_run_g60", as_of)

    # -- H2S concentration proxy (for the within-sour test, §H) ---------------
    pop = _attach_h2s_proxy(pop)

    # -- cumulative-volume clock (throughput vs exposure mechanism test) ------
    pop["t_vol"] = _attach_cumulative_volume(pop, as_of)

    # -- КВЧ (low coverage; descriptive only) --------------------------------
    if with_kvch:
        pop["kvch"] = _load_kvch(pop)
        pop["log_kvch"] = np.log1p(pop["kvch"].clip(lower=0))
        kvch_by = _coverage_by_stratum(pop, "kvch", as_of)
    else:
        pop["kvch"] = np.nan
        pop["log_kvch"] = np.nan
        kvch_by = pd.DataFrame()

    notes.append(
        "h2s-continuous kept for the WITHIN-SOUR test only (well_median/pad_median rows "
        "carry real within-stratum signal; the plan's exclusion was for field-fallback, "
        "not these). The sour/nonsour STRATA come from the Свод «Кислый/Некислый» label, "
        f"which matches proxy>~{SOUR_PROXY_THRESHOLD_MG_L:.0f} mg/l; the legacy 10 mg/l "
        "threshold is far below any Vt_sour value (sour q05≈122, nonsour q90≈1.5 mg/l) and "
        "does not drive these strata."
    )
    notes.append(
        f"Qnom source mix (all fields): {pop['kpod_qnom_source'].value_counts().to_dict()}; "
        "type_parse tier is small — U-shape shown robust to dropping it in §3."
    )

    by_stratum = _frame_coverage_by_stratum(pop, as_of)
    cov = FrameCoverage(
        t0=t0cov.as_frame(), kpod=kcov.as_frame(), kpod_g60=kcov60.as_frame(),
        kvch=kvch_by, by_stratum=by_stratum, notes=notes,
    )
    return pop, cov


def _coverage_by_stratum(pop: pd.DataFrame, col: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-stratum present/absent count for ``col``, split by event/censor."""
    g = pop[pop["field"] == FIELD].copy()
    g["_present"] = g[col].notna()
    rows = []
    for (h2s, con_grp), sub in g.groupby(["h2s_class", "contractor_group"]):
        for ev, lab in ((1, "event"), (0, "censor")):
            s = sub[sub["event"] == ev]
            rows.append({
                "h2s_class": h2s, "contractor_group": con_grp, "kind": lab,
                "n": len(s), "n_present": int(s["_present"].sum()),
                "share_present": round(s["_present"].mean(), 3) if len(s) else np.nan,
            })
    return pd.DataFrame(rows)


def _frame_coverage_by_stratum(pop: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    g = pop[pop["field"] == FIELD]
    rows = []
    for (h2s, con_grp), sub in g.groupby(["h2s_class", "contractor_group"]):
        rows.append({
            "stratum": f"{FIELD}_{h2s}_{con_grp}",
            "n_runs": len(sub), "n_events": int(sub["event"].sum()),
            "n_censored": int((sub["event"] == 0).sum()),
            "n_running": int(sub["end"].isna().sum()),
            "kpod_run": int(sub["kpod_run"].notna().sum()),
            "ql_t0": int(sub["ql"].notna().sum()),
            "kvch": int(sub["kvch"].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("n_events", ascending=False)


# ===========================================================================
# Kpod U-shape protocol (plan §3)
# ===========================================================================

def km_rmst_mrl(durations: np.ndarray, events: np.ndarray, horizon: float | None = None) -> tuple[float, float]:
    """(RMST(0) to horizon, MRL(0)=mean via KM tail-extended) — the reporting
    standard (``feedback_report_rmst_mrl``).  MRL(0) here is the KM-integrated mean
    to the last observed time (a lower bound when the tail is censored)."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    dur = np.asarray(durations, dtype=float)
    ev = np.asarray(events, dtype=float)
    mask = np.isfinite(dur) & (dur > 0)
    dur, ev = dur[mask], ev[mask]
    if len(dur) < 5 or ev.sum() < 1:
        return np.nan, np.nan
    kmf = KaplanMeierFitter().fit(dur, ev)
    h = float(horizon) if horizon is not None else float(np.quantile(dur, 0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmst = float(restricted_mean_survival_time(kmf, t=h))
        mrl = float(restricted_mean_survival_time(kmf, t=float(dur.max())))
    return round(rmst, 1), round(mrl, 1)


def binned_table(
    df: pd.DataFrame, col: str, bins: list[float], labels: list[str], *,
    clock: str = CLOCK_PRIMARY,
) -> pd.DataFrame:
    """Honest picture before any smoothing (plan §3.1): per bin of ``col`` — n runs,
    n events, censored share, KM RMST(0)/MRL(0), event rate per 1000 op-days."""
    d = df[df[col].notna() & df[clock].notna()].copy()
    d["_bin"] = pd.cut(d[col], bins=bins, labels=labels, right=False)
    rows = []
    for lab, sub in d.groupby("_bin", observed=True):
        rmst, mrl = km_rmst_mrl(sub[clock].to_numpy(), sub[EVENT_COL].to_numpy())
        exposure = float(sub[clock].sum())
        rows.append({
            "bin": lab, "col": col, "n_runs": len(sub), "n_events": int(sub[EVENT_COL].sum()),
            "censored_share": round((sub[EVENT_COL] == 0).mean(), 3),
            "km_rmst0": rmst, "km_mrl0": mrl,
            "rate_per_1000_opd": round(1000.0 * sub[EVENT_COL].sum() / exposure, 3) if exposure else np.nan,
        })
    return pd.DataFrame(rows)


def kpod_binned_table(df: pd.DataFrame, kpod_col: str, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    return binned_table(df, kpod_col, KPOD_BINS, KPOD_BIN_LABELS, clock=clock).rename(
        columns={"bin": "kpod_bin"})


def _prep_cox(df: pd.DataFrame, covariates: list[str], *, clock: str) -> pd.DataFrame:
    keep = list(dict.fromkeys(covariates + [EVENT_COL, clock, "stratum", CLUSTER]))
    work = df[keep].copy()
    work[clock] = pd.to_numeric(work[clock], errors="coerce")
    return work.dropna(subset=covariates + [clock, EVENT_COL])


def _zscore(s: pd.Series) -> tuple[pd.Series, float, float]:
    mu, sd = float(s.mean()), float(s.std(ddof=0))
    sd = sd if sd > 0 else 1.0
    return (s - mu) / sd, mu, sd


def fit_quadratic_cox(df: pd.DataFrame, kpod_col: str, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """β1·K + β2·K² stratified Cox — test β2 > 0 (convexity / U-shape)."""
    d = df.copy()
    d["_K"] = d[kpod_col]
    d["_K2"] = d[kpod_col] ** 2
    work = _prep_cox(d, ["_K", "_K2"], clock=clock)
    res = fit_cause_specific_cox(
        work, covariates=["_K", "_K2"], event_col=EVENT_COL, duration_col=clock,
        strata=["stratum"], cluster_col=CLUSTER, min_events=MIN_EVENTS_PER_PARAM * 2,
    )
    if not res.success:
        return pd.DataFrame([{"kpod_col": kpod_col, "note": res.message}])
    s = res.summary.copy()
    s.insert(0, "kpod_col", kpod_col)
    s["n"], s["n_events"] = res.n, res.n_events
    return s


def _spline_basis(x: pd.Series, df_spline: int = 4):
    """Natural cubic-spline (restricted) basis via patsy ``cr``, centered so the
    linear predictor is a log-HR relative to the sample mean Kpod."""
    from patsy import dmatrix

    dm = dmatrix(f"cr(x, df={df_spline}) - 1", {"x": x.to_numpy()}, return_type="dataframe")
    dm.columns = [f"sp{i}" for i in range(dm.shape[1])]
    dm.index = x.index
    return dm - dm.mean(axis=0)


def fit_spline_cox(
    df: pd.DataFrame, kpod_col: str, *, clock: str = CLOCK_PRIMARY, df_spline: int = 4,
    grid: np.ndarray | None = None, permutation: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Penalized-ish spline Cox (df 3-4): log-HR vs Kpod with a delta-method CI band.

    Returns ``(curve_df, meta)`` where ``curve_df`` has kpod / log_hr / lo / hi over
    a grid and ``meta`` carries the permutation-null p-value for the deviance gain."""
    from lifelines import CoxPHFitter

    d = df[df[kpod_col].notna()].copy()
    basis = _spline_basis(d[kpod_col], df_spline)
    cols = list(basis.columns)
    d = pd.concat([d, basis], axis=1)
    work = _prep_cox(d, cols, clock=clock)
    if int(work[EVENT_COL].sum()) < MIN_EVENTS_PER_PARAM * len(cols):
        return pd.DataFrame(), {"note": "too few events for spline", "kpod_col": kpod_col}

    def _fit(frame):
        cph = CoxPHFitter(penalizer=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(frame[cols + [clock, EVENT_COL, "stratum"]], duration_col=clock,
                    event_col=EVENT_COL, strata=["stratum"])
        return cph

    cph = _fit(work)
    beta = cph.params_.reindex(cols).to_numpy()
    cov = cph.variance_matrix_.reindex(index=cols, columns=cols).to_numpy()

    if grid is None:
        grid = np.linspace(float(np.quantile(d[kpod_col], 0.02)),
                            float(np.quantile(d[kpod_col], 0.98)), 60)
    from patsy import dmatrix, build_design_matrices
    di = dmatrix(f"cr(x, df={df_spline}) - 1", {"x": d[kpod_col].to_numpy()}, return_type="dataframe")
    design_info = di.design_info
    Bg = np.asarray(build_design_matrices([design_info], {"x": grid})[0])
    Bg = Bg - di.to_numpy().mean(axis=0)  # same centering as the fit basis
    log_hr = Bg @ beta
    var = np.einsum("ij,jk,ik->i", Bg, cov, Bg)
    se = np.sqrt(np.clip(var, 0, None))
    curve = pd.DataFrame({"kpod": grid, "log_hr": log_hr,
                          "lo": log_hr - 1.96 * se, "hi": log_hr + 1.96 * se})

    meta = {"kpod_col": kpod_col, "df_spline": df_spline, "n": res_n(work),
            "n_events": int(work[EVENT_COL].sum())}

    # Permutation null: deviance gain of the spline vs an intercept-only stratified
    # fit, refit on permuted Kpod (plan §3 guard: the U-shape is overfit-prone).
    # Skipped inside bootstrap (permutation=False) — otherwise O(boot × perm) fits.
    if permutation:
        ll_full = cph.log_likelihood_
        rng = np.random.default_rng(0)
        perm_gains = []
        base_ll = _null_stratified_ll(work, clock)
        for _ in range(200):
            wperm = work.copy()
            perm_vals = rng.permutation(d.loc[work.index, kpod_col].to_numpy())
            pb = _spline_basis(pd.Series(perm_vals, index=work.index), df_spline)
            wperm[cols] = pb.to_numpy()
            try:
                llp = _fit(wperm).log_likelihood_
            except Exception:
                continue
            perm_gains.append(llp - base_ll)
        obs_gain = ll_full - base_ll
        meta["deviance_gain"] = round(2 * obs_gain, 2)
        meta["p_permutation"] = round(
            float(np.mean([g >= obs_gain for g in perm_gains])) if perm_gains else np.nan, 4)
    return curve, meta


def res_n(work: pd.DataFrame) -> int:
    return int(len(work))


def _null_stratified_ll(work: pd.DataFrame, clock: str) -> float:
    """Log-lik of the covariate-free stratified Cox (baseline for deviance gains)."""
    from lifelines import CoxPHFitter
    tmp = work[[clock, EVENT_COL, "stratum"]].copy()
    tmp["_z"] = 0.0
    # A zero column is dropped by lifelines; use a tiny jitter-free constant trick:
    # fit an unpenalized model with a single near-constant is singular, so instead
    # compute the partial log-lik at beta=0 via a 1-covariate fit with penalizer.
    cph = CoxPHFitter(penalizer=1e6)
    tmp["_z"] = np.linspace(-1e-6, 1e-6, len(tmp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(tmp, duration_col=clock, event_col=EVENT_COL, strata=["stratum"])
    return cph.log_likelihood_


def fit_hinge_cox(
    df: pd.DataFrame, kpod_col: str, *, clock: str = CLOCK_PRIMARY,
    left_knot: float = HINGE_LEFT_KNOT, right_knot: float = HINGE_RIGHT_KNOT,
) -> pd.DataFrame:
    """Hinge decomposition (the deliverable form, plan §3.4):
    ``left = (knot_L − K)₊``, ``right = (K − knot_R)₊`` → two slopes with CIs.

    Coefficients are per 0.1 of Kpod (natural unit), so a row reads directly as
    "left penalty per 0.1 of underloading" / "right penalty per 0.1 above nominal"."""
    d = df.copy()
    d["_left"] = np.clip(left_knot - d[kpod_col], 0, None) * 10.0   # per 0.1 Kpod
    d["_right"] = np.clip(d[kpod_col] - right_knot, 0, None) * 10.0
    work = _prep_cox(d, ["_left", "_right"], clock=clock)
    res = fit_cause_specific_cox(
        work, covariates=["_left", "_right"], event_col=EVENT_COL, duration_col=clock,
        strata=["stratum"], cluster_col=CLUSTER, min_events=MIN_EVENTS_PER_PARAM * 2,
    )
    if not res.success:
        return pd.DataFrame([{"kpod_col": kpod_col, "note": res.message}])
    s = res.summary.copy()
    s["covariate"] = s["covariate"].map({"_left": "underload_per_0.1", "_right": "overload_per_0.1"})
    s.insert(0, "kpod_col", kpod_col)
    s["left_knot"], s["right_knot"] = left_knot, right_knot
    s["n"], s["n_events"] = res.n, res.n_events
    return s


def exposure_share_contest(
    df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY,
    mean_cols=("kpod_run", "kpod_freq_run"),
    share_cols=("frac_days_kpod_below", "frac_days_kpod_above"),
) -> pd.DataFrame:
    """Mean-based vs exposure-share Kpod forms in the same stratified Cox (plan §3.5):
    compare partial log-lik and concordance / coefficient stability."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    candidates = {
        "kpod_run_mean": ["kpod_run"],
        "kpod_freq_run_mean": ["kpod_freq_run"],
        "exposure_shares": list(share_cols),
        "mean_plus_shares": ["kpod_run", *share_cols],
    }
    rows = []
    for name, cols in candidates.items():
        work = _prep_cox(df, cols, clock=clock)
        if int(work[EVENT_COL].sum()) < MIN_EVENTS_PER_PARAM * len(cols):
            rows.append({"model": name, "note": "insufficient events"})
            continue
        cph = CoxPHFitter(penalizer=0.01)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(work[cols + [clock, EVENT_COL, "stratum"]], duration_col=clock,
                        event_col=EVENT_COL, strata=["stratum"])
        except Exception as exc:
            rows.append({"model": name, "note": f"fit failed: {exc}"})
            continue
        rows.append({
            "model": name, "n": len(work), "n_events": int(work[EVENT_COL].sum()),
            "log_lik": round(cph.log_likelihood_, 2),
            "concordance": round(cph.concordance_index_, 4),
            "AIC_partial": round(cph.AIC_partial_, 2),
        })
    return pd.DataFrame(rows)


def risk_min_kpod(
    df: pd.DataFrame, kpod_col: str, *, clock: str = CLOCK_PRIMARY, n_boot: int = 300,
) -> dict:
    """Risk-minimizing Kpod (the spline's argmin log-HR) with a bootstrap CI (§3.6)."""
    curve, _ = fit_spline_cox(df, kpod_col, clock=clock)
    if curve.empty:
        return {"kpod_col": kpod_col, "note": "no spline"}
    kmin = float(curve.loc[curve["log_hr"].idxmin(), "kpod"])
    d = df[df[kpod_col].notna()].copy()
    rng = np.random.default_rng(1)
    mins = []
    for _ in range(n_boot):
        samp = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        try:
            cv, _ = fit_spline_cox(samp, kpod_col, clock=clock,
                                   grid=curve["kpod"].to_numpy(), permutation=False)
            if not cv.empty:
                mins.append(float(cv.loc[cv["log_hr"].idxmin(), "kpod"]))
        except Exception:
            continue
    lo, hi = (np.nanpercentile(mins, [2.5, 97.5]) if mins else (np.nan, np.nan))
    return {"kpod_col": kpod_col, "risk_min_kpod": round(kmin, 3),
            "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3), "n_boot": len(mins)}


# ===========================================================================
# Model stage (plan §4): stratified Cox, per-covariate natural + SD units
# ===========================================================================

#: Natural-unit deltas for the formula table (per this step of the raw covariate).
NATURAL_UNITS = {
    "ql": ("per +10 m3/d", 10.0),
    "qg": ("per +10 m3/d", 10.0),
    "kpod_run": ("per +0.1 Kpod", 0.1),
    "kpod_freq_run": ("per +0.1 Kpod", 0.1),
    "kpod_t0": ("per +0.1 Kpod", 0.1),
    "glf": ("per +0.1 GLF", 0.1),
    "kvch": ("per +10 mg/l", 10.0),
    "log_ql": ("per +1 log-unit", 1.0),
    "log1p_qg": ("per +1 log-unit", 1.0),
    "frac_days_kpod_below": ("per +10% days", 0.1),
    "frac_days_kpod_above": ("per +10% days", 0.1),
    "freq_run": ("per +1 Hz", 1.0),
    "freq_t0": ("per +1 Hz", 1.0),
    "frac_days_freq_over": ("per +10% days", 0.1),
}


def stratified_cox(
    df: pd.DataFrame, covariates: list[str], *, clock: str = CLOCK_PRIMARY,
    per_stratum: bool = False, min_events: int = MIN_EVENTS_PER_PARAM,
) -> pd.DataFrame:
    """Stratified Cox (strata = h2s_class x contractor_group via ``stratum``), global b.

    Reports HR per 1 SD (standardized) AND per natural unit -- the formula table needs
    natural units.  ``per_stratum`` refits within each stratum for a stability check.
    """
    frames = []
    groups = ([("global", df)] if not per_stratum
              else [(s, g) for s, g in df.groupby("stratum")])
    for label, g in groups:
        work = g.copy()
        # Global fit stratifies on h2s × contractor; a within-stratum fit has nothing
        # left to stratify on, so it runs against a single constant stratum.
        if label == "global":
            strata = ["stratum"]
        else:
            work["_one"] = "all"
            strata = ["_one"]
        stds = {}
        zcols = []
        for cov in covariates:
            if cov not in work.columns:
                continue
            z = f"z_{cov}"
            work[z], mu, sd = _zscore(work[cov])
            stds[z] = (cov, sd)
            zcols.append(z)
        res = fit_cause_specific_cox(
            work, covariates=zcols, event_col=EVENT_COL, duration_col=clock,
            strata=strata, cluster_col=CLUSTER, min_events=min_events * max(len(zcols), 1),
        )
        if not res.success:
            frames.append(pd.DataFrame([{"scope": label, "note": res.message,
                                         "n": res.n, "n_events": res.n_events}]))
            continue
        s = res.summary.copy()
        s["scope"] = label
        s["n"], s["n_events"] = res.n, res.n_events
        # Translate the standardized b back to natural units.
        raw, sd = zip(*[stds[z] for z in s["covariate"]])
        s["covariate_raw"] = raw
        s["beta_per_sd"] = s["coef"]
        s["hr_per_sd"] = s["hr"]
        beta_nat = s["coef"] / np.array(sd)                 # b per 1 raw unit
        s["beta_per_unit"] = beta_nat
        nat = [NATURAL_UNITS.get(r, ("per +1 unit", 1.0)) for r in raw]
        s["natural_unit"] = [n[0] for n in nat]
        step = np.array([n[1] for n in nat])
        s["hr_per_natural"] = np.exp(beta_nat * step)
        # 95% CI on the natural-unit HR (delta method on the z-scale se).
        se_nat = s["se"].to_numpy() / np.array(sd) * step
        s["hr_nat_ci_lo"] = np.exp(beta_nat * step - 1.96 * se_nat)
        s["hr_nat_ci_hi"] = np.exp(beta_nat * step + 1.96 * se_nat)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def cox_decomposition(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY,
                      kpod_col: str = "kpod_run") -> pd.DataFrame:
    """Kpod-only / Ql-only / joint (plan §3 guard): Kpod = Ql/Qnom is mechanically
    correlated with Ql, so if the joint fit kills both, the data cannot separate rate
    from loading.  Qnom spread is what would separate them -- reported in the caller."""
    out = []
    specs = {"kpod_only": [kpod_col], "ql_only": ["log_ql"], "joint": [kpod_col, "log_ql"]}
    for name, cols in specs.items():
        s = stratified_cox(df, cols, clock=clock)
        s.insert(0, "spec", name)
        out.append(s)
    return pd.concat(out, ignore_index=True)


def speed_load_decomposition(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY,
                             freq_col: str = "freq_run", kpod_col: str = "kpod_run") -> pd.DataFrame:
    """Does raising **frequency** and/or **Kpod = Ql/Qnom** hurt reliability?

    Frequency (running speed), Kpod (off-design load) and Ql (throughput) are one
    causal chain — the affinity law makes higher speed drive higher flow drive higher
    Kpod — so they are collinear by construction.  This fits the ladder of nested
    models that shows which term survives when the others are held: a coefficient that
    keeps its sign and magnitude across specs is a distinct effect; one that collapses
    when a chain-mate enters was that chain-mate in disguise."""
    specs = {
        "freq_only": [freq_col],
        "kpod_only": [kpod_col],
        "ql_only": ["log_ql"],
        "freq_plus_kpod": [freq_col, kpod_col],
        "freq_plus_ql": [freq_col, "log_ql"],
        "freq_kpod_ql": [freq_col, kpod_col, "log_ql"],
    }
    out = []
    for name, cols in specs.items():
        s = stratified_cox(df, cols, clock=clock)
        s.insert(0, "spec", name)
        out.append(s)
    return pd.concat(out, ignore_index=True)


#: The operating chain, in the order a deployment should prefer to drop them (Ql is
#: the most often available on the plan side, Kpod the most often missing since it
#: needs Q_nominal).  Availability patterns are the non-empty subsets of this.
OPERATING_CHAIN = ["freq_run", "kpod_run", "log_ql"]


def availability_coefficients(
    df: pd.DataFrame, *, chain: list[str] | None = None, clock: str = CLOCK_PRIMARY,
) -> pd.DataFrame:
    """Coefficient set **per covariate-availability pattern** — the deployment recipe
    for missing collinear inputs.

    freq / Kpod / Ql are one affinity-law chain, so a coefficient is *conditional* on
    which chain-mates are in the model: the freq multiplier fitted with Kpod present
    (``freq+kpod``) is the effect of freq *holding Kpod fixed*; the freq multiplier
    fitted with Kpod absent (``freq`` alone) is *marginal* over Kpod — it absorbs the
    part of the chain Kpod would have carried, and is therefore larger.

    **Application rule.** When scoring a well, look up the row whose ``pattern`` matches
    exactly the covariates you actually have for it, and use those coefficients.  Do NOT
    take the full-model (all-present) coefficient and feed it an imputed value for a
    missing chain-mate — with this much collinearity that double-counts (the full-model
    freq coef is already netted of Kpod, and an imputed-from-freq Kpod adds nothing new).
    The available-case sub-model is the clean, unbiased predictor for that pattern.

    Returns one row per (pattern, covariate): HR per natural unit + 95 % CI + p, and the
    complete-case n / n_events the pattern was fitted on."""
    from itertools import combinations

    chain = chain or OPERATING_CHAIN
    rows = []
    for k in range(1, len(chain) + 1):
        for sub in combinations(chain, k):
            pattern = "+".join(c.replace("_run", "").replace("log_", "") for c in sub)
            s = stratified_cox(df, list(sub), clock=clock)
            for _, r in s.iterrows():
                if pd.isna(r.get("hr_per_natural")):
                    continue
                rows.append({
                    "pattern": pattern, "n_covariates": k, "covariate": r["covariate_raw"],
                    "natural_unit": r["natural_unit"],
                    "hr_per_natural": round(float(r["hr_per_natural"]), 4),
                    "hr_ci_lo": round(float(r["hr_nat_ci_lo"]), 4),
                    "hr_ci_hi": round(float(r["hr_nat_ci_hi"]), 4),
                    "p": round(float(r["p"]), 4),
                    "n": int(r["n"]), "n_events": int(r["n_events"]),
                    "role": "marginal" if k == 1 else ("conditional" if k == len(chain) else "partial"),
                })
    return pd.DataFrame(rows)


#: Natural step per operating covariate for the life-multiplier reporting.
CHAIN_STEPS = {"log_ql": 1.0, "freq_run": 1.0, "kpod_run": 0.1}
CHAIN_UNIT = {"log_ql": "per +1 log-unit (e-fold Ql)", "freq_run": "per +1 Hz", "kpod_run": "per +0.1 Kpod"}


def availability_life_multipliers(
    df: pd.DataFrame, *, h2s_class: str = "nonsour", chain: list[str] | None = None,
    clock: str = CLOCK_PRIMARY, penalizer: float = 0.02,
) -> pd.DataFrame:
    """Per-class **life-multiplier** coefficients for every covariate-availability pattern —
    the deployment recipe when only some of {Ql, freq, Kpod} are known.

    For each non-empty subset of the affinity chain we fit a Weibull **AFT** on that class,
    **with contractor dummies (slb/oth) held in every fit**, on the available-case rows
    (all subset covariates present).  The reported ``life_mult`` is the multiplicative shift
    of TTF per natural step, exp(β_AFT·step) — so the three-covariate row is the
    **split (conditional)** effect of each covariate holding the other two fixed, and the
    one/two-covariate rows are the **fallbacks** (each absorbs the chain-mates it omits).

    **Application.** Score a well with the row-set whose ``pattern`` matches exactly the
    covariates available for it; multiply the base by ``Π life_mult_i^((X_i−X_ref,i)/step_i)``
    and the contractor multiplier.  Do NOT impute a missing chain-mate and use the 3-way row
    (double-counts).  Returns one row per (pattern, covariate) incl. contractor terms, with
    life_mult + 95% CI + p + the available-case n/n_events."""
    from itertools import combinations
    from lifelines import WeibullAFTFitter

    chain = chain or list(CHAIN_STEPS)
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s_class)].copy()
    g[clock] = pd.to_numeric(g[clock], errors="coerce")
    g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL])
    g["slb"] = (g["contractor_group"] == "slb").astype(float)
    g["oth"] = (g["contractor_group"] == "oth").astype(float)
    rows = []
    for k in range(1, len(chain) + 1):
        for sub in combinations(chain, k):
            pattern = "+".join(c.replace("_run", "").replace("log_", "") for c in sub)
            cols = list(sub) + ["slb", "oth"]
            d = g.dropna(subset=list(sub))[[clock, EVENT_COL, *cols]].copy()
            if int(d[EVENT_COL].sum()) < 15:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aft = WeibullAFTFitter(penalizer=penalizer).fit(d, clock, EVENT_COL)
            lam = aft.summary.xs("lambda_")
            role = "marginal" if k == 1 else ("split/conditional" if k == len(chain) else "partial")
            for c in cols:
                if c not in lam.index:
                    continue
                step = CHAIN_STEPS.get(c, 1.0)                       # contractor: per level (step 1)
                coef, se = float(lam.loc[c, "coef"]), float(lam.loc[c, "se(coef)"])
                unit = CHAIN_UNIT.get(c, "vs brt")
                is_chain = c in CHAIN_STEPS
                rows.append({
                    "h2s_class": h2s_class, "pattern": pattern, "n_covariates": k,
                    "covariate": c, "kind": "operating" if is_chain else "contractor",
                    "natural_unit": unit,
                    "life_mult": round(float(np.exp(coef * step)), 4),
                    "ci_lo": round(float(np.exp((coef - 1.96 * se) * step)), 4),
                    "ci_hi": round(float(np.exp((coef + 1.96 * se) * step)), 4),
                    "p": round(float(lam.loc[c, "p"]), 4),
                    "role": role if is_chain else "contractor",
                    "n": int(len(d)), "n_events": int(d[EVENT_COL].sum()),
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Per-stratum consolidated deliverable: baseline survival + coefficients + the
# TTF-adjustment equation
# ===========================================================================

#: Covariates carried into the per-stratum adjustment equation (the operating chain;
#: Qg/GLF are null and left out of the readable equation).
EQUATION_COVARIATES = ["freq_run", "kpod_run", "log_ql"]


#: v3.1 Ql covariate shape: log with limits (winsorized at these quantiles).  Chosen by
#: AIC over {linear, log, log+quadratic, spline df3, log+hinge}: capping wins (ΔAIC −3.1
#: vs plain log) and its γ (−0.31) matches the empirical binned-median slope (−0.32) —
#: the plain-log flatness was leverage from the Ql≈1 point and saturation above ~q95.
QL_CAP_QUANTILES = (0.05, 0.95)


def _capped_log_ql(g: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """log(Ql) winsorized at the class q05–q95 (the v3.1 'log with limits' shape)."""
    lo = float(g["ql"].quantile(QL_CAP_QUANTILES[0]))
    hi = float(g["ql"].quantile(QL_CAP_QUANTILES[1]))
    return np.log(g["ql"].clip(lo, hi)), lo, hi


def revised_operating_model(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """The revised Vt operating model (v3.1): **baseline + capped-log-Ql + contractor**.

    v2 (Ql-only) under-predicted brt; v3 added the contractor covariate (empirical,
    mechanism unidentified — plausibly informative censoring).  v3.1 refines the Ql
    *shape*: **log with limits** — Ql winsorized at the class q05–q95 before the log —
    which wins on AIC and recovers the empirical median slope (γ ≈ −0.31 vs binned −0.32);
    outside the caps the effect is held flat (no extrapolation beyond observed support).

        TTF(well) = M0(h2s) · (clip(Ql, lo, hi) / Ql_ref)^γ · contractor_mult(contractor)

    Fitted per h2s class with a Weibull AFT on ``capped log_ql + contractor``
    (reference = brt).  Ql is null on sour (H2S dominates) but carried for symmetry.
    Returns one row per (h2s_class, term), with the cap limits on the Ql row."""
    from lifelines import WeibullAFTFitter

    rows = []
    for h2s in ("nonsour", "sour"):
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
        g = g[g["ql"].notna()].copy()
        g[clock] = pd.to_numeric(g[clock], errors="coerce")
        g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"])
        if int(g[EVENT_COL].sum()) < 20:
            rows.append({"h2s_class": h2s, "term": "(model)", "note": "too few Ql-covered events"})
            continue
        g["lqc"], cap_lo, cap_hi = _capped_log_ql(g)
        qref = float(g["lqc"].median())
        dummies = pd.get_dummies(g["contractor_group"], prefix="c", drop_first=True).astype(float)
        ref = sorted(g["contractor_group"].unique())[0]      # 'brt' sorts first
        X = pd.concat([g[[clock, EVENT_COL, "lqc"]], dummies], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft = WeibullAFTFitter(penalizer=0.01).fit(X, clock, EVENT_COL)
            lam = aft.summary.xs("lambda_")
            # M0 = median at Ql_ref for the reference contractor (all dummies = 0).
            ref_row = pd.DataFrame({**{"lqc": [qref]}, **{c: [0.0] for c in dummies.columns}})
            m0 = float(aft.predict_median(ref_row).iloc[0])
            # Weibull params of S(t|x)=exp(-(t/eta)^beta): shape beta is CONDITIONAL (one
            # per class, covariates held), eta at the reference (brt, Ql_ref).  NB this
            # conditional beta (~1.1-1.4) differs from the marginal per-stratum beta (<1
            # for nonsour) — the marginal <1 is frailty across wells, not a fit clock.
            beta = float(np.exp(aft.summary.xs("rho_").loc["Intercept", "coef"]))
            eta_ref = float(np.exp(lam.loc["Intercept", "coef"] + lam.loc["lqc", "coef"] * qref))
        # Ql term
        s = lam.loc["lqc"]
        rows.append({
            "h2s_class": h2s, "term": "Ql", "reference": f"{ref}, Ql={np.exp(qref):.0f} m3/d",
            "weibull_beta": round(beta, 3), "eta_ref_d": round(eta_ref, 0),
            "baseline_median_d": round(m0, 0), "ql_ref_m3d": round(float(np.exp(qref)), 0),
            "ql_cap_lo": round(cap_lo, 0), "ql_cap_hi": round(cap_hi, 0),
            "coef": round(float(s["coef"]), 3),
            "life_mult": round(float(np.exp(s["coef"])), 3),
            "life_mult_unit": "per e-fold Ql (capped)",
            "ci": f"[{np.exp(s['coef lower 95%']):.2f}, {np.exp(s['coef upper 95%']):.2f}]",
            "p": round(float(s["p"]), 4),
        })
        # contractor terms
        for c in dummies.columns:
            cg = c[2:]
            sc = lam.loc[c]
            rows.append({
                "h2s_class": h2s, "term": f"contractor:{cg}", "reference": f"vs {ref}",
                "coef": round(float(sc["coef"]), 3),
                "life_mult": round(float(np.exp(sc["coef"])), 3),
                "life_mult_unit": f"vs {ref}",
                "ci": f"[{np.exp(sc['coef lower 95%']):.2f}, {np.exp(sc['coef upper 95%']):.2f}]",
                "p": round(float(sc["p"]), 4),
            })
    return pd.DataFrame(rows)


def informative_censoring_test(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Does ГТМ-censoring explain the contractor residual? — the competing-risks test.

    Two probes on Vt_nonsour: (a) P(genuine failure by t) via cause-specific 1−KM (ГТМ
    censored) vs the Aalen–Johansen CIF (ГТМ a competing event) — if the brt/slb gap
    shrinks under CIF, part of the KM advantage was censoring bias; (b) the AFT slb
    multiplier on the cause-specific vs all-cause estimand.  Verdict (2026-07-22): the
    CIF gap is ~5–12 % smaller and the slb multiplier moves 0.79 → 0.87 — informative
    censoring explains **about a third** of the residual, not all of it."""
    from lifelines import AalenJohansenFitter, KaplanMeierFitter, WeibullAFTFitter
    from analysis.workflows.production_risk import esp_population as _pop

    ns = df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour")].copy()
    ns["_reason"] = ns["pull_reason"].map(_pop._norm).str.casefold()
    ns["ev_cr"] = np.where(ns[EVENT_COL] == 1, 1,
                           np.where(ns["end"].notna() & ns["_reason"].isin(_pop.WORKOVER_REASONS), 2, 0))
    ns[clock] = pd.to_numeric(ns[clock], errors="coerce")
    ns = ns[ns[clock] > 0]

    rows = []
    for t in (180, 365, 730):
        rec = {"check": f"p_fail_by_{t}d"}
        for cg in ("brt", "slb"):
            g = ns[ns["contractor_group"] == cg]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KaplanMeierFitter().fit(g[clock], g[EVENT_COL])
                pkm = 1 - float(km.survival_function_at_times(t).iloc[0])
                aj = AalenJohansenFitter(calculate_variance=False).fit(
                    g[clock], g["ev_cr"], event_of_interest=1)
                cd = aj.cumulative_density_
                pcif = float(cd[cd.index <= t].iloc[-1, 0]) if (cd.index <= t).any() else 0.0
            rec[f"{cg}_km"] = round(pkm, 3)
            rec[f"{cg}_cif"] = round(pcif, 3)
        rec["gap_km"] = round(rec["slb_km"] - rec["brt_km"], 3)
        rec["gap_cif"] = round(rec["slb_cif"] - rec["brt_cif"], 3)
        rows.append(rec)

    q = ns[ns["ql"].notna()].copy()
    q["lqc"], _, _ = _capped_log_ql(q)
    q["slb"] = (q["contractor_group"] == "slb").astype(float)
    q["oth"] = (q["contractor_group"] == "oth").astype(float)
    q["pulled"] = q["end"].notna().astype(int)
    for lab, ev in (("cause_specific", EVENT_COL), ("all_cause", "pulled")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = WeibullAFTFitter(penalizer=0.01).fit(
                q[[clock, ev, "lqc", "slb", "oth"]].rename(columns={ev: "_E"}), clock, "_E")
        lam = a.summary.xs("lambda_")
        rows.append({"check": f"aft_slb_mult_{lab}",
                     "slb_mult": round(float(np.exp(lam.loc["slb", "coef"])), 3),
                     "p": round(float(lam.loc["slb", "p"]), 3),
                     "gamma_ql": round(float(lam.loc["lqc", "coef"]), 3)})
    return pd.DataFrame(rows)


def replicate_v31_on_ya(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """External validity: refit the exact v3.1 design (capped log-Ql + contractor, AFT)
    on Ya_nonsour (3–4× Vt's events) and stack against Vt.

    Verdict (2026-07-22): **replicates** — Ya γ = −0.255 [−0.33, −0.19] (p<1e-5) vs Vt
    −0.305, and the contractor multipliers match (slb 0.805 p=0.008 on Ya vs 0.792 on
    Vt; oth shorter on both).  The slb≈0.8 term is a replicated effect, not Vt noise."""
    from lifelines import WeibullAFTFitter

    rows = []
    for fld in ("Vt", "Ya"):
        g = df[(df["field"] == fld) & (df["h2s_class"] == "nonsour") & df["ql"].notna()].copy()
        g[clock] = pd.to_numeric(g[clock], errors="coerce")
        g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"])
        if int(g[EVENT_COL].sum()) < 30:
            continue
        g["lqc"], lo, hi = _capped_log_ql(g)
        qref = float(g["lqc"].median())
        dm = pd.get_dummies(g["contractor_group"], prefix="c", drop_first=True).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = WeibullAFTFitter(penalizer=0.01).fit(
                pd.concat([g[[clock, EVENT_COL, "lqc"]], dm], axis=1), clock, EVENT_COL)
            lam = a.summary.xs("lambda_")
            m0 = float(a.predict_median(pd.DataFrame(
                {**{"lqc": [qref]}, **{c: [0.0] for c in dm.columns}})).iloc[0])
        s = lam.loc["lqc"]
        row = {"field": fld, "n": len(g), "n_events": int(g[EVENT_COL].sum()),
               "caps": f"[{lo:.0f},{hi:.0f}]", "ql_ref": round(float(np.exp(qref)), 0),
               "m0_median_d": round(m0, 0), "gamma_ql": round(float(s["coef"]), 3),
               "gamma_ci": f"[{s['coef lower 95%']:.3f},{s['coef upper 95%']:.3f}]",
               "gamma_p": round(float(s["p"]), 5)}
        for c in dm.columns:
            sc = lam.loc[c]
            row[f"mult_{c[2:]}"] = round(float(np.exp(sc["coef"])), 3)
            row[f"p_{c[2:]}"] = round(float(sc["p"]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_holdout_validation(df: pd.DataFrame, *, cutoff: str = "2024-01-01",
                                clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Out-of-sample gate: fit v3.1 on installs < ``cutoff``, evaluate on the ≥ cutoff
    cohort — the repo's standard before a layer may feed the forecast.

    Nonsour only: the Ql-covered **sour** population is essentially all 2024+ (1 pre-2024
    run, 0 events), so a sour holdout is impossible — reported as untestable, not skipped
    silently.  Evaluation on the test cohort: per-contractor predicted (Ql-mixture) median
    vs observed KM median, predicted vs observed S(t) at 180/365 d, and the γ refitted on
    test alone (coefficient stability)."""
    from lifelines import WeibullAFTFitter, KaplanMeierFitter

    rows = []
    cut = pd.Timestamp(cutoff)
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour") & df["ql"].notna()].copy()
    g[clock] = pd.to_numeric(g[clock], errors="coerce")
    g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"])
    g["install"] = pd.to_datetime(g["install"])
    train, test = g[g["install"] < cut].copy(), g[g["install"] >= cut].copy()

    def _design(d, caps=None):
        d = d.copy()
        if caps is None:
            d["lqc"], lo, hi = _capped_log_ql(d)
        else:
            lo, hi = caps
            d["lqc"] = np.log(d["ql"].clip(lo, hi))
        for cg in ("slb", "oth"):
            d[cg] = (d["contractor_group"] == cg).astype(float)
        return d, (lo, hi)

    train, caps = _design(train)
    test, _ = _design(test, caps=caps)          # test uses TRAIN caps (no peeking)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = WeibullAFTFitter(penalizer=0.01).fit(
            train[[clock, EVENT_COL, "lqc", "slb", "oth"]], clock, EVENT_COL)
        gam_train = float(aft.summary.xs("lambda_").loc["lqc", "coef"])
        aft_test = WeibullAFTFitter(penalizer=0.01).fit(
            test[[clock, EVENT_COL, "lqc", "slb", "oth"]], clock, EVENT_COL)
        gam_test = float(aft_test.summary.xs("lambda_").loc["lqc", "coef"])
    rows.append({"scope": "nonsour_gamma_stability", "train": round(gam_train, 3),
                 "test": round(gam_test, 3),
                 "n_events_train": int(train[EVENT_COL].sum()),
                 "n_events_test": int(test[EVENT_COL].sum()),
                 "verdict": "ok" if np.sign(gam_train) == np.sign(gam_test) else "sign flip"})

    tl = np.linspace(1, float(test[clock].quantile(0.99)) * 2, 500)
    train_ev = train.groupby("contractor_group")[EVENT_COL].sum().to_dict()
    for scope, sub in [("nonsour_all", test)] + [
            (f"nonsour_{cg}", test[test["contractor_group"] == cg]) for cg in ("brt", "slb", "oth")]:
        if int(sub[EVENT_COL].sum()) < 10:
            rows.append({"scope": f"holdout_median:{scope}", "verdict": "thin"})
            continue
        cg_key = scope.rsplit("_", 1)[-1]
        if cg_key in train_ev and train_ev.get(cg_key, 0) < 10:
            # The contractor coefficient cannot be learned from the train era at all —
            # its holdout row measures nothing (e.g. oth: 2 pre-2024 runs, 1 event).
            rows.append({"scope": f"holdout_median:{scope}",
                         "verdict": f"not evaluable — train has {train_ev.get(cg_key, 0)} events for {cg_key}"})
            continue
        S = aft.predict_survival_function(sub[["lqc", "slb", "oth"]], times=tl).mean(axis=1).values
        pred_med = float(tl[np.argmin(np.abs(S - 0.5))]) if S.min() < 0.5 else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KaplanMeierFitter().fit(sub[clock], sub[EVENT_COL])
        obs_med = float(km.median_survival_time_)
        rec = {"scope": f"holdout_median:{scope}", "observed": round(obs_med, 0),
               "predicted": round(pred_med, 0) if np.isfinite(pred_med) else np.nan,
               "ratio": round(pred_med / obs_med, 2) if obs_med > 0 and np.isfinite(pred_med) else np.nan,
               "n_events_test": int(sub[EVENT_COL].sum())}
        for t in (180, 365):
            Sp = float(aft.predict_survival_function(sub[["lqc", "slb", "oth"]], times=[t]).mean(axis=1).iloc[0])
            So = float(km.survival_function_at_times(t).iloc[0])
            rec[f"S{t}_pred"] = round(Sp, 3)
            rec[f"S{t}_obs"] = round(So, 3)
        rec["verdict"] = ("ok" if np.isfinite(rec.get("ratio", np.nan)) and 0.8 <= rec["ratio"] <= 1.25
                          else "check")
        rows.append(rec)

    rows.append({"scope": "sour", "verdict": "UNTESTABLE — Ql-covered sour population is "
                 "all 2024+ (1 pre-cutoff run, 0 events); no temporal holdout possible"})
    return pd.DataFrame(rows)


def bootstrap_model_cis(df: pd.DataFrame, *, n_boot: int = 200, seed: int = 7,
                        clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Bootstrap CIs on the **composed** v3.1 deliverables (M0, γ, contractor mults and
    the per-contractor predicted medians) — the coefficient CIs are asymptotic, but the
    quantities the equation actually serves had none.

    Cluster bootstrap **by well** (runs of one well are dependent), n_boot=200 per the
    repo standard; percentile 2.5/97.5.  Each draw refits the AFT and predicts on the
    ORIGINAL per-contractor covariate sets."""
    from lifelines import WeibullAFTFitter

    rng = np.random.default_rng(seed)
    out_rows = []
    for h2s in ("nonsour", "sour"):
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s) & df["ql"].notna()].copy()
        g[clock] = pd.to_numeric(g[clock], errors="coerce")
        g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"])
        if int(g[EVENT_COL].sum()) < 20:
            continue
        g["lqc"], lo, hi = _capped_log_ql(g)
        qref = float(g["lqc"].median())
        for cg in ("slb", "oth"):
            g[cg] = (g["contractor_group"] == cg).astype(float)
        wells = g["code"].unique()
        by_well = {w: sub for w, sub in g.groupby("code")}
        tl = np.linspace(1, float(g[clock].quantile(0.99)) * 2, 400)

        stats: dict[str, list] = {k: [] for k in
                                  ["gamma", "m0", "mult_slb", "mult_oth",
                                   "med_brt", "med_slb", "med_oth"]}
        for _ in range(n_boot):
            draw = rng.choice(wells, size=len(wells), replace=True)
            b = pd.concat([by_well[w] for w in draw], ignore_index=True)
            if int(b[EVENT_COL].sum()) < 15 or b["contractor_group"].nunique() < 2:
                continue
            X = b[[clock, EVENT_COL, "lqc", "slb", "oth"]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aft = WeibullAFTFitter(penalizer=0.01).fit(X, clock, EVENT_COL)
            except Exception:
                continue
            lam = aft.summary.xs("lambda_")
            stats["gamma"].append(float(lam.loc["lqc", "coef"]))
            stats["mult_slb"].append(float(np.exp(lam.loc["slb", "coef"])) if "slb" in lam.index else np.nan)
            stats["mult_oth"].append(float(np.exp(lam.loc["oth", "coef"])) if "oth" in lam.index else np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats["m0"].append(float(aft.predict_median(
                    pd.DataFrame({"lqc": [qref], "slb": [0.0], "oth": [0.0]})).iloc[0]))
            # composed medians on the ORIGINAL per-contractor covariates
            for cg in ("brt", "slb", "oth"):
                sub = g[g["contractor_group"] == cg]
                if len(sub) < 15:
                    stats[f"med_{cg}"].append(np.nan)
                    continue
                S = aft.predict_survival_function(sub[["lqc", "slb", "oth"]], times=tl).mean(axis=1).values
                stats[f"med_{cg}"].append(float(tl[np.argmin(np.abs(S - 0.5))]) if S.min() < 0.5 else np.nan)

        for name, vals in stats.items():
            v = np.array([x for x in vals if np.isfinite(x)])
            if len(v) < 50:
                continue
            out_rows.append({"h2s_class": h2s, "quantity": name,
                             "point": round(float(np.median(v)), 3),
                             "ci_lo": round(float(np.percentile(v, 2.5)), 3),
                             "ci_hi": round(float(np.percentile(v, 97.5)), 3),
                             "n_boot_ok": len(v)})
    return pd.DataFrame(out_rows)


def model_consistency_checks(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Cross-checks that the v3.1 model, the per-stratum baselines and the covariate data
    tell one consistent story.  Reported, never silently passed:

      * per-stratum **predicted vs observed median on the SAME (Ql-covered) sample** —
        comparing the model to the all-runs baseline mixes in the uncovered (mostly
        pre-2018, longer-lived) runs and fakes a miss;
      * covariate hygiene counts (Ql≤5 leverage, t0 windows <10 days, kpod>4,
        type_parse Qnom tier) — the rows the shape/robustness passes guard against;
      * the h2s-confounding check: pooled-Vt Ql coefficient WITHOUT h2s stratification
        vs stratified (sour runs at *higher* Ql with *shorter* life, so an unstratified
        fit inflates the Ql effect — the per-class fit is required)."""
    from lifelines import WeibullAFTFitter, KaplanMeierFitter

    rows = []
    for h2s in ("nonsour", "sour"):
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
        g = g[g["ql"].notna()]
        g[clock] = pd.to_numeric(g[clock], errors="coerce")
        g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"])
        if int(g[EVENT_COL].sum()) < 20:
            continue
        g = g.copy()
        g["lqc"], _, _ = _capped_log_ql(g)
        g["slb"] = (g["contractor_group"] == "slb").astype(float)
        g["oth"] = (g["contractor_group"] == "oth").astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft = WeibullAFTFitter(penalizer=0.01).fit(
                g[[clock, EVENT_COL, "lqc", "slb", "oth"]], clock, EVENT_COL)
        tl = np.linspace(1, float(g[clock].quantile(0.99)) * 2, 600)
        for cg, sub in g.groupby("contractor_group"):
            n_ev = int(sub[EVENT_COL].sum())
            if n_ev < 10:
                continue
            S = aft.predict_survival_function(sub[["lqc", "slb", "oth"]], times=tl).mean(axis=1).values
            pred = float(tl[np.argmin(np.abs(S - 0.5))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs = float(KaplanMeierFitter().fit(sub[clock], sub[EVENT_COL]).median_survival_time_)
            rows.append({
                "check": "pred_vs_obs_median_same_sample", "scope": f"Vt_{h2s}_{cg}",
                "observed": round(obs, 0), "predicted": round(pred, 0),
                "ratio": round(pred / obs, 2) if obs > 0 else np.nan, "n_events": n_ev,
                "verdict": "ok" if obs > 0 and 0.85 <= pred / obs <= 1.18 else "check",
            })

    vt = df[df["field"] == FIELD]
    hygiene = {
        "ql_leq_5_leverage": int((vt["ql"] <= 5).sum()),
        "t0_window_lt_10d": int((vt["t0_n_days"] < 10).sum()),
        "kpod_gt_4": int((vt["kpod_run"] > 4).sum()),
        "qnom_type_parse_tier": int((vt["kpod_qnom_source"] == "type_parse").sum()),
        "t_cal_leq_1": int((pd.to_numeric(vt["t_cal"], errors="coerce") <= 1).sum()),
    }
    for k, v in hygiene.items():
        rows.append({"check": f"hygiene:{k}", "scope": "Vt", "observed": v,
                     "verdict": "ok" if v <= 30 else "check"})
    return pd.DataFrame(rows)


def per_stratum_survival_ttf(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Per Vt stratum: fitted baseline Weibull (shape β, scale η on ``clock``) and the
    baseline TTF summaries + the covariate reference (stratum-mean) each adjustment is
    taken relative to.  Weibull fitted directly on the analysis clock so the baseline is
    consistent with the covariate fits (RMST/MRL/median reporting, never B50)."""
    from math import gamma
    from lifelines import WeibullFitter

    rows = []
    for stratum, g in df[df["field"] == FIELD].groupby("stratum"):
        d = g[[clock, EVENT_COL]].copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[(d[clock] > 0)].dropna()
        n_ev = int(d[EVENT_COL].sum())
        rec = {"stratum": stratum, "h2s_class": g["h2s_class"].iloc[0],
               "contractor": g["contractor_group"].iloc[0], "n_runs": len(g), "n_events": n_ev}
        if n_ev < 8:
            rec["note"] = "too few events"
            rows.append(rec)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter().fit(d[clock], d[EVENT_COL])
        beta, eta = float(wf.rho_), float(wf.lambda_)
        rmst, mean, median = _weibull_life(beta, eta, horizon=max(5.0 * eta, 3000.0))
        rec.update({"clock": clock, "weibull_beta": round(beta, 3), "weibull_eta": round(eta, 1),
                    "ttf_rmst0": rmst, "ttf_mean": mean, "ttf_median": median})
        for c in EQUATION_COVARIATES:
            rec[f"{c}_ref"] = round(float(g[c].mean()), 3) if g[c].notna().any() else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    # For reference, the clock each stratum uses in the SHIPPED production bundle
    # (esp_models): this analysis re-fits everything on ``clock`` (t_cal) so the
    # baseline is consistent with the t_cal covariate coefficients, but the shipped
    # forecast bundle mixes clocks per stratum — surfaced here so the two aren't confused.
    try:
        shipped = _load_vt_baselines()[["stratum", "clock"]].rename(columns={"clock": "shipped_clock"})
        out = out.merge(shipped, on="stratum", how="left")
    except Exception:
        out["shipped_clock"] = np.nan
    return out


def per_h2s_coefficients(df: pd.DataFrame, *, covariates: list[str] | None = None,
                         clock: str = CLOCK_PRIMARY, shape_by_class: dict | None = None) -> pd.DataFrame:
    """Covariate coefficients fitted **separately within sour and within nonsour** (and
    pooled), stratified by contractor — the answer to "do sour and nonsour differ?".

    Each row also carries the **life multiplier per natural step** = HR^(−1/β_weibull)
    using that class's mean Weibull shape, i.e. the factor the equation multiplies TTF
    by per +1 natural unit of the covariate."""
    covariates = covariates or EQUATION_COVARIATES
    shape_by_class = shape_by_class or {}
    rows = []
    for label, sub in [("sour", df[(df["field"] == FIELD) & (df["h2s_class"] == "sour")]),
                       ("nonsour", df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour")]),
                       ("pooled", df[df["field"] == FIELD])]:
        work = sub.copy()
        # within a class the only thing left to stratify on is contractor; pooled keeps
        # the full h2s×contractor stratum.
        work["stratum"] = (work["contractor_group"] if label != "pooled"
                           else work["field"] + "_" + work["h2s_class"] + "_" + work["contractor_group"])
        beta_shape = shape_by_class.get(label, np.nan)
        for cov in covariates:
            s = stratified_cox(work, [cov], clock=clock)
            r = s[s.get("covariate_raw", "") == cov]
            if r.empty:
                rows.append({"h2s_class": label, "covariate": cov, "note": "no fit"})
                continue
            rr = r.iloc[0]
            hr = float(rr["hr_per_natural"])
            life_mult = hr ** (-1.0 / beta_shape) if np.isfinite(beta_shape) and beta_shape > 0 else np.nan
            rows.append({
                "h2s_class": label, "covariate": cov, "natural_unit": rr["natural_unit"],
                "hr_per_natural": round(hr, 4), "hr_ci_lo": round(float(rr["hr_nat_ci_lo"]), 4),
                "hr_ci_hi": round(float(rr["hr_nat_ci_hi"]), 4), "p": round(float(rr["p"]), 4),
                "n_events": int(rr["n_events"]), "weibull_shape": round(beta_shape, 3) if np.isfinite(beta_shape) else np.nan,
                "life_mult_per_step": round(life_mult, 4) if np.isfinite(life_mult) else np.nan,
                "verdict": "effect" if rr["p"] < 0.05 else "null",
            })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def contractor_gap_analysis(
    df: pd.DataFrame, *, h2s: str = "nonsour", a: str = "brt", b: str = "slb",
    clock: str = CLOCK_PRIMARY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Why do two contractor strata differ in baseline life? — covariate balance,
    cohort robustness, and how much of the gap the operating covariates explain.

    Returns ``(balance, cohort_weibull, adjusted_hr)``: median balance of a/b on every
    covariate + install cohort; the Weibull baseline for each on all-history and the
    2024+ cohort (is the gap a cohort artifact?); and the b-vs-a hazard ratio unadjusted
    then adjusted for the operating covariates (does the gap shrink?)."""
    from lifelines import WeibullFitter
    from analysis.workflows.production_risk import esp_population as _pop

    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
    try:
        g = _pop.attach_equipment(g)
    except Exception:
        pass
    ga, gb = g[g["contractor_group"] == a], g[g["contractor_group"] == b]

    def _med(s):
        s = pd.to_numeric(s, errors="coerce")
        return round(float(s.median()), 2) if s.notna().any() else np.nan

    metrics = ["ql", "qg", "glf", "freq_run", "kpod_run", "wcut", "h2s_mg_l", "kvch",
               "setting_depth_m", "max_corr_class", clock]
    brows = [
        {"metric": "n_runs", a: len(ga), b: len(gb)},
        {"metric": "n_events", a: int(ga[EVENT_COL].sum()), b: int(gb[EVENT_COL].sum())},
        {"metric": "install_median", a: str(pd.to_datetime(ga["install"]).median().date()),
         b: str(pd.to_datetime(gb["install"]).median().date())},
        {"metric": "pct_install_2024plus",
         a: round(100 * (pd.to_datetime(ga["install"]) >= pd.Timestamp(C.MC_INSTALL_COHORT_START)).mean(), 1),
         b: round(100 * (pd.to_datetime(gb["install"]) >= pd.Timestamp(C.MC_INSTALL_COHORT_START)).mean(), 1)},
    ]
    for m in metrics:
        if m in g.columns:
            brows.append({"metric": f"{m}_median", a: _med(ga[m]), b: _med(gb[m])})
    balance = pd.DataFrame(brows)

    wrows = []
    start = pd.Timestamp(C.MC_INSTALL_COHORT_START)
    for lab, sub in [(f"{a}_all", ga), (f"{b}_all", gb),
                     (f"{a}_2024plus", ga[pd.to_datetime(ga["install"]) >= start]),
                     (f"{b}_2024plus", gb[pd.to_datetime(gb["install"]) >= start])]:
        d = sub[[clock, EVENT_COL]].copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0].dropna()
        if int(d[EVENT_COL].sum()) < 8:
            wrows.append({"group": lab, "n_events": int(d[EVENT_COL].sum()), "note": "thin"})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter().fit(d[clock], d[EVENT_COL])
        wrows.append({"group": lab, "n_events": int(d[EVENT_COL].sum()),
                      "weibull_eta": round(float(wf.lambda_), 1), "weibull_beta": round(float(wf.rho_), 3),
                      "median": round(float(wf.median_survival_time_), 1)})
    cohort_weibull = pd.DataFrame(wrows)

    g["_b"] = (g["contractor_group"] == b).astype(float)
    # Fair comparison: run the whole HR ladder on ONE complete-case sample (rows with the
    # rate covariates present), so "unadjusted" and "adjusted" are not on different rows.
    # An unadjusted HR on the full sample vs an adjusted HR on the covered subset would
    # confound the covariate effect with the coverage cohort.
    common = g.dropna(subset=["freq_run", "kpod_run", "log_ql", "glf"])
    arows = []
    for lab, cols in {"unadjusted": ["_b"],
                      "adj_freq_kpod_ql": ["_b", "freq_run", "kpod_run", "log_ql"],
                      "adj_plus_glf": ["_b", "freq_run", "kpod_run", "log_ql", "glf"]}.items():
        d = common
        res = fit_cause_specific_cox(d, covariates=cols, event_col=EVENT_COL, duration_col=clock,
                                     strata=["field"], cluster_col=CLUSTER, min_events=15)
        if res.success and not res.summary[res.summary["covariate"] == "_b"].empty:
            r = res.summary[res.summary["covariate"] == "_b"].iloc[0]
            arows.append({"spec": lab, f"hr_{b}_vs_{a}": round(float(r["hr"]), 3),
                          "p": round(float(r["p"]), 4), "n_events": res.n_events})
        else:
            arows.append({"spec": lab, "note": res.message})
    adjusted_hr = pd.DataFrame(arows)
    return balance, cohort_weibull, adjusted_hr


def contractor_pull_composition(df: pd.DataFrame, *, h2s: str = "nonsour",
                                clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Is a contractor's shorter life genuine FAILURES or more ГТМ (planned pulls)?

    Splits each run into genuine_failure / ГТМ-ППР pull / running, and compares the
    cause-specific-failure median (ГТМ censored) with the all-cause median (any pull an
    event) and the failure rate per 1000 op-days.  A higher failure RATE ⇒ real
    reliability gap; more ГТМ with a similar failure rate ⇒ the short runs are
    intervention-driven, not a reliability difference."""
    from lifelines import WeibullFitter
    from analysis.workflows.production_risk import esp_population as _pop

    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
    g["_reason"] = g["pull_reason"].map(_pop._norm).str.casefold()
    rows = []
    for cg, sub in g.groupby("contractor_group"):
        d = sub.copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0]
        n = len(d)
        n_fail = int(d[EVENT_COL].sum())
        gtm_mask = d["end"].notna() & (d[EVENT_COL] == 0) & d["_reason"].isin(_pop.WORKOVER_REASONS)
        n_gtm = int(gtm_mask.sum())
        n_run = int(d["end"].isna().sum())
        opd = pd.to_numeric(d["t_mix"], errors="coerce").sum()
        # Observed ГТМ/ППР interval (install → planned pull) — descriptive, not
        # censoring-corrected.  Both clocks reported.
        gtm_cal = pd.to_numeric(d.loc[gtm_mask, clock], errors="coerce")
        gtm_op = pd.to_numeric(d.loc[gtm_mask, "t_mix"], errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_cs = float(WeibullFitter().fit(d[clock], d[EVENT_COL]).median_survival_time_) if n_fail >= 8 else np.nan
            med_ac = float(WeibullFitter().fit(d[clock], d["end"].notna().astype(int)).median_survival_time_) if d["end"].notna().sum() >= 8 else np.nan
        rows.append({
            "contractor": cg, "n_runs": n, "n_failure": n_fail, "n_gtm_ppr": n_gtm, "n_running": n_run,
            "pct_failure": round(100 * n_fail / n, 1), "pct_gtm_ppr": round(100 * n_gtm / n, 1),
            "fail_rate_per_1000_opd": round(1000 * n_fail / opd, 2) if opd else np.nan,
            "median_cause_specific": round(med_cs, 0) if np.isfinite(med_cs) else np.nan,
            "median_all_cause": round(med_ac, 0) if np.isfinite(med_ac) else np.nan,
            "gtm_interval_mean_cal": round(float(gtm_cal.mean()), 0) if len(gtm_cal) else np.nan,
            "gtm_interval_median_cal": round(float(gtm_cal.median()), 0) if len(gtm_cal) else np.nan,
            "gtm_interval_median_op": round(float(gtm_op.median()), 0) if gtm_op.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def contractor_rate_cohorts(df: pd.DataFrame, *, h2s: str = "nonsour",
                            clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Separate contractor from rate by matching/reversing Ql: high-rate brt vs low-rate
    slb, and the rate-matched overlap band.

    brt (mostly low-rate) and slb (mostly high-rate) are Ql-shifted but overlap plenty in
    100–500; the well-powered band comparison is :func:`contractor_rate_band_comparison`.
    Here the reversal cohorts show that low-rate slb outlives high-rate brt, i.e. the naive
    brt advantage is a rate effect."""
    from lifelines import WeibullFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s) & df["ql"].notna()].copy()
    brt, slb = g[g["contractor_group"] == "brt"], g[g["contractor_group"] == "slb"]
    ql_brt, ql_slb = float(brt["ql"].median()), float(slb["ql"].median())

    def _row(sub, label, definition):
        d = sub.copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0].dropna(subset=[clock])
        n_ev = int(d[EVENT_COL].sum())
        rec = {"cohort": label, "definition": definition, "n_runs": len(d), "n_events": n_ev,
               "ql_median": round(float(d["ql"].median()), 0) if len(d) else np.nan}
        if n_ev >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec["ttf_median"] = round(float(WeibullFitter().fit(d[clock], d[EVENT_COL]).median_survival_time_), 0)
            rec["sufficient"] = "yes"
        else:
            rec["ttf_median"] = np.nan
            rec["sufficient"] = "NO (events<8)"
        return rec

    rows = [
        _row(brt[brt["ql"] > ql_slb], "brt_high_rate", f"brt, Ql>{ql_slb:.0f} (>slb median)"),
        _row(slb[slb["ql"] < ql_brt], "slb_low_rate", f"slb, Ql<{ql_brt:.0f} (<brt median)"),
        _row(brt[(brt["ql"] >= ql_brt) & (brt["ql"] <= ql_slb)], "brt_matched_band",
             f"brt, Ql in [{ql_brt:.0f},{ql_slb:.0f}]"),
        _row(slb[(slb["ql"] >= ql_brt) & (slb["ql"] <= ql_slb)], "slb_matched_band",
             f"slb, Ql in [{ql_brt:.0f},{ql_slb:.0f}]"),
    ]
    return pd.DataFrame(rows)


def contractor_rate_band_comparison(df: pd.DataFrame, *, h2s: str = "nonsour",
                                    clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """brt-vs-slb head-to-head within overlapping Ql bands — the well-powered
    rate-matched comparison.

    brt and slb overlap substantially in 100–500 m³/d; comparing within that band matches
    on rate with enough events to be conclusive.  A band HR ≈ 1 ⇒ no contractor effect
    once rate is held; the raw baseline gap is the rate difference, not the contractor."""
    from lifelines import WeibullFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s) & df["ql"].notna()].copy()

    def _med(sub):
        d = sub[[clock, EVENT_COL]].copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0].dropna()
        if int(d[EVENT_COL].sum()) < 8:
            return len(d), int(d[EVENT_COL].sum()), np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return len(d), int(d[EVENT_COL].sum()), round(float(WeibullFitter().fit(d[clock], d[EVENT_COL]).median_survival_time_), 0)

    rows = []
    for lo, hi in [(100, 500), (200, 500), (100, 200), (200, 300), (300, 500)]:
        band = g[(g["ql"] >= lo) & (g["ql"] < hi)].copy()
        brt, slb = band[band["contractor_group"] == "brt"], band[band["contractor_group"] == "slb"]
        nb, eb, mb = _med(brt)
        ns_, es, ms = _med(slb)
        band["slb"] = (band["contractor_group"] == "slb").astype(float)
        res = fit_cause_specific_cox(band, covariates=["slb"], event_col=EVENT_COL,
                                     duration_col=clock, strata=["field"], cluster_col=CLUSTER, min_events=10)
        hr = round(float(res.summary["hr"].iloc[0]), 2) if res.success else np.nan
        p = round(float(res.summary["p"].iloc[0]), 3) if res.success else np.nan
        rows.append({
            "ql_band": f"{lo}-{hi}", "brt_n": nb, "brt_events": eb, "brt_median": mb,
            "slb_n": ns_, "slb_events": es, "slb_median": ms,
            "hr_slb_vs_brt": hr, "p": p, "n_events": (eb + es),
        })
    return pd.DataFrame(rows)


def contractor_ql_counterfactual(df: pd.DataFrame, *, h2s: str = "nonsour",
                                 clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """How much of the brt/slb median gap does the **Ql hazard alone** reproduce?

    Fits a Ql-only Cox (per estimand), predicts the median at each contractor's mean
    log-Ql, and compares to the observed per-contractor Weibull median.  Reported for the
    cause-specific (failure) and all-cause ("pulled") estimands.  Ql-predicted ≈ observed
    ⇒ the gap is throughput; Ql-predicted much closer to 1 than observed ⇒ the gap is the
    contractor/build package, not Ql."""
    from lifelines import CoxPHFitter, WeibullFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
    g["pulled"] = g["end"].notna().astype(int)
    ql = {cg: float(g[g["contractor_group"] == cg]["log_ql"].mean()) for cg in ("brt", "slb")}
    rows = []
    for est, ev in (("cause_specific", EVENT_COL), ("all_cause_pulled", "pulled")):
        d = g[[clock, ev, "log_ql"]].copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter().fit(d, clock, ev)
            pred = cph.predict_median(pd.DataFrame({"log_ql": [ql["brt"], ql["slb"]]}))
        pb, ps = float(pred.iloc[0]), float(pred.iloc[1])

        def _obs(cg):
            gg = g[g["contractor_group"] == cg][[clock, ev]].copy()
            gg[clock] = pd.to_numeric(gg[clock], errors="coerce")
            gg = gg[gg[clock] > 0].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(WeibullFitter().fit(gg[clock], gg[ev]).median_survival_time_)
        ob, os = _obs("brt"), _obs("slb")
        pr, orr = ps / pb, os / ob
        rows.append({
            "estimand": est, "ql_hr_per_log_unit": round(float(np.exp(cph.params_["log_ql"])), 3),
            "ql_pred_median_brt": round(pb, 0), "ql_pred_median_slb": round(ps, 0),
            "ql_pred_ratio_slb_brt": round(pr, 3),
            "observed_median_brt": round(ob, 0), "observed_median_slb": round(os, 0),
            "observed_ratio_slb_brt": round(orr, 3),
            "pct_gap_explained_by_ql": round(100 * (1 - pr) / (1 - orr), 0) if orr < 1 else np.nan,
        })
    return pd.DataFrame(rows)


def contractor_equipment(df: pd.DataFrame, *, h2s: str = "nonsour",
                         clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Break down the Big «Группа исполнения УЭЦН» (pump execution group) + pump make +
    sizing by contractor — to test whether equipment execution explains the brt/slb gap.

    It does not, because the field is **collinear with the manufacturer**: Borets pumps
    are «Н2/Н3-ЛЧ» + ЭЦНД*ИК* (corrosion/wear-marked in the type string), REDA pumps are
    «N2/N3» + MT/S/D — near-uniform within each contractor, so it re-expresses "different
    maker", not an independent covariate.  The H2/H3 number is not cross-comparable either
    (its within-maker H3-vs-H2 median TTF flips sign), reported here so that is visible."""
    from lifelines import WeibullFitter
    from analysis.data.equipment_big import load_equipment_big
    from analysis.workflows.production_risk import crosswalk

    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy().sort_values("install")
    big = load_equipment_big()
    big = big[big["is_esp_strict"] == True].copy()  # noqa: E712
    big["code"] = big["well_key"].map(crosswalk.norm_well)
    big["install"] = pd.to_datetime(big["install_date"], errors="coerce")
    big = big.dropna(subset=["code", "install"]).sort_values("install")
    ecols = ["pump_exec_group", "pump_exec_lch", "gno_type", "q_nom_m3d",
             "ped_power_kw", "ped_max_temp_c", "stages", "any_corr_protection"]
    m = pd.merge_asof(
        g[["code", "install", "contractor_group", EVENT_COL, clock]],
        big[["code", "install", *ecols]],
        on="install", by="code", tolerance=pd.Timedelta(days=15), direction="nearest")

    def _median_ttf(sub):
        d = sub[[clock, EVENT_COL]].copy()
        d[clock] = pd.to_numeric(d[clock], errors="coerce")
        d = d[d[clock] > 0].dropna()
        if int(d[EVENT_COL].sum()) < 10:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return round(float(WeibullFitter().fit(d[clock], d[EVENT_COL]).median_survival_time_), 0)

    rows = []
    for cg, sub in m.groupby("contractor_group"):
        make = sub["gno_type"].astype(str).str.extract(r"([А-Яа-яA-Za-z]+)")[0].value_counts()
        rows.append({
            "contractor": cg, "n_matched": int(sub["pump_exec_group"].notna().sum()),
            "lch_share": round(float(sub["pump_exec_lch"].mean()), 3) if sub["pump_exec_lch"].notna().any() else np.nan,
            "h3_share": round(float((sub["pump_exec_group"] == "H3").mean()), 3),
            "q_nom_median": round(float(pd.to_numeric(sub["q_nom_m3d"], errors="coerce").median()), 0),
            "ped_power_kw_median": round(float(pd.to_numeric(sub["ped_power_kw"], errors="coerce").median()), 0),
            "ped_max_temp_median": round(float(pd.to_numeric(sub["ped_max_temp_c"], errors="coerce").median()), 0),
            "stages_median": round(float(pd.to_numeric(sub["stages"], errors="coerce").median()), 0),
            "corr_protection_share": round(float((sub["any_corr_protection"] == True).mean()), 3),  # noqa: E712
            "top_pump_family": ", ".join(f"{k}:{v}" for k, v in make.head(3).items()),
            "median_ttf_H2": _median_ttf(sub[sub["pump_exec_group"] == "H2"]),
            "median_ttf_H3": _median_ttf(sub[sub["pump_exec_group"] == "H3"]),
        })
    return pd.DataFrame(rows)


def weibull_shape_by_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Weibull shape β on each clock (calendar / op-time / cumulative volume) by h2s
    class — the aging-vs-infant diagnostic.

    β < 1 = decreasing hazard (infant / early-failure dominated); β > 1 = wear-out.
    Note β stays < ~1 on **every** clock here — the throughput result is about which
    variable *paces* failure, NOT a wear-out shape (which would need β_volume > 1)."""
    from lifelines import WeibullFitter
    rows = []
    for cls in ("nonsour", "sour"):
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == cls)]
        rec = {"h2s_class": cls}
        for clock, name in (("t_cal", "beta_calendar"), ("t_mix", "beta_op_time"), ("t_vol", "beta_cum_volume")):
            d = g[[clock, EVENT_COL]].copy()
            d[clock] = pd.to_numeric(d[clock], errors="coerce")
            d = d[d[clock] > 0].dropna()
            if int(d[EVENT_COL].sum()) < 8:
                rec[name] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec[name] = round(float(WeibullFitter().fit(d[clock], d[EVENT_COL]).rho_), 3)
        rec["shape_reading"] = ("infant/early-failure (all β<1)" if cls == "nonsour"
                                else "near-constant on calendar (β≈1); no wear-out on any clock")
        rows.append(rec)
    return pd.DataFrame(rows)


def sour_effect_interaction(df: pd.DataFrame, *, covariates: list[str] | None = None,
                            clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Test whether each covariate's effect **differs** between sour and nonsour, and
    return the pieces the shrinkage needs.

    Fits ``z + z·I(sour)`` stratified by h2s×contractor: main = nonsour effect,
    interaction δ = sour − nonsour.  Non-significant δ ⇒ the data cannot reject a shared
    effect (borrowing the nonsour trend for sour is honest); significant δ ⇒ sour truly
    differs.  ``z`` is standardised over all Vt so main and interaction share a scale;
    ``sd`` is returned to convert back to natural units."""
    covariates = covariates or EQUATION_COVARIATES
    rows = []
    for cov in covariates:
        d = df[(df["field"] == FIELD) & df[cov].notna()].copy()
        d["z"], _, sd = _zscore(d[cov])
        d["z_sour"] = d["z"] * (d["h2s_class"] == "sour").astype(float)
        res = fit_cause_specific_cox(
            d, covariates=["z", "z_sour"], event_col=EVENT_COL, duration_col=clock,
            strata=["stratum"], cluster_col=CLUSTER, min_events=10)
        if not res.success:
            rows.append({"covariate": cov, "note": res.message})
            continue
        m = res.summary.set_index("covariate")
        _, step = NATURAL_UNITS.get(cov, ("per +1 unit", 1.0))
        rows.append({
            "covariate": cov, "sd": round(float(sd), 4),
            "nonsour_hr_sd": round(float(m.loc["z", "hr"]), 3), "nonsour_p": round(float(m.loc["z", "p"]), 4),
            "beta_ns_z": float(m.loc["z", "coef"]),
            "interaction_coef": round(float(m.loc["z_sour", "coef"]), 4),
            "interaction_se": float(m.loc["z_sour", "se"]),
            "interaction_p": round(float(m.loc["z_sour", "p"]), 4),
            "step": step,
            "sour_differs": "yes" if m.loc["z_sour", "p"] < 0.05 else "no (shared effect not rejected)",
        })
    return pd.DataFrame(rows)


def shrink_sour_toward_pooled(interaction: pd.DataFrame, shape_sour: float) -> pd.DataFrame:
    """Honest way to keep sour covariates in the equation: **positive-part James–Stein
    shrinkage of the sour deviation δ toward 0** (i.e. sour toward the nonsour trend).

    ``δ_EB = δ · max(0, 1 − se_δ²/δ²)`` — driven directly by the interaction test, so a
    non-significant δ (freq) collapses toward the nonsour effect (trend retained,
    attenuated) while a significant δ (Kpod/Ql) keeps sour near its own ~null. The sour
    coefficient is then ``β_nonsour + δ_EB``, always the same sign as nonsour, attenuated
    by exactly how much the data says sour differs."""
    rows = []
    for _, r in interaction.iterrows():
        if "beta_ns_z" not in r or pd.isna(r.get("beta_ns_z")):
            continue
        b_ns, delta, se_d = float(r["beta_ns_z"]), float(r["interaction_coef"]), float(r["interaction_se"])
        sd, step = float(r["sd"]), float(r["step"])
        js = max(0.0, 1.0 - (se_d ** 2) / (delta ** 2)) if delta != 0 else 0.0
        delta_eb = delta * js
        b_sour_z = b_ns + delta_eb
        hr_ns = float(np.exp(b_ns / sd * step))
        hr_sour = float(np.exp(b_sour_z / sd * step))
        life = hr_sour ** (-1.0 / shape_sour) if shape_sour > 0 else np.nan
        rows.append({
            "covariate": r["covariate"], "natural_unit": NATURAL_UNITS.get(r["covariate"], ("", 1))[0],
            "nonsour_hr_per_step": round(hr_ns, 4),
            "interaction_p": r["interaction_p"], "js_keep_deviation": round(js, 3),
            "sour_shrunk_hr_per_step": round(hr_sour, 4),
            "sour_shrunk_life_mult_per_step": round(life, 4) if np.isfinite(life) else np.nan,
            "verdict": ("trend retained (δ n.s.)" if r["interaction_p"] >= 0.05
                        else "attenuated (δ significant, sour differs)"),
        })
    return pd.DataFrame(rows)


def clock_mechanism_test(df: pd.DataFrame, *, covariates: list[str] | None = None) -> pd.DataFrame:
    """Cumulative throughput vs exposure time: fit each operating covariate on three
    clocks — calendar (`t_cal`), operating time (`t_mix`), cumulative volume (`t_vol`) —
    within each h2s class.

    Reading: an effect that **persists on op-time but vanishes on the volume clock** is a
    *cumulative-throughput* (wear/work) effect — failures sit at a fixed volume budget and
    rate only fills it faster. An effect that vanishes already on op-time is *exposure*.
    (Ql is mechanically inside t_vol, so read freq/Kpod as the clean discriminators.)"""
    covariates = covariates or EQUATION_COVARIATES
    rows = []
    for cls, sub in [("sour", df[(df["field"] == FIELD) & (df["h2s_class"] == "sour")]),
                     ("nonsour", df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour")])]:
        work = sub.copy()
        work["stratum"] = work["contractor_group"]
        for cov in covariates:
            for clock in (CLOCK_PRIMARY, CLOCK_SENS, "t_vol"):
                d = _prep_cox(work.assign(_c=work[cov]), ["_c"], clock=clock)
                if int(d[EVENT_COL].sum()) < 10:
                    rows.append({"h2s_class": cls, "covariate": cov, "clock": clock, "note": "thin"})
                    continue
                d["z"], _, _ = _zscore(d["_c"])
                res = fit_cause_specific_cox(
                    d.drop(columns="_c"), covariates=["z"], event_col=EVENT_COL,
                    duration_col=clock, strata=["stratum"], cluster_col=CLUSTER, min_events=10)
                if res.success and not res.summary.empty:
                    r = res.summary.iloc[0]
                    rows.append({"h2s_class": cls, "covariate": cov,
                                 "clock": {"t_cal": "calendar", "t_mix": "op_time", "t_vol": "cum_volume"}[clock],
                                 "hr_per_sd": round(float(r["hr"]), 3), "p": round(float(r["p"]), 4),
                                 "n_events": res.n_events})
                else:
                    rows.append({"h2s_class": cls, "covariate": cov, "clock": clock, "note": res.message})
    return pd.DataFrame(rows)


RMST_HORIZON_D = 730  # planning horizon τ for RMST(0, τ); ~2 yr fleet-planning window


def _ttf_quantities_block(revised: pd.DataFrame, tau: float = RMST_HORIZON_D) -> str:
    """Derive the covariate-explicit TTF reporting quantities from the v3.1 params.

    v3.1 is an AFT model: every covariate enters through a single time-scale factor
    ``φ(x) = contractor_mult · (clip(Ql,lo,hi)/Ql_ref)^γ`` (φ = 1 at the brt reference,
    Ql = Ql_ref), so ``η(x) = η_ref·φ(x)``.  Hence

        median(x) = φ(x) · η_ref · (ln2)^(1/β)
        MRL(0|x)  = E[T|x]      = φ(x) · η_ref · Γ(1 + 1/β)          (mean; full tail)
        RMST(0,τ|x) = ∫₀^τ S    = φ(x) · η_ref · Γ(1 + 1/β) · P(1/β, (τ/(η_ref·φ(x)))^β)
                                = φ(x) · RMST_ref(0, τ/φ(x))

    where P is the regularized lower incomplete gamma (``scipy.special.gammainc``).
    median and MRL(0) scale **linearly** in φ(x); RMST(0,τ) does not — the horizon τ is
    fixed, so longer-lived wells push more life past τ and RMST saturates below φ·mean.
    Returns a markdown block (equations + a per-stratum table at Ql = Ql_ref)."""
    from scipy.special import gamma as _gamma, gammainc as _gammainc

    ql = revised[revised["term"] == "Ql"].set_index("h2s_class")
    con = revised[revised["term"].str.startswith("contractor")]
    rows = []
    for cls, r in ql.iterrows():
        beta = float(r["weibull_beta"]); eta_ref = float(r["eta_ref_d"])
        mean_ref = eta_ref * _gamma(1 + 1 / beta)
        med_ref = eta_ref * np.log(2) ** (1 / beta)
        mults = {"brt": 1.0}
        for _, cr in con[con["h2s_class"] == cls].iterrows():
            mults[cr["term"].split(":", 1)[1]] = float(cr["life_mult"])
        for name, phi in mults.items():  # φ = contractor_mult at Ql = Ql_ref
            eta = eta_ref * phi
            rows.append({
                "h2s_class": cls, "contractor": name, "phi": round(phi, 3),
                "median_d": round(phi * med_ref),
                "MRL0_mean_d": round(phi * mean_ref),
                f"RMST_0_{int(tau)}d": round(phi * mean_ref * _gammainc(1 / beta, (tau / eta) ** beta)),
            })
    tbl = pd.DataFrame(rows)
    return "\n".join([
        f"### TTF reporting quantities — MRL(0) and RMST(0, τ={int(tau)} d), covariate-explicit\n",
        "All three derive from `S(t|x)=exp(−(t/η(x))^β)` with the **AFT time-scale**",
        "`φ(x)=contractor_mult·(clip(Ql,lo,hi)/Ql_ref)^γ` (φ=1 at brt, Ql=Ql_ref):\n",
        "```text",
        "median(x)   = φ(x) · η_ref · (ln2)^(1/β)",
        "MRL(0|x)    = φ(x) · η_ref · Γ(1+1/β)                       (mean; whole tail)",
        "RMST(0,τ|x) = φ(x) · η_ref · Γ(1+1/β) · P(1/β, (τ/(η_ref·φ(x)))^β)",
        "            = φ(x) · RMST_ref(0, τ/φ(x))     [P = regularized lower incomplete Γ]",
        "```",
        "median and MRL(0) scale **linearly** in φ; RMST(0,τ) is sub-linear — τ is fixed, so",
        "the longer-lived wells push more life beyond τ and RMST saturates below φ·mean.",
        f"**Report RMST(0,{int(tau)} d) as the headline TTF** (no tail extrapolation, always",
        "estimable), MRL(0)=mean beside it (full expected life), median as descriptor only.\n",
        _md_table(tbl),
    ])


def write_ttf_equation(baseline: pd.DataFrame, coef: pd.DataFrame, path: Path,
                       shrink: pd.DataFrame | None = None,
                       interaction: pd.DataFrame | None = None,
                       revised: pd.DataFrame | None = None) -> None:
    """Write the per-stratum TTF-adjustment equation deliverable (markdown)."""
    lm = coef[coef["h2s_class"].isin(("sour", "nonsour")) & coef.get("life_mult_per_step").notna()]
    lines = ["# Vt TTF adjustment — baselines, coefficients, equation\n"]

    if revised is not None and not revised.empty:
        base = revised[revised["term"] == "Ql"]
        lines += [
            "## ⭐ RECOMMENDED MODEL (v3.1) — baseline × capped-log-Ql × contractor\n",
            "**Structure:** `TTF(well) = M0(h2s) · (clip(Ql, lo, hi)/Ql_ref)^γ · contractor_mult`.",
            "Ql (throughput) is the operating driver; the shape is **log with limits** — Ql",
            "winsorized at the class q05–q95 (`ql_cap_lo/hi`) before the log.  Chosen by AIC over",
            "linear/log/quadratic/spline/hinge; capping removes the low-Ql leverage and the >q95",
            "saturation, and its γ matches the empirical binned-median slope.  Outside the caps the",
            "effect is held FLAT (no extrapolation beyond observed support).  A **contractor",
            "covariate** absorbs the brt/slb residual that Ql alone under-predicts (mechanism NOT",
            "identified, n.s. on nonsour, plausibly informative censoring — brt is more heavily",
            "ГТМ-censored); included **empirically for calibration**, labelled unexplained.\n",
            "M0 is the median at Ql_ref for the reference contractor (brt); `contractor_mult` is the",
            "life multiplier vs brt.  Fitted per h2s class (Weibull AFT, capped `log_ql + contractor`).",
            "Same-sample calibration: 4/6 strata within ±10% (`model_consistency_checks.csv`,",
            "`model_validation_v31.png`); thin oth groups flagged.\n",
            "**Full survival function** (needs the Weibull params below):",
            "`S(t|x) = exp( −(t / η(x))^β )`, `η(x) = η_ref · (clip(Ql,lo,hi)/Ql_ref)^γ · contractor_mult`,",
            "`γ = ln(life_mult)`, β and η_ref from the Ql rows.  Median = η(x)·(ln2)^(1/β).\n",
            "> β here is the **conditional** Weibull shape (covariates held): nonsour β≈1.15,",
            "> sour β≈1.37 — near-constant, NOT the marginal per-stratum β<1 of §2/table 1",
            "> (that <1 is frailty across wells, not a within-profile aging shape).\n",
            _md_table(revised[revised["term"] == "Ql"][
                ["h2s_class", "weibull_beta", "eta_ref_d", "ql_ref_m3d", "ql_cap_lo", "ql_cap_hi",
                 "baseline_median_d", "life_mult", "ci", "p"]]),
            "\n" + _md_table(revised[revised["term"].str.startswith("contractor")][
                ["h2s_class", "term", "life_mult", "ci", "p"]]),
            "\n" + _ttf_quantities_block(revised),
            "\n> ⚠ **Caveat.** Ql is a directional trend with modest individual predictive power",
            "> (concordance ≈0.58). The capping fixed the earlier median-slope miscalibration — γ",
            "> now matches the empirical binned-median slope (−0.31 vs −0.32). The contractor",
            "> multiplier is empirical, mechanism ~⅓ ГТМ-censoring / rest unattributed",
            "> (`informative_censoring_test.csv`); the Ql slope and contractor multiplier both",
            "> **replicate on Ya** (`replicate_v31_on_ya.csv`); γ transfers out-of-sample but M0",
            "> drifts with install vintage (`temporal_holdout_validation.csv`). Report",
            "> RMST(0)+MRL(0), never B50.\n",
            "The per-stratum v1 (marginal) tables below are retained for reference / diagnostics.\n",
            "---\n",
        ]

    lines += ["## Clock\n",
             "Every baseline and coefficient here is fitted on **`t_cal` (calendar days)** —",
             "uniformly — so the baseline and the covariate multipliers share one clock and",
             "the equation is self-consistent. The `shipped_clock` column in table 1 shows what",
             "each stratum uses in the *production forecast bundle* (esp_models), which mixes",
             "clocks per stratum; that is a different deliverable, not used here.\n",
             "## Equation\n",
             "For a run in stratum *s* with operating covariates X (only those actually",
             "available for the well — see the availability table), the covariate-adjusted",
             "time-to-failure is a multiplicative shift of that stratum's baseline:\n",
             "```text",
             "TTF(X) = TTF_base(s) * PROD_i  m_i(class_s) ^ ((X_i - X_ref,i(s)) / step_i)",
             "",
             "  TTF_base(s)  = stratum baseline Weibull mean or median (table 1)",
             "  m_i          = life multiplier per natural step (table 2 nonsour; table 3 sour)",
             "  step_i       = freq: 1 Hz | Kpod: 0.1 | log_ql: 1 log-unit",
             "  X_ref,i(s)   = stratum-mean covariate (table 1); no shift at the reference",
             "```",
             "Equivalently TTF(X) = TTF_base * exp( SUM_i gamma_i (X_i - X_ref,i) ), gamma_i = ln(m_i)/step_i.\n",
             "## Table 1 — per-stratum baseline survival + TTF + covariate reference\n",
             _md_table(baseline),
             "\n## Table 2 — life multiplier per natural step, NONSOUR (the measured effect)\n",
             _md_table(lm[lm["h2s_class"] == "nonsour"]) if not lm.empty else "(none)"]

    if shrink is not None and not shrink.empty:
        lines += [
            "\n## Table 3 — SOUR multipliers (partial-pooling / shrinkage)\n",
            "Sour's *own* fit is null, but keeping the covariates in the sour equation is",
            "honest only where the sour-vs-nonsour difference is not significant. The sour",
            "coefficient is **β_nonsour + δ_EB**, where the sour deviation δ is shrunk by",
            "positive-part James–Stein (`δ_EB = δ·max(0, 1 − se²/δ²)`), driven by the",
            "interaction test: **freq keeps the nonsour trend** (δ n.s.) while **Kpod / Ql**",
            "**attenuate to ~1** (δ significant — the data prefers no sour effect).",
            "`js_keep_deviation`=0 ⇒ sour collapses to the nonsour effect, 1 ⇒ keeps its own",
            "deviation. Use `sour_shrunk_life_mult_per_step` as m_i for sour.\n",
            _md_table(shrink),
        ]
    if interaction is not None and not interaction.empty:
        lines += ["\n## Sour-vs-nonsour interaction test (the honesty gate for table 3)\n",
                  _md_table(interaction)]

    lines += [
        "\n## Worked example\n",
        "nonsour_brt pump +5 Hz above its stratum mean → median TTF ≈ 573 · 0.965⁵ ≈ 479 d.",
        "A sour pump +5 Hz: freq keeps a (small) trend via table 3; Kpod/Ql ≈ ×1.",
        "\nReporting: RMST(0) + MRL(0)/median, never B50. All multipliers observational",
        "(wells not randomised to speed/rate) — a relation, not a causal lever.\n"]
    path.write_text("\n".join(lines), encoding="utf-8")


def underload_harm_test(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY,
                        frac_col: str = "frac_days_kpod_below") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is chronic **low Kpod** (много суток с Kpod<0.7) harmful? — the direct test.

    The physics prior says sustained underloading (downthrust / heating / gas locking)
    should shorten life.  This fits the guarded ``frac_days_kpod_below_0.7`` dose as a
    hazard covariate — alone and **holding the mean level, running frequency and Ql
    fixed** — so a genuine underload-instability effect would show as a *positive* HR
    even after the productivity confound (low Kpod ↔ low-rate well) is adjusted out.

    Returns ``(binned, cox)``: a KM/rate table by frac-below band and the per-spec HR
    (per +10 % of days below 0.7)."""
    binned = binned_table(df, frac_col, [0, 0.25, 0.5, 0.75, 1.0001],
                          ["0-25%", "25-50%", "50-75%", "75-100%"], clock=clock)
    specs = {
        "frac_below_alone": [frac_col],
        "adj_mean_kpod": [frac_col, "kpod_run"],
        "adj_freq": [frac_col, "freq_run"],
        "adj_ql": [frac_col, "log_ql"],
        "adj_all": [frac_col, "kpod_run", "freq_run", "log_ql"],
    }
    rows = []
    for name, cols in specs.items():
        s = stratified_cox(df, cols, clock=clock)
        r = s[s.get("covariate_raw", "") == frac_col]
        if r.empty:
            rows.append({"spec": name, "note": "no fit"})
            continue
        rr = r.iloc[0]
        rows.append({"spec": name, "hr_per_10pct_days": round(float(rr["hr_per_natural"]), 4),
                     "p": round(float(rr["p"]), 4), "n_events": int(rr["n_events"]),
                     "direction": "HARMFUL" if rr["hr_per_natural"] > 1 else "protective"})
    return binned, pd.DataFrame(rows)


def sour_threshold_report(df: pd.DataFrame) -> pd.DataFrame:
    """H2S proxy distribution by the Свод sour label + separation at candidate
    thresholds — shows the effective sour boundary is ~120 mg/l, not 10."""
    g = df[(df["field"] == FIELD) & df["h2s_mg_l"].notna()]
    rows = []
    for cls, sub in g.groupby("h2s_class"):
        v = sub["h2s_mg_l"]
        rows.append({"h2s_class": cls, "n": len(v),
                     "q05": round(float(v.quantile(0.05)), 1), "q50": round(float(v.quantile(0.5)), 1),
                     "q95": round(float(v.quantile(0.95)), 1), "max": round(float(v.max()), 1)})
    dist = pd.DataFrame(rows)
    sour = g[g["h2s_class"] == "sour"]["h2s_mg_l"]
    nons = g[g["h2s_class"] == "nonsour"]["h2s_mg_l"]
    sep = pd.DataFrame([
        {"threshold_mg_l": t, "sour_above_pct": round(100 * (sour > t).mean(), 1),
         "nonsour_above_pct": round(100 * (nons > t).mean(), 1)}
        for t in (10, 50, 100, 120, 200, 300)
    ])
    dist["_kind"], sep["_kind"] = "distribution", "separation"
    return pd.concat([dist, sep], ignore_index=True)


#: Candidate H2S thresholds (mg/l) for the prevalence sweep.
H2S_THRESHOLDS = (10, 50, 100, 120, 200, 300, 500, 700)


def h2s_well_prevalence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of **wells** (unique, not runs) above each H2S threshold — how prevalent
    the exposure actually is, and how little the exact cutoff matters.

    Returns ``(vt_by_threshold, by_scope)``.  Per well the proxy is ~constant
    (well/pad median), so wells are deduped by code and averaged.  Coverage is made
    explicit: ``pct_of_data`` (among wells with an H2S measurement) vs ``pct_of_all``
    (missing counted as below threshold).  ``by_scope`` gives Vt-by-contractor and the
    per-field context at the 10 and 120 mg/l cutoffs."""
    def _wells(g):
        return g.groupby("code").agg(
            h2s=("h2s_mg_l", "mean"),
            sour=("h2s_class", lambda s: (s == "sour").any())).reset_index()

    vt = _wells(df[df["field"] == FIELD])
    n_all, n_data = len(vt), int(vt["h2s"].notna().sum())
    wd = vt[vt["h2s"].notna()]
    rows = []
    for t in H2S_THRESHOLDS:
        rows.append({
            "threshold_mg_l": t,
            "pct_wells_gt_of_data": round(100 * (wd["h2s"] > t).mean(), 1),
            "pct_wells_gt_of_all": round(100 * (vt["h2s"].fillna(0) > t).mean(), 1),
        })
    vt_by_threshold = pd.DataFrame(rows)
    vt_by_threshold.attrs["n_wells"] = n_all
    vt_by_threshold.attrs["n_with_data"] = n_data

    scope_rows = []
    # Vt by contractor
    for cg, g in df[df["field"] == FIELD].groupby("contractor_group"):
        w = _wells(g); wdc = w[w["h2s"].notna()]
        scope_rows.append({
            "scope": f"Vt_{cg}", "n_wells": len(w), "n_with_data": len(wdc),
            "coverage_pct": round(100 * len(wdc) / max(len(w), 1), 0),
            "sour_labeled_pct": round(100 * w["sour"].mean(), 1),
            "pct_gt_10_of_data": round(100 * (wdc["h2s"] > 10).mean(), 1) if len(wdc) else np.nan,
            "pct_gt_120_of_data": round(100 * (wdc["h2s"] > 120).mean(), 1) if len(wdc) else np.nan,
            "median_h2s": round(float(wdc["h2s"].median()), 1) if len(wdc) else np.nan,
        })
    # field context
    for fld, g in df.groupby("field"):
        w = _wells(g); wdc = w[w["h2s"].notna()]
        if len(wdc) < 10:
            continue
        scope_rows.append({
            "scope": f"field:{fld}", "n_wells": len(w), "n_with_data": len(wdc),
            "coverage_pct": round(100 * len(wdc) / max(len(w), 1), 0),
            "sour_labeled_pct": round(100 * w["sour"].mean(), 1),
            "pct_gt_10_of_data": round(100 * (wdc["h2s"] > 10).mean(), 1),
            "pct_gt_120_of_data": round(100 * (wdc["h2s"] > 120).mean(), 1),
            "median_h2s": round(float(wdc["h2s"].median()), 1),
        })
    return vt_by_threshold, pd.DataFrame(scope_rows)


def h2s_within_sour(df: pd.DataFrame, *, clock: str = CLOCK_PRIMARY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within the SOUR stratum, does MORE H2S shorten life? (concentration gradient
    above the sour threshold — the flag already carries the threshold effect).

    Uses well/pad-median proxy rows only; stratifies by contractor (h2s_class is
    constant = sour here).  Returns ``(binned, cox)``."""
    s = df[(df["field"] == FIELD) & (df["h2s_class"] == "sour") & df["log_h2s"].notna()].copy()
    s["stratum"] = s["contractor_group"]        # within sour, only contractor is left
    if len(s) < 20:
        return pd.DataFrame(), pd.DataFrame([{"note": f"only {len(s)} sour runs with h2s"}])
    s["_hb"] = pd.qcut(s["h2s_mg_l"], 3, labels=["low", "mid", "high"])
    brows = []
    for lab, sub in s.groupby("_hb", observed=True):
        rmst, mrl = km_rmst_mrl(sub[clock].to_numpy(), sub[EVENT_COL].to_numpy())
        exp = float(sub[clock].sum())
        brows.append({"h2s_band": lab, "h2s_med": round(float(sub["h2s_mg_l"].median())),
                      "n": len(sub), "n_events": int(sub[EVENT_COL].sum()), "km_rmst0": rmst,
                      "rate_per_1000_opd": round(1000 * sub[EVENT_COL].sum() / exp, 2) if exp else np.nan})
    rows = []
    for name, cols in {"h2s_alone": ["log_h2s"], "h2s+freq": ["log_h2s", "freq_run"],
                       "h2s+kpod+ql": ["log_h2s", "kpod_run", "log_ql"]}.items():
        res = fit_cause_specific_cox(
            _prep_cox(s, cols, clock=clock), covariates=cols, event_col=EVENT_COL,
            duration_col=clock, strata=["stratum"], cluster_col=CLUSTER, min_events=10)
        if res.success and not res.summary[res.summary["covariate"] == "log_h2s"].empty:
            r = res.summary[res.summary["covariate"] == "log_h2s"].iloc[0]
            rows.append({"spec": name, "hr_per_efold_h2s": round(float(r["hr"]), 3),
                         "p": round(float(r["p"]), 4), "n_events": res.n_events})
        else:
            rows.append({"spec": name, "note": res.message})
    return pd.DataFrame(brows), pd.DataFrame(rows)


def schoenfeld_ph(df: pd.DataFrame, covariates: list[str], *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Per-covariate Schoenfeld PH test (plan §4 diagnostics)."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    work = _prep_cox(df, covariates, clock=clock)
    for c in covariates:
        work[c], _, _ = _zscore(work[c])
    fit_df = work[covariates + [clock, EVENT_COL, "stratum"]].copy()
    cph = CoxPHFitter(penalizer=0.01)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(fit_df, duration_col=clock, event_col=EVENT_COL, strata=["stratum"])
            zph = proportional_hazard_test(cph, fit_df, time_transform="rank")
    except Exception as exc:
        return pd.DataFrame([{"note": f"PH test failed: {exc}"}])
    r = zph.summary.reset_index().rename(columns={"index": "covariate"})
    return r[["covariate", "test_statistic", "p"]]


def vif_correlation(df: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VIF per covariate + Pearson correlation matrix on the complete-case subset."""
    work = df[covariates].dropna()
    corr = work.corr()
    vifs = []
    X = ((work - work.mean()) / work.std(ddof=0)).to_numpy()
    for i, c in enumerate(covariates):
        others = np.delete(X, i, axis=1)
        y = X[:, i]
        try:
            beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), others]), y, rcond=None)
            yhat = np.column_stack([np.ones(len(y)), others]) @ beta
            r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
            vifs.append(1.0 / max(1 - r2, 1e-9))
        except Exception:
            vifs.append(np.nan)
    return pd.DataFrame({"covariate": covariates, "vif": np.round(vifs, 2)}), corr.round(3)


# ===========================================================================
# Deliverable (plan §5): PH-route + Weibull AFT life multipliers
# ===========================================================================

def _load_vt_baselines() -> pd.DataFrame:
    p = Path(ESP_MODELS_REF)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[4] / ESP_MODELS_REF
    em = pd.read_csv(p)
    return em[em["field"] == FIELD].copy()


def _weibull_life(beta: float, eta: float, horizon: float) -> tuple[float, float, float]:
    """(RMST(0) to horizon, MRL(0)=mean, median) of a single Weibull S=exp(-(t/eta)^beta)."""
    from math import gamma
    t = np.linspace(0, horizon, 4000)
    S = np.exp(-((t / eta) ** beta))
    rmst = float(np.trapz(S, t))
    mean = float(eta * gamma(1 + 1 / beta))
    median = float(eta * (np.log(2)) ** (1 / beta))
    return round(rmst, 1), round(mean, 1), round(median, 1)


def ph_life_multipliers(cox_summary: pd.DataFrame, *, horizon: float = 1500.0) -> pd.DataFrame:
    """PH route (plan §5.1): for a single-Weibull baseline, S(t|dX)=S0(t)^exp(b.dX) is
    Weibull with the same shape and eta' = eta.exp(b.dX)^(-1/shape), so the life
    multiplier (RMST-, mean-, median-ratio) is exactly ``exp(b_cox.d)^(-1/b_weibull)``.

    Uses each Vt stratum's shipped single-Weibull baseline; ``cox_summary`` supplies
    per-natural-unit b (global stratified fit)."""
    base = _load_vt_baselines()
    single = base[base["w1"] == 0.0]  # single-Weibull strata (all per-stratum Vt rows)
    rows = []
    cs = cox_summary[cox_summary["scope"] == "global"] if "scope" in cox_summary else cox_summary
    for _, cr in cs.iterrows():
        raw = cr.get("covariate_raw")
        beta_cox = float(cr.get("beta_per_unit", np.nan))
        unit_lbl, step = NATURAL_UNITS.get(raw, ("per +1 unit", 1.0))
        for _, b in single.iterrows():
            shape = float(b["beta1"])
            hr = float(np.exp(beta_cox * step))
            mult = hr ** (-1.0 / shape)   # life multiplier per natural step
            rows.append({
                "covariate": raw, "stratum": b["stratum"], "natural_unit": unit_lbl,
                "weibull_shape": round(shape, 3), "hr_per_step": round(hr, 4),
                "ph_life_multiplier": round(mult, 4),
            })
    return pd.DataFrame(rows)


def fit_aft(df: pd.DataFrame, covariates: list[str], *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Weibull AFT companion (plan §5.2): eta(X)=eta0.exp(g.X), g IS the life multiplier
    exponent -- the most readable simple relation.  Stratum entered as design dummies
    (AFT has no free per-stratum baseline; the plan compares AFT vs PH-derived)."""
    from lifelines import WeibullAFTFitter

    work = _prep_cox(df, covariates, clock=clock).copy()
    steps = {}
    for c in covariates:
        _, step = NATURAL_UNITS.get(c, ("per +1 unit", 1.0))
        steps[c] = step
    dummies = pd.get_dummies(work["stratum"], prefix="str", drop_first=True).astype(float)
    X = pd.concat([work[covariates + [clock, EVENT_COL]], dummies], axis=1)
    aft = WeibullAFTFitter(penalizer=0.01)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft.fit(X, duration_col=clock, event_col=EVENT_COL)
    except Exception as exc:
        return pd.DataFrame([{"note": f"AFT failed: {exc}"}])
    lam = aft.summary.xs("lambda_", level=0)  # log-scale (eta) coefficients
    rows = []
    for c in covariates:
        if c not in lam.index:
            continue
        gamma_per_unit = float(lam.loc[c, "coef"])
        step = steps[c]
        rows.append({
            "covariate": c, "natural_unit": NATURAL_UNITS.get(c, ("per +1 unit", 1.0))[0],
            "gamma_per_unit": round(gamma_per_unit, 5),
            "aft_life_multiplier": round(float(np.exp(gamma_per_unit * step)), 4),
            "p": round(float(lam.loc[c, "p"]), 4),
        })
    return pd.DataFrame(rows)


def formula_table(cox_summary: pd.DataFrame, aft: pd.DataFrame, ph: pd.DataFrame,
                  support: pd.DataFrame) -> pd.DataFrame:
    """Final one-row-per-covariate deliverable: TTF multiplier ~ exp(g.dX) with CI,
    natural units, stratum validity, support range, and the observational caveat.

    AFT vs PH-derived compared; disagreement >15% => PH violated, quote AFT (plan §5.2)."""
    cs = cox_summary[cox_summary.get("scope", "global") == "global"].copy()
    ph_agg = (ph.groupby("covariate")["ph_life_multiplier"].mean().to_dict()
              if not ph.empty and "covariate" in ph else {})
    aft_map = aft.set_index("covariate")["aft_life_multiplier"].to_dict() if "covariate" in aft else {}
    sup = support.set_index("covariate") if not support.empty and "covariate" in support else pd.DataFrame()
    rows = []
    for _, r in cs.iterrows():
        raw = r.get("covariate_raw")
        if raw is None:
            continue
        aft_m = aft_map.get(raw, np.nan)
        ph_m = ph_agg.get(raw, np.nan)
        disagree = (abs(aft_m - ph_m) / abs(ph_m) if np.isfinite(aft_m) and np.isfinite(ph_m) and ph_m else np.nan)
        quote = "AFT (PH violated)" if (np.isfinite(disagree) and disagree > 0.15) else "AFT/PH agree"
        srange = ""
        if not sup.empty and raw in sup.index:
            srange = f"[{sup.loc[raw, 'q05']:.2f}, {sup.loc[raw, 'q95']:.2f}]"
        rows.append({
            "covariate": raw, "natural_unit": r.get("natural_unit"),
            "hr_per_step": round(float(r.get("hr_per_natural", np.nan)), 4),
            "aft_life_multiplier": round(aft_m, 4) if np.isfinite(aft_m) else np.nan,
            "ph_life_multiplier": round(ph_m, 4) if np.isfinite(ph_m) else np.nan,
            "aft_vs_ph_disagree": round(disagree, 3) if np.isfinite(disagree) else np.nan,
            "quote": quote, "p": round(float(r.get("p", np.nan)), 4),
            "support_q05_q95": srange,
            "caveat": "observational -- wells not randomized to this covariate; may confound with well productivity/completion",
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Honesty gates (plan §6)
# ===========================================================================

def replicate(df: pd.DataFrame, covariates: list[str], *, clock: str = CLOCK_PRIMARY) -> pd.DataFrame:
    """Re-check each covariate on Ya and Mc(2024+), same stratified design (plan §6).

    A Vt effect that fails replication is reported Vt-specific-unconfirmed, not
    generalized (``project_hazard_catboost`` precedent: Kpod/Ql died on replication)."""
    out = []
    for fld in ("Vt", "Ya", "Mc"):
        g = df[df["field"] == fld].copy()
        if fld in C.MC_COHORT_FIELDS:
            g = g[g["install"] >= pd.Timestamp(C.MC_INSTALL_COHORT_START)]
        for cov in covariates:
            s = stratified_cox(g, [cov], clock=clock)
            row = s[s.get("scope", "global") == "global"]
            if row.empty or "hr_per_natural" not in row:
                out.append({"field": fld, "covariate": cov, "note": "no fit / thin"})
                continue
            r = row.iloc[0]
            out.append({
                "field": fld, "covariate": cov, "n_events": int(r.get("n_events", 0)),
                "hr_per_natural": round(float(r.get("hr_per_natural", np.nan)), 4),
                "beta_per_unit": round(float(r.get("beta_per_unit", np.nan)), 5),
                "p": round(float(r.get("p", np.nan)), 4),
            })
    res = pd.DataFrame(out)
    # Flag sign agreement with Vt.
    if "beta_per_unit" in res:
        vt = res[res["field"] == "Vt"].set_index("covariate")["beta_per_unit"].to_dict()
        res["sign_matches_vt"] = [
            (np.sign(b) == np.sign(vt.get(c, np.nan))) if pd.notna(b) and c in vt else np.nan
            for c, b in zip(res.get("covariate", []), res.get("beta_per_unit", []))
        ]
    return res


def support_overlap(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Quantile support per stratum (NOT min..max -- ``support_overlap`` false-positive)."""
    rows = []
    g = df[df["field"] == FIELD]
    for cov in covariates:
        for stratum, sub in g.groupby("stratum"):
            v = sub[cov].dropna()
            if len(v) < 5:
                rows.append({"covariate": cov, "stratum": stratum, "n": len(v)})
                continue
            rows.append({
                "covariate": cov, "stratum": stratum, "n": len(v),
                "q05": round(float(v.quantile(0.05)), 3),
                "q50": round(float(v.quantile(0.50)), 3),
                "q95": round(float(v.quantile(0.95)), 3),
            })
    return pd.DataFrame(rows)


def support_range(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Pooled Vt quantile support (feeds the formula table's no-extrapolation range)."""
    g = df[df["field"] == FIELD]
    rows = []
    for cov in covariates:
        v = g[cov].dropna()
        if len(v) < 5:
            continue
        rows.append({"covariate": cov, "n": len(v),
                     "q05": round(float(v.quantile(0.05)), 3),
                     "q95": round(float(v.quantile(0.95)), 3)})
    return pd.DataFrame(rows)


def assert_nfailures(df: pd.DataFrame) -> pd.DataFrame:
    """Assert current per-stratum failure counts against the shipped esp_models.csv
    (standing population trap).  The reworked 2026-07-22 Svod legitimately shifts
    counts vs the 2026-07-20 refit -- the table surfaces the delta, it does not fail."""
    base = _load_vt_baselines()[["stratum", "n_runs", "n_failures"]].rename(
        columns={"n_runs": "n_runs_ref", "n_failures": "n_fail_ref"})
    cur = (df[df["field"] == FIELD].groupby("stratum")
           .agg(n_runs_cur=("event", "size"), n_fail_cur=("event", "sum")).reset_index())
    m = cur.merge(base, on="stratum", how="outer")
    m["n_fail_delta"] = m["n_fail_cur"] - m["n_fail_ref"]
    return m


def km_covered_vs_uncovered(df: pd.DataFrame, cov_col: str, *, clock: str = CLOCK_PRIMARY,
                            path: Path | None = None) -> pd.DataFrame:
    """Complete-case bias check (plan §4): KM of runs WITH vs WITHOUT ``cov_col``.
    A large gap means the modelled subset is a biased slice."""
    from lifelines import KaplanMeierFitter

    g = df[df["field"] == FIELD].copy()
    g["_covered"] = g[cov_col].notna()
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for covered, sub in g.groupby("_covered"):
        dur = pd.to_numeric(sub[clock], errors="coerce")
        mask = dur.notna() & (dur > 0)
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter().fit(dur[mask], sub.loc[mask, EVENT_COL], label=f"covered={covered}")
        kmf.plot_survival_function(ax=ax, ci_show=True)
        rmst, mrl = km_rmst_mrl(dur[mask].to_numpy(), sub.loc[mask, EVENT_COL].to_numpy())
        rows.append({"covered": bool(covered), "n": int(mask.sum()),
                     "n_events": int(sub.loc[mask, EVENT_COL].sum()),
                     "km_rmst0": rmst, "km_mrl0": mrl})
    ax.set_title(f"Vt KM by {cov_col} coverage (complete-case bias check)")
    ax.set_xlabel(f"{clock} (days)"); ax.set_ylabel("S(t)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================

def fig_spline(curves: dict, path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, cv in curves.items():
        if cv is None or cv.empty:
            continue
        ax.plot(cv["kpod"], cv["log_hr"], label=name, lw=2)
        ax.fill_between(cv["kpod"], cv["lo"], cv["hi"], alpha=0.15)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.axvline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(_tt("Kpod", "Кпод")); ax.set_ylabel(_tt("log-HR (rel. mean Kpod)", "log-HR (отн. среднего Кпод)"))
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_binned_km(df: pd.DataFrame, col: str, path: Path, *, clock: str = CLOCK_PRIMARY,
                  bins: list[float] | None = None, labels: list[str] | None = None) -> None:
    from lifelines import KaplanMeierFitter
    if bins is None:
        bins, labels = (FREQ_BINS, FREQ_BIN_LABELS) if col.startswith("freq") else (KPOD_BINS, KPOD_BIN_LABELS)
    d = df[df[col].notna() & df[clock].notna()].copy()
    d["_bin"] = pd.cut(d[col], bins=bins, labels=labels, right=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lab, sub in d.groupby("_bin", observed=True):
        dur = pd.to_numeric(sub[clock], errors="coerce")
        mask = dur.notna() & (dur > 0)
        if mask.sum() < 8:
            continue
        KaplanMeierFitter().fit(dur[mask], sub.loc[mask, EVENT_COL], label=str(lab)).plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(f"Vt KM by bin ({col}, {clock})", fontsize=10)
    ax.set_xlabel(f"{clock} (days)"); ax.set_ylabel("S(t)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_corr_heatmap(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Covariate correlation (Vt complete-case)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_km_weibull_grid(df: pd.DataFrame, baseline: pd.DataFrame, path: Path,
                        *, clock: str = CLOCK_PRIMARY) -> None:
    """Per-stratum KM survival + fitted Weibull overlay with β/η/median annotated —
    the fit-quality visual for the per-stratum baselines."""
    from lifelines import KaplanMeierFitter
    strata = list(baseline.dropna(subset=["weibull_beta"])["stratum"])
    ncol = 3
    nrow = int(np.ceil(len(strata) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow), squeeze=False)
    for ax, stratum in zip(axes.ravel(), strata):
        g = df[(df["field"] == FIELD) & (df["stratum"] == stratum)].copy()
        dur = pd.to_numeric(g[clock], errors="coerce")
        mask = dur.notna() & (dur > 0)
        b = baseline[baseline["stratum"] == stratum].iloc[0]
        beta, eta = float(b["weibull_beta"]), float(b["weibull_eta"])
        kmf = KaplanMeierFitter().fit(dur[mask], g.loc[mask, EVENT_COL])
        kmf.plot_survival_function(ax=ax, ci_show=True, color="#4C78A8", label="KM")
        t = np.linspace(0, float(dur[mask].quantile(0.98)), 300)
        ax.plot(t, np.exp(-((t / eta) ** beta)), color="#E4572E", lw=2, ls="--",
                label=f"Weibull β={beta:.2f} η={eta:.0f}")
        ax.set_title(f"{stratum}\nn_ev={int(b['n_events'])}  med={b['ttf_median']:.0f}d  mean={b['ttf_mean']:.0f}d",
                     fontsize=8.5)
        ax.set_xlabel(f"{clock} (d)", fontsize=8); ax.set_ylabel("S(t)", fontsize=8)
        ax.legend(fontsize=7); ax.set_ylim(0, 1.02)
    for ax in axes.ravel()[len(strata):]:
        ax.axis("off")
    fig.suptitle(f"Vt per-stratum survival: KM + fitted Weibull ({clock})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _v31_class_params(revised: pd.DataFrame, cls: str) -> dict:
    """Pull the v3.1 params for an h2s class out of ``revised_operating_model`` output."""
    ql = revised[(revised["h2s_class"] == cls) & (revised["term"] == "Ql")].iloc[0]
    mults = {"brt": 1.0}
    for _, r in revised[(revised["h2s_class"] == cls)
                        & revised["term"].str.startswith("contractor")].iterrows():
        mults[r["term"].split(":", 1)[1]] = float(r["life_mult"])
    return {"beta": float(ql["weibull_beta"]), "eta_ref": float(ql["eta_ref_d"]),
            "ql_ref": float(ql["ql_ref_m3d"]), "lo": float(ql["ql_cap_lo"]),
            "hi": float(ql["ql_cap_hi"]), "gamma": float(ql["coef"]), "mult": mults}


def _v31_eta_per_run(g: pd.DataFrame, p: dict) -> np.ndarray:
    """η_i = η_ref · contractor_mult · (clip(Ql,lo,hi)/Ql_ref)^γ per run; Ql-missing → φ_ql=1
    (the ТМ-06 deployment fallback: score contractor-only when Ql is absent)."""
    ql = pd.to_numeric(g["ql"], errors="coerce").clip(p["lo"], p["hi"])
    phi_ql = np.where(ql.notna(), (ql / p["ql_ref"]) ** p["gamma"], 1.0)
    mult = g["contractor_group"].map(p["mult"]).fillna(1.0).to_numpy()
    return p["eta_ref"] * mult * phi_ql


def fig_km_v31_per_stratum(df: pd.DataFrame, revised: pd.DataFrame, path: Path,
                           *, clock: str = CLOCK_PRIMARY) -> None:
    """Per-stratum KM + **v3.1 (conditional) model** overlay — the in-scope fit picture.

    Unlike ``fig_km_weibull_grid`` (marginal per-stratum Weibull, β<1 frailty), this overlays
    the deployed v3.1 law: for every run η_i = η_ref·contractor_mult·(clip(Ql)/Ql_ref)^γ with
    the **conditional** class β, then the stratum curve is the run-averaged S̄(t)=mean_i
    exp(−(t/η_i)^β). Legend shows the conditional β and the model median (S̄ crosses 0.5)."""
    from lifelines import KaplanMeierFitter
    order = [("nonsour", "brt"), ("nonsour", "slb"), ("nonsour", "oth"),
             ("sour", "brt"), ("sour", "slb"), ("sour", "oth")]
    params = {c: _v31_class_params(revised, c) for c in ("nonsour", "sour")}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), squeeze=False)
    for ax, (cls, con) in zip(axes.ravel(), order):
        p = params[cls]; beta = p["beta"]
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == cls)
               & (df["contractor_group"] == con)].copy()
        dur = pd.to_numeric(g[clock], errors="coerce")
        mask = dur.notna() & (dur > 0)
        g, dur = g[mask], dur[mask]
        ne = int(g[EVENT_COL].sum())
        kmf = KaplanMeierFitter().fit(dur, g[EVENT_COL])
        kmf.plot_survival_function(ax=ax, ci_show=True, color="#4C78A8",
                                   label=_tt("KM (empirical)", "КМ (эмпир.)"))
        t = np.linspace(0, float(dur.quantile(0.98)), 300)
        eta_i = _v31_eta_per_run(g, p)
        sbar = np.exp(-(t[:, None] / eta_i[None, :]) ** beta).mean(axis=1)
        med = float(np.interp(-0.5, -sbar, t)) if sbar.min() < 0.5 else float("nan")
        ax.plot(t, sbar, color="#E4572E", lw=2.2, ls="--",
                label=f"v3.1 β={beta:.2f}  " + _tt(f"med={med:.0f}d", f"мед={med:.0f}сут"))
        ax.set_title(f"Vt_{cls}_{con}   n={len(g)} " + _tt(f"ev={ne}", f"отк={ne}"), fontsize=9)
        ax.set_xlabel(f"{clock} " + _tt("(d)", "(сут)"), fontsize=8)
        ax.set_ylabel("S(t)", fontsize=8)
        ax.legend(fontsize=7.5); ax.set_ylim(0, 1.02)
    fig.suptitle(_tt("Vt per-stratum survival: KM + v3.1 conditional model "
                     "(η=η_ref·contractor·(clip Ql/Ql_ref)^γ, run-averaged)",
                     "Выживаемость Vt по стратам: КМ + условная модель v3.1 "
                     "(η=η_ref·подрядчик·(clip Qж/Qж_ref)^γ, усреднение по пробегам)"), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=130); plt.close(fig)


def fig_model_validation(df: pd.DataFrame, path: Path, *, clock: str = CLOCK_PRIMARY) -> None:
    """Three-panel visual validation of the v3.1 model:

    (a) **Cox–Snell residuals** per h2s class — if the AFT is calibrated, the cumulative
        hazard of the residuals lies on the 45° line;
    (b) **predicted vs observed median per stratum** (same Ql-covered sample) — points on
        the diagonal = the model reproduces all six strata;
    (c) **Ql shape check (nonsour)** — spline Cox log-HR vs Ql with the v3.1 capped-linear
        overlay and the cap limits marked: shows the log-with-limits shape is what the
        data supports (flat outside the caps)."""
    from lifelines import WeibullAFTFitter, NelsonAalenFitter, CoxPHFitter

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))

    # ---- (a) Cox–Snell per class ----
    ax = axes[0]
    for h2s, color in (("nonsour", "#4C78A8"), ("sour", "#E4572E")):
        g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)].copy()
        g = g[g["ql"].notna()]
        g[clock] = pd.to_numeric(g[clock], errors="coerce")
        g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"]).copy()
        if int(g[EVENT_COL].sum()) < 20:
            continue
        g["lqc"], _, _ = _capped_log_ql(g)
        g["slb"] = (g["contractor_group"] == "slb").astype(float)
        g["oth"] = (g["contractor_group"] == "oth").astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft = WeibullAFTFitter(penalizer=0.01).fit(
                g[[clock, EVENT_COL, "lqc", "slb", "oth"]], clock, EVENT_COL)
            # r_i = -log S_pred(t_i|x_i): Exp(1) under a correct model.
            S = [float(aft.predict_survival_function(g.iloc[[i]][["lqc", "slb", "oth"]],
                                                     times=[g[clock].iloc[i]]).iloc[0, 0])
                 for i in range(len(g))]
            r = -np.log(np.clip(S, 1e-10, 1.0))
            naf = NelsonAalenFitter().fit(r, event_observed=g[EVENT_COL])
        ax.step(naf.cumulative_hazard_.index, naf.cumulative_hazard_.values[:, 0],
                where="post", color=color, lw=2, label=f"{h2s} (n_ev={int(g[EVENT_COL].sum())})")
    lim = 3.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, label=_tt("perfect calibration", "идеальная калибровка"))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(_tt("Cox–Snell residual", "остаток Кокса–Снелла"))
    ax.set_ylabel(_tt("cumulative hazard of residuals", "кум. интенсивность остатков"))
    ax.set_title(_tt("(a) Cox–Snell calibration", "(а) калибровка Кокса–Снелла"), fontsize=10)
    ax.legend(fontsize=8)

    # ---- (b) predicted vs observed median per stratum ----
    ax = axes[1]
    cc = model_consistency_checks(df, clock=clock)
    pv = cc[cc["check"] == "pred_vs_obs_median_same_sample"]
    colors = {"nonsour": "#4C78A8", "sour": "#E4572E"}
    for _, r in pv.iterrows():
        cls = "sour" if "_sour_" in r["scope"] else "nonsour"
        ax.plot(r["observed"], r["predicted"], "o", ms=9, color=colors[cls])
        ax.annotate(r["scope"].replace("Vt_", ""), (r["observed"], r["predicted"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    m = max(pv["observed"].max(), pv["predicted"].max()) * 1.15
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.fill_between([0, m], [0, m * 0.85], [0, m * 1.18], color="grey", alpha=0.12,
                    label=_tt("±15/18% band", "полоса ±15/18%"))
    ax.set_xlabel(_tt("observed KM median (d)", "набл. медиана КМ (сут)"))
    ax.set_ylabel(_tt("v3.1 predicted median (d)", "медиана прогноза v3.1 (сут)"))
    ax.set_title(_tt("(b) per-stratum medians (same sample)", "(б) медианы по стратам (та же выборка)"),
                 fontsize=10); ax.legend(fontsize=8)

    # ---- (c) Ql shape (nonsour): spline vs capped-linear ----
    ax = axes[2]
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour")].copy()
    g = g[g["ql"].notna()]
    g[clock] = pd.to_numeric(g[clock], errors="coerce")
    g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "ql"]).copy()
    g["slb"] = (g["contractor_group"] == "slb").astype(float)
    g["oth"] = (g["contractor_group"] == "oth").astype(float)
    lq = np.log(g["ql"])
    basis = _spline_basis(lq, 3)
    cols = list(basis.columns)
    work = pd.concat([g[[clock, EVENT_COL, "slb", "oth"]], basis], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=0.01).fit(work, clock, EVENT_COL)
    from patsy import dmatrix, build_design_matrices
    di = dmatrix("cr(x, df=3) - 1", {"x": lq.to_numpy()}, return_type="dataframe")
    grid = np.linspace(float(lq.quantile(0.01)), float(lq.quantile(0.99)), 80)
    Bg = np.asarray(build_design_matrices([di.design_info], {"x": grid})[0]) - di.to_numpy().mean(axis=0)
    beta = cph.params_.reindex(cols).to_numpy()
    covm = cph.variance_matrix_.reindex(index=cols, columns=cols).to_numpy()
    log_hr = Bg @ beta
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Bg, covm, Bg), 0, None))
    ax.plot(np.exp(grid), log_hr, color="#4C78A8", lw=2.2,
            label=_tt("spline Cox log-HR (df3)", "сплайн Cox log-HR (df3)"))
    ax.fill_between(np.exp(grid), log_hr - 1.96 * se, log_hr + 1.96 * se, alpha=0.15, color="#4C78A8")
    # capped-linear overlay
    lqc, cap_lo, cap_hi = _capped_log_ql(g)
    work2 = pd.concat([g[[clock, EVENT_COL, "slb", "oth"]], lqc.rename("lqc")], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = CoxPHFitter(penalizer=0.01).fit(work2, clock, EVENT_COL)
    b2 = float(c2.params_["lqc"])
    line = b2 * (np.clip(grid, np.log(cap_lo), np.log(cap_hi)) - float(lqc.mean()))
    line -= line.mean() - log_hr.mean()   # align vertical offset for shape comparison
    ax.plot(np.exp(grid), line, color="#E4572E", lw=2.2, ls="--",
            label=_tt(f"v3.1 capped log-linear (β={b2:.2f})", f"v3.1 лог-линейно с отсечками (β={b2:.2f})"))
    for cap in (cap_lo, cap_hi):
        ax.axvline(cap, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(_tt("Ql (m³/d, log scale)", "Qж (м³/сут, лог. шкала)")); ax.set_ylabel("log-HR")
    ax.set_title(_tt("(c) Ql shape: spline vs log-with-limits",
                     "(в) форма Qж: сплайн vs лог-с-отсечками"), fontsize=10); ax.legend(fontsize=8)

    fig.suptitle(_tt("Vt v3.1 model validation — calibration, per-stratum fit, covariate shape",
                     "Валидация модели Vt v3.1 — калибровка, подгонка по стратам, форма ковариаты"), fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_km_by_covariate_bin(df: pd.DataFrame, col: str, bins: list, labels: list[str],
                            path: Path, *, unit: str, title_note: str, field: str = FIELD,
                            h2s_class: str = "nonsour",
                            clock: str = CLOCK_PRIMARY, min_events: int = 8) -> None:
    """KM survival curves per covariate bin (``h2s_class`` pooled, ``field``) — the
    covariate→survival relationship shown directly: one KM per band, coloured low→high,
    with **n, events and median in every legend entry** so the data volume behind each
    curve is visible."""
    from lifelines import KaplanMeierFitter
    import matplotlib.cm as cm
    g = df[(df["field"] == field) & (df["h2s_class"] == h2s_class) & df[col].notna()].copy()
    g[clock] = pd.to_numeric(g[clock], errors="coerce")
    g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL])
    g["_b"] = pd.cut(g[col], bins=bins, labels=labels, right=False)
    cmap = cm.get_cmap("viridis_r")
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for i, lab in enumerate(labels):
        sub = g[g["_b"] == lab]
        ne = int(sub[EVENT_COL].sum())
        if ne < min_events:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmf = KaplanMeierFitter().fit(sub[clock], sub[EVENT_COL])
        med = kmf.median_survival_time_
        lbl = _tt(f"{lab}  (n={len(sub)}, ev={ne}, med={med:.0f}d)",
                  f"{lab}  (n={len(sub)}, отк={ne}, мед={med:.0f}сут)")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=cmap(i / max(len(labels) - 1, 1)), lw=2.3,
                                   label=lbl)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(f"{clock} " + _tt("(days)", "(сут)")); ax.set_ylabel("S(t)"); ax.set_ylim(0, 1.02)
    ax.set_title(_tt(f"{field}_{h2s_class}: Kaplan–Meier by {unit} band — {title_note}",
                     f"{field}_{h2s_class}: Каплан–Мейер по полосам {unit} — {title_note}"), fontsize=10)
    ax.legend(fontsize=8, title=f"{unit}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_nonsour_km_by_ql_bin(df: pd.DataFrame, path: Path, *, clock: str = CLOCK_PRIMARY) -> None:
    fig_km_by_covariate_bin(
        df, "ql", [0, 100, 200, 300, 450, 700, np.inf],
        ["<100", "100-200", "200-300", "300-450", "450-700", ">700"], path,
        unit="Ql (m³/d)", title_note="survival falls with throughput", clock=clock)


def fig_nonsour_ql_bin_trend(df: pd.DataFrame, path: Path, *, clock: str = CLOCK_PRIMARY) -> None:
    """Bar chart of observed median TTF per Ql bin (Vt_nonsour, brt+slb pooled) with the
    v2 log-linear Ql trend overlaid — does the fitted trend match the binned data?"""
    from lifelines import WeibullFitter, WeibullAFTFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour") & df["ql"].notna()].copy()
    g[clock] = pd.to_numeric(g[clock], errors="coerce")
    g = g[g[clock] > 0].dropna(subset=[clock, EVENT_COL, "log_ql"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = WeibullAFTFitter(penalizer=0.01).fit(g[[clock, EVENT_COL, "log_ql"]], clock, EVENT_COL)

    bins = [0, 100, 150, 200, 300, 450, 700, np.inf]
    labels = ["<100", "100-150", "150-200", "200-300", "300-450", "450-700", ">700"]
    g["qb"] = pd.cut(g["ql"], bins=bins, labels=labels, right=False)
    obs, fit, nev, qmed = [], [], [], []
    for lab in labels:
        sub = g[g["qb"] == lab]
        ne = int(sub[EVENT_COL].sum())
        nev.append(ne)
        if ne >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs.append(float(WeibullFitter().fit(sub[clock], sub[EVENT_COL]).median_survival_time_))
            qm = float(sub["log_ql"].median())
            qmed.append(np.exp(qm))
            fit.append(float(aft.predict_median(pd.DataFrame({"log_ql": [qm]})).iloc[0]))
        else:
            obs.append(np.nan); fit.append(np.nan); qmed.append(np.nan)

    # Empirical median trend: OLS of log(bin median) on log(bin Ql) — the honest
    # median slope, steeper than the constant-shape AFT scale slope.
    ok = np.isfinite(np.array(obs)) & np.isfinite(np.array(qmed))
    lx, ly = np.log(np.array(qmed)[ok]), np.log(np.array(obs)[ok])
    g_emp, b_emp = np.polyfit(lx, ly, 1)
    emp = np.where(np.isfinite(np.array(qmed)), np.exp(b_emp + g_emp * np.log(np.array(qmed))), np.nan)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.bar(x, obs, color="#4C78A8", width=0.62, label=_tt("observed KM/Weibull median", "набл. медиана КМ/Вейбулл"))
    ax.plot(x, emp, "s--", color="#2E7D32", lw=2.2, ms=6,
            label=_tt(f"empirical median trend  γ={g_emp:.2f}",
                      f"эмпир. тренд медиан  γ={g_emp:.2f}"))
    for xi, (o, ne) in enumerate(zip(obs, nev)):
        if np.isfinite(o):
            ax.text(xi, o + 12, _tt(f"n_ev={ne}", f"отк={ne}"), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(Qж≈{q:.0f})" if np.isfinite(q) else l
                                          for l, q in zip(labels, qmed)], fontsize=8)
    ax.set_xlabel(_tt("Ql band (m³/d)", "полоса Qж (м³/сут)"))
    ax.set_ylabel(_tt("median TTF (t_cal, days)", "медиана ННО (t_cal, сут)"))
    ax.set_title(_tt("Vt_nonsour (brt+slb): observed median TTF by Ql bin + empirical trend",
                     "Vt_nonsour (brt+slb): набл. медиана ННО по бинам Qж + эмпирический тренд"), fontsize=10)
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_nonsour_km_ql_fit(df: pd.DataFrame, path: Path, *, clock: str = CLOCK_PRIMARY) -> None:
    """Vt_nonsour KM (pooled, brt, slb) with the v2 Ql-model fit overlaid — visual test
    that the Ql-driven model reproduces brt/slb from **rate alone** (no contractor term).

    Left panel: single-point fit at each contractor's median vs mean Ql (they nearly
    coincide, and match slb but under-shoot brt's low-Ql tail).  Right panel: the honest
    fit — the **Ql-distribution mixture** (average predicted S over each contractor's own
    Ql values), which reproduces both curves."""
    from lifelines import KaplanMeierFitter, WeibullAFTFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == "nonsour")].copy()
    d = g[[clock, EVENT_COL, "log_ql", "contractor_group"]].copy()
    d[clock] = pd.to_numeric(d[clock], errors="coerce")
    d = d[d[clock] > 0].dropna(subset=[clock, EVENT_COL])
    dq = d.dropna(subset=["log_ql"]).copy()
    dq["slb"] = (dq["contractor_group"] == "slb").astype(float)
    dq["oth"] = (dq["contractor_group"] == "oth").astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft_q = WeibullAFTFitter(penalizer=0.01).fit(dq[[clock, EVENT_COL, "log_ql"]], clock, EVENT_COL)
        aft_qc = WeibullAFTFitter(penalizer=0.01).fit(dq[[clock, EVENT_COL, "log_ql", "slb", "oth"]], clock, EVENT_COL)
    tmax = float(d[clock].quantile(0.97))
    tl = np.linspace(1, tmax, 200)
    colors = {"pooled": "#333333", "brt": "#4C78A8", "slb": "#E4572E"}

    def _km(ax):
        for label, sub in [("pooled (brt+slb)", d), ("brt", d[d["contractor_group"] == "brt"]),
                           ("slb", d[d["contractor_group"] == "slb"])]:
            key = "pooled" if label.startswith("pooled") else label
            KaplanMeierFitter().fit(sub[clock], sub[EVENT_COL], label=f"KM {label} (n={len(sub)})").plot_survival_function(
                ax=ax, ci_show=(key == "pooled"), color=colors[key], lw=2)

    def _mixture(ax, aft, cols, title):
        _km(ax)
        for cg in ("brt", "slb"):
            sub = dq[dq["contractor_group"] == cg]
            sf = aft.predict_survival_function(sub[cols], times=tl)
            ax.plot(tl, sf.mean(axis=1).values, ls="--", lw=2.4, color=colors[cg],
                    label=f"fit {cg} (n={len(sub)})")
        ax.set_title(title, fontsize=10)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _mixture(axL, aft_q, ["log_ql"], "v2: Ql only — under-predicts brt")
    _mixture(axR, aft_qc, ["log_ql", "slb", "oth"], "v3: Ql + contractor — reproduces both")
    for ax in (axL, axR):
        ax.set_xlabel(f"{clock} (days)"); ax.set_ylabel("S(t)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7.5)
    fig.suptitle("Vt_nonsour: KM (pooled / brt / slb) vs Ql-model fit — v2 (Ql) vs v3 (Ql+contractor)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_contractor_overlap(df: pd.DataFrame, path: Path, *, h2s: str = "nonsour") -> None:
    """Grouped bar charts of Ql and Kpod for brt vs slb (within-contractor shares) —
    visualises how little the two contractors overlap in rate/loading."""
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)]
    brt, slb = g[g["contractor_group"] == "brt"], g[g["contractor_group"] == "slb"]
    panels = [
        ("ql", [0, 100, 200, 300, 400, 500, np.inf],
         ["<100", "100-200", "200-300", "300-400", "400-500", ">500"],
         _tt("Ql (m³/d)", "Qж (м³/сут)")),
        ("kpod_run", KPOD_BINS, KPOD_BIN_LABELS, _tt("Kpod (guarded run mean)", "Кпод (guarded, среднее пробега)")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    _med = _tt("medians", "медианы")
    for ax, (col, bins, labels, xlabel) in zip(axes, panels):
        b = pd.cut(brt[col].dropna(), bins=bins, labels=labels, right=False).value_counts(normalize=True).reindex(labels, fill_value=0) * 100
        s = pd.cut(slb[col].dropna(), bins=bins, labels=labels, right=False).value_counts(normalize=True).reindex(labels, fill_value=0) * 100
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w / 2, b.values, w, label=f"brt (n={brt[col].notna().sum()})", color="#4C78A8")
        ax.bar(x + w / 2, s.values, w, label=f"slb (n={slb[col].notna().sum()})", color="#E4572E")
        bm, sm = brt[col].median(), slb[col].median()
        ax.set_title(f"{xlabel}   {_med}: brt={bm:.0f} / slb={sm:.0f}"
                     if col == "ql" else f"{xlabel}   {_med}: brt={bm:.2f} / slb={sm:.2f}", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_ylabel(_tt("% of contractor's runs", "% пробегов подрядчика")); ax.legend(fontsize=8)
    fig.suptitle(_tt(f"Vt_{h2s} brt vs slb — rate (Ql) and loading (Kpod) overlap",
                     f"Vt_{h2s} brt vs slb — перекрытие по дебиту (Qж) и загрузке (Кпод)"), fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_contractor_km(df: pd.DataFrame, path: Path, *, h2s: str = "nonsour",
                      clock: str = CLOCK_PRIMARY) -> None:
    """KM (empirical) survival per contractor for one h2s class — brt / slb / oth overlaid,
    with n, events and median in each legend entry.  Shows the raw per-contractor doживаемость
    behind the brt/slb «lesson»."""
    from lifelines import KaplanMeierFitter
    g = df[(df["field"] == FIELD) & (df["h2s_class"] == h2s)]
    colors = {"brt": "#4C78A8", "slb": "#E4572E", "oth": "#2E7D32"}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for cg in ("brt", "slb", "oth"):
        sub = g[g["contractor_group"] == cg].copy()
        dur = pd.to_numeric(sub[clock], errors="coerce")
        m = dur.notna() & (dur > 0)
        if int(sub.loc[m, EVENT_COL].sum()) < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmf = KaplanMeierFitter().fit(dur[m], sub.loc[m, EVENT_COL])
        med = kmf.median_survival_time_
        lbl = _tt(f"{cg}  (n={int(m.sum())}, ev={int(sub.loc[m, EVENT_COL].sum())}, med={med:.0f}d)",
                  f"{cg}  (n={int(m.sum())}, отк={int(sub.loc[m, EVENT_COL].sum())}, мед={med:.0f}сут)")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[cg], lw=2.2, label=lbl)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(f"{clock} " + _tt("(days)", "(сут)")); ax.set_ylabel("S(t)"); ax.set_ylim(0, 1.02)
    ax.set_title(_tt(f"Vt_{h2s}: KM survival by contractor",
                     f"Vt_{h2s}: доживаемость по подрядчикам (КМ, факт)"), fontsize=10)
    ax.legend(fontsize=9, title=_tt("contractor", "подрядчик"))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_support_hist(df: pd.DataFrame, cov: str, path: Path) -> None:
    g = df[df["field"] == FIELD]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for stratum, sub in g.groupby("stratum"):
        v = sub[cov].dropna()
        if len(v) < 5:
            continue
        ax.hist(v, bins=25, histtype="step", label=f"{stratum} (n={len(v)})", density=True, lw=1.5)
    ax.set_title(_tt(f"Vt {cov} support by stratum", f"Vt {cov}: обеспеченность по стратам"), fontsize=10)
    ax.set_xlabel(cov); ax.set_ylabel(_tt("density", "плотность")); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ===========================================================================
# Orchestrator
# ===========================================================================

#: Kpod variants carried through the U-shape ladder (plan §3: both plain and
#: freq-adjusted, in t0 and guarded-run form).
KPOD_VARIANTS = ["kpod_run", "kpod_freq_run", "kpod_t0", "kpod_freq_t0"]
#: Model-stage covariate set.  Running frequency (``freq_run``) is a first-class
#: covariate — the operating speed the pump is driven at — alongside Kpod (off-design
#: load) and Ql (throughput).  h2s-continuous dropped; contractor/h2s are strata.
MODEL_COVARIATES = ["freq_run", "kpod_run", "log_ql", "log1p_qg", "glf"]


def _w(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run(as_of: pd.Timestamp | None = None, *, with_kvch: bool = True,
        n_boot: int = 200) -> Path:
    """Full Vt TTF-covariate workflow; writes tables + figures to results_dir.

    Returns the results directory.  Everything the plan §7 lists is produced;
    thin-slice strata (events < ~10/param) get descriptive treatment only."""
    out = results_dir("production_risk_vt_ttf_covariates")
    tdir = out / "tables"; fdir = out / "figures"
    tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)

    df, cov = build_frame(as_of=as_of, with_kvch=with_kvch)
    vt = df[df["field"] == FIELD].copy()

    # -- coverage + notes ----------------------------------------------------
    _w(cov.by_stratum, tdir / "coverage_by_stratum.csv")
    _w(cov.t0, tdir / "coverage_t0.csv")
    _w(cov.kpod, tdir / "coverage_kpod_guard30.csv")
    _w(cov.kpod_g60, tdir / "coverage_kpod_guard60.csv")
    if not cov.kvch.empty:
        _w(cov.kvch, tdir / "coverage_kvch.csv")
    (out / "NOTES.md").write_text(
        "# Vt TTF ~ operating covariates — run notes\n\n"
        + "\n".join(f"- {n}" for n in cov.notes) + "\n", encoding="utf-8")

    # -- honesty gate: n_failures assertion + KM covered/uncovered -----------
    _w(assert_nfailures(df), tdir / "gate_nfailures_vs_esp_models.csv")
    _w(km_covered_vs_uncovered(df, "kpod_run", path=fdir / "km_kpod_covered_vs_uncovered.png"),
       tdir / "gate_km_kpod_coverage.csv")

    # -- Kpod U-shape ladder (§3) --------------------------------------------
    binned = pd.concat([kpod_binned_table(vt, v).assign(kpod_col=v) for v in KPOD_VARIANTS],
                       ignore_index=True)
    _w(binned, tdir / "kpod_binned_descriptive.csv")

    quad = pd.concat([fit_quadratic_cox(vt, v) for v in KPOD_VARIANTS], ignore_index=True)
    _w(quad, tdir / "kpod_quadratic_cox.csv")

    hinge = pd.concat([fit_hinge_cox(vt, v) for v in KPOD_VARIANTS], ignore_index=True)
    _w(hinge, tdir / "kpod_hinge_slopes.csv")

    # Spline curves + permutation null (primary variants), risk-min Kpod.
    curves, spline_meta, riskmin = {}, [], []
    for v in ("kpod_run", "kpod_freq_run"):
        cv, meta = fit_spline_cox(vt, v)
        curves[v] = cv
        spline_meta.append(meta)
        riskmin.append(risk_min_kpod(vt, v, n_boot=n_boot))
    _w(pd.DataFrame(spline_meta), tdir / "kpod_spline_meta.csv")
    _w(pd.DataFrame(riskmin), tdir / "kpod_risk_min.csv")
    fig_spline({k: v for k, v in curves.items() if v is not None and not v.empty},
               fdir / "kpod_spline_logHR.png", title="Vt spline log-HR vs Kpod (t_cal, stratified)")
    fig_binned_km(vt, "kpod_run", fdir / "kpod_binned_km.png")

    _w(exposure_share_contest(vt), tdir / "kpod_exposure_share_contest.csv")

    # Guard-60 sensitivity for the §3.5 contest (does the winner flip?).
    vt_g60 = vt.rename(columns={"kpod_run_g60": "kpod_run", "kpod_freq_run_g60": "kpod_freq_run",
                                "frac_days_kpod_below_g60": "frac_days_kpod_below",
                                "frac_days_kpod_above_g60": "frac_days_kpod_above"})
    _w(exposure_share_contest(vt_g60), tdir / "kpod_exposure_share_contest_guard60.csv")

    # -- is low Kpod harmful? (direct underload-dose test) -------------------
    ul_binned, ul_cox = underload_harm_test(vt)
    _w(ul_binned, tdir / "kpod_underload_binned.csv")
    _w(ul_cox, tdir / "kpod_underload_harm_test.csv")

    # -- H2S: effective sour threshold + within-sour concentration gradient --
    _w(sour_threshold_report(vt), tdir / "h2s_sour_threshold.csv")
    # % of WELLS above threshold (prevalence of exposure; all fields for context).
    h2s_thr, h2s_scope = h2s_well_prevalence(df)
    _w(h2s_thr, tdir / "h2s_well_prevalence_vt.csv")
    _w(h2s_scope, tdir / "h2s_well_prevalence_by_scope.csv")
    h2s_binned, h2s_cox = h2s_within_sour(vt)
    if not h2s_binned.empty:
        _w(h2s_binned, tdir / "h2s_within_sour_binned.csv")
    _w(h2s_cox, tdir / "h2s_within_sour_cox.csv")

    # -- running-frequency effect (does raising speed hurt reliability?) -----
    freq_binned = pd.concat(
        [binned_table(vt, c, FREQ_BINS, FREQ_BIN_LABELS).assign(freq_col=c)
         for c in ("freq_run", "freq_t0")], ignore_index=True)
    _w(freq_binned, tdir / "freq_binned_descriptive.csv")
    # Linear frequency HR (t0 setpoint vs run-developed), single-covariate stratified.
    freq_lin = pd.concat([stratified_cox(vt, [c]).assign(freq_variant=c)
                          for c in ("freq_run", "freq_t0")], ignore_index=True)
    _w(freq_lin, tdir / "freq_linear_cox.csv")
    fig_binned_km(vt, "freq_run", fdir / "freq_binned_km.png")
    # The key answer: frequency vs Kpod vs Ql, nested (they are one affinity-law chain).
    _w(speed_load_decomposition(vt), tdir / "cox_freq_kpod_ql_decomposition.csv")
    # Deployment recipe for missing collinear inputs: coefficient set per availability
    # pattern (the freq multiplier changes when Kpod is absent — marginal vs conditional).
    _w(availability_coefficients(vt), tdir / "cox_availability_coefficients.csv")
    _w(pd.concat([availability_life_multipliers(vt, h2s_class=c) for c in ("nonsour", "sour")],
                 ignore_index=True), tdir / "availability_life_multipliers.csv")

    # -- model stage (§4) ----------------------------------------------------
    cox_global = stratified_cox(vt, MODEL_COVARIATES)
    _w(cox_global, tdir / "cox_stratified_global.csv")
    _w(stratified_cox(vt, MODEL_COVARIATES, per_stratum=True), tdir / "cox_per_stratum.csv")
    _w(cox_decomposition(vt), tdir / "cox_kpod_ql_joint.csv")
    _w(schoenfeld_ph(vt, MODEL_COVARIATES), tdir / "cox_schoenfeld_ph.csv")
    vif, corr = vif_correlation(vt, MODEL_COVARIATES)
    _w(vif, tdir / "cox_vif.csv")
    fig_corr_heatmap(corr, fdir / "covariate_corr_heatmap.png")

    # Qnom spread (what separates rate from loading, §3 guard) + type_parse robustness.
    qnom_spread = (vt.groupby("stratum")["nominal_flow_m3d"]
                   .agg(["count", "median", "std",
                         lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]))
    qnom_spread.columns = ["n", "median", "std", "q05", "q95"]
    _w(qnom_spread.reset_index(), tdir / "qnom_spread_by_stratum.csv")
    no_parse = vt[vt["kpod_qnom_source"] != "type_parse"]
    _w(cox_decomposition(no_parse), tdir / "cox_kpod_ql_joint_no_typeparse.csv")

    # -- t_mix sensitivity pass ----------------------------------------------
    _w(stratified_cox(vt, MODEL_COVARIATES, clock=CLOCK_SENS), tdir / "cox_stratified_global_tmix.csv")

    # -- deliverable (§5) ----------------------------------------------------
    support = support_range(vt, MODEL_COVARIATES)
    _w(support, tdir / "support_range_pooled.csv")
    ph = ph_life_multipliers(cox_global)
    _w(ph, tdir / "ph_life_multipliers.csv")
    aft = fit_aft(vt, MODEL_COVARIATES)
    _w(aft, tdir / "aft_life_multipliers.csv")
    _w(formula_table(cox_global, aft, ph, support), tdir / "FORMULA_TABLE.csv")

    # -- per-stratum consolidated deliverable + TTF-adjustment equation ------
    base_ttf = per_stratum_survival_ttf(vt)
    _w(base_ttf, tdir / "per_stratum_survival_ttf.csv")
    shape_by_class = {}
    for cls in ("sour", "nonsour"):
        b = base_ttf[(base_ttf["h2s_class"] == cls) & base_ttf.get("weibull_beta").notna()]
        shape_by_class[cls] = float(b["weibull_beta"].mean()) if not b.empty else np.nan
    shape_by_class["pooled"] = float(base_ttf["weibull_beta"].mean(skipna=True))
    h2s_coef = per_h2s_coefficients(vt, shape_by_class=shape_by_class)
    _w(h2s_coef, tdir / "per_h2s_coefficients.csv")
    # Sour: interaction test (does it differ?) + honest shrinkage of its coefficients.
    interaction = sour_effect_interaction(vt)
    _w(interaction.drop(columns=["beta_ns_z", "interaction_se"], errors="ignore"),
       tdir / "sour_effect_interaction.csv")
    sour_shrunk = shrink_sour_toward_pooled(interaction, shape_by_class.get("sour", np.nan))
    _w(sour_shrunk, tdir / "sour_shrunk_coefficients.csv")
    revised = revised_operating_model(vt)
    _w(revised, tdir / "revised_operating_model_v2.csv")
    # Consistency + hygiene cross-checks and the visual validation of the model.
    _w(model_consistency_checks(vt), tdir / "model_consistency_checks.csv")
    fig_model_validation(vt, fdir / "model_validation_v31.png")
    # Contractor-residual mechanism (competing risks) + external validity on Ya.
    _w(informative_censoring_test(vt), tdir / "informative_censoring_test.csv")
    _w(replicate_v31_on_ya(df), tdir / "replicate_v31_on_ya.csv")
    # OOS gate + bootstrap CIs on the composed deliverables.
    _w(temporal_holdout_validation(vt), tdir / "temporal_holdout_validation.csv")
    _w(bootstrap_model_cis(vt, n_boot=200), tdir / "bootstrap_model_cis.csv")
    write_ttf_equation(base_ttf, h2s_coef, out / "TTF_ADJUSTMENT_EQUATION.md",
                       shrink=sour_shrunk,
                       interaction=interaction.drop(columns=["beta_ns_z", "interaction_se"], errors="ignore"),
                       revised=revised)
    # Per-stratum survival fit-quality visual: KM + fitted Weibull overlay (marginal, β<1
    # frailty) and the in-scope v3.1 conditional-model overlay.
    fig_km_weibull_grid(vt, base_ttf, fdir / "km_weibull_per_stratum.png")
    fig_km_v31_per_stratum(vt, revised, fdir / "km_v31_per_stratum.png")

    # -- mechanism: cumulative throughput vs exposure time -------------------
    _w(clock_mechanism_test(vt), tdir / "clock_mechanism_throughput_vs_exposure.csv")
    # Aging diagnostic: β<1 on every clock ⇒ NOT wear-out (throughput paces, not accrues).
    _w(weibull_shape_by_clock(vt), tdir / "weibull_shape_by_clock.csv")
    # Why do nonsour brt & slb differ ~2×? balance + cohort + covariate adjustment.
    cg_bal, cg_wb, cg_hr = contractor_gap_analysis(vt)
    _w(cg_bal, tdir / "contractor_gap_balance.csv")
    _w(cg_wb, tdir / "contractor_gap_weibull_cohort.csv")
    _w(cg_hr, tdir / "contractor_gap_adjusted_hr.csv")
    _w(contractor_equipment(vt), tdir / "contractor_equipment_exec_group.csv")
    _w(contractor_pull_composition(vt), tdir / "contractor_pull_composition.csv")
    _w(contractor_ql_counterfactual(vt), tdir / "contractor_ql_counterfactual.csv")
    _w(contractor_rate_cohorts(vt), tdir / "contractor_rate_matched_cohorts.csv")
    _w(contractor_rate_band_comparison(vt), tdir / "contractor_rate_band_comparison.csv")
    fig_contractor_overlap(vt, fdir / "contractor_ql_kpod_overlap.png")
    fig_contractor_km(vt, fdir / "nonsour_contractor_km.png", h2s="nonsour")
    fig_nonsour_km_ql_fit(vt, fdir / "nonsour_km_ql_fit.png")
    fig_nonsour_ql_bin_trend(vt, fdir / "nonsour_ql_bin_trend.png")
    fig_nonsour_km_by_ql_bin(vt, fdir / "nonsour_km_by_ql_bin.png")
    fig_km_by_covariate_bin(vt, "freq_run", list(FREQ_BINS), FREQ_BIN_LABELS,
                            fdir / "nonsour_km_by_freq_bin.png",
                            unit="частота, Гц", title_note="survival falls with running frequency")
    fig_km_by_covariate_bin(vt, "kpod_run", list(KPOD_BINS), KPOD_BIN_LABELS,
                            fdir / "nonsour_km_by_kpod_bin.png",
                            unit="Кпод", title_note="survival falls with Kpod (rate proxy, no U-shape)")
    # GLF (ГЖФ): KM by bin shows NO ordering (the null, more informative than a support hist).
    fig_km_by_covariate_bin(vt, "glf", [0, 50, 100, 200, np.inf], ["<50", "50-100", "100-200", ">200"],
                            fdir / "nonsour_km_by_glf_bin.png",
                            unit="GLF", title_note="no ordering by GLF — the null covariate")
    # Same bin-KM visuals on Vt_SOUR (~110 events) — COARSE 3-band edges (thinner stratum).
    # Ql shows a weak monotone ordering but OPPOSITE in sign to nonsour and n.s. (p≈0.6,
    # confounded — worst sour wells run at low rate); freq/Kpod are non-monotone noise.
    for col, bins, labs, unit, fname, note in [
            ("ql", [0, 200, 400, np.inf], ["<200", "200-400", ">400"], "Ql (m³/d)", "sour_km_by_ql_bin.png",
             "weak POSITIVE ordering (opposite nonsour, n.s., confounded)"),
            ("freq_run", [0, 48, 52, np.inf], ["<48", "48-52", ">52"], "частота, Гц", "sour_km_by_freq_bin.png",
             "no monotone ordering (H₂S corrosion dominates)"),
            ("kpod_run", [0, 0.7, 1.0, np.inf], ["<0.7", "0.7-1.0", ">1.0"], "Кпод", "sour_km_by_kpod_bin.png",
             "no monotone ordering (H₂S corrosion dominates)")]:
        fig_km_by_covariate_bin(vt, col, bins, labs, fdir / fname, unit=unit,
                                h2s_class="sour", min_events=6, title_note=note)
    # Same bin-KM visuals on Ya (the replication field) for Ql / frequency / Kpod.
    for col, bins, labs, unit, fname in [
            ("ql", [0, 100, 200, 300, 450, 700, np.inf],
             ["<100", "100-200", "200-300", "300-450", "450-700", ">700"], "Ql (m³/d)", "ya_km_by_ql_bin.png"),
            ("freq_run", list(FREQ_BINS), FREQ_BIN_LABELS, "частота, Гц", "ya_km_by_freq_bin.png"),
            ("kpod_run", list(KPOD_BINS), KPOD_BIN_LABELS, "Кпод", "ya_km_by_kpod_bin.png")]:
        fig_km_by_covariate_bin(df, col, bins, labs, fdir / fname, unit=unit,
                                title_note="replication field", field="Ya")

    # -- honesty gates: replication + support overlap ------------------------
    _w(replicate(df, MODEL_COVARIATES), tdir / "gate_replication_ya_mc.csv")
    _w(support_overlap(vt, MODEL_COVARIATES), tdir / "gate_support_overlap_by_stratum.csv")
    for c in MODEL_COVARIATES:
        fig_support_hist(vt, c, fdir / f"support_hist_{c}.png")

    return out


if __name__ == "__main__":  # pragma: no cover
    print(run())
