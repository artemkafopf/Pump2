"""Why θ_Ql is extrapolated with a *prior* slope above the fit peak — the evidence.

The composed model (:mod:`vt_composed_model`) continues θ_Ql above its rational's peak with
``θ = θ_peak·(Ql/Ql_peak)^γ``.  This module is the reproducible record of **why γ is a prior
and not an estimate**, and of the range over which θ_Ql may be evaluated at all.

Run it whenever γ, the Ql knot grid, or the population changes.  Three pieces of evidence:

1. :func:`support_table` — how much data actually lives above the top knot.  The fleet DOES
   have runs up there (Vt: 21 runs / 14 events, max Ql 1433; fleet: 150 / 98 above 823), so
   "unobserved" is the wrong reason to clamp.  But **no field has a single run above Ql 1500**,
   so anything above the fleet max is unbacked by construction.

2. :func:`free_slope_fits` — Weibull-PH with the shipped power law as a fixed offset and one
   free *excess* log-slope on ``log(Ql/knee)₊``, stratum-specific baselines, profile-likelihood
   CI.  Every CI contains 0 and spans ≈[−1, +1], with negative point estimates: the tail needs
   no slope different from the fitted exponent, and what pull there is goes *downward* — the
   known high-Ql workover-contamination / collider artifact (same family as the failure-only
   brt/slb reversal), which is why it is not believed.

3. :func:`extended_knot_refit` — refit the v3.2 hybrid with extra Ql knots above 823 under its
   own monotone (non-negative increment) constraint.  It assigns **zero** increment there
   (Δloglik < 0.003): the constraint is binding at its lower bound.

4. :func:`dense_knot_shape` — refit the polyline on 9- and 11-knot grids to look for curvature
   the 5-knot grid could not resolve.  There is none: the denser fits are a **staircase** of
   near-vertical jumps between dead-flat stretches, the jumps **move when the grid moves**
   (nonsour's low-Ql step lands at 100–160 on one grid and 120–175 on the next), and the whole
   exercise buys ~1.0 loglik for 4–6 extra parameters.  The chord across the supported range
   (log-log slope ≈0.38 nonsour) matches the fitted exponent.

Conclusion (2026-07-27): the layer is a **single fitted power law** — see
:data:`vt_composed_model.QL_COEF`.  There is no extrapolation prior left to choose, so the
former γ is retired; :func:`tail_sensitivity` only bounds how much an alternative tail slope
*could* matter.  **Caveat kept deliberately visible:** the plateau above Ql ≈600 is the one
feature stable across all knot grids, and preferring a straight power law over it is a
judgement that the plateau is contaminated (it sits exactly where the monotone constraint binds
against a downward signal, in the band whose crude hazard reverses), not a clean fit victory.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import vt_composed_model as M
from analysis.workflows.production_risk import vt_ttf_covariates as V
from analysis.workflows.production_risk import vt_v32_hybrid as H

SLUG = "production_risk_vt_ql_extrapolation"
CLOCK = H.CLOCK
EVENT_COL = H.EVENT_COL

#: γ candidates priced in the comparison table (the shipped value must be among them).
GAMMA_GRID = (0.0, 0.10, 0.15, 0.20, 0.30)
#: Ql values the comparison is reported at — spans the full documented evaluation window.
REPORT_QL = (0.0, 100.0, 250.0, 450.0, 823.0, 1250.0, 1433.0, 1500.0, 2000.0)
#: Where the free-slope fits are allowed to kink (top knot, and the rational's peak).
KNEES = (823.0, 695.0)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def prepare_fleet(as_of: pd.Timestamp | None = None,
                  cached: pd.DataFrame | None = None) -> pd.DataFrame:
    """All fields on the v3.2 relabeled population — the fleet gives γ what power it has."""
    if cached is not None:
        df = cached
    else:
        prev = C.SOUR_WELL_LEVEL_ALL_RUNS
        C.SOUR_WELL_LEVEL_ALL_RUNS = True
        try:
            df, _ = V.build_frame(as_of=as_of)
        finally:
            C.SOUR_WELL_LEVEL_ALL_RUNS = prev
    d = df.copy()
    d[CLOCK] = pd.to_numeric(d[CLOCK], errors="coerce")
    d["ql"] = pd.to_numeric(d["ql"], errors="coerce")
    d = d[d[CLOCK] > 0].dropna(subset=[CLOCK, EVENT_COL, "ql"]).copy()
    d["stratum_key"] = d["field"].astype(str) + "_" + d["h2s_class"].astype(str)
    return d


# ---------------------------------------------------------------------------
# 1. Support above the top knot
# ---------------------------------------------------------------------------
def support_table(d: pd.DataFrame) -> pd.DataFrame:
    """Runs / events above each Ql threshold, per field.  Answers 'is it observed at all?'."""
    rows = []
    for fld, g in d.groupby("field"):
        r = {"field": fld, "n": len(g), "events": int(g[EVENT_COL].sum()),
             "ql_max": round(float(g["ql"].max()), 1),
             "ql_p95": round(float(g["ql"].quantile(0.95)), 1),
             "ql_p99": round(float(g["ql"].quantile(0.99)), 1)}
        for thr in (823, 1000, 1200, 1500, 2000):
            hi = g[g["ql"] > thr]
            r[f"n_gt{thr}"] = len(hi)
            r[f"ev_gt{thr}"] = int(hi[EVENT_COL].sum())
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("field").reset_index(drop=True)
    tot = {"field": "FLEET", "n": int(out["n"].sum()), "events": int(out["events"].sum()),
           "ql_max": float(out["ql_max"].max()),
           "ql_p95": round(float(d["ql"].quantile(0.95)), 1),
           "ql_p99": round(float(d["ql"].quantile(0.99)), 1)}
    for thr in (823, 1000, 1200, 1500, 2000):
        tot[f"n_gt{thr}"] = int(out[f"n_gt{thr}"].sum())
        tot[f"ev_gt{thr}"] = int(out[f"ev_gt{thr}"].sum())
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


def band_hazard(d: pd.DataFrame, field: str = "Vt") -> pd.DataFrame:
    """Crude per-year hazard by Ql band — shows the (small-n) reversal at the very top."""
    g = d[d["field"] == field]
    rows = []
    for lo, hi in [(250, 450), (450, 823), (823, 1000), (1000, 1200), (1200, 1500)]:
        b = g[(g["ql"] >= lo) & (g["ql"] < hi)]
        if not len(b):
            continue
        rows.append({"field": field, "ql_lo": lo, "ql_hi": hi, "n": len(b),
                     "events": int(b[EVENT_COL].sum()),
                     "exposure_d": round(float(b[CLOCK].sum())),
                     "crude_hazard_per_yr": round(float(b[EVENT_COL].sum())
                                                  / float(b[CLOCK].sum()) * 365.0, 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Free high-Ql log-slope, with a profile-likelihood CI
# ---------------------------------------------------------------------------
def _theta_shipped_flat(stratum: str, ql) -> np.ndarray:
    """The shipped power law as a fixed offset.

    The free slope fitted on top of it therefore measures the **excess** slope the data wants
    above the knee, relative to the single fitted exponent.  γ_excess ≈ 0 ⇒ one power law is
    enough and no extrapolation prior is needed.
    """
    return M.theta_ql(stratum, ql)


def _make_nll(d: pd.DataFrame, knee: float):
    keys = sorted(d["stratum_key"].unique())
    s = d["stratum_key"].map({k: i for i, k in enumerate(keys)}).to_numpy()
    t = d[CLOCK].to_numpy(float)
    e = d[EVENT_COL].to_numpy(float)
    q = d["ql"].to_numpy(float)
    cls = d["h2s_class"].where(d["h2s_class"].isin(list(M.STRATA)), "nonsour")
    off = np.log(np.array([float(_theta_shipped_flat(c, x)) for c, x in zip(cls, q)]))
    x = np.maximum(np.log(q / knee), 0.0)
    K = len(keys)

    def nll(p: np.ndarray) -> float:
        beta, lneta, g = p[:K], p[K:2 * K], p[2 * K]
        if np.any(beta <= 0):
            return 1e18
        b, le = beta[s], lneta[s]
        lp = off + g * x
        lt = np.log(t)
        base = np.clip(b * (lt - le), -700.0, 700.0)
        ll = e * (lp + np.log(b) - le + (b - 1.0) * (lt - le)) - np.exp(lp) * np.exp(base)
        v = -float(np.sum(ll))
        return v if np.isfinite(v) else 1e18

    return nll, K, q, e


def free_slope_fit(d: pd.DataFrame, knee: float, label: str,
                   ql_cap: float | None = None) -> dict:
    """One free-γ fit + profile-likelihood 95% CI (χ²₁/2 = 1.92 drop)."""
    if ql_cap is not None:
        d = d[d["ql"] <= ql_cap]
    nll, K, q, e = _make_nll(d, knee)
    bnd = [(0.2, 4.0)] * K + [(np.log(20.0), np.log(5e4))] * K + [(-2.0, 3.0)]
    p0 = np.concatenate([np.full(K, 1.1), np.full(K, np.log(400.0)), [0.15]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bnd)

        def profile(gv: float) -> float:
            rr = minimize(lambda p: nll(np.concatenate([p, [gv]])), r.x[:2 * K],
                          method="L-BFGS-B", bounds=bnd[:2 * K])
            return float(rr.fun)

        grid = np.linspace(-1.0, 2.0, 121)
        ok = grid[np.array([profile(g) for g in grid]) <= r.fun + 1.92]
    above = q > knee
    return {"fit": label, "knee": knee, "ql_cap": ql_cap, "n": len(d),
            "n_above_knee": int(above.sum()), "events_above_knee": int(e[above].sum()),
            "gamma_hat": round(float(r.x[2 * K]), 3),
            "ci_lo": round(float(ok.min()), 2) if len(ok) else np.nan,
            "ci_hi": round(float(ok.max()), 2) if len(ok) else np.nan,
            "ci_width": round(float(ok.max() - ok.min()), 2) if len(ok) else np.nan,
            "identified": bool(len(ok) and (ok.max() - ok.min()) < 0.5)}


def free_slope_fits(d: pd.DataFrame) -> pd.DataFrame:
    """The full identifiability panel: Vt alone, the fleet, Ya, and a capped-tail variant."""
    vt, ya = d[d["field"] == "Vt"], d[d["field"] == "Ya"]
    rows = []
    for knee in KNEES:
        rows.append(free_slope_fit(vt, knee, "Vt only"))
        rows.append(free_slope_fit(d, knee, "FLEET (all fields)"))
    rows.append(free_slope_fit(ya, KNEES[0], "Ya only"))
    rows.append(free_slope_fit(d, KNEES[0], "FLEET, tail capped", ql_cap=1200.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Does the monotone-constrained v3.2 fit want to climb above 823?
# ---------------------------------------------------------------------------
def _rewire_ql_knots(knots: tuple[float, ...]) -> None:
    """Patch :mod:`vt_v32_hybrid`'s Ql knot grid and every constant derived from it."""
    H.QL_KNOTS = tuple(knots)
    H._KNOTS["Ql"] = H.QL_KNOTS
    H._N_QL = len(H.QL_KNOTS) - 1
    H._QL_PIN_IX = H.QL_KNOTS.index(H.QL_PIN)
    H._N_INC = H._N_QL + H._KP_L + H._KP_R + H._FR_L + H._FR_R


def extended_knot_refit(cached: pd.DataFrame | None = None,
                        as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Refit v3.2 with extra Ql knots above the top one; report the increment it assigns.

    Restores the module's original knot grid on exit — this must not leak into other fits.
    """
    base = (47.0, 100.0, 250.0, 450.0, 823.0)
    grids = [base, base + (1150.0,), base + (1150.0, 1433.0),
             base + (1000.0, 1200.0, 1433.0)]
    original = H.QL_KNOTS
    rows = []
    try:
        for knots in grids:
            _rewire_ql_knots(knots)
            vt = H.prepare_frame(as_of=as_of, cached=cached)
            fits = {s: H.fit_stratum(vt[vt["h2s_class"] == s]) for s in H.STRATA}
            model = H.build_model(fits)
            for s in H.STRATA:
                th = model.theta[("Ql", s)]
                top = float(th[-1])
                at823 = float(np.interp(823.0, knots, th))
                rows.append({
                    "knots": "|".join(f"{k:.0f}" for k in knots),
                    "n_knots_above_823": sum(k > 823.0 for k in knots),
                    "stratum": s,
                    "theta_823": round(at823, 4),
                    "theta_top_knot": round(top, 4),
                    "top_knot": knots[-1],
                    "implied_gamma": round(float(np.log(top / at823)
                                                 / np.log(knots[-1] / 823.0)), 4)
                    if knots[-1] > 823.0 else np.nan,
                    "loglik": round(fits[s].loglik, 4),
                    "cc_n": fits[s].cc_n, "cc_events": fits[s].cc_events,
                })
    finally:
        _rewire_ql_knots(original)
    return pd.DataFrame(rows)


#: Denser Ql knot grids for the shape probe.  ``base`` is the v3.2 grid; ``x2`` roughly
#: doubles it on log-spaced quantile-ish cuts; ``x2_hi`` also splits the high tail.
DENSE_GRIDS = {
    "base_5": (47.0, 100.0, 250.0, 450.0, 823.0),
    "dense_9": (47.0, 100.0, 160.0, 250.0, 340.0, 450.0, 600.0, 823.0, 1100.0),
    "dense_11": (47.0, 80.0, 120.0, 175.0, 250.0, 340.0, 450.0, 600.0, 823.0,
                 1100.0, 1433.0),
}


def dense_knot_shape(cached: pd.DataFrame | None = None,
                     as_of: pd.Timestamp | None = None,
                     grids: dict | None = None) -> pd.DataFrame:
    """Refit the Ql polyline on DENSER knot grids and read off the local log-log slope.

    Purpose: the deployed layer is a single power law, i.e. a straight line in log θ vs log Ql.
    A 5-knot polyline is too coarse to confirm or refute that.  Doubling the knots exposes any
    real curvature — if each segment's local slope hovers around the fitted exponent, the power
    law is the right summary; systematic drift in the slopes would say it is not.

    Per segment reports the local slope ``Δlnθ / ΔlnQl`` and the support (runs / events) that
    segment actually rests on, so thin segments can be discounted.
    """
    grids = DENSE_GRIDS if grids is None else grids
    original = H.QL_KNOTS
    rows = []
    try:
        for name, knots in grids.items():
            _rewire_ql_knots(tuple(knots))
            vt = H.prepare_frame(as_of=as_of, cached=cached)
            fits = {s: H.fit_stratum(vt[vt["h2s_class"] == s]) for s in H.STRATA}
            model = H.build_model(fits)
            for s in H.STRATA:
                th = model.theta[("Ql", s)]
                g = vt[vt["h2s_class"] == s]
                ql = pd.to_numeric(g["ql"], errors="coerce")
                for i in range(len(knots) - 1):
                    lo, hi = knots[i], knots[i + 1]
                    band = g[(ql >= lo) & (ql < hi)]
                    rows.append({
                        "grid": name, "n_knots": len(knots), "stratum": s,
                        "ql_lo": lo, "ql_hi": hi,
                        "theta_lo": round(float(th[i]), 4),
                        "theta_hi": round(float(th[i + 1]), 4),
                        "local_slope": round(float(np.log(th[i + 1] / th[i])
                                                   / np.log(hi / lo)), 4),
                        "n_in_band": len(band),
                        "events_in_band": int(band[EVENT_COL].sum()),
                        "tail_slope_c": round(M.QL_COEF[s][0], 4),
                        "loglik": round(fits[s].loglik, 3),
                        "cc_n": fits[s].cc_n, "cc_events": fits[s].cc_events,
                    })
    finally:
        _rewire_ql_knots(original)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Sensitivity: what if the slope above the dense range were NOT the fitted one?
# ---------------------------------------------------------------------------
def _theta_with_tail(stratum: str, ql, tail: float, knee: float = None) -> np.ndarray:
    """Shipped power law below the knee; an alternative constant slope ``tail`` above it."""
    knee = M.QL_HI if knee is None else knee
    q = np.maximum(np.asarray(ql, float), M.QL_LO)
    base = M.theta_ql(stratum, np.minimum(q, knee))
    return base * np.where(q > knee, (np.maximum(q, knee) / knee) ** tail, 1.0)


def tail_sensitivity(tails=GAMMA_GRID, ql_values=REPORT_QL) -> pd.DataFrame:
    """Price alternative high-Ql slopes against the shipped single power law.

    The shipped model has NO tail parameter — its slope above the dense range is just the
    fitted exponent, listed here as ``tail == c`` so the comparison includes it.  This table
    exists to bound how much the unbacked region could matter, not to reintroduce a knob.
    """
    rows = []
    for s in M.STRATA:
        c = M.QL_COEF[s][0]
        for tail in tuple(tails) + (c,):
            for q in ql_values:
                th = float(_theta_with_tail(s, q, tail))
                b = M.BASELINE[s]
                eta_eff = b["eta0"] * th ** (-1.0 / b["beta0"])
                r0 = M.rmst_ref(s)
                r = M.rmst(eta_eff, b["beta0"])
                rows.append({"tail_slope": round(tail, 4), "stratum": s, "ql": q,
                             "theta_ql": round(th, 4), "rmst730": round(r, 1),
                             "rmst_mult": round(r / r0, 4),
                             "is_shipped": abs(tail - c) < 1e-9,
                             "backed_by_data": q <= M.QL_UNBACKED_ABOVE})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(*, as_of: pd.Timestamp | None = None, cached: pd.DataFrame | None = None,
        write: bool = True) -> dict:
    """Rebuild the whole γ evidence base and write tables + the 0–2000 figure."""
    d = prepare_fleet(as_of=as_of, cached=cached)
    out = {
        "support": support_table(d),
        "band_hazard": band_hazard(d, "Vt"),
        "free_slope": free_slope_fits(d),
        "extended_knots": extended_knot_refit(cached=cached, as_of=as_of),
        "dense_knots": dense_knot_shape(cached=cached, as_of=as_of),
        "tail_sensitivity": tail_sensitivity(),
    }
    if write:
        _write_outputs(out)
    return out


def _write_outputs(res: dict) -> Path:
    out = results_dir(SLUG)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
    _fig_knot_density(res["dense_knots"], figures)
    fig_ql_multipliers(figures)
    return out


# --- the deliverable chart: θ_Ql and its RMST(0,730) multiplier, 0 → 2000 ----
#: Stratum colours, fixed by entity (never by rank) and shared with the other Vt figures.
#: Validated as a categorical pair: ΔE 18.1 protan / 27.2 normal vs a light surface.
STRATUM_COLOR = {"nonsour": "#2471a3", "sour": "#c0392b"}
STRATUM_LABEL = {"nonsour": "несернистые", "sour": "сернистые"}

#: The v3.2 hybrid's FITTED θ_Ql polyline — what the deployed smooth form must reproduce.
#: Kept here (not imported) so the figure shows the fit as of the smoothing, independent of
#: any later refit; ``extended_knot_refit`` is what re-derives them.
V32_KNOTS = (47.0, 100.0, 250.0, 450.0, 823.0)
V32_THETA = {"nonsour": (0.820, 0.820, 1.000, 1.483, 1.644),
             "sour": (0.927, 0.927, 1.000, 1.159, 1.159)}


def fig_ql_multipliers(figures: Path | None = None, *, annotate_at=(250.0, 823.0, 1433.0)):
    """θ_Ql (hazard) and the RMST(0,730) multiplier it implies, over the full 0–2000 window.

    Two panels, one measure each — never a shared axis.  Support is shown as three shaded
    bands (fitted / extrapolated-but-observed / unbacked) so no one reads the right-hand
    end as an estimate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = M.QL_EVAL_RANGE
    q = np.linspace(lo, hi, 900)
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(15, 6.2))

    bands = [(M.QL_LO, M.QL_HI, "#000000", .045),                # dense support
             (M.QL_HI, M.QL_UNBACKED_ABOVE, "#000000", .085),    # thin, but still observed
             (M.QL_UNBACKED_ABOVE, hi, "#000000", .16)]          # no observations anywhere
    for ax in (ax_t, ax_r):
        for x0, x1, col, a in bands:
            ax.axvspan(x0, x1, color=col, alpha=a, lw=0, zorder=0)
        ax.axvspan(lo, M.QL_LO, color="#000000", alpha=.085, lw=0, zorder=0)
        ax.axvline(M.QL_REF, ls=":", color="#555555", lw=1.0, zorder=1)
        ax.axvline(M.QL_UNBACKED_ABOVE, ls="--", color="#333333", lw=1.2, zorder=1)
        ax.grid(alpha=.18, lw=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(lo, hi * 1.10)          # headroom so the 2000 end-labels stay inside
        ax.set_xlabel("Ql, м³/сут")
        # tick on the values that mean something, not on a generic 250-step grid
        ax.set_xticks([0, M.QL_REF, 500, M.QL_HI, M.QL_UNBACKED_ABOVE, hi])
        ax.set_xticklabels(["0", "250\nопорная", "500", "823\nкрай подгонки",
                            "1433\nмакс. по флоту", "2000"], fontsize=8.5)

    for i, s in enumerate(M.STRATA):
        c = STRATUM_COLOR[s]
        b = M.BASELINE[s]
        th = M.theta_ql(s, q)
        r0 = M.rmst_ref(s)

        def to_rmst(t: float) -> float:
            return M.rmst(b["eta0"] * float(t) ** (-1 / b["beta0"]), b["beta0"]) / r0

        rm = np.array([to_rmst(t) for t in th])
        ax_t.plot(q, th, color=c, lw=2.0, zorder=4, label=STRATUM_LABEL[s])
        ax_r.plot(q, rm, color=c, lw=2.0, zorder=4,
                  label=f"{STRATUM_LABEL[s]} (база {r0:.0f} сут)")

        # the FITTED polyline underneath: knots as hollow diamonds + the piecewise-linear
        # interpolation the smooth curve is rendering.  Mark style (not hue) carries the
        # fit-vs-deployed distinction — hue stays reserved for the stratum.
        kx = np.asarray(V32_KNOTS, float)
        kth = np.asarray(V32_THETA[s], float)
        dense = np.linspace(kx[0], kx[-1], 300)
        poly = np.exp(np.interp(dense, kx, np.log(kth)))
        ax_t.plot(dense, poly, color=c, lw=1.1, ls=(0, (5, 3)), alpha=.75, zorder=3)
        ax_r.plot(dense, [to_rmst(t) for t in poly], color=c, lw=1.1, ls=(0, (5, 3)),
                  alpha=.75, zorder=3)
        ax_t.plot(kx, kth, "D", mfc="white", mec=c, mew=1.7, ms=7, zorder=6)
        ax_r.plot(kx, [to_rmst(t) for t in kth], "D", mfc="white", mec=c, mew=1.7, ms=7,
                  zorder=6)

        for x in annotate_at:
            tv = float(M.theta_ql(s, x))
            on_knot = any(abs(x - k) < 1e-6 for k in V32_KNOTS)
            for ax, yv, fmt in ((ax_t, tv, f"{tv:.2f}"), (ax_r, to_rmst(tv), f"×{to_rmst(tv):.2f}")):
                if not on_knot:                      # a knot already carries its own diamond
                    ax.plot([x], [yv], "o", color=c, ms=7, mec="white", mew=1.6, zorder=5)
                ax.annotate(fmt, (x, yv), textcoords="offset points", xytext=(8, 7),
                            fontsize=9, color="#222222", zorder=7)
        # end of window: label to the RIGHT of the curve, in the headroom margin
        tv_end = float(M.theta_ql(s, hi))
        for ax, yv, fmt in ((ax_t, tv_end, f"{tv_end:.2f}"),
                            (ax_r, to_rmst(tv_end), f"×{to_rmst(tv_end):.2f}")):
            ax.plot([hi], [yv], "o", color=c, ms=7, mec="white", mew=1.6, zorder=5)
            # stagger vertically: the two strata's RMST multipliers coincide at the far end
            ax.annotate(fmt, (hi, yv), textcoords="offset points", xytext=(9, 5 - 13 * i),
                        fontsize=9, color=c, fontweight="bold", zorder=6)

    ax_t.axhline(1.0, color="#888888", lw=.9, zorder=1)
    ax_r.axhline(1.0, color="#888888", lw=.9, zorder=1)
    ax_t.set_ylabel("θ_Ql — множитель риска отказа")
    ax_r.set_ylabel("множитель RMST(0,730)")
    ax_t.set_title("θ_Ql: множитель интенсивности отказов", fontsize=11)
    ax_r.set_title("во что это превращается по наработке (RMST 0–730 сут)", fontsize=11)
    # legend: hue = stratum, mark style = deployed smooth form vs the fitted polyline
    from matplotlib.lines import Line2D
    form_keys = [Line2D([], [], color="#555555", lw=2.0,
                        label="модель (гладкая, в VBA)"),
                 Line2D([], [], color="#555555", lw=1.1, ls=(0, (5, 3)),
                        label="полилиния v3.2 (5 узлов)"),
                 Line2D([], [], color="#555555", marker="D", mfc="white", mec="#555555",
                        mew=1.7, ms=7, ls="none", label="узлы полилинии")]
    ax_t.legend(handles=ax_t.get_legend_handles_labels()[0][:2] + form_keys,
                fontsize=8.5, loc="lower right")
    ax_r.legend(fontsize=9, loc="lower left")

    # support-band key, placed in-panel so the title can stay short
    cs = "  ".join(f"c({STRATUM_LABEL[s]})={M.QL_COEF[s][0]:.3f}" for s in M.STRATA)
    ax_t.text(0.015, 0.28,
              "фон = обеспеченность данными:\n"
              f"светлый — плотные данные · средний — до макс. по флоту "
              f"({M.QL_UNBACKED_ABOVE:.0f}) · тёмный — наблюдений нет\n"
              f"наклон хвоста (излома нет): {cs}",
              transform=ax_t.transAxes, va="top", ha="left", fontsize=7.6, color="#333333",
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=.85))
    # the two guide lines are labelled once (left panel only) — both panels share the x-axis,
    # and repeating them on the right collides with the converging end-of-range labels
    # (the guide lines are named on the x-axis ticks — no rotated in-plot labels to collide)
    fig.suptitle(f"Vt — слой Ql на диапазоне {lo:.0f}–{hi:.0f} м³/сут", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if figures is not None:
        fig.savefig(figures / "theta_ql_and_rmst_multipliers.png", dpi=140)
        plt.close(fig)
        return figures / "theta_ql_and_rmst_multipliers.png"
    return fig


def _fig_knot_density(dense: pd.DataFrame, figures: Path) -> None:
    """Knot-density probe on LOG-LOG axes, where the fitted power law is a straight line.

    Overlays the isotonic polyline refitted on 5 / 9 / 11 knots.  If the layer really had
    curvature, the denser polylines would bend away from the line consistently.  Instead they
    form a staircase whose steps MOVE when the grid moves — the signature of a monotone-
    constrained estimator on sparse data, not of a real shape.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_style = {"base_5": ("#7f8c8d", 1.3, (0, (6, 3))),
                  "dense_9": ("#e67e22", 1.6, (0, (3, 2))),
                  "dense_11": ("#8e44ad", 1.6, "-")}
    q = np.geomspace(M.QL_LO, M.QL_EVAL_RANGE[1], 400)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, s in zip(axes, M.STRATA):
        c = M.QL_COEF[s][0]
        ax.plot(q, M.theta_ql(s, q), color=STRATUM_COLOR[s], lw=2.8, zorder=5,
                label=f"модель (хвост c={c:.3f})")
        for name, g in dense[dense["stratum"] == s].groupby("grid"):
            g = g.sort_values("ql_lo")
            xs = list(g["ql_lo"]) + [float(g["ql_hi"].iloc[-1])]
            ys = list(g["theta_lo"]) + [float(g["theta_hi"].iloc[-1])]
            col, lw, ls = grid_style.get(name, ("#333333", 1.4, "-"))
            ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=3,
                    label=f"полилиния, {int(g['n_knots'].iloc[0])} узлов")
            ax.plot(xs, ys, "o", color=col, ms=3.5, zorder=4)
        ax.axvspan(M.QL_UNBACKED_ABOVE, M.QL_EVAL_RANGE[1], color="#000000", alpha=.10, lw=0)
        ax.axvline(M.QL_REF, ls=":", color="#555555", lw=1.0)
        ax.axhline(1.0, color="#888888", lw=.9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Ql, м³/сут (лог)"); ax.set_ylabel("θ (лог)")
        ax.set_title(f"{STRATUM_LABEL[s]} — в лог-лог степенная модель это ПРЯМАЯ",
                     fontsize=10.5)
        ax.grid(alpha=.25, which="both"); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Проверка формы слоя Ql: удвоение числа узлов не выявляет кривизну — "
                 "полилиния распадается в лестницу, ступени которой смещаются вместе с сеткой",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(figures / "theta_ql_knot_density.png", dpi=140)
    plt.close(fig)
