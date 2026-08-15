"""Field v4 — the unified v4 structure fitted for **every field the calculators offer**.

The workbooks (`ПикПолка`, `NPV_УВЧ`) have always had a field cell — «УН» / «Участок недр» —
but it only drove downtime days and repair cost; the survival model itself was Vt-only, later
Vt + Ya.  This module fits the same structure for the whole КРС_ЭПУ roster and emits the
parameter blocks the VBA reads, so switching the field cell switches the model.

Structure, identical everywhere (see :mod:`unified_v4`)::

    h(t | x) = (β₀/η₀)(t/η₀)^(β₀−1) · HR_contractor · θ_Qnom(q) · θ_Kpod(k) · θ_freq(Δf)

**One knot grid for all fields.**  Complexity is controlled by the per-field ridge
(:data:`unified_v4.RIDGE`), never by moving knots.  Two reasons: the deploy block stays a
rectangle the VBA can read with one knot row, and a shrunk arm degrades toward θ ≡ 1 — the
honest null — whereas a re-cut grid quietly changes what the curve *means* field to field.

**What differs per field is support, not form:**

======  =============  ==============  ====================================================
code    repo field     runs / events   contractor
======  =============  ==============  ====================================================
Ya      Ya             2145 / 1217     brt · slb · oth
Vt      Vt ns + sr      626 /  313     brt · slb · oth, two H₂S strata
Az      Az              315 /  162     brt · slb  (oth: 5 window runs → reference)
Ic      Ic              290 /  163     brt · slb  (oth: 8 field runs → reference)
Au      **Za**          297 /  153     brt · slb · oth
Mc, Mr  Mc (incl. Mr)   162 /   42     brt only — **no contrast exists**
======  =============  ==============  ====================================================

Standing rules that are not negotiable here (``.claude/skills/field-k1k2``):

* every censored run stays in — running pumps, ГТМ pulls, unknown-reason pulls;
* **Мирнинский is a 2024+ install cohort**, filtered on install date, never left-truncated;
* Ya is a single H₂S class and Vt's two are never pooled;
* RMST(0,730) is the headline life, with a KM control beside it.

A layer that does not earn its keep out of sample is still *fitted* — so the evidence is
visible — but is flagged in ``layers.csv`` with its CV delta.  Frequency is the standing
example: it is negative on Vt and inside noise on Ya, and ships as an operator-set prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival.polyline_ph import ArmSpec, FREE
from analysis.paths import results_dir
from analysis.workflows.production_risk import unified_v4 as U

SLUG = "production_risk_field_v4"

#: Calculator field code → repo field label, in roster order.  ``Au``/``Mr`` are aliases:
#: the population build folds the ``AU*`` pads into ``Za`` and the ``MR`` pad into ``Mc``.
ROSTER = ("Ya", "Vt", "Az", "Ic", "Au", "Mc", "Mr", "Fleet")

#: Deploy keys the VBA looks up, in the order they are written to ``TuneBlock``/``QnomBlock``.
#: ``Mr`` reuses ``Mc``'s row rather than getting its own — same cohort, same fit.
DEPLOY_ALIAS = {"Au": "Za", "Mr": "Mc"}

#: The pooled fallback for a field with no model of its own.
#:
#: Several codes the workbooks price — Ki (Кийский), Ma (Марковский), Bt (Большетирский) —
#: have no stratum in the survival mart at all, and Da has 50 events with a single contractor.
#: Falling back to Vt, as the VBA did, silently prices those wells on a sour-capable field with
#: an unusually steep nameplate slope.  ``Fleet`` is fitted on **every** run in the mart instead:
#: a fleet-average baseline, with contractor levels taken **within field** (field fixed effects
#: in the stage-1 window fit, so the level is not a field-mix artefact) and one pooled θ_Qnom.
FLEET_KEY = "Fleet"

#: Calculator codes that have no field model and are served by :data:`FLEET_KEY`.
FLEET_CODES = ("Da", "Ki", "Ma", "Bt")


@dataclass
class FieldRun:
    """One field's fit plus everything needed to audit it."""
    field: str
    fits: dict                                    # stratum -> unified_v4.StratumFit
    contractor: dict                              # stage-1 diagnostics (adjusted)
    contractor_plain: dict                        # same fit without the size adjusters
    frame: pd.DataFrame
    cv: dict = dc_field(default_factory=dict)     # layer -> out-of-sample Δ log-lik


def _repo_field(code: str) -> str:
    return DEPLOY_ALIAS.get(code, code)


def fleet_frame(cached: pd.DataFrame, fields=None) -> pd.DataFrame:
    """Every field's modelling frame, stacked, with ``stratum`` = the field label.

    Stage 1 uses that label as a fixed effect; stages 2–3 collapse it to ``Fleet`` so the
    baseline and θ_Qnom are genuinely fleet-average — which is what an unmodelled field needs.
    """
    fields = fields or sorted({_repo_field(c) for c in ROSTER} | {"Da"})
    parts = []
    for f in fields:
        d = U.prepare_field(f, cached=cached)
        d["stratum"] = f                                  # field FE for the contractor stage
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fit_one(code: str, *, cached: pd.DataFrame, n_boot: int = 0, with_cv: bool = True,
            seed: int = 7) -> FieldRun:
    """Fit one field end to end, including the audit fit without size adjusters."""
    if code == FLEET_KEY:
        d = fleet_frame(cached)
        cc = U.complete_case(d)
        lvl = U.fit_contractor_levels(cc)                 # within-field, size-adjusted
        plain = U.fit_contractor_levels(cc, size_adjust=False)
        d = d.copy()
        d["stratum"] = FLEET_KEY                          # pooled baseline + θ_Qnom
        r = U.fit_field(d, field=FLEET_KEY, level=lvl["level"], n_boot=n_boot, seed=seed)
        cv = layer_cv(r["frame"], FLEET_KEY) if with_cv else {}
        return FieldRun(field=FLEET_KEY, fits=r["fits"], contractor=lvl,
                        contractor_plain=plain, frame=r["frame"], cv=cv)
    field = _repo_field(code)
    d = U.prepare_field(field, cached=cached)
    cc = U.complete_case(d)
    lvl = U.fit_contractor_levels(cc)
    plain = U.fit_contractor_levels(cc, size_adjust=False)
    r = U.fit_field(d, field=field, level=lvl["level"], n_boot=n_boot, seed=seed)
    cv = layer_cv(r["frame"], field) if with_cv else {}
    return FieldRun(field=field, fits=r["fits"], contractor=lvl, contractor_plain=plain,
                    frame=r["frame"], cv=cv)


def layer_cv(cc: pd.DataFrame, field: str) -> dict:
    """Out-of-sample gain of Kpod and freq over ``Qnom + contractor`` alone.

    The same 5-fold, well-clustered comparison that produced
    :data:`unified_v4.LAYER_CV_DELTA` — re-measured per field rather than assumed to transfer.
    """
    ridge = U.RIDGE.get(field, U.RIDGE_DEFAULT)
    qn = ArmSpec("qnom", "qnom", U.QNOM_KNOTS, U.QNOM_REF, FREE, ridge["qnom"])
    base = U.cross_validate(cc, (qn,))
    out = {"qnom_only": round(base, 2)}
    for name, col, knots, pin in (("kpod", "kpod_run", U.KPOD_KNOTS, U.KPOD_REF),
                                  ("freq", "freq_dev", U.FREQ_KNOTS, U.FREQ_REF)):
        a = ArmSpec(name, col, knots, pin, FREE, ridge[name])
        out[name] = round(U.cross_validate(cc, (qn, a)) - base, 2)
    return out


def run(*, as_of=None, cached: pd.DataFrame | None = None, roster=ROSTER,
        n_boot: int = 0, with_cv: bool = True, write: bool = True, seed: int = 7) -> dict:
    """Fit every field on the roster off one shared frame."""
    frame = cached if cached is not None else U.build_cached_frame(as_of)
    runs: dict = {}
    for code in roster:
        field = _repo_field(code)
        if field in runs:                      # Mr → Mc, Au → Za: one fit serves both codes
            continue
        runs[field] = fit_one(code, cached=frame, n_boot=n_boot, with_cv=with_cv, seed=seed)
    out = {"runs": runs, "baselines": baseline_table(runs), "layers": layer_table(runs),
           "contractor": contractor_table(runs), "support": support_table(runs),
           "tune_block": tune_block(runs), "qnom_block": qnom_block(runs)}
    if write:
        _write_outputs(out)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _weibull_all_runs(g: pd.DataFrame) -> tuple:
    """Covariate-free Weibull on **every** run of the stratum, complete case or not.

    The layered fit necessarily runs on complete cases — a row with no Kpod cannot enter a
    Kpod layer — and on the thin fields that is a big cut (Mc 42 events → 27, Az 162 → 108).
    This is the control: if the layered baseline has drifted away from the all-runs one, the
    complete-case selection is doing something, and the deployed RMST_ref inherits it.
    """
    from scipy.optimize import brentq

    t = pd.to_numeric(g[U.CLOCK], errors="coerce").to_numpy(float)
    e = pd.to_numeric(g[U.EVENT_COL], errors="coerce").to_numpy(float)
    m = np.isfinite(t) & (t > 0) & np.isfinite(e)
    t, e = t[m], e[m]
    d = float(e.sum())
    if d < 3:
        return (np.nan, np.nan, np.nan)
    lt = np.log(t)

    def score(b):                     # profiled MLE: η closed-form given β
        w = t ** b
        return 1.0 / b + float((e * lt).sum() / d) - float((w * lt).sum() / w.sum())

    try:
        beta = brentq(score, 0.05, 6.0)
    except ValueError:
        return (np.nan, np.nan, np.nan)
    eta = float(((t ** beta).sum() / d) ** (1.0 / beta))
    return (round(beta, 4), round(eta, 1), round(U.rmst(eta, beta), 1))


def baseline_table(runs: dict) -> pd.DataFrame:
    rows = []
    for field, r in runs.items():
        for s, f in r.fits.items():
            g = r.frame[r.frame["stratum"] == s]
            b_all, e_all, rmst_all = _weibull_all_runs(g)
            rows.append({"field": field, "stratum": s, "n": f.n, "events": f.events,
                         "beta_all_runs": b_all, "eta_all_runs": e_all,
                         "rmst730_all_runs": rmst_all,
                         "beta0": round(f.beta0, 4), "eta0": round(f.eta0, 1),
                         "rmst730_ref": round(U.rmst(f.eta0, f.beta0), 1),
                         # MRL(0) = the Weibull mean ηΓ(1+1/β).  Reported beside RMST because
                         # with β < 1 it draws heavily on a tail nobody observed.
                         "mrl0_ref": round(f.eta0 * _gamma1p(1.0 / f.beta0), 1),
                         "concordance": round(f.concordance, 4),
                         "hr_brt": 1.0,
                         "hr_slb": round(f.contractor.get("slb", 1.0), 4),
                         "hr_oth": round(f.contractor.get("oth", 1.0), 4),
                         "cv_kpod": r.cv.get("kpod", np.nan),
                         "cv_freq": r.cv.get("freq", np.nan)})
    return pd.DataFrame(rows)


def _gamma1p(x: float) -> float:
    from scipy.special import gamma
    return float(gamma(1.0 + x))


def layer_table(runs: dict) -> pd.DataFrame:
    """Every layer at its knots, with the deployed θ beside the fitted one."""
    knots = {"qnom": U.QNOM_KNOTS, "kpod": U.KPOD_KNOTS, "freq": U.FREQ_KNOTS}
    rows = []
    for field, r in runs.items():
        for s, f in r.fits.items():
            for name, kn in knots.items():
                th = np.asarray(f.theta[name], float)
                dep = (U.deploy_theta_qnom(th) if name == "qnom" else th)
                mult = f.rmst_mult_curve(name, np.asarray(kn, float))
                for k, t, dp, m in zip(kn, th, dep, mult):
                    rows.append({"field": field, "stratum": s, "layer": name, "x": k,
                                 "theta_fitted": round(float(t), 4),
                                 "theta_deployed": round(float(dp), 4),
                                 "clamped": bool(abs(float(dp) - float(t)) > 1e-9),
                                 "rmst_mult": round(float(m), 4),
                                 "shared_shape": name in f.shared_shape,
                                 "cv_delta": r.cv.get(name, np.nan)})
    return pd.DataFrame(rows)


def contractor_table(runs: dict) -> pd.DataFrame:
    """The de-double-counting audit: level with and without the size adjusters.

    ``qnom_geomean`` / ``kpod_median`` are what the plain level was silently absorbing — a
    contractor whose pumps are bigger than the reference gets a level that already contains
    part of θ_Qnom, and θ_Qnom then applies it again.
    """
    rows = []
    for field, r in runs.items():
        ov = r.frame[(r.frame["ql"] >= r.contractor["window"][0])
                     & (r.frame["ql"] <= r.contractor["window"][1])]
        for cg in (U.CONTRACTOR_REF, *U.CONTRACTOR_TERMS):
            m = ov["contractor_group"] == cg
            q = ov.loc[m, "qnom"].astype(float)
            rows.append({
                "field": field, "contractor": cg,
                "level_plain": round(r.contractor_plain["level"].get(cg, 1.0), 4),
                "level_adjusted": round(r.contractor["level"].get(cg, 1.0), 4),
                "window_n": int(m.sum()), "window_events": int(ov.loc[m, U.EVENT_COL].sum()),
                "qnom_geomean": round(float(np.exp(np.log(q).mean())), 1) if len(q) else np.nan,
                "kpod_median": round(float(ov.loc[m, "kpod_run"].median()), 3) if m.any()
                else np.nan,
                "unsupported": cg in r.contractor.get("unsupported", {}),
                "size_hr_qnom_per_efold": round(r.contractor["size_terms"].get("lq", np.nan), 3),
                "size_hr_kpod_per_efold": round(r.contractor["size_terms"].get("lk", np.nan), 3),
            })
    return pd.DataFrame(rows)


def support_table(runs: dict) -> pd.DataFrame:
    """Where each (stratum, contractor) actually has nameplates — outside this θ extrapolates."""
    rows = []
    for field, r in runs.items():
        for (s, cg), g in r.frame.groupby(["stratum", "contractor_group"]):
            q = g["qnom"].astype(float)
            rows.append({"field": field, "stratum": s, "contractor": cg,
                         "n": len(g), "events": int(g[U.EVENT_COL].sum()),
                         "qnom_p10": round(float(q.quantile(0.10))),
                         "qnom_p90": round(float(q.quantile(0.90))),
                         "qnom_max": round(float(q.max()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deploy blocks — exactly what the VBA reads
# ---------------------------------------------------------------------------
def _alias_pairs() -> list[tuple[str, str]]:
    """Deploy rows that copy another row: pads (Au→Za, Mr→Mc) and the fleet-served codes."""
    return list(DEPLOY_ALIAS.items()) + [(c, FLEET_KEY) for c in FLEET_CODES]


def deploy_key(stratum: str) -> str:
    """Stratum → the key the workbook looks up.  Vt keeps its H₂S suffix; nothing else has one."""
    return stratum


def tune_block(runs: dict) -> pd.DataFrame:
    """``TuneBlock``: one row per key — β, RMST_ref (days), and the three contractor levels."""
    rows = []
    for field, r in runs.items():
        for s, f in r.fits.items():
            rows.append({"key": deploy_key(s), "beta": round(f.beta0, 7),
                         "rmst_ref_days": round(U.rmst(f.eta0, f.beta0), 6),
                         "brt": 1.0,
                         "slb": round(f.contractor.get("slb", 1.0), 5),
                         "oth": round(f.contractor.get("oth", 1.0), 5)})
    for code, target in _alias_pairs():
        src = [r for r in rows if r["key"] == target]
        if src:
            rows.append({**src[0], "key": code})
    return pd.DataFrame(rows)


def qnom_block(runs: dict) -> pd.DataFrame:
    """``QnomBlock``: one row per key with the **deployed** θ at the shared knots."""
    rows = []
    for field, r in runs.items():
        for s, f in r.fits.items():
            th = U.deploy_theta_qnom(np.asarray(f.theta["qnom"], float))
            rows.append({"key": deploy_key(s),
                         **{f"q{int(k)}": round(float(t), 6)
                            for k, t in zip(U.QNOM_KNOTS, th)}})
    for code, target in _alias_pairs():
        src = [r for r in rows if r["key"] == target]
        if src:
            rows.append({**src[0], "key": code})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_outputs(out: dict) -> Path:
    d = results_dir(SLUG)
    t = d / "tables"
    out["baselines"].to_csv(t / "baselines.csv", index=False, encoding="utf-8-sig")
    out["layers"].to_csv(t / "layers.csv", index=False, encoding="utf-8-sig")
    out["contractor"].to_csv(t / "contractor_levels_audit.csv", index=False,
                             encoding="utf-8-sig")
    out["support"].to_csv(t / "support.csv", index=False, encoding="utf-8-sig")
    out["tune_block"].to_csv(t / "deploy_tune_block.csv", index=False, encoding="utf-8-sig")
    out["qnom_block"].to_csv(t / "deploy_qnom_block.csv", index=False, encoding="utf-8-sig")
    km_control(out["runs"], d / "figures" / "km_control.png")
    return d


def km_control(runs: dict, path: Path) -> Path:
    """Mandatory control: model survival against Kaplan–Meier, per stratum.

    A parametric baseline that has drifted off the KM is the one failure mode this workflow
    cannot detect from the likelihood alone.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    items = [(f, s, fit) for f, r in runs.items() for s, fit in r.fits.items()]
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (field, s, fit) in zip(axes.ravel(), items):
        g = runs[field].frame
        g = g[g["stratum"] == s]
        km = KaplanMeierFitter().fit(g[U.CLOCK], g[U.EVENT_COL])
        km.plot_survival_function(ax=ax, ci_show=True, color="#444444", label="KM")
        t = np.linspace(1, 1460, 400)
        ax.plot(t, np.exp(-((t / fit.eta0) ** fit.beta0)), color="#c0392b", lw=1.8,
                label="model (θ=1)")
        ax.set_title(f"{s}  n={fit.n}, отк.={fit.events}", fontsize=9)
        ax.set_xlim(0, 1460)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
