"""Мирнинский (Mc) failure/ГТМ competing-risks accounting.

The Mc failure-count forecast matches actual monthly failures.  The question this
module answers is the natural follow-up: *shouldn't it predict MORE, with the
surplus consumed by workovers (ГТМ)?*  The answer depends on two things and this
module quantifies both on the Mc 2024+ cohort:

1. **Hazard shape.**  If λ_fail is flat (no wear-out), pumps pulled for ГТМ were
   not about to fail, so removing ГТМ adds no failure surplus; and renewing a
   pump to age 0 manufactures no extra infant exposure.  If λ_fail has an infant
   spike, every ГТМ renewal re-enters that spike and ADDS failure exposure.
2. **Whether ГТМ is de-facto preventive** (pulls target pumps about to fail).  If
   so the observed-failure hazard understates the latent no-ГТМ hazard.

**One clock for everything.**  The known ``tte`` trap mixes clocks — events timed
on Наработка, censorings on calendar — which biases survival up (on Mc 60/61 open
runs fell through to calendar).  Here **every** run, event, pull and open pump is
timed on ``t_cal`` (calendar: ``pull-install`` or ``as_of-install`` while running),
the single clock that is always available and never imputed.  We assert no open run
gets a different clock than events.  Note: the day-8 infant spike documented for Mc
lives on the *operating*-day clock; on ``t_cal`` downtime smears it out, so the
calendar hazard reads flat — that is a property of the clock, stated, not hidden.

The estimators (cause-specific Weibull, band hazards, AJ CIF, renewal engine,
informative-censoring test) live in :mod:`competing_risks_core` and are shared
byte-for-byte with the Vt accounting; this module keeps only the Mc-specific
population build, canonical comparison, figures and orchestration.

Outputs → ``results_dir("production_risk_mc_competing_risks")``.  Do NOT commit.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.survival import cif as CIF
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import competing_risks_core as core
from analysis.workflows.production_risk import esp_population as P

# Re-export the shared engine so existing consumers (tests, CLI) keep using
# ``mc_competing_risks.<name>`` unchanged.
from analysis.workflows.production_risk.competing_risks_core import (  # noqa: F401
    CENSORED,
    FAILURE,
    GTM,
    CAUSE_ORDER,
    HAZARD_BANDS,
    FLAT_RENEWAL_AGE,
    CIF_EVAL_TIMES,
    classify_cause,
    CauseFit,
    fit_cause_specific,
    piecewise_hazard,
    cif_vs_km,
    BandHazards,
    band_hazards_from_pop,
    simulate_renewal,
    observed_window_prediction,
    monthly_decomposition,
    combined_decomposition,
)

# Мирнинский telemetry prefixes (Мс + Мр pads share the field/stratum).
MC_TELEMETRY_PREFIXES = ("MC", "MR")


# --------------------------------------------------------------------------- #
# Population (single t_cal clock) — Mc-specific                                  #
# --------------------------------------------------------------------------- #
def build_population(as_of: date | pd.Timestamp | None = None) -> pd.DataFrame:
    """Mc 2024+ cohort, one row per run, on the single ``t_cal`` clock.

    Columns: ``code`` (well id, also the bootstrap cluster), ``install``, ``end``,
    ``t_cal`` (duration for every fit and the renewal), ``cause_code``,
    ``cause`` (label), ``is_open`` (running), ``current_age`` (= t_cal for open
    runs), ``contractor_group``.
    """
    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    pop = P.add_time_scales(P.build(asof, gtm_is_failure=False), asof)
    mc = P.apply_mc_cohort(pop)
    mc = mc[mc["field"].isin(C.MC_COHORT_FIELDS)].copy()

    ended = mc["end"].notna() & (mc["end"] <= asof)
    mc["is_open"] = ~ended
    mc["cause_code"] = [
        classify_cause(r, u, e)
        for r, u, e in zip(mc["pull_reason"], mc["has_failed_unit"], ended)
    ]
    mc["cause"] = mc["cause_code"].map({CENSORED: "running", FAILURE: "failure", GTM: "gtm"})
    mc["current_age"] = mc["t_cal"].astype(float)
    mc["well_key"] = mc["code"]

    # Single-clock invariant: every run — open and closed, event and censor — is
    # timed on t_cal (calendar).  For open runs that means t_cal == as_of - install,
    # NOT the run population's ``tte`` (the mixed-clock trap).  Assert open runs are
    # on the calendar clock alongside events.
    op = mc[mc["is_open"]]
    if len(op):
        cal_age = (asof - op["install"]).dt.days.clip(lower=P.DAY_ZERO_TTE).astype(float)
        mismatch = int((~np.isclose(op["t_cal"], cal_age)).sum())
        if mismatch:
            raise AssertionError(
                f"{mismatch} open Mc runs not on the calendar clock (t_cal != as_of-install) "
                "— refusing to fit on a mixed clock."
            )
        if not np.isfinite(mc["t_cal"]).all() or (mc["t_cal"] <= 0).any():
            raise AssertionError("non-finite or non-positive t_cal in the Mc cohort.")
    return mc.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Informative-censoring test — Mc telemetry wrappers                             #
# --------------------------------------------------------------------------- #
def _mc_well_mean_qliq() -> pd.Series:
    """Well-level mean liquid rate for Мирнинский (a rate proxy)."""
    return core.well_mean_qliq(MC_TELEMETRY_PREFIXES)


def informative_censoring_level1(pop: pd.DataFrame) -> pd.DataFrame:
    """Level 1 association test on the Мирнинский telemetry (see core)."""
    return core.informative_censoring_level1(pop, MC_TELEMETRY_PREFIXES)


def informative_censoring_level2(
    pop: pd.DataFrame, *, window_days: int = 21, min_runs: int = 6
) -> pd.DataFrame:
    """Level 2 telemetry-precursor test on the Мирнинский telemetry (see core)."""
    return core.informative_censoring_level2(
        pop, MC_TELEMETRY_PREFIXES, window_days=window_days, min_runs=min_runs
    )


# --------------------------------------------------------------------------- #
# Reconcile with the canonical forecast — Mc-specific                           #
# --------------------------------------------------------------------------- #
# The canonical cal-c0-k2 Mc build (results/production_risk_mc_svodprognoz/
# 2026-07-17) forecasts, for 2026-07 … 2027-06, a 12-month failure total of 48.60
# on a fleet that GROWS 67 -> 79 because it adds the plan's ВНС new wells.  Our
# competing-risks forecast is seeded from the currently-running pumps only, so the
# absolute totals differ by the ВНС universe, NOT by the competing-risks modelling.
CANONICAL_12M_FAILURE_TOTAL = 48.60
CANONICAL_SOURCE = "results/production_risk_mc_svodprognoz/2026-07-17 (Сводпрогноз_Mc.xlsx, cal c0 k2)"


def canonical_comparison(
    pop: pd.DataFrame,
    hazards: BandHazards,
    decomp: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Compare column (1) to the canonical build and to a single-hazard renewal."""
    fc = decomp[decomp["segment"] == "forecast"]
    cr_total = float(fc["m1_obs_failures"].sum())
    n_running = int(pop["is_open"].sum())

    single_total = float(fc["m4_latent_nogtm_failures"].sum())
    move_pct = 100.0 * (cr_total - single_total) / single_total if single_total > 0 else float("nan")
    rows = [
        {
            "comparison": "competing_risks_col1_vs_single_hazard_same_fleet",
            "competing_risks_12m": round(cr_total, 2),
            "reference_12m": round(single_total, 2),
            "move_pct": round(move_pct, 1),
            "materially_moves_gt5pct": bool(abs(move_pct) > 5.0) if np.isfinite(move_pct) else False,
            "note": f"isolates the competing-risks modelling change on the identical running fleet ({n_running} pumps)",
        },
        {
            "comparison": "competing_risks_col1_vs_canonical_absolute",
            "competing_risks_12m": round(cr_total, 2),
            "reference_12m": CANONICAL_12M_FAILURE_TOTAL,
            "move_pct": round(100.0 * (cr_total - CANONICAL_12M_FAILURE_TOTAL) / CANONICAL_12M_FAILURE_TOTAL, 1),
            "materially_moves_gt5pct": None,
            "note": f"NOT apples-to-apples: canonical fleet grows 67->79 incl. plan ВНС; ours is the {n_running} running pumps only. Source: {CANONICAL_SOURCE}",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures (Russian labels) + orchestrator — Mc-specific                         #
# --------------------------------------------------------------------------- #
def _cif_curves_for_plot(pop: pd.DataFrame, tmax: float):
    t = pop["t_cal"].to_numpy(float)
    ec = pop["cause_code"].to_numpy(int)
    grid = np.linspace(0.0, tmax, 400)
    cf = CIF.aalen_johansen_cif(t, ec, FAILURE)
    cg = CIF.aalen_johansen_cif(t, ec, GTM)
    naive = CIF.all_cause_km(t, (ec == FAILURE).astype(int))
    return grid, cf.at(grid), cg.at(grid), naive.at(grid)


def figure_cif_stack(pop: pd.DataFrame, path: Path, tmax: float = 730.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, cif_f, cif_g, naive = _cif_curves_for_plot(pop, tmax)
    running = 1.0 - cif_f - cif_g
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.stackplot(
        grid, cif_f, cif_g, running,
        labels=["Отказ (AJ CIF)", "ГТМ (AJ CIF)", "В работе"],
        colors=["#c0392b", "#e67e22", "#d9e2ec"], alpha=0.9,
    )
    ax.plot(grid, naive, "k--", lw=1.8, label="Наивная 1−KM (отказ, ГТМ цензурир.)")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Наработка (календарные дни, t_cal)")
    ax.set_ylabel("Кумулятивная доля")
    ax.set_title("Мирнинский: конкурирующие риски отказ/ГТМ (Aalen–Johansen)\nнаивная кривая переоценивает частоту отказов")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_hazard_shapes(pop: pd.DataFrame, fits: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hf = piecewise_hazard(pop, FAILURE)
    hg = piecewise_hazard(pop, GTM)
    centers = [(lo + (hi if np.isfinite(hi) else lo + 365)) / 2 for lo, hi in HAZARD_BANDS]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.step(centers, hf["hazard_per_1000d"], where="mid", color="#c0392b", lw=2, label="λ_отказ (эмпир., по бэндам)")
    ax.step(centers, hg["hazard_per_1000d"], where="mid", color="#e67e22", lw=2, label="λ_ГТМ (эмпир., по бэндам)")
    grid = np.linspace(1, 730, 400)
    for cause, color in (("failure", "#7b241c"), ("gtm", "#af601a")):
        f = fits[cause]
        haz = (f.beta / f.eta) * np.power(grid / f.eta, f.beta - 1.0) * 1000.0
        ax.plot(grid, haz, "--", color=color, lw=1.3,
                label=f"{cause} Вейбулл β={f.beta:.2f}")
    ax.set_xlabel("Возраст насоса (календарные дни)")
    ax.set_ylabel("Опасность на 1000 дней")
    ax.set_title("Мирнинский: форма опасности по причинам\nλ_отказ ≈ плоская (нет износа; инфант-всплеск — на op-часах, не на календаре)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_decomposition(decomp: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = decomp.copy()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    obs = d["segment"] == "observed"
    fc = d["segment"] == "forecast"
    ax.bar(x[obs], d.loc[obs, "m1_obs_failures"], color="#2e86c1", alpha=0.85, label="Модель, отказы (наблюд. окно)")
    ax.bar(x[fc], d.loc[fc, "m1_obs_failures"], color="#c0392b", alpha=0.9, label="Прогноз отказов (кол.1)")
    ax.plot(x[fc], d.loc[fc, "m4_latent_nogtm_failures"], "o-", color="#117864", lw=1.8,
            label="Латентный контрфакт без ГТМ (кол.4)")
    ax.scatter(x[obs], d.loc[obs, "actual_failures"], color="black", zorder=5, s=28, label="Факт отказов")
    ax.axvline(x[fc][0] - 0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["month"], rotation=90, fontsize=7)
    ax.set_ylabel("Отказы в месяц")
    ax.set_title("Мирнинский: помесячная декомпозиция — прогноз vs факт vs латентный контрфакт\nлатентный ≈ наблюдаемому (НЕ 2×): плоская опасность ⇒ ГТМ не «съедает» массу отказов")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_analysis(
    as_of: date | pd.Timestamp | None = None,
    *,
    horizon_months: int = 12,
    n_boot: int = 300,
    write_figures: bool = True,
) -> dict:
    """End-to-end: build the Mc cohort, run tasks A–E, write tables/figures.

    Returns a dict of the produced frames and the output directory.
    """
    from analysis.paths import results_dir

    asof = pd.Timestamp(as_of if as_of is not None else C.SVOD_OPEN_ASOF)
    out = results_dir("production_risk_mc_competing_risks")
    tables, figures = out / "tables", out / "figures"

    pop = build_population(asof)
    hazards = band_hazards_from_pop(pop)

    # A — cause-specific fits
    fits = {c: fit_cause_specific(pop, code, n_boot=n_boot)
            for c, code in (("failure", FAILURE), ("gtm", GTM))}
    params = pd.DataFrame(
        [
            {
                "cause": f.cause, "n_events": f.n_events, "n_at_risk": f.n_at_risk,
                "beta": round(f.beta, 4), "beta_ci_lo": round(f.beta_lo, 3), "beta_ci_hi": round(f.beta_hi, 3),
                "eta": round(f.eta, 1), "eta_ci_lo": round(f.eta_lo, 1), "eta_ci_hi": round(f.eta_hi, 1),
                "rmst_0_730": round(f.rmst_0_730, 1), "mrl_0": round(f.mrl_0, 1),
                "n_boot_ok": f.n_boot_ok,
            }
            for f in fits.values()
        ]
    )
    haz_bands = pd.concat([piecewise_hazard(pop, FAILURE), piecewise_hazard(pop, GTM)], ignore_index=True)
    params.to_csv(tables / "cause_specific_params.csv", index=False, encoding="utf-8-sig")
    haz_bands.to_csv(tables / "cause_specific_hazard_bands.csv", index=False, encoding="utf-8-sig")

    # cross-check event counts vs esp_models.csv Mc row
    _assert_event_counts(pop, fits)

    # B — CIF vs naive KM
    cif = cif_vs_km(pop)
    cif.to_csv(tables / "cif_vs_km.csv", index=False, encoding="utf-8-sig")

    # C — informative censoring
    l1 = informative_censoring_level1(pop)
    l2 = informative_censoring_level2(pop)
    l1.to_csv(tables / "informative_censoring.csv", index=False, encoding="utf-8-sig")
    if len(l2):
        l2.to_csv(tables / "informative_censoring_telemetry_level2.csv", index=False, encoding="utf-8-sig")

    # D — decomposition (observed calibration window + 12-month forecast)
    obs = observed_window_prediction(pop, hazards)
    decomp = combined_decomposition(pop, hazards, obs, as_of=asof, horizon_months=horizon_months)
    decomp.to_csv(tables / "monthly_decomposition.csv", index=False, encoding="utf-8-sig")

    # E — canonical comparison
    canon = canonical_comparison(pop, hazards, decomp, as_of=asof, horizon_months=horizon_months)
    canon.to_csv(tables / "canonical_comparison.csv", index=False, encoding="utf-8-sig")

    if write_figures:
        figure_cif_stack(pop, figures / "cif_stack_vs_naive_km.png")
        figure_hazard_shapes(pop, fits, figures / "cause_specific_hazard_shapes.png")
        figure_decomposition(decomp, figures / "monthly_decomposition.png")

    return {
        "out": out, "pop": pop, "fits": fits, "params": params, "hazard_bands": haz_bands,
        "cif": cif, "l1": l1, "l2": l2, "obs": obs, "decomp": decomp, "canonical": canon,
    }


def _assert_event_counts(pop: pd.DataFrame, fits: dict) -> None:
    """Cross-check the fitted failure event count against esp_models.csv (repo rule)."""
    import warnings

    n_fail = fits["failure"].n_events
    try:
        reg = pd.read_csv(C.model_registry_path(), encoding="utf-8-sig")
        mc = reg[reg["stratum"].astype(str).str.startswith("Mc")]
        reg_fail = int(mc["n_failures"].sum()) if not mc.empty else -1
    except Exception as exc:  # pragma: no cover - registry optional
        warnings.warn(f"could not read esp_models.csv for event-count cross-check ({exc})")
        return
    if reg_fail >= 0 and abs(n_fail - reg_fail) > max(10, int(0.3 * reg_fail)):
        warnings.warn(
            f"Mc failure count {n_fail} diverges sharply from esp_models.csv {reg_fail} "
            "— check for a population defect before trusting the fit."
        )


__all__ = [
    "run_analysis",
    "build_population",
    "classify_cause",
    "fit_cause_specific",
    "piecewise_hazard",
    "cif_vs_km",
    "informative_censoring_level1",
    "informative_censoring_level2",
    "BandHazards",
    "band_hazards_from_pop",
    "simulate_renewal",
    "observed_window_prediction",
    "monthly_decomposition",
    "combined_decomposition",
    "canonical_comparison",
    "figure_cif_stack",
    "figure_hazard_shapes",
    "figure_decomposition",
]
