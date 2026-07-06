"""Phase A T6 — Extended Cox inference fix for the Vt H₂S γ.

The published γ p-values / R² in ``results/vt_failure/2026-06-29/`` come from OLS on
points sampled from survival curves — autocorrelated, with an arbitrary n on the grid
variant.  Those are **not valid statistics**.  This replaces them with a proper
partial-likelihood estimate:

  Primary:    episode-split the 366 sour/nonsour Vt runs at every distinct failure
              time and fit the time-interaction Cox
              h(t) = h₀(t)·exp(β·sour + γ·sour·log t)
              via lifelines ``CoxTimeVaryingFitter``.  → citable γ, SE, p.
  Cross-check: cluster bootstrap over wells (percentile CI for γ). Verify the old
              curve-regression M4/M5 point estimates (+0.079 KM, +0.343 mixture) fall
              inside the honest interval.

Episode-split convention (2026-07-06 correction): episodes are cut at **every distinct
failure time**, so each episode's ``stop`` is exactly the risk-set evaluation time and
``sour·log(stop)`` equals ``sour·log(t)`` for the case *and* every at-risk subject —
the exact time-interaction partial likelihood.  The first shipped version cut at fixed
30-day intervals with the covariate at episode *end*: cases then carried
``log(own failure time)`` while at-risk controls carried ``log(grid edge)`` — a
systematically lower covariate for cases, i.e. a mechanical negative bias on γ
(shipped γ = −0.851; corrected γ ≈ +0.2).  Do not reintroduce grid-based splitting
with end-of-episode covariates.

Produces, under ``results/phase_a_extended_cox_fix/<date>/``:
  tables/extended_cox_subject_level.csv  — β, γ, SE, p, HR from the split-data Cox
  tables/gamma_bootstrap_ci.csv          — γ percentile CI + M4/M5 containment check

Run:
    python scripts/run/phase_a_extended_cox_fix.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter

from analysis.paths import results_dir
from analysis.workflows.vt_failure.data import load_analysis_df
from analysis.models.survival.extended_cox import fit_extended_cox_log_log
# B0.2: the corrected episode-split now lives in a reusable, tested module.
from analysis.models.survival.time_interaction_cox import (
    episode_split as _episode_split_generic,
)

warnings.filterwarnings("ignore")

N_BOOTSTRAP = 300
# Old curve-regression γ point estimates to check for containment (review Act IV).
M4_GAMMA_KM = 0.079
M5_GAMMA_MIX = 0.343


def _subject_frame() -> pd.DataFrame:
    df = load_analysis_df()
    vt = df[df["is_vt"] & df["h2s_label"].isin(["Кислый", "Некислый"])].copy()
    vt = vt[vt["duration"] > 0]
    out = pd.DataFrame({
        "well_key": vt["well_key"].astype(str).values,
        "duration": vt["duration"].astype(float).values,
        "event": vt["event"].astype(int).values,
        "sour": (vt["h2s_label"] == "Кислый").astype(int).values,
    }).reset_index(drop=True)
    out["subject_id"] = np.arange(len(out))
    return out


def episode_split(subjects: pd.DataFrame) -> pd.DataFrame:
    """Long-format split at every distinct failure time (exact risk-set covariate).

    Thin adapter over ``time_interaction_cox.episode_split`` (B0.2) that keeps the
    ``sour`` / ``sour_logt`` column names this script's ``_fit_ctv`` expects.  The
    corrected split convention (cut at every distinct failure time; covariate at
    the shared risk-set time) lives in that reusable, tested module.
    """
    split = _episode_split_generic(
        subjects, duration_col="duration", event_col="event",
        covariate_col="sour", id_col="subject_id", cluster_col="well_key",
    )
    return split.rename(columns={"cov": "sour", "cov_logt": "sour_logt"})


def _curve_regression_gamma(subjects: pd.DataFrame) -> float | None:
    """M4-style curve-regression γ: KM(sour) vs KM(nonsour) → log-log OLS slope.

    This reproduces the *old* (descriptive-only) estimator so its uncertainty can be
    bootstrapped honestly. Returns None if a group has too few failures.
    """
    g1 = subjects[subjects["sour"] == 1]
    g0 = subjects[subjects["sour"] == 0]
    if int(g1["event"].sum()) < 5 or int(g0["event"].sum()) < 5:
        return None
    km1, km0 = KaplanMeierFitter(), KaplanMeierFitter()
    km1.fit(g1["duration"], g1["event"])
    km0.fit(g0["duration"], g0["event"])
    try:
        res = fit_extended_cox_log_log(km0.survival_function_, km1.survival_function_)
        return float(res.gamma)
    except Exception:
        return None


def _fit_ctv(split: pd.DataFrame, cluster: bool = True) -> "CoxTimeVaryingFitter":
    # lifelines 0.30.x CoxTimeVaryingFitter has no robust/cluster SE; the model SE
    # is reported for reference, but the citable uncertainty is the well bootstrap.
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(split[["subject_id", "start", "stop", "event", "sour", "sour_logt"]],
            id_col="subject_id", event_col="event",
            start_col="start", stop_col="stop")
    return ctv


def main() -> None:
    out = results_dir("phase_a_extended_cox_fix")
    tbl = out / "tables"

    subjects = _subject_frame()
    print(f"[T6] Vt sour/nonsour runs: {len(subjects)}  "
          f"(sour={int(subjects['sour'].sum())}, events={int(subjects['event'].sum())})")

    split = episode_split(subjects)
    print(f"[T6] Episode-split into {len(split)} intervals at every distinct failure time")

    # ── Primary: subject-level partial-likelihood Cox ─────────────────────────
    ctv = _fit_ctv(split, cluster=True)
    s = ctv.summary
    beta = float(s.loc["sour", "coef"])
    gamma = float(s.loc["sour_logt", "coef"])
    gamma_se = float(s.loc["sour_logt", "se(coef)"])
    gamma_p = float(s.loc["sour_logt", "p"])
    beta_p = float(s.loc["sour", "p"])
    prim = pd.DataFrame([
        {"term": "sour (β)", "coef": round(beta, 4), "se": round(float(s.loc["sour", "se(coef)"]), 4),
         "HR": round(float(np.exp(beta)), 3), "p": round(beta_p, 5)},
        {"term": "sour·log t (γ)", "coef": round(gamma, 4), "se": round(gamma_se, 4),
         "HR_per_logt": round(float(np.exp(gamma)), 3), "p": round(gamma_p, 5)},
    ])
    prim.to_csv(tbl / "extended_cox_subject_level.csv", index=False, encoding="utf-8-sig")
    print("\n[T6] PRIMARY — subject-level time-interaction Cox (partial likelihood):")
    print(prim.to_string(index=False))
    print(f"     γ_cox = {gamma:+.4f} (model SE {gamma_se:.4f}) on the hazard scale "
          f"[β·sour + γ·sour·log t]")
    print("     NOTE: γ_cox is a DIFFERENT estimand than the old curve-regression γ "
          "(log-log slope);")
    print("     the two are not directly comparable — see the curve-regression CI below.")

    # ── Bootstrap #1: subject-level Cox γ CI (over wells) ─────────────────────
    print(f"\n[T6] Well bootstrap of the Cox γ ({N_BOOTSTRAP} draws)...")
    wells = subjects["well_key"].unique()
    rng = np.random.default_rng(42)

    def _resample(seed_wells: np.ndarray) -> pd.DataFrame:
        parts = []
        for k, w in enumerate(seed_wells):
            s = subjects[subjects["well_key"] == w].copy()
            s["subject_id"] = s["subject_id"].astype(str) + f"_{k}"
            s["well_key"] = f"{w}_{k}"
            parts.append(s)
        return pd.concat(parts, ignore_index=True)

    cox_g, curve_g = [], []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(wells, size=len(wells), replace=True)
        bsub = _resample(drawn)
        try:
            cox_g.append(float(_fit_ctv(episode_split(bsub), cluster=False)
                               .summary.loc["sour_logt", "coef"]))
        except Exception:
            pass
        cg = _curve_regression_gamma(bsub)
        if cg is not None:
            curve_g.append(cg)

    def _ci(a):
        a = np.asarray(a)
        if len(a) < 20:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    cox_med, cox_lo, cox_hi = _ci(cox_g)
    pd.DataFrame([{
        "gamma_cox_point": round(gamma, 4),
        "gamma_cox_boot_median": round(cox_med, 4) if np.isfinite(cox_med) else None,
        "gamma_cox_ci_lo": round(cox_lo, 4) if np.isfinite(cox_lo) else None,
        "gamma_cox_ci_hi": round(cox_hi, 4) if np.isfinite(cox_hi) else None,
        "n_boot_ok": int(len(cox_g)),
    }]).to_csv(tbl / "gamma_cox_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(f"     Cox γ = {gamma:+.3f}  well-bootstrap 95% CI [{cox_lo:+.3f}, {cox_hi:+.3f}] "
          f"→ the citable subject-level uncertainty")

    # ── Bootstrap #2: curve-regression γ CI + M4/M5 containment (T6.2) ────────
    curve_point = _curve_regression_gamma(subjects)
    curve_med, curve_lo, curve_hi = _ci(curve_g)
    pd.DataFrame([{
        "gamma_curve_point": round(curve_point, 4) if curve_point is not None else None,
        "gamma_curve_boot_median": round(curve_med, 4) if np.isfinite(curve_med) else None,
        "gamma_curve_ci_lo": round(curve_lo, 4) if np.isfinite(curve_lo) else None,
        "gamma_curve_ci_hi": round(curve_hi, 4) if np.isfinite(curve_hi) else None,
        "n_boot_ok": int(len(curve_g)),
        "M4_km_gamma": M4_GAMMA_KM,
        "M4_inside_ci": bool(np.isfinite(curve_lo) and curve_lo <= M4_GAMMA_KM <= curve_hi),
        "M5_mix_gamma": M5_GAMMA_MIX,
        "M5_inside_ci": bool(np.isfinite(curve_lo) and curve_lo <= M5_GAMMA_MIX <= curve_hi),
    }]).to_csv(tbl / "gamma_curve_regression_ci.csv", index=False, encoding="utf-8-sig")
    print(f"\n[T6] Curve-regression γ (old estimand) = "
          f"{curve_point:+.3f}  well-bootstrap 95% CI [{curve_lo:+.3f}, {curve_hi:+.3f}]")
    print(f"     M4(+{M4_GAMMA_KM}) inside={bool(np.isfinite(curve_lo) and curve_lo<=M4_GAMMA_KM<=curve_hi)}, "
          f"M5(+{M5_GAMMA_MIX}) inside={bool(np.isfinite(curve_lo) and curve_lo<=M5_GAMMA_MIX<=curve_hi)}")

    print(f"\n[T6] Outputs written to: {out}")


if __name__ == "__main__":
    main()
