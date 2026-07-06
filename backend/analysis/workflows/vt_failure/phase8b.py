"""Phase 8b — Extended Cox model for H2S in Vt (time-varying coefficient).

Population : Vt field, all runs with known H2S classification ("raw" baseline,
             no infant-mortality filter).
Covariate  : is_acidic  (X=0 Некислый, X=1 Кислый)
Duration   : duration column (ttf_true_best_days, falling back to run_days)

Models fit
----------
1. Conventional Cox PH  — CoxPHFitter(is_acidic), cluster-robust on well_key
2. Extended Cox          — log-log OLS  y(t) = β* + γ·log(t)
   with Weibull shape correction β = β* − log[β₀/(β₀+γ)]

Outputs
-------
tables/
    p8b_summary.csv           — side-by-side comparison table
    p8b_extended_cox_points.csv — (time, y_obs, y_fit) for audit
figures/
    p8b_loglog.png            — log(−log S) vs log(t) for both groups
    p8b_y_regression.png      — y(t) with fitted line + 95% pointwise CI
    p8b_tv_hr.png             — time-varying HR(t) vs conventional HR
    p8b_schoenfeld.png        — Schoenfeld residuals from conventional Cox
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from analysis.models.survival.extended_cox import fit_extended_cox_log_log
from analysis.models.survival.weibull_model import fit_basic_weibull


# ── colour palette ────────────────────────────────────────────────────────────
_C = {"acidic": "#d62728", "neutral": "#2ca02c"}

_LABEL_ACIDIC  = "Кислый (H₂S)"
_LABEL_NEUTRAL = "Некислый"


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare_vt_h2s(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Vt field with known H2S, add binary is_acidic, drop trivial durations."""
    vt = df[df["is_vt"]].copy()
    known = vt[vt["h2s_label"].isin(["Кислый", "Некислый"])].copy()
    known["is_acidic"] = (known["h2s_label"] == "Кислый").astype(float)
    # drop zero / negative durations
    known = known[known["duration"] > 0].reset_index(drop=True)
    return known


def _fit_km(sub: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(sub["duration"], sub["event"])
    return kmf


def _fit_weibull_beta(sub: pd.DataFrame) -> float:
    """Fit single-group Weibull and return β (shape parameter)."""
    d = sub["duration"].to_numpy(float)
    e = sub["event"].to_numpy(int)
    try:
        res = fit_basic_weibull(d, e)
        return float(res["beta"])
    except Exception:
        return 1.0


# ── figure 1 — log(−log S) vs log(t) ─────────────────────────────────────────

def _plot_loglog(km0: KaplanMeierFitter, km1: KaplanMeierFitter, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for kmf, label, colour in [
        (km0, _LABEL_NEUTRAL, _C["neutral"]),
        (km1, _LABEL_ACIDIC,  _C["acidic"]),
    ]:
        sf = kmf.survival_function_
        t = sf.index.values
        s = sf.iloc[:, 0].values
        valid = (s > 0) & (s < 1) & (t > 0)
        ax.plot(
            np.log(t[valid]),
            np.log(-np.log(s[valid])),
            label=label,
            color=colour,
            linewidth=1.8,
        )

    ax.set_xlabel("log(t)  [t in days]")
    ax.set_ylabel("log(−log S(t))")
    ax.set_title(
        "Log-log survival plot — H₂S groups (Vt)\n"
        "Parallel lines ⇒ proportional hazards"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8b_loglog.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── figure 2 — y(t) regression ───────────────────────────────────────────────

def _plot_y_regression(res, out: Path) -> None:
    times = res.times
    y_obs = res.y_observed
    y_fit = res.y_fitted

    t_grid = np.linspace(times.min(), times.max(), 200)
    hr_lo, hr_hi = res.hr_ci(t_grid)
    # convert back to log-log difference scale (they're log(HR))
    y_lo = np.log(np.clip(hr_lo, 1e-9, None))
    y_hi = np.log(np.clip(hr_hi, 1e-9, None))
    log_t_grid = np.log(t_grid)
    log_t_fit = np.log(times)
    y_grid = res.beta_star + res.gamma * log_t_grid

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log(times), y_obs, s=20, alpha=0.6, color="#555", label="y(t) observed")
    ax.plot(log_t_grid, y_grid, color="#1f77b4", linewidth=2,
            label=f"β* + γ·log(t)  γ={res.gamma:+.3f}  p={res.p_gamma:.3f}")
    ax.fill_between(log_t_grid, y_lo, y_hi, alpha=0.15, color="#1f77b4", label="95% CI")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="y=0  (proportional hazards)")

    ax.set_xlabel("log(t)  [t in days]")
    ax.set_ylabel("y(t) = log(−log S₁) − log(−log S₀)")
    ax.set_title(
        f"Extended Cox — H₂S effect over time (Vt)\n"
        f"β*={res.beta_star:+.3f}  γ={res.gamma:+.3f}  R²={res.r_squared:.3f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8b_y_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── figure 3 — time-varying HR ───────────────────────────────────────────────

def _plot_tv_hr(res, cox_hr: float, cox_ci: tuple[float, float], out: Path) -> None:
    t_max = float(res.times.max())
    t_grid = np.linspace(1, t_max, 300)
    hr_tv = res.hr_profile(t_grid)
    hr_lo, hr_hi = res.hr_ci(t_grid)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t_grid, hr_tv, color="#d62728", linewidth=2, label="Extended Cox HR(t)")
    ax.fill_between(t_grid, hr_lo, hr_hi, alpha=0.15, color="#d62728", label="95% CI")

    # Conventional Cox HR
    ax.axhline(cox_hr, color="#1f77b4", linewidth=1.8, linestyle="--",
               label=f"Conventional Cox HR={cox_hr:.3f}")
    ax.axhspan(cox_ci[0], cox_ci[1], alpha=0.10, color="#1f77b4")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Hazard Ratio  (Кислый vs Некислый)")
    ax.set_title("Time-varying HR for H₂S (Vt)\nExtended Cox vs conventional Cox")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8b_tv_hr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── figure 4 — Schoenfeld residuals ──────────────────────────────────────────

def _plot_schoenfeld(cph: CoxPHFitter, sub: pd.DataFrame, out: Path) -> None:
    try:
        ph = proportional_hazard_test(cph, sub, time_transform="rank")
        p_val = float(ph.summary["p"].iloc[0])
    except Exception:
        p_val = np.nan

    try:
        resid = cph.compute_residuals(sub, kind="schoenfeld")
        col = resid.columns[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(resid.index, resid[col], s=18, alpha=0.5, color="#555")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        # smoothed trend
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(resid.index)
        t_sorted = resid.index.values[sorted_idx]
        r_sorted = resid[col].values[sorted_idx]
        if len(r_sorted) > 10:
            smooth = uniform_filter1d(r_sorted, size=max(3, len(r_sorted) // 10))
            ax.plot(t_sorted, smooth, color="#d62728", linewidth=1.5, label="Trend")
        p_text = f"Schoenfeld p={p_val:.3f}" if np.isfinite(p_val) else "Schoenfeld p=N/A"
        ax.set_xlabel("Rank-transformed time")
        ax.set_ylabel("Schoenfeld residual  (is_acidic)")
        ax.set_title(f"Proportional hazard test — H₂S (Vt)\n{p_text}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "p8b_schoenfeld.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    return p_val


# ── conventional Cox ─────────────────────────────────────────────────────────

def _fit_conventional_cox(sub: pd.DataFrame) -> dict:
    fit_df = sub[["duration", "event", "is_acidic", "well_key"]].dropna().copy()
    cph = CoxPHFitter()
    try:
        cph.fit(
            fit_df,
            duration_col="duration",
            event_col="event",
            cluster_col="well_key",
            robust=True,
        )
    except Exception as exc:
        return {"error": str(exc)}

    row = cph.summary.loc["is_acidic"]
    ph_p = np.nan
    try:
        ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_p = float(ph.summary["p"].iloc[0])
    except Exception:
        pass

    return {
        "cox_hr": float(row["exp(coef)"]),
        "cox_ci_lo": float(row["exp(coef) lower 95%"]),
        "cox_ci_hi": float(row["exp(coef) upper 95%"]),
        "cox_p": float(row["p"]),
        "cox_concordance": float(cph.concordance_index_),
        "cox_ph_p": ph_p,
        "cox_ph_ok": bool(ph_p > 0.05) if np.isfinite(ph_p) else None,
        "_cph_obj": cph,
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run(df: pd.DataFrame, out: Path) -> dict:
    """
    Parameters
    ----------
    df  : full analysis dataframe from data.load_analysis_df()
    out : output directory (tables parent); figures go to sibling figures/phase8b_extended_cox/
    """
    tbl_out = out
    fig_out = out.parent.parent / "figures" / "phase8b_extended_cox"
    tbl_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # ── 1. data prep ─────────────────────────────────────────────────────────
    sub = _prepare_vt_h2s(df)
    n_total = len(sub)
    n_acidic  = int((sub["is_acidic"] == 1).sum())
    n_neutral = int((sub["is_acidic"] == 0).sum())
    n_fail_ac = int(sub.loc[sub["is_acidic"] == 1, "event"].sum())
    n_fail_ne = int(sub.loc[sub["is_acidic"] == 0, "event"].sum())

    print(f"\n{'=' * 60}")
    print("PHASE 8b — EXTENDED COX  (H₂S, Vt, raw baseline)")
    print(f"{'=' * 60}")
    print(f"  Vt runs with known H₂S : {n_total}")
    print(f"  Кислый  (X=1)          : {n_acidic}  ({n_fail_ac} failures)")
    print(f"  Некислый (X=0)          : {n_neutral} ({n_fail_ne} failures)")

    if n_fail_ac < 10 or n_fail_ne < 10:
        print("  ⚠ Too few failures in one group — skipping phase.")
        return results

    # ── 2. KM fits ───────────────────────────────────────────────────────────
    neutral_sub = sub[sub["is_acidic"] == 0]
    acidic_sub  = sub[sub["is_acidic"] == 1]

    km_neutral = _fit_km(neutral_sub)
    km_acidic  = _fit_km(acidic_sub)

    # ── 3. Baseline Weibull shape β₀ ─────────────────────────────────────────
    beta0 = _fit_weibull_beta(neutral_sub)
    print(f"  Baseline Weibull β₀    : {beta0:.3f}")

    # ── 4. Extended Cox ───────────────────────────────────────────────────────
    exc_res = fit_extended_cox_log_log(
        km_baseline_sf=km_neutral.survival_function_,
        km_treatment_sf=km_acidic.survival_function_,
        baseline_weibull_beta=beta0,
    )
    results["extended_cox"] = exc_res.summary_dict()

    # ── 5. Conventional Cox ───────────────────────────────────────────────────
    cox = _fit_conventional_cox(sub)
    results["conventional_cox"] = {k: v for k, v in cox.items() if not k.startswith("_")}
    cph_obj = cox.get("_cph_obj")

    # ── 6. Comparison table ───────────────────────────────────────────────────
    rows = [
        {
            "model": "Conventional Cox PH",
            "parameter": "HR (is_acidic)",
            "estimate": round(cox.get("cox_hr", np.nan), 3),
            "ci_lo": round(cox.get("cox_ci_lo", np.nan), 3),
            "ci_hi": round(cox.get("cox_ci_hi", np.nan), 3),
            "p_value": round(cox.get("cox_p", np.nan), 4),
            "ph_test_p": round(cox.get("cox_ph_p", np.nan), 4),
            "note": "Assumes constant HR over time",
        },
        {
            "model": "Extended Cox (log-log OLS)",
            "parameter": "γ  (log-t interaction)",
            "estimate": round(exc_res.gamma, 3),
            "ci_lo": round(exc_res.gamma - 1.96 * exc_res.se_gamma, 3),
            "ci_hi": round(exc_res.gamma + 1.96 * exc_res.se_gamma, 3),
            "p_value": round(exc_res.p_gamma, 4),
            "ph_test_p": np.nan,
            "note": "γ≠0 ⇒ time-varying HR",
        },
        {
            "model": "Extended Cox (log-log OLS)",
            "parameter": "β* (corrected intercept)",
            "estimate": round(exc_res.beta_star, 3),
            "ci_lo": round(exc_res.beta_star - 1.96 * exc_res.se_beta_star, 3),
            "ci_hi": round(exc_res.beta_star + 1.96 * exc_res.se_beta_star, 3),
            "p_value": round(exc_res.p_beta_star, 4),
            "ph_test_p": np.nan,
            "note": f"Weibull correction: β₀={beta0:.3f}, β={exc_res.beta:.3f}",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tbl_out / "p8b_summary.csv", index=False, encoding="utf-8-sig")
    results["summary_df"] = summary_df

    # audit points
    audit_df = pd.DataFrame({
        "time_days": exc_res.times,
        "y_observed": exc_res.y_observed,
        "y_fitted": exc_res.y_fitted,
    })
    audit_df.to_csv(tbl_out / "p8b_extended_cox_points.csv", index=False, encoding="utf-8-sig")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    _plot_loglog(km_neutral, km_acidic, fig_out)

    _plot_y_regression(exc_res, fig_out)

    _plot_tv_hr(
        exc_res,
        cox_hr=cox.get("cox_hr", 1.0),
        cox_ci=(cox.get("cox_ci_lo", 1.0), cox.get("cox_ci_hi", 1.0)),
        out=fig_out,
    )

    if cph_obj is not None:
        fit_df = sub[["duration", "event", "is_acidic", "well_key"]].dropna().copy()
        _plot_schoenfeld(cph_obj, fit_df, fig_out)

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print(f"\n  {'Model':<30}  {'Parameter':<25}  {'Estimate':>10}  {'95% CI':>22}  {'p':>8}")
    print("  " + "-" * 105)
    for row in rows:
        lo = row["ci_lo"]
        hi = row["ci_hi"]
        print(
            f"  {row['model']:<30}  {row['parameter']:<25}  "
            f"{row['estimate']:>10.3f}  [{lo:.3f}, {hi:.3f}]  {row['p_value']:>8.4f}"
        )

    gamma = exc_res.gamma
    if exc_res.p_gamma < 0.05:
        direction = "decreasing" if gamma < 0 else "increasing"
        print(f"\n  γ={gamma:+.3f} is significant (p={exc_res.p_gamma:.3f}): "
              f"H₂S effect is {direction} over time — PH violated.")
    else:
        print(f"\n  γ={gamma:+.3f} (p={exc_res.p_gamma:.3f}): "
              f"no significant time interaction — conventional Cox HR is adequate.")

    ph_ok = cox.get("cox_ph_ok")
    if ph_ok is False:
        print(f"  Schoenfeld test p={cox.get('cox_ph_p', np.nan):.3f}: "
              "proportional hazard assumption REJECTED for conventional Cox.")
    elif ph_ok is True:
        print(f"  Schoenfeld test p={cox.get('cox_ph_p', np.nan):.3f}: "
              "proportional hazard assumption not rejected.")

    n_pts = exc_res.n_points
    r2 = exc_res.r_squared
    print(f"\n  Extended Cox log-log regression: n_points={n_pts}, R²={r2:.3f}")
    print(f"  n_total={n_total}  "
          f"Кислый={n_acidic} (fail={n_fail_ac})  "
          f"Некислый={n_neutral} (fail={n_fail_ne})")
    print(f"\n  Outputs → {tbl_out}  &  {fig_out}")
    print(f"{'=' * 60}\n")

    return results
