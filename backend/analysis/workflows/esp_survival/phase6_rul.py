"""Phase 6 — Remaining Useful Life (RUL) calculator.

For a pump with known (stratum, covariates, current age t₀):
  S_conditional(t | t₀) = [S_base(t) · cox_adj(t)] / [S_base(t₀) · cox_adj(t₀)]

  where:
    S_base(t) = w₁·exp(-(t/η₁)^β₁) + w₂·exp(-(t/η₂)^β₂)  (stratum mixture)
    cox_adj(t, X) = exp(Σ βᵢ·Xᵢ)  (static covariates; extended terms add Xᵢ·log t)

  RUL = ∫_{t₀}^{∞} S_conditional(t | t₀) dt

Success criteria:
  - RUL decreasing with age for pumps past ~100 days (wear-out regime)
  - Vt sour RUL at t₀=60d < Vt non-sour at same age
  - Within Vt sour: Шлюмберже RUL > Борец > Новые технологии at same age
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    latent_survival,
)


EPSILON = 1e-9
T_HORIZON_DAYS = 2000
N_GRID = 2000


def _base_survival(t: np.ndarray, model: TwoComponentLatentWeibullModel) -> np.ndarray:
    return np.asarray(latent_survival(t, model), dtype=float)


def _cox_adjustment(
    t: np.ndarray,
    coef: dict[str, float],
    X_static: dict[str, float],
    X_extended: dict[str, float] | None = None,
) -> np.ndarray:
    """Time-varying Cox adjustment factor exp(Σ βᵢ·Xᵢ + Σ γⱼ·Xⱼ·log(t))."""
    static_lp = sum(coef.get(k, 0.0) * v for k, v in X_static.items())
    log_t = np.log(np.clip(t, EPSILON, None))
    ext_lp = np.zeros_like(t, dtype=float)
    if X_extended:
        for k, v in X_extended.items():
            ext_lp += coef.get(k, 0.0) * v * log_t
    return np.exp(np.clip(static_lp + ext_lp, -20.0, 20.0))


def compute_rul(
    t0: float,
    model: TwoComponentLatentWeibullModel,
    cox_coef: dict[str, float] | None = None,
    X_static: dict[str, float] | None = None,
    X_extended: dict[str, float] | None = None,
    t_horizon: float = T_HORIZON_DAYS,
    n_grid: int = N_GRID,
) -> dict[str, float]:
    """Compute conditional RUL = E[T - t₀ | T > t₀].

    Parameters
    ----------
    t0 : current age in days
    model : stratum mixture model
    cox_coef : Extended Cox coefficient dict {covariate: beta}
    X_static : static covariate values {covariate: value}
    X_extended : covariates entering as X·log(t) {covariate: value}
    """
    t_grid = np.linspace(t0 + 1e-6, t_horizon, n_grid)
    coef = cox_coef or {}
    X_s = X_static or {}
    X_e = X_extended or {}

    S_base_t0 = float(_base_survival(np.array([t0 + EPSILON]), model)[0])
    adj_t0 = float(_cox_adjustment(np.array([t0 + EPSILON]), coef, X_s, X_e)[0])
    denom = S_base_t0 * adj_t0

    if denom < EPSILON:
        return {"rul": float("nan"), "t0": t0, "p50_remaining": float("nan")}

    S_base_grid = _base_survival(t_grid, model)
    adj_grid = _cox_adjustment(t_grid, coef, X_s, X_e)
    S_cond = (S_base_grid * adj_grid) / denom
    S_cond = np.clip(S_cond, 0.0, 1.0)

    # RUL = integral of conditional survival (remaining time from t0)
    rul = float(np.trapz(S_cond, t_grid - t0))

    # P50 remaining life: find remaining time r such that S_cond(t0+r) ≈ 0.5
    t_remaining = t_grid - t0
    p50_remaining = float("nan")
    idx_below = np.where(S_cond <= 0.5)[0]
    if len(idx_below) > 0:
        p50_remaining = float(t_remaining[idx_below[0]])

    return {
        "rul": round(rul, 1),
        "t0": t0,
        "p50_remaining_days": round(p50_remaining, 1) if not np.isnan(p50_remaining) else None,
    }


def compute_rul_profile(
    model: TwoComponentLatentWeibullModel,
    t0_values: list[float],
    cox_coef: dict[str, float] | None = None,
    X_static: dict[str, float] | None = None,
    X_extended: dict[str, float] | None = None,
) -> pd.DataFrame:
    """RUL at multiple current ages."""
    rows = []
    for t0 in t0_values:
        r = compute_rul(t0, model, cox_coef, X_static, X_extended)
        rows.append(r)
    return pd.DataFrame(rows)


def run(
    df: pd.DataFrame,
    out_dir: Path,
    models: dict[str, TwoComponentLatentWeibullModel],
    cox_coef: dict[str, float] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_grid = list(range(0, 901, 15))  # 0 to 900 days in steps of 15

    # Representative covariate profiles (field median)
    def _field_medians(stratum: str) -> dict[str, float]:
        g = df[df["stratum"] == stratum]
        vals: dict[str, float] = {}
        for col in ["frequency", "motor_load", "water_cut", "glr", "p_bot"]:
            if col in g.columns:
                med = g[col].median()
                if not np.isnan(med):
                    vals[col] = float(med)
        return vals

    # ── RUL profiles for key strata ──────────────────────────────────────────
    plot_strata = ["Vt_sour", "Vt_nonsour", "Ya_nonsour"]
    plot_strata = [s for s in plot_strata if s in models]

    all_profiles: dict[str, pd.DataFrame] = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Vt_sour": "crimson", "Vt_nonsour": "steelblue", "Ya_nonsour": "darkorange"}

    for stratum in plot_strata:
        model = models[stratum]
        X_s = _field_medians(stratum)
        profile = compute_rul_profile(model, t0_grid, cox_coef=cox_coef, X_static=X_s)
        all_profiles[stratum] = profile
        ax.plot(profile["t0"], profile["rul"],
                color=colors.get(stratum, "gray"), lw=2, label=stratum)

    ax.set_xlabel("Current age t₀ (days)")
    ax.set_ylabel("RUL (expected remaining days)")
    ax.set_title("Phase 6 — RUL vs Current Age by Stratum")
    ax.legend()
    ax.set_xlim(0, 600)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_dir / "phase6_rul_profiles.png", dpi=150)
    plt.close(fig)

    # ── Vt sour: contractor comparison at fixed ages ─────────────────────────
    vt_sour_model = models.get("Vt_sour")
    if vt_sour_model is not None:
        contractor_profiles: dict[str, pd.DataFrame] = {}
        contractor_colors = {
            "Шлюмберже": "steelblue",
            "Борец": "darkorange",
            "Новые технологии": "crimson",
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        for contractor, color in contractor_colors.items():
            g_sub = df[(df["stratum"] == "Vt_sour") & (df["contractor"] == contractor)]
            if len(g_sub) < 5:
                continue
            X_s = {}
            for col in ["frequency", "motor_load", "water_cut", "p_bot"]:
                if col in g_sub.columns:
                    med = g_sub[col].median()
                    if not np.isnan(med):
                        X_s[col] = float(med)
            # Add contractor HR from Cox if available
            contractor_coef = cox_coef or {}
            profile = compute_rul_profile(vt_sour_model, t0_grid,
                                          cox_coef=contractor_coef, X_static=X_s)
            contractor_profiles[contractor] = profile
            ax.plot(profile["t0"], profile["rul"], color=color, lw=2, label=contractor)

        ax.set_xlabel("Current age t₀ (days)")
        ax.set_ylabel("RUL (expected remaining days)")
        ax.set_title("Phase 6 — Vt sour RUL by Contractor")
        ax.legend()
        ax.set_xlim(0, 300)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(out_dir / "phase6_vt_sour_contractor_rul.png", dpi=150)
        plt.close(fig)

    # ── Snapshot table at key ages ────────────────────────────────────────────
    snapshot_rows = []
    for stratum, profile in all_profiles.items():
        for t0 in [30, 60, 90, 180, 365]:
            row_match = profile[profile["t0"] == t0]
            if row_match.empty:
                continue
            r = row_match.iloc[0]
            snapshot_rows.append({
                "stratum": stratum,
                "t0_days": t0,
                "rul_days": r["rul"],
                "p50_remaining_days": r.get("p50_remaining_days"),
            })
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df.to_csv(out_dir / "phase6_rul_snapshot.csv", index=False, encoding="utf-8-sig")

    print("\n[Phase 6] RUL snapshot (days) at key ages:")
    pivot = snapshot_df.pivot(index="t0_days", columns="stratum", values="rul_days")
    print(pivot.to_string())

    # ── Success check ─────────────────────────────────────────────────────────
    checks: list[str] = []
    # Check 1: RUL decreasing at ages past 100 days in Ya_nonsour
    if "Ya_nonsour" in all_profiles:
        p = all_profiles["Ya_nonsour"]
        rul_100 = p.loc[p["t0"] == 90, "rul"].values
        rul_200 = p.loc[p["t0"] == 180, "rul"].values
        if len(rul_100) and len(rul_200):
            if float(rul_100[0]) > float(rul_200[0]):
                checks.append("✓ RUL decreasing with age (wear-out regime)")
            else:
                checks.append("✗ RUL not decreasing — check model")

    # Check 2: Vt sour RUL at t0=60d < Vt nonsour
    for t0_check in [60]:
        sour_rul = snapshot_df.loc[(snapshot_df["stratum"]=="Vt_sour") & (snapshot_df["t0_days"]==t0_check), "rul_days"]
        nonsour_rul = snapshot_df.loc[(snapshot_df["stratum"]=="Vt_nonsour") & (snapshot_df["t0_days"]==t0_check), "rul_days"]
        if len(sour_rul) and len(nonsour_rul):
            if float(sour_rul.values[0]) < float(nonsour_rul.values[0]):
                checks.append(f"✓ Vt sour RUL at t₀={t0_check}d < Vt non-sour")
            else:
                checks.append(f"✗ Vt sour RUL at t₀={t0_check}d ≥ Vt non-sour — unexpected")

    print("\n[Phase 6] Success checks:")
    for c in checks:
        print(f"  {c}")

    print(f"\n[Phase 6] Complete.")
    return {"rul_profiles": all_profiles, "snapshot": snapshot_df}
