"""Phase 3 — Mixture diagnostics: bootstrap CI, GOF, component assignment.

For each fitted mixture:
  1. Bootstrap CI (100 resamples) on (β₁, β₂, w₁)
  2. Goodness-of-fit: RMSE between mixture and KM, compared to single Weibull
  3. Component assignment r_j1 per pump (E-step posterior)
  4. Cross-validate r_j1 > 0.5 against is_early_failure (Ранний/Преждевременный)
     and against is_h2s_cause (for Vt sour)
  5. Per-stratum w₁ comparison — operational interpretability
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_posterior_given_failure,
    latent_posterior_given_survival,
    latent_survival,
)
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.models.survival.weibull_model import fit_basic_weibull


NUM_BOOTSTRAP = 50


def _km_frame(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events)
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]
    et = kmf.event_table.reset_index()
    et = et[["event_at", "at_risk"]].rename(columns={"event_at": "time", "at_risk": "n_risk"})
    merged = sf.merge(et, on="time", how="left")
    merged["n_risk"] = merged["n_risk"].ffill().fillna(0)
    return merged[merged["time"] > 0].reset_index(drop=True)


def _bootstrap_ci(
    durations: np.ndarray,
    events: np.ndarray,
    model: TwoComponentLatentWeibullModel,
    n_boot: int = NUM_BOOTSTRAP,
    fix_beta1: float | None = None,
    fix_beta2: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Non-parametric bootstrap CI on (β₁, β₂, w₁) via raw-data resampling + EM.

    For two-stage strata, pass fix_beta1/fix_beta2 to keep the bootstrap
    estimator consistent with the constrained point estimate.
    """
    rng = np.random.default_rng(42)
    n = len(durations)
    beta1_boots, beta2_boots, w1_boots = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t_b = durations[idx]
        e_b = events[idx]
        try:
            res = fit_latent_weibull_em(
                t_b, e_b,
                initial_model=model,
                num_starts=3,
                max_iter=150,
                fix_beta1=fix_beta1,
                fix_beta2=fix_beta2,
            )
            beta1_boots.append(res.model.component_1.beta)
            beta2_boots.append(res.model.component_2.beta)
            w1_boots.append(res.model.weight_1)
        except Exception:
            pass

    def ci(arr: list[float]) -> tuple[float, float]:
        if len(arr) < 10:
            return (float("nan"), float("nan"))
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "beta1_ci": ci(beta1_boots),
        "beta2_ci": ci(beta2_boots),
        "w1_ci": ci(w1_boots),
    }


def _compute_component_assignment(
    g: pd.DataFrame,
    model: TwoComponentLatentWeibullModel,
) -> pd.DataFrame:
    """Compute r_j1 (posterior probability of belonging to C1) for each pump.

    For failed pumps: use posterior_given_failure at their observed TTE.
    For censored pumps: use posterior_given_survival at their observed TTE.
    """
    g = g.copy()
    ttes = g["tte"].to_numpy(dtype=float)
    events = g["event"].to_numpy(dtype=int)

    r1_fail = np.asarray(latent_posterior_given_failure(ttes, model)[0], dtype=float)
    r1_surv = np.asarray(latent_posterior_given_survival(ttes, model)[0], dtype=float)

    g["r_j1"] = np.where(events == 1, r1_fail, r1_surv)
    g["component_assignment"] = np.where(g["r_j1"] > 0.5, "C1_early", "C2_wearout")
    return g


def _single_weibull_rmse(km: pd.DataFrame, durations: np.ndarray, events: np.ndarray) -> float:
    """RMSE of best-fit single Weibull against KM survival curve."""
    try:
        w = fit_basic_weibull(durations, events)
        t = km["time"].to_numpy(dtype=float)
        s_km = km["survival"].to_numpy(dtype=float)
        s_wb = np.exp(-(t / w["eta"]) ** w["beta"])
        return float(np.sqrt(np.mean((s_wb - s_km) ** 2)))
    except Exception:
        return float("nan")


def run(
    df: pd.DataFrame,
    out_dir: Path,
    models: dict[str, TwoComponentLatentWeibullModel],
    kmf_cache: dict[str, "KaplanMeierFitter"],  # noqa: F821
    *,
    global_beta1: float | None = None,
    global_beta2: float | None = None,
    two_stage_strata: list[str] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    two_stage_set = set(two_stage_strata or [])

    diag_rows = []
    assign_frames: dict[str, pd.DataFrame] = {}

    for stratum, model in models.items():
        g = df[df["stratum"] == stratum].copy()
        durations = g["tte"].to_numpy(dtype=float)
        events = g["event"].to_numpy(dtype=int)
        km = _km_frame(durations, events)

        # Mixture RMSE vs KM
        t_km = km["time"].to_numpy(dtype=float)
        s_km = km["survival"].to_numpy(dtype=float)
        s_mix = np.asarray(latent_survival(t_km, model), dtype=float)
        mix_rmse = float(np.sqrt(np.mean((s_mix - s_km) ** 2)))

        sw_rmse = _single_weibull_rmse(km, durations, events)

        # Delta RMSE: positive means mixture is better
        delta_rmse = sw_rmse - mix_rmse

        # Bootstrap CI — use fixed shapes for two-stage strata so CI is consistent
        # with the constrained point estimate
        print(f"  [Phase 3] {stratum}: bootstrapping CI...")
        is_two_stage = stratum in two_stage_set
        ci_dict = _bootstrap_ci(
            durations, events, model, n_boot=NUM_BOOTSTRAP,
            fix_beta1=global_beta1 if is_two_stage else None,
            fix_beta2=global_beta2 if is_two_stage else None,
        )

        # Component assignment
        g_assigned = _compute_component_assignment(g, model)
        assign_frames[stratum] = g_assigned

        # Cross-tab r_j1 vs early failure sign (0=mature, 1=early, -1=unknown)
        early_known = g_assigned[g_assigned["is_early_failure"] >= 0].copy()
        corr_early = float("nan")
        if len(early_known) > 5:
            corr_early = float(early_known["r_j1"].corr(early_known["is_early_failure"]))

        # H2S cause correlation (Vt sour only)
        corr_h2s = float("nan")
        if "sour" in stratum:
            h2s_known = g_assigned[g_assigned["event"] == 1]
            if len(h2s_known) > 5:
                corr_h2s = float(h2s_known["r_j1"].corr(h2s_known["is_h2s_cause"]))

        # Fraction correctly classified (r_j1 > 0.5 for early, < 0.5 for mature)
        correct_frac = float("nan")
        if len(early_known) > 5:
            pred_early = (g_assigned[g_assigned["is_early_failure"] >= 0]["r_j1"] > 0.5).astype(int)
            true_early = g_assigned[g_assigned["is_early_failure"] >= 0]["is_early_failure"]
            correct_frac = float((pred_early == true_early).mean())

        diag_rows.append({
            "stratum": stratum,
            "n_failures": int(events.sum()),
            "beta1": round(model.component_1.beta, 4),
            "beta1_lo95": round(ci_dict["beta1_ci"][0], 4),
            "beta1_hi95": round(ci_dict["beta1_ci"][1], 4),
            "beta2": round(model.component_2.beta, 4),
            "beta2_lo95": round(ci_dict["beta2_ci"][0], 4),
            "beta2_hi95": round(ci_dict["beta2_ci"][1], 4),
            "w1": round(model.weight_1, 4),
            "w1_lo95": round(ci_dict["w1_ci"][0], 4),
            "w1_hi95": round(ci_dict["w1_ci"][1], 4),
            "mix_rmse": round(mix_rmse, 6),
            "single_weibull_rmse": round(sw_rmse, 6),
            "delta_rmse_improvement": round(delta_rmse, 6),
            "corr_rj1_early_sign": round(corr_early, 4) if not np.isnan(corr_early) else None,
            "corr_rj1_h2s_cause": round(corr_h2s, 4) if not np.isnan(corr_h2s) else None,
            "classification_accuracy": round(correct_frac, 4) if not np.isnan(correct_frac) else None,
        })

        print(
            f"  [Phase 3] {stratum}: "
            f"β₂={model.component_2.beta:.3f} [{ci_dict['beta2_ci'][0]:.3f}–{ci_dict['beta2_ci'][1]:.3f}]  "
            f"w₁={model.weight_1:.3f} [{ci_dict['w1_ci'][0]:.3f}–{ci_dict['w1_ci'][1]:.3f}]  "
            f"mix_RMSE={mix_rmse:.5f}  sw_RMSE={sw_rmse:.5f}  ΔΔ={delta_rmse:+.5f}"
        )

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "phase3_diagnostics.csv", index=False, encoding="utf-8-sig")

    # Save component assignments for each stratum
    for stratum, af in assign_frames.items():
        safe_name = stratum.replace(" ", "_").replace("/", "_")
        cols = ["well_key", "pad_key", "tte", "event", "r_j1", "component_assignment",
                "is_early_failure", "is_h2s_cause", "contractor"]
        out_cols = [c for c in cols if c in af.columns]
        af[out_cols].to_csv(
            out_dir / f"component_assignment_{safe_name}.csv",
            index=False, encoding="utf-8-sig",
        )

    # ── Component assignment plot: Vt sour ────────────────────────────────────
    if "Vt_sour" in assign_frames:
        af = assign_frames["Vt_sour"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # r_j1 distribution by is_early_failure
        for is_early, label, color in [(1, "Early (Ранний/Преждев.)", "crimson"),
                                        (0, "Mature (Многосуточный)", "steelblue")]:
            sub = af[af["is_early_failure"] == is_early]["r_j1"]
            if not sub.empty:
                axes[0].hist(sub, bins=20, alpha=0.6, label=label, color=color)
        axes[0].axvline(0.5, ls="--", color="k", lw=1)
        axes[0].set_xlabel("r_j1 (P[C1 early | pump])")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Vt sour — r_j1 by failure sign")
        axes[0].legend(fontsize=7)

        # r_j1 distribution by is_h2s_cause
        for is_h2s, label, color in [(1, "H2S cause confirmed", "crimson"),
                                      (0, "Other cause", "steelblue")]:
            sub = af[(af["event"] == 1) & (af["is_h2s_cause"] == is_h2s)]["r_j1"]
            if not sub.empty:
                axes[1].hist(sub, bins=20, alpha=0.6, label=label, color=color)
        axes[1].axvline(0.5, ls="--", color="k", lw=1)
        axes[1].set_xlabel("r_j1 (P[C1 early | pump])")
        axes[1].set_title("Vt sour — r_j1 by H2S cause attribution")
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(out_dir / "phase3_vt_sour_component_assignment.png", dpi=150)
        plt.close(fig)

    # ── w₁ comparison across strata ──────────────────────────────────────────
    if not diag_df.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(diag_df) * 0.8), 4))
        strata_labels = diag_df["stratum"].tolist()
        w1_vals = diag_df["w1"].tolist()
        lo = diag_df["w1_lo95"].tolist()
        hi = diag_df["w1_hi95"].tolist()
        colors = ["crimson" if "sour" in s else "steelblue" for s in strata_labels]
        bars = ax.bar(strata_labels, w1_vals, color=colors, alpha=0.7)
        # CI error bars
        err_lo = [max(0.0, w - l) for w, l in zip(w1_vals, lo)]
        err_hi = [max(0.0, h - w) for w, h in zip(w1_vals, hi)]
        ax.errorbar(strata_labels, w1_vals,
                    yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=4, lw=1.5)
        ax.set_ylabel("w₁ (early-failure fraction)")
        ax.set_title("Phase 3 — Early-failure fraction w₁ by stratum")
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / "phase3_w1_comparison.png", dpi=150)
        plt.close(fig)

    print(f"\n[Phase 3] Complete. {len(diag_df)} strata diagnosed.")
    return {"diagnostics": diag_df, "component_assignments": assign_frames}
