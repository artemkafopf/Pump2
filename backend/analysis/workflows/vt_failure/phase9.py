"""Phase 9 — Vt Deep-Dive: environment comparison + RMST mix decomposition."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time

from .config import (
    CATEGORY_COLORS, FAILURE_CATEGORIES, FREQ_GROUP_COLORS,
    MIN_FAILURES_KM, MIN_FAILURES_WEIBULL, VT_FIELD,
)


def _rmst(T: np.ndarray, E: np.ndarray, tau: float) -> float:
    """Restricted mean survival time up to horizon tau."""
    if E.sum() < 5:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    try:
        return float(restricted_mean_survival_time(kmf, t=tau))
    except Exception:
        return np.nan


def _compute_rmst_tau(df: pd.DataFrame) -> float:
    """Set tau = 75th percentile of global observed failure times."""
    fail_times = df.loc[df["event"] == 1, "duration"].dropna()
    if fail_times.empty:
        return 365.0
    return float(fail_times.quantile(0.75))


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    vt  = df[df["is_vt"]].copy()
    gl  = df.copy()
    non_vt = df[~df["is_vt"]].copy()

    # ------------------------------------------------------------------
    # 9a — Environment comparison table: Vt vs Global
    # ------------------------------------------------------------------
    def _env_row(sub: pd.DataFrame, label: str) -> dict:
        n = len(sub)
        n_ev = int(sub["event"].sum())
        return {
            "scope": label,
            "n_runs": n,
            "n_failures": n_ev,
            "median_ttf_d": sub.loc[sub["event"] == 1, "duration"].median(),
            "avg_freq_hz": sub["freq_w_mean"].mean(),
            "pct_above_55hz": (sub["freq_w_mean"] > 55).mean() * 100,
            "median_h2s_mg_l": sub["h2s_proxy_mg_l"].median(),
            "pct_acidic": (sub["h2s_label"] == "Кислый").mean() * 100,
            "avg_glf": sub["avg_glf"].mean(),
            "avg_kpod": sub["avg_kpod"].mean(),
            "median_ca_kg": sub["cum_calcium_load_kg"].median() if "cum_calcium_load_kg" in sub.columns else np.nan,
            "median_cl_kg": sub["cum_chloride_load_kg"].median() if "cum_chloride_load_kg" in sub.columns else np.nan,
            "median_so4_kg": sub["cum_sulfate_load_kg"].median() if "cum_sulfate_load_kg" in sub.columns else np.nan,
            "median_salt_kg": sub["cum_salt_load_kg"].median() if "cum_salt_load_kg" in sub.columns else np.nan,
            "median_gypsum": sub["cum_gypsum_scale_proxy"].median() if "cum_gypsum_scale_proxy" in sub.columns else np.nan,
            "pct_tele_coverage": (sub["n_freq_valid_days"] >= 7).mean() * 100,
        }

    env_table = pd.DataFrame([_env_row(vt, "Vt"), _env_row(gl, "GLOBAL"), _env_row(non_vt, "non-Vt")])
    env_table = env_table.round(2)
    env_table.to_csv(out / "p9a_environment_comparison.csv", index=False, encoding="utf-8-sig")
    results["env_table"] = env_table
    print("\n" + "=" * 60 + "\nPHASE 9 — VT DEEP-DIVE\n" + "=" * 60)
    print("9a — Environment comparison:")
    print(env_table.to_string(index=False))

    # ------------------------------------------------------------------
    # 9b — RMST category mix decomposition
    # ------------------------------------------------------------------
    tau = _compute_rmst_tau(df)
    results["rmst_tau"] = tau
    print(f"\n9b — RMST horizon τ = {tau:.0f} days (75th pctile of global failure times)")

    # Global category weights (among failures)
    gl_fail = gl[gl["event"] == 1]
    gl_cat_counts = gl_fail["failure_category"].value_counts()
    gl_cat_weights = (gl_cat_counts / gl_cat_counts.sum()).to_dict()

    # Vt category weights
    vt_fail = vt[vt["event"] == 1]
    vt_cat_counts = vt_fail["failure_category"].value_counts()
    vt_cat_weights = (vt_cat_counts / vt_cat_counts.sum()).to_dict() if len(vt_fail) > 0 else {}

    rmst_rows = []
    for cat in FAILURE_CATEGORIES:
        # RMST within Vt for this category (all runs — not just failures)
        g_vt = vt[vt["failure_category"] == cat].dropna(subset=["duration"])
        T_vt = g_vt["duration"].values.astype(float)
        E_vt = g_vt["event"].values.astype(float)
        rmst_vt = _rmst(T_vt, E_vt, tau)

        # RMST within Global for this category
        g_gl = gl[gl["failure_category"] == cat].dropna(subset=["duration"])
        T_gl = g_gl["duration"].values.astype(float)
        E_gl = g_gl["event"].values.astype(float)
        rmst_gl = _rmst(T_gl, E_gl, tau)

        rmst_rows.append({
            "category": cat,
            "n_vt_runs": len(g_vt),
            "n_gl_runs": len(g_gl),
            "vt_cat_weight": vt_cat_weights.get(cat, 0.0),
            "gl_cat_weight": gl_cat_weights.get(cat, 0.0),
            "rmst_vt": rmst_vt,
            "rmst_gl": rmst_gl,
        })

    rmst_df = pd.DataFrame(rmst_rows)

    # Vt observed RMST overall
    vt_T = vt["duration"].dropna().values.astype(float)
    vt_E = vt.loc[vt["duration"].notna(), "event"].values.astype(float)
    rmst_vt_overall = _rmst(vt_T, vt_E, tau)

    gl_T = gl["duration"].dropna().values.astype(float)
    gl_E = gl.loc[gl["duration"].notna(), "event"].values.astype(float)
    rmst_gl_overall = _rmst(gl_T, gl_E, tau)

    # Mix effect: substitute global weights into Vt within-category RMST
    rmst_mix_adjusted = sum(
        gl_cat_weights.get(r["category"], 0) * r["rmst_vt"]
        for _, r in rmst_df.iterrows()
        if np.isfinite(r["rmst_vt"])
    )
    # Within-category effect: substitute global within-category RMST with Vt weights
    rmst_withincategory_adjusted = sum(
        vt_cat_weights.get(r["category"], 0) * r["rmst_gl"]
        for _, r in rmst_df.iterrows()
        if np.isfinite(r["rmst_gl"])
    )

    decomp = {
        "tau_days": tau,
        "rmst_global_overall": rmst_gl_overall,
        "rmst_vt_overall": rmst_vt_overall,
        "rmst_vt_minus_global": rmst_vt_overall - rmst_gl_overall if np.isfinite(rmst_vt_overall) and np.isfinite(rmst_gl_overall) else np.nan,
        "rmst_mix_adjusted": rmst_mix_adjusted,
        "mix_effect_days": rmst_mix_adjusted - rmst_vt_overall if np.isfinite(rmst_mix_adjusted) else np.nan,
        "rmst_withincategory_adjusted": rmst_withincategory_adjusted,
        "withincategory_effect_days": rmst_withincategory_adjusted - rmst_gl_overall if np.isfinite(rmst_withincategory_adjusted) else np.nan,
    }
    rmst_df.to_csv(out / "p9b_rmst_by_category.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([decomp]).to_csv(out / "p9b_rmst_decomposition.csv", index=False, encoding="utf-8-sig")
    results["rmst_decomp"] = decomp
    results["rmst_by_category"] = rmst_df

    print(f"\n  RMST global overall: {rmst_gl_overall:.1f}d")
    print(f"  RMST Vt overall:    {rmst_vt_overall:.1f}d")
    print(f"  Gap (Vt - Global):  {rmst_vt_overall - rmst_gl_overall:.1f}d")
    if np.isfinite(decomp["mix_effect_days"]):
        print(f"  Mix effect:          {decomp['mix_effect_days']:.1f}d  "
              f"({'Vt has more fast-failing categories' if decomp['mix_effect_days'] < 0 else 'Vt has less fast-failing categories'})")
    if np.isfinite(decomp["withincategory_effect_days"]):
        print(f"  Within-cat effect:   {decomp['withincategory_effect_days']:.1f}d  "
              f"({'Vt fails faster within categories' if decomp['withincategory_effect_days'] < 0 else 'Global fails faster within categories'})")

    # RMST figure
    valid_cats = rmst_df[(rmst_df["rmst_vt"].notna()) | (rmst_df["rmst_gl"].notna())]
    if not valid_cats.empty:
        x = np.arange(len(valid_cats))
        fig, ax = plt.subplots(figsize=(11, 5))
        w = 0.35
        vt_vals = [v if np.isfinite(v) else 0 for v in valid_cats["rmst_vt"].values]
        gl_vals = [v if np.isfinite(v) else 0 for v in valid_cats["rmst_gl"].values]
        ax.bar(x - w/2, vt_vals, w, label="Vt", color="#D65F5F", alpha=0.85)
        ax.bar(x + w/2, gl_vals, w, label="Global", color="#4878CF", alpha=0.75)
        ax.axhline(rmst_vt_overall, color="#D65F5F", linestyle="--", linewidth=1, label=f"Vt overall RMST={rmst_vt_overall:.0f}d")
        ax.axhline(rmst_gl_overall, color="#4878CF", linestyle="--", linewidth=1, label=f"Global overall RMST={rmst_gl_overall:.0f}d")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_cats["category"].values, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"RMST (days, τ={tau:.0f}d)")
        ax.set_title(f"Phase 9b — RMST by failure category: Vt vs Global\n(τ = {tau:.0f}d = 75th pctile global failure times)")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "p9b_rmst_by_category.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9c — Within-Vt frequency effect
    # ------------------------------------------------------------------
    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    vt_freq = vt[vt["freq_group"].isin(grp_order)]
    vt_grp_counts = vt_freq.groupby("freq_group", observed=True).agg(
        n_runs=("row_id", "count"), n_fail=("event", "sum")
    ).reset_index()
    print("\n9c — Within-Vt frequency group counts:")
    print(vt_grp_counts.to_string(index=False))

    if vt_freq["event"].sum() >= MIN_FAILURES_KM:
        fig, ax = plt.subplots(figsize=(9, 6))
        for grp, color in FREQ_GROUP_COLORS.items():
            sub = vt_freq[vt_freq["freq_group"] == grp].dropna(subset=["duration"])
            T = sub["duration"].values.astype(float)
            E = sub["event"].values.astype(float)
            if E.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label=f"{grp} (n={len(T)}, ev={int(E.sum())})")
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2.0, ci_show=True, ci_alpha=0.12)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("S(t)")
        ax.set_title(f"Phase 9c — Within-Vt: KM by frequency group")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "p9c_km_vt_by_freq.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9d — H2S interaction within Vt
    # ------------------------------------------------------------------
    h2s_rows = []
    for h2s_class in ["Кислый", "Некислый"]:
        g = vt[vt["h2s_label"] == h2s_class]
        n = len(g)
        n_ev = int(g["event"].sum())
        if n == 0:
            continue
        eligible_90 = g[(g["duration"] >= 90) | (g["event"] == 1)]
        n_inf = int(((eligible_90["event"] == 1) & (eligible_90["duration"] < 90)).sum())
        h2s_rows.append({
            "h2s_class": h2s_class,
            "n_runs": n,
            "n_failures": n_ev,
            "fail_rate_pct": n_ev / n * 100,
            "infant_rate_90d_pct": n_inf / len(eligible_90) * 100 if len(eligible_90) > 0 else np.nan,
            "median_ttf_d": g.loc[g["event"] == 1, "duration"].median(),
        })
    h2s_interaction = pd.DataFrame(h2s_rows)
    h2s_interaction.to_csv(out / "p9d_h2s_interaction_vt.csv", index=False, encoding="utf-8-sig")
    results["h2s_interaction_vt"] = h2s_interaction
    print("\n9d — H2S interaction within Vt:")
    print(h2s_interaction.to_string(index=False))

    if len(h2s_rows) >= 2:
        fig, ax = plt.subplots(figsize=(9, 6))
        h2s_colors = {"Кислый": "#d62728", "Некислый": "#2ca02c"}
        for _, row in h2s_interaction.iterrows():
            hc = row["h2s_class"]
            sub = vt[vt["h2s_label"] == hc].dropna(subset=["duration"])
            T = sub["duration"].values.astype(float)
            E = sub["event"].values.astype(float)
            if E.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label=f"{hc} (n={len(T)}, ev={int(E.sum())})")
            kmf.plot_survival_function(ax=ax, color=h2s_colors.get(hc, "#888"),
                                       linewidth=2.0, ci_show=True, ci_alpha=0.12)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("S(t)")
        ax.set_title("Phase 9d — Within-Vt: KM by H2S class (Кислый vs Некислый)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "p9d_km_vt_by_h2s.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9e — Hypothesis verdict table (auto-filled from data)
    # ------------------------------------------------------------------
    verdict_rows = []
    # H1: Vt has more fast-failing categories?
    vt_fast_cats = sum(vt_cat_weights.get(c, 0) for c in ["КЛ (R-0)", "ПЭД (R-0)", "Слом вала"])
    gl_fast_cats = sum(gl_cat_weights.get(c, 0) for c in ["КЛ (R-0)", "ПЭД (R-0)", "Слом вала"])
    verdict_rows.append({
        "hypothesis": "Vt has more fast-failing categories",
        "key_metric": f"Vt fast-cat weight={vt_fast_cats:.0%}, Global={gl_fast_cats:.0%}",
        "direction": "SUPPORTED" if vt_fast_cats > gl_fast_cats else "NOT SUPPORTED",
    })
    # H2: Vt has harsher H2S?
    vt_h2s = vt["h2s_proxy_mg_l"].median()
    gl_h2s = df["h2s_proxy_mg_l"].median()
    verdict_rows.append({
        "hypothesis": "Vt has harsher H2S environment",
        "key_metric": f"Vt median H2S={vt_h2s:.1f} mg/L, Global={gl_h2s:.1f} mg/L",
        "direction": "SUPPORTED" if vt_h2s > gl_h2s * 1.5 else "PARTIAL" if vt_h2s > gl_h2s else "NOT SUPPORTED",
    })
    # H3: RMST gap
    gap = decomp.get("rmst_vt_minus_global", np.nan)
    verdict_rows.append({
        "hypothesis": "Vt has shorter survival overall",
        "key_metric": f"RMST gap={gap:.1f}d (Vt - Global)" if np.isfinite(gap) else "N/A",
        "direction": "SUPPORTED" if np.isfinite(gap) and gap < -10 else "MARGINAL" if np.isfinite(gap) and gap < 0 else "NOT SUPPORTED",
    })
    # H4: Within-category worse in Vt?
    wc_eff = decomp.get("withincategory_effect_days", np.nan)
    verdict_rows.append({
        "hypothesis": "Vt fails faster within each category",
        "key_metric": f"Within-cat effect={wc_eff:.1f}d" if np.isfinite(wc_eff) else "N/A",
        "direction": "SUPPORTED" if np.isfinite(wc_eff) and wc_eff < 0 else "NOT SUPPORTED",
    })
    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(out / "p9e_hypothesis_verdicts.csv", index=False, encoding="utf-8-sig")
    results["verdicts"] = verdict_df
    print("\n9e — Hypothesis verdicts:")
    print(verdict_df.to_string(index=False))

    return results
