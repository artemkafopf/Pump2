"""TRF capacity hypothesis test.

Four tests for the idea that pumps have a fixed cumulative Hz-day budget:

  T1  KW test on TRF-at-failure distributions across frequency groups
  T2  Weibull fitted in TRF-space (beta > 1 would mean wear-out in Hz-days)
  T3  Weibull AFT with ln(freq) as stress covariate (coeff ≈ -1 supports capacity)
  T4  Scatter: TRF vs TTF coloured by freq group
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, WeibullFitter
from scipy.stats import kruskal, mannwhitneyu

from .config import FREQ_GROUP_COLORS


def _mw_pair(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ---------------------------------------------------------------
    # Prepare: failures with TRF and duration available
    # ---------------------------------------------------------------
    fails = df[
        (df["event"] == 1)
        & df["total_freq_hz_days"].notna()
        & df["duration"].notna()
        & df["freq_w_mean"].notna()
        & (df["duration"] > 0)
    ].copy()

    fails["trf_at_failure"] = fails["total_freq_hz_days"]
    fails["ln_freq"] = np.log(fails["freq_w_mean"])

    grp_labels = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    grp_short  = {"Low (≤50 Hz)": "Low", "Normal (50–55 Hz)": "Normal", "High (>55 Hz)": "High"}
    colors     = FREQ_GROUP_COLORS

    # ---------------------------------------------------------------
    # T1 — TRF at failure distribution comparison
    # ---------------------------------------------------------------
    groups_trf = {g: fails.loc[fails["freq_group"] == g, "trf_at_failure"] for g in grp_labels}
    kw_groups  = [v for v in groups_trf.values() if len(v) >= 2]
    kw_stat, kw_p = (kruskal(*kw_groups) if len(kw_groups) >= 2 else (np.nan, np.nan))

    trf_summary_rows = []
    for grp in grp_labels:
        s = groups_trf[grp]
        if len(s) == 0:
            continue
        ttf_grp = fails.loc[fails["freq_group"] == grp, "duration"]
        trf_summary_rows.append({
            "freq_group": grp,
            "n_failures": len(s),
            "median_TRF_hz_days": round(s.median(), 0),
            "p25_TRF": round(s.quantile(0.25), 0),
            "p75_TRF": round(s.quantile(0.75), 0),
            "cv_TRF": round(s.std() / s.mean(), 3),
            "median_TTF_days": round(ttf_grp.median(), 1),
            "cv_TTF": round(ttf_grp.std() / ttf_grp.mean(), 3),
        })
    trf_summary = pd.DataFrame(trf_summary_rows)
    trf_summary.to_csv(out / "trf_t1_summary.csv", index=False, encoding="utf-8-sig")
    results["t1_kw_p"] = float(kw_p)
    results["t1_kw_stat"] = float(kw_stat)
    results["t1_trf_summary"] = trf_summary

    # Pairwise Mann-Whitney
    mw_rows = []
    for i, ga in enumerate(grp_labels):
        for gb in grp_labels[i + 1:]:
            s_a = groups_trf[ga]
            s_b = groups_trf[gb]
            _, p = _mw_pair(s_a, s_b)
            mw_rows.append({"group_a": ga, "group_b": gb, "mw_p": round(p, 4) if np.isfinite(p) else np.nan})
    pd.DataFrame(mw_rows).to_csv(out / "trf_t1_mannwhitney.csv", index=False, encoding="utf-8-sig")

    # Box-plot: TRF at failure & TTF side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bp_data_trf = [groups_trf[g].values for g in grp_labels if len(groups_trf[g]) > 0]
    bp_data_ttf = [fails.loc[fails["freq_group"] == g, "duration"].values for g in grp_labels if len(groups_trf[g]) > 0]
    bp_labels   = [grp_short[g] for g in grp_labels if len(groups_trf[g]) > 0]
    bp_colors   = [colors.get(g, "#888") for g in grp_labels if len(groups_trf[g]) > 0]

    for ax, bp_data, ylabel, title_suffix in zip(
        axes,
        [bp_data_trf, bp_data_ttf],
        ["Hz-days", "days"],
        ["TRF at failure", "TTF at failure"],
    ):
        bplot = ax.boxplot(bp_data, patch_artist=True, notch=False, showfliers=False)
        for patch, c in zip(bplot["boxes"], bp_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(bp_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Failures only — {title_suffix}\n(KW p={kw_p:.3f})" if title_suffix == "TRF at failure" else f"Failures only — {title_suffix}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("TRF Capacity Test T1: TRF and TTF distributions at failure by freq group", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "trf_t1_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # T2 — Weibull fit using TRF as time axis (all runs with TRF)
    #       Censored if event == 0; but for censored runs TRF is partial
    # ---------------------------------------------------------------
    # Use ALL rows with TRF and duration > 0
    trf_all = df[df["total_freq_hz_days"].notna() & (df["duration"] > 0) & df["freq_w_mean"].notna()].copy()
    trf_all["trf_at_failure"] = trf_all["total_freq_hz_days"]

    weibull_trf_rows = []
    for grp in grp_labels:
        sub = trf_all[trf_all["freq_group"] == grp]
        if sub["event"].sum() < 10:
            continue
        wf = WeibullFitter()
        try:
            wf.fit(sub["trf_at_failure"], event_observed=sub["event"])
            weibull_trf_rows.append({
                "freq_group": grp,
                "n_runs": len(sub),
                "n_fail": int(sub["event"].sum()),
                "beta_trf": round(float(wf.rho_), 3),
                "eta_trf_hz_days": round(float(wf.lambda_), 0),
                "median_trf_hz_days": round(float(wf.median_survival_time_), 0),
                "beta_95lo": round(float(wf.summary.loc["rho_", "coef lower 95%"]), 3),
                "beta_95hi": round(float(wf.summary.loc["rho_", "coef upper 95%"]), 3),
            })
        except Exception:
            pass

    # Also global and Vt
    for scope, sub_df in [("GLOBAL", trf_all), ("Vt", trf_all[trf_all["field"] == "Vt"])]:
        if sub_df["event"].sum() < 10:
            continue
        wf = WeibullFitter()
        try:
            wf.fit(sub_df["trf_at_failure"], event_observed=sub_df["event"])
            weibull_trf_rows.append({
                "freq_group": scope,
                "n_runs": len(sub_df),
                "n_fail": int(sub_df["event"].sum()),
                "beta_trf": round(float(wf.rho_), 3),
                "eta_trf_hz_days": round(float(wf.lambda_), 0),
                "median_trf_hz_days": round(float(wf.median_survival_time_), 0),
                "beta_95lo": round(float(wf.summary.loc["rho_", "coef lower 95%"]), 3),
                "beta_95hi": round(float(wf.summary.loc["rho_", "coef upper 95%"]), 3),
            })
        except Exception:
            pass

    weibull_trf_df = pd.DataFrame(weibull_trf_rows)
    weibull_trf_df.to_csv(out / "trf_t2_weibull_trf_axis.csv", index=False, encoding="utf-8-sig")
    results["t2_weibull_trf"] = weibull_trf_df

    # Forest plot: beta in TRF space
    if len(weibull_trf_rows) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(weibull_trf_rows) * 0.8)))
        y_pos = list(range(len(weibull_trf_rows)))
        for i, row in enumerate(weibull_trf_rows):
            b  = row["beta_trf"]
            lo = row["beta_95lo"]
            hi = row["beta_95hi"]
            c  = colors.get(row["freq_group"], "#666")
            ax.scatter(b, i, color=c, s=80, zorder=3)
            ax.hlines(i, lo, hi, color=c, linewidth=3, alpha=0.6)
            ax.text(hi + 0.02, i, f"β={b:.3f} [{lo:.3f}–{hi:.3f}]  (η={row['eta_trf_hz_days']:.0f} Hz-d)",
                    va="center", fontsize=8)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="β=1 (exponential)")
        ax.axvline(0.8, color="gray",  linestyle=":",  linewidth=1, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["freq_group"] for r in weibull_trf_rows], fontsize=8)
        ax.set_xlabel("Weibull shape β  (TRF axis — Hz-days)")
        ax.set_title("TRF Capacity Test T2: Weibull β when time = cumulative Hz-days\nβ>1 → wear-out in Hz-day space (supports capacity hypothesis)")
        ax.legend(fontsize=7)
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "trf_t2_weibull_forest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------
    # T3 — Weibull AFT: TTF ~ ln(freq) + field_Vt
    #       If TRF capacity is real: coeff_ln_freq ≈ -1
    # ---------------------------------------------------------------
    aft_df = df[
        df["freq_w_mean"].notna()
        & df["duration"].notna()
        & (df["duration"] > 0)
        & df["freq_group"].isin(grp_labels)
    ].copy()
    aft_df["ln_freq"] = np.log(aft_df["freq_w_mean"])
    aft_df["is_vt"] = (aft_df["field"] == "Vt").astype(float)

    aft_rows = []
    for label, covs in [
        ("AFT_univariate",     ["ln_freq"]),
        ("AFT_plus_field",     ["ln_freq", "is_vt"]),
    ]:
        sub = aft_df[covs + ["duration", "event"]].dropna()
        if sub["event"].sum() < 20:
            continue
        try:
            waft = WeibullAFTFitter()
            waft.fit(sub, duration_col="duration", event_col="event")
            summ = waft.summary
            row = {"model": label, "n_fail": int(sub["event"].sum())}
            for cov in covs:
                # AFT summary has MultiIndex (param, covariate)
                try:
                    s_row = summ.xs(cov, level="covariate").loc["lambda_"]
                    row[f"coef_{cov}"]   = round(float(s_row["coef"]), 4)
                    row[f"p_{cov}"]      = round(float(s_row["p"]), 4)
                    row[f"ci_lo_{cov}"]  = round(float(s_row["coef lower 95%"]), 4)
                    row[f"ci_hi_{cov}"]  = round(float(s_row["coef upper 95%"]), 4)
                except (KeyError, TypeError):
                    pass
            rho_row = summ.xs("Intercept", level="covariate").loc["rho_"]
            row["rho"] = round(float(np.exp(rho_row["coef"])), 3)
            aft_rows.append(row)
        except Exception as exc:
            aft_rows.append({"model": label, "error": str(exc)})

    aft_df_out = pd.DataFrame(aft_rows)
    aft_df_out.to_csv(out / "trf_t3_aft.csv", index=False, encoding="utf-8-sig")
    results["t3_aft"] = aft_df_out

    # ---------------------------------------------------------------
    # T4 — Scatter TRF vs TTF coloured by freq group
    # ---------------------------------------------------------------
    plot_df = fails.sample(min(len(fails), 1200), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (scope, mask_col) in zip(axes, [("Global", None), ("Vt only", "is_vt")]):
        sub = plot_df if mask_col is None else plot_df[plot_df[mask_col]]
        for grp in grp_labels:
            g = sub[sub["freq_group"] == grp]
            if len(g) == 0:
                continue
            ax.scatter(
                g["duration"], g["trf_at_failure"],
                c=colors.get(grp, "#888"), alpha=0.35, s=14,
                label=f"{grp_short[grp]} (n={len(g)})",
            )
        # TRF capacity prediction lines: TRF = freq × TTF → lines at freq = 45, 53, 59 Hz
        t_range = np.linspace(1, plot_df["duration"].quantile(0.95), 200)
        for freq_ref, c_ref, lab in [(45.2, "#5DBB63", "45 Hz"), (53.2, "#3060AC", "53 Hz"), (59.1, "#D65F5F", "59 Hz")]:
            ax.plot(t_range, freq_ref * t_range, linestyle="--", linewidth=1, color=c_ref, alpha=0.5, label=f"TRF=freq×TTF ({lab})")

        ax.set_xlabel("TTF (days)")
        ax.set_ylabel("TRF at failure (Hz-days)")
        ax.set_title(f"T4 Scatter ({scope})\nConstant TRF capacity = along dashed lines")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("TRF Capacity Test T4: if capacity is fixed, each group lies on its own dashed line", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "trf_t4_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Eta scaling check: TRF capacity → η_low / η_high ≈ freq_low / freq_high
    # (Weibull fitted in TTF-space; if η ∝ 1/freq then capacity is constant)
    # ---------------------------------------------------------------
    eta_rows = []
    median_freq_by_grp = df.groupby("freq_group", observed=True)["freq_w_mean"].median()
    for grp in grp_labels:
        sub = df[df["freq_group"] == grp]
        if sub["event"].sum() < 10:
            continue
        wf = WeibullFitter()
        try:
            wf.fit(sub["duration"], event_observed=sub["event"])
            mf = float(median_freq_by_grp.get(grp, np.nan))
            eta_rows.append({
                "freq_group": grp,
                "median_freq_hz": round(mf, 1),
                "eta_ttf_days": round(float(wf.lambda_), 1),
                "beta_ttf": round(float(wf.rho_), 3),
                "predicted_eta_if_capacity_model": round(
                    float(eta_rows[0]["eta_ttf_days"] * eta_rows[0]["median_freq_hz"] / mf), 1
                ) if len(eta_rows) > 0 else np.nan,
            })
        except Exception:
            pass
    if eta_rows:
        # Fill in predicted_eta_if_capacity_model after first row exists
        eta_df = pd.DataFrame(eta_rows)
        ref_eta  = eta_df.iloc[0]["eta_ttf_days"]
        ref_freq = eta_df.iloc[0]["median_freq_hz"]
        eta_df["predicted_eta_if_capacity"] = (ref_eta * ref_freq / eta_df["median_freq_hz"]).round(1)
        eta_df["ratio_actual_predicted"] = (eta_df["eta_ttf_days"] / eta_df["predicted_eta_if_capacity"]).round(3)
        eta_df.to_csv(out / "trf_t5_eta_scaling.csv", index=False, encoding="utf-8-sig")
        results["t5_eta_scaling"] = eta_df

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRF CAPACITY HYPOTHESIS TEST")
    print("=" * 60)

    print("\n--- T1: TRF at failure by frequency group ---")
    print(trf_summary.to_string(index=False))
    print(f"\nKruskal-Wallis (TRF at failure): stat={kw_stat:.2f}, p={kw_p:.4f}")
    print("Interpretation: p<0.05 means TRF at failure differs across groups → NOT a fixed capacity")

    print("\n--- T2: Weibull β in TRF (Hz-day) space ---")
    if not weibull_trf_df.empty:
        print(weibull_trf_df[["freq_group", "n_fail", "beta_trf", "beta_95lo", "beta_95hi", "eta_trf_hz_days"]].to_string(index=False))
    print("Interpretation: β>1 in Hz-day space would mean wear-out in cumulative frequency")

    print("\n--- T3: Weibull AFT — coefficient on ln(freq) ---")
    if not aft_df_out.empty:
        show_cols = [c for c in ["model", "n_fail", "coef_ln_freq", "ci_lo_ln_freq", "ci_hi_ln_freq", "p_ln_freq", "rho"] if c in aft_df_out.columns]
        print(aft_df_out[show_cols].to_string(index=False))
    print("Interpretation: coef_ln_freq ≈ -1 supports TRF capacity (lower freq → proportionally longer TTF)")
    print("               coef_ln_freq not significant → freq has no acceleration effect")

    if "t5_eta_scaling" in results:
        print("\n--- T5: Weibull η (scale) vs predicted if TRF capacity held ---")
        print(results["t5_eta_scaling"][["freq_group", "median_freq_hz", "eta_ttf_days", "predicted_eta_if_capacity", "ratio_actual_predicted"]].to_string(index=False))
        print("Interpretation: ratio≈1 → capacity model holds; ratio>>1 or <<1 → rejects it")

    return results
