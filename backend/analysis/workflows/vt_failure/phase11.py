"""Phase 11 — Sensitivity Checks."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from .config import (
    FREQ_GROUP_COLORS, INFANT_THRESHOLDS, MAJOR_FIELDS, MIN_FAILURES_KM,
)


def _km_median(T: np.ndarray, E: np.ndarray) -> float:
    if E.sum() < 5:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    return float(kmf.median_survival_time_)


def _logrank_p(T1, E1, T2, E2) -> float:
    if E1.sum() < 5 or E2.sum() < 5:
        return np.nan
    try:
        return float(logrank_test(T1, T2, E1, E2).p_value)
    except Exception:
        return np.nan


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]

    # ------------------------------------------------------------------
    # S1 — TTF definition: calendar run_days vs ttf_true_best_days
    # ------------------------------------------------------------------
    s1_rows = []
    for grp in grp_order:
        sub = df[df["freq_group"] == grp]
        for col_label, col in [("calendar_TTF", "run_days"), ("true_TTF", "duration")]:
            g = sub.dropna(subset=[col])
            T = g[col].values.astype(float)
            E = g["event"].values.astype(float)
            s1_rows.append({
                "ttf_type": col_label,
                "freq_group": grp,
                "n_fail": int(E.sum()),
                "median_ttf": _km_median(T, E),
            })

    # High vs Normal logrank with each TTF definition
    for col_label, col in [("calendar", "run_days"), ("true", "duration")]:
        sub_n = df[(df["freq_group"] == "Normal (50–55 Hz)")].dropna(subset=[col])
        sub_h = df[(df["freq_group"] == "High (>55 Hz)")].dropna(subset=[col])
        p = _logrank_p(
            sub_n[col].values.astype(float), sub_n["event"].values.astype(float),
            sub_h[col].values.astype(float), sub_h["event"].values.astype(float),
        )
        s1_rows.append({"ttf_type": col_label, "freq_group": "High vs Normal logrank p", "n_fail": np.nan, "median_ttf": p})

    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv(out / "p11_s1_ttf_definition.csv", index=False, encoding="utf-8-sig")
    results["s1_ttf_definition"] = s1_df

    # ------------------------------------------------------------------
    # S2 — Frequency band boundary sensitivity (±5 Hz)
    # ------------------------------------------------------------------
    s2_rows = []
    for freq_high_min in [50.0, 52.5, 55.0, 57.5, 60.0]:
        # Redefine groups with this threshold
        df2 = df.copy()
        df2["freq_group_alt"] = pd.cut(
            df2["freq_w_mean"],
            bins=[-np.inf, freq_high_min - 5, freq_high_min, np.inf],
            labels=["Low", "Normal", "High"],
            right=True,
        ).astype("string").fillna("<no freq>")

        sub_n = df2[(df2["freq_group_alt"] == "Normal")].dropna(subset=["duration"])
        sub_h = df2[(df2["freq_group_alt"] == "High")].dropna(subset=["duration"])
        p = _logrank_p(
            sub_n["duration"].values.astype(float), sub_n["event"].values.astype(float),
            sub_h["duration"].values.astype(float), sub_h["event"].values.astype(float),
        )
        med_n = _km_median(sub_n["duration"].values.astype(float), sub_n["event"].values.astype(float))
        med_h = _km_median(sub_h["duration"].values.astype(float), sub_h["event"].values.astype(float))
        s2_rows.append({
            "freq_high_threshold_hz": freq_high_min,
            "n_normal": len(sub_n),
            "n_high": len(sub_h),
            "n_fail_high": int(sub_h["event"].sum()),
            "median_ttf_normal": med_n,
            "median_ttf_high": med_h,
            "logrank_p_hi_vs_normal": p,
        })
    s2_df = pd.DataFrame(s2_rows)
    s2_df.to_csv(out / "p11_s2_freq_threshold.csv", index=False, encoding="utf-8-sig")
    results["s2_freq_threshold"] = s2_df

    # Check stability: does p < 0.05 for most thresholds?
    sig_count = (s2_df["logrank_p_hi_vs_normal"].dropna() < 0.05).sum()
    results["s2_threshold_stable"] = sig_count >= 3
    print(f"  S2: freq threshold sensitivity — {sig_count}/{len(s2_df)} thresholds show p<0.05")

    # ------------------------------------------------------------------
    # S3 — Infant mortality threshold stability
    # ------------------------------------------------------------------
    s3_rows = []
    for thr in INFANT_THRESHOLDS:
        for grp in grp_order:
            eligible = df[((df["duration"] >= thr) | (df["event"] == 1)) & (df["freq_group"] == grp)]
            n_inf = int(((eligible["event"] == 1) & (eligible["duration"] < thr)).sum())
            n_el  = len(eligible)
            s3_rows.append({
                "threshold_d": thr,
                "freq_group": grp,
                "infant_rate_pct": n_inf / n_el * 100 if n_el > 0 else np.nan,
                "n_eligible": n_el,
                "n_infant": n_inf,
            })
    s3_df = pd.DataFrame(s3_rows)
    s3_df.to_csv(out / "p11_s3_infant_threshold.csv", index=False, encoding="utf-8-sig")
    results["s3_infant_threshold"] = s3_df

    # Is high>normal consistently across thresholds?
    hi_rates = s3_df[s3_df["freq_group"] == "High (>55 Hz)"].set_index("threshold_d")["infant_rate_pct"]
    nm_rates = s3_df[s3_df["freq_group"] == "Normal (50–55 Hz)"].set_index("threshold_d")["infant_rate_pct"]
    common_thrs = hi_rates.index.intersection(nm_rates.index)
    hi_above = sum(hi_rates[t] > nm_rates[t] for t in common_thrs if np.isfinite(hi_rates.get(t, np.nan)) and np.isfinite(nm_rates.get(t, np.nan)))
    results["s3_stable"] = hi_above >= 3
    print(f"  S3: infant threshold stability — high freq > normal in {hi_above}/{len(common_thrs)} thresholds")

    # ------------------------------------------------------------------
    # S4 — Unit of analysis: well-level (first run per well)
    # ------------------------------------------------------------------
    df_well = df.sort_values("mount_date").groupby("well_key", observed=True).first().reset_index()
    s4_rows = []
    for grp in grp_order:
        sub = df_well[df_well["freq_group"] == grp].dropna(subset=["duration"])
        T = sub["duration"].values.astype(float)
        E = sub["event"].values.astype(float)
        s4_rows.append({
            "level": "well_first_run",
            "freq_group": grp,
            "n": len(sub),
            "n_fail": int(E.sum()),
            "median_ttf": _km_median(T, E),
        })
    sub_n_w = df_well[(df_well["freq_group"] == "Normal (50–55 Hz)")].dropna(subset=["duration"])
    sub_h_w = df_well[(df_well["freq_group"] == "High (>55 Hz)")].dropna(subset=["duration"])
    p_well = _logrank_p(
        sub_n_w["duration"].values.astype(float), sub_n_w["event"].values.astype(float),
        sub_h_w["duration"].values.astype(float), sub_h_w["event"].values.astype(float),
    )
    s4_rows.append({"level": "well_first_run", "freq_group": "High vs Normal logrank p", "n": np.nan, "n_fail": np.nan, "median_ttf": p_well})
    s4_df = pd.DataFrame(s4_rows)
    s4_df.to_csv(out / "p11_s4_unit_of_analysis.csv", index=False, encoding="utf-8-sig")
    results["s4_unit_analysis"] = s4_df

    # ------------------------------------------------------------------
    # S5 — RMST horizon sensitivity (50th, 75th, 90th pctile)
    # ------------------------------------------------------------------
    from lifelines.utils import restricted_mean_survival_time
    fail_times = df.loc[df["event"] == 1, "duration"].dropna()
    taus = {
        "tau_50th": float(fail_times.quantile(0.50)),
        "tau_75th": float(fail_times.quantile(0.75)),
        "tau_90th": float(fail_times.quantile(0.90)),
    }
    s5_rows = []
    for tau_label, tau in taus.items():
        for scope_label, scope_df in [("global", df), ("vt", df[df["is_vt"]])]:
            sub = scope_df.dropna(subset=["duration"])
            T = sub["duration"].values.astype(float)
            E = sub["event"].values.astype(float)
            if E.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(T, E)
            try:
                rmst = restricted_mean_survival_time(kmf, t=tau)
            except Exception:
                rmst = np.nan
            s5_rows.append({"tau_label": tau_label, "tau_days": tau, "scope": scope_label, "rmst": rmst})
    s5_df = pd.DataFrame(s5_rows)
    s5_df.to_csv(out / "p11_s5_rmst_horizon.csv", index=False, encoding="utf-8-sig")
    results["s5_rmst_horizon"] = s5_df

    # ------------------------------------------------------------------
    # Summary figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # S2 — freq threshold vs logrank p
    ax = axes[0]
    ax.plot(s2_df["freq_high_threshold_hz"], s2_df["logrank_p_hi_vs_normal"],
            "o-", color="#D65F5F", linewidth=2, markersize=7)
    ax.axhline(0.05, color="gray", linestyle="--", label="p=0.05")
    ax.set_xlabel("High-freq threshold (Hz)")
    ax.set_ylabel("Log-rank p (High vs Normal)")
    ax.set_title("S2 — Sensitivity: freq band threshold")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # S3 — infant rate by threshold
    ax2 = axes[1]
    for grp, color in [("High (>55 Hz)", "#D65F5F"), ("Normal (50–55 Hz)", "#6ACC65"), ("Low (≤50 Hz)", "#4878CF")]:
        sub = s3_df[s3_df["freq_group"] == grp]
        if sub.empty:
            continue
        ax2.plot(sub["threshold_d"], sub["infant_rate_pct"], "o-", color=color, label=grp, linewidth=2)
    ax2.set_xlabel("Infant threshold (days)")
    ax2.set_ylabel("Infant failure rate (%)")
    ax2.set_title("S3 — Sensitivity: infant mortality threshold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Phase 11 — Sensitivity Checks")
    fig.tight_layout()
    fig.savefig(out / "p11_sensitivity_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 11 — SENSITIVITY CHECKS")
    print("=" * 60)
    print("S1 — TTF definition (calendar vs true):")
    print(s1_df[s1_df["freq_group"] == "High vs Normal logrank p"].to_string(index=False))
    print("\nS2 — Freq band threshold:")
    print(s2_df[["freq_high_threshold_hz", "n_fail_high", "logrank_p_hi_vs_normal"]].to_string(index=False))
    print(f"\nS4 — Well-level logrank p (High vs Normal): {p_well:.4f}" if np.isfinite(p_well) else "\nS4 — insufficient data")
    print("\nS5 — RMST horizon sensitivity:")
    print(s5_df.to_string(index=False))

    return results
