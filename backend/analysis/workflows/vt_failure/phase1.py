"""Phase 1 — Descriptive Baseline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    INFANT_THRESHOLDS, MAJOR_FIELDS, VT_FIELD, FREQ_GROUP_COLORS,
)


def _summary_for_group(g: pd.DataFrame, duration_col: str = "duration") -> pd.Series:
    n_runs = len(g)
    n_fail = int(g["event"].sum())
    dur = g[duration_col].dropna()
    fail_dur = g.loc[g["event"] == 1, duration_col].dropna()
    mature_dur = g.loc[(g["event"] == 1) & (g[duration_col] >= 90), duration_col].dropna()
    infant_counts = {}
    for thr in INFANT_THRESHOLDS:
        eligible = g[(g[duration_col] >= thr) | (g["event"] == 1)]
        n_inf = int(((eligible["event"] == 1) & (eligible[duration_col] < thr)).sum())
        infant_counts[f"infant_rate_{thr}d"] = n_inf / len(eligible) if len(eligible) > 0 else np.nan

    row = {
        "n_runs": n_runs,
        "n_fail": n_fail,
        "fail_rate_pct": n_fail / n_runs * 100 if n_runs > 0 else np.nan,
        "median_ttf_all": dur.median() if len(dur) > 0 else np.nan,
        "median_ttf_failures": fail_dur.median() if len(fail_dur) > 0 else np.nan,
        "median_ttf_mature": mature_dur.median() if len(mature_dur) > 0 else np.nan,
        "avg_freq_hz": g["freq_w_mean"].mean(),
        "pct_high_freq": (g["freq_w_mean"] > 55).mean() * 100,
        "median_h2s_mg_l": g["h2s_proxy_mg_l"].median(),
        "pct_acidic": (g["h2s_label"] == "Кислый").mean() * 100,
        "avg_glf": g["avg_glf"].mean(),
        "avg_kpod": g["avg_kpod"].mean(),
        "pct_high_glf": (g["avg_glf"] > 0.25).mean() * 100,
        "median_trf_per_day": g["trf_per_day"].median(),
        "median_tlf_per_day": g["tlf_per_day"].median(),
        "median_ca_load_kg": g["cum_calcium_load_kg"].median() if "cum_calcium_load_kg" in g.columns else np.nan,
        "median_cl_load_kg": g["cum_chloride_load_kg"].median() if "cum_chloride_load_kg" in g.columns else np.nan,
        "median_so4_load_kg": g["cum_sulfate_load_kg"].median() if "cum_sulfate_load_kg" in g.columns else np.nan,
        "median_salt_kg": g["cum_salt_load_kg"].median() if "cum_salt_load_kg" in g.columns else np.nan,
        "median_gypsum": g["cum_gypsum_scale_proxy"].median() if "cum_gypsum_scale_proxy" in g.columns else np.nan,
    }
    row.update(infant_counts)
    return pd.Series(row)


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ----- By field -----
    field_stats = (
        df.groupby("field", observed=True)
        .apply(_summary_for_group, include_groups=False)
        .reset_index()
    )
    global_row = _summary_for_group(df).to_frame().T.assign(field="GLOBAL")
    field_stats = pd.concat([global_row, field_stats], ignore_index=True)
    field_stats = field_stats.round(2)
    field_stats.to_csv(out / "p1_field_summary.csv", index=False, encoding="utf-8-sig")
    results["field_summary"] = field_stats

    # ----- By frequency group -----
    grp_stats = (
        df[df["freq_group"] != "<no freq data>"]
        .groupby("freq_group", observed=True)
        .apply(_summary_for_group, include_groups=False)
        .reset_index()
    )
    grp_stats = grp_stats.round(2)
    grp_stats.to_csv(out / "p1_freq_group_summary.csv", index=False, encoding="utf-8-sig")
    results["freq_group_summary"] = grp_stats

    # ----- Failure category mix by field -----
    cat_mix = (
        df[df["event"] == 1]
        .groupby(["field", "failure_category"], observed=True)["row_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Add percentage columns
    cat_cols = [c for c in cat_mix.columns if c != "field"]
    cat_mix["total"] = cat_mix[cat_cols].sum(axis=1)
    for c in cat_cols:
        cat_mix[f"{c}_pct"] = (cat_mix[c] / cat_mix["total"] * 100).round(1)
    cat_mix.to_csv(out / "p1_category_mix_by_field.csv", index=False, encoding="utf-8-sig")
    results["category_mix"] = cat_mix

    # ----- Vt vs Global comparison figure -----
    vt_row = field_stats[field_stats["field"] == VT_FIELD].iloc[0]
    gl_row = field_stats[field_stats["field"] == "GLOBAL"].iloc[0]

    metrics = {
        "Avg freq (Hz)": ("avg_freq_hz", "Hz"),
        "% High freq (>55 Hz)": ("pct_high_freq", "%"),
        "Median H2S (mg/L)": ("median_h2s_mg_l", "mg/L"),
        "% Acidic (Кислый)": ("pct_acidic", "%"),
        "Avg GLF": ("avg_glf", ""),
        "Avg Kpod": ("avg_kpod", ""),
        "Median TTF all (days)": ("median_ttf_all", "d"),
        "Median TTF failures (days)": ("median_ttf_failures", "d"),
        "Infant rate 90d (%)": ("infant_rate_90d", "%"),
        "Failure rate (%)": ("fail_rate_pct", "%"),
    }
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (label, (col, unit)) in zip(axes.flat, metrics.items()):
        vt_val = float(vt_row[col]) if col in vt_row.index else np.nan
        gl_val = float(gl_row[col]) if col in gl_row.index else np.nan
        colors = ["#D65F5F", "#4878CF"]
        bars = ax.bar(["Vt", "Global"], [vt_val, gl_val], color=colors, width=0.5)
        ax.set_title(label, fontsize=8, pad=4)
        ax.set_ylabel(unit, fontsize=7)
        for bar, val in zip(bars, [vt_val, gl_val]):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase 1 — Vt vs Global: key operating and failure metrics", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "p1_vt_vs_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Category mix figure: Vt vs Global (failures only) -----
    from .config import FAILURE_CATEGORIES as _FAIL_CATS

    def _cat_pcts(sub_df):
        counts = sub_df[sub_df["event"] == 1]["failure_category"].value_counts()
        total = counts.sum()
        return {cat: counts.get(cat, 0) / total * 100 if total > 0 else 0.0 for cat in _FAIL_CATS}

    vt_pcts = _cat_pcts(df[df["is_vt"]])
    gl_pcts = _cat_pcts(df)
    cat_names = _FAIL_CATS
    x = np.arange(len(cat_names))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x - width / 2, [vt_pcts[c] for c in cat_names], width, label="Vt", color="#D65F5F", alpha=0.85)
    ax2.bar(x + width / 2, [gl_pcts[c] for c in cat_names], width, label="Global", color="#4878CF", alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Share of failures (%)")
    ax2.set_title("Phase 1 — Failure category mix: Vt vs Global (failures only)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "p1_category_mix_vt_vs_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ----- Print summary -----
    print("\n" + "=" * 60)
    print("PHASE 1 — DESCRIPTIVE BASELINE")
    print("=" * 60)
    print(field_stats[["field", "n_runs", "n_fail", "median_ttf_failures",
                        "avg_freq_hz", "median_h2s_mg_l", "pct_acidic"]].to_string(index=False))
    print(f"\nFreq group summary:")
    print(grp_stats[["freq_group", "n_runs", "n_fail", "median_ttf_failures"]].to_string(index=False))

    return results
