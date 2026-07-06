"""Phase 6 — Cumulative Duty and Chemistry-Proxy Comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FAILURE_CATEGORIES, FREQ_GROUP_COLORS, VT_FIELD


ION_COLS = [
    ("cum_calcium_load_kg",  "Calcium load (kg)"),
    ("cum_chloride_load_kg", "Chloride load (kg)"),
    ("cum_sulfate_load_kg",  "Sulfate load (kg)"),
    ("cum_salt_load_kg",     "Combined salt load (kg)"),
    ("cum_gypsum_scale_proxy", "Gypsum scale proxy"),
]


def _box_by_group(df: pd.DataFrame, col: str, group_col: str, groups: list,
                   colors: list, ylabel: str, title: str, out_path: Path) -> None:
    data = [df.loc[df[group_col] == g, col].dropna().values for g in groups]
    data = [d for d in data if len(d) > 0]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(groups[:len(data)], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    grp_colors = [FREQ_GROUP_COLORS[g] for g in grp_order]

    # ------------------------------------------------------------------
    # 6.1  Normalised TRF (TRF/TTF = mean freq hz) by freq group
    # ------------------------------------------------------------------
    # trf_per_day ≡ mean operating frequency; show vs freq_group as sanity check
    for metric, label, fname in [
        ("trf_per_day", "TRF/TTF (mean operating Hz)", "trf_per_day"),
        ("tlf_per_day", "TLF/TTF (mean qliq m³/day)", "tlf_per_day"),
    ]:
        if metric not in df.columns:
            continue
        _box_by_group(
            df[df["freq_group"].isin(grp_order)],
            metric, "freq_group", grp_order, grp_colors,
            label,
            f"Phase 6 — {label} by frequency group",
            out / f"p6_box_{fname}_by_freq.png",
        )

    # ------------------------------------------------------------------
    # 6.2  Duty metric stats table
    # ------------------------------------------------------------------
    duty_rows = []
    for grp in grp_order:
        sub = df[df["freq_group"] == grp]
        duty_rows.append({
            "freq_group": grp,
            "n_runs": len(sub),
            "median_trf_per_day": sub["trf_per_day"].median(),
            "median_tlf_per_day": sub["tlf_per_day"].median(),
            "n_zero_tlf": (sub["total_liquid_m3"].fillna(0) < 1).sum(),
            "pct_zero_tlf": (sub["total_liquid_m3"].fillna(0) < 1).mean() * 100,
        })
    duty_df = pd.DataFrame(duty_rows)
    duty_df.to_csv(out / "p6_duty_by_freq_group.csv", index=False, encoding="utf-8-sig")
    results["duty_by_freq"] = duty_df

    # ------------------------------------------------------------------
    # 6.3  Ion-proxy distributions by field (Vt vs other major fields)
    # ------------------------------------------------------------------
    ion_field_rows = []
    fields_to_compare = ["GLOBAL", VT_FIELD] + [f for f in ["Ya", "Az", "Za", "Ic"] if f != VT_FIELD]
    for ion_col, ion_label in ION_COLS:
        if ion_col not in df.columns:
            continue
        for field in fields_to_compare:
            sub = df if field == "GLOBAL" else df[df["field"] == field]
            vals = sub[ion_col].dropna()
            if vals.empty:
                continue
            ion_field_rows.append({
                "field": field,
                "ion": ion_col,
                "ion_label": ion_label,
                "n_not_null": len(vals),
                "pct_coverage": len(vals) / len(sub) * 100,
                "median": vals.median(),
                "p25": vals.quantile(0.25),
                "p75": vals.quantile(0.75),
                "mean": vals.mean(),
            })
    ion_field_df = pd.DataFrame(ion_field_rows)
    if not ion_field_df.empty:
        ion_field_df.to_csv(out / "p6_ion_by_field.csv", index=False, encoding="utf-8-sig")
        results["ion_by_field"] = ion_field_df

    # ------------------------------------------------------------------
    # 6.4  Ion proxy vs failure category (correlation / box plots)
    # ------------------------------------------------------------------
    corr_rows = []
    for ion_col, ion_label in ION_COLS:
        if ion_col not in df.columns:
            continue
        sub = df[df["event"] == 1].dropna(subset=[ion_col, "failure_category"])
        if len(sub) < 20:
            continue
        # Median by category
        cat_med = sub.groupby("failure_category")[ion_col].median().reset_index()
        cat_med.columns = ["category", "median_val"]
        cat_med["ion"] = ion_label

        # Kruskal-Wallis across categories
        from scipy.stats import kruskal
        cat_groups = [sub.loc[sub["failure_category"] == c, ion_col].values
                      for c in FAILURE_CATEGORIES
                      if ((sub["failure_category"] == c) & sub[ion_col].notna()).sum() >= 5]
        if len(cat_groups) >= 2:
            try:
                kw_stat, kw_p = kruskal(*cat_groups)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
            corr_rows.append({
                "ion": ion_label,
                "kruskal_stat": round(kw_stat, 2) if np.isfinite(kw_stat) else np.nan,
                "kruskal_p": round(kw_p, 4) if np.isfinite(kw_p) else np.nan,
                "best_cat": cat_med.set_index("category")["median_val"].idxmax() if not cat_med.empty else "",
            })

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(out / "p6_ion_kruskal_by_category.csv", index=False, encoding="utf-8-sig")
        results["ion_kruskal"] = corr_df

    # ------------------------------------------------------------------
    # 6.5  Ion pairwise correlation matrix (global)
    # ------------------------------------------------------------------
    avail = [c for c, _ in ION_COLS if c in df.columns and df[c].notna().sum() > 30]
    if len(avail) >= 2:
        corr_mat = df[avail].corr(method="spearman").round(3)
        corr_mat.to_csv(out / "p6_ion_pairwise_corr.csv", encoding="utf-8-sig")
        results["ion_pairwise_corr"] = corr_mat

        # Figure
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels([c.replace("cum_", "").replace("_load_kg", "")
                             .replace("_scale_proxy", "") for c in avail], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(avail)))
        ax.set_yticklabels([c.replace("cum_", "").replace("_load_kg", "")
                             .replace("_scale_proxy", "") for c in avail], fontsize=8)
        plt.colorbar(im, ax=ax, label="Spearman r")
        for i in range(len(avail)):
            for j in range(len(avail)):
                ax.text(j, i, f"{corr_mat.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Phase 6 — Ion proxy pairwise Spearman correlations")
        fig.tight_layout()
        fig.savefig(out / "p6_ion_corr_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 6.6  Scatter: ion proxy vs duration (failures only, by scope)
    # ------------------------------------------------------------------
    for scope_label, scope_df in [("global", df), ("vt", df[df["is_vt"]])]:
        fails = scope_df[scope_df["event"] == 1].dropna(subset=["duration"])
        if len(fails) < 20:
            continue
        primary_ion = "cum_salt_load_kg" if "cum_salt_load_kg" in fails.columns else None
        if primary_ion is None:
            continue
        fails_clean = fails.dropna(subset=[primary_ion])
        if len(fails_clean) < 10:
            continue
        r, p = spearmanr(fails_clean["duration"], fails_clean[primary_ion])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(fails_clean["duration"], fails_clean[primary_ion],
                   alpha=0.4, s=20, color="#4878CF")
        ax.set_xlabel("Duration / TTF (days)")
        ax.set_ylabel("Cumulative salt load (kg)")
        ax.set_title(f"Phase 6 — Cum. salt load vs TTF ({scope_label}, failures only)\n"
                     f"Spearman r={r:.3f}, p={p:.4f}  [DESCRIPTIVE ONLY — censoring bias]")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"p6_salt_vs_ttf_{scope_label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 6 — DUTY AND CHEMISTRY PROXIES")
    print("=" * 60)
    print(duty_df.to_string(index=False))
    if not ion_field_df.empty:
        print("\nIon coverage and medians by field:")
        print(ion_field_df[["field", "ion", "n_not_null", "pct_coverage", "median"]].to_string(index=False))

    return results
