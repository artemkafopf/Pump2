"""Phase 2 — Infant Mortality Analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency

from .config import (
    FAILURE_CATEGORIES, INFANT_THRESHOLDS, MAJOR_FIELDS, VT_FIELD,
)


def infant_rate(g: pd.DataFrame, threshold: int, duration_col: str = "duration") -> dict:
    """Compute infant failure rate for group g at given threshold.

    Denominator = runs where duration >= threshold OR event == 1.
    Numerator   = runs where event == 1 AND duration < threshold.
    """
    eligible = g[(g[duration_col] >= threshold) | (g["event"] == 1)].copy()
    n_eligible = len(eligible)
    n_infant = int(((eligible["event"] == 1) & (eligible[duration_col] < threshold)).sum())
    n_mature = int(((eligible["event"] == 1) & (eligible[duration_col] >= threshold)).sum())
    n_censored_before = int(((g["event"] == 0) & (g[duration_col] < threshold)).sum())
    return {
        "n_eligible": n_eligible,
        "n_infant_fail": n_infant,
        "n_mature_fail": n_mature,
        "n_censored_excluded": n_censored_before,
        "infant_rate_pct": n_infant / n_eligible * 100 if n_eligible > 0 else np.nan,
        "infant_share_of_fail_pct": n_infant / (n_infant + n_mature) * 100 if (n_infant + n_mature) > 0 else np.nan,
    }


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ------------------------------------------------------------------
    # 2.1  Infant rate by field × freq group × threshold
    # ------------------------------------------------------------------
    rows = []
    for field in MAJOR_FIELDS + ["GLOBAL"]:
        sub = df if field == "GLOBAL" else df[df["field"] == field]
        for grp in ["All", "Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]:
            if grp == "All":
                g = sub
            else:
                g = sub[sub["freq_group"] == grp]
            if len(g) == 0:
                continue
            for thr in INFANT_THRESHOLDS:
                ir = infant_rate(g, thr)
                rows.append({"field": field, "freq_group": grp, "threshold_d": thr, **ir})
    infant_table = pd.DataFrame(rows)
    infant_table.to_csv(out / "p2_infant_rate_table.csv", index=False, encoding="utf-8-sig")
    results["infant_table"] = infant_table

    # ------------------------------------------------------------------
    # 2.2  Heatmap: infant rate by field × freq group at primary threshold (90d)
    # ------------------------------------------------------------------
    primary_thr = 90
    pivot_data = infant_table[
        (infant_table["threshold_d"] == primary_thr) &
        (infant_table["freq_group"] != "All")
    ].pivot(index="field", columns="freq_group", values="infant_rate_pct")

    if not pivot_data.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot_data) * 0.5)))
        im = ax.imshow(pivot_data.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=min(100, pivot_data.values[np.isfinite(pivot_data.values)].max() * 1.1 if np.isfinite(pivot_data.values).any() else 50))
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns, rotation=15, ha="right")
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        plt.colorbar(im, ax=ax, label="Infant failure rate (%)")
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color="white" if val > 20 else "black", fontsize=8)
        ax.set_title(f"Phase 2 — Infant failure rate (threshold={primary_thr}d)\nby field × frequency group")
        fig.tight_layout()
        fig.savefig(out / "p2_infant_rate_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2.3  Infant rate by failure category × threshold (global, failures only)
    # ------------------------------------------------------------------
    cat_rows = []
    for cat in FAILURE_CATEGORIES + ["<missing>"]:
        g = df[(df["event"] == 1) & (df["failure_category"] == cat)]
        if len(g) == 0:
            continue
        for thr in INFANT_THRESHOLDS:
            n_inf = int((g["duration"] < thr).sum())
            total = len(g)
            cat_rows.append({
                "category": cat,
                "threshold_d": thr,
                "n_failures": total,
                "n_infant": n_inf,
                "infant_share_pct": n_inf / total * 100,
            })
    cat_infant = pd.DataFrame(cat_rows)
    cat_infant.to_csv(out / "p2_infant_by_category.csv", index=False, encoding="utf-8-sig")
    results["infant_by_category"] = cat_infant

    # Bar chart at primary threshold
    cat_90 = cat_infant[cat_infant["threshold_d"] == primary_thr].sort_values("infant_share_pct", ascending=True)
    if not cat_90.empty:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.barh(cat_90["category"], cat_90["infant_share_pct"], color="#D65F5F", alpha=0.85)
        ax2.set_xlabel(f"Infant share of failures (< {primary_thr}d) (%)")
        ax2.set_title(f"Phase 2 — % of failures that are infant (< {primary_thr}d) by category")
        for i, (_, row) in enumerate(cat_90.iterrows()):
            ax2.text(row["infant_share_pct"] + 0.5, i,
                     f"{row['infant_share_pct']:.0f}% (n={row['n_failures']})", va="center", fontsize=8)
        ax2.grid(True, axis="x", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out / "p2_infant_by_category.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # ------------------------------------------------------------------
    # 2.4  Vt: observed vs standardised infant rate (mix adjustment)
    # ------------------------------------------------------------------
    vt_df = df[df["is_vt"]]
    gl_df = df

    # Global category weights among eligible failures
    for thr in [primary_thr]:
        gl_eligible = gl_df[(gl_df["duration"] >= thr) | (gl_df["event"] == 1)]
        gl_fail = gl_eligible[gl_eligible["event"] == 1]
        if gl_fail.empty:
            continue
        gl_cat_weights = gl_fail["failure_category"].value_counts(normalize=True)

        vt_eligible = vt_df[(vt_df["duration"] >= thr) | (vt_df["event"] == 1)]
        if vt_eligible.empty:
            continue

        # Per-category infant rate within Vt
        vt_cat_infant = {}
        for cat in FAILURE_CATEGORIES:
            g = vt_eligible[(vt_eligible["failure_category"] == cat)]
            if len(g) == 0:
                vt_cat_infant[cat] = np.nan
                continue
            n_inf = int(((g["event"] == 1) & (g["duration"] < thr)).sum())
            vt_cat_infant[cat] = n_inf / len(g) * 100

        # Vt observed rate
        vt_obs = infant_rate(vt_df, thr)["infant_rate_pct"]
        # Standardised rate: global weights × Vt within-category rates
        std_rate = sum(
            gl_cat_weights.get(cat, 0) * (vt_cat_infant.get(cat, np.nan) / 100)
            for cat in FAILURE_CATEGORIES
            if np.isfinite(vt_cat_infant.get(cat, np.nan))
        ) * 100

        results[f"vt_infant_obs_{thr}d"] = vt_obs
        results[f"vt_infant_standardised_{thr}d"] = std_rate
        results[f"vt_infant_global_{thr}d"] = infant_rate(gl_df, thr)["infant_rate_pct"]

    # ------------------------------------------------------------------
    # 2.5  Fisher test: infant vs mature split across freq groups (per field)
    # ------------------------------------------------------------------
    fisher_rows = []
    for field in MAJOR_FIELDS + ["GLOBAL"]:
        sub = df if field == "GLOBAL" else df[df["field"] == field]
        for thr in [primary_thr]:
            eligible = sub[(sub["duration"] >= thr) | (sub["event"] == 1)]
            counts = eligible.groupby("freq_group", observed=True).apply(
                lambda g: pd.Series({
                    "n_infant": int(((g["event"] == 1) & (g["duration"] < thr)).sum()),
                    "n_mature": int(((g["event"] == 1) & (g["duration"] >= thr)).sum() +
                                    (g["event"] == 0).sum()),
                }), include_groups=False
            ).reset_index()

            lo = counts[counts["freq_group"] == "Low (≤50 Hz)"]
            hi = counts[counts["freq_group"] == "High (>55 Hz)"]
            if lo.empty or hi.empty:
                continue
            table = np.array([
                [int(lo["n_infant"].iloc[0]), int(lo["n_mature"].iloc[0])],
                [int(hi["n_infant"].iloc[0]), int(hi["n_mature"].iloc[0])],
            ])
            if table.sum() > 0 and table.min() >= 0:
                try:
                    _, p = fisher_exact(table, alternative="two-sided")
                except Exception:
                    p = np.nan
                lo_rate = table[0, 0] / table[0].sum() * 100 if table[0].sum() > 0 else np.nan
                hi_rate = table[1, 0] / table[1].sum() * 100 if table[1].sum() > 0 else np.nan
                fisher_rows.append({
                    "field": field, "threshold_d": thr,
                    "low_infant_rate_pct": lo_rate, "high_infant_rate_pct": hi_rate,
                    "abs_diff_pct": hi_rate - lo_rate if np.isfinite(lo_rate) and np.isfinite(hi_rate) else np.nan,
                    "fisher_p": round(p, 4),
                    "bonferroni_p": round(min(p * len(MAJOR_FIELDS), 1.0), 4) if np.isfinite(p) else np.nan,
                    "sig_bonferroni": p * len(MAJOR_FIELDS) < 0.05 if np.isfinite(p) else False,
                })

    fisher_df = pd.DataFrame(fisher_rows)
    fisher_df.to_csv(out / "p2_infant_fisher_tests.csv", index=False, encoding="utf-8-sig")
    results["infant_fisher"] = fisher_df

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 — INFANT MORTALITY")
    print("=" * 60)
    thr = primary_thr
    vt_obs = results.get(f"vt_infant_obs_{thr}d", np.nan)
    vt_std = results.get(f"vt_infant_standardised_{thr}d", np.nan)
    vt_gl  = results.get(f"vt_infant_global_{thr}d", np.nan)
    print(f"Infant rate at {thr}d threshold:")
    print(f"  Global: {vt_gl:.1f}%")
    print(f"  Vt observed: {vt_obs:.1f}%")
    print(f"  Vt mix-standardised (global weights): {vt_std:.1f}%")
    if np.isfinite(vt_std) and np.isfinite(vt_obs):
        mix_explains = abs(vt_obs - vt_std) < abs(vt_obs - vt_gl) * 0.5
        print(f"  → Mix {'partly explains' if mix_explains else 'does NOT explain'} elevated Vt infant rate")
    if not fisher_df.empty:
        print("\nFisher test (Low vs High freq, infant split):")
        print(fisher_df[["field", "threshold_d", "low_infant_rate_pct", "high_infant_rate_pct",
                          "abs_diff_pct", "fisher_p", "sig_bonferroni"]].to_string(index=False))

    return results
