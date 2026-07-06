"""Phase 7 — Confounding Test (stratified Spearman correlations)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FAILURE_CATEGORIES, MAJOR_FIELDS, VT_FIELD


def _spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(both) < 5:
        return np.nan, np.nan, len(both)
    r, p = spearmanr(both["x"], both["y"])
    return float(r), float(p), len(both)


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # Use only failure events for Spearman (censored would dominate and create
    # artifactual correlations). Note this is the biased step — survival models are primary.
    failures_df = df[df["event"] == 1].copy()

    exposure_col = "freq_signed_exposure"  # signed_freq_exposure from mart
    if exposure_col not in df.columns:
        exposure_col = "freq_above_55hz_pct"

    rows = []

    # ------------------------------------------------------------------
    # 7a — Raw correlation (all failures, global)
    # ------------------------------------------------------------------
    r, p, n = _spearman_safe(failures_df[exposure_col], failures_df["duration"])
    rows.append({
        "stratification": "NONE (raw)",
        "field": "GLOBAL",
        "category": "ALL",
        "n": n,
        "spearman_r": r,
        "p_value": p,
        "note": "Biased — excludes all censored runs",
    })

    # ------------------------------------------------------------------
    # 7b — Within-field stratification
    # ------------------------------------------------------------------
    for field in MAJOR_FIELDS + [VT_FIELD]:
        sub = failures_df[failures_df["field"] == field]
        r, p, n = _spearman_safe(sub[exposure_col], sub["duration"])
        rows.append({
            "stratification": "within_field",
            "field": field,
            "category": "ALL",
            "n": n,
            "spearman_r": r,
            "p_value": p,
            "note": "",
        })

    # ------------------------------------------------------------------
    # 7c — Within-category stratification (global)
    # ------------------------------------------------------------------
    for cat in FAILURE_CATEGORIES:
        sub = failures_df[failures_df["failure_category"] == cat]
        r, p, n = _spearman_safe(sub[exposure_col], sub["duration"])
        rows.append({
            "stratification": "within_category",
            "field": "GLOBAL",
            "category": cat,
            "n": n,
            "spearman_r": r,
            "p_value": p,
            "note": "",
        })

    # ------------------------------------------------------------------
    # 7c — Within-category × within-field (Vt)
    # ------------------------------------------------------------------
    vt_fail = failures_df[failures_df["is_vt"]]
    for cat in FAILURE_CATEGORIES:
        sub = vt_fail[vt_fail["failure_category"] == cat]
        r, p, n = _spearman_safe(sub[exposure_col], sub["duration"])
        rows.append({
            "stratification": "within_category",
            "field": VT_FIELD,
            "category": cat,
            "n": n,
            "spearman_r": r,
            "p_value": p,
            "note": "",
        })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out / "p7_spearman_correlations.csv", index=False, encoding="utf-8-sig")
    results["spearman"] = corr_df

    # ------------------------------------------------------------------
    # 7d — Interpretation: does r survive stratification?
    # ------------------------------------------------------------------
    raw_r = rows[0]["spearman_r"]
    within_field_rs = [r["spearman_r"] for r in rows if r["stratification"] == "within_field" and np.isfinite(r["spearman_r"])]
    within_cat_rs = [r["spearman_r"] for r in rows if r["stratification"] == "within_category" and r["field"] == "GLOBAL" and np.isfinite(r["spearman_r"])]

    interpretation = {
        "raw_r": raw_r,
        "within_field_mean_r": np.mean(within_field_rs) if within_field_rs else np.nan,
        "within_field_same_sign": all(np.sign(r) == np.sign(raw_r) for r in within_field_rs if np.isfinite(r)) if within_field_rs else False,
        "within_cat_mean_r": np.mean(within_cat_rs) if within_cat_rs else np.nan,
        "within_cat_same_sign": all(np.sign(r) == np.sign(raw_r) for r in within_cat_rs if np.isfinite(r)) if within_cat_rs else False,
    }
    if np.isfinite(raw_r) and within_field_rs:
        if abs(np.mean(within_field_rs)) < 0.5 * abs(raw_r):
            interpretation["field_confounding"] = "LIKELY — correlation substantially attenuated within fields"
        else:
            interpretation["field_confounding"] = "UNLIKELY — correlation persists within fields"
    if np.isfinite(raw_r) and within_cat_rs:
        if abs(np.mean(within_cat_rs)) < 0.5 * abs(raw_r):
            interpretation["category_confounding"] = "LIKELY — correlation substantially attenuated within categories"
        else:
            interpretation["category_confounding"] = "UNLIKELY — correlation persists within categories"

    results["interpretation"] = interpretation
    pd.DataFrame([interpretation]).to_csv(out / "p7_interpretation.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 7e — Summary forest plot of Spearman r by stratification layer
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    layers = [
        ("NONE (raw)", "Raw (global)", axes[0]),
        ("within_field", "Within field", axes[1]),
        ("within_category", "Within category (global)", axes[2]),
    ]
    for strat, label, ax in layers:
        sub = corr_df[corr_df["stratification"] == strat].dropna(subset=["spearman_r"])
        if sub.empty:
            ax.set_visible(False)
            continue
        y = range(len(sub))
        colors = ["#D65F5F" if r < 0 else "#4878CF" for r in sub["spearman_r"]]
        ax.barh(list(y), sub["spearman_r"].values, color=colors, alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(list(y))
        grp_labels = sub["field"].values if strat == "within_field" else sub["category"].values
        ax.set_yticklabels(grp_labels, fontsize=8)
        ax.set_xlabel(f"Spearman r ({exposure_col} vs TTF)")
        ax.set_title(label)
        ax.set_xlim(-1, 1)
        ax.grid(True, axis="x", alpha=0.2)

    fig.suptitle("Phase 7 — Confounding test: Spearman r by stratification layer\n"
                 "(failures only — biased; survival models are primary evidence)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "p7_spearman_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7 — CONFOUNDING TEST")
    print("=" * 60)
    print("⚠  Spearman on failures only — biased where censoring rates differ across groups.")
    print("   These are directional indicators; survival models in Phase 8 are primary.\n")
    print(corr_df[["stratification", "field", "category", "n", "spearman_r", "p_value"]].to_string(index=False))
    for k, v in interpretation.items():
        print(f"  {k}: {v}")

    return results
