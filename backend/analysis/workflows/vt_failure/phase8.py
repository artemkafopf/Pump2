"""Phase 8 — Cox Proportional Hazards (sequential covariate adjustment)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from .config import (
    MAJOR_FIELDS, MIN_FAILURES_COX, VT_FIELD,
)


def _encode_freq_group(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode freq_group with Normal as reference."""
    df2 = df.copy()
    df2["freq_low"]  = (df2["freq_group"] == "Low (≤50 Hz)").astype(float)
    df2["freq_high"] = (df2["freq_group"] == "High (>55 Hz)").astype(float)
    return df2


def _fit_cox(
    df: pd.DataFrame,
    covariates: list[str],
    duration_col: str = "duration",
    event_col: str = "event",
    cluster_col: str = "well_key",
    label: str = "",
) -> dict:
    """Fit one Cox model. Returns dict with HR, CI, p, c-index."""
    sub = df[covariates + [duration_col, event_col, cluster_col]].dropna().copy()
    if sub[event_col].sum() < MIN_FAILURES_COX:
        return {"label": label, "error": f"n_fail={int(sub[event_col].sum())} < {MIN_FAILURES_COX}"}

    cph = CoxPHFitter()
    try:
        cph.fit(
            sub,
            duration_col=duration_col,
            event_col=event_col,
            cluster_col=cluster_col,
            robust=True,  # cluster-robust SEs
        )
    except Exception as exc:
        return {"label": label, "error": str(exc)}

    summary = cph.summary.copy()
    result = {"label": label, "n_total": len(sub), "n_fail": int(sub[event_col].sum())}

    for var in covariates:
        if var in summary.index:
            row = summary.loc[var]
            result[f"{var}_hr"]    = round(float(row["exp(coef)"]), 3)
            result[f"{var}_ci_lo"] = round(float(row["exp(coef) lower 95%"]), 3)
            result[f"{var}_ci_hi"] = round(float(row["exp(coef) upper 95%"]), 3)
            result[f"{var}_p"]     = round(float(row["p"]), 4)

    result["concordance"] = round(float(cph.concordance_index_), 3)

    # PH test (Schoenfeld)
    try:
        ph_result = proportional_hazard_test(cph, sub, time_transform="rank")
        ph_summary = ph_result.summary
        result["ph_test_global_p"] = round(float(ph_summary["p"].min()), 4)
        result["ph_assumption_ok"] = bool(ph_summary["p"].min() > 0.05)
    except Exception:
        result["ph_test_global_p"] = np.nan
        result["ph_assumption_ok"] = None

    result["_cph_obj"] = cph  # keep for plotting
    return result


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # Need ≥ MIN_FAILURES_COX globally
    if df["event"].sum() < MIN_FAILURES_COX:
        print(f"Phase 8: global n_fail={df['event'].sum()} < {MIN_FAILURES_COX}, skipping Cox.")
        return results

    df2 = _encode_freq_group(df).copy()

    # Encode field as dummy (Ya as reference — largest)
    fields_present = [f for f in MAJOR_FIELDS if (df2["field"] == f).sum() > 0]
    ref_field = "Ya"
    for f in fields_present:
        if f != ref_field:
            df2[f"field_{f}"] = (df2["field"] == f).astype(float)

    # Encode H2S class
    df2["is_acidic"] = (df2["h2s_label"] == "Кислый").astype(float)

    # Encode mount_year (standardised)
    yr_mean = df2["mount_year"].mean()
    yr_std  = df2["mount_year"].std()
    df2["mount_year_std"] = (df2["mount_year"] - yr_mean) / (yr_std + 1e-9)

    # GLF normalised
    glf_mean = df2["avg_glf"].mean()
    glf_std  = df2["avg_glf"].std()
    df2["avg_glf_std"] = (df2["avg_glf"] - glf_mean) / (glf_std + 1e-9)

    field_dummies = [f"field_{f}" for f in fields_present if f != ref_field and f"field_{f}" in df2.columns]

    # ------------------------------------------------------------------
    # Sequential models
    # ------------------------------------------------------------------
    models = [
        ("M1_univariate",    ["freq_low", "freq_high"]),
        ("M2_plus_field",    ["freq_low", "freq_high"] + field_dummies),
        ("M3_plus_year",     ["freq_low", "freq_high"] + field_dummies + ["mount_year_std"]),
        ("M4_plus_h2s_glf",  ["freq_low", "freq_high"] + field_dummies + ["mount_year_std", "is_acidic", "avg_glf_std"]),
    ]

    # Add salt/ion if coverage > 60%
    if "cum_salt_load_kg" in df2.columns and df2["cum_salt_load_kg"].notna().mean() > 0.6:
        # Log-transform and standardise
        df2["log_salt"] = np.log1p(df2["cum_salt_load_kg"].clip(lower=0))
        salt_std = df2["log_salt"].std()
        df2["log_salt_std"] = (df2["log_salt"] - df2["log_salt"].mean()) / (salt_std + 1e-9)
        models.append(
            ("M5_plus_salt", ["freq_low", "freq_high"] + field_dummies + ["mount_year_std", "is_acidic", "avg_glf_std", "log_salt_std"])
        )

    all_model_results = []
    cox_objs = {}

    for label, covariates in models:
        res = _fit_cox(df2, covariates, label=label)
        all_model_results.append(res)
        if "_cph_obj" in res:
            cox_objs[label] = res.pop("_cph_obj")

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------
    rows_for_table = []
    for res in all_model_results:
        row = {k: v for k, v in res.items() if not k.startswith("_")}
        rows_for_table.append(row)
    cox_table = pd.DataFrame(rows_for_table)
    cox_table.to_csv(out / "p8_cox_model_table.csv", index=False, encoding="utf-8-sig")
    results["cox_table"] = cox_table

    # ------------------------------------------------------------------
    # HR forest plot for primary freq covariate across models
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, max(4, len(models) * 1.2)))
    y_pos = list(range(len(models)))
    for i, (label, _) in enumerate(models):
        res = all_model_results[i]
        hr_hi = res.get("freq_high_hr", np.nan)
        lo_hi = res.get("freq_high_ci_lo", np.nan)
        hi_hi = res.get("freq_high_ci_hi", np.nan)
        if np.isfinite(hr_hi):
            ax.scatter(hr_hi, i, color="#D65F5F", s=80, zorder=3)
            if np.isfinite(lo_hi) and np.isfinite(hi_hi):
                ax.hlines(i, lo_hi, hi_hi, color="#D65F5F", linewidth=3, alpha=0.6)
        ax.text(max(hi_hi if np.isfinite(hi_hi) else hr_hi, hr_hi) + 0.05, i,
                f"HR={hr_hi:.3f} [{lo_hi:.3f}–{hi_hi:.3f}]" if all(np.isfinite([hr_hi, lo_hi, hi_hi])) else "N/A",
                va="center", fontsize=8)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for lbl, _ in models], fontsize=8)
    ax.set_xlabel("Hazard Ratio for High Freq (>55 Hz) vs Normal (50–55 Hz)")
    ax.set_title("Phase 8 — Cox HR for High Freq: sequential adjustment\n(cluster-robust SEs on well_key)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "p8_cox_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Vt-only Cox (if n_fail sufficient)
    # ------------------------------------------------------------------
    vt_df = df2[df2["is_vt"]].copy()
    if vt_df["event"].sum() >= MIN_FAILURES_COX:
        vt_covariates = ["freq_low", "freq_high", "is_acidic", "avg_glf_std", "mount_year_std"]
        vt_res = _fit_cox(vt_df, vt_covariates, label="M_vt_only")
        vt_row = {k: v for k, v in vt_res.items() if not k.startswith("_")}
        pd.DataFrame([vt_row]).to_csv(out / "p8_cox_vt_only.csv", index=False, encoding="utf-8-sig")
        results["cox_vt"] = vt_row

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 8 — COX PROPORTIONAL HAZARDS")
    print("=" * 60)
    show_cols = ["label", "n_fail", "freq_high_hr", "freq_high_ci_lo", "freq_high_ci_hi",
                 "freq_high_p", "concordance", "ph_assumption_ok"]
    show_cols = [c for c in show_cols if c in cox_table.columns]
    print(cox_table[show_cols].to_string(index=False))
    if "cox_vt" in results:
        print(f"\nVt-only Cox: HR_high={results['cox_vt'].get('freq_high_hr','N/A')} "
              f"[{results['cox_vt'].get('freq_high_ci_lo','N/A')}–{results['cox_vt'].get('freq_high_ci_hi','N/A')}]")

    return results
