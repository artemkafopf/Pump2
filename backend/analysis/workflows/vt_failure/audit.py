"""Step 0 — Data Audit: feasibility checks before any survival analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FREQ_HIGH_MIN, FREQ_LOW_MAX, FREQ_VERY_HIGH_MIN,
    INFANT_THRESHOLDS, MAJOR_FIELDS, MIN_FAILURES_KM, MIN_FAILURES_WEIBULL,
    VT_FIELD,
)


def run(df: pd.DataFrame, out: Path) -> dict:
    """Run all data-audit checks. Returns a summary dict. Saves CSV/text to *out*."""
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ------------------------------------------------------------------
    # 1. Count table: runs × failures by frequency band × field
    # ------------------------------------------------------------------
    df2 = df.copy()
    df2["freq_band"] = pd.cut(
        df2["freq_w_mean"],
        bins=[-np.inf, FREQ_LOW_MAX, FREQ_HIGH_MIN, FREQ_VERY_HIGH_MIN, np.inf],
        labels=["≤50 Hz", "50–55 Hz", "55–58 Hz", ">58 Hz"],
        right=True,
    ).astype("string").fillna("<no telemetry>")

    band_field = (
        df2.groupby(["field", "freq_band"], observed=True)
        .agg(n_runs=("row_id", "count"), n_fail=("event", "sum"))
        .reset_index()
    )
    band_global = (
        df2.groupby("freq_band", observed=True)
        .agg(n_runs=("row_id", "count"), n_fail=("event", "sum"))
        .reset_index()
        .assign(field="GLOBAL")
    )
    band_table = pd.concat([band_global, band_field], ignore_index=True)
    band_table.to_csv(out / "s0_freq_band_counts.csv", index=False, encoding="utf-8-sig")
    results["freq_band_table"] = band_table

    # Check if very-high (>58 Hz) group is viable
    very_high_global = band_table.loc[
        (band_table["field"] == "GLOBAL") & (band_table["freq_band"] == ">58 Hz"), "n_fail"
    ]
    vh_n = int(very_high_global.iloc[0]) if not very_high_global.empty else 0
    results["very_high_60hz_viable"] = vh_n >= MIN_FAILURES_WEIBULL
    results["very_high_60hz_n_fail"] = vh_n

    # ------------------------------------------------------------------
    # 2. Telemetry coverage
    # ------------------------------------------------------------------
    tele = (
        df2.groupby("field", observed=True)
        .apply(lambda g: pd.Series({
            "n_runs": len(g),
            "n_with_freq": (g["n_freq_valid_days"] >= 7).sum(),
            "pct_with_freq": (g["n_freq_valid_days"] >= 7).mean() * 100,
            "pct_using_true_ttf": g["using_true_ttf"].mean() * 100,
        }), include_groups=False)
        .reset_index()
    )
    global_tele = pd.Series({
        "field": "GLOBAL",
        "n_runs": len(df2),
        "n_with_freq": (df2["n_freq_valid_days"] >= 7).sum(),
        "pct_with_freq": (df2["n_freq_valid_days"] >= 7).mean() * 100,
        "pct_using_true_ttf": df2["using_true_ttf"].mean() * 100,
    })
    tele = pd.concat([pd.DataFrame([global_tele]), tele], ignore_index=True)
    tele.to_csv(out / "s0_telemetry_coverage.csv", index=False, encoding="utf-8-sig")
    results["telemetry"] = tele

    # ------------------------------------------------------------------
    # 3. Failure category coverage
    # ------------------------------------------------------------------
    cat_cov = df2.groupby("event", observed=True).apply(
        lambda g: pd.Series({
            "n_runs": len(g),
            "n_with_category": (g["failure_category"] != "<missing>").sum(),
            "pct_with_category": (g["failure_category"] != "<missing>").mean() * 100,
        }), include_groups=False
    ).reset_index()
    cat_cov.to_csv(out / "s0_category_coverage.csv", index=False, encoding="utf-8-sig")
    results["category_coverage"] = cat_cov

    # ------------------------------------------------------------------
    # 4. H2S coverage
    # ------------------------------------------------------------------
    h2s_cov = (
        df2.groupby("field", observed=True)
        .apply(lambda g: pd.Series({
            "n_runs": len(g),
            "pct_h2s_proxy": g["h2s_proxy_mg_l"].notna().mean() * 100,
            "pct_h2s_excel": g["h2s_class_excel"].isin(["Кислый", "Некислый"]).mean() * 100,
            "pct_h2s_any": g["h2s_label"].isin(["Кислый", "Некислый"]).mean() * 100,
            "median_h2s": g["h2s_proxy_mg_l"].median(),
        }), include_groups=False)
        .reset_index()
    )
    h2s_global = pd.Series({
        "field": "GLOBAL",
        "n_runs": len(df2),
        "pct_h2s_proxy": df2["h2s_proxy_mg_l"].notna().mean() * 100,
        "pct_h2s_excel": df2["h2s_class_excel"].isin(["Кислый", "Некислый"]).mean() * 100,
        "pct_h2s_any": df2["h2s_label"].isin(["Кислый", "Некислый"]).mean() * 100,
        "median_h2s": df2["h2s_proxy_mg_l"].median(),
    })
    h2s_cov = pd.concat([pd.DataFrame([h2s_global]), h2s_cov], ignore_index=True)
    h2s_cov.to_csv(out / "s0_h2s_coverage.csv", index=False, encoding="utf-8-sig")
    results["h2s_coverage"] = h2s_cov

    # ------------------------------------------------------------------
    # 5. Ion-proxy coverage and pairwise correlations
    # ------------------------------------------------------------------
    ion_cols = ["cum_calcium_load_kg", "cum_chloride_load_kg", "cum_sulfate_load_kg",
                "cum_salt_load_kg", "cum_gypsum_scale_proxy"]
    ion_cov = pd.DataFrame({
        "column": ion_cols,
        "pct_not_null": [df2[c].notna().mean() * 100 if c in df2.columns else 0.0 for c in ion_cols],
    })
    ion_cov.to_csv(out / "s0_ion_coverage.csv", index=False, encoding="utf-8-sig")
    results["ion_coverage"] = ion_cov

    avail_ions = [c for c in ion_cols if c in df2.columns and df2[c].notna().mean() > 0.3]
    if avail_ions:
        corr = df2[avail_ions].corr(method="spearman").round(3)
        corr.to_csv(out / "s0_ion_correlations.csv", encoding="utf-8-sig")
        results["ion_correlations"] = corr

    # ------------------------------------------------------------------
    # 6. Duty metric coverage
    # ------------------------------------------------------------------
    duty_cov = pd.DataFrame({
        "metric": ["total_freq_hz_days", "total_liquid_m3", "trf_per_day", "tlf_per_day"],
        "pct_not_null": [
            df2["total_freq_hz_days"].notna().mean() * 100,
            df2["total_liquid_m3"].notna().mean() * 100,
            df2["trf_per_day"].notna().mean() * 100,
            df2["tlf_per_day"].notna().mean() * 100,
        ],
        "n_zero_or_near_zero": [
            (df2["total_freq_hz_days"].fillna(0) < 1).sum(),
            (df2["total_liquid_m3"].fillna(0) < 1).sum(),
            0, 0,
        ],
    })
    duty_cov.to_csv(out / "s0_duty_coverage.csv", index=False, encoding="utf-8-sig")
    results["duty_coverage"] = duty_cov

    # ------------------------------------------------------------------
    # 7. Mount-year × frequency trend (temporal confounding check)
    # ------------------------------------------------------------------
    yr_freq = (
        df2.groupby(["field", "mount_year"], observed=True)["freq_w_mean"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_freq_hz", "count": "n_runs"})
    )
    yr_freq.to_csv(out / "s0_year_freq_trend.csv", index=False, encoding="utf-8-sig")
    results["year_freq_trend"] = yr_freq

    # ------------------------------------------------------------------
    # 8. Multiple-run wells
    # ------------------------------------------------------------------
    runs_per_well = df2.groupby("well_key")["row_id"].count()
    n_multi = (runs_per_well > 1).sum()
    total_wells = len(runs_per_well)
    results["multi_run_wells"] = {
        "n_wells": total_wells,
        "n_multi_run": int(n_multi),
        "pct_multi_run": float(n_multi / total_wells * 100),
        "max_runs_per_well": int(runs_per_well.max()),
    }

    # ------------------------------------------------------------------
    # 9. Subgroup feasibility summary
    # ------------------------------------------------------------------
    feasibility_rows = []
    for field in MAJOR_FIELDS + ["GLOBAL"]:
        sub = df2 if field == "GLOBAL" else df2[df2["field"] == field]
        for grp_label, mask in [
            ("All", sub["event"] >= 0),
            ("High freq", sub["freq_group"] == "High (>55 Hz)"),
            ("Low freq", sub["freq_group"] == "Low (≤50 Hz)"),
        ]:
            g = sub[mask]
            nf = int(g["event"].sum())
            feasibility_rows.append({
                "field": field,
                "group": grp_label,
                "n_runs": len(g),
                "n_fail": nf,
                "km_ok": nf >= MIN_FAILURES_KM,
                "weibull_ok": nf >= MIN_FAILURES_WEIBULL,
            })
    feasibility = pd.DataFrame(feasibility_rows)
    feasibility.to_csv(out / "s0_feasibility.csv", index=False, encoding="utf-8-sig")
    results["feasibility"] = feasibility

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 0 — DATA AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total runs: {len(df2):,}  |  Total failures: {int(df2['event'].sum()):,}")
    print(f"Very-high freq (>58 Hz) global n_fail={vh_n} — "
          f"{'VIABLE for Weibull/Cox' if results['very_high_60hz_viable'] else 'TOO SPARSE, drop 60 Hz framing'}")
    vt_rows = df2[df2["field"] == VT_FIELD]
    print(f"Vt: {len(vt_rows)} runs, {int(vt_rows['event'].sum())} failures")
    print(f"  avg freq_w_mean: {vt_rows['freq_w_mean'].mean():.1f} Hz  "
          f"  median H2S: {vt_rows['h2s_proxy_mg_l'].median():.1f} mg/L")
    print(f"Multiple-run wells: {n_multi}/{total_wells} ({n_multi/total_wells*100:.0f}%)")
    print(f"Telemetry (≥7 freq days) global: {(df2['n_freq_valid_days']>=7).mean()*100:.0f}%")
    print(f"H2S proxy global coverage: {df2['h2s_proxy_mg_l'].notna().mean()*100:.0f}%")
    print(f"Outputs → {out}")

    return results
