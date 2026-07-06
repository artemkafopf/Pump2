"""Phase 0 — Data audit: stratum viability, covariate coverage, pad feasibility."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CHEMISTRY_COLS,
    OPERATIONAL_IMPUTE_COLS,
    MIN_FAILURES_INDEPENDENT,
    MIN_FAILURES_TWO_STAGE,
    stratum_summary,
)

# All candidate covariates grouped by tier
TIER1_COLS = ["frequency", "motor_load", "water_cut", "glr", "p_bot", "p_bubble",
              "q_actual", "q_nominal"]
TIER2_COLS = CHEMISTRY_COLS
TIER3_COLS = ["execution_group", "corrosion_resistance", "esp_size", "tubing_id",
              "tubing_grade", "curvature", "setting_depth", "well_type",
              "n_stages", "current_noload", "current_nominal", "two_stage_sep"]
TIER4_COLS = ["opz_count", "sko_esp_count"]
ALL_CANDIDATE_COLS = TIER1_COLS + TIER2_COLS + TIER3_COLS + TIER4_COLS


def _coverage_pct(series: pd.Series) -> float:
    present = series.notna()
    if len(series) == 0:
        return 0.0
    return round(100.0 * present.sum() / len(series), 1)


def run(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Stratum summary ────────────────────────────────────────────────────
    strat = stratum_summary(df)
    strat.to_csv(out_dir / "stratum_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[Phase 0] Stratum summary:")
    print(strat.to_string(index=False))

    # ── 2. Per-stratum × covariate coverage ──────────────────────────────────
    # Use pre-imputation indicator columns where available (_imputed == 0 means originally present)
    cov_rows = []
    available_cols = [c for c in ALL_CANDIDATE_COLS if c in df.columns]
    for stratum, g in df.groupby("stratum"):
        for col in available_cols:
            imputed_col = f"{col}_imputed"
            if imputed_col in df.columns:
                # 1 = was missing originally; 0 = originally present
                orig_present = (g[imputed_col] == 0).sum()
                cov_pct = round(100.0 * orig_present / max(len(g), 1), 1)
            else:
                orig_present = int(g[col].notna().sum())
                cov_pct = _coverage_pct(g[col])
            cov_rows.append({
                "stratum": stratum,
                "covariate": col,
                "n_total": len(g),
                "n_originally_present": int(orig_present),
                "coverage_pct": cov_pct,
            })
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(out_dir / "covariate_coverage.csv", index=False, encoding="utf-8-sig")

    # Print compact pivot for key columns
    key_cols = ["frequency", "motor_load", "water_cut", "glr", "p_bot",
                "ph", "kvch", "h2s_conc", "cl"]
    key_cov = cov_df[cov_df["covariate"].isin([c for c in key_cols if c in available_cols])]
    pivot = key_cov.pivot(index="stratum", columns="covariate", values="coverage_pct")
    print("\n[Phase 0] Key covariate coverage (%):")
    print(pivot.to_string())

    # ── 3. Pad feasibility audit ─────────────────────────────────────────────
    pad_rows = []
    for (field, pad), g in df.groupby(["field_clean", "pad_key"]):
        n_wells = g["well_key"].nunique()
        for col in CHEMISTRY_COLS:
            if col not in df.columns:
                continue
            vals = g[col].dropna()
            pad_rows.append({
                "field": field,
                "pad_key": pad,
                "covariate": col,
                "n_wells_on_pad": n_wells,
                "n_runs": len(g),
                "n_measured": len(vals),
                "cv_pct": round(100.0 * vals.std() / vals.mean(), 1) if len(vals) > 1 and vals.mean() != 0 else None,
                "single_well_pad": n_wells == 1,
            })
    pad_df = pd.DataFrame(pad_rows)
    pad_df.to_csv(out_dir / "pad_feasibility.csv", index=False, encoding="utf-8-sig")

    n_single_well_pads = pad_df[pad_df["covariate"] == "cl"]["single_well_pad"].sum()
    n_total_pads = pad_df[pad_df["covariate"] == "cl"].shape[0]
    print(f"\n[Phase 0] Single-well pads (no within-pad averaging): "
          f"{int(n_single_well_pads)}/{n_total_pads} "
          f"({100*n_single_well_pads/max(n_total_pads,1):.0f}%)")

    # ── 4. Contractor × field audit ──────────────────────────────────────────
    contr_rows = []
    for (field, contractor), g in df.groupby(["field_clean", "contractor"]):
        n_fail = int(g["event"].sum())
        b50 = g.loc[g["event"] == 1, "tte"].median()
        contr_rows.append({
            "field": field,
            "contractor": contractor,
            "n_runs": len(g),
            "n_failures": n_fail,
            "empirical_B50": round(b50, 1) if not np.isnan(b50) else None,
            "sparse": n_fail < 10,
        })
    contr_df = pd.DataFrame(contr_rows).sort_values(
        ["field", "n_failures"], ascending=[True, False]
    )
    contr_df.to_csv(out_dir / "contractor_field_audit.csv", index=False, encoding="utf-8-sig")

    print("\n[Phase 0] Contractor × field (n_failures):")
    pivot_c = contr_df.pivot(index="contractor", columns="field", values="n_failures").fillna(0).astype(int)
    print(pivot_c.to_string())

    # ── 5. Vt sour detailed breakdown ────────────────────────────────────────
    vt_sour = df[df["stratum"] == "Vt_sour"]
    if not vt_sour.empty:
        print(f"\n[Phase 0] Vt sour (n={len(vt_sour)}):")
        print(f"  Failures: {int(vt_sour['event'].sum())} | "
              f"Censored: {int((vt_sour['event']==0).sum())}")
        print(f"  H2S cause: {int(vt_sour['is_h2s_cause'].sum())} / {len(vt_sour)} "
              f"({100*vt_sour['is_h2s_cause'].mean():.0f}%)")
        print(f"  Median TTE: {vt_sour['tte'].median():.0f} days")

        # Failure node distribution
        node_counts = vt_sour["failure_node"].value_counts().head(6)
        print("  Top failure nodes:")
        for node, cnt in node_counts.items():
            print(f"    {node}: {cnt} ({100*cnt/len(vt_sour):.0f}%)")

        vt_sour_contr = (
            vt_sour.groupby("contractor")
            .agg(n=("tte", "count"), median_tte=("tte", "median"),
                 n_fail=("event", "sum"))
            .reset_index()
        )
        print("  Contractor breakdown:")
        print(vt_sour_contr.to_string(index=False))

    return {
        "strat_summary": strat,
        "cov_coverage": cov_df,
        "pad_feasibility": pad_df,
        "contractor_field": contr_df,
    }
