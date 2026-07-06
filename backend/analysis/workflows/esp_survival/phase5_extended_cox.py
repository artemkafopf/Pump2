"""Phase 5 — Covariate exploration and Extended Cox model.

Steps:
  5a. Spearman correlation screening — flag |ρ| > 0.65 pairs
  5b. Univariate Cox with each Tier 1+2 candidate
  5c. Multivariate Extended Cox (lifelines CoxPHFitter)
      - Standard Cox terms for static covariates
      - Extended terms X + X:log(t) for PH violators
      - cluster_col='well_key' for robust SE
  5d. Schoenfeld test + VIF check
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# VIF requires statsmodels
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# Candidate covariates (by tier) — use log-transformed versions where noted
TIER1_CANDIDATES = [
    "frequency",
    "motor_load",
    "water_cut",
    "delta_bep",
    "glr",
    "p_bot",
    "p_bubble",
]
TIER2_CANDIDATES = [
    "log_cl",
    "log_so4",
    "log_ca",
    "log_hco3",
    "log_kvch",
    "log_h2s_conc",
    "log_ca_so4",
    "ph",
]
TIER3_CANDIDATES = [
    "curvature",
    "setting_depth",
    "n_stages_ratio",
    "current_ratio",
    "two_stage_sep",
]

ALL_CONTINUOUS = TIER1_CANDIDATES + TIER2_CANDIDATES + TIER3_CANDIDATES

# Extended terms (X·log(t)) for operational covariates
EXTENDED_CANDIDATES = ["frequency", "motor_load"]

# Final VBA-ready model: covariates that pass PH test AND are physically interpretable.
# motor_load excluded — Schoenfeld p=2e-20 (massive PH violation), VIF=31 when paired
# with its x_logt term. Covered by stratum structure instead.
FINAL_MODEL_COVARIATES = [
    "delta_bep",      # BEP deviation — primary operational stress, PH OK, p=0.000
    "p_bot",          # bottomhole pressure, PH OK, p=0.013, 88-91% coverage
    "n_stages_ratio", # stages/nominal_flow — equipment stress proxy, PH OK, p=0.000
    # Excluded: water_cut (p=0.648 in multivariate), glr (Schoenfeld violation),
    #           motor_load (Schoenfeld p=2e-20, VIF=31 with x_logt term)
]

SPEARMAN_THRESHOLD = 0.65
UNIVARIATE_P_THRESHOLD = 0.10
VIF_THRESHOLD = 5.0


def _prepare_cox_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise continuous covariates (zero mean, unit std) for numeric stability."""
    cdf = df.copy()
    for col in ALL_CONTINUOUS:
        if col in cdf.columns:
            vals = pd.to_numeric(cdf[col], errors="coerce")
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                cdf[f"{col}_std"] = (vals - mu) / sigma
    return cdf


def run(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use only failed + censored runs with valid stratum
    analysis = df[df["tte"] > 0].copy()
    analysis["log_tte"] = np.log(analysis["tte"].clip(lower=1e-6))
    analysis["stratum_code"] = analysis["stratum"]

    # Add extended (X·log(t)) terms to the DataFrame
    for col in EXTENDED_CANDIDATES:
        if col in analysis.columns:
            analysis[f"{col}_x_logt"] = analysis[col] * analysis["log_tte"]

    cdf = _prepare_cox_df(analysis)

    # ── 5a. Spearman correlation screening ────────────────────────────────────
    avail = [c for c in ALL_CONTINUOUS if c in cdf.columns]
    corr_mat = cdf[avail].corr(method="spearman", numeric_only=True)
    corr_mat.to_csv(out_dir / "phase5a_spearman_corr.csv", encoding="utf-8-sig")

    flagged_pairs = []
    for i, ci in enumerate(avail):
        for j, cj in enumerate(avail):
            if j <= i:
                continue
            rho = corr_mat.loc[ci, cj]
            if abs(rho) > SPEARMAN_THRESHOLD:
                flagged_pairs.append({"var1": ci, "var2": cj, "spearman_rho": round(rho, 3)})
    flag_df = pd.DataFrame(flagged_pairs)
    flag_df.to_csv(out_dir / "phase5a_high_corr_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"\n[Phase 5a] High-correlation pairs (|ρ| > {SPEARMAN_THRESHOLD}):")
    for _, row in flag_df.iterrows():
        print(f"  {row['var1']} ↔ {row['var2']}: ρ={row['spearman_rho']}")

    # ── 5b. Univariate Cox ────────────────────────────────────────────────────
    uni_rows = []
    for col in avail:
        sub = cdf[["tte", "event", "stratum_code", col, "well_key"]].dropna()
        if len(sub) < 30 or int(sub["event"].sum()) < 10:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(
                sub,
                duration_col="tte",
                event_col="event",
                strata=["stratum_code"],
                formula=col,
                cluster_col="well_key",
                robust=True,
            )
            s = cph.summary
            hr = float(np.exp(s.loc[col, "coef"]))
            p = float(s.loc[col, "p"])
            ci_lo = float(np.exp(s.loc[col, "coef lower 95%"]))
            ci_hi = float(np.exp(s.loc[col, "coef upper 95%"]))
            c_idx = float(cph.concordance_index_)
            uni_rows.append({
                "covariate": col, "hr": round(hr, 4), "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4), "p_value": round(p, 4),
                "c_index": round(c_idx, 4),
                "significant_univariate": p < UNIVARIATE_P_THRESHOLD,
            })
        except Exception as exc:
            uni_rows.append({"covariate": col, "note": str(exc)[:80]})

    uni_df = pd.DataFrame(uni_rows)
    uni_df.to_csv(out_dir / "phase5b_univariate_cox.csv", index=False, encoding="utf-8-sig")

    sig_cols = uni_df.loc[uni_df.get("significant_univariate", False) == True, "covariate"].tolist()
    print(f"\n[Phase 5b] Univariate significant (p < {UNIVARIATE_P_THRESHOLD}): {sig_cols}")

    # ── 5c. Multivariate Extended Cox ─────────────────────────────────────────
    # Keep significant univariate terms; add X·log(t) for extended candidates
    formula_parts = [c for c in sig_cols if c in cdf.columns]

    # Add contractor as categorical
    if "contractor" in cdf.columns:
        viable_contractors = (
            cdf.groupby("contractor")["event"].sum()
        )
        viable_contractors = viable_contractors[viable_contractors >= 10].index.tolist()
        if len(viable_contractors) >= 2:
            ref = viable_contractors[0]
            raw_codes = pd.Categorical(
                cdf["contractor"].where(cdf["contractor"].isin(viable_contractors)),
                categories=viable_contractors,
            ).codes.astype("float")
            cdf["contractor_coded"] = np.where(raw_codes == -1.0, np.nan, raw_codes)
            formula_parts.append("contractor_coded")

    # Add extended X·log(t) terms for operational covariates
    for col in EXTENDED_CANDIDATES:
        ext_col = f"{col}_x_logt"
        if col in sig_cols and ext_col in cdf.columns:
            formula_parts.append(ext_col)

    if not formula_parts:
        print("\n[Phase 5c] No significant univariate terms found — skipping multivariate.")
        return {"univariate": uni_df, "multivariate": pd.DataFrame(), "vif": pd.DataFrame()}

    formula_parts_clean = list(dict.fromkeys(formula_parts))  # deduplicate
    multi_cols = ["tte", "event", "stratum_code", "well_key"] + formula_parts_clean
    multi_df = cdf[[c for c in multi_cols if c in cdf.columns]].dropna()

    print(f"\n[Phase 5c] Multivariate Cox formula: {' + '.join(formula_parts_clean)}")
    print(f"  Dataset: {len(multi_df)} rows, {int(multi_df['event'].sum())} failures")

    multi_result_df = pd.DataFrame()
    schoenfeld_df = pd.DataFrame()
    try:
        cph_multi = CoxPHFitter()
        cph_multi.fit(
            multi_df,
            duration_col="tte",
            event_col="event",
            strata=["stratum_code"],
            formula=" + ".join(formula_parts_clean),
            cluster_col="well_key",
            robust=True,
        )
        multi_result_df = cph_multi.summary.reset_index()
        multi_result_df.to_csv(out_dir / "phase5c_multivariate_cox.csv",
                               index=False, encoding="utf-8-sig")

        # Schoenfeld test
        try:
            sch = proportional_hazard_test(cph_multi, multi_df, time_transform="log")
            schoenfeld_df = sch.summary.reset_index()
            schoenfeld_df.to_csv(out_dir / "phase5c_schoenfeld_test.csv",
                                 index=False, encoding="utf-8-sig")
            # Column name varies by lifelines version: 'covariate' or index name
            name_col = schoenfeld_df.columns[0]
            violations = schoenfeld_df[schoenfeld_df["p"] < 0.05]
            if violations.empty:
                print("[Phase 5c] Schoenfeld: no PH violations detected.")
            else:
                print(f"[Phase 5c] Schoenfeld: PH violations — {violations[name_col].tolist()}")
        except Exception as exc:
            print(f"[Phase 5c] Schoenfeld test failed: {exc}")

        print(f"[Phase 5c] Concordance: {cph_multi.concordance_index_:.4f}")

    except Exception as exc:
        print(f"[Phase 5c] Multivariate fit failed: {exc}")

    # ── 5d. VIF check ─────────────────────────────────────────────────────────
    vif_df = pd.DataFrame()
    if HAS_STATSMODELS and not multi_df.empty:
        try:
            X_vif = multi_df[[c for c in formula_parts_clean if c in multi_df.columns]].dropna()
            from statsmodels.tools.tools import add_constant
            X_vif_const = add_constant(X_vif, has_constant="add")
            vif_rows = []
            for i, col in enumerate(X_vif.columns):
                vif = float(variance_inflation_factor(X_vif_const.values, i + 1))
                vif_rows.append({"covariate": col, "vif": round(vif, 3)})
            vif_df = pd.DataFrame(vif_rows)
            vif_df.to_csv(out_dir / "phase5d_vif.csv", index=False, encoding="utf-8-sig")
            high_vif = vif_df[vif_df["vif"] > VIF_THRESHOLD]
            if high_vif.empty:
                print(f"[Phase 5d] VIF: all covariates < {VIF_THRESHOLD}.")
            else:
                print(f"[Phase 5d] VIF > {VIF_THRESHOLD}: {high_vif['covariate'].tolist()}")
        except Exception as exc:
            print(f"[Phase 5d] VIF computation failed: {exc}")

    # ── Forest plot of HRs ────────────────────────────────────────────────────
    if not multi_result_df.empty:
        try:
            col_hr = "exp(coef)"
            col_lo = "exp(coef) lower 95%"
            col_hi = "exp(coef) upper 95%"
            col_name = multi_result_df.columns[0]  # first column is covariate name
            plot_df = multi_result_df[
                [col_name, col_hr, col_lo, col_hi, "p"]
            ].copy()
            plot_df = plot_df.dropna()
            fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.4)))
            y_pos = range(len(plot_df))
            ax.barh(list(y_pos), np.log(plot_df[col_hr]),
                    xerr=[np.log(plot_df[col_hr]) - np.log(plot_df[col_lo]),
                          np.log(plot_df[col_hi]) - np.log(plot_df[col_hr])],
                    height=0.5, color=[
                        "crimson" if p < 0.05 else "steelblue"
                        for p in plot_df["p"]
                    ],
                    capsize=3)
            ax.axvline(0, ls="--", color="k", lw=1)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(plot_df[col_name], fontsize=8)
            ax.set_xlabel("log(HR)")
            ax.set_title("Phase 5 — Multivariate Cox HR (red = p < 0.05)")
            plt.tight_layout()
            fig.savefig(out_dir / "phase5c_forest_plot.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[Phase 5c] Forest plot failed: {exc}")

    # ── 5e. Final VBA-ready Cox model ─────────────────────────────────────────
    # Trimmed to covariates that pass PH + have physical interpretation.
    # motor_load excluded (Schoenfeld p=2e-20, VIF=31 with x_logt interaction).
    final_cols = [c for c in FINAL_MODEL_COVARIATES if c in cdf.columns]
    final_model_df = pd.DataFrame()
    cox_export_df = pd.DataFrame()

    print(f"\n[Phase 5e] Final VBA model covariates: {final_cols}")
    final_data_cols = ["tte", "event", "stratum_code", "well_key"] + final_cols
    final_data = cdf[[c for c in final_data_cols if c in cdf.columns]].dropna()
    print(f"  Dataset: {len(final_data)} rows, {int(final_data['event'].sum())} failures")

    try:
        cph_final = CoxPHFitter()
        cph_final.fit(
            final_data,
            duration_col="tte",
            event_col="event",
            strata=["stratum_code"],
            formula=" + ".join(final_cols),
            cluster_col="well_key",
            robust=True,
        )
        final_model_df = cph_final.summary.reset_index()
        final_model_df.to_csv(out_dir / "phase5e_final_cox.csv",
                              index=False, encoding="utf-8-sig")

        # Schoenfeld on final model
        try:
            sch_final = proportional_hazard_test(cph_final, final_data, time_transform="log")
            sch_final.summary.reset_index().to_csv(
                out_dir / "phase5e_schoenfeld_final.csv", index=False, encoding="utf-8-sig"
            )
            violations_final = sch_final.summary[sch_final.summary["p"] < 0.05]
            if violations_final.empty:
                print("[Phase 5e] Schoenfeld: no PH violations in final model.")
            else:
                print(f"[Phase 5e] Schoenfeld violations: {violations_final.index.tolist()}")
        except Exception as exc:
            print(f"[Phase 5e] Schoenfeld failed: {exc}")

        print(f"[Phase 5e] Concordance: {cph_final.concordance_index_:.4f}")

        # Export clean coefficient file for VBA import
        name_col = final_model_df.columns[0]
        cox_export_df = pd.DataFrame({
            "covariate": final_model_df[name_col],
            "beta": final_model_df["coef"].round(6),
            "hr": final_model_df["exp(coef)"].round(4),
            "hr_lo95": final_model_df["exp(coef) lower 95%"].round(4),
            "hr_hi95": final_model_df["exp(coef) upper 95%"].round(4),
            "p": final_model_df["p"].round(4),
        })
        cox_export_df.to_csv(out_dir / "cox_coefficients.csv",
                             index=False, encoding="utf-8-sig")

        print("\n[Phase 5e] VBA-ready coefficients:")
        print(cox_export_df[["covariate", "beta", "hr", "p"]].to_string(index=False))

    except Exception as exc:
        print(f"[Phase 5e] Final model fit failed: {exc}")

    print(f"\n[Phase 5] Complete.")
    return {
        "spearman": flag_df,
        "univariate": uni_df,
        "multivariate": multi_result_df,
        "schoenfeld": schoenfeld_df,
        "vif": vif_df,
        "final_model": final_model_df,
        "cox_export": cox_export_df,
    }
