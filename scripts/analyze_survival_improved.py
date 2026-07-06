#!/usr/bin/env python3
"""
Improved Bayesian survival analysis: Global / Ya / Vt.

Improvements over the Codex analysis:
  1. Uses ttf_true_best_days from mart__vt_freq55 (telemetry-corrected TTF).
  2. Proper MCMC: 12,000 iter x 3 chains -> ~1,800 effective draws.
  3. Soft-probability regression: uses P(comp_1) as a continuous [0,1] target
     rather than a hard binary threshold.
  4. Unweighted regression (no max_probability bias).
  5. K-fold cross-validation of the mount_year x gypsum interaction.
  6. Contractor-stratified mount_year effect to test confounding.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
)
from analysis.regularized_logistic import (
    RegularizedLogisticConfig,
    fit_regularized_logistic_model,
)
from analysis.paths import results_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH   = REPO_ROOT / "data" / "warehouse" / "pump2.db"
TABLE     = "mart__vt_freq55"
DUR_COL   = "ttf_true_best_days"
EVT_COL   = "event"
_PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
FEAT_CSV  = _PROMPT_DIR / "tables" / "analysis_dataset.csv"
_SLUG = "survival_improved"
OUT_DIR = results_dir(_SLUG)

MCMC_CFG = BayesianLatentWeibullConfig(
    n_components=2,
    n_iter=12_000,
    burn_in=3_000,
    thin=5,
    n_chains=3,
    mu_log_eta=5.0,
    sigma_log_eta=1.5,
    mu_log_beta=0.0,
    sigma_log_beta=0.7,
    random_seed=42,
)

NUMERIC_FEATURES = [
    "mount_year",
    "log1p_h2s_effective_mg_l",
    "avg_kpod",
    "freq_above_55hz_pct",
    "freq_below_45hz_pct",
    "freq_signed_exposure",
    "freq_w_mean",
    "log1p_tlf_per_day",
    "log1p_calcium_load_per_day",
    "log1p_chloride_load_per_day",
    "log1p_sulfate_load_per_day",
    "log1p_gypsum_proxy_per_day",
    "low_h2s_flag_numeric",
]
CATEGORICAL_FEATURES = ["contractor", "pump_family"]

RIDGE_CFG = RegularizedLogisticConfig(
    l2_penalty=1.0,
    max_iter=300,
    standardize_numeric=True,
    add_missing_indicators=True,
)

# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    # The feature CSV already contains ttf_true_best_days, event, field, and
    # all covariates — it was built from the mart. Load it directly.
    df = pd.read_csv(FEAT_CSV, low_memory=False)

    df = df.dropna(subset=[DUR_COL, EVT_COL])
    df = df[df[DUR_COL] > 0].copy()
    df[EVT_COL] = df[EVT_COL].astype(int)

    # Log-transforms (create only if the raw column exists and log version absent)
    for raw, log_name in [
        ("h2s_effective_mg_l",    "log1p_h2s_effective_mg_l"),
        ("tlf_per_day",           "log1p_tlf_per_day"),
        ("calcium_load_per_day",  "log1p_calcium_load_per_day"),
        ("chloride_load_per_day", "log1p_chloride_load_per_day"),
        ("sulfate_load_per_day",  "log1p_sulfate_load_per_day"),
        ("gypsum_proxy_per_day",  "log1p_gypsum_proxy_per_day"),
    ]:
        if log_name not in df.columns:
            if raw in df.columns:
                df[log_name] = np.log1p(pd.to_numeric(df[raw], errors="coerce").clip(lower=0))
            else:
                df[log_name] = np.nan

    # h2s fallback chain
    if "h2s_effective_mg_l" not in df.columns:
        for src in ("direct_h2s_mg_l", "h2s_proxy_mg_l"):
            if src in df.columns:
                df["h2s_effective_mg_l"] = pd.to_numeric(df[src], errors="coerce")
                df["log1p_h2s_effective_mg_l"] = np.log1p(df["h2s_effective_mg_l"].clip(lower=0))
                break

    if "low_h2s_flag_numeric" not in df.columns:
        if "high_h2s_flag" in df.columns:
            df["low_h2s_flag_numeric"] = (
                pd.to_numeric(df["high_h2s_flag"], errors="coerce").fillna(0) == 0
            ).astype(float)
        else:
            df["low_h2s_flag_numeric"] = np.nan

    if "freq_signed_exposure" not in df.columns:
        if "freq_above_55hz_pct" in df.columns and "freq_below_45hz_pct" in df.columns:
            df["freq_signed_exposure"] = (
                pd.to_numeric(df["freq_above_55hz_pct"], errors="coerce") -
                pd.to_numeric(df["freq_below_45hz_pct"], errors="coerce")
            )

    return df


def make_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "Global": df.copy(),
        "Vt":     df[df["field"] == "Vt"].copy(),
        "Ya":     df[df["field"] == "Ya"].copy(),
    }


# ---------------------------------------------------------------------------
# Step 1: Bayesian survival fits
# ---------------------------------------------------------------------------

def fit_group(name: str, df: pd.DataFrame, out_dir: Path) -> dict:
    n_fail = int(df[EVT_COL].sum())
    print(f"\n[Survival] Fitting {name}: n={len(df)}, failures={n_fail} ...")
    t0 = time.perf_counter()
    result = fit_bayesian_latent_weibull(
        df[[DUR_COL, EVT_COL, "row_id"]].rename(columns={"row_id": "id"}),
        duration_column=DUR_COL,
        event_column=EVT_COL,
        id_column="id",
        config=MCMC_CFG,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")

    gdir = out_dir / name.lower()
    gdir.mkdir(parents=True, exist_ok=True)

    # Save latent probabilities keyed by row_id
    obs_probs = result.observation_posterior_probabilities()
    obs_probs = obs_probs.rename(columns={"id": "row_id"})
    obs_probs.to_csv(gdir / "observation_latent_probabilities.csv", index=False)

    result.posterior_summary_frame().to_csv(gdir / "posterior_summary.csv", index=False)
    result.posterior_life_quantiles_frame().to_csv(gdir / "posterior_life_quantiles.csv", index=False)

    # Print summary
    summary = result.posterior_summary_frame()
    quants  = result.posterior_life_quantiles_frame()
    diag    = result.diagnostics_frame()

    comps = [c for c in sorted(summary["component"].unique()) if c != "all"]
    print(f"\n  === {name} posterior (K=2, {n_fail} failures, {len(df)-n_fail} censored) ===")
    for comp in comps:
        chunk = summary[summary["component"] == comp]
        params = {r["parameter"]: r for _, r in chunk.iterrows()}
        eta  = params.get("eta", {})
        beta = params.get("beta", {})
        wt   = params.get("weight", {})
        print(f"  {comp}: w={wt.get('mean','?'):.3f} | eta={eta.get('mean','?'):.1f}d [{eta.get('q2_5','?'):.0f}-{eta.get('q97_5','?'):.0f}] | beta={beta.get('mean','?'):.3f} [{beta.get('q2_5','?'):.3f}-{beta.get('q97_5','?'):.3f}]")

    for _, row in quants.iterrows():
        print(f"  {row['label']:>4s}: {row['mean']:7.1f}d  [95%CI {row['q2_5']:.1f}-{row['q97_5']:.1f}]")

    # KM reference
    km = kaplan_meier_frame(df[DUR_COL].values, df[EVT_COL].values)
    for target_s, label in [(0.50, "B50"), (0.25, "B75"), (0.10, "B90")]:
        subset = km[km["survival"] <= target_s]
        t_km = subset["time"].iloc[0] if not subset.empty else float("nan")
        print(f"  KM {label}: {t_km:.0f}d")

    for _, row in diag.iterrows():
        print(f"  Chain {int(row['chain'])}: {int(row['saved_draws'])} draws  acc_eta={row['acceptance_eta_mean']:.2f}  acc_beta={row['acceptance_beta_mean']:.2f}")

    return {
        "name": name,
        "result": result,
        "obs_probs": obs_probs,
        "out_dir": gdir,
    }


# ---------------------------------------------------------------------------
# Step 2: Merge latent probabilities with features
# ---------------------------------------------------------------------------

def build_analysis_dataset(df_full: pd.DataFrame, fit_outputs: list[dict]) -> pd.DataFrame:
    """Join every group's p_short back onto the full feature table."""
    frames = []
    for fo in fit_outputs:
        probs = fo["obs_probs"].copy()
        probs["group"] = fo["name"]
        frames.append(probs)
    all_probs = pd.concat(frames, ignore_index=True)

    # p_short = P(component_1 | data)
    if "component_1_probability" in all_probs.columns:
        all_probs["p_short"] = all_probs["component_1_probability"]
    else:
        raise KeyError("component_1_probability column missing from latent probabilities")

    merged = all_probs.merge(
        df_full.drop(columns=[DUR_COL, EVT_COL], errors="ignore"),
        on="row_id",
        how="left",
        suffixes=("", "_feat"),
    )
    return merged


# ---------------------------------------------------------------------------
# Step 3: Soft-probability univariate associations
# ---------------------------------------------------------------------------

def spearman_associations(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Spearman rho between p_short and each numeric feature."""
    rows = []
    for feat in NUMERIC_FEATURES:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        y = df["p_short"]
        valid = x.notna() & y.notna()
        if valid.sum() < 20:
            continue
        rho, pval = spearmanr(x[valid], y[valid])
        rows.append({"group": group_name, "feature": feat, "n": int(valid.sum()),
                     "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Step 4: Soft-probability multivariate regression
# ---------------------------------------------------------------------------

def run_multivariate(df: pd.DataFrame, group_name: str, model_name: str,
                     extra_numeric: list[str] | None = None) -> dict:
    """Ridge logistic with p_short as CONTINUOUS target (fractional response).

    No sample weighting (fixes the max_probability bias from Codex).
    """
    numeric = list(NUMERIC_FEATURES) + (extra_numeric or [])
    result = fit_regularized_logistic_model(
        df,
        target_column="p_short",
        numeric_features=numeric,
        categorical_features=CATEGORICAL_FEATURES,
        sample_weight_column=None,      # UNWEIGHTED — key improvement
        id_column="row_id",
        config=RIDGE_CFG,
    )
    m = result.metrics
    return {
        "group": group_name,
        "model": model_name,
        "n": m["rows"],
        "positive_rate": m["positive_rate"],
        "converged": m["converged"],
        "pseudo_r2": m["mcfadden_pseudo_r2"],
        "auc": m["auc_weighted"],
        "brier": m["brier_score"],
        "result_obj": result,
    }


# ---------------------------------------------------------------------------
# Step 5: Cross-validate the interaction term
# ---------------------------------------------------------------------------

def cross_validate_interaction(df: pd.DataFrame, group_name: str, n_folds: int = 5) -> pd.DataFrame:
    """K-fold CV comparing model with vs without mount_year x gypsum interaction."""
    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    rows = []
    for fold_i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()

        for use_interaction in [False, True]:
            extra = ["mount_year_x_log1p_gypsum_proxy_per_day"] if use_interaction else []
            try:
                res = fit_regularized_logistic_model(
                    train_df,
                    target_column="p_short",
                    numeric_features=list(NUMERIC_FEATURES) + extra,
                    categorical_features=CATEGORICAL_FEATURES,
                    sample_weight_column=None,
                    config=RIDGE_CFG,
                )
                # Predict on test
                from analysis.regularized_logistic import design_matrix_from_coefficient_table
                X_test = design_matrix_from_coefficient_table(test_df, res.coefficient_table)
                beta   = res.coefficient_table["coefficient"].to_numpy()
                preds  = expit(X_test @ beta)
                y_test = pd.to_numeric(test_df["p_short"], errors="coerce").to_numpy()
                valid  = np.isfinite(preds) & np.isfinite(y_test)
                brier  = float(np.mean((preds[valid] - y_test[valid]) ** 2))
                # Rank correlation as AUC-proxy for continuous target
                from scipy.stats import spearmanr as _sp
                rho_test, _ = _sp(preds[valid], y_test[valid])
                rows.append({
                    "group": group_name, "fold": fold_i,
                    "interaction": use_interaction,
                    "brier": brier, "spearman_rho": rho_test,
                })
            except Exception as exc:
                rows.append({
                    "group": group_name, "fold": fold_i,
                    "interaction": use_interaction,
                    "brier": float("nan"), "spearman_rho": float("nan"),
                    "error": str(exc),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6: Contractor-stratified mount_year effect
# ---------------------------------------------------------------------------

def contractor_stratified_mount_year(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Spearman rho between mount_year and p_short within each contractor."""
    if "contractor" not in df.columns or "mount_year" not in df.columns:
        return pd.DataFrame()
    rows = []
    for ctr, sub in df.groupby("contractor"):
        x = pd.to_numeric(sub["mount_year"], errors="coerce")
        y = sub["p_short"]
        valid = x.notna() & y.notna()
        if valid.sum() < 15:
            continue
        rho, pval = spearmanr(x[valid], y[valid])
        rows.append({"group": group_name, "contractor": ctr, "n": int(valid.sum()),
                     "rho_mount_year_vs_pshort": rho, "p_value": pval})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_comparison(fit_outputs: list[dict]) -> None:
    print("\n\n" + "=" * 78)
    print("  SURVIVAL COMPARISON (properly converged MCMC, ttf_true_best_days)")
    print("=" * 78)
    header = f"{'Parameter':<35} {'Global':>13} {'Vt':>13} {'Ya':>13}"
    print(header)
    print("-" * 78)

    def _get(fo, comp_idx, param):
        s = fo["result"].posterior_summary_frame()
        comps = [c for c in sorted(s["component"].unique()) if c != "all"]
        if comp_idx >= len(comps):
            return "n/a"
        c = comps[comp_idx]
        row = s[(s["component"] == c) & (s["parameter"] == param)]
        if row.empty:
            return "n/a"
        return f"{row.iloc[0]['mean']:.3f}"

    def _quant(fo, label):
        q = fo["result"].posterior_life_quantiles_frame()
        r = q[q["label"] == label]
        return f"{r.iloc[0]['mean']:.1f}d" if not r.empty else "n/a"

    fmap = {fo["name"]: fo for fo in fit_outputs}
    grps = ["Global", "Vt", "Ya"]

    vr = {g: fmap[g]["result"].validation_report for g in grps}
    print(f"{'N (runs)':<35} " + " ".join(f"{vr[g].rows_after_validation:>13}" for g in grps))
    print(f"{'Failures':<35} " + " ".join(f"{vr[g].failure_count:>13}" for g in grps))
    print("-" * 78)

    for ci, label in [(0, "Component 1 (short-lived)"), (1, "Component 2 (long-lived)")]:
        print(f"{label}")
        for param, pname in [("weight", "  weight"), ("eta", "  eta (days)"), ("beta", "  beta (shape)")]:
            vals = " ".join(f"{_get(fmap[g], ci, param):>13}" for g in grps)
            print(f"{pname:<35} {vals}")
        print("-" * 78)

    for label in ["B10", "B25", "B50", "B90"]:
        vals = " ".join(f"{_quant(fmap[g], label):>13}" for g in grps)
        print(f"{'Mixture ' + label:<35} {vals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data ...")
    df_full = load_data()
    groups  = make_groups(df_full)
    for name, gdf in groups.items():
        print(f"  {name:6s}: n={len(gdf)}, failures={int(gdf[EVT_COL].sum())} ({gdf[EVT_COL].mean():.1%}), median_ttf={gdf[DUR_COL].median():.0f}d")

    # ---- Step 1: Bayesian survival fits ----
    print("\n" + "="*60)
    print(" STEP 1: BAYESIAN SURVIVAL FITS")
    print("="*60)
    fit_outputs = []
    for name, gdf in groups.items():
        fo = fit_group(name, gdf, OUT_DIR)
        fit_outputs.append(fo)

    print_comparison(fit_outputs)

    # ---- Step 2: Build feature-enriched analysis dataset ----
    print("\n\n" + "="*60)
    print(" STEP 2: MERGING LATENT PROBS WITH FEATURES")
    print("="*60)
    analysis_df = build_analysis_dataset(df_full, fit_outputs)
    print(f"  Analysis dataset: n={len(analysis_df)}, p_short: mean={analysis_df['p_short'].mean():.3f}, median={analysis_df['p_short'].median():.3f}")
    print(f"  Groups: {analysis_df['group'].value_counts().to_dict()}")

    # Add interaction terms
    if "mount_year" in analysis_df.columns and "log1p_gypsum_proxy_per_day" in analysis_df.columns:
        my = pd.to_numeric(analysis_df["mount_year"], errors="coerce")
        gyp = analysis_df["log1p_gypsum_proxy_per_day"]
        my_std = float(my.std()) or 1e-8
        my_z = (my - float(my.mean())) / my_std
        analysis_df["mount_year_x_log1p_gypsum_proxy_per_day"] = my_z * gyp

    analysis_df.to_csv(OUT_DIR / "analysis_dataset_with_pshort.csv", index=False)

    # ---- Step 3: Soft-probability univariate associations ----
    print("\n\n" + "="*60)
    print(" STEP 3: SOFT-PROBABILITY UNIVARIATE ASSOCIATIONS")
    print("="*60)
    all_assoc = []
    for group_name in ["Global", "Vt", "Ya"]:
        gdf = analysis_df[analysis_df["group"] == group_name].copy()
        assoc = spearman_associations(gdf, group_name)
        all_assoc.append(assoc)
        print(f"\n  {group_name} (n={len(gdf)}, mean_p_short={gdf['p_short'].mean():.3f}):")
        for _, row in assoc.iterrows():
            sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "  ")
            print(f"    {row['feature']:<38} rho={row['spearman_rho']:+.3f}  p={row['p_value']:.3e} {sig}")

    assoc_df = pd.concat(all_assoc, ignore_index=True)
    assoc_df.to_csv(OUT_DIR / "soft_spearman_associations.csv", index=False)

    # ---- Step 4: Multivariate soft-probability regression ----
    print("\n\n" + "="*60)
    print(" STEP 4: MULTIVARIATE RIDGE REGRESSION (soft p_short target)")
    print("="*60)
    mv_results = []
    for group_name in ["Global", "Vt", "Ya"]:
        gdf = analysis_df[analysis_df["group"] == group_name].copy()
        print(f"\n  {group_name} -- main effects:")
        r_base = run_multivariate(gdf, group_name, "base")
        print(f"    n={r_base['n']}  pseudo_R2={r_base['pseudo_r2']:.4f}  AUC={r_base['auc']:.4f}")
        mv_results.append(r_base)

        print(f"  {group_name} -- with interaction:")
        r_int = run_multivariate(gdf, group_name, "interaction",
                                 extra_numeric=["mount_year_x_log1p_gypsum_proxy_per_day"])
        print(f"    n={r_int['n']}  pseudo_R2={r_int['pseudo_r2']:.4f}  AUC={r_int['auc']:.4f}  delta_R2={r_int['pseudo_r2']-r_base['pseudo_r2']:+.4f}")
        mv_results.append(r_int)

        # Print top coefficients for Vt
        if group_name == "Vt":
            coef = r_int["result_obj"].coefficient_table
            coef_sorted = coef.sort_values("z_value", key=abs, ascending=False)
            print(f"\n  Top coefficients (Vt interaction model):")
            for _, row in coef_sorted.head(15).iterrows():
                sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "  ")
                print(f"    {row['term']:<45} coef={row['coefficient']:+.3f}  OR={row['odds_ratio']:.3f}  z={row['z_value']:+.2f}  p={row['p_value']:.3e} {sig}")

    # Save coefficient tables
    coef_frames = []
    for r in mv_results:
        ct = r["result_obj"].coefficient_table.copy()
        ct["group"] = r["group"]
        ct["model"] = r["model"]
        coef_frames.append(ct)
    pd.concat(coef_frames, ignore_index=True).to_csv(OUT_DIR / "multivariate_coefficients.csv", index=False)

    # ---- Step 5: Cross-validate interaction ----
    print("\n\n" + "="*60)
    print(" STEP 5: 5-FOLD CROSS-VALIDATION (mount_year x gypsum interaction)")
    print("="*60)
    cv_results = []
    for group_name in ["Global", "Vt", "Ya"]:
        gdf = analysis_df[analysis_df["group"] == group_name].copy()
        if len(gdf) < 50:
            print(f"  {group_name}: skipped (n={len(gdf)} < 50)")
            continue
        cv = cross_validate_interaction(gdf, group_name)
        cv_results.append(cv)

        for use_int in [False, True]:
            sub = cv[cv["interaction"] == use_int]
            label = "with interaction   " if use_int else "without interaction"
            print(f"  {group_name} {label}: "
                  f"mean_brier={sub['brier'].mean():.4f}  "
                  f"mean_rho={sub['spearman_rho'].mean():.4f}")

    if cv_results:
        pd.concat(cv_results, ignore_index=True).to_csv(OUT_DIR / "cv_interaction_results.csv", index=False)

    # ---- Step 6: Contractor-stratified mount_year ----
    print("\n\n" + "="*60)
    print(" STEP 6: CONTRACTOR-STRATIFIED mount_year EFFECT")
    print("="*60)
    strat_frames = []
    for group_name in ["Global", "Vt", "Ya"]:
        gdf = analysis_df[analysis_df["group"] == group_name].copy()
        strat = contractor_stratified_mount_year(gdf, group_name)
        strat_frames.append(strat)
        print(f"\n  {group_name}:")
        print(strat[["contractor","n","rho_mount_year_vs_pshort","p_value"]].to_string(index=False))

    if strat_frames:
        pd.concat(strat_frames, ignore_index=True).to_csv(OUT_DIR / "contractor_stratified_mount_year.csv", index=False)

    print(f"\n\nAll outputs -> {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
