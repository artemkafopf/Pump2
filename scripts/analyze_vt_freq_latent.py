#!/usr/bin/env python3
"""
Run A re-fit for Vt + frequency stratification analysis.

Questions:
  a) mean_freq > 55 Hz vs <= 55 Hz
  b) pct_days_above_55hz > 50% vs <= 50%

Uses mart data (ttf_true_best_days) so sample sizes match the reported Run A.
Saves latent probabilities, KM tables, and group summaries.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.bayesian_latent_weibull import (
from analysis.paths import results_dir
    BayesianLatentWeibullConfig,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
)

DB_PATH = REPO_ROOT / "data" / "warehouse" / "pump2.db"
_SLUG = "vt_freq_latent"
OUT_DIR = results_dir(_SLUG)
MIN_FREQ_DAYS = 10          # minimum valid freq days to include in group comparisons

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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_vt() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT row_id, ttf_true_best_days, event, "
        "freq_w_mean, freq_above_55hz_pct, freq_signed_exposure, "
        "n_freq_valid_days, n_freq_above_55hz "
        "FROM mart__vt_freq55 WHERE field = 'Vt'",
        conn,
    )
    conn.close()
    df = df[df.ttf_true_best_days.notna() & (df.ttf_true_best_days > 0)].copy()
    df["event"] = df["event"].astype(int)
    return df


# ---------------------------------------------------------------------------
# KM utilities
# ---------------------------------------------------------------------------

def km_at(df: pd.DataFrame, time_points: list[float]) -> dict:
    km = kaplan_meier_frame(df["ttf_true_best_days"].values, df["event"].values)
    result = {}
    for tp in time_points:
        row = km[km["time"] <= tp]
        result[f"S({tp:.0f}d)"] = float(row["survival"].iloc[-1]) if not row.empty else 1.0
    for q, label in [(0.5, "B50"), (0.25, "B75"), (0.1, "B90")]:
        subset = km[km["survival"] <= q]
        result[label] = float(subset["time"].iloc[0]) if not subset.empty else float("nan")
    return result


def log_rank_test(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Wilcoxon-Gehan log-rank approximation via Mann-Whitney on TTF."""
    _, p = mannwhitneyu(df1["ttf_true_best_days"], df2["ttf_true_best_days"], alternative="two-sided")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    vt = load_vt()
    print(f"Vt: n={len(vt)}, failures={vt.event.sum()} ({vt.event.mean():.1%})")
    print(f"  median TTF={vt.ttf_true_best_days.median():.0f}d, "
          f"freq_w_mean valid={vt.freq_w_mean.notna().sum()}, "
          f"freq_pct valid={vt.freq_above_55hz_pct.notna().sum()}")

    # --- Fit Run A ---
    print("\nFitting Run A (mart Vt, K=2) ...")
    result = fit_bayesian_latent_weibull(
        vt[["row_id", "ttf_true_best_days", "event"]].rename(columns={"row_id": "id"}),
        duration_column="ttf_true_best_days",
        event_column="event",
        id_column="id",
        config=MCMC_CFG,
    )

    # Save diagnostics
    diag = result.diagnostics_frame()
    ps   = result.posterior_summary_frame()
    pq   = result.posterior_life_quantiles_frame()
    ps.to_csv(OUT_DIR / "posterior_summary.csv", index=False)
    pq.to_csv(OUT_DIR / "posterior_life_quantiles.csv", index=False)

    print("\n  Convergence:")
    for _, row in diag.iterrows():
        print(f"    Chain {int(row['chain'])}: {int(row['saved_draws'])} draws  "
              f"acc_eta={row['acceptance_eta_mean']:.2f}  acc_beta={row['acceptance_beta_mean']:.2f}")

    comps = [c for c in sorted(ps["component"].unique()) if c != "all"]
    print("\n  Posterior summary:")
    for comp in comps:
        chunk = ps[ps["component"] == comp]
        params = {r["parameter"]: r for _, r in chunk.iterrows()}
        w  = params.get("weight", {})
        e  = params.get("eta", {})
        b  = params.get("beta", {})
        ess_w = w.get("ess", float("nan"))
        rhat_w = w.get("rhat", float("nan"))
        print(f"    {comp}: w={w.get('mean', float('nan')):.3f}  "
              f"eta={e.get('mean', float('nan')):.1f}d  "
              f"beta={b.get('mean', float('nan')):.3f}  "
              f"ESS_w={ess_w:.0f}  R-hat_w={rhat_w:.4f}")

    print("\n  Life quantiles (mixture):")
    for _, row in pq.iterrows():
        print(f"    {row['label']:>4s}: {row['mean']:.1f}d  "
              f"median={row['q50']:.1f}d  [95%CI {row['q2_5']:.1f}-{row['q97_5']:.1f}]")

    # --- Latent probabilities ---
    obs_probs = result.observation_posterior_probabilities()
    obs_probs = obs_probs.rename(columns={"id": "row_id"})
    merged = vt.merge(obs_probs[["row_id", "component_1_probability",
                                  "component_2_probability",
                                  "most_probable_component",
                                  "max_probability",
                                  "classification_entropy"]], on="row_id", how="left")
    merged.to_csv(OUT_DIR / "vt_latent_with_freq.csv", index=False)

    # --- Define frequency groups ---
    # Restrict comparisons to runs with sufficient freq coverage
    has_freq = merged["n_freq_valid_days"] >= MIN_FREQ_DAYS
    cov_n = has_freq.sum()
    print(f"\n  Runs with >={MIN_FREQ_DAYS} valid freq days: {cov_n} / {len(merged)}")

    TIME_POINTS = [90, 180, 365, 730]

    for q_label, col, threshold, description in [
        ("(a) mean_freq",   "freq_w_mean",          55.0,  "mean freq > 55 Hz"),
        ("(b) pct_above55", "freq_above_55hz_pct",   0.50,  ">50% days above 55 Hz"),
    ]:
        print(f"\n{'='*68}")
        print(f"  Frequency criterion {q_label}: {description}")
        print(f"{'='*68}")

        # Build groups: high / low / no-data
        base = merged[has_freq & merged[col].notna()].copy()
        hi  = base[base[col] >  threshold]
        lo  = base[base[col] <= threshold]
        na  = merged[~has_freq | merged[col].isna()]

        print(f"  High group  ({col} > {threshold}):  n={len(hi)}, "
              f"failures={hi.event.sum()} ({hi.event.mean():.1%}), "
              f"median_ttf={hi.ttf_true_best_days.median():.0f}d")
        print(f"  Low group   ({col} <= {threshold}): n={len(lo)}, "
              f"failures={lo.event.sum()} ({lo.event.mean():.1%}), "
              f"median_ttf={lo.ttf_true_best_days.median():.0f}d")
        print(f"  No-data:                          n={len(na)}")

        # KM at time points
        km_hi = km_at(hi, TIME_POINTS) if len(hi) >= 5 else {}
        km_lo = km_at(lo, TIME_POINTS) if len(lo) >= 5 else {}
        km_na = km_at(na, TIME_POINTS) if len(na) >= 5 else {}

        print(f"\n  Kaplan-Meier survival S(t):")
        hdr = f"  {'Time':>8}   {'High':>10}   {'Low':>10}   {'No-data':>10}"
        print(hdr)
        for tp in TIME_POINTS:
            key = f"S({tp:.0f}d)"
            h_val = f"{km_hi.get(key, float('nan')):.3f}" if km_hi else "   n/a"
            l_val = f"{km_lo.get(key, float('nan')):.3f}" if km_lo else "   n/a"
            n_val = f"{km_na.get(key, float('nan')):.3f}" if km_na else "   n/a"
            print(f"  {tp:>8}d   {h_val:>10}   {l_val:>10}   {n_val:>10}")

        for label in ["B50", "B75", "B90"]:
            h_val = f"{km_hi.get(label, float('nan')):.0f}d" if km_hi else "n/a"
            l_val = f"{km_lo.get(label, float('nan')):.0f}d" if km_lo else "n/a"
            n_val = f"{km_na.get(label, float('nan')):.0f}d" if km_na else "n/a"
            print(f"  {label:>8}:   {h_val:>10}   {l_val:>10}   {n_val:>10}")

        # Latent membership by group
        print(f"\n  P(component_1 = short-lived) by group:")
        for label, grp in [("High", hi), ("Low", lo), ("No-data", na)]:
            p = grp["component_1_probability"].dropna()
            if len(p) == 0:
                continue
            print(f"  {label:>10}: mean={p.mean():.3f}  median={p.median():.3f}  "
                  f"sd={p.std():.3f}  n={len(p)}")

        # Mann-Whitney on raw TTF (high vs low)
        if len(hi) >= 5 and len(lo) >= 5:
            p_mw = log_rank_test(hi, lo)
            print(f"\n  Mann-Whitney p-value (high vs low TTF): {p_mw:.4f}")
            if len(hi) < 30:
                print(f"  WARNING: high-freq group n={len(hi)} — low statistical power.")

        # Mann-Whitney on p_short (high vs low)
        if len(hi) >= 5 and len(lo) >= 5:
            p1_hi = hi["component_1_probability"].dropna()
            p1_lo = lo["component_1_probability"].dropna()
            if len(p1_hi) >= 3 and len(p1_lo) >= 3:
                _, p_mw2 = mannwhitneyu(p1_hi, p1_lo, alternative="two-sided")
                print(f"  Mann-Whitney p-value (high vs low P_short): {p_mw2:.4f}")

        # Save group table
        rows = []
        for label, grp in [("high", hi), ("low", lo), ("no_data", na)]:
            if len(grp) == 0:
                continue
            km_vals = km_at(grp, TIME_POINTS) if len(grp) >= 5 else {}
            p1 = grp["component_1_probability"].dropna()
            rows.append({
                "criterion": q_label, "group": label, "n": len(grp),
                "failures": int(grp.event.sum()), "failure_rate": grp.event.mean(),
                "median_ttf": grp.ttf_true_best_days.median(),
                "p_short_mean": p1.mean() if len(p1) else float("nan"),
                "p_short_sd": p1.std() if len(p1) else float("nan"),
                **km_vals,
            })
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"group_summary_{q_label.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            index=False)

    # --- Overall frequency vs p_short correlation (within runs with freq data) ---
    print(f"\n{'='*68}")
    print("  Spearman: frequency features vs P(component_1 = short-lived)")
    print(f"{'='*68}")
    from scipy.stats import spearmanr
    freq_feats = ["freq_w_mean", "freq_above_55hz_pct", "freq_signed_exposure"]
    for col in freq_feats:
        x = pd.to_numeric(merged[col], errors="coerce")
        y = merged["component_1_probability"]
        valid = x.notna() & y.notna() & (merged["n_freq_valid_days"] >= MIN_FREQ_DAYS)
        if valid.sum() < 20:
            continue
        rho, pval = spearmanr(x[valid], y[valid])
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {col:<30} rho={rho:+.3f}  p={pval:.4f}  n={valid.sum()} {sig}")

    # --- KM for all Vt (no freq split) as reference ---
    print(f"\n{'='*68}")
    print("  Overall Vt KM (all runs, Run A reference)")
    print(f"{'='*68}")
    km_all = km_at(vt, TIME_POINTS)
    for tp in TIME_POINTS:
        key = f"S({tp:.0f}d)"
        print(f"  S({tp}d) = {km_all.get(key, float('nan')):.3f}")
    for label in ["B50", "B75", "B90"]:
        print(f"  {label} = {km_all.get(label, float('nan')):.0f}d")

    print(f"\n  Outputs saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
