#!/usr/bin/env python3
"""
Bayesian Weibull latent membership vs high-frequency operation.
Runs for a specified field (or 'Global' = all fields).

Usage:
  python analyze_freq_latent.py [--field Global|Vt|Ya|...]

Questions:
  a) mean_freq > 55 Hz
  b) pct_days_above_55hz > 50%
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
)
from analysis.paths import results_dir

DB_PATH = REPO_ROOT / "data" / "warehouse" / "pump2.db"
MIN_FREQ_DAYS = 10

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

TIME_POINTS = [90, 180, 365, 730]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_population(field: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT row_id, field, ttf_true_best_days, event, "
        "freq_w_mean, freq_above_55hz_pct, freq_signed_exposure, "
        "n_freq_valid_days, n_freq_above_55hz "
        "FROM mart__vt_freq55",
        conn,
    )
    conn.close()
    if field != "Global":
        df = df[df["field"] == field].copy()
    df = df[df.ttf_true_best_days.notna() & (df.ttf_true_best_days > 0)].copy()
    df["event"] = df["event"].astype(int)
    return df


# ---------------------------------------------------------------------------
# KM helpers
# ---------------------------------------------------------------------------

def km_stats(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {}
    km = kaplan_meier_frame(df["ttf_true_best_days"].values, df["event"].values)
    result = {}
    for tp in TIME_POINTS:
        row = km[km["time"] <= tp]
        result[f"S({tp}d)"] = float(row["survival"].iloc[-1]) if not row.empty else 1.0
    for surv_thresh, label in [(0.5, "B50"), (0.25, "B75"), (0.1, "B90")]:
        subset = km[km["survival"] <= surv_thresh]
        result[label] = float(subset["time"].iloc[0]) if not subset.empty else float("nan")
    return result


def mw_p(s1: pd.Series, s2: pd.Series) -> float:
    s1, s2 = s1.dropna(), s2.dropna()
    if len(s1) < 3 or len(s2) < 3:
        return float("nan")
    _, p = mannwhitneyu(s1, s2, alternative="two-sided")
    return float(p)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72

def print_group_block(label: str, hi: pd.DataFrame, lo: pd.DataFrame,
                      na: pd.DataFrame, col: str, threshold: float) -> list[dict]:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)

    def grp_line(name: str, grp: pd.DataFrame) -> None:
        print(f"  {name:<12}  n={len(grp):>4}  "
              f"failures={grp.event.sum():>3} ({grp.event.mean():.1%})  "
              f"median_ttf={grp.ttf_true_best_days.median():>5.0f}d")

    grp_line(f"> {threshold}", hi)
    grp_line(f"<= {threshold}", lo)
    grp_line("no telemetry", na)

    # KM table
    km_hi = km_stats(hi)
    km_lo = km_stats(lo)
    km_na = km_stats(na)

    print(f"\n  Kaplan-Meier survival:")
    print(f"  {'Time':>8}   {'High':>10}   {'Low':>10}   {'No-data':>10}")
    for tp in TIME_POINTS:
        key = f"S({tp}d)"
        print(f"  {tp:>7}d   "
              f"{km_hi.get(key, float('nan')):>10.3f}   "
              f"{km_lo.get(key, float('nan')):>10.3f}   "
              f"{km_na.get(key, float('nan')):>10.3f}")
    for bl in ["B50", "B75", "B90"]:
        hv = f"{km_hi[bl]:.0f}d" if bl in km_hi and not np.isnan(km_hi[bl]) else "n/a"
        lv = f"{km_lo[bl]:.0f}d" if bl in km_lo and not np.isnan(km_lo[bl]) else "n/a"
        nv = f"{km_na[bl]:.0f}d" if bl in km_na and not np.isnan(km_na[bl]) else "n/a"
        print(f"  {bl:>8}:   {hv:>10}   {lv:>10}   {nv:>10}")

    # Latent membership
    print(f"\n  P(component_1 = short-lived):")
    rows_out = []
    for name, grp in [("High", hi), ("Low", lo), ("No-data", na)]:
        p1 = grp["component_1_probability"].dropna()
        if len(p1) == 0:
            continue
        print(f"  {name:<12}  mean={p1.mean():.3f}  median={p1.median():.3f}  "
              f"sd={p1.std():.3f}  n={len(p1)}")
        km_v = km_stats(grp)
        rows_out.append({
            "criterion": label, "group": name.lower(),
            "n": len(grp), "failures": int(grp.event.sum()),
            "failure_rate": grp.event.mean(),
            "median_ttf": grp.ttf_true_best_days.median(),
            "p_short_mean": p1.mean(), "p_short_sd": p1.std(),
            **km_v,
        })

    # Tests
    ttf_p = mw_p(hi.ttf_true_best_days, lo.ttf_true_best_days)
    ps_p  = mw_p(hi.component_1_probability, lo.component_1_probability)
    print(f"\n  Mann-Whitney p (TTF high vs low):    {ttf_p:.4f}")
    print(f"  Mann-Whitney p (P_short high vs low): {ps_p:.4f}")
    if len(hi) < 50:
        print(f"  WARNING: high-freq group n={len(hi)} — low power.")

    return rows_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="Global")
    args = parser.parse_args()
    field = args.field

    out_dir = results_dir(f"freq_latent_{field.lower()}")

    # Load
    df = load_population(field)
    print(f"\nPopulation: {field}  n={len(df)}  failures={df.event.sum()} ({df.event.mean():.1%})")
    print(f"  median_ttf={df.ttf_true_best_days.median():.0f}d")
    has_freq = df.n_freq_valid_days >= MIN_FREQ_DAYS
    print(f"  runs with >={MIN_FREQ_DAYS} valid freq days: {has_freq.sum()} ({has_freq.mean():.1%})")
    sub = df[has_freq]
    print(f"  freq_w_mean > 55Hz:      {(sub.freq_w_mean > 55).sum():>4} ({(sub.freq_w_mean > 55).mean():.1%})")
    print(f"  pct_above_55 > 50%%:     {(sub.freq_above_55hz_pct > 0.5).sum():>4} ({(sub.freq_above_55hz_pct > 0.5).mean():.1%})")

    # Fit
    print(f"\nFitting K=2 Bayesian Weibull mixture ({field}) ...")
    result = fit_bayesian_latent_weibull(
        df[["row_id", "ttf_true_best_days", "event"]].rename(columns={"row_id": "id"}),
        duration_column="ttf_true_best_days",
        event_column="event",
        id_column="id",
        config=MCMC_CFG,
    )

    ps = result.posterior_summary_frame()
    pq = result.posterior_life_quantiles_frame()
    ps.to_csv(out_dir / "posterior_summary.csv", index=False)
    pq.to_csv(out_dir / "posterior_life_quantiles.csv", index=False)

    # Convergence
    diag = result.diagnostics_frame()
    print("\n  Convergence:")
    for _, row in diag.iterrows():
        print(f"    Chain {int(row['chain'])}: {int(row['saved_draws'])} draws  "
              f"acc_eta={row['acceptance_eta_mean']:.2f}  acc_beta={row['acceptance_beta_mean']:.2f}")

    comps = [c for c in sorted(ps["component"].unique()) if c != "all"]
    print("\n  Posterior summary:")
    for comp in comps:
        chunk = ps[ps["component"] == comp]
        p = {r["parameter"]: r for _, r in chunk.iterrows()}
        print(f"    {comp}:  w={p['weight']['mean']:.3f}  "
              f"eta={p['eta']['mean']:.1f}d [{p['eta']['q2_5']:.0f}-{p['eta']['q97_5']:.0f}]  "
              f"beta={p['beta']['mean']:.3f} [{p['beta']['q2_5']:.3f}-{p['beta']['q97_5']:.3f}]  "
              f"ESS={p['weight']['ess']:.0f}  R-hat={p['weight']['rhat']:.4f}")

    print("\n  Life quantiles:")
    for _, row in pq.iterrows():
        print(f"    {row['label']:>4s}: {row['mean']:.1f}d  [95%CI {row['q2_5']:.1f}-{row['q97_5']:.1f}]")

    # Latent probs — reset index so boolean masks align with merged
    obs = result.observation_posterior_probabilities().rename(columns={"id": "row_id"})
    merged = df.merge(
        obs[["row_id", "component_1_probability", "component_2_probability",
             "most_probable_component", "max_probability", "classification_entropy"]],
        on="row_id", how="left",
    ).reset_index(drop=True)
    # Recompute has_freq on merged so index alignment is guaranteed
    has_freq = merged["n_freq_valid_days"] >= MIN_FREQ_DAYS
    merged.to_csv(out_dir / "latent_with_freq.csv", index=False)

    # --- KM reference (full population) ---
    print(f"\n{SEP}")
    print(f"  {field} overall KM (reference)")
    print(SEP)
    km_all = km_stats(merged)
    for tp in TIME_POINTS:
        print(f"  S({tp}d) = {km_all.get(f'S({tp}d)', float('nan')):.3f}")
    for bl in ["B50", "B75", "B90"]:
        print(f"  {bl}     = {km_all.get(bl, float('nan')):.0f}d")

    # --- Group analyses ---
    all_rows = []
    for label, col, threshold in [
        ("(a) mean_freq > 55 Hz",        "freq_w_mean",          55.0),
        ("(b) >50% days above 55 Hz",    "freq_above_55hz_pct",   0.50),
    ]:
        base = merged[has_freq & merged[col].notna()].copy()
        hi   = base[base[col] >  threshold]
        lo   = base[base[col] <= threshold]
        na   = merged[~has_freq | merged[col].isna()]
        rows = print_group_block(label, hi, lo, na, col, threshold)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_dir / "group_summary.csv", index=False)

    # --- Spearman: frequency vs P(short) ---
    print(f"\n{SEP}")
    print("  Spearman: frequency features vs P(component_1 = short-lived)")
    print(SEP)
    freq_cols = ["freq_w_mean", "freq_above_55hz_pct", "freq_signed_exposure"]
    for col in freq_cols:
        x = pd.to_numeric(merged[col], errors="coerce")
        y = merged["component_1_probability"]
        valid = x.notna() & y.notna() & has_freq
        if valid.sum() < 20:
            continue
        rho, pval = spearmanr(x[valid], y[valid])
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {col:<32}  rho={rho:+.3f}  p={pval:.4f}  n={valid.sum()} {sig}")

    # --- Spearman: frequency vs TTF directly ---
    print(f"\n  Spearman: frequency features vs TTF directly (no model)")
    for col in freq_cols:
        x = pd.to_numeric(merged[col], errors="coerce")
        y = merged["ttf_true_best_days"]
        valid = x.notna() & y.notna() & has_freq
        if valid.sum() < 20:
            continue
        rho, pval = spearmanr(x[valid], y[valid])
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {col:<32}  rho={rho:+.3f}  p={pval:.4f}  n={valid.sum()} {sig}")

    # --- Per-field breakdown of high-freq effect ---
    if field == "Global":
        print(f"\n{SEP}")
        print("  Per-field: fraction high-freq and failure rates")
        print(SEP)
        sub_f = merged[has_freq & merged["freq_w_mean"].notna()].copy()
        sub_f["high_a"] = sub_f["freq_w_mean"] > 55
        sub_f["high_b"] = sub_f["freq_above_55hz_pct"] > 0.5
        for fld, grp in sub_f.groupby("field"):
            n = len(grp)
            if n < 10:
                continue
            ha = grp.high_a.sum()
            hb = grp.high_b.sum()
            fail_hi = grp[grp.high_a]["event"].mean() if ha > 0 else float("nan")
            fail_lo = grp[~grp.high_a]["event"].mean() if (n-ha) > 0 else float("nan")
            rho, pval = spearmanr(grp["freq_w_mean"], grp["component_1_probability"])
            print(f"  {fld:<6}  n={n:>4}  pct_hi_a={ha/n:.2f}  "
                  f"fail_hi={fail_hi:.2f}  fail_lo={fail_lo:.2f}  "
                  f"rho(freq,pshort)={rho:+.3f}  p={pval:.3f}")

    print(f"\n  Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
