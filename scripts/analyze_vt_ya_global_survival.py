#!/usr/bin/env python3
"""
Bayesian Weibull mixture analysis: Vt vs Ya vs Global subpopulations.

Loads mart__vt_freq55 from the SQLite warehouse, fits a K=2 Bayesian
Weibull mixture to each of the three groups, and prints a comparative
summary of component parameters, life quantiles, and hazard shapes.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from analysis.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
)

# -- Config --------------------------------------------------------------------

DB_PATH = _REPO_ROOT / "data" / "warehouse" / "pump2.db"
TABLE   = "mart__vt_freq55"
DUR_COL = "ttf_true_best_days"
EVT_COL = "event"

# MCMC: enough for stable posteriors, not so many that it takes forever
MCMC_CFG = BayesianLatentWeibullConfig(
    n_components=2,
    n_iter=12_000,
    burn_in=3_000,
    thin=5,
    n_chains=3,
    mu_log_eta=5.0,    # prior center at exp(5)~=150 days
    sigma_log_eta=1.5, # covers ~10-1500 days
    mu_log_beta=0.0,   # prior center at exp(0)=1.0
    sigma_log_beta=0.7,
    random_seed=42,
)

# -- Data loading --------------------------------------------------------------

def load_groups() -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT field, {DUR_COL}, {EVT_COL} FROM {TABLE}",
        conn,
    )
    conn.close()

    df = df.dropna(subset=[DUR_COL, EVT_COL])
    df = df[df[DUR_COL] > 0].copy()
    df[EVT_COL] = df[EVT_COL].astype(int)

    return {
        "Global": df.copy(),
        "Vt":     df[df["field"] == "Vt"].copy(),
        "Ya":     df[df["field"] == "Ya"].copy(),
    }


# -- Printing helpers ----------------------------------------------------------

def _hazard_shape(beta: float) -> str:
    if beta < 0.8:
        return "infant mortality (beta<1, decreasing hazard)"
    elif beta > 1.2:
        return "wear-out (beta>1, increasing hazard)"
    else:
        return "roughly constant hazard (beta~1)"


def print_group_summary(name: str, result) -> None:
    sep = "-" * 62
    print(f"\n{'='*62}")
    print(f"  Group: {name}")
    print(f"{'='*62}")

    vr = result.validation_report
    print(f"  n={vr.rows_after_validation}  "
          f"failures={vr.failure_count} ({vr.failure_count/vr.rows_after_validation:.1%})  "
          f"censored={vr.censored_count}")
    print(sep)

    # Component posteriors (skip mixture-level "all" row)
    summary = result.posterior_summary_frame()
    comps = [c for c in sorted(summary["component"].unique()) if c != "all"]
    for comp in comps:
        chunk = summary[summary["component"] == comp]
        params = {row["parameter"]: row for _, row in chunk.iterrows()}
        eta_row  = params.get("eta")
        beta_row = params.get("beta")
        w_row    = params.get("weight")

        eta_mean  = eta_row["mean"]   if eta_row  is not None else float("nan")
        beta_mean = beta_row["mean"]  if beta_row is not None else float("nan")
        w_mean    = w_row["mean"]     if w_row    is not None else float("nan")
        eta_ci    = (eta_row["q2_5"],  eta_row["q97_5"])  if eta_row  is not None else (float("nan"), float("nan"))
        beta_ci   = (beta_row["q2_5"], beta_row["q97_5"]) if beta_row is not None else (float("nan"), float("nan"))

        print(f"\n  {comp}:")
        print(f"    weight   = {w_mean:.3f}")
        print(f"    eta(days)= {eta_mean:7.1f}  [95% CI: {eta_ci[0]:.1f} - {eta_ci[1]:.1f}]")
        print(f"    beta     = {beta_mean:7.3f}  [95% CI: {beta_ci[0]:.3f} - {beta_ci[1]:.3f}]")
        print(f"    shape    : {_hazard_shape(beta_mean)}")

    # Life quantiles (mixture-level)
    print(f"\n  {sep}")
    print("  Life quantiles (mixture):")
    qlf = result.posterior_life_quantiles_frame()
    for _, row in qlf.iterrows():
        print(f"    {row['label']:>4s}: {row['mean']:7.1f}d  "
              f"median={row['q50']:7.1f}d  "
              f"[95% CI: {row['q2_5']:.1f} - {row['q97_5']:.1f}]")

    # KM reference
    km = kaplan_meier_frame(result.cleaned_df[DUR_COL], result.cleaned_df[EVT_COL])
    km50 = km[km["survival"] <= 0.50]
    km25 = km[km["survival"] <= 0.25]
    km10 = km[km["survival"] <= 0.10]
    km50_t = km50["time"].iloc[0] if not km50.empty else float("nan")
    km25_t = km25["time"].iloc[0] if not km25.empty else float("nan")
    km10_t = km10["time"].iloc[0] if not km10.empty else float("nan")
    print(f"\n  KM reference: B50={km50_t:.0f}d  B75={km25_t:.0f}d  B90={km10_t:.0f}d")

    # Diagnostics
    diag = result.diagnostics_frame()
    print(f"\n  {sep}")
    print("  Diagnostics:")
    for _, row in diag.iterrows():
        print(f"    Chain {int(row['chain'])}: {int(row['saved_draws'])} draws  "
              f"acc_eta={row['acceptance_eta_mean']:.2f}  "
              f"acc_beta={row['acceptance_beta_mean']:.2f}")


def print_comparison_table(results: dict[str, object]) -> None:
    print(f"\n\n{'='*75}")
    print("  COMPARISON TABLE")
    print(f"{'='*75}")
    header = f"{'Metric':<34} {'Global':>12} {'Vt':>12} {'Ya':>12}"
    print(header)
    print("-" * 75)

    def _get_param(result, comp_idx: int, param: str) -> str:
        s = result.posterior_summary_frame()
        comps = [c for c in sorted(s["component"].unique()) if c != "all"]
        if comp_idx >= len(comps):
            return "n/a"
        comp = comps[comp_idx]
        chunk = s[s["component"] == comp]
        row = chunk[chunk["parameter"] == param]
        if row.empty:
            return "n/a"
        return f"{row.iloc[0]['mean']:.3f}"

    def _get_quant(result, label: str) -> str:
        q = result.posterior_life_quantiles_frame()
        row = q[q["label"] == label]
        if row.empty:
            return "n/a"
        return f"{row.iloc[0]['mean']:.1f}d"

    rows_vals = {g: str(results[g].validation_report.rows_after_validation) for g in ["Global","Vt","Ya"]}
    fail_vals = {g: f"{results[g].validation_report.failure_count} ({results[g].validation_report.failure_count/results[g].validation_report.rows_after_validation:.1%})" for g in ["Global","Vt","Ya"]}

    def _row(label: str, vals: dict) -> None:
        print(f"{label:<34} {vals.get('Global','n/a'):>12} {vals.get('Vt','n/a'):>12} {vals.get('Ya','n/a'):>12}")

    _row("N (runs)", rows_vals)
    _row("Failures", fail_vals)
    print("-" * 75)

    for ci in [0, 1]:
        comp_label = f"Component {ci+1}"
        _row(f"{comp_label} weight",
             {g: _get_param(results[g], ci, "weight") for g in ["Global","Vt","Ya"]})
        _row(f"{comp_label} eta (scale, days)",
             {g: _get_param(results[g], ci, "eta") for g in ["Global","Vt","Ya"]})
        _row(f"{comp_label} beta (shape)",
             {g: _get_param(results[g], ci, "beta") for g in ["Global","Vt","Ya"]})
        print("-" * 75)

    for label in ["B10", "B25", "B50", "B90"]:
        _row(f"Mixture {label} life",
             {g: _get_quant(results[g], label) for g in ["Global","Vt","Ya"]})


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print(f"Loading data from {DB_PATH} ...")
    groups = load_groups()
    for name, df in groups.items():
        n_fail = int(df[EVT_COL].sum())
        print(f"  {name:6s}: n={len(df)}, failures={n_fail} ({n_fail/len(df):.1%}), "
              f"median_ttf={df[DUR_COL].median():.0f}d")

    results: dict[str, object] = {}

    for name, df in groups.items():
        print(f"\nFitting K=2 Bayesian Weibull mixture - {name} "
              f"(n={len(df)}, {int(df[EVT_COL].sum())} failures) ...")
        t0 = time.perf_counter()
        result = fit_bayesian_latent_weibull(
            df,
            duration_column=DUR_COL,
            event_column=EVT_COL,
            config=MCMC_CFG,
        )
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s")
        results[name] = result
        print_group_summary(name, result)

    print_comparison_table(results)

    # Save CSV summary
    rows = []
    for name, result in results.items():
        vr = result.validation_report
        s  = result.posterior_summary_frame()
        q  = result.posterior_life_quantiles_frame()
        comps = sorted(s["component"].unique())
        for i, comp in enumerate(comps):
            chunk = s[s["component"] == comp]
            params = {row["parameter"]: row for _, row in chunk.iterrows()}
            for param in ["weight", "eta", "beta"]:
                if param not in params:
                    continue
                r = params[param]
                rows.append(dict(
                    group=name,
                    component=comp,
                    n=vr.rows_after_validation,
                    n_failures=vr.failure_count,
                    parameter=param,
                    mean=r["mean"],
                    sd=r["sd"],
                    q2_5=r["q2_5"],
                    q50=r["q50"],
                    q97_5=r["q97_5"],
                ))
        for _, qrow in q.iterrows():
            rows.append(dict(
                group=name,
                component="mixture",
                n=vr.rows_after_validation,
                n_failures=vr.failure_count,
                parameter=qrow["label"],
                mean=qrow["mean"],
                sd=qrow["sd"],
                q2_5=qrow["q2_5"],
                q50=qrow["q50"],
                q97_5=qrow["q97_5"],
            ))

    out_path = _REPO_ROOT / "analysis" / "vt_ya_global_survival_comparison.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nComparison CSV saved -> {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
