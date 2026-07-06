#!/usr/bin/env python3
"""
CLI for Bayesian Latent Weibull Mixture survival analysis.

Supports both:
  - Fixed-K Bayesian Weibull mixture  (--mode fixed, default)
  - Unknown-K birth-death MCMC        (--mode birth-death)

Usage examples
--------------
# Fixed K=2 from CSV, duration column 'days', event column 'failed':
  python scripts/bayesian_weibull_cli.py \\
      --data data/runs.csv \\
      --duration days \\
      --event failed \\
      --n-components 2 \\
      --output results/

# Unknown-K from Excel, duration computed from dates:
  python scripts/bayesian_weibull_cli.py \\
      --data data/runs.xlsx --sheet Sheet1 \\
      --start-date start_date --end-date end_date \\
      --event event_flag \\
      --mode birth-death \\
      --k-max 8 \\
      --n-iter 12000 --burn-in 3000 \\
      --output results_bd/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the backend package is importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from analysis.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
    load_survival_file,
)
from analysis.weibull_birth_death import (
    BirthDeathWeibullConfig,
    fit_birth_death_weibull,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian Latent Weibull Mixture — survival analysis CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Data ---
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", required=True, help="Path to CSV or Excel input file.")
    data_group.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first sheet).")
    data_group.add_argument("--duration", dest="duration_column", default=None,
                            help="Column with observed duration. Omit if using --start-date / --end-date.")
    data_group.add_argument("--event", dest="event_column", required=True,
                            help="Binary event column: 1=failure, 0=censored.")
    data_group.add_argument("--id", dest="id_column", default=None,
                            help="Unique identifier column (auto-generated if omitted).")
    data_group.add_argument("--start-date", dest="start_date_column", default=None,
                            help="Start date column for computing duration.")
    data_group.add_argument("--end-date", dest="end_date_column", default=None,
                            help="End date column for computing duration.")

    # --- Model ---
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--mode", choices=["fixed", "birth-death"], default="fixed",
                             help="'fixed': fixed-K Gibbs+MH; 'birth-death': unknown-K BD MCMC. (default: fixed)")
    model_group.add_argument("--n-components", type=int, default=2,
                             help="Number of mixture components for --mode fixed. (default: 2)")
    model_group.add_argument("--k-max", type=int, default=10,
                             help="Maximum K for --mode birth-death. (default: 10)")
    model_group.add_argument("--lambda-k", type=float, default=3.0,
                             help="Poisson prior rate on K for --mode birth-death. (default: 3.0)")
    model_group.add_argument("--birth-rate", type=float, default=3.0,
                             help="Fixed birth rate beta for the BD process. (default: 3.0)")

    # --- MCMC ---
    mcmc_group = parser.add_argument_group("MCMC")
    mcmc_group.add_argument("--n-iter", type=int, default=5_000,
                            help="Total MCMC iterations including burn-in. (default: 5000)")
    mcmc_group.add_argument("--burn-in", type=int, default=1_000,
                            help="Burn-in iterations to discard. (default: 1000)")
    mcmc_group.add_argument("--thin", type=int, default=5,
                            help="Save every Nth post-burn-in sample. (default: 5)")
    mcmc_group.add_argument("--n-chains", type=int, default=2,
                            help="Number of independent chains. (default: 2)")
    mcmc_group.add_argument("--seed", type=int, default=42,
                            help="Base random seed. (default: 42)")

    # --- Priors ---
    prior_group = parser.add_argument_group("Priors")
    prior_group.add_argument("--mu-log-eta", type=float, default=5.0,
                             help="Prior mean of log(eta). (default: 5.0)")
    prior_group.add_argument("--sigma-log-eta", type=float, default=1.5,
                             help="Prior SD of log(eta). (default: 1.5)")
    prior_group.add_argument("--mu-log-beta", type=float, default=0.0,
                             help="Prior mean of log(beta). (default: 0.0)")
    prior_group.add_argument("--sigma-log-beta", type=float, default=0.7,
                             help="Prior SD of log(beta). (default: 0.7)")

    # --- Output ---
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output", default="bayesian_weibull_output",
                           help="Directory for output files. (default: bayesian_weibull_output)")
    out_group.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90],
                           help="Future failure probability horizons in days. (default: 30 60 90)")

    return parser


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_banner(mode: str) -> None:
    print("=" * 60)
    print("  Bayesian Latent Weibull Mixture — Survival Analysis")
    print(f"  Mode: {'Fixed-K Gibbs+MH' if mode == 'fixed' else 'Unknown-K Birth-Death MCMC'}")
    print("=" * 60)


def _print_validation(report) -> None:
    print(f"\n[Data]")
    print(f"  Rows before validation : {report.rows_before_validation}")
    print(f"  Rows after  validation : {report.rows_after_validation}")
    print(f"  Removed                : {report.removed_rows}")
    print(f"  Failures               : {report.failure_count}")
    print(f"  Right-censored         : {report.censored_count}")
    if report.notes:
        for note in report.notes:
            print(f"  Note: {note}")


def _print_fixed_k_summary(result) -> None:
    summary = result.posterior_summary_frame()
    print(f"\n[Posterior Summary — K={result.config.n_components}]")
    for component in summary["component"].unique():
        chunk = summary.loc[summary["component"] == component]
        print(f"  {component}:")
        for _, row in chunk.iterrows():
            param = row["parameter"]
            mean = row["mean"]
            lo = row["q2_5"]
            hi = row["q97_5"]
            print(f"    {param:>12s} = {mean:8.3f}  [95% CI: {lo:.3f} – {hi:.3f}]")

    qlf = result.posterior_life_quantiles_frame()
    print(f"\n[Mixture Life Quantiles]")
    for _, row in qlf.iterrows():
        print(f"  {row['label']:>4s}: mean={row['mean']:.1f}  median={row['q50']:.1f}"
              f"  [95% CI: {row['q2_5']:.1f} – {row['q97_5']:.1f}]")

    diag = result.diagnostics_frame()
    print(f"\n[Diagnostics]")
    for _, row in diag.iterrows():
        print(f"  Chain {int(row['chain'])}: {int(row['saved_draws'])} draws"
              f"  acc_eta={row['acceptance_eta_mean']:.2f}"
              f"  acc_beta={row['acceptance_beta_mean']:.2f}")

    print(f"\n[Limitations]")
    for lim in result.limitations:
        print(f"  - {lim}")


def _print_bd_summary(result) -> None:
    k_frame = result.posterior_k_frame()
    print(f"\n[Posterior K Distribution]  P(K=1|data) = {result.p_single_weibull():.3f}")
    for _, row in k_frame.iterrows():
        bar = "#" * int(round(row["probability"] * 40))
        print(f"  K={int(row['k']):>2d}  {row['probability']:.3f}  {bar}")

    modal_summary = result.posterior_summary_frame()
    if not modal_summary.empty:
        k_vals = [d.k for d in result.all_draws]
        import numpy as np
        modal_k = int(np.bincount(k_vals).argmax())
        print(f"\n[Posterior Summary — modal K={modal_k}]")
        for comp in modal_summary["component"].unique():
            chunk = modal_summary.loc[modal_summary["component"] == comp]
            print(f"  {comp}:")
            for _, row in chunk.iterrows():
                print(f"    {row['parameter']:>12s} = {row['mean']:8.3f}"
                      f"  [95% CI: {row['q2_5']:.3f} – {row['q97_5']:.3f}]")

    qlf = result.posterior_life_quantiles_frame()
    print(f"\n[Mixture Life Quantiles]")
    for _, row in qlf.iterrows():
        print(f"  {row['label']:>4s}: mean={row['mean']:.1f}  median={row['q50']:.1f}"
              f"  [95% CI: {row['q2_5']:.1f} – {row['q97_5']:.1f}]")

    diag = result.diagnostics_frame()
    print(f"\n[Diagnostics]")
    for _, row in diag.iterrows():
        print(f"  Chain {int(row['chain'])}: {int(row['saved_draws'])} draws"
              f"  births={int(row['birth_count'])}"
              f"  deaths={int(row['death_count'])}"
              f"  bd_events={int(row['bd_event_count'])}")

    print(f"\n[Limitations]")
    for lim in result.limitations:
        print(f"  - {lim}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _print_banner(args.mode)

    # Load data
    print(f"\nLoading data from: {args.data}")
    sheet_arg = args.sheet
    if sheet_arg is not None:
        try:
            sheet_arg = int(sheet_arg)
        except (ValueError, TypeError):
            pass
    try:
        df = load_survival_file(args.data, sheet_name=sheet_arg)
    except Exception as exc:
        print(f"ERROR loading data: {exc}", file=sys.stderr)
        return 1

    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns.")

    # Fit
    t_start = time.perf_counter()

    if args.mode == "fixed":
        config = BayesianLatentWeibullConfig(
            n_components=args.n_components,
            n_iter=args.n_iter,
            burn_in=args.burn_in,
            thin=args.thin,
            n_chains=args.n_chains,
            mu_log_eta=args.mu_log_eta,
            sigma_log_eta=args.sigma_log_eta,
            mu_log_beta=args.mu_log_beta,
            sigma_log_beta=args.sigma_log_beta,
            random_seed=args.seed,
        )
        print(f"\nFitting fixed-K={args.n_components} Weibull mixture  "
              f"({args.n_iter} iters × {args.n_chains} chains)...")
        try:
            result = fit_bayesian_latent_weibull(
                df,
                duration_column=args.duration_column,
                event_column=args.event_column,
                id_column=args.id_column,
                start_date_column=args.start_date_column,
                end_date_column=args.end_date_column,
                config=config,
            )
        except Exception as exc:
            print(f"ERROR during fitting: {exc}", file=sys.stderr)
            return 1

        elapsed = time.perf_counter() - t_start
        print(f"  Done in {elapsed:.1f}s.")
        _print_validation(result.validation_report)
        _print_fixed_k_summary(result)

        print(f"\nExporting results to: {args.output}/")
        try:
            result.export_bundle(args.output)
            # Also save life quantiles
            result.posterior_life_quantiles_frame().to_csv(
                Path(args.output) / "posterior_life_quantiles.csv", index=False
            )
        except Exception as exc:
            print(f"ERROR during export: {exc}", file=sys.stderr)
            return 1

    else:   # birth-death
        bd_config = BirthDeathWeibullConfig(
            lambda_k=args.lambda_k,
            k_max=args.k_max,
            birth_rate=args.birth_rate,
            n_iter=args.n_iter,
            burn_in=args.burn_in,
            thin=args.thin,
            n_chains=args.n_chains,
            mu_log_eta=args.mu_log_eta,
            sigma_log_eta=args.sigma_log_eta,
            mu_log_beta=args.mu_log_beta,
            sigma_log_beta=args.sigma_log_beta,
            random_seed=args.seed,
        )
        print(f"\nFitting unknown-K birth-death Weibull mixture  "
              f"(K_max={args.k_max}, {args.n_iter} iters × {args.n_chains} chains)...")
        try:
            result = fit_birth_death_weibull(
                df,
                duration_column=args.duration_column,
                event_column=args.event_column,
                id_column=args.id_column,
                start_date_column=args.start_date_column,
                end_date_column=args.end_date_column,
                config=bd_config,
            )
        except Exception as exc:
            print(f"ERROR during fitting: {exc}", file=sys.stderr)
            return 1

        elapsed = time.perf_counter() - t_start
        print(f"  Done in {elapsed:.1f}s.")
        _print_validation(result.validation_report)
        _print_bd_summary(result)

        print(f"\nExporting results to: {args.output}/")
        try:
            result.export_bundle(args.output)
        except Exception as exc:
            print(f"ERROR during export: {exc}", file=sys.stderr)
            return 1

    print(f"\nAll outputs saved to: {Path(args.output).resolve()}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
