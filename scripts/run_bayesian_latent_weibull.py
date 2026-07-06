from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis import BayesianLatentWeibullConfig, fit_bayesian_latent_weibull, load_survival_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a fixed-K Bayesian latent Weibull mixture with right-censored data.")
    parser.add_argument("--input", required=True, help="Path to a CSV or Excel file.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index.")
    parser.add_argument("--duration-column", default=None, help="Duration column. Omit when deriving from dates.")
    parser.add_argument("--event-column", required=True, help="Event column where 1=failure and 0=censored.")
    parser.add_argument("--id-column", default=None, help="Unique observation ID column.")
    parser.add_argument("--start-date-column", default=None, help="Start date column when duration is absent.")
    parser.add_argument("--end-date-column", default=None, help="End date column when duration is absent.")
    parser.add_argument("--k", type=int, default=2, help="Fixed number of mixture components.")
    parser.add_argument("--iterations", type=int, default=2000, help="Total MCMC iterations per chain.")
    parser.add_argument("--burn-in", type=int, default=1000, help="Burn-in iterations per chain.")
    parser.add_argument("--thin", type=int, default=5, help="Thinning interval.")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--unknown-k", action="store_true", help="Enable birth/death MCMC over K.")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum K when unknown-k is enabled.")
    parser.add_argument("--lambda-k", type=float, default=3.0, help="Truncated Poisson prior mean for K.")
    parser.add_argument("--birth-death-probability", type=float, default=0.25, help="Probability of attempting a birth/death move each iteration.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported outputs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    sheet_value: str | int | None = args.sheet
    if isinstance(sheet_value, str) and sheet_value.isdigit():
        sheet_value = int(sheet_value)

    dataframe = load_survival_file(args.input, sheet_name=sheet_value)
    config = BayesianLatentWeibullConfig(
        n_components=args.k,
        n_iter=args.iterations,
        burn_in=args.burn_in,
        thin=args.thin,
        n_chains=args.chains,
        random_seed=args.seed,
        use_unknown_k=bool(args.unknown_k),
        k_max=args.k_max,
        lambda_k=args.lambda_k,
        birth_death_probability=args.birth_death_probability,
    )
    result = fit_bayesian_latent_weibull(
        dataframe,
        duration_column=args.duration_column,
        event_column=args.event_column,
        id_column=args.id_column,
        start_date_column=args.start_date_column,
        end_date_column=args.end_date_column,
        config=config,
    )
    result.export_bundle(args.output_dir)

    summary = result.posterior_summary_frame()
    diagnostics = result.diagnostics_frame()
    print("Bayesian latent Weibull fit completed.")
    print(f"Rows after validation: {result.validation_report.rows_after_validation}")
    print(f"Failures: {result.validation_report.failure_count}")
    print(f"Censored: {result.validation_report.censored_count}")
    print(f"Saved draws per chain: {result.chains[0].weights_relabeled.shape[0] if result.chains else 0}")
    if result.config.use_unknown_k:
        print("Posterior K distribution:")
        print(result.posterior_k_frame().to_string(index=False))
        print("")
    print("")
    print("Posterior summary preview:")
    print(summary.head(12).to_string(index=False))
    print("")
    print("Diagnostics:")
    print(diagnostics.to_string(index=False))
    print("")
    print(f"Outputs written to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
