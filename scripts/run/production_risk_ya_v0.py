"""Fit the Ya **v0** base model — cause-specific k1/k2 latent Weibull for failure AND ГТМ,
composed as competing risks — over cohort × clock, and write tables + figures.

Thin CLI wrapper — all logic lives in
``backend/analysis/workflows/production_risk/ya_v0.py``.

    python scripts/run/production_risk_ya_v0.py [--n-boot 200] [--num-starts 60] [--no-write]

The EM likelihood is multimodal, so ``--num-starts`` is a correctness knob, not a speed one:
drop it below ~12 only for a smoke test.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import ya_v0 as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT,
                    help="well-cluster bootstrap resamples per cause (0 = skip)")
    ap.add_argument("--num-starts", type=int, default=M.EM_STARTS,
                    help="EM restarts for each cause-specific k2 fit (likelihood is multimodal)")
    ap.add_argument("--scan-starts", type=int, default=M.SCAN_EM_STARTS,
                    help="EM restarts inside the install-cutoff sweep (a sensitivity, cheaper)")
    ap.add_argument("--no-write", action="store_true", help="fit only, write nothing")
    args = ap.parse_args()

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)

    run = M.run(
        num_starts=args.num_starts,
        n_boot=args.n_boot,
        scan_starts=args.scan_starts,
        write=not args.no_write,
    )

    print("=== POPULATION ===")
    print(run.tables["population"].to_string(index=False))

    print("\n=== CLOCK AUDIT — availability by outcome decides eligibility ===")
    print(run.tables["clock_audit"].to_string(index=False))

    print("\n=== CLOCK VERDICT ===")
    print(run.tables["clock_verdict"].to_string(index=False))

    print("\n=== CAUSE-SPECIFIC FITS (RMST(0,730) headline, MRL(0) alongside) ===")
    print(run.tables["cause_fits"].to_string(index=False))

    print("\n=== COMPETING-RISKS CHECK (parametric vs Aalen-Johansen vs naive 1-KM) ===")
    print(run.tables["cif_check"].to_string(index=False))

    print("\n=== INSTALL-CUTOFF SWEEP (is 'modern' a real cohort or a censoring artifact?) ===")
    print(run.tables["cohort_scan"].to_string(index=False))

    print("\n=== P(pull within 365 d | alive at age a) ===")
    print(run.tables["window_probability"].to_string(index=False))

    print("\n=== CAUSE ASSIGNMENT AUDIT ===")
    print(run.tables["cause_assignment_audit"].to_string(index=False))

    if not args.no_write:
        print("\nwrote ->", results_dir(M.SLUG))


if __name__ == "__main__":
    main()
