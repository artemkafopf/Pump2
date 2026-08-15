"""Fit **Ya v2.1** — the Ya k1/k2 hybrid with **Qnom** (nameplate BEP capacity) as the rate
arm instead of Ql, and write tables + figures.

Same model and procedure as Ya v2 (``production_risk_ya_k1k2_hybrid.py``); the only change is
the rate covariate — ``run(rate="Qnom")``.  Writes to its own slug so it never overwrites v2.

    python scripts/run/production_risk_ya_v21.py [--n-boot 200] [--no-write]
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import ya_k1k2_hybrid as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=200,
                    help="well-cluster bootstrap resamples for the θ intervals (0 = skip)")
    ap.add_argument("--num-starts", type=int, default=60,
                    help="EM restarts for the k2 baseline (the likelihood is multimodal)")
    ap.add_argument("--no-write", action="store_true", help="fit only, write nothing")
    args = ap.parse_args()

    pd.set_option("display.width", 220)
    run = M.run(rate="Qnom", n_boot=args.n_boot, num_starts=args.num_starts,
                write=not args.no_write)
    m = run.model

    print(f"=== Ya {m.version} BASELINE (rate arm = {m.rate_name}; "
          f"ref: brt, Qnom {m.rate_arm.pin:g}, Kpod 0.8, f = nominal) ===")
    print(run.baseline_table.T.to_string())

    print("\n=== COVERAGE ===")
    print(run.coverage.T.to_string())

    print("\n=== MODEL SPEC (θ and its RMST(0,730) life multiplier) ===")
    print(run.spec.to_string(index=False))

    print("\n=== APPLICABILITY (θ clamped flat outside; assumption, not a fit) ===")
    for k, v in m.applicability.items():
        print(f"  {k}: {v[0]} .. {v[1]}")

    print("\n=== COMPOSITION EXAMPLES (multiply θ, convert ONCE) ===")
    ref = m.compose()
    print(f"  reference (Qnom 200)      : RMST730 {ref['rmst_ref']:.1f} d, MRL0 {ref['mrl_ref']:.1f} d")
    for label, kw in (
        ("big pump (Qnom 900)", dict(qnom=900.0)),
        ("Qnom 900 + 65 Hz", dict(qnom=900.0, freq_dev=15.0)),
        ("Qnom 900 + 65 Hz + oth", dict(qnom=900.0, freq_dev=15.0, contractor="oth")),
        ("small pump (Qnom 50)", dict(qnom=50.0)),
    ):
        r = m.compose(**kw)
        print(f"  {label:<26}: θ={r['theta_total']:.3f} -> RMST730 {r['rmst']:.1f} d "
              f"(×{r['rmst_mult']:.3f}), MRL0 {r['mrl']:.1f} d, median {r['median']:.0f} d")

    if not args.no_write:
        print("\nwrote ->", results_dir(m.slug))


if __name__ == "__main__":
    main()
