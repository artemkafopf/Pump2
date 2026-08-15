"""Thin CLI for the full-Vt survival model — new baseline mixture-Weibull (k1/k2) +
proportional-hazard layers (Ql primary + contractor from v3.1; Kpod/freq constrained minor).

    PYTHONIOENCODING=utf-8 python scripts/run/vt_physics_model.py [--n-boot 200] [--lam 6] [--quick]

Writes tables + Russian figures under
``results/production_risk_vt_physics_model/<date>/``.  See
``backend/analysis/workflows/production_risk/vt_physics_model.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import vt_physics_model as M  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=200, help="well-cluster bootstrap resamples (0 to skip)")
    p.add_argument("--lam", type=float, default=M.POLY_PENALIZER,
                   help="ridge on the Kpod/freq polyline knots (bigger = more minor)")
    p.add_argument("--failures-only", action="store_true",
                   help="drop ALL censored runs and fit on failures alone (BIASED variant)")
    p.add_argument("--quick", action="store_true", help="n_boot=40 fast smoke run")
    args = p.parse_args(argv)

    n_boot = 40 if args.quick else args.n_boot
    run = M.run(n_boot=n_boot, lam_poly=args.lam, failures_only=args.failures_only, write=True)

    if args.failures_only:
        print("*** FAILURES-ONLY variant (censored runs dropped) — BIASED; for comparison only ***")
    b = run.baseline
    print(f"Baseline full Vt: selected {b.model_kind}  "
          f"(k1 AIC {b.aic_k1:.0f} vs k2 AIC {b.aic_k2:.0f}, ΔAIC {b.delta_aic:+.1f})")
    print(f"  k1: β={b.beta1:.3f} η={b.eta1:.0f} | k2 modes ≈ {b.k2_short_life:.0f} / {b.k2_long_life:.0f} d "
          f"(≈ absorbed sour/nonsour)")
    print(f"  RMST(0,730)={run.baseline_table['rmst730'].iloc[0]:.0f}  "
          f"MRL(0)={run.baseline_table['mrl0'].iloc[0]:.0f}  n={b.n} events={b.events}")
    print("\nPrimary hazards (Ql + contractor from v3.1):")
    cols = ["label", "hr", "hr_ci_lo", "hr_ci_hi", "p", "hr_unit"]
    print(run.hazard_layers_table[cols].to_string(index=False))
    print("\nConstrained MINOR hazard θ per knot (>1 = higher risk vs reference):")
    print(run.knot_thetas[["arm", "knot", "theta_hazard", "life_mult_approx",
                           "theta_ci_lo", "theta_ci_hi"]].to_string(index=False))
    print(f"\nOutputs → results/{M.SLUG}/<date>/  (tables/: baseline, hazard_layers, "
          "knot_thetas, ttf_summary, model_comparison, coverage; figures/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
