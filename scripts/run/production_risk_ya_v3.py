"""Fit and print the Ya v3 blended-rate model.

Thin CLI wrapper — logic in ``backend/analysis/workflows/production_risk/ya_v3.py``.

    python scripts/run/production_risk_ya_v3.py
    python scripts/run/production_risk_ya_v3.py --w 1.0     # pin the blend to pure Qnom
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import ya_v3 as V3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w", type=float, default=None,
                    help="pin the mixing weight (default: profile maximum)")
    ap.add_argument("--no-profile", action="store_true", help="skip the w profile (faster)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    run = V3.run(w=args.w, profile=not args.no_profile, write=not args.no_write)
    m = run.model
    print("=== Ya v3 — blended rate, freq, contractor.  No Kpod. ===")
    print(run.baseline_table.to_string(index=False))
    if len(run.profile):
        print("\n=== profile likelihood over w ===")
        print(run.profile.to_string(index=False))
        inci = run.profile[run.profile.in_ci95]
        print(f"  ML w = {m.layers.w:.2f};  95% interval "
              f"[{inci['w'].min():.2f}, {inci['w'].max():.2f}]  "
              f"-> the weight is NOT identified; only w = 0 (pure Ql) is rejected")
    print("\n=== layers ===")
    print(run.spec.to_string(index=False))
    print("\n=== worked example: oth, Qnom 500, Ql 450, +10 Hz ===")
    r = m.compose(contractor="oth", qnom=500, ql=450, freq_dev=10)
    print(f"  theta_parts = { {k: round(v, 4) for k, v in r['theta_parts'].items()} }")
    print(f"  theta_total = {r['theta_total']:.3f} -> eta_eff = {r['eta_eff']:.1f} d")
    print(f"  RMST730 = {r['rmst']:.1f} d (ref {r['rmst_ref']:.1f}, mult {r['rmst_mult']:.3f})")


if __name__ == "__main__":
    main()
