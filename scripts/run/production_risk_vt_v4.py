"""Fit and print the Vt v4 nameplate-rate model.

Thin CLI wrapper — logic in
``backend/analysis/workflows/production_risk/vt_v4.py``.

    python scripts/run/production_risk_vt_v4.py
    python scripts/run/production_risk_vt_v4.py --kpod-overlay
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import vt_v4 as V  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kpod-overlay", action="store_true",
                    help="switch on the Ya Kpod bathtub (default OFF: theta_Kpod = 1.0). "
                         "Requires a realised rate at prediction time.")
    ap.add_argument("--no-write", action="store_true", help="do not write tables/figures")
    args = ap.parse_args()

    run = V.run(write=not args.no_write, overlay=args.kpod_overlay or None)
    m = run.model

    print("=== Vt v4 — reference: brt, Qnom 250, 50 Hz, Kpod 0.8 ===")
    print(run.baseline_table.to_string(index=False))

    print(f"\n=== contractor levels (from Ql {V.CONTRACTOR_WINDOW[0]:.0f}-"
          f"{V.CONTRACTOR_WINDOW[1]:.0f}, brt = 1) ===")
    for s in V.STRATA:
        f = m.fits[s]
        for cg in V.CGS:
            lo, hi, p = f.level_ci[cg]
            flag = "" if cg == V.CONTRACTOR_REF else ("  *" if p < 0.05 else "   (n.s.)")
            print(f"  {s:8s} {cg:4s} {f.level[cg]:.3f}"
                  + ("" if cg == V.CONTRACTOR_REF else f" [{lo:.2f}, {hi:.2f}] p={p:.3f}") + flag)

    print("\n=== theta_Qnom ===")
    for s in V.STRATA:
        print(f"  {s:8s} " + "  ".join(f"{k:.0f}:{v:.3f}"
                                       for k, v in zip(V.QNOM_KNOTS, m.fits[s].theta_qnom)))
    print(f"\n=== theta_freq (transferred from Ya, clamped [42, 65]) ===")
    print("  " + "  ".join(f"{f}:{float(V.theta_freq(f)):.3f}" for f in (42, 45, 50, 55, 60, 65)))
    print(f"\n=== theta_Kpod overlay {'ON' if args.kpod_overlay else 'OFF (= 1.0, the null)'} ===")
    print("  " + "  ".join(f"{k}:{float(V.theta_kpod(k, overlay=args.kpod_overlay)):.3f}"
                           for k in (0.2, 0.6, 0.8, 1.0, 1.2, 1.5)))

    print("\n=== confident window per contractor (Qnom p10-p90) ===")
    print(run.support_table.drop(columns=["note"]).to_string(index=False))

    print("\n=== worked example: nonsour, slb, Qnom 600, 55 Hz ===")
    r = m.compose("nonsour", contractor="slb", qnom=600, freq=55)
    print(f"  theta_parts = { {k: round(v, 4) for k, v in r['theta_parts'].items()} }")
    print(f"  theta_total = {r['theta_total']:.3f} -> eta_eff = {r['eta_eff']:.1f} d")
    print(f"  RMST730 = {r['rmst']:.1f} d (ref {r['rmst_ref']:.1f}, mult {r['rmst_mult']:.3f}); "
          f"median = {r['median']:.0f} d; in_support = {r['in_support']}")


if __name__ == "__main__":
    main()
