"""Print the Vt composed operating-life model spec + a worked example.

Thin CLI wrapper — logic in
``backend/analysis/workflows/production_risk/vt_composed_model.py`` (the VBA-port source).

    python scripts/run/production_risk_vt_composed_model.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import vt_composed_model as C  # noqa: E402


def main() -> None:
    print("=== Vt composed model — reference (brt, Ql 250, freq 50, Kpod 0.8) ===")
    for s in C.STRATA:
        b = C.BASELINE[s]
        print(f"  {s:8s}: beta0={b['beta0']}, eta0={b['eta0']}, RMST730_ref={C.rmst_ref(s):.1f} d")

    print("\n=== contractor θ (brt=1) ===")
    for s in C.STRATA:
        print(f"  {s:8s}: " + ", ".join(f"{c}={v}" for c, v in C.CONTRACTOR[s].items()))

    print("\n=== θ at a grid (Ql per stratum; freq/Kpod shared) ===")
    print(f"  Ql window {C.QL_EVAL_RANGE[0]:.0f}–{C.QL_EVAL_RANGE[1]:.0f} m³/d; single fitted "
          f"smooth curve through the v3.2 knots, tail slope c=" +
          ", ".join(f"{s} {C.QL_COEF[s][0]:.4f}" for s in C.STRATA) +
          f"; no run observed above Ql {C.QL_UNBACKED_ABOVE:.0f}")
    print("  Ql  :", {q: {s: round(float(C.theta_ql(s, q)), 3) for s in C.STRATA}
                      for q in (0, 100, 250, 450, 823, 1250, 1433, 2000)})
    print("  freq:", {f: round(float(C.theta_freq(f)), 3) for f in (42, 45, 50, 55, 60, 65)})
    print("  Kpod:", {k: round(float(C.theta_kpod(k)), 3) for k in (0.2, 0.6, 0.8, 1.0, 1.2, 1.5)})

    print("\n=== worked example: sour, oth, Ql 450, freq 60, Kpod 1.0 ===")
    r = C.compose("sour", contractor="oth", ql=450, freq=60, kpod=1.0)
    print(f"  θ_parts = {r.theta_parts}")
    print(f"  θ_total = {r.theta_total:.3f}  -> η_eff = {r.eta_eff:.1f} d")
    print(f"  RMST730 = {r.rmst:.1f} d (ref {r.rmst_ref:.1f}, mult {r.rmst_mult:.3f}); "
          f"median = {r.median:.0f} d; MRL = {r.mrl:.0f} d")


if __name__ == "__main__":
    main()
