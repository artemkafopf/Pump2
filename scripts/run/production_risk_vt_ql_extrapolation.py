"""Rebuild the evidence base for the θ_Ql high-rate extrapolation slope γ.

Thin CLI wrapper — logic in
``backend/analysis/workflows/production_risk/vt_ql_extrapolation.py``.

    python scripts/run/production_risk_vt_ql_extrapolation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import vt_composed_model as M  # noqa: E402
from analysis.workflows.production_risk import vt_ql_extrapolation as X  # noqa: E402


def main() -> None:
    res = X.run()

    print("=== 1. support above the top knot (is high Ql observed at all?) ===")
    cols = ["field", "n", "events", "ql_max", "ql_p99",
            "n_gt823", "ev_gt823", "n_gt1000", "n_gt1500"]
    print(res["support"][cols].to_string(index=False))

    print("\n=== Vt crude hazard by Ql band (the top bands are small-n) ===")
    print(res["band_hazard"].to_string(index=False))

    print("\n=== 2. EXCESS high-Ql slope over the fitted power law — profile-likelihood CI ===")
    print(res["free_slope"].to_string(index=False))
    print("  -> point estimates all negative, every CI contains 0 and spans ≈[-1,+1]:")
    print("     no evidence the tail needs a slope different from the fitted exponent.")

    print("\n=== 3. monotone-constrained v3.2 refit with knots above 823 ===")
    print(res["extended_knots"].to_string(index=False))
    print("  -> zero increment assigned above 823, with the monotonicity constraint binding")
    print("     at its lower bound.")

    print("\n=== 4. knot-density probe: does more resolution reveal curvature? ===")
    dense = res["dense_knots"]
    for grid, g in dense.groupby("grid", sort=False):
        ll = g.groupby("stratum")["loglik"].first().to_dict()
        print(f"  {grid:9s} ({int(g['n_knots'].iloc[0])} knots)  loglik " +
              ", ".join(f"{k} {v:.3f}" for k, v in ll.items()))
    print(dense[["grid", "stratum", "ql_lo", "ql_hi", "local_slope",
                 "n_in_band", "events_in_band"]].to_string(index=False))
    print("  -> the denser polylines are a STAIRCASE whose steps move with the grid, for")
    print("     ~1.0 loglik across 4-6 extra parameters: no evidence of real curvature.")

    print(f"\n=== 5. sensitivity to an alternative high-Ql slope (the shipped model has none) ===")
    ts = res["tail_sensitivity"]
    for s, g in ts.groupby("stratum"):          # per stratum: the shipped c differs, so a
        c = M.QL_COEF[s][0]                     # joint pivot would be mostly NaN
        print(f"  -- {s}: RMST(0,730) by assumed slope above Ql {M.QL_HI:.0f} "
              f"(shipped = fitted c = {c:.4f})")
        print(g.pivot_table(index="ql", columns="tail_slope", values="rmst730").to_string())
    print("\nSHIPPED: v3.2 knots reproduced, tail slope c = " +
          ", ".join(f"{s} {M.QL_COEF[s][0]:.4f}" for s in M.STRATA) +
          f"; no observation anywhere above Ql {M.QL_UNBACKED_ABOVE:.0f}")


if __name__ == "__main__":
    main()
