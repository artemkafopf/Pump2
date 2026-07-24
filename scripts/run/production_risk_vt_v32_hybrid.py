"""Fit the Vt v3.2 hybrid θ-model and write tables + figures.

Thin CLI wrapper — all logic lives in
``backend/analysis/workflows/production_risk/vt_v32_hybrid.py``.

    python scripts/run/production_risk_vt_v32_hybrid.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import vt_v32_hybrid as M  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 220)
    run = M.run(write=True)

    print("=== BASELINE (reference point: brt, Ql 250, Kpod 0.8, freq = nominal) ===")
    print(run.baseline_table.to_string(index=False))

    print("\n=== MODEL SPEC ===")
    for stratum in ("Vt_nonsour", "Vt_sour"):
        sub = run.spec[run.spec["stratum"] == stratum]
        print(f"\n-- {stratum} --")
        print(sub[["component", "scope", "term", "knot", "theta", "rmst730_mult"]]
              .to_string(index=False))

    print("\n=== APPLICABILITY (theta clamped flat outside; assumption, not a fit) ===")
    for k, v in M.APPLICABILITY.items():
        print(f"  {k}: {v[0]} .. {v[1]}")

    print("\n=== COMPOSITION EXAMPLE (multiply theta, convert ONCE) ===")
    for stratum in ("nonsour", "sour"):
        r = run.model.compose(stratum, contractor="oth", ql=823.0, kpod=1.2, freq_dev=0.0)
        print(f"  Vt_{stratum}: theta_total={r['theta_total']:.3f} -> "
              f"RMST730 {r['rmst']:.1f} d (ref {r['rmst_ref']:.1f}, mult {r['rmst_mult']:.3f}), "
              f"median {r['median']:.0f} d")

    print("\nwrote ->", results_dir(M.SLUG))


if __name__ == "__main__":
    main()
