"""CLI: pump sizing in the infant band — the half of the failures the landmark excludes.

    python scripts/run/pump_sizing_infant.py

Companion to ``pump_sizing_landmark.py``.  Thin wrapper; logic lives in
:mod:`analysis.workflows.production_risk.pump_sizing_infant`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.workflows.production_risk import pump_sizing_infant as PSI  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", type=str, default=None, help="analysis date (YYYY-MM-DD)")
    ap.add_argument("--field", action="append", default=None,
                    help="per-field scope (repeatable; default Ya, Vt, Mc)")
    args = ap.parse_args()

    res = PSI.run(as_of=pd.Timestamp(args.as_of) if args.as_of else None,
                  fields=tuple(args.field) if args.field else ("Ya", "Vt", "Mc"))

    cov = res["coverage"]
    print(f"runs {cov.n_with_qnom}/{cov.n_runs_total} with Qnom, {cov.n_events} events "
          f"({cov.n_with_prior_deliverability} with prior-run deliverability)")
    print(f"events by band: {cov.n_events_by_band}")

    print("\n-- counting-process validity gate --")
    print(res["gate"].round(6).to_string(index=False))
    if not res["piecewise"].empty:
        print("\n-- size effect across follow-up (HR per e-fold Qnom) --")
        cols = [c for c in ["label", "strata", "band", "n_events_in_band", "HR_per_e_fold",
                            "HR_lo", "HR_hi", "p", "HR_per_doubling", "note"]
                if c in res["piecewise"].columns]
        print(res["piecewise"][cols].round(4).to_string(index=False))
    if not res["homogeneity"].empty:
        print("\n-- is the effect the same in every band? --")
        print(res["homogeneity"].round(4).to_string(index=False))
    if not res["terciles"].empty:
        print("\n-- survival by pump-size tercile (non-parametric) --")
        print(res["terciles"].round(3).to_string(index=False))
    if not res["cost"].empty:
        print("\n-- cost of doubling, measured FROM INSTALL (all bands) --")
        print(res["cost"][["label", "n", "n_events", "HR_per_size_multiple",
                           "rmst_ref_days", "rmst_lost_days", "rmst_lost_pct"]]
              .round(3).to_string(index=False))
    if not res["node_test"].empty:
        print("\n-- do infant failures change MODE with pump size? --")
        print(res["node_test"].round(4).to_string(index=False))
    print(f"\nwrote {res['out_dir']}")


if __name__ == "__main__":
    main()
