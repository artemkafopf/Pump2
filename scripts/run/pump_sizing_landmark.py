"""CLI: pump sizing vs operating life, day-90 landmark design.

    python scripts/run/pump_sizing_landmark.py
    python scripts/run/pump_sizing_landmark.py --landmark 60
    python scripts/run/pump_sizing_landmark.py --landmark 60 --landmark 90 --landmark 120

Thin wrapper — all logic lives in
:mod:`analysis.workflows.production_risk.pump_sizing_landmark`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import pump_sizing_landmark as PSL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmark", type=int, action="append", default=None,
                    help="landmark in operating days (repeatable; default 90). "
                         "Half of all failures happen before day 90, so the 60-day "
                         "variant is not optional decoration — run it.")
    ap.add_argument("--as-of", type=str, default=None, help="analysis date (YYYY-MM-DD)")
    ap.add_argument("--field", action="append", default=None,
                    help="per-field scope (repeatable; default Ya, Vt, Mc)")
    args = ap.parse_args()

    landmarks = args.landmark or [PSL.LANDMARK_OP_DAYS]
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    fields = tuple(args.field) if args.field else ("Ya", "Vt", "Mc")

    root = results_dir(PSL.SLUG)
    for L in landmarks:
        out = root if len(landmarks) == 1 else results_dir(f"{PSL.SLUG}/L{L}")
        print(f"\n=== landmark = {L} operating days -> {out} ===")
        res = PSL.run(as_of=as_of, landmark_op_days=L, fields=fields, out_dir=out)

        cov = res["coverage"]
        print(f"eligible {cov.n_eligible}/{cov.n_runs_total} runs "
              f"({cov.n_eligible / cov.n_runs_total:.1%}), {cov.n_events_eligible} events; "
              f"{cov.share_of_events_lost():.1%} of failures occur before the landmark")
        if not res["audit"].empty:
            print("\n-- landmark selection audit (does it drop the big pumps?) --")
            print(res["audit"].to_string(index=False))
        if not res["mediation"].empty:
            print("\n-- mediation: does free gas carry the size effect? --")
            print(res["mediation"].round(4).to_string(index=False))
        if not res["within_well"].empty:
            print("\n-- within-well (size read inside the well) --")
            cols = [c for c in ["label", "min_ratio", "n_informative_wells",
                                "n_informative_runs", "n_events", "HR_log_qnom",
                                "HR_log_qnom_lo", "HR_log_qnom_hi", "p_log_qnom",
                                "HR_run_seq", "p_run_seq", "note"]
                    if c in res["within_well"].columns]
            print(res["within_well"][cols].round(4).to_string(index=False))
        if not res["km"].empty:
            print("\n-- Weibull RMST vs Kaplan-Meier control --")
            print(res["km"][["label", "n", "n_events", "weibull_rmst", "km_rmst",
                             "rel_gap", "weibull_mrl", "flag"]].round(3).to_string(index=False))
        if not res["cost"].empty:
            print("\n-- what upsizing costs (RMST from the landmark) --")
            print(res["cost"][["label", "source", "size_multiple", "HR_per_size_multiple",
                               "rmst_ref_days", "rmst_lost_days", "rmst_lost_pct",
                               "observed_dql_m3d", "days_lost_per_m3d"]]
                  .round(3).to_string(index=False))
    print(f"\nwrote {root}")


if __name__ == "__main__":
    main()
