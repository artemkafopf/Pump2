"""CLI: find pumps whose speed was raised mid-run from 50-55 Hz to 60+ Hz."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.features.freq_regime import HIGH_THRESHOLD, LOW_BAND, MIN_PURITY, MIN_SEGMENT_DAYS
from analysis.workflows.production_risk.freq_regime_shift import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-band", nargs=2, type=float, default=list(LOW_BAND),
                        metavar=("LO", "HI"), help="pre-shift frequency band, Hz")
    parser.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD,
                        help="post-shift frequency threshold, Hz")
    parser.add_argument("--min-segment-days", type=int, default=MIN_SEGMENT_DAYS,
                        help="minimum operating days in each regime")
    parser.add_argument("--min-purity", type=float, default=MIN_PURITY,
                        help="minimum share of a segment's days inside its band")
    parser.add_argument("--no-mc-cohort", action="store_true",
                        help="disable the standing Мирнинский installs-2024+ cohort filter")
    parser.add_argument("--run-date", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    result = run(
        run_date=args.run_date,
        mc_cohort=not args.no_mc_cohort,
        low_band=(args.low_band[0], args.low_band[1]),
        high_threshold=args.high_threshold,
        min_segment_days=args.min_segment_days,
        min_purity=args.min_purity,
    )
    stats = result["stats"]
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    detected = result["detected"]
    if not detected.empty:
        cols = ["well", "field", "contractor", "install_date", "shift_date",
                "pre_median_hz", "post_median_hz", "shift_op_day", "days_shift_to_stop",
                "event", "failed_node"]
        print()
        print(detected[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
