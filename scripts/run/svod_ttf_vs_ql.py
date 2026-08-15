"""CLI wrapper: Свод TTF (ННО) vs Ql — Ya vs Vt/nonsour/sour, failures vs all pulls.

Heavy logic lives in ``analysis.workflows.production_risk.svod_ttf_vs_ql``
(CLAUDE.md rule 3).

    python scripts/run/svod_ttf_vs_ql.py
    python scripts/run/svod_ttf_vs_ql.py --variants failures
    python scripts/run/svod_ttf_vs_ql.py --rates ql_start   # first-month rate only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import svod_ttf_vs_ql as Q  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["failures", "all_closed"],
                    choices=["failures", "all_closed"],
                    help="Row denominator: genuine failures and/or all pulls (default: both)")
    ap.add_argument("--rates", nargs="+", default=["ql_svod", "ql_start"],
                    choices=list(Q.RATES),
                    help="Rate x-axis: Свод snapshot and/or first-operating-month mean")
    ap.add_argument("--overlays", nargs="+", default=list(Q.OVERLAYS),
                    choices=list(Q.OVERLAYS),
                    help="How individual runs are drawn under the binned curve: "
                         "none, scatter (coloured by outcome), heatmap (column-"
                         "normalised density)")
    ap.add_argument("--bins", type=int, default=10, help="Max quantile bins per group")
    ap.add_argument("--no-mc-cohort", action="store_true",
                    help="Keep pre-2024 Мирнинский installs (contrast only — see CLAUDE.md §5)")
    args = ap.parse_args()

    out = Q.run(
        variants=tuple(args.variants),
        rates=tuple(args.rates),
        overlays=tuple(args.overlays),
        bins=args.bins,
        mc_cohort=not args.no_mc_cohort,
    )
    print(f"Wrote Свод TTF-vs-Ql panels to: {out}")


if __name__ == "__main__":
    main()
