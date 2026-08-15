"""CLI wrapper: Свод TTF (ННО) vs nameplate Qnom and vs contractor.

Heavy logic lives in ``analysis.workflows.production_risk.svod_ttf_by_design``
(CLAUDE.md rule 3).  The measured-rate (Ql) counterpart is ``svod_ttf_vs_ql.py``.

    python scripts/run/svod_ttf_by_design.py
    python scripts/run/svod_ttf_by_design.py --axes contractor
    python scripts/run/svod_ttf_by_design.py --axes qnom --overlays heatmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import svod_ttf_by_design as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["failures", "all_closed"],
                    choices=["failures", "all_closed"],
                    help="Row denominator: genuine failures and/or all pulls (default: both)")
    ap.add_argument("--axes", nargs="+", default=["qnom", "contractor"],
                    choices=["qnom", "contractor"],
                    help="Which design axis to plot (default: both)")
    ap.add_argument("--overlays", nargs="+", default=list(D.OVERLAYS),
                    choices=list(D.OVERLAYS),
                    help="How individual runs are drawn under the Qnom curves "
                         "(the contractor figure is categorical and ignores this)")
    ap.add_argument("--bins", type=int, default=10, help="Max quantile bins per group")
    ap.add_argument("--no-mc-cohort", action="store_true",
                    help="Keep pre-2024 Мирнинский installs (contrast only — CLAUDE.md §5)")
    args = ap.parse_args()

    out = D.run(
        variants=tuple(args.variants),
        axes=tuple(args.axes),
        overlays=tuple(args.overlays),
        bins=args.bins,
        mc_cohort=not args.no_mc_cohort,
    )
    print(f"Wrote Свод TTF-by-design panels to: {out}")


if __name__ == "__main__":
    main()
