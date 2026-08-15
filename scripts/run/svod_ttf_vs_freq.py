"""CLI wrapper: Свод TTF (ННО) vs frequency, before and after the fitted Ql layer.

Heavy logic lives in ``analysis.workflows.production_risk.svod_ttf_vs_freq``
(CLAUDE.md rule 3).  Siblings: ``svod_ttf_vs_ql.py``, ``svod_ttf_by_design.py``.

    python scripts/run/svod_ttf_vs_freq.py
    python scripts/run/svod_ttf_vs_freq.py --variants all_closed --overlays heatmap
    python scripts/run/svod_ttf_vs_freq.py --modes qnom   # Vt v4 / Ya v2.1 layer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import svod_ttf_vs_freq as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["failures", "all_closed"],
                    choices=["failures", "all_closed"],
                    help="Row denominator: genuine failures and/or all pulls (default: both)")
    ap.add_argument("--modes", nargs="+", default=["ql", "qnom"],
                    choices=["ql", "qnom"],
                    help="Which shipped rate layer to extract: ql (Ya v2 / Vt v3.2) "
                         "and/or qnom (Ya v2.1 / Vt v4)")
    ap.add_argument("--overlays", nargs="+", default=list(F.OVERLAYS),
                    choices=list(F.OVERLAYS),
                    help="How individual runs are drawn under the binned curves")
    ap.add_argument("--bins", type=int, default=10, help="Max quantile bins per group")
    ap.add_argument("--no-mc-cohort", action="store_true",
                    help="Keep pre-2024 Мирнинский installs (contrast only — CLAUDE.md §5)")
    args = ap.parse_args()

    out = F.run(
        variants=tuple(args.variants),
        modes=tuple(args.modes),
        overlays=tuple(args.overlays),
        bins=args.bins,
        mc_cohort=not args.no_mc_cohort,
    )
    print(f"Wrote Свод TTF-vs-freq panels to: {out}")


if __name__ == "__main__":
    main()
