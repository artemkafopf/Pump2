"""CLI wrapper: Vt TTF ~ ESP operating-covariate relations.

Heavy logic lives in ``analysis.workflows.production_risk.vt_ttf_covariates``; this
is a thin entry point (CLAUDE.md rule 3).  See the workflow plan
``agents/analyses/vt_ttf_covariates_plan.md``.

    python scripts/run/vt_ttf_covariates.py
    python scripts/run/vt_ttf_covariates.py --no-kvch --n-boot 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import vt_ttf_covariates as V  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None,
                    help="Analysis date YYYY-MM-DD (default: config.SVOD_OPEN_ASOF)")
    ap.add_argument("--no-kvch", action="store_true", help="Skip the low-coverage КВЧ join")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="Bootstrap draws for the risk-min Kpod CI")
    args = ap.parse_args()

    import pandas as pd
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    out = V.run(as_of=as_of, with_kvch=not args.no_kvch, n_boot=args.n_boot)
    print(f"Wrote Vt TTF covariate outputs to: {out}")


if __name__ == "__main__":
    main()
