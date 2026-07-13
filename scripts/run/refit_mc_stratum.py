"""Refit Mc/MR production-risk survival rows and write a new VBA bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd

from analysis.workflows.production_risk.mc_refit import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-workbook", type=Path, default=None)
    parser.add_argument("--equipment-big", type=Path, default=None)
    parser.add_argument("--source-bundle-date", default="2026-07-08")
    parser.add_argument("--target-bundle-date", default="2026-07-13")
    parser.add_argument("--cutoff", default="2026-06-30")
    args = parser.parse_args()

    result = run(
        prediction_workbook_path=args.prediction_workbook,
        equipment_big_path=args.equipment_big,
        source_bundle_date=args.source_bundle_date,
        target_bundle_date=args.target_bundle_date,
        cutoff=pd.Timestamp(args.cutoff),
    )
    print(f"bundle={result.bundle_dir}")
    print(f"report={result.report_dir}")
    print(result.fit_rows.to_string(index=False))


if __name__ == "__main__":
    main()
