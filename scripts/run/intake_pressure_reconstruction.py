"""Reconstruct pump intake pressure (Рприем) and propagate it into free gas at intake.

Thin CLI wrapper — all logic lives in ``backend/analysis``:

* physics   : ``analysis.features.intake_pressure``
* ML        : ``analysis.models.ml.intake_pressure_ml``
* scoring   : ``analysis.workflows.intake_pressure.validate``
* β         : ``analysis.workflows.intake_pressure.gas_at_intake``

Usage::

    python scripts/run/intake_pressure_reconstruction.py
    python scripts/run/intake_pressure_reconstruction.py --no-leak-demo --no-pvt-sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import pandas as pd

from analysis.workflows.intake_pressure.run import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="warehouse sqlite path (default: pump2.db)")
    ap.add_argument("--no-leak-demo", action="store_true",
                    help="skip the deliberately-ungrouped split kept to size the leakage")
    ap.add_argument("--no-pvt-sweep", action="store_true",
                    help="skip the free_gas PVT sensitivity sweep")
    ap.add_argument("--no-score", action="store_true",
                    help="reuse today's model_scores.csv instead of re-running the "
                         "~20 min validation; regenerates the imputation only")
    args = ap.parse_args()

    res = run(args.db,
              include_leak_demo=not args.no_leak_demo,
              run_pvt_sweep=not args.no_pvt_sweep,
              score=not args.no_score)

    pd.set_option("display.width", 200)
    print(f"\nwrote -> {res['out_dir']}\n")
    print(res["scores"][["model", "regime", "n", "mae", "med_ae", "r2"]].to_string(index=False))
    s = res["summary"]
    print(f"\ngap rows {s['gap_rows']:,}  imputed {s['gap_rows_imputed']:,} "
          f"({100 * s['gap_imputed_share']:.1f}%)")
    print(f"median physics-vs-ML disagreement: {s['median_realization_spread_atm']:.2f} atm")
    # ASCII only: the Windows console here is cp1251 and a literal beta raises
    # UnicodeEncodeError, which would kill the script *after* every artefact is written.
    print(f"beta rank stability (well-level Spearman): "
          f"{s['beta_rank_stability']['spearman']:.4f}")


if __name__ == "__main__":
    main()
