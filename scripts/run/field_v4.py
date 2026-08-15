"""Fit the v4 structure for every field the calculators offer.

    python scripts/run/field_v4.py                    # all fields, CV on, tables + KM control
    python scripts/run/field_v4.py --no-cv            # skip the out-of-sample gate (fast)
    python scripts/run/field_v4.py --boot 400         # bootstrap bands on θ_Qnom
    python scripts/run/field_v4.py --fields Ya Vt Mc  # a subset

Writes ``results/production_risk_field_v4/<date>/`` — baselines, layers, the contractor
de-double-counting audit, support windows, and the two deploy blocks the VBA reads.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.workflows.production_risk import field_v4 as F  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", nargs="*", default=list(F.ROSTER),
                   help=f"subset of {' '.join(F.ROSTER)}")
    p.add_argument("--boot", type=int, default=0, help="bootstrap resamples for θ_Qnom")
    p.add_argument("--no-cv", action="store_true", help="skip the 5-fold layer gate")
    p.add_argument("--no-write", action="store_true")
    a = p.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    out = F.run(roster=tuple(a.fields), n_boot=a.boot, with_cv=not a.no_cv,
                write=not a.no_write)

    print("\n=== baselines")
    print(out["baselines"].to_string(index=False))
    print("\n=== contractor levels — plain vs size-adjusted")
    print(out["contractor"].to_string(index=False))
    print("\n=== θ_Qnom deployed")
    q = out["layers"]
    print(q[q["layer"] == "qnom"].pivot(index="stratum", columns="x",
                                        values="theta_deployed").to_string())
    clamped = q[q["clamped"]]
    if len(clamped):
        print("\nclamped knots (deploy guard — fitted curve fell above the reference):")
        print(clamped[["stratum", "layer", "x", "theta_fitted", "theta_deployed"]]
              .to_string(index=False))
    print("\n=== deploy blocks")
    print(out["tune_block"].to_string(index=False))
    print(out["qnom_block"].to_string(index=False))


if __name__ == "__main__":
    main()
