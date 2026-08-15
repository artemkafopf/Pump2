"""Fit the unified v4 model (Ya, Vt_nonsour, Vt_sour) — free polylines on every layer.

    python scripts/run/unified_v4.py                 # fit + tables + figure
    python scripts/run/unified_v4.py --boot 400      # with bootstrap bands
    python scripts/run/unified_v4.py --cv            # re-derive the layer/ridge choices
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.models.survival.polyline_ph import (  # noqa: E402
    FREE, MONO, TENT, ArmSpec, eval_polyline, fit_polyline_ph)
from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import unified_v4 as U  # noqa: E402


# The CV machinery lives in the backend module so `field_v4` gates its layers the same way.
_held_out_loglik = U.held_out_loglik
cross_validate = U.cross_validate


def _cv_frame(field: str, cached=None) -> pd.DataFrame:
    cc = U.complete_case(U.prepare_field(field, cached=cached))
    cc["off_contractor"] = U.contractor_offset(cc, U.fit_contractor_levels(cc)["level"])
    if cc["stratum"].nunique() > 1:
        cc["sour"] = (cc["stratum"] == "Vt_sour").astype(float)
    return cc


def report_cv(cached=None) -> pd.DataFrame:
    """Reproduce the two decisions baked into the module: which layers, and how much ridge.

    Prints the shape comparison as well — the tent's deficit against the free polyline is the
    measurement behind :data:`unified_v4.LAYER_CV_DELTA`'s warning.
    """
    rows = []
    for field in ("Ya", "Vt"):
        cc = _cv_frame(field, cached)
        r = U.RIDGE[field]
        qn = ArmSpec("qnom", "qnom", U.QNOM_KNOTS, U.QNOM_REF, FREE, r["qnom"])
        base = cross_validate(cc, (qn,))
        rows.append({"field": field, "model": "qnom only", "shape": "-",
                     "cv": round(base, 2), "delta": 0.0})
        for name, col, knots, pin in (("kpod", "kpod_run", U.KPOD_KNOTS, U.KPOD_REF),
                                      ("freq", "freq_dev", U.FREQ_KNOTS, U.FREQ_REF)):
            for shape in (FREE, MONO, TENT):
                a = ArmSpec(name, col, knots, pin, shape, r[name])
                v = cross_validate(cc, (qn, a))
                rows.append({"field": field, "model": f"qnom + {name}", "shape": shape,
                             "cv": round(v, 2), "delta": round(v - base, 2)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=200, help="cluster-bootstrap resamples (0 = off)")
    ap.add_argument("--cv", action="store_true", help="re-derive layer and ridge choices")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if args.cv:
        report_cv().to_csv(results_dir(U.SLUG) / "tables" / "cv_layer_choice.csv",
                           index=False, encoding="utf-8-sig")
        return

    obj = U.run(n_boot=args.boot, write=not args.no_write)
    print(U.baseline_table(obj).to_string(index=False))
    print()
    print(U.layer_table(obj).to_string(index=False))
    print(f"\nVt shared-shape LR: {obj.lr['Vt']}")
    if not args.no_write:
        print(f"\nwritten to {results_dir(U.SLUG)}")


if __name__ == "__main__":
    main()
