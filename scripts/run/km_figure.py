"""CLI: график Kaplan-Meier МРП против ННО для месторождения/страты.

    python scripts/run/km_figure.py --field Ya
    python scripts/run/km_figure.py --field Ya --contractor brt slb --label "brt+slb"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import km_figure as K  # noqa: E402

FIELD_TITLES = {"Ya": "Ярактинское НГКМ", "Mc": "Мирнинский УН", "Vt": "Верхнечонское НГКМ"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="Ya")
    ap.add_argument("--as-of", default="2026-07-01")
    ap.add_argument("--h2s", default="nonsour")
    ap.add_argument("--contractor", nargs="*", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    contractor = args.contractor or None
    data = K.build_data(args.as_of, args.field, h2s=args.h2s, contractor=contractor)

    title = FIELD_TITLES.get(args.field, args.field)
    title += " — Kaplan-Meier"
    if args.label:
        title += f", {args.label}"
    elif contractor:
        title += f", {'+'.join(contractor)}"

    extra = (f"Sum t/N завышает: МРП {data['naive_mrp']:.0f} против RMST {data['rmst_all']:.0f}; "
             f"ННО {data['naive_nno']:.0f} против RMST {data['rmst_fail']:.0f}")
    fig = K.build_figure(data, title, subtitle_extra=extra)

    out = results_dir(f"production_risk_km_{args.field.lower()}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    name = f"km_{args.field.lower()}"
    if args.label or contractor:
        name += "_" + (args.label or "+".join(contractor)).replace("+", "_")
    path = out / "figures" / f"{name}.png"
    fig.savefig(path, dpi=140)
    print(f"пробегов={data['n']} подъёмов={data['pulls']} отказов={data['failures']} "
          f"в работе={data['running']}")
    print(f"tau={data['tau']:.0f} | МРП RMST={data['rmst_all']:.0f} (Sum t/N={data['naive_mrp']:.0f}) "
          f"| ННО RMST={data['rmst_fail']:.0f} (Sum t/N={data['naive_nno']:.0f})")
    print(f"график: {path}")


if __name__ == "__main__":
    main()
