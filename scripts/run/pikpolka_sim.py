"""Run the ПикПолка / NPV_УВЧ calculator from Python and sweep its inputs.

    # reproduce the workbook, then show what our fixes changed
    python scripts/run/pikpolka_sim.py --workbook "d:/.../Копия Калькулятор_NPV_УВЧ_V02.xlsm" --validate

    # УВЧ study: is the frequency uplift worth it, and where does Δ NPV change sign?
    python scripts/run/pikpolka_sim.py --workbook ... --sweep "branch1:freq=50,55,60,65" \
                                        --sweep "branch0:ql0=400,600,800,1000"

    # how much of the answer is the operator's frequency prior?
    python scripts/run/pikpolka_sim.py --workbook ... --sweep "ctrl:t60=1.0,1.15,1.3,1.6"

    # the same well on every field's model
    python scripts/run/pikpolka_sim.py --workbook ... --sweep "layers_key=Ya,Vt_sour,Az,Ic,Za,Mc"

``--fields-from-fit`` swaps the two hardcoded Vt strata for the full per-field parameter set
produced by ``scripts/run/field_v4.py``, so ``layers_key`` can name any field on the roster.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import pikpolka_sim as S  # noqa: E402

SLUG = "production_risk_pikpolka_sim"


def _parse_axis(spec: str):
    """``key=v1,v2,v3`` → (key, [values]) with numbers parsed as numbers."""
    key, _, rhs = spec.partition("=")
    vals = []
    for v in rhs.split(","):
        v = v.strip()
        try:
            vals.append(int(v) if v.lstrip("-").isdigit() else float(v))
        except ValueError:
            vals.append(v)
    return key.strip(), vals


def _blocks_from_fit(path: Path | None):
    """TuneBlock/QnomBlock from a ``field_v4`` run — falls back to the module defaults."""
    if path is None:
        candidates = sorted((REPO_ROOT / "results" / "production_risk_field_v4").glob("*/tables"))
        if not candidates:
            raise SystemExit("no field_v4 results found — run scripts/run/field_v4.py first")
        path = candidates[-1]
    tune = pd.read_csv(path / "deploy_tune_block.csv")
    qnom = pd.read_csv(path / "deploy_qnom_block.csv")
    t = {r["key"]: (r["beta"], r["rmst_ref_days"], r["brt"], r["slb"], r["oth"])
         for _, r in tune.iterrows()}
    qcols = [c for c in qnom.columns if c.startswith("q")]
    q = {r["key"]: tuple(float(r[c]) for c in qcols) for _, r in qnom.iterrows()}
    return t, q, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True)
    p.add_argument("--layout", default="npv", choices=("npv", "pikpolka"))
    p.add_argument("--validate", action="store_true",
                   help="check the port against the workbook's cached results")
    p.add_argument("--sweep", action="append", default=[],
                   help="repeatable, e.g. --sweep \"branch1:freq=50,55,60\"")
    p.add_argument("--fields-from-fit", nargs="?", const="", default=None,
                   help="use field_v4's deploy blocks (optionally a tables/ path)")
    p.add_argument("--preclamp", action="store_true",
                   help="run the pre-clamp sour θ_Qnom, i.e. exactly what the sheet ships")
    p.add_argument("--out", default=None, help="write the sweep to results/<slug>/<date>/")
    a = p.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    tune, qnom = None, (S.QNOM_PRECLAMP if a.preclamp else None)
    if a.fields_from_fit is not None:
        tune, qnom, src = _blocks_from_fit(Path(a.fields_from_fit) if a.fields_from_fit
                                           else None)
        print(f"parameters from {src}")

    cfg = S.load_config(a.workbook, layout=a.layout)
    print(f"\n{cfg.field}  кислый={cfg.sour}  ключ={cfg.key}  netback={cfg.netback:,.0f} ₽/т  "
          f"простой={cfg.downtime_days} сут  сетка={cfg.days} сут")
    for b in cfg.branches:
        print(f"  {b.name}: Ql={b.ql0:g}  f={b.freq:g} Гц  Qном={b.nominal0:g}  {b.contractor}")

    if a.validate:
        print("\n=== port vs workbook (pre-clamp parameters — what the sheet is running)")
        print(S.validate_against_workbook(a.workbook, layout=a.layout).to_string(index=False))
        print("\n=== with the sour clamp + ladder fix (what the rewired sheet will do)")
        print(S.validate_against_workbook(a.workbook, layout=a.layout,
                                          qnom_block=S.QNOM).to_string(index=False))

    base = S.simulate(cfg, tune=tune, qnom_block=qnom)
    print("\n=== base case")
    print(base["summary"].to_string(index=False))
    print(f"ННО в опорной точке (brt, Qном 250, 50 Гц, Кпод в плато): "
          f"{base['life_at_ref']:.1f} сут")
    if "d_nno" in base:
        print(f"Δ ННО = {base['d_nno']} сут   Δ NPV = {base['d_npv']:,.0f} ₽")

    if a.sweep:
        axes = dict(_parse_axis(s) for s in a.sweep)
        print(f"\n=== sweep over {list(axes)}  "
              f"({len(list(pd.MultiIndex.from_product(list(axes.values()))))} points)")
        g = S.sweep(cfg, tune=tune, qnom_block=qnom, **axes)
        cols = list(axes) + ["branch", "nno_days", "failures", "npv", "d_nno", "d_npv"]
        print(g[cols].to_string(index=False))
        if a.out is not None:
            d = results_dir(SLUG)
            f = d / "tables" / "sweep.csv"
            g.to_csv(f, index=False, encoding="utf-8-sig")
            print(f"\nwritten: {f}")


if __name__ == "__main__":
    main()
