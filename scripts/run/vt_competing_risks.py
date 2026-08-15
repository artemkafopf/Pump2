"""CLI for the Вахитовское (Vt) failure/ГТМ competing-risks accounting (lab H2S).

Thin wrapper around ``analysis.workflows.production_risk.vt_competing_risks``.
Windows: run with ``PYTHONIOENCODING=utf-8``.  Outputs go to
``results/production_risk_vt_competing_risks/<date>/``.  Do NOT commit results.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

if getattr(sys, "frozen", False):
    RUNTIME_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    sys.path.insert(0, str(RUNTIME_ROOT / "backend"))
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import vt_competing_risks as V


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", type=_parse_date, default=C.SVOD_OPEN_ASOF,
                    help="analysis date / running-fleet snapshot (default: SVOD_OPEN_ASOF)")
    ap.add_argument("--horizon-months", type=int, default=12, help="forecast horizon in months")
    ap.add_argument("--n-boot", type=int, default=300, help="well-cluster bootstrap resamples (>=200)")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG figures")
    args = ap.parse_args()

    res = V.run_analysis(
        as_of=args.as_of,
        horizon_months=args.horizon_months,
        n_boot=args.n_boot,
        write_figures=not args.no_figures,
    )

    print("=" * 74)
    print("Vt competing-risks accounting (H2S from lab only)")
    print("=" * 74)
    print(f"  output          : {res['out']}")
    pop = res["pop"]
    print(f"  Vt runs         : {len(pop)}  (running={int(pop['is_open'].sum())}, "
          f"failures={int((pop['cause_code'] == V.FAILURE).sum())}, "
          f"ГТМ={int((pop['cause_code'] == V.GTM).sum())})")
    print("  --- lab coverage ---")
    for _, r in res["coverage"].iterrows():
        print(f"    {r['lab_class']:<12}: wells={r['wells']:>3} runs={r['runs']:>3} "
              f"отк={r['failures']:>3} ГТМ={r['gtm']:>3} откр={r['open_runs']:>3}")
    print("  --- λ_fail shape (all contractors) ---")
    for cls in V.FIT_CLASSES:
        p = res["params"]
        row = p[(p["lab_class"] == cls) & (p["contractor_cell"] == "all") & (p["cause"] == "failure")]
        if len(row):
            r = row.iloc[0]
            print(f"    {cls:<12}: β={r['beta']:.2f} η={r['eta']:.0f} "
                  f"RMST(0,730)={r['rmst_0_730']:.0f} n_отк={r['n_failures']}")
    print("  --- k1/k2 survival on lab split (event=отказ, ГТМ+раб. цензур., t_cal) ---")
    k = res["k1k2"]
    for _, r in k[k["selected"]].iterrows():
        print(f"    {r['stratum']:<26} {r['model_kind']:<13} "
              f"w1={r['w1']:.2f} β1={r['beta1']:.2f} η1={r['eta1']:.0f} β2={r['beta2']:.2f} η2={r['eta2']:.0f} "
              f"B50={r['b50']:.0f} RMST={r['rmst_0_730']:.0f} (n_отк={r['n_failures']}, Δaic={r['delta_aic_k1_minus_k2']:.1f})")
    print("  --- latent bound (Task D) ---")
    for _, r in res["latent"].iterrows():
        print(f"    {r['lab_class']:<12}: RMST λ_fail base={r['lambda_fail_rmst_baseline']:.0f} "
              f"-> верхн.={r['lambda_fail_rmst_upper_bound']:.0f} "
              f"(перекодир. ГТМ={r['n_gtm_recoded_as_failure']}/{r['n_gtm_total']})")
    print("=" * 74)


if __name__ == "__main__":
    main()
