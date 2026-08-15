"""CLI for the Мирнинский (Mc) failure/ГТМ competing-risks accounting.

Thin wrapper around ``analysis.workflows.production_risk.mc_competing_risks``.
Windows: run with ``PYTHONIOENCODING=utf-8``.  Outputs go to
``results/production_risk_mc_competing_risks/<date>/``.  Do NOT commit results.
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
from analysis.workflows.production_risk import mc_competing_risks as M


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

    res = M.run_analysis(
        as_of=args.as_of,
        horizon_months=args.horizon_months,
        n_boot=args.n_boot,
        write_figures=not args.no_figures,
    )

    print("=" * 70)
    print("Mc competing-risks accounting")
    print("=" * 70)
    print(f"  output          : {res['out']}")
    pop = res["pop"]
    print(f"  Mc 2024+ runs   : {len(pop)}  (running={int(pop['is_open'].sum())}, "
          f"failures={int((pop['cause_code'] == M.FAILURE).sum())}, "
          f"ГТМ={int((pop['cause_code'] == M.GTM).sum())})")
    for c, f in res["fits"].items():
        print(f"  λ_{c:<8}: beta={f.beta:.3f} [{f.beta_lo:.2f},{f.beta_hi:.2f}]  "
              f"eta={f.eta:.0f}  n_ev={f.n_events}  RMST(0,730)={f.rmst_0_730:.0f}")
    fc = res["decomp"][res["decomp"]["segment"] == "forecast"]
    print(f"  12-мес прогноз  : отказы(1)={fc['m1_obs_failures'].sum():.1f}  "
          f"ГТМ(2)={fc['m2_gtm'].sum():.1f}  всего(3)={fc['m3_total_pulls'].sum():.1f}")
    print(f"  латент.контрфакт: без ГТМ(4)={fc['m4_latent_nogtm_failures'].sum():.1f}  "
          f"инфант-эффект(5)={fc['m5_infant_renewal_effect'].sum():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
