"""CLI wrapper: Свод ННО vs setpoint — marginal vs field+Ql residual.

Heavy logic lives in ``analysis.workflows.production_risk.svod_nno_decomposition``
(CLAUDE.md rule 3).

    python scripts/run/svod_nno_decomposition.py
    python scripts/run/svod_nno_decomposition.py --variants failures
    python scripts/run/svod_nno_decomposition.py --log-response   # old log-days panel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import svod_nno_decomposition as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["failures", "all_closed"],
                    choices=["failures", "all_closed"],
                    help="Row denominator: genuine failures only (default) and/or all closed runs")
    ap.add_argument("--bins", type=int, default=10, help="Quantile bins per x-axis")
    ap.add_argument("--log-response", action="store_true",
                    help="Residuals in log-days (the pre-2026-07-24 panel) instead of days")
    ap.add_argument("--by-field", nargs="*", default=["freq", "kpod", "delta"],
                    choices=["kpod", "delta", "freq"],
                    help="x-axes to also split per field (default: freq; empty to skip)")
    ap.add_argument("--freq-adjusted", nargs="*", default=["kpod", "delta"],
                    choices=["kpod", "delta", "freq"],
                    help="x-axes to ALSO rebuild with frequency removed from the control set")
    ap.add_argument("--qnom-controlled", action="store_true",
                    help="ALSO emit TTF vs freq & Kpod cleaned of field + Qnom (pump size) "
                         "for Ya and Vt — the 'rate hazard is Qnom not Ql' check")
    ap.add_argument("--no-mc-cohort", action="store_true",
                    help="Keep pre-2024 Мирнинский installs (contrast only — see CLAUDE.md §5)")
    args = ap.parse_args()

    out = D.run(
        variants=tuple(args.variants),
        bins=args.bins,
        log_response=args.log_response,
        mc_cohort=not args.no_mc_cohort,
        by_field_vars=tuple(args.by_field),
        freq_adjusted_vars=tuple(args.freq_adjusted),
    )
    if args.qnom_controlled:
        D.run_qnom_controlled(
            variants=tuple(args.variants),
            bins=args.bins,
            log_response=args.log_response,
            mc_cohort=not args.no_mc_cohort,
        )
    print(f"Wrote Свод ННО decomposition to: {out}")


if __name__ == "__main__":
    main()
