"""CLI for production-risk integration of ESP failure forecasts into the plan."""
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
from analysis.workflows.production_risk.run import run


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-excel", action="store_true", help="skip the companion workbook")
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    ap.add_argument("--bundle-date", default=C.BUNDLE_DATE, help="esp_survival_vba_models bundle date")
    ap.add_argument("--pp-master", type=Path, default=None, help="override production-plan master workbook")
    ap.add_argument("--gtm-schedule", type=Path, default=None, help="override DF04/GTM schedule workbook")
    ap.add_argument("--prediction-workbook", type=Path, default=None, help="override workbook with sheet Свод")
    ap.add_argument("--techregime-workbook", type=Path, default=None, help="override current techregime workbook")
    ap.add_argument("--equipment-big", type=Path, default=None, help="override WellsArtificialLiftBig workbook")
    def _positive_days(value: str) -> int:
        days = int(value)
        if days < 1:
            raise argparse.ArgumentTypeError("downtime days must be >= 1")
        return days

    ap.add_argument("--downtime-days", type=_positive_days, default=None, help="override repair downtime days for all scenarios (>= 1)")
    ap.add_argument("--scope-policy", default=C.ESP_SCOPE_CONSERVATIVE, help="ESP scope policy")
    ap.add_argument("--p90", type=float, default=C.CHANGEOUT_P90_THRESHOLD, help="changeout threshold")
    ap.add_argument("--full-tables", action="store_true",
                    help="also write the detailed CSV/parquet/json analysis tree (default: only the 3 workbooks)")
    args = ap.parse_args()
    cfg = C.RunConfig(
        forecast_start=args.forecast_start,
        horizon_end=args.horizon_end,
        bundle_date=args.bundle_date,
        pp_master_path=args.pp_master,
        gtm_schedule_path=args.gtm_schedule,
        prediction_workbook_path=args.prediction_workbook,
        techregime_workbook_path=args.techregime_workbook,
        equipment_big_path=args.equipment_big,
        downtime_override_days=args.downtime_days,
        esp_scope_policy=args.scope_policy,
        changeout_p90=args.p90,
        write_excel=not args.no_excel,
        full_tables=args.full_tables,
    )
    run(cfg)


if __name__ == "__main__":
    main()
