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


def _run_check(bundle_date: str) -> int:
    """Fast self-test: load config + bundle + model, print the active state, and exit.

    Verifies a (re)built EXE is wired correctly in ~1-2 s instead of running the full
    ~5-min forecast.  Reads only small CSVs (no plan/telemetry workbooks).
    """
    frozen = bool(getattr(sys, "frozen", False))
    from analysis.workflows.production_risk import failure_rate as fr
    from analysis.workflows.production_risk.survival import HazardLayer, StrataModel

    problems: list[str] = []
    try:
        reg = StrataModel(bundle_date=bundle_date)
        n_strata = len(reg.df)
    except Exception as exc:  # noqa: BLE001
        n_strata = -1
        problems.append(f"strata registry failed to load: {exc}")
    try:
        hz = HazardLayer(bundle_date=bundle_date)
        static = list(hz.coeffs["covariate"]) if not hz.coeffs.empty else []
    except Exception as exc:  # noqa: BLE001
        static = []
        problems.append(f"hazard layer failed to load: {exc}")

    if not C.observed_failures_path(bundle_date).exists():
        problems.append("bundle is missing esp_observed_failures.csv")
    if fr._CALIBRATION_FACTORS or fr._REPORTING_FIELD_CALIBRATION_FACTORS:
        problems.append("manual calibration factors are ACTIVE (should be retired)")

    print("=" * 64)
    print("Pump2ProductionRisk self-check")
    print("=" * 64)
    print(f"  mode            : {'frozen EXE' if frozen else 'source'}")
    print(f"  bundle_date     : {bundle_date}")
    print(f"  bundle dir      : {C.bundle_dir(bundle_date)}")
    print(f"  strata rows     : {n_strata}")
    print(f"  observed file   : {'present' if C.observed_failures_path(bundle_date).exists() else 'MISSING'}")
    print(f"  manual calib    : {'retired' if not (fr._CALIBRATION_FACTORS or fr._REPORTING_FIELD_CALIBRATION_FACTORS) else 'ACTIVE (!)'}")
    print("  -- stress dials (base_p50 uses none of these) --")
    print(f"  Ql hazard       : {C.QL_HAZARD_ENABLED}   [{C.QL_HAZARD_SOURCE}]")
    print(f"  Kpod hazard     : {C.KPOD_HAZARD_ENABLED}   beta_over={C.KPOD_HAZARD_BETA_OVER} beta_under={C.KPOD_HAZARD_BETA_UNDER} "
          f"band=[{C.KPOD_HAZARD_K_LO},{C.KPOD_HAZARD_K_HI}]   [{C.KPOD_HAZARD_SOURCE}]")
    print(f"  Uncertainty     : {C.UNCERTAINTY_HAZARD_ENABLED}   [{C.UNCERTAINTY_HAZARD_SOURCE}]")
    print(f"  static Cox rows : {len(static)} enabled")
    for cov in static:
        print(f"                    - {cov}")
    print("=" * 64)
    if problems:
        print("SELF-CHECK FAILED:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("OK — model wired and bundle readable.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fast self-test: load config+bundle+model, print the active dial/state, and exit (no forecast)")
    ap.add_argument("--no-excel", action="store_true", help="skip the companion workbook")
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    ap.add_argument("--bundle-date", default=C.BUNDLE_DATE, help="esp_survival_vba_models bundle date")
    ap.add_argument("--pp-master", type=Path, default=None, help="override production-plan master workbook")
    ap.add_argument("--gtm-schedule", type=Path, default=None, help="override DF04/GTM schedule workbook")
    ap.add_argument("--prediction-workbook", type=Path, default=None, help="override workbook with sheet Свод")
    ap.add_argument("--techregime-workbook", type=Path, default=None, help="override current techregime workbook")
    ap.add_argument("--equipment-big", type=Path, default=None, help="override WellsArtificialLiftBig workbook")
    ap.add_argument("--fact-through", default=None, help="last complete fact month for observed failure rates (YYYY-MM)")
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
    ap.add_argument("--catboost-compare", action="store_true",
                    help="add the parallel CatBoost failure-rate line for comparison (heavy; Python/CLI only)")
    args = ap.parse_args()
    if args.check:
        sys.exit(_run_check(args.bundle_date))
    cfg = C.RunConfig(
        forecast_start=args.forecast_start,
        horizon_end=args.horizon_end,
        bundle_date=args.bundle_date,
        pp_master_path=args.pp_master,
        gtm_schedule_path=args.gtm_schedule,
        prediction_workbook_path=args.prediction_workbook,
        techregime_workbook_path=args.techregime_workbook,
        equipment_big_path=args.equipment_big,
        fact_through_month=args.fact_through or C.DEFAULT_FACT_THROUGH_MONTH,
        downtime_override_days=args.downtime_days,
        esp_scope_policy=args.scope_policy,
        changeout_p90=args.p90,
        write_excel=not args.no_excel,
        full_tables=args.full_tables,
        enable_catboost_compare=args.catboost_compare,
    )
    run(cfg)


if __name__ == "__main__":
    main()
