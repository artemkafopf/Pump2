"""Pipeline orchestrator: runs all warehouse build steps in dependency order.

Usage:
    python scripts/pipeline.py                  # run all steps
    python scripts/pipeline.py --step ingest    # run only the ingest step
    python scripts/pipeline.py --step ttf       # run only proc__ttf_true
    python scripts/pipeline.py --check         # run quality checks only

Steps in dependency order:
  1. ingest         → raw__v03_runs, raw__v03_failures
  1b. equipment_big → raw__equipment_big, feat__run_equipment
  2. daily_merged   → proc__daily_merged
  3. daily_lab      → proc__daily_lab
  4. daily_op       → proc__daily_operating
  5. daily_ppt      → proc__daily_precipitate
  6. h2s            → proc__h2s_proxy
  7. ttf            → proc__ttf_true
  8. run_features   → feat__run_mature, feat__run_whole_life
  9. freq_exposure  → feat__run_freq_exposure
  10. mart_vt       → mart__vt_freq55
  11. mart_weibull  → mart__weibull_input
  check             → quality checks (no writes)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.db import get_warehouse_conn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pump2 warehouse pipeline orchestrator.")
    parser.add_argument("--step", default="all", help="Which step to run (default: all).")
    parser.add_argument("--check", action="store_true", help="Run quality checks only.")
    return parser.parse_args()


_STEPS: dict[str, str] = {
    "ingest": "scripts.ingest.ingest_v03",
    "equipment_big": "scripts.ingest.ingest_equipment_big",
    "daily_merged": "scripts.processing.build_daily_merged",
    "daily_lab": "scripts.processing.build_daily_lab",
    "daily_op": "scripts.processing.build_daily_operating",
    "daily_ppt": "scripts.processing.build_daily_precipitate",
    "h2s": "scripts.processing.build_h2s_proxy",
    "ttf": "scripts.processing.build_ttf_true",
    "run_features": "scripts.features.build_run_features",
    "freq_exposure": "scripts.features.build_freq_exposure",
    "mart_vt": "scripts.marts.build_mart_vt_freq55",
    "mart_weibull": "scripts.marts.build_mart_weibull",
}

_STEP_ORDER = [
    "ingest",
    "equipment_big",
    "daily_merged",
    "daily_lab",
    "daily_op",
    "daily_ppt",
    "h2s",
    "ttf",
    "run_features",
    "freq_exposure",
    "mart_vt",
    "mart_weibull",
]


def _run_step(name: str, conn) -> None:
    import importlib
    module = importlib.import_module(_STEPS[name])
    module.run(conn=conn)


def main() -> None:
    args = _parse_args()

    if args.check:
        from scripts.quality.check_pipeline import run as run_check
        run_check()
        return

    conn = get_warehouse_conn()
    try:
        if args.step == "all":
            print("[pipeline] Running full pipeline...\n")
            for step in _STEP_ORDER:
                _run_step(step, conn)
        elif args.step in _STEPS:
            print(f"[pipeline] Running step: {args.step}\n")
            _run_step(args.step, conn)
        else:
            print(f"[pipeline] Unknown step: {args.step!r}")
            print(f"  Available steps: {', '.join(_STEP_ORDER)}")
            sys.exit(1)

        print("\n[pipeline] Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
