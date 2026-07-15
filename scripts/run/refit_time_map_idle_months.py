"""Workstream B - build production-risk time map and idle-month decision tables."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk, failure_rate
from analysis.workflows.production_risk.time_map import (
    build_run_month_dataset,
    fit_time_map,
    idle_hazard_table,
    load_daily_operating,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _load_population(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["install_date", "end_date"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--population",
        type=Path,
        default=Path("results/esp_survival_big_censored_refit/2026-07-15/fit_population.csv"),
        help="Workstream-A fit_population.csv",
    )
    ap.add_argument("--out-date", default=date.today().isoformat())
    ap.add_argument("--pp-master", type=Path, default=None)
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    args = ap.parse_args()

    out = results_dir("esp_time_map_refit", args.out_date)
    tables = out / "tables"

    print("[1/5] loading A population + production plan ...")
    pop = _load_population(args.population)
    plan = crosswalk.load_plan(args.forecast_start, args.horizon_end, master_path=args.pp_master)
    well_field = failure_rate.build_well_field(plan)

    print("[2/5] loading daily operating telemetry ...")
    daily = load_daily_operating()

    print("[3/5] building run-month mapping dataset ...")
    run_months = build_run_month_dataset(pop, daily, plan)
    run_months.to_csv(tables / "time_map_run_month_dataset.csv", index=False, encoding="utf-8-sig")

    print("[4/5] fitting age/season uptime map ...")
    uptime_map, metrics = fit_time_map(run_months)

    print("[5/5] measuring idle-month hazard ...")
    idle = idle_hazard_table(run_months, well_field)
    esp_time_map = pd.concat([uptime_map, idle.reindex(columns=uptime_map.columns.union(idle.columns))], ignore_index=True)
    # Keep the bundle-facing file at the run root and the verbose copies in tables/.
    esp_time_map.to_csv(out / "esp_time_map.csv", index=False, encoding="utf-8-sig")
    esp_time_map.to_csv(tables / "esp_time_map.csv", index=False, encoding="utf-8-sig")
    uptime_map.to_csv(tables / "uptime_map_rows.csv", index=False, encoding="utf-8-sig")
    idle.to_csv(tables / "idle_hazard_decision.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables / "time_map_reconstruction_metrics.csv", index=False, encoding="utf-8-sig")

    focus = idle[idle["field"].isin(["GLOBAL", "Мирнинский УН", "Ярактинский УН", "Верхнетирский УН"])].copy()
    manifest = {
        "population": str(args.population),
        "n_population_runs": int(len(pop)),
        "n_run_months": int(len(run_months)),
        "n_map_rows": int(len(esp_time_map)),
        "idle_decision": (
            "fit_idle_hazard_fraction"
            if bool((focus["decision"] == "fit_idle_hazard_fraction").any())
            else "attribute_to_last_producing_month"
        ),
        "focus_idle": focus[
            [
                "field",
                "idle_hazard_fraction",
                "idle_failures",
                "producing_failures",
                "idle_hazard_per_month",
                "producing_hazard_per_month",
                "decision",
            ]
        ].to_dict("records"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\noutputs -> {out}")
    if not metrics.empty:
        print("\n=== reconstruction metrics ===")
        print(metrics.to_string(index=False))
    if not focus.empty:
        print("\n=== idle hazard decision (focus) ===")
        print(focus[["field", "idle_hazard_fraction", "idle_failures", "producing_failures", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
