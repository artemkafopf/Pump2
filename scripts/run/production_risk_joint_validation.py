"""Workstream D - joint validation gate for the production-risk refit.

This runner intentionally does not ship a bundle or rebuild the EXE.  It builds
temporary validation bundles from the frozen Workstream-A survival fit plus
Workstream-B time-map candidates, disables manual calibration factors in-process,
and emits the D0/D1 tables requested by the joint refit plan.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk, failure_rate
from analysis.workflows.production_risk.survival import StrataModel
from analysis.workflows.production_risk.time_map import TimeMap, fit_time_map


FOCUS_FIELDS = ["Мирнинский УН", "Ярактинский УН", "Верхнетирский УН", failure_rate.GLOBAL_LABEL]
FACT_FIRST = "2024-01"
FACT_LAST = "2026-06"


@dataclass(frozen=True)
class BundleSpec:
    name: str
    bundle_dir: Path
    bundle_date: str
    observed_mode: str = "recorded"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _hash_split(values: pd.Series) -> pd.Series:
    return np.where((pd.util.hash_pandas_object(values.astype(str), index=False) % 5) == 0, "holdout", "train")


def _predict_rows_with_map(rows: pd.DataFrame, tmap: TimeMap) -> np.ndarray:
    pred: list[float] = []
    for _, row in rows.iterrows():
        uptime, _ = tmap.predict_uptime(
            row.get("model_field"),
            float(row.get("age_op_start_observed", 0.0)),
            int(row["calendar_month"]),
            fallback=1.0,
        )
        pred.append(float(row["calendar_days"]) * uptime)
    return np.asarray(pred, dtype=float)


def _recon_row(split: str, model: str, g: pd.DataFrame, pred: np.ndarray, total_col: str = "op_days_treg") -> dict:
    actual = g["op_days_treg"].to_numpy(dtype=float)
    out = {
        "split": split,
        "model": model,
        "n_run_months": int(len(g)),
        "monthly_op_day_mae": float(np.mean(np.abs(pred - actual))) if len(g) else np.nan,
    }
    recon = pd.DataFrame({"run_idx": g["run_idx"].to_numpy(), "actual": actual, "pred": pred})
    by_run = recon.groupby("run_idx", as_index=False).sum()
    if total_col == "tte":
        totals = g.groupby("run_idx")["tte"].first().rename("actual_total")
        by_run = by_run.merge(totals, on="run_idx", how="left")
        denom = by_run["actual_total"].replace(0, np.nan)
        rel = ((by_run["pred"] - by_run["actual_total"]).abs() / denom).replace([np.inf, -np.inf], np.nan)
    else:
        denom = by_run["actual"].replace(0, np.nan)
        rel = ((by_run["pred"] - by_run["actual"]).abs() / denom).replace([np.inf, -np.inf], np.nan)
    rel = rel.dropna()
    out["median_total_nno_abs_pct"] = float(rel.median()) if not rel.empty else np.nan
    return out


def true_time_map_holdout(run_months: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the map on train wells only, then score train and holdout symmetrically."""
    fit = run_months[
        run_months["daily_coverage"].ge(0.5)
        & run_months["uptime_treg"].notna()
        & run_months["model_field"].notna()
    ].copy()
    fit["split"] = _hash_split(fit["well_code"])
    train = fit[fit["split"].eq("train")].copy()
    time_map, _ = fit_time_map(train)
    tmap = TimeMap(time_map)
    scalar = train.groupby("model_field")["uptime_treg"].mean().to_dict()
    global_scalar = float(train["uptime_treg"].mean())

    rows: list[dict] = []
    for split, g in fit.groupby("split"):
        pred_scalar = np.asarray(
            [
                float(row["calendar_days"])
                * float(scalar.get(row.get("model_field"), global_scalar))
                for _, row in g.iterrows()
            ],
            dtype=float,
        )
        pred_map = _predict_rows_with_map(g, tmap)
        scale_frame = pd.DataFrame({"run_idx": g["run_idx"].to_numpy(), "pred": pred_map, "tte": g["tte"].to_numpy(dtype=float)})
        scale = scale_frame.groupby("run_idx").agg(pred_sum=("pred", "sum"), tte=("tte", "first"))
        scale["factor"] = np.divide(
            scale["tte"].to_numpy(dtype=float),
            scale["pred_sum"].to_numpy(dtype=float),
            out=np.ones(len(scale), dtype=float),
            where=scale["pred_sum"].to_numpy(dtype=float) > 0,
        )
        pred_map_scaled = pred_map * g["run_idx"].map(scale["factor"]).to_numpy(dtype=float)
        rows.append(_recon_row(split, "scalar_field_trainfit", g, pred_scalar))
        rows.append(_recon_row(split, "age_season_map_trainfit_unscaled", g, pred_map))
        rows.append(_recon_row(split, "age_season_map_scaled_to_run_nno", g, pred_map_scaled, total_col="tte"))

    metrics = pd.DataFrame(rows)
    hold = metrics[metrics["split"].eq("holdout")].set_index("model")
    if {"scalar_field_trainfit", "age_season_map_trainfit_unscaled"}.issubset(hold.index):
        scalar_mae = float(hold.at["scalar_field_trainfit", "monthly_op_day_mae"])
        map_mae = float(hold.at["age_season_map_trainfit_unscaled", "monthly_op_day_mae"])
        improvement = (scalar_mae - map_mae) / scalar_mae if scalar_mae > 0 else np.nan
        metrics["holdout_map_mae_improvement_vs_scalar"] = improvement
        metrics["time_map_gate"] = "pass" if np.isfinite(improvement) and improvement >= 0.10 else "near_null"
    else:
        metrics["holdout_map_mae_improvement_vs_scalar"] = np.nan
        metrics["time_map_gate"] = "insufficient_holdout"
    return time_map, metrics


def idle_reattribution(run_months: pd.DataFrame, well_field: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = run_months.copy()
    df["reporting_field"] = df["well_code"].map(well_field).fillna("")
    df = df[df["month"].between(FACT_FIRST, FACT_LAST) & df["plan_producing"].notna()].copy()
    df["status"] = np.where(df["plan_producing"].astype(bool), "producing", "idle")

    failure_rows: list[dict] = []
    for run_idx, g in df.groupby("run_idx"):
        fail = g[g["is_failure_month"].astype(bool)]
        if fail.empty:
            continue
        recorded = fail.sort_values("month").iloc[-1]
        before = g[g["month"].le(str(recorded["month"]))].sort_values("month")
        prod = before[before["plan_producing"].astype(bool)]
        assigned = prod.iloc[-1] if not prod.empty else recorded
        failure_rows.append(
            {
                "run_idx": int(run_idx),
                "well_code": recorded["well_code"],
                "field": recorded["reporting_field"],
                "recorded_month": recorded["month"],
                "recorded_status": recorded["status"],
                "reattributed_month": assigned["month"],
                "reattributed_status": assigned["status"],
                "moved": bool(recorded["month"] != assigned["month"]),
            }
        )
    failures = pd.DataFrame(failure_rows)

    rows: list[dict] = []
    detail_counts: list[pd.DataFrame] = []
    for mode in ("recorded", "reattributed"):
        month_col = "recorded_month" if mode == "recorded" else "reattributed_month"
        status_col = "recorded_status" if mode == "recorded" else "reattributed_status"
        if not failures.empty:
            counts = (
                failures.groupby(["field", month_col], as_index=False)
                .size()
                .rename(columns={month_col: "month", "size": "observed_failures"})
            )
            global_counts = counts.groupby("month", as_index=False)["observed_failures"].sum().assign(field=failure_rate.GLOBAL_LABEL)
            counts = pd.concat([counts, global_counts], ignore_index=True)
            counts["observed_mode"] = mode
            detail_counts.append(counts)
        for field, den in list(df.groupby("reporting_field")) + [(failure_rate.GLOBAL_LABEL, df)]:
            if mode == "recorded":
                f = failures[failures["field"].eq(field)] if field != failure_rate.GLOBAL_LABEL else failures
                status_counts = f["recorded_status"].value_counts()
            else:
                f = failures[failures["field"].eq(field)] if field != failure_rate.GLOBAL_LABEL else failures
                status_counts = f["reattributed_status"].value_counts()
            den_counts = den["status"].value_counts()
            idle_months = int(den_counts.get("idle", 0))
            prod_months = int(den_counts.get("producing", 0))
            idle_fail = int(status_counts.get("idle", 0))
            prod_fail = int(status_counts.get("producing", 0))
            idle_h = idle_fail / idle_months if idle_months else np.nan
            prod_h = prod_fail / prod_months if prod_months else np.nan
            frac = idle_h / prod_h if np.isfinite(idle_h) and np.isfinite(prod_h) and prod_h > 0 else 0.0
            rows.append(
                {
                    "field": field,
                    "observed_mode": mode,
                    "idle_months": idle_months,
                    "idle_failures": idle_fail,
                    "idle_hazard_per_month": idle_h,
                    "producing_months": prod_months,
                    "producing_failures": prod_fail,
                    "producing_hazard_per_month": prod_h,
                    "idle_hazard_fraction": float(np.clip(frac, 0.0, 1.0)),
                    "raw_idle_hazard_fraction": frac,
                    "moved_failures": int(f["moved"].sum()) if not f.empty else 0,
                }
            )
    summary = pd.DataFrame(rows)
    recorded = summary[summary["observed_mode"].eq("recorded")][["field", "raw_idle_hazard_fraction"]].rename(
        columns={"raw_idle_hazard_fraction": "recorded_raw_idle_fraction"}
    )
    summary = summary.merge(recorded, on="field", how="left")
    ratio = np.divide(
        summary["raw_idle_hazard_fraction"].to_numpy(dtype=float),
        summary["recorded_raw_idle_fraction"].to_numpy(dtype=float),
        out=np.full(len(summary), np.nan),
        where=summary["recorded_raw_idle_fraction"].to_numpy(dtype=float) > 0,
    )
    summary["fraction_vs_recorded"] = ratio
    summary["real_idle_gate"] = np.where(
        (summary["observed_mode"].eq("reattributed"))
        & (summary["idle_failures"] >= 3)
        & (summary["raw_idle_hazard_fraction"] >= 0.25)
        & (summary["fraction_vs_recorded"] >= 0.5),
        "pass_real_idle",
        np.where(summary["observed_mode"].eq("reattributed"), "treat_as_misdated_or_sensitivity", ""),
    )

    adjusted_counts = pd.concat(detail_counts, ignore_index=True) if detail_counts else pd.DataFrame(
        columns=["field", "month", "observed_failures", "observed_mode"]
    )
    return summary, adjusted_counts


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def make_bundle(
    root: Path,
    name: str,
    models_path: Path,
    time_map: pd.DataFrame | None,
    source_bundle_date: str,
) -> BundleSpec:
    bundle_date = f"validation_{name}"
    bundle_dir = root / "bundles" / bundle_date
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    models = pd.read_csv(models_path, encoding="utf-8-sig").copy()
    if "uptime_factor" not in models.columns:
        models["uptime_factor"] = 1.0
    models.to_csv(bundle_dir / "esp_models.csv", index=False, encoding="utf-8-sig")
    _copy_if_exists(C.hazard_coeffs_path(source_bundle_date), bundle_dir / "esp_cox_coeffs.csv")
    _copy_if_exists(C.run_covariates_path(source_bundle_date), bundle_dir / "esp_run_covariates.csv")
    if time_map is not None and not time_map.empty:
        time_map.to_csv(bundle_dir / "esp_time_map.csv", index=False, encoding="utf-8-sig")
    return BundleSpec(name=name, bundle_dir=bundle_dir, bundle_date=bundle_date)


@contextmanager
def calibration_disabled():
    old_field = dict(failure_rate._CALIBRATION_FACTORS)
    old_reporting = dict(failure_rate._REPORTING_FIELD_CALIBRATION_FACTORS)
    old_weight = failure_rate._SURVIVAL_WEIGHT_FIELDS
    try:
        failure_rate._CALIBRATION_FACTORS.clear()
        failure_rate._REPORTING_FIELD_CALIBRATION_FACTORS.clear()
        failure_rate._SURVIVAL_WEIGHT_FIELDS = frozenset()
        yield
    finally:
        failure_rate._CALIBRATION_FACTORS.clear()
        failure_rate._CALIBRATION_FACTORS.update(old_field)
        failure_rate._REPORTING_FIELD_CALIBRATION_FACTORS.clear()
        failure_rate._REPORTING_FIELD_CALIBRATION_FACTORS.update(old_reporting)
        failure_rate._SURVIVAL_WEIGHT_FIELDS = old_weight


def _empty_projection() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario",
            "wid",
            "month",
            "expected_failures",
            "model_field",
        ]
    )


def run_failure_rate_for_bundle(plan, esp_source, cfg: C.RunConfig, bundle: BundleSpec) -> pd.DataFrame:
    model = StrataModel(registry_path=bundle.bundle_dir / "esp_models.csv", bundle_date=bundle.bundle_date)
    # StrataModel resolves esp_time_map through C.time_map_path(bundle_date).  The
    # validation bundles are outside C.BUNDLE_ROOT, so bind the candidate map directly.
    model.time_map = TimeMap.from_csv(bundle.bundle_dir / "esp_time_map.csv")
    with calibration_disabled():
        res = failure_rate.compute(
            plan,
            esp_source,
            _empty_projection(),
            cfg,
            scenario_id=C.PRIMARY_SCENARIO_ID,
            model=model,
        )
    out = res.monthly.copy()
    out["d_scenario"] = bundle.name
    out["observed_mode"] = bundle.observed_mode
    return out


def _apply_adjusted_observed(monthly: pd.DataFrame, adjusted_counts: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = monthly.copy()
    if adjusted_counts.empty:
        return out
    repl = adjusted_counts[adjusted_counts["observed_mode"].eq(mode)][["field", "month", "observed_failures"]]
    out = out.drop(columns=["observed_failures"], errors="ignore").merge(repl, on=["field", "month"], how="left")
    out["observed_failures"] = out["observed_failures"].fillna(0.0)
    fleet_pos = out["fleet_size"].to_numpy(dtype=float) > 0
    out["observed_rate"] = np.divide(
        out["observed_failures"].to_numpy(dtype=float),
        out["fleet_size"].to_numpy(dtype=float),
        out=np.full(len(out), np.nan),
        where=fleet_pos,
    )
    return out


def summarize_ladder(monthly_frames: list[pd.DataFrame], adjusted_counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    long_frames: list[pd.DataFrame] = []
    for frame in monthly_frames:
        scenario = str(frame["d_scenario"].iloc[0])
        if scenario == "A_plus_idle_reattributed":
            frame = _apply_adjusted_observed(frame, adjusted_counts, "reattributed")
            frame["observed_mode"] = "reattributed"
        else:
            frame = _apply_adjusted_observed(frame, adjusted_counts, "recorded")
            frame["observed_mode"] = "recorded_a_population"
        long_frames.append(frame)
        fact = frame[frame["month"].between(FACT_FIRST, FACT_LAST)].copy()
        for field, g in fact.groupby("field"):
            obs = float(g["observed_failures"].fillna(0).sum())
            pred = float(g["predicted_failures"].fillna(0).sum())
            rate_mask = g["observed_rate"].notna() & g["predicted_rate"].notna()
            mae = float((g.loc[rate_mask, "observed_rate"] - g.loc[rate_mask, "predicted_rate"]).abs().mean()) if bool(rate_mask.any()) else np.nan
            rows.append(
                {
                    "d_scenario": scenario,
                    "field": field,
                    "observed_mode": str(frame["observed_mode"].iloc[0]),
                    "observed_failures": obs,
                    "predicted_failures": pred,
                    "fact_model_ratio": obs / pred if pred > 0 else np.nan,
                    "monthly_rate_mae": mae,
                    "passes_0p90_1p10": bool(pred > 0 and 0.90 <= obs / pred <= 1.10),
                }
            )
    return pd.DataFrame(rows), pd.concat(long_frames, ignore_index=True)


def _gated_idle_map(time_map_full: pd.DataFrame, reattr: pd.DataFrame) -> pd.DataFrame:
    uptime = time_map_full[time_map_full["row_type"].eq("uptime")].copy()
    idle = time_map_full[time_map_full["row_type"].eq("idle_hazard")].copy()
    passed = set(
        reattr[
            reattr["observed_mode"].eq("reattributed")
            & reattr["real_idle_gate"].eq("pass_real_idle")
        ]["field"].astype(str)
    )
    # Keep GLOBAL as a fallback only if the global re-attribution passes; otherwise
    # field rows that failed the gate must not inherit a global idle fraction.
    idle = idle[idle["field"].astype(str).isin(passed)].copy()
    return pd.concat([uptime, idle], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-date", default=date.today().isoformat())
    ap.add_argument("--population", type=Path, default=Path("results/esp_survival_big_censored_refit/2026-07-15/fit_population.csv"))
    ap.add_argument("--models", type=Path, default=Path("results/esp_survival_big_censored_refit/2026-07-15/esp_models_big_censored_refit.csv"))
    ap.add_argument("--run-months", type=Path, default=Path("results/esp_time_map_refit/2026-07-15/tables/time_map_run_month_dataset.csv"))
    ap.add_argument("--time-map", type=Path, default=Path("results/esp_time_map_refit/2026-07-15/esp_time_map.csv"))
    ap.add_argument("--source-bundle-date", default=C.BUNDLE_DATE)
    ap.add_argument("--pp-master", type=Path, default=None)
    ap.add_argument("--prediction-workbook", type=Path, default=None)
    ap.add_argument("--equipment-big", type=Path, default=None)
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    args = ap.parse_args()

    out = results_dir("production_risk_joint_validation", args.out_date)
    tables = out / "tables"
    print("[1/5] loading D inputs ...")
    run_months = pd.read_csv(args.run_months, encoding="utf-8-sig")
    time_map_full = pd.read_csv(args.time_map, encoding="utf-8-sig")

    print("[2/5] D0 true time-map holdout ...")
    trained_time_map, holdout = true_time_map_holdout(run_months)
    holdout.to_csv(tables / "D_time_map_holdout.csv", index=False, encoding="utf-8-sig")

    print("[3/5] D0 idle re-attribution ...")
    plan = crosswalk.load_plan(args.forecast_start, args.horizon_end, master_path=args.pp_master)
    well_field = failure_rate.build_well_field(plan)
    idle_reattr, adjusted_counts = idle_reattribution(run_months, well_field)
    idle_reattr.to_csv(tables / "D_idle_reattribution.csv", index=False, encoding="utf-8-sig")
    adjusted_counts.to_csv(tables / "D_observed_failures_recorded_vs_reattributed.csv", index=False, encoding="utf-8-sig")

    print("[4/5] building temporary validation bundles ...")
    uptime_only = time_map_full[time_map_full["row_type"].eq("uptime")].copy()
    gated_idle = _gated_idle_map(time_map_full, idle_reattr)
    bundles = [
        make_bundle(out, "A_only", args.models, None, args.source_bundle_date),
        make_bundle(out, "A_plus_time_placement", args.models, uptime_only, args.source_bundle_date),
        make_bundle(out, "A_plus_idle_reattributed", args.models, uptime_only, args.source_bundle_date),
        make_bundle(out, "A_plus_idle_hazard", args.models, gated_idle, args.source_bundle_date),
    ]

    print("[5/5] replaying fact window with manual factors disabled ...")
    cfg = C.RunConfig(
        forecast_start=args.forecast_start,
        horizon_end=args.horizon_end,
        bundle_date=args.source_bundle_date,
        pp_master_path=args.pp_master,
        prediction_workbook_path=args.prediction_workbook,
        equipment_big_path=args.equipment_big,
        fact_through_month=FACT_LAST,
        write_excel=False,
        full_tables=False,
    )
    esp_source = crosswalk.load_esp_source(args.source_bundle_date, args.prediction_workbook)
    monthly_frames = []
    for b in bundles:
        print(f"      replaying {b.name} ...", flush=True)
        monthly_frames.append(run_failure_rate_for_bundle(plan, esp_source, cfg, b))
    ladder, monthly = summarize_ladder(monthly_frames, adjusted_counts)
    ladder.to_csv(tables / "D_scenario_ladder_fact_model.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(tables / "D_scenario_ladder_monthly.csv", index=False, encoding="utf-8-sig")
    trained_time_map.to_csv(tables / "D_time_map_trainfit.csv", index=False, encoding="utf-8-sig")
    gated_idle.to_csv(tables / "D_time_map_gated_idle_candidate.csv", index=False, encoding="utf-8-sig")

    focus = ladder[ladder["field"].isin(FOCUS_FIELDS)].copy()
    focus.to_csv(tables / "D_scenario_ladder_focus.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "script": "scripts/run/production_risk_joint_validation.py",
        "population": str(args.population),
        "models": str(args.models),
        "run_months": str(args.run_months),
        "time_map": str(args.time_map),
        "manual_calibration": "disabled in-process",
        "hazard_c": "not evaluated; C workstream not refit",
        "outputs": [
            "tables/D_time_map_holdout.csv",
            "tables/D_idle_reattribution.csv",
            "tables/D_scenario_ladder_fact_model.csv",
            "tables/D_scenario_ladder_monthly.csv",
        ],
    }
    (out / "D_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\noutputs -> {out}")
    print("\n=== D0 time-map holdout ===")
    print(holdout.to_string(index=False))
    print("\n=== D0 idle re-attribution focus ===")
    print(idle_reattr[idle_reattr["field"].isin(FOCUS_FIELDS)].to_string(index=False))
    print("\n=== D1 scenario ladder focus ===")
    cols = ["d_scenario", "field", "observed_failures", "predicted_failures", "fact_model_ratio", "monthly_rate_mae", "passes_0p90_1p10"]
    print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
