"""Workflow entrypoint for production-risk forecasting."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import RESULTS_ROOT, resolve_equipment_big_path, results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk, export_excel, failure_rate as failure_rate_mod, layers, repair_compat
from analysis.workflows.production_risk.survival import HazardLayer, StrataModel


def _stamped_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")


def _write_csv_unlocked(df, path: Path, **kwargs) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        stamped = _stamped_path(path)
        df.to_csv(stamped, **kwargs)
        return stamped


def _write_parquet_unlocked(df, path: Path) -> Path:
    try:
        df.to_parquet(path)
        return path
    except PermissionError:
        stamped = _stamped_path(path)
        df.to_parquet(stamped)
        return stamped
    except ImportError:
        # No parquet engine (the frozen EXE excludes pyarrow to save ~80 MB) —
        # fall back to compressed CSV with identical content.
        alt = path.with_suffix(".csv.gz")
        try:
            df.to_csv(alt, index=False, encoding="utf-8-sig")
            return alt
        except PermissionError:
            stamped = _stamped_path(alt)
            df.to_csv(stamped, index=False, encoding="utf-8-sig")
            return stamped


def _update_manifest(out_dir: Path, cfg: C.RunConfig, inputs: list[str]) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["script"] = "scripts/run/production_risk.py"
    manifest["inputs"] = inputs
    manifest["config"] = {
        "forecast_start": cfg.forecast_start.isoformat(),
        "horizon_end": cfg.horizon_end.isoformat(),
        "bundle_date": cfg.bundle_date,
        "pp_master_path": str(cfg.pp_master_path) if cfg.pp_master_path else None,
        "gtm_schedule_path": str(cfg.gtm_schedule_path) if cfg.gtm_schedule_path else None,
        "prediction_workbook_path": str(cfg.prediction_workbook_path) if cfg.prediction_workbook_path else None,
        "techregime_workbook_path": str(cfg.techregime_workbook_path) if cfg.techregime_workbook_path else None,
        "equipment_big_path": str(cfg.equipment_big_path) if cfg.equipment_big_path else None,
        "downtime_override_days": cfg.downtime_override_days,
        "esp_scope_policy": cfg.esp_scope_policy,
        "changeout_p90": cfg.changeout_p90,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_summary(production: dict[str, object], workover: object, changeout: object, audit) -> None:
    monthly = production["monthly"]
    primary = monthly[monthly["scenario"] == C.PRIMARY_SCENARIO_ID].copy()
    if primary.empty:
        primary = monthly.copy()
    annual = (
        primary.assign(year=primary["month"].str[:4])
        .groupby("year")
        .agg(
            total_planned_oil_t=("total_planned_oil_t", "sum"),
            oil_loss_t=("oil_loss_t", "sum"),
            expected_failures=("expected_failures", "sum"),
            uncovered_planned_oil_t=("uncovered_planned_oil_t", "sum"),
        )
    )
    print("\n" + "=" * 72)
    print("PRODUCTION-RISK SUMMARY")
    print("=" * 72)
    for year, row in annual.iterrows():
        risk_pct = 100.0 * row["oil_loss_t"] / row["total_planned_oil_t"] if row["total_planned_oil_t"] else 0.0
        print(
            f"  {year}: plan={row['total_planned_oil_t']:,.0f} t  "
            f"loss={row['oil_loss_t']:,.0f} t ({risk_pct:.2f}%)  "
            f"uncovered={row['uncovered_planned_oil_t']:,.0f} t  "
            f"exp failures={row['expected_failures']:.1f}"
        )
    included = int((audit["scope_label"] != "excluded_non_esp").sum()) if not audit.empty else 0
    excluded = int((audit["scope_label"] == "excluded_non_esp").sum()) if not audit.empty else 0
    print(f"\n  included wells: {included}")
    print(f"  excluded by conservative ESP scope: {excluded}")
    if not workover.empty:
        peak = workover[workover["scenario"] == C.PRIMARY_SCENARIO_ID]
        if peak.empty:
            peak = workover
        peak_row = peak.sort_values("reactive_esp_failures", ascending=False).iloc[0]
        print(
            f"  peak reactive month: {peak_row['month']} "
            f"({peak_row['reactive_esp_failures']:.2f} expected ESP failures)"
        )
    if not changeout.empty:
        top = changeout.head(5)
        print("\n  top 5 changeout candidates:")
        for _, row in top.iterrows():
            print(
                f"    {row['wid']:12} {row['plan_field'][:24]:24} "
                f"P90(base)={row.get(f'p_fail_90d_{C.PRIMARY_SCENARIO_ID}', float('nan')):.3f}"
            )


def _write_full_tables(
    cfg, plan, esp_source, gtm, projection, production, workover, changeout, audit,
    repair_forecast, repair_forecast_stress, global_downtime, field_downtime,
    failure_rate=None, failure_rate_stress=None,
) -> Path:
    """Detailed CSV/parquet/json analysis tree (opt-in via cfg.full_tables)."""
    import pandas as pd

    out = results_dir(cfg.output_slug)
    tables = out / "tables"
    _write_parquet_unlocked(projection, out / "models" / "projection.parquet")
    _write_csv_unlocked(production["monthly"], tables / "A_production_at_risk_monthly.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(production["by_well"], tables / "A_production_at_risk_by_well.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(production["by_field"], tables / "A_production_at_risk_by_field.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(workover, tables / "B_workover_load_monthly.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(changeout, tables / "C_changeout_priority.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(audit, tables / "D_mapping_audit.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(plan.anomalies, tables / "D_runtime_anomalies.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(repair_compat.rows_frame(repair_forecast), tables / "E_repair_forecast_rows.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(repair_compat.rows_frame(repair_forecast_stress), tables / "E_repair_forecast_hazard_rows.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(pd.DataFrame(repair_forecast["monthly_summary"]), tables / "E_repair_forecast_monthly.csv", index=False, encoding="utf-8-sig")
    _write_csv_unlocked(pd.DataFrame(repair_forecast_stress["monthly_summary"]), tables / "E_repair_forecast_hazard_monthly.csv", index=False, encoding="utf-8-sig")
    repair_compat.write_json(out / "models" / "repair_forecast.json", repair_forecast)
    repair_compat.write_json(out / "models" / "repair_forecast_hazard.json", repair_forecast_stress)
    downtime_rows = [{"field": "GLOBAL", "n": global_downtime.n, "p25": global_downtime.p25,
                      "p50": global_downtime.p50, "p75": global_downtime.p75, "mean": global_downtime.mean}]
    for field, stats in sorted(field_downtime.items()):
        downtime_rows.append({"field": field, "n": stats.n, "p25": stats.p25, "p50": stats.p50,
                              "p75": stats.p75, "mean": stats.mean})
    _write_csv_unlocked(pd.DataFrame(downtime_rows), tables / "D_downtime_summary.csv", index=False, encoding="utf-8-sig")
    if failure_rate is not None:
        _write_csv_unlocked(failure_rate.monthly, tables / "F_failure_rate_monthly.csv", index=False, encoding="utf-8-sig")
    if failure_rate_stress is not None:
        _write_csv_unlocked(
            failure_rate_stress.monthly,
            tables / "F_failure_rate_hazard_monthly.csv",
            index=False,
            encoding="utf-8-sig",
        )
    _update_manifest(out, cfg, [
        str(plan.master_path),
        str(esp_source.workbook_path),
        str(cfg.gtm_schedule_path or crosswalk.resolve_gtm_schedule_path()),
        str(cfg.techregime_workbook_path or crosswalk.resolve_techregime_workbook_path()),
        str(cfg.equipment_big_path or resolve_equipment_big_path()),
        str(C.model_registry_path(cfg.bundle_date)),
        str(C.hazard_coeffs_path(cfg.bundle_date)),
        str(C.run_covariates_path(cfg.bundle_date)),
        str(C.time_map_path(cfg.bundle_date)),
    ])
    return out


def run(cfg: C.RunConfig | None = None, write_excel: bool | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    write_excel = cfg.write_excel if write_excel is None else write_excel
    deliverables = RESULTS_ROOT
    deliverables.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading plan ...")
    plan = crosswalk.load_plan(cfg.forecast_start, cfg.horizon_end, master_path=cfg.pp_master_path)
    print(f"      producers={len(plan.producers)}  months={plan.fwd_months[0]}..{plan.fwd_months[-1]}")

    print("[2/6] loading ESP source + bundle ...")
    esp_source = crosswalk.load_esp_source(cfg.bundle_date, cfg.prediction_workbook_path)
    gtm = crosswalk.load_gtm(cfg.gtm_schedule_path)
    current_tr = crosswalk.load_current_techregime_status(cfg.techregime_workbook_path)
    model = StrataModel(bundle_date=cfg.bundle_date)
    hazard = HazardLayer(bundle_date=cfg.bundle_date)
    global_downtime, field_downtime = crosswalk.derive_downtime_quantiles(esp_source)
    print(
        f"      source={esp_source.workbook_path.name}  "
        f"cutoff={esp_source.source_cutoff.date() if esp_source.source_cutoff else 'n/a'}  "
        f"gtm rows={len(gtm)}  tr current statuses={len(current_tr)}"
    )

    print("[3/6] building well states ...")
    states, audit = layers.build_well_states(plan, esp_source, gtm, model, cfg, current_status=current_tr)
    src_mix = Counter(state.state_label for state in states)
    scope_mix = Counter(state.scope_label for state in states)
    print(f"      state={dict(src_mix)}  scope={dict(scope_mix)}")

    print("[4/6] deterministic scenario projection ...")
    projection = layers.run_projection(
        states,
        cfg.scenarios,
        model,
        hazard,
        global_downtime,
        field_downtime,
        plan.fwd_months,
        cfg.changeout_p90,
        downtime_override_days=cfg.downtime_override_days,
    )

    print("[5/6] building planner outputs ...")
    production = layers.production_at_risk(projection, plan)
    workover = layers.workover_load(projection, gtm, plan.fwd_months)
    changeout = layers.changeout(production["by_well"])
    repair_forecast = repair_compat.build(projection, states, cfg, scenario_id=C.PRIMARY_SCENARIO_ID)
    repair_forecast_stress = repair_compat.build(projection, states, cfg, scenario_id=C.STRESS_SCENARIO_ID)
    failure_rate = failure_rate_mod.compute(
        plan, esp_source, projection, cfg, scenario_id=C.PRIMARY_SCENARIO_ID, model=model
    )
    failure_rate_stress = failure_rate_mod.compute(
        plan, esp_source, projection, cfg, scenario_id=C.STRESS_SCENARIO_ID, model=model
    )
    print(
        f"      failure-rate: fields={len(failure_rate.fields)}  "
        f"Свод∩master={int(failure_rate.coverage['svod_in_master'])}  "
        f"last fact month={failure_rate.coverage['last_observed_month']}  "
        f"hazard fields={len(failure_rate_stress.fields)}"
    )
    fallback_share = failure_rate.coverage.get("global_pooled_share_by_field", {})
    if isinstance(fallback_share, dict):
        top_fallback = sorted(
            ((k, float(v)) for k, v in fallback_share.items() if k != failure_rate_mod.GLOBAL_LABEL),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]
        if top_fallback:
            details = ", ".join(f"{field}={share:.0%}" for field, share in top_fallback if share > 0)
            if details:
                print(f"      Global_Pooled share (top УН): {details}")
    print(f"      charts rendered: {len(failure_rate.chart_fields)} of {len(failure_rate.fields)} fields")

    _print_summary(production, workover, changeout, audit)

    written: list[Path] = []
    if write_excel:
        print("[6/6] writing deliverable workbooks ...")
        written = [
            export_excel.write(
                deliverables / "Риск_добычи_УЭЦН_2026_2027.xlsx", production, workover, changeout, audit
            ),
            repair_compat.write_excel(deliverables / "Прогноз_ремонтов.xlsx", repair_forecast, failure_rate=failure_rate),
            repair_compat.write_excel(
                deliverables / "Прогноз_ремонтов_hazard.xlsx",
                repair_forecast_stress,
                failure_rate=failure_rate_stress,
            ),
        ]
        for path in written:
            print(f"      wrote {path}")
    else:
        print("[6/6] deliverable workbooks skipped (--no-excel)")

    tables_dir = None
    if cfg.full_tables:
        print("      writing full analysis tables ...")
        tables_dir = _write_full_tables(
            cfg, plan, esp_source, gtm, projection, production, workover, changeout, audit,
            repair_forecast, repair_forecast_stress, global_downtime, field_downtime,
            failure_rate=failure_rate,
            failure_rate_stress=failure_rate_stress,
        )
        print(f"      tables → {tables_dir}")

    print(f"\nresults → {deliverables}")
    return {
        "out": deliverables,
        "tables_dir": tables_dir,
        "deliverables": written,
        "plan": plan,
        "esp_source": esp_source,
        "states": states,
        "projection": projection,
        "production": production,
        "workover": workover,
        "changeout": changeout,
        "audit": audit,
        "repair_forecast": repair_forecast,
        "repair_forecast_stress": repair_forecast_stress,
        "failure_rate": failure_rate,
        "failure_rate_stress": failure_rate_stress,
    }


if __name__ == "__main__":
    run(write_excel=False)
