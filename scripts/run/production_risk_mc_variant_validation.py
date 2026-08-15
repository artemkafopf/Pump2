"""Workstream E - Mc vintage variant validation ladder.

This runner is a follow-up to Workstream D.  It does not ship a bundle or
rebuild the EXE.  It clones the Workstream-A registry, swaps only the
``Mc_nonsour_Pooled`` baseline to each candidate vintage row, then replays the
D fact/model ladder with the placement map and gated idle-hazard candidate.
"""
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
from scripts.run import production_risk_joint_validation as D


FOCUS_FIELDS = ["Мирнинский УН", "Ярактинский УН", "Верхнетирский УН", failure_rate.GLOBAL_LABEL]


def mc_variant_model_registry(base_models: pd.DataFrame, variant: pd.Series) -> pd.DataFrame:
    """Return a registry with exactly one Mc pooled nonsour row replaced."""
    required = ["label", "model_kind", "w1", "beta1", "eta1", "beta2", "eta2", "b20", "b50", "b80"]
    missing = [c for c in required if c not in variant.index or pd.isna(variant[c])]
    if missing:
        raise ValueError(f"Mc variant row is missing required values: {missing}")

    out = base_models.copy()
    out = out[~out["stratum"].isin(["Mc_nonsour_Pooled", "Mc_nonsour_brt"])].reset_index(drop=True)
    row = {
        "stratum": "Mc_nonsour_Pooled",
        "model_kind": variant["model_kind"],
        "w1": variant["w1"],
        "beta1": variant["beta1"],
        "eta1": variant["eta1"],
        "beta2": variant["beta2"],
        "eta2": variant["eta2"],
        "b20": variant["b20"],
        "b50": variant["b50"],
        "b80": variant["b80"],
        "n_runs": variant.get("n_runs"),
        "n_failures": variant.get("n_failures"),
        "n_censored": variant.get("n_censored"),
        "n_big_censored": variant.get("n_big_censored"),
        "delta_aic": variant.get("delta_aic"),
        "clock": variant.get("clock", "ttf_mix_svod_plus_big_nno"),
        "field": "Mc",
        "h2s_class": "nonsour",
        "contractor_group": "Pooled",
        "mc_window": variant["label"],
    }
    for col in out.columns:
        row.setdefault(col, pd.NA)
    return pd.concat([out, pd.DataFrame([row], columns=out.columns)], ignore_index=True)


def write_variant_models(base_models: pd.DataFrame, variants: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for _, variant in variants.iterrows():
        label = str(variant["label"])
        note = variant.get("note", "")
        note_text = "" if pd.isna(note) else str(note).strip()
        if note_text:
            rows.append({"mc_window": label, "status": "skipped", "reason": note_text})
            continue
        registry = mc_variant_model_registry(base_models, variant)
        path = out_dir / f"esp_models_mc_{label}.csv"
        registry.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "mc_window": label,
                "status": "written",
                "models_path": str(path),
                "model_kind": variant.get("model_kind"),
                "b50": variant.get("b50"),
                "b80": variant.get("b80"),
                "n_runs": variant.get("n_runs"),
                "n_failures": variant.get("n_failures"),
                "max_abs_dS_reliable": variant.get("max_abs_dS_reliable"),
                "max_abs_dS_on_0_b80": variant.get("max_abs_dS_on_0_b80"),
                "accept_dS_lt_0p05": variant.get("accept_dS_lt_0p05"),
                "b50_in_km_ci": variant.get("b50_in_km_ci"),
            }
        )
    return pd.DataFrame(rows)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-date", default=date.today().isoformat())
    ap.add_argument("--models", type=Path, default=Path("results/esp_survival_big_censored_refit/2026-07-15/esp_models_big_censored_refit.csv"))
    ap.add_argument("--mc-variants", type=Path, default=Path("results/esp_survival_big_censored_refit/2026-07-15/mc_vintage_variants.csv"))
    ap.add_argument("--run-months", type=Path, default=Path("results/esp_time_map_refit/2026-07-15/tables/time_map_run_month_dataset.csv"))
    ap.add_argument("--time-map", type=Path, default=Path("results/esp_time_map_refit/2026-07-15/esp_time_map.csv"))
    ap.add_argument("--source-bundle-date", default=C.BUNDLE_DATE)
    ap.add_argument("--pp-master", type=Path, default=None)
    ap.add_argument("--prediction-workbook", type=Path, default=None)
    ap.add_argument("--equipment-big", type=Path, default=None)
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    args = ap.parse_args()

    out = results_dir("production_risk_mc_variant_validation", args.out_date)
    tables = out / "tables"
    model_dir = out / "variant_models"

    print("[1/4] loading E inputs ...")
    base_models = pd.read_csv(args.models, encoding="utf-8-sig")
    variants = pd.read_csv(args.mc_variants, encoding="utf-8-sig")
    run_months = pd.read_csv(args.run_months, encoding="utf-8-sig")
    time_map_full = pd.read_csv(args.time_map, encoding="utf-8-sig")

    print("[2/4] rebuilding D idle gate ...")
    plan = crosswalk.load_plan(args.forecast_start, args.horizon_end, master_path=args.pp_master)
    well_field = failure_rate.build_well_field(plan)
    idle_reattr, adjusted_counts = D.idle_reattribution(run_months, well_field)
    gated_idle = D._gated_idle_map(time_map_full, idle_reattr)
    idle_reattr.to_csv(tables / "E_idle_reattribution.csv", index=False, encoding="utf-8-sig")
    gated_idle.to_csv(tables / "E_time_map_gated_idle_candidate.csv", index=False, encoding="utf-8-sig")

    print("[3/4] writing Mc variant registries ...")
    registry_index = write_variant_models(base_models, variants, model_dir)
    registry_index.to_csv(tables / "E_mc_variant_registry_index.csv", index=False, encoding="utf-8-sig")

    cfg = C.RunConfig(
        forecast_start=args.forecast_start,
        horizon_end=args.horizon_end,
        bundle_date=args.source_bundle_date,
        pp_master_path=args.pp_master,
        prediction_workbook_path=args.prediction_workbook,
        equipment_big_path=args.equipment_big,
        fact_through_month=D.FACT_LAST,
        write_excel=False,
        full_tables=False,
    )
    esp_source = crosswalk.load_esp_source(args.source_bundle_date, args.prediction_workbook)

    print("[4/4] replaying Mc variants with manual factors disabled ...")
    monthly_frames = []
    for rec in registry_index[registry_index["status"].eq("written")].itertuples(index=False):
        name = f"E_mc_{rec.mc_window}"
        print(f"      replaying {name} ...", flush=True)
        bundle = D.make_bundle(out, name, Path(rec.models_path), gated_idle, args.source_bundle_date)
        monthly_frames.append(D.run_failure_rate_for_bundle(plan, esp_source, cfg, bundle))

    if not monthly_frames:
        raise RuntimeError("No usable Mc variants were written")

    ladder, monthly = D.summarize_ladder(monthly_frames, adjusted_counts)
    ladder = ladder.merge(
        registry_index.add_prefix("variant_"),
        left_on=ladder["d_scenario"].str.replace("E_mc_", "", regex=False),
        right_on="variant_mc_window",
        how="left",
    ).drop(columns=["key_0"], errors="ignore")
    ladder["passes_e_gate"] = ladder["field"].eq("Мирнинский УН") & ladder["passes_0p90_1p10"]

    monthly.to_csv(tables / "E_mc_variant_monthly.csv", index=False, encoding="utf-8-sig")
    ladder.to_csv(tables / "E_mc_variant_ladder.csv", index=False, encoding="utf-8-sig")
    focus = ladder[ladder["field"].isin(FOCUS_FIELDS)].copy()
    focus.to_csv(tables / "E_mc_variant_ladder_focus.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "script": "scripts/run/production_risk_mc_variant_validation.py",
        "models": str(args.models),
        "mc_variants": str(args.mc_variants),
        "run_months": str(args.run_months),
        "time_map": str(args.time_map),
        "scenario": "D placement map + gated idle hazard, Mc registry row varied only",
        "manual_calibration": "disabled in-process",
        "outputs": [
            "tables/E_mc_variant_registry_index.csv",
            "tables/E_mc_variant_ladder.csv",
            "tables/E_mc_variant_ladder_focus.csv",
            "tables/E_mc_variant_monthly.csv",
        ],
    }
    (out / "E_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\noutputs -> {out}")
    cols = [
        "d_scenario",
        "field",
        "observed_failures",
        "predicted_failures",
        "fact_model_ratio",
        "passes_0p90_1p10",
        "variant_b50",
        "variant_b80",
        "variant_accept_dS_lt_0p05",
        "variant_b50_in_km_ci",
    ]
    print("\n=== E Mc variant ladder focus ===")
    print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
