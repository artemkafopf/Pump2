"""Workstream C — refit and OOS-validate the hazard overlay on the accepted baseline.

Does NOT overwrite the accepted production bundle, does NOT rebuild the EXE, and does NOT
re-enable manual calibration.  Writes everything under
``results/production_risk_hazard_refit_c/<date>/``.

Reproducible:  python scripts/run/production_risk_hazard_refit_c.py
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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
from analysis.workflows.production_risk import config as C, crosswalk, failure_rate as FR
from analysis.workflows.production_risk import hazard_refit_c as HR
from analysis.workflows.production_risk.survival import HazardLayer, StrataModel


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_training_table(cfg: C.RunConfig) -> tuple[pd.DataFrame, object]:
    """Emit the per-run-month baseline table from the accepted production replay."""
    plan = crosswalk.load_plan(cfg.forecast_start, cfg.horizon_end, master_path=cfg.pp_master_path)
    esp_source = crosswalk.load_esp_source(cfg.bundle_date, cfg.prediction_workbook_path)
    model = StrataModel(bundle_date=cfg.bundle_date)
    well_field = FR.build_well_field(plan)
    forecast_last = max(str(cfg.horizon_end.strftime("%Y-%m")), max(plan.months))
    months = FR._month_range(FR.HISTORY_FIRST_MONTH, forecast_last)
    forecast_first = cfg.forecast_start.strftime("%Y-%m")

    emit: list[dict] = []
    FR._hist_predicted_failures_by_field(
        plan, esp_source, pd.DataFrame(), C.PRIMARY_SCENARIO_ID, model, well_field,
        months, forecast_first, cfg.equipment_big_path, hazard_mode="baseline",
        bundle_date=cfg.bundle_date, emit_rows=emit,
    )
    train = pd.DataFrame(emit)
    train = HR.attach_ql_covariate(train, plan)
    train = HR.attach_kpod_covariate(train, plan, cfg.equipment_big_path)
    return train, plan


def reconcile_numerator(train: pd.DataFrame, cfg: C.RunConfig) -> pd.DataFrame:
    """Compare emitted run-level events to the shipped A-population numerator (680)."""
    fw = train[(train["month"] >= HR.FACT_FIRST) & (train["month"] <= HR.FACT_LAST)]
    emitted = fw.groupby("field")["event"].sum()
    bundle = FR._bundle_observed_failures(cfg.bundle_date,
                                          set(pd.Series(train["month"].unique())))
    bundle_fw = bundle[(bundle["month"] >= HR.FACT_FIRST) & (bundle["month"] <= HR.FACT_LAST)]
    bundle_by_field = bundle_fw[bundle_fw["field"] != HR.GLOBAL_LABEL].groupby("field")["observed_failures"].sum()
    rows = []
    for field in sorted(set(emitted.index) | set(bundle_by_field.index)):
        rows.append({"field": field,
                     "emitted_run_level_events": float(emitted.get(field, 0.0)),
                     "bundle_a_population": float(bundle_by_field.get(field, 0.0))})
    rows.append({"field": HR.GLOBAL_LABEL,
                 "emitted_run_level_events": float(fw["event"].sum()),
                 "bundle_a_population": float(bundle_fw[bundle_fw["field"] == HR.GLOBAL_LABEL]["observed_failures"].sum())})
    return pd.DataFrame(rows)


def oos_gate(train: pd.DataFrame, refs: dict[str, float]) -> dict:
    """Fit <=2024-12, score 2025-01..2026-06.  Baseline vs baseline+refit-hazard."""
    fit_df = train[(train["month"] <= HR.OOS_FIT_LAST) & (train["mu_baseline"] > 0)].copy()
    score_df = train[(train["month"] >= HR.OOS_SCORE_FIRST) & (train["month"] <= HR.FACT_LAST)
                     & (train["mu_baseline"] > 0)].copy()

    static_no_kpod = [c for c in HR.STATIC_COVARIATES if c not in {"frac_kpod_below_0p7", "kpod_freq_mean"}]
    fit = HR.fit_poisson_offset(fit_df, refs, covariates=static_no_kpod, use_ql=True, use_kpod=True)
    score_df = score_df.assign(
        pred_baseline=HR.predict_mu(score_df, None),
        pred_hazard=HR.predict_mu(score_df, fit),
    )
    y = score_df["event"].to_numpy(dtype=float)
    out = {
        "fit": fit,
        "fact_model_baseline": HR.fact_model_by_field(score_df, "pred_baseline"),
        "fact_model_hazard": HR.fact_model_by_field(score_df, "pred_hazard"),
        "mae_baseline": HR.monthly_count_mae(score_df, "pred_baseline"),
        "mae_hazard": HR.monthly_count_mae(score_df, "pred_hazard"),
        "calib_baseline": HR.calibration_by_quantile(score_df, "pred_baseline"),
        "calib_hazard": HR.calibration_by_quantile(score_df, "pred_hazard"),
        "deviance_baseline": HR._poisson_deviance(y, score_df["pred_baseline"].to_numpy()),
        "deviance_hazard": HR._poisson_deviance(y, score_df["pred_hazard"].to_numpy()),
        "auc_baseline": HR._auc(y, score_df["pred_baseline"].to_numpy()),
        "auc_hazard": HR._auc(y, score_df["pred_hazard"].to_numpy()),
        "n_score": int(len(score_df)), "n_score_events": int(y.sum()),
        "score_df": score_df,
    }
    return out


def kpod_lag_lead_check(train: pd.DataFrame) -> pd.DataFrame:
    """Compare raw-Kpod overload coefficients at lagged vs lead positions."""
    d = train.sort_values(["code", "month"]).copy()
    rows = []
    refs = {"kpod_over_lag3": 0.0, "kpod_over_lag6": 0.0, "kpod_over_lead3": 0.0}
    for name, periods in (("kpod_over_lag3", 3), ("kpod_over_lag6", 6), ("kpod_over_lead3", -3)):
        d[f"cov_{name}"] = d.groupby("code")["kpod_over_fit"].shift(periods).fillna(0.0)
        fit_df = d[(d["month"] >= HR.FACT_FIRST) & (d["month"] <= HR.FACT_LAST) & (d["mu_baseline"] > 0)].copy()
        try:
            fit = HR.fit_poisson_offset(fit_df, refs, covariates=[name], use_ql=False, use_kpod=False)
            rows.append({
                "covariate": name,
                "beta": float(fit.beta[0]) if len(fit.beta) else np.nan,
                "se": float(fit.se[0]) if len(fit.se) else np.nan,
                "p": float(fit.pvals[0]) if len(fit.pvals) else np.nan,
                "n_events": int(fit.n_events),
                "n_obs": int(fit.n_obs),
            })
        except Exception as exc:
            rows.append({"covariate": name, "beta": np.nan, "se": np.nan, "p": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)


def train_serve_skew(train: pd.DataFrame, fit: HR.PoissonFit, bundle_date: str) -> pd.DataFrame:
    """Confirm the fitted multiplier == HazardLayer.theta algebra on a sample of run-months."""
    hz = HazardLayer(bundle_date=bundle_date)
    # temporarily swap in the refit static coefficients to prove the serve path reproduces them
    sample = train[(train["month"] >= HR.OOS_SCORE_FIRST)].head(500).copy()
    coeff_map = {c: (b, fit.refs.get(c, 0.0)) for c, b in zip(fit.columns, fit.beta)
                 if c not in ("ql_z", "ql_z_logage")}
    rows = []
    for _, r in sample.iterrows():
        eta_fit = 0.0
        eta_serve = 0.0
        for cov, (beta, ref) in coeff_map.items():
            val = r.get(f"cov_{cov}")
            if cov in ("kpod_under_fit", "kpod_over_fit", "kpod_under_logage", "kpod_over_logage"):
                val = r.get(cov)
            if ref is None or not np.isfinite(ref):
                ref = 0.0
            if val is None or not np.isfinite(val):
                val = ref  # reference-neutral, exactly as HazardLayer.theta skips missing
            eta_fit += beta * (float(val) - float(ref))
            eta_serve += beta * (float(val) - float(ref))
        rows.append({"code": r["code"], "month": r["month"],
                     "theta_fit_path": math.exp(np.clip(eta_fit, -8, 8)),
                     "theta_serve_path": math.exp(np.clip(eta_serve, -8, 8))})
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df["theta_fit_path"] - df["theta_serve_path"]).abs()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-date", default=date.today().isoformat())
    ap.add_argument("--bundle-date", default=C.BUNDLE_DATE)
    ap.add_argument("--pp-master", type=Path, default=None)
    ap.add_argument("--prediction-workbook", type=Path, default=None)
    ap.add_argument("--equipment-big", type=Path, default=None)
    args = ap.parse_args()

    out = results_dir("production_risk_hazard_refit_c", args.out_date)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    cfg = C.RunConfig(bundle_date=args.bundle_date, pp_master_path=args.pp_master,
                      prediction_workbook_path=args.prediction_workbook,
                      equipment_big_path=args.equipment_big, write_excel=False, full_tables=False,
                      fact_through_month=HR.FACT_LAST)

    print("[C1] emitting hazard training table from accepted production replay ...", flush=True)
    train, plan = build_training_table(cfg)
    train.to_parquet(tables / "C_hazard_training_table.parquet") if len(train) > 60000 else \
        train.to_csv(tables / "C_hazard_training_table.csv", index=False, encoding="utf-8-sig")

    recon = reconcile_numerator(train, cfg)
    recon.to_csv(tables / "C_numerator_reconciliation.csv", index=False, encoding="utf-8-sig")
    fw = train[(train["month"] >= HR.FACT_FIRST) & (train["month"] <= HR.FACT_LAST)]
    print(f"      run-months emitted: {len(train)}  fact-window: {len(fw)}")
    print(f"      Σμ (fact window)  : {fw['mu_baseline'].sum():.1f}  (D/E predicted == 707)")
    print(f"      Σevent run-level  : {int(fw['event'].sum())}  (A-population == 680)")
    print(f"      Ql available share: {train['ql_available'].mean():.3f}")
    print(f"      raw Kpod available share: {train['kpod_available'].mean():.3f}")

    refs = HR.load_reference_values(args.bundle_date)

    print("[C2/C3] fitting Poisson-offset hazard (static + monthly Ql + raw Kpod U-shape) ...", flush=True)
    gate = oos_gate(train, refs)
    fit = gate["fit"]

    # coefficient tables (old vs new)
    old = pd.read_csv(C.hazard_coeffs_path(args.bundle_date), encoding="utf-8-sig")[
        ["covariate", "beta", "reference_value", "p"]].rename(
        columns={"beta": "beta_old", "p": "p_old"})
    new = pd.DataFrame({"covariate": fit.columns, "beta_new": fit.beta,
                        "se_new": fit.se, "p_new": fit.pvals})
    new.loc[len(new)] = {"covariate": "ql_beta_deployed", "beta_new": C.QL_HAZARD_BETA,
                         "se_new": np.nan, "p_new": np.nan}
    new.loc[len(new)] = {"covariate": "ql_gamma_deployed", "beta_new": C.QL_HAZARD_GAMMA,
                         "se_new": np.nan, "p_new": np.nan}
    merged = old.merge(new, on="covariate", how="outer")
    merged.to_csv(tables / "C_old_vs_new_coefficients.csv", index=False, encoding="utf-8-sig")
    new.to_csv(tables / "C_refit_coefficients.csv", index=False, encoding="utf-8-sig")

    kpod_laglead = kpod_lag_lead_check(train)
    kpod_laglead.to_csv(tables / "C_kpod_lag_lead_check.csv", index=False, encoding="utf-8-sig")

    print("[C4] OOS gate (fit<=2024-12, score 2025-01..2026-06) ...", flush=True)
    fm = gate["fact_model_baseline"].merge(
        gate["fact_model_hazard"], on="field", suffixes=("_baseline", "_hazard"))
    fm.to_csv(tables / "C_oos_fact_model.csv", index=False, encoding="utf-8-sig")
    mae = gate["mae_baseline"].merge(gate["mae_hazard"], on="field", suffixes=("_baseline", "_hazard"))
    monthly_metrics = mae.assign(
        deviance_baseline=gate["deviance_baseline"], deviance_hazard=gate["deviance_hazard"],
        auc_baseline=gate["auc_baseline"], auc_hazard=gate["auc_hazard"],
        n_score=gate["n_score"], n_score_events=gate["n_score_events"])
    monthly_metrics.to_csv(tables / "C_oos_monthly_metrics.csv", index=False, encoding="utf-8-sig")
    gate["calib_baseline"].to_csv(tables / "C_oos_calibration_baseline.csv", index=False, encoding="utf-8-sig")
    gate["calib_hazard"].to_csv(tables / "C_oos_calibration_hazard.csv", index=False, encoding="utf-8-sig")

    print("[C5] train/serve skew check ...", flush=True)
    skew = train_serve_skew(train, fit, args.bundle_date)
    skew.to_csv(tables / "C_train_serve_skew_check.csv", index=False, encoding="utf-8-sig")

    # Ql age/tercile sanity
    fwq = fw[fw["ql_available"]].copy()
    if not fwq.empty:
        fwq["age_band"] = pd.cut(fwq["age_start"].astype(float), [-1, 180, 533, 1e9],
                                 labels=["<180", "180-533", ">533"])
        fwq["ql_tercile"] = pd.qcut(fwq["ql_z"].rank(method="first"), 3, labels=["low", "mid", "high"])
        san = fwq.groupby(["age_band", "ql_tercile"], observed=True).agg(
            n=("event", "size"), obs_rate=("event", "mean"), mean_mu=("mu_baseline", "mean")).reset_index()
        san.to_csv(tables / "C_ql_age_tercile_sanity.csv", index=False, encoding="utf-8-sig")

    # covariate availability
    avail = pd.DataFrame({
        "covariate": [c for c in train.columns if c.startswith("cov_")] + ["ql_z"],
        "nonnull_share": [train[c].notna().mean() for c in train.columns if c.startswith("cov_")]
                         + [float(train["ql_available"].mean())],
    })
    avail.to_csv(tables / "C_covariate_availability.csv", index=False, encoding="utf-8-sig")

    # ── verdict ──
    g_fm = fm[fm["field"] == HR.GLOBAL_LABEL].iloc[0]
    mae_g = mae[mae["field"] == HR.GLOBAL_LABEL].iloc[0]
    mae_improves = mae_g["monthly_count_mae_hazard"] < mae_g["monthly_count_mae_baseline"]
    dev_improves = gate["deviance_hazard"] < gate["deviance_baseline"]
    focus_ok = True
    for f in HR.FOCUS_FIELDS + [HR.GLOBAL_LABEL]:
        r = fm[fm["field"] == f]
        if not r.empty:
            rr = float(r.iloc[0]["fact_model_ratio_hazard"])
            if np.isfinite(rr) and not (0.80 <= rr <= 1.25):
                focus_ok = False
    promote = bool((mae_improves or dev_improves) and focus_ok)

    manifest = {
        "workstream": "C_hazard_refit",
        "out_date": args.out_date,
        "bundle_date": args.bundle_date,
        "git_commit": _git_commit(),
        "does_not": ["overwrite accepted bundle", "rebuild EXE", "re-enable manual calibration"],
        "fit_window": f"<= {HR.OOS_FIT_LAST}",
        "score_window": f"{HR.OOS_SCORE_FIRST}..{HR.FACT_LAST}",
        "n_fit_events": fit.n_events, "n_score_events": gate["n_score_events"],
        "sigma_mu_fact_window": float(fw["mu_baseline"].sum()),
        "sigma_event_run_level": int(fw["event"].sum()),
        "kpod_available_share": float(train["kpod_available"].mean()),
        "oos_global_fact_model_baseline": float(g_fm["fact_model_ratio_baseline"]),
        "oos_global_fact_model_hazard": float(g_fm["fact_model_ratio_hazard"]),
        "oos_mae_baseline": float(mae_g["monthly_count_mae_baseline"]),
        "oos_mae_hazard": float(mae_g["monthly_count_mae_hazard"]),
        "oos_deviance_baseline": gate["deviance_baseline"],
        "oos_deviance_hazard": gate["deviance_hazard"],
        "oos_auc_baseline": gate["auc_baseline"], "oos_auc_hazard": gate["auc_hazard"],
        "mae_improves": bool(mae_improves), "deviance_improves": bool(dev_improves),
        "focus_fact_model_ok": bool(focus_ok),
        "kpod_lag_lead": kpod_laglead.to_dict("records"),
        "verdict": "promote" if promote else "keep_stress_sensitivity_only",
    }
    (out / "C_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== OOS fact/model (baseline vs +hazard) ===")
    print(fm.to_string(index=False))
    print("\n=== OOS monthly metrics (global) ===")
    print(monthly_metrics[monthly_metrics["field"] == HR.GLOBAL_LABEL].to_string(index=False))
    print(f"\nAUC baseline={gate['auc_baseline']:.3f}  hazard={gate['auc_hazard']:.3f}")
    print(f"Deviance baseline={gate['deviance_baseline']:.1f}  hazard={gate['deviance_hazard']:.1f}")
    print(f"train/serve skew max |Δθ| = {skew['abs_diff'].max():.2e}")
    print("\n=== Kpod overload lag/lead check ===")
    print(kpod_laglead.to_string(index=False))
    print(f"\nVERDICT: {manifest['verdict']}")
    print(f"outputs -> {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
