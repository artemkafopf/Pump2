"""CatBoost vs Weibull monthly ESP failure-rate comparison (the deliverable).

Runs the production-risk pipeline with the CatBoost comparison flag on, then renders the
verdict over the fact window (2024-01..fact_through), per УН and fleet-global:

  * ``figures/`` — Fact vs Weibull-base vs Weibull-hazard vs CatBoost(cross-fit) monthly
    failure-rate, per major УН and global;
  * ``tables/F_failure_rate_compare_monthly.csv`` — every model line on one table;
  * ``tables/verdict_mae.csv`` — per-УН and pooled monthly-rate MAE/RMSE of each model line
    vs fact (cross-fit CatBoost is the headline; in-sample is the memorisation-bounded
    secondary row);
  * ``tables/catboost_coverage.csv`` — CatBoost covariate-coverage share and past-b90
    tail-dependence share per УН.

The Weibull path and its numbers are untouched; this only *adds* the CatBoost line.
VBA / launcher / EXE are deliberately not involved (comparison runs from Python/CLI only).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk.failure_rate import GLOBAL_LABEL
from analysis.workflows.production_risk.run import run

# Line styling is prescribed by the handoff (§8.4): five lines per panel is unreadable, so
# the in-sample CatBoost line lives in the tables, not the headline chart.
_STYLE = {
    "observed_rate": dict(label="Факт", color="#1F77B4", ls="-", lw=2.2),
    "predicted_rate": dict(label="Прогноз (Weibull, база)", color="#D62728", ls="--", lw=1.8),
    "weibull_hazard_rate": dict(label="Прогноз (Weibull, hazard)", color="#FF7F0E", ls=":", lw=1.8),
    "predicted_rate_catboost_xfit": dict(label="Прогноз (CatBoost, cross-fit)", color="#2CA02C", ls="-.", lw=1.8),
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _fact_window(monthly: pd.DataFrame, last_obs: str) -> pd.DataFrame:
    m = monthly[(monthly["month"] >= "2024-01")].copy()
    if last_obs:
        m = m[m["month"] <= last_obs]
    return m


def _build_compare(base_m: pd.DataFrame, haz_m: pd.DataFrame, cov: dict) -> pd.DataFrame:
    haz = haz_m[["field", "month", "predicted_failures", "predicted_rate"]].rename(
        columns={"predicted_failures": "weibull_hazard_pred", "predicted_rate": "weibull_hazard_rate"}
    )
    out = base_m.merge(haz, on=["field", "month"], how="left")
    share_by_field = cov.get("catboost_covariate_share_by_field", {}) or {}
    tail_by_field = cov.get("catboost_past_b90_share_by_field", {}) or {}
    out["catboost_covariate_share"] = out["field"].map(lambda f: share_by_field.get(f, float("nan")))
    out["catboost_past_b90_share"] = out["field"].map(lambda f: tail_by_field.get(f, float("nan")))
    cols = {
        "field": "field",
        "month": "month",
        "fleet_size": "fleet_size",
        "observed_failures": "observed_failures",
        "predicted_failures": "weibull_base_pred",
        "weibull_hazard_pred": "weibull_hazard_pred",
        "predicted_failures_catboost": "catboost_pred",
        "predicted_failures_catboost_xfit": "catboost_xfit_pred",
        "observed_rate": "observed_rate",
        "predicted_rate": "weibull_base_rate",
        "weibull_hazard_rate": "weibull_hazard_rate",
        "predicted_rate_catboost": "catboost_rate",
        "predicted_rate_catboost_xfit": "catboost_xfit_rate",
        "catboost_covariate_share": "catboost_covariate_share",
        "catboost_past_b90_share": "catboost_past_b90_share",
    }
    present = {k: v for k, v in cols.items() if k in out.columns}
    return out[list(present)].rename(columns=present)


def _verdict(compare: pd.DataFrame, last_obs: str) -> pd.DataFrame:
    """Per-УН and pooled monthly-rate MAE/RMSE of each model line vs fact, fact window only."""
    fw = _fact_window(compare, last_obs)
    fw = fw[fw["observed_rate"].notna()]
    model_cols = {
        "weibull_base": "weibull_base_rate",
        "weibull_hazard": "weibull_hazard_rate",
        "catboost_insample": "catboost_rate",
        "catboost_xfit": "catboost_xfit_rate",
    }
    rows = []
    for field, g in fw.groupby("field"):
        row = {"field": field, "n_months": int(len(g))}
        obs = g["observed_rate"].to_numpy(float)
        for name, col in model_cols.items():
            if col not in g.columns:
                continue
            pred = g[col].to_numpy(float)
            mask = np.isfinite(pred) & np.isfinite(obs)
            if mask.sum() == 0:
                row[f"mae_{name}"] = float("nan")
                row[f"rmse_{name}"] = float("nan")
                continue
            err = pred[mask] - obs[mask]
            row[f"mae_{name}"] = float(np.mean(np.abs(err)))
            row[f"rmse_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        # headline: which line tracks fact best (by MAE)
        maes = {name: row.get(f"mae_{name}", float("nan")) for name in model_cols}
        finite = {k: v for k, v in maes.items() if np.isfinite(v)}
        row["best_by_mae"] = min(finite, key=finite.get) if finite else ""
        rows.append(row)
    verdict = pd.DataFrame(rows)
    # sort with ГЛОБАЛЬНО first, then by fleet-weight (n_months desc as a proxy)
    verdict["_g"] = verdict["field"] != GLOBAL_LABEL
    return verdict.sort_values(["_g", "field"]).drop(columns="_g").reset_index(drop=True)


def _plot_field(compare: pd.DataFrame, field: str, last_obs: str, out_png: Path) -> None:
    g = _fact_window(compare[compare["field"] == field], last_obs).sort_values("month")
    if g.empty:
        return
    x = pd.to_datetime(g["month"] + "-01")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for col, st in _STYLE.items():
        if col not in g.columns:
            continue
        y = g[col].to_numpy(float)
        ax.plot(x, y, label=st["label"], color=st["color"], linestyle=st["ls"], linewidth=st["lw"])
    label = "Флот (все УН)" if field == GLOBAL_LABEL else field
    ax.set_title(f"Интенсивность отказов УЭЦН — {label}", fontsize=11)
    ax.set_ylabel("отказов / скв. в мес")
    ax.set_xlabel("Месяц")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    ap.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    ap.add_argument("--bundle-date", default=C.BUNDLE_DATE)
    ap.add_argument("--pp-master", type=Path, default=None)
    ap.add_argument("--gtm-schedule", type=Path, default=None)
    ap.add_argument("--prediction-workbook", type=Path, default=None)
    ap.add_argument("--techregime-workbook", type=Path, default=None)
    ap.add_argument("--equipment-big", type=Path, default=None)
    ap.add_argument("--fact-through", default=None, help="last complete fact month (YYYY-MM)")
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
        fact_through_month=args.fact_through or C.DEFAULT_FACT_THROUGH_MONTH,
        write_excel=False,
        enable_catboost_compare=True,
    )
    res = run(cfg, write_excel=False)
    fr_base = res["failure_rate"]
    fr_haz = res["failure_rate_stress"]
    cov = fr_base.coverage
    last_obs = str(cov.get("last_observed_month", "") or "")

    compare = _build_compare(fr_base.monthly, fr_haz.monthly, cov)
    verdict = _verdict(compare, last_obs)

    out = results_dir("catboost_vs_weibull_failure_rate")
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    compare.to_csv(tables / "F_failure_rate_compare_monthly.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(tables / "verdict_mae.csv", index=False, encoding="utf-8-sig")

    share = cov.get("catboost_covariate_share_by_field", {}) or {}
    tail = cov.get("catboost_past_b90_share_by_field", {}) or {}
    cov_rows = [
        {"field": f, "catboost_covariate_full_share": share.get(f, float("nan")),
         "past_b90_eval_share": tail.get(f, float("nan"))}
        for f in sorted(set(share) | set(tail))
    ]
    cov_rows.append({
        "field": GLOBAL_LABEL,
        "catboost_covariate_full_share": cov.get("catboost_covariate_share", float("nan")),
        "past_b90_eval_share": cov.get("catboost_past_b90_share", float("nan")),
    })
    pd.DataFrame(cov_rows).to_csv(tables / "catboost_coverage.csv", index=False, encoding="utf-8-sig")

    for field in fr_base.chart_fields:
        safe = "global" if field == GLOBAL_LABEL else "".join(c if c.isalnum() else "_" for c in str(field))[:40]
        _plot_field(compare, field, last_obs, figures / f"failure_rate_{safe}.png")

    print("\n" + "=" * 78)
    print("CATBOOST vs WEIBULL — FAILURE-RATE VERDICT (fact window, monthly-rate MAE)")
    print("=" * 78)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = [c for c in ["field", "n_months", "mae_weibull_base", "mae_weibull_hazard",
                        "mae_catboost_insample", "mae_catboost_xfit", "best_by_mae"] if c in verdict.columns]
    print(verdict[show].to_string(index=False))
    print(f"\nCatBoost covariate-full share (global): {cov.get('catboost_covariate_share', float('nan')):.3f}")
    print(f"Past-b90 eval share (global): {cov.get('catboost_past_b90_share', float('nan')):.3f}")
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
