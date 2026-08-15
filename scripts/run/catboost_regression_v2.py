"""Run CatBoost direct TTF/RUL regression temporal holdouts."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from analysis.models.ml.regression_ttf import (
    EQUIPMENT_FEATURES,
    LANDMARKS,
    QUANTILES,
    RULRegressor,
    TTFRegressor,
    build_regression_frame,
    conditional_ipcw_weights,
    ipcw_weights,
)
from analysis.models.survival.temporal_holdout import stratum_baseline_surv, temporal_split
from analysis.paths import results_dir

CUTOFFS = ("2023-12-31", "2022-12-31")
HORIZONS = (90, 180, 365)


def _km_quantiles(train: pd.DataFrame, test: pd.DataFrame, probs=(0.9, 0.5, 0.1)) -> pd.DataFrame:
    from lifelines import KaplanMeierFitter

    def fit(g):
        km = KaplanMeierFitter()
        km.fit(g["tte"].to_numpy(float), g["event"].to_numpy(int))
        return km

    global_km = fit(train)
    by_s = {s: fit(g) for s, g in train.groupby("stratum_key", observed=True) if int(g["event"].sum()) >= 5}

    def q_for(km, surv_prob):
        sf = km.survival_function_.reset_index()
        time_col, surv_col = sf.columns[:2]
        hit = sf[sf[surv_col] <= surv_prob]
        if hit.empty:
            return float(sf[time_col].max())
        return float(hit[time_col].iloc[0])

    rows = []
    for s in test["stratum_key"]:
        km = by_s.get(s, global_km)
        vals = sorted(q_for(km, p) for p in probs)
        rows.append(vals)
    return pd.DataFrame(rows, index=test.index, columns=["b10_op_d", "b50_op_d", "b90_op_d"])


def _weighted_quantile_metrics(test: pd.DataFrame, pred: pd.DataFrame, weights: pd.Series, label: str) -> dict:
    fail = test["event"].astype(int).eq(1)
    w = weights.reindex(test.index).fillna(0.0)
    mask = fail & np.isfinite(pred["b50_op_d"]) & (w > 0)
    t = test.loc[mask, "tte"].astype(float)
    ww = w.loc[mask]
    b10 = pred.loc[mask, "b10_op_d"].astype(float)
    b50 = pred.loc[mask, "b50_op_d"].astype(float)
    b90 = pred.loc[mask, "b90_op_d"].astype(float)

    def wmean(x):
        return float(np.average(x, weights=ww)) if len(x) and float(ww.sum()) > 0 else float("nan")

    def pinball(q, yhat):
        e = t - yhat
        return wmean(np.maximum(q * e, (q - 1.0) * e))

    try:
        cidx = float(concordance_index(test["tte"].to_numpy(float), pred["b50_op_d"].to_numpy(float), test["event"].to_numpy(int)))
    except Exception:
        cidx = float("nan")
    return {
        "model": label,
        "n_fail_weighted": float(ww.sum()) if len(ww) else 0.0,
        "mae_b50_ipcw": wmean(np.abs(t - b50)),
        "pinball_b10": pinball(0.1, b10),
        "pinball_b50": pinball(0.5, b50),
        "pinball_b90": pinball(0.9, b90),
        "coverage_b10_b90": wmean((t >= b10) & (t <= b90)),
        "coverage_t_le_b50": wmean(t <= b50),
        "cindex_b50": cidx,
    }


def _ttf_holdout(frame: pd.DataFrame, cutoff: str, include_equipment: bool) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = temporal_split(frame, cutoff, date_col="install_date")
    model = TTFRegressor(include_equipment=include_equipment).fit(split.train)
    pred = model.predict_ttf(split.test)
    base = _km_quantiles(split.train, split.test)
    w_test = ipcw_weights(split.test)
    rows = []
    for label, p in (("catboost", pred), ("stratum_km_quantile", base)):
        row = _weighted_quantile_metrics(split.test, p, w_test, label)
        row["cutoff"] = cutoff
        rows.append(row)
    for h in HORIZONS:
        surv = stratum_baseline_surv(split.train, split.test, h)
        rows.append({"cutoff": cutoff, "model": f"stratum_km_surv_h{h}", "horizon_op_d": h, "cindex_surv": np.nanmean(surv)})
    fi = pd.DataFrame()
    if getattr(model.mean_model_, "get_feature_importance", None) is not None:
        imp = model.mean_model_.get_feature_importance(type="PredictionValuesChange")
        fi = pd.DataFrame({"feature": model.features_, "importance": imp, "cutoff": cutoff, "model": "mean_ttf"})
    return rows, pred, base, fi


def _rul_holdout(frame: pd.DataFrame, cutoff: str, include_equipment: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = temporal_split(frame, cutoff, date_col="install_date")
    model = RULRegressor(include_equipment=include_equipment).fit(split.train)
    rows = []
    calibration = []
    for lmk in LANDMARKS:
        test_l = split.test[split.test["tte"].astype(float) > float(lmk)].copy()
        if test_l.empty:
            continue
        pred = model.predict_rul(test_l, lmk)
        pred = model.to_calendar(pred, test_l["stratum_key"])
        metric_df = test_l.copy()
        metric_df["tte"] = metric_df["tte"].astype(float) - float(lmk)
        metric_df["run_days"] = pd.to_numeric(metric_df["run_days"], errors="coerce")
        w = conditional_ipcw_weights(test_l.assign(age_op_d=float(lmk)), split.train)
        tmp = pred.rename(columns={"rul_b10_op_d": "b10_op_d", "rul_b50_op_d": "b50_op_d", "rul_b90_op_d": "b90_op_d"})
        row = _weighted_quantile_metrics(metric_df, tmp, w, "catboost_rul")
        row.update({"cutoff": cutoff, "landmark_op_d": lmk})
        rows.append(row)
        calibration.append({
            "cutoff": cutoff,
            "landmark_op_d": lmk,
            "model": "catboost_rul",
            "coverage_b10_b90": row["coverage_b10_b90"],
            "coverage_t_le_b50": row["coverage_t_le_b50"],
        })
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def main() -> None:
    out = results_dir("catboost_regression_v2")
    match_table = out / "tables" / "equipment_match_rate.csv"
    frame = build_regression_frame(include_equipment=True, equipment_match_table=match_table)

    holdout_rows, fi_rows, cal_rows, rul_rows = [], [], [], []
    ablation_rows = []
    for cutoff in CUTOFFS:
        rows, _, _, fi = _ttf_holdout(frame, cutoff, include_equipment=True)
        holdout_rows.extend(rows)
        fi_rows.append(fi)
        rrows, crows = _rul_holdout(frame, cutoff, include_equipment=True)
        rul_rows.append(rrows)
        cal_rows.append(crows)

        rows_noeq, _, _, _ = _ttf_holdout(frame.drop(columns=[c for c in EQUIPMENT_FEATURES + ["equipment_matched"] if c in frame.columns]), cutoff, include_equipment=False)
        cb = next(r for r in rows if r["model"] == "catboost")
        cb_no = next(r for r in rows_noeq if r["model"] == "catboost")
        ablation_rows.append({
            "cutoff": cutoff,
            "cindex_with_equipment": cb["cindex_b50"],
            "cindex_without_equipment": cb_no["cindex_b50"],
            "delta_cindex": cb["cindex_b50"] - cb_no["cindex_b50"],
        })

    pd.DataFrame(holdout_rows).to_csv(out / "tables" / "holdout_metrics.csv", index=False)
    pd.concat(rul_rows, ignore_index=True).to_csv(out / "tables" / "rul_holdout_metrics.csv", index=False)
    pd.concat(cal_rows, ignore_index=True).to_csv(out / "tables" / "calibration.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "tables" / "equipment_ablation.csv", index=False)
    pd.concat(fi_rows, ignore_index=True).to_csv(out / "tables" / "feature_importance.csv", index=False)
    uf = RULRegressor().fit(frame).uptime_factors_.rename("uptime_factor").reset_index()
    uf.to_csv(out / "tables" / "uptime_factors.csv", index=False)

    (out / "reports" / "README.md").write_text(
        "# CatBoost Direct Regression v2\n\n"
        "Gates confirmed: IPCW primary; RMST pseudo-observation cross-check planned; "
        "TTF quantiles 0.1/0.5/0.9; RUL landmarks 0/30/90/180/365/730 op-days; "
        "cutoffs 2023-12-31 and 2022-12-31; full fleet/all strata; exploratory "
        "equipment block included with ablation; per-mode regression skipped.\n\n"
        "This run is an independent Weibull-free regression cross-check. Outputs are "
        "in operating days unless explicitly suffixed `_cal_d`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
