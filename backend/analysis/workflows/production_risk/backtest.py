"""Out-of-time backtest of the ESP baseline failure model against actual outcomes.

Self-contained by design: reads ``esp_models.csv`` and the v03 failure register directly
and implements the mixture survival S(t) inline, so it does not depend on the (actively
evolving) rest of the workflow.  Answers the trust question the deliverables rest on:
*do the model's per-well failure probabilities hold up out-of-time?*

Method (leakage-safe): at each as-of date A, find the pump that was actually running on A,
compute its age and predicted P(fail within H days) from A's information only, then compare
to whether it actually failed in (A, A+H].  Only fully-observed windows (A+H <= data cutoff)
are scored; runs whose observation ends inside the window without a failure are censored out.

Reported: discrimination (AUC vs an age-only baseline), calibration (quintiles), Brier.
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openpyxl

from analysis.paths import RESULTS_ROOT, resolve_v03_all_path

_BUNDLE = RESULTS_ROOT / "esp_survival_vba_models" / "2026-07-08"
MODELS_CSV = _BUNDLE / "esp_models.csv"
COEFF_CSV = _BUNDLE / "esp_cox_coeffs.csv"
RUNCOV_CSV = _BUNDLE / "esp_run_covariates.csv"
_COV_SKIP = {"well_key_norm", "run_seq", "install_date", "stratum_key", "covariate_available", "clock"}
HORIZONS = (90, 180, 365)
# Monthly as-of grid; each is used only when its full window lands before the data cutoff.
ASOF_GRID = [datetime(y, m, 1) for y in (2024, 2025) for m in (1, 3, 5, 7, 9, 11)]
_CODE_RE = re.compile(r"^([A-ZА-Я]+_\d+)")


def _norm(x):
    if x is None:
        return None
    s = str(x).strip().upper().replace(" ", "")
    m = _CODE_RE.match(s)
    return m.group(1) if m else (s or None)


def _ctr(v):
    s = str(v or "")
    return "brt" if "Борец" in s else "slb" if "Шлюмберже" in s else ("oth" if s.strip() else "Pooled")


def _sour(v):
    s = str(v or "")
    return "sour" if ("исл" in s and "екисл" not in s) else "nonsour"


def _load_models() -> dict[str, dict]:
    return pd.read_csv(MODELS_CSV, encoding="utf-8-sig").set_index("stratum").to_dict("index")


def _resolve(models, field, sour, ctr):
    for k in (f"{field}_{sour}_{ctr}", f"{field}_{sour}_Pooled", "Global_Pooled"):
        if k in models:
            return models[k], k
    return models["Global_Pooled"], "Global_Pooled"


def _S(t, pr):
    t = max(0.0, t)
    return (pr["w1"] * math.exp(-((t / pr["eta1"]) ** pr["beta1"]))
            + (1 - pr["w1"]) * math.exp(-((t / pr["eta2"]) ** pr["beta2"])))


def _load_hazard() -> list[tuple[str, float, float]]:
    """Enabled θ coefficients (covariate, beta, reference_value) from esp_cox_coeffs.csv."""
    cf = pd.read_csv(COEFF_CSV, encoding="utf-8-sig")
    cf = cf[cf["enabled"].astype(str).str.upper() == "TRUE"]
    return [(str(r["covariate"]), float(r["beta"]), float(r["reference_value"])) for _, r in cf.iterrows()]


def _load_run_covariates() -> dict[tuple[str, int], dict[str, float]]:
    """Per-(well, run_seq) covariate vectors from esp_run_covariates.csv (as the workbook uses)."""
    df = pd.read_csv(RUNCOV_CSV, encoding="utf-8-sig")
    cov_cols = [c for c in df.columns if c not in _COV_SKIP]
    out: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in df.iterrows():
        code = _norm(row["well_key_norm"])
        seq = row["run_seq"]
        if code is None or pd.isna(seq):
            continue
        d: dict[str, float] = {}
        for c in cov_cols:
            v = row[c]
            if pd.notna(v):
                try:
                    d[c] = float(v)
                except (TypeError, ValueError):
                    pass
        out[(code, int(seq))] = d
    return out


def _theta(covs, coeffs) -> float:
    """exp(Σ βᵢ·(xᵢ − refᵢ)) over covariates that are present (mirrors HazardLayer.theta)."""
    if not covs:
        return 1.0
    total, used = 0.0, 0
    for cov, beta, ref in coeffs:
        v = covs.get(cov)
        if v is None or not np.isfinite(v):
            continue
        total += beta * (v - ref)
        used += 1
    return math.exp(total) if used else 1.0


def _apply_theta(pr, theta):
    """Shift the mixture scale by θ (mirrors HazardLayer.apply_theta)."""
    if theta <= 0 or abs(theta - 1.0) < 1e-12:
        return pr
    out = dict(pr)
    out["eta1"] = pr["eta1"] / (theta ** (1.0 / pr["beta1"]))
    out["eta2"] = pr["eta2"] / (theta ** (1.0 / pr["beta2"]))
    return out


def _load_runs() -> tuple[dict[str, list[dict]], datetime]:
    wb = openpyxl.load_workbook(resolve_v03_all_path(), read_only=True, data_only=True)
    ws = wb["Свод"]
    runs: dict[str, list[dict]] = defaultdict(list)
    cutoff = datetime(2000, 1, 1)
    for r in ws.iter_rows(min_row=2, values_only=True):
        code = _norm(r[2])
        if code is None:
            continue
        mount = r[6] if isinstance(r[6], datetime) else None
        stop = r[7] if isinstance(r[7], datetime) else None
        for d in (mount, stop):
            if d is not None:
                cutoff = max(cutoff, d)
        runs[code].append(dict(field=str(r[0]).strip() if r[0] else None,
                               ctr=_ctr(r[3]), sour=_sour(r[74]),
                               mount=mount, stop=stop, ff=r[75], seq_ord=None))
    wb.close()
    # covariate file keys runs by within-well ordinal (mount order), NOT v03 «Нспуска».
    for rr in runs.values():
        for i, run in enumerate(sorted([x for x in rr if x["mount"]], key=lambda x: x["mount"]), start=1):
            run["seq_ord"] = i
    return runs, cutoff


def _active_at(rr, A):
    cand = [x for x in rr if x["mount"] and x["mount"] <= A and (x["stop"] is None or x["stop"] > A)]
    return max(cand, key=lambda x: x["mount"]) if cand else None


def _auc(y, score) -> float:
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def _build(models, runs, cutoff, horizon, coeffs, covmap) -> pd.DataFrame:
    rows, excluded = [], 0
    for A in ASOF_GRID:
        if A + timedelta(days=horizon) > cutoff:
            continue
        Wend = A + timedelta(days=horizon)
        for code, rr in runs.items():
            run = _active_at(rr, A)
            if run is None:
                continue
            pr, skey = _resolve(models, run["field"], run["sour"], run["ctr"])
            up = float(pr.get("uptime_factor", 0.9)) or 0.9
            age = max(0.0, (A - run["mount"]).days) * up
            s0 = _S(age, pr)
            pfail = 0.0 if s0 <= 0 else max(0.0, min(1.0, 1 - _S(age + horizon * up, pr) / s0))
            # hazard-adjusted prediction from the same run's θ covariates
            covs = covmap.get((code, run["seq_ord"])) if run["seq_ord"] is not None else None
            theta = _theta(covs, coeffs)
            pr_h = _apply_theta(pr, theta)
            s0h = _S(age, pr_h)
            pfail_h = 0.0 if s0h <= 0 else max(0.0, min(1.0, 1 - _S(age + horizon * up, pr_h) / s0h))
            t_end = run["stop"] if run["stop"] is not None else cutoff
            if run["ff"] == 1 and run["stop"] is not None and A < run["stop"] <= Wend:
                y = 1
            elif t_end > Wend:
                y = 0
            else:
                excluded += 1
                continue
            rows.append(dict(code=code, asof=A.date().isoformat(), stratum=skey,
                             age=age, pfail=pfail, pfail_hazard=pfail_h, theta=theta,
                             has_cov=bool(covs), y=y, horizon=horizon))
    df = pd.DataFrame(rows)
    df.attrs["excluded_censored"] = excluded
    return df


def _calibration(df, q=5) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.qcut(d.pfail.rank(method="first"), q, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), pred=("pfail", "mean"), obs=("y", "mean")).reset_index()


def run_backtest() -> dict:
    models = _load_models()
    coeffs = _load_hazard()
    covmap = _load_run_covariates()
    runs, cutoff = _load_runs()
    out_dir = RESULTS_ROOT / "production_risk_backtest" / datetime.today().date().isoformat()
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"data cutoff = {cutoff.date()}   as-of grid = {len(ASOF_GRID)} dates   "
          f"θ coeffs = {len(coeffs)}   run-covariate rows = {len(covmap)}")
    summary_rows, cal_frames, byasof_frames = [], [], []
    for H in HORIZONS:
        df = _build(models, runs, cutoff, H, coeffs, covmap)
        cov = df[df["has_cov"]]                        # subset where θ actually differs from 1
        for label, d in (("all", df), ("with_covariates", cov)):
            if d.empty:
                continue
            summary_rows.append(dict(
                horizon=H, set=label, n=len(d),
                theta_ne1_pct=round(100 * (d.theta.round(6) != 1.0).mean(), 1),
                obs_rate=round(d.y.mean(), 3),
                pred_base=round(d.pfail.mean(), 3), pred_hazard=round(d.pfail_hazard.mean(), 3),
                auc_base=round(_auc(d.y.values, d.pfail.values), 3),
                auc_hazard=round(_auc(d.y.values, d.pfail_hazard.values), 3),
                auc_age_only=round(_auc(d.y.values, d.age.values), 3),
                brier_base=round(float(np.mean((d.pfail - d.y) ** 2)), 4),
                brier_hazard=round(float(np.mean((d.pfail_hazard - d.y) ** 2)), 4),
            ))
        cal_frames.append(_calibration(df).assign(horizon=H))
        for _, g in df.groupby("asof"):
            byasof_frames.append(dict(horizon=H, asof=g["asof"].iloc[0], n=len(g),
                                      obs=round(g.y.mean(), 3),
                                      auc_base=round(_auc(g.y.values, g.pfail.values), 3),
                                      auc_hazard=round(_auc(g.y.values, g.pfail_hazard.values), 3)))

    summary = pd.DataFrame(summary_rows)
    cal_all = pd.concat(cal_frames, ignore_index=True)
    byasof = pd.DataFrame(byasof_frames)
    summary.to_csv(out_dir / "tables" / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    cal_all.to_csv(out_dir / "tables" / "backtest_calibration.csv", index=False, encoding="utf-8-sig")
    byasof.to_csv(out_dir / "tables" / "backtest_by_asof.csv", index=False, encoding="utf-8-sig")

    print("\n=== BACKTEST: baseline vs baseline+hazard (θ overlay), identical observations ===")
    print(summary.to_string(index=False))
    print("\nΔAUC (hazard − base):")
    for _, r in summary.iterrows():
        print(f"  H={r['horizon']:>3} {r['set']:<15} n={r['n']:>5}  ΔAUC={r['auc_hazard']-r['auc_base']:+.3f}  "
              f"ΔBrier={r['brier_hazard']-r['brier_base']:+.4f}")
    print(f"\nresults -> {out_dir}")
    return dict(summary=summary, calibration=cal_all, by_asof=byasof, out_dir=out_dir)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run_backtest()
