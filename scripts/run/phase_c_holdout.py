"""Phase C C5 — temporal holdout: the stack's first out-of-sample test
(output: phase_c_holdout).

Split by install date (train = installs ≤ cutoff, test = installs > cutoff), fit on
train only, score on test with censoring-aware metrics.  Two models compared:

  A. baseline  — stratum-only KM survival at the horizon (the K=2/Weibull null)
  B. θ         — stratified Cox on the merged θ covariates (C4)

Metrics on test: C-index at each horizon (well-cluster bootstrap CI), IPCW Brier at
90/180/365 d, and a calibration table.  Horizons respect the young test cohort's
follow-up: n-at-risk is reported per horizon.  The deliverable is the **honest
delta B − A** — if θ adds little out of sample, that is the finding.

Primary cutoff 2023-12-31; sensitivity cutoff 2022-12-31.

Run:
    python scripts/run/phase_c_holdout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.models.survival.temporal_holdout import (
    temporal_split, stratum_baseline_surv, ipcw_brier, cindex_at,
    cindex_bootstrap_ci, n_at_risk, calibration_table, _censoring_km,
)

# Merged θ covariates (the C4 VIF-kept survivors + numeric cohort adjusters).
THETA_NUMERIC = [
    "log_h2s_proxy_mg_l", "log_glf_mean_opdays",
    "freq_above_55hz_pct_early", "load_mean", "load_std_early", "kpod_freq_mean",
    "frac_kpod_below_0p7", "n_freq_steps_per_100d", "n_restarts_per_100d",
    "log_motor_power_kw", "curvature_missing", "vg_m", "pbubble_atm", "nominal_freq_hz",
    "log_run_seq", "log_days_since_prev_failure", "install_pre2020", "install_2023plus",
]
HORIZONS = [90, 180, 365]
CUTOFFS = [("2023-12-31", "primary"), ("2022-12-31", "sensitivity")]


def _fit_theta(train: pd.DataFrame):
    """Stratified Cox on θ covariates; drop <5-event strata; ridge fallback."""
    keep = ["well_key", "tte", "event", "stratum_key"] + THETA_NUMERIC
    tr = train[keep].copy()
    tr["tte"] = pd.to_numeric(tr["tte"], errors="coerce")
    tr = tr[tr["tte"] > 0].dropna(subset=THETA_NUMERIC + ["tte", "event"])
    ev = tr.groupby("stratum_key", observed=True)["event"].transform("sum")
    tr = tr[ev >= 5]
    # drop zero-variance covariates on this sample
    cols = [c for c in THETA_NUMERIC if tr[c].nunique(dropna=True) >= 2]
    tr = tr[["well_key", "tte", "event", "stratum_key"] + cols]
    for pen in (0.0, 0.1):
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(tr, "tte", "event", strata=["stratum_key"], cluster_col="well_key",
                    formula=" + ".join(cols), robust=True)
            return cph, cols, tr
        except Exception:
            continue
    return None, cols, tr


def _theta_surv_at(cph, test: pd.DataFrame, cols: list[str], horizon: float,
                   valid_strata: set) -> np.ndarray:
    """Per-test-subject S(horizon) from the θ Cox (NaN where stratum unseen)."""
    out = np.full(len(test), np.nan)
    mask = test["stratum_key"].isin(valid_strata).to_numpy()
    if mask.any():
        sub = test.loc[mask, cols + ["stratum_key"]]
        try:
            sf = cph.predict_survival_function(sub, times=[horizon])
            out[mask] = sf.iloc[0].to_numpy()
        except Exception:
            pass
    return out


def run_cutoff(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = temporal_split(df, cutoff)
    train_all, test_all = split.train, split.test

    # complete-case test/train on θ covariates (telemetry subpopulation)
    cc_cols = THETA_NUMERIC + ["tte", "event"]
    train = train_all[train_all["tte"] > 0].dropna(subset=cc_cols)
    test = test_all[test_all["tte"] > 0].dropna(subset=cc_cols).reset_index(drop=True)

    cph, cols, tr_used = _fit_theta(train)
    valid_strata = set(tr_used["stratum_key"].unique())
    # evaluate only on test rows whose stratum was fit on train
    test = test[test["stratum_key"].isin(valid_strata)].reset_index(drop=True)
    cens = _censoring_km(tr_used)

    rows = []
    for h in HORIZONS:
        s_base = stratum_baseline_surv(tr_used, test, h)
        s_theta = _theta_surv_at(cph, test, cols, h, valid_strata) if cph is not None else np.full(len(test), np.nan)
        ci_b = cindex_at(test, s_base); ci_t = cindex_at(test, s_theta)
        ci_t_lo, ci_t_hi = cindex_bootstrap_ci(test, s_theta, n_boot=150)
        br_b, nb = ipcw_brier(test, s_base, h, cens)
        br_t, nt = ipcw_brier(test, s_theta, h, cens)
        rows.append({
            "cutoff": cutoff, "horizon_d": h,
            "n_test": len(test), "n_at_risk": n_at_risk(test, h),
            "cindex_baseline": round(ci_b, 4), "cindex_theta": round(ci_t, 4),
            "cindex_theta_ci_lo": round(ci_t_lo, 4) if np.isfinite(ci_t_lo) else None,
            "cindex_theta_ci_hi": round(ci_t_hi, 4) if np.isfinite(ci_t_hi) else None,
            "cindex_delta": round(ci_t - ci_b, 4) if np.isfinite(ci_t) and np.isfinite(ci_b) else None,
            "brier_baseline": round(br_b, 4), "brier_theta": round(br_t, 4),
            "brier_delta": round(br_t - br_b, 4) if np.isfinite(br_t) and np.isfinite(br_b) else None,
        })
    deltas = pd.DataFrame(rows)
    calib = calibration_table(test, _theta_surv_at(cph, test, cols, 180, valid_strata), 180)
    calib.insert(0, "cutoff", cutoff)
    print(f"[C5] cutoff={cutoff}: train_all={len(train_all)}/{int(train_all.event.sum())}ev, "
          f"test_all={len(test_all)}/{int(test_all.event.sum())}ev; "
          f"complete-case test evaluated on n={len(test)}")
    return deltas, calib


def main() -> None:
    out = results_dir("phase_c_holdout")
    tbl, figs = out / "tables", out / "figures"

    df = build_competing_risks_df(tte_col="ttf_mix")

    all_deltas, all_calib = [], []
    for cutoff, _ in CUTOFFS:
        d, c = run_cutoff(df, cutoff)
        all_deltas.append(d); all_calib.append(c)
        print(d.to_string(index=False)); print()
    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas.to_csv(tbl / "c5_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_calib, ignore_index=True).to_csv(
        tbl / "c5_calibration.csv", index=False, encoding="utf-8-sig")

    # figure: C-index baseline vs θ per horizon (primary cutoff)
    prim = deltas[deltas["cutoff"] == "2023-12-31"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(prim)); w = 0.35
    ax1.bar(x - w/2, prim["cindex_baseline"], w, label="baseline (strata)", color="#B0B0B0")
    ax1.bar(x + w/2, prim["cindex_theta"], w, label="+ θ", color="#4C78A8")
    ax1.axhline(0.5, color="k", ls=":", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{h}d" for h in prim["horizon_d"]])
    ax1.set_ylabel("C-index (test)"); ax1.set_ylim(0.4, 0.8)
    ax1.set_title("Out-of-sample discrimination"); ax1.legend(fontsize=8)
    ax2.bar(x - w/2, prim["brier_baseline"], w, label="baseline", color="#B0B0B0")
    ax2.bar(x + w/2, prim["brier_theta"], w, label="+ θ", color="#F58518")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{h}d" for h in prim["horizon_d"]])
    ax2.set_ylabel("IPCW Brier (lower better)")
    ax2.set_title("Out-of-sample calibration/accuracy"); ax2.legend(fontsize=8)
    fig.suptitle("C5 — temporal holdout (train ≤ 2023-12-31, test > 2023-12-31)", fontsize=11)
    fig.tight_layout(); fig.savefig(figs / "c5_holdout_bars.png", dpi=140); plt.close(fig)

    print(f"[C5] Outputs written to: {out}")


if __name__ == "__main__":
    main()
