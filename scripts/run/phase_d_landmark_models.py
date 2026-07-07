"""Phase D — D3 + D4: landmark risk models and the alarm verdict.

The prediction view and the phase's deliverable.  On the **install-cohort** split
(train installs ≤ 2023-12-31, test after — the C5 convention; leakage rule 3), two
models compete out of sample for every (horizon H ∈ {30,60,90} × guard g ∈ {3,7,14}):

  A. **baseline** — age + stratum only: conditional KM window risk
     ``1 − S_str(L+g+H)/S_str(L)`` (the null to beat);
  B. **dynamic** — a stratified landmark Cox on op_age + screened trailing-window
     dynamics (so B = baseline's information *plus* dynamics; the delta isolates what
     dynamics add).

Judged on the test split only, per §0.1 capacity (top-5% / top-10%):
dynamic AUC + IPCW Brier vs baseline; precision / recall / lead-time / false-alarms
per pump-year; well-cluster bootstrap CIs on every delta.  Screening happens on train
only.  Secondary: the landmark-calendar split as a robustness read.

Outputs under ``results/phase_d_landmark_models/<date>/`` (+ ``phase_d_alarm_eval``):
  tables/alarm_grid.csv            — AUC/Brier/precision/recall/false-alarm across H×g
  tables/delta_ci.csv              — bootstrap CI of dynamic−baseline AUC per H×g
  tables/primary_operational.csv   — precision/recall/lead-time/FA at H=60,g=7
  tables/calibration_primary.csv   — predicted vs observed window risk (dynamic)
  tables/robustness_calendar.csv   — primary H×g on the landmark-calendar split
  tables/feature_set.csv           — screened dynamic features (train)
  figures/auc_delta.png, figures/lead_time.png
  logs/alarm.txt                   — includes the verdict line

Run (needs phase_d_landmarks.py first):
    python scripts/run/phase_d_landmark_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.landmark_features import (
    load_latest_landmark_frame, HORIZONS, GUARD_GAPS, DEFAULT_GUARD, WINDOWS,
)
from analysis.models.survival.landmark_cox import (
    install_cohort_split, landmark_calendar_split, fit_landmark_cox,
    predict_window_risk, baseline_window_risk,
)
from analysis.models.survival.screening_cox import univariate_screen, correlation_prune
from analysis.models.survival import temporal_holdout as th
from analysis.models.survival.alarm_metrics import (
    precision_recall_at_capacity, lead_time_distribution, false_alarms_per_pump_year,
    dynamic_auc, cluster_bootstrap_delta,
)

PRIMARY_H, PRIMARY_G = 60, DEFAULT_GUARD   # H=60 / g=7 (user-confirmed primary)
_STEMS = ["kprod_mean", "kprod_slope", "qliq_slope", "load_mean", "load_slope",
          "load_std", "load_exc", "freq_mean", "freq_std", "freq_step", "freq_days55",
          "restarts", "longest_idle", "rintake_mean", "rintake_slope", "rzab_slope",
          "drawdown", "watercut_mean", "watercut_jump", "gasfactor_mean", "gasfactor_slope"]


def _zscore(frame, cols, train_mask):
    d = frame.copy()
    for c in cols:
        mu, sd = d.loc[train_mask, c].mean(), d.loc[train_mask, c].std()
        d[c] = (d[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return d


def _screen(train, cands):
    uni = univariate_screen(train, cands, duration_col="tte_land", event_col="event_land")
    survivors = uni.loc[uni["pass_screen"], "covariate"].tolist()
    cindex = uni.set_index("covariate")["c_index"].to_dict()
    kept, _ = correlation_prune(train, survivors, cindex)
    return kept


def _eval_one(train, test, features, H, g, fit=None):
    """Score baseline + dynamic on the test split for one (H, g); return eval frame.

    ``fit`` (the landmark Cox) is horizon-agnostic — only the scoring time u=g+H
    differs — so the caller fits once and passes it in to avoid refitting per cell.
    """
    u = g + H
    y_col = f"y_H{H}_g{g}"
    base_risk = baseline_window_risk(train, test, g, H)
    if fit is None:
        fit = fit_landmark_cox(train, features)
    dyn_risk = predict_window_risk(fit, test, u)
    # dynamic falls back to baseline where the Cox couldn't score (unseen stratum etc.)
    dyn_risk = np.where(np.isfinite(dyn_risk), dyn_risk, base_risk)
    ev = test[["row_id", "well_key", "op_age", "tte_land", "event_land", y_col]].copy()
    ev = ev.rename(columns={y_col: "y"})
    ev["terminal_op_age"] = ev["op_age"] + ev["tte_land"]
    ev["base_risk"] = base_risk
    ev["dyn_risk"] = dyn_risk
    return ev, fit


def _metrics_row(ev, H, g, split_key):
    """Core out-of-sample metrics for one eval frame."""
    row = {"split": split_key, "H": H, "g": g,
           "n_test_landmarks": len(ev),
           "n_pos": int((ev["y"] == 1).sum()), "n_neg": int((ev["y"] == 0).sum())}
    row["auc_base"] = dynamic_auc(ev, "base_risk")
    row["auc_dyn"] = dynamic_auc(ev, "dyn_risk")
    row["auc_delta"] = (round(row["auc_dyn"] - row["auc_base"], 4)
                        if np.isfinite(row["auc_dyn"]) and np.isfinite(row["auc_base"]) else np.nan)
    # IPCW Brier at u=g+H on the landmark durations (reuse temporal_holdout)
    cens = th._censoring_km(ev, dur="tte_land", ev="event_land")
    for name, col in (("base", "base_risk"), ("dyn", "dyn_risk")):
        br, _ = th.ipcw_brier(ev.assign(_s=1 - ev[col]), (1 - ev[col]).to_numpy(),
                              g + H, cens, dur="tte_land", ev="event_land")
        row[f"brier_{name}"] = round(br, 4) if np.isfinite(br) else np.nan
    # operational at top-5% / 10% for the dynamic model
    pr = precision_recall_at_capacity(ev, "dyn_risk", k_fracs=(0.05, 0.10))
    prb = precision_recall_at_capacity(ev, "base_risk", k_fracs=(0.05, 0.10))
    for _, r in pr.iterrows():
        k = int(r["k_frac"] * 100)
        row[f"prec{k}_dyn"] = r["precision"]; row[f"recall{k}_dyn"] = r["recall"]
        row[f"fa{k}_dyn"] = false_alarms_per_pump_year(ev, "dyn_risk", r["k_frac"])
    for _, r in prb.iterrows():
        k = int(r["k_frac"] * 100)
        row[f"prec{k}_base"] = r["precision"]; row[f"recall{k}_base"] = r["recall"]
    return row


def main() -> None:
    out = results_dir("phase_d_landmark_models")
    eval_out = results_dir("phase_d_alarm_eval")
    tbl, figs, logs = out / "tables", out / "figures", out / "logs"
    log: list[str] = []

    frame = load_latest_landmark_frame()
    cands = [f"{s}_w{w}" for w in WINDOWS for s in _STEMS if f"{s}_w{w}" in frame.columns]
    zcols = cands + ["op_age"]
    dt = pd.to_datetime(frame["install_dt"], errors="coerce")
    train_mask = dt <= pd.Timestamp("2023-12-31")
    fz = _zscore(frame, zcols, train_mask)
    # keep an un-scaled op_age for baseline/age; z-scored op_age_z drives the Cox
    fz["op_age_z"] = fz["op_age"]
    fz["op_age"] = frame["op_age"]

    split = install_cohort_split(fz)
    train, test = split.train, split.test
    log.append(f"[D3] install-cohort split @ {split.cutoff}: train {len(train)} / "
               f"test {len(test)} landmarks")

    kept = _screen(train, cands)
    features = ["op_age_z"] + kept                     # dynamic = age + stratum + dynamics
    pd.DataFrame({"feature": features}).to_csv(tbl / "feature_set.csv", index=False)
    log.append(f"[D3] dynamic model features (age + {len(kept)} screened dynamics): "
               + ", ".join(features))
    fit = fit_landmark_cox(train, features)            # horizon-agnostic; fit once
    log.append(f"[D3] landmark Cox fit: success={fit.success} ({fit.message}); "
               f"{len(fit.strata_levels)} train strata")

    # ── the H × g alarm grid (primary split) ────────────────────────────────
    grid_rows, delta_rows = [], []
    primary_ev = None
    for H in HORIZONS:
        for g in GUARD_GAPS:
            ev, _ = _eval_one(train, test, features, H, g, fit=fit)
            ev_scored = ev[ev["y"].isin([0, 1])].reset_index(drop=True)
            grid_rows.append(_metrics_row(ev_scored, H, g, "install_cohort"))
            # bootstrap CI on the AUC delta (dynamic − baseline)
            def _delta(f):
                a = dynamic_auc(f, "dyn_risk"); b = dynamic_auc(f, "base_risk")
                return a - b if np.isfinite(a) and np.isfinite(b) else np.nan
            pt, lo, hi = cluster_bootstrap_delta(ev_scored, _delta, n_boot=300)
            delta_rows.append({"H": H, "g": g, "auc_delta": round(pt, 4),
                               "ci_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
                               "ci_hi": round(hi, 4) if np.isfinite(hi) else np.nan})
            if H == PRIMARY_H and g == PRIMARY_G:
                primary_ev = ev

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(tbl / "alarm_grid.csv", index=False, encoding="utf-8-sig")
    (eval_out / "tables" / "alarm_grid.csv").write_text(
        grid.to_csv(index=False), encoding="utf-8-sig")
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(tbl / "delta_ci.csv", index=False, encoding="utf-8-sig")

    # ── primary operational detail + verdict line ───────────────────────────
    pe = primary_ev[primary_ev["y"].isin([0, 1])].reset_index(drop=True)
    op_rows = []
    for k in (0.05, 0.10):
        lead = lead_time_distribution(primary_ev, "dyn_risk", k)
        pr = precision_recall_at_capacity(pe, "dyn_risk", k_fracs=(k,)).iloc[0]
        prb = precision_recall_at_capacity(pe, "base_risk", k_fracs=(k,)).iloc[0]
        op_rows.append({
            "k_pct": int(k * 100),
            "precision_dyn": pr["precision"], "recall_dyn": pr["recall"],
            "recall_base": prb["recall"],
            "false_alarms_per_pump_year": false_alarms_per_pump_year(pe, "dyn_risk", k),
            "lead_time_median": round(lead["median"], 1) if lead["n"] else np.nan,
            "lead_time_q1": round(lead["q1"], 1) if lead["n"] else np.nan,
            "lead_time_q3": round(lead["q3"], 1) if lead["n"] else np.nan,
            "n_caught_runs": lead["n"],
        })
    op = pd.DataFrame(op_rows)
    op.to_csv(tbl / "primary_operational.csv", index=False, encoding="utf-8-sig")

    # calibration of the dynamic window risk
    cal = th.calibration_table(pe.assign(_dummy=0), pe["dyn_risk"].to_numpy(),
                               PRIMARY_H + PRIMARY_G, dur="tte_land", ev="event_land")
    if not cal.empty:
        cal.to_csv(tbl / "calibration_primary.csv", index=False, encoding="utf-8-sig")

    prim = grid[(grid["H"] == PRIMARY_H) & (grid["g"] == PRIMARY_G)].iloc[0]
    pd10 = op[op["k_pct"] == 10].iloc[0]
    dci = delta[(delta["H"] == PRIMARY_H) & (delta["g"] == PRIMARY_G)].iloc[0]
    verdict = (
        f"[VERDICT] At H={PRIMARY_H}d and top-10% capacity, the dynamic model catches "
        f"{pd10['recall_dyn']*100:.0f}% of failures (median lead {pd10['lead_time_median']}d, "
        f"{pd10['false_alarms_per_pump_year']} false alarms/pump-year) vs "
        f"{pd10['recall_base']*100:.0f}% for age+stratum alone. "
        f"AUC dyn {prim['auc_dyn']} vs base {prim['auc_base']} "
        f"(delta {dci['auc_delta']} [{dci['ci_lo']}, {dci['ci_hi']}])."
    )
    material = np.isfinite(dci["ci_lo"]) and dci["ci_lo"] > 0
    verdict += ("  -> dynamics add material, interval-backed lift (D5 watchlist earned)."
                if material else
                "  -> HONEST NULL: CI includes 0; dynamics do not beat age+stratum "
                "out of sample (D5 -> telemetry-improvement recommendation).")
    log.append(verdict)

    # ── secondary robustness: landmark-calendar split (primary H×g) ─────────
    cal_split = landmark_calendar_split(fz)
    rob_rows = []
    if len(cal_split.test) > 50:
        ev2, _ = _eval_one(cal_split.train, cal_split.test, features, PRIMARY_H, PRIMARY_G)
        ev2s = ev2[ev2["y"].isin([0, 1])].reset_index(drop=True)
        rob_rows.append(_metrics_row(ev2s, PRIMARY_H, PRIMARY_G, "landmark_calendar"))
    if rob_rows:
        pd.DataFrame(rob_rows).to_csv(tbl / "robustness_calendar.csv",
                                      index=False, encoding="utf-8-sig")
        r = rob_rows[0]
        log.append(f"[D4] robustness (landmark-calendar split): AUC dyn {r['auc_dyn']} "
                   f"vs base {r['auc_base']} (delta {r['auc_delta']})")

    # ── figures ─────────────────────────────────────────────────────────────
    _fig_auc_delta(delta, figs / "auc_delta.png")
    _fig_lead(primary_ev, figs / "lead_time.png")

    (logs / "alarm.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"[D4] outputs -> {out}  (+ {eval_out})")


def _fig_auc_delta(delta, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    lab = [f"H{int(r.H)}/g{int(r.g)}" for r in delta.itertuples()]
    y = np.arange(len(delta))
    ax.errorbar(delta["auc_delta"], y,
                xerr=[delta["auc_delta"] - delta["ci_lo"], delta["ci_hi"] - delta["auc_delta"]],
                fmt="o", color="#4C78A8", ecolor="#999", capsize=3)
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("AUC(dynamic) − AUC(age+stratum), test split (95% well-cluster CI)")
    ax.set_title("Phase D — does dynamics beat age+stratum out of sample?", fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_lead(primary_ev, path):
    lead = lead_time_distribution(primary_ev, "dyn_risk", 0.10)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if lead["n"]:
        ax.hist(lead["values"], bins=20, color="#F58518", alpha=0.85)
        ax.axvline(lead["median"], color="k", ls="--", label=f"median {lead['median']:.0f}d")
        ax.legend(fontsize=8)
    ax.set_xlabel("operating days from first alarm to failure (top-10%)")
    ax.set_ylabel("failing runs")
    ax.set_title(f"Phase D lead-time distribution (H={PRIMARY_H}, g={PRIMARY_G})", fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
