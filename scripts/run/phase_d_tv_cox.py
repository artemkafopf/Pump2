"""Phase D — D2: time-varying Cox (inference view — which dynamics carry hazard).

Episode-splits the landmark frame (start = landmark, stop = next landmark/event/censor)
and fits a stratified time-varying Cox on a **train-only** screened feature subset —
all-cause first, then the two Phase-B mode views (hydraulic, electro-thermal) via the
cause-specific event indicators.  Purpose is *engineering knowledge*, not prediction:
which trailing-window dynamics move the hazard and on which failure mode.

Features are z-scored (train stats) so HRs are per-SD and comparable; the screening
cascade (univariate p<0.10 → Spearman prune) is adapted from ``screening_cox`` and run
on train only (leakage rule 3).  Every output carries the §0.2-style causal disclaimer:
dynamics are *even more* confounded by indication than static settings — a rising load
trend may be the operator responding, not the pump failing.

Outputs under ``results/phase_d_tv_cox/<date>/``:
  tables/screen_univariate.csv          — train univariate screen
  tables/tvcox_all_cause.csv            — all-cause per-SD HRs
  tables/tvcox_hydraulic.csv            — hydraulic cause-specific HRs
  tables/tvcox_electro-thermal.csv      — electro-thermal cause-specific HRs
  tables/prior_check.csv                — Phase B/C prior corroboration table
  logs/tv_cox.txt

Run (needs phase_d_landmarks.py first):
    python scripts/run/phase_d_tv_cox.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.landmark_features import load_latest_landmark_frame, WINDOWS
from analysis.models.survival.landmark_cox import (
    install_cohort_split, build_episode_frame, fit_time_varying_cox,
)
from analysis.models.survival.screening_cox import univariate_screen, correlation_prune

# Well-covered dynamics candidates (exclude days_since_restart — mostly "no restart",
# and the pctdrop which is a run-relative transform of the level).
_STEMS = ["kprod_mean", "kprod_slope", "qliq_slope", "load_mean", "load_slope",
          "load_std", "load_exc", "freq_mean", "freq_std", "freq_step", "freq_days55",
          "restarts", "longest_idle", "rintake_mean", "rintake_slope", "rzab_slope",
          "drawdown", "watercut_mean", "watercut_jump", "gasfactor_mean", "gasfactor_slope"]

# Phase B/C priors to check (feature stem → expected mode).
PRIORS = {
    "freq_step": "electro-thermal", "freq_std": "electro-thermal",
    "restarts": "electro-thermal", "freq_days55": "electro-thermal",
    "kprod_mean": "hydraulic", "kprod_slope": "hydraulic", "qliq_slope": "hydraulic",
    "gasfactor_mean": "hydraulic",
}


def _zscore(frame: pd.DataFrame, cols: list[str], train_mask: pd.Series) -> pd.DataFrame:
    d = frame.copy()
    for c in cols:
        mu = d.loc[train_mask, c].mean()
        sd = d.loc[train_mask, c].std()
        d[c] = (d[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return d


def main() -> None:
    out = results_dir("phase_d_tv_cox")
    tbl, logs = out / "tables", out / "logs"
    log: list[str] = []

    frame = load_latest_landmark_frame()
    cands = [f"{s}_w{w}" for w in WINDOWS for s in _STEMS if f"{s}_w{w}" in frame.columns]
    # prefer the 30d window as primary; keep both, screening prunes the correlated one
    dt = pd.to_datetime(frame["install_dt"], errors="coerce")
    train_mask = dt <= pd.Timestamp("2023-12-31")
    frame_z = _zscore(frame, cands, train_mask)
    split = install_cohort_split(frame_z)
    train = split.train
    log.append(f"[D2] train landmarks {len(train)}; candidates {len(cands)} "
               "(z-scored, per-SD HRs); DISCLAIMER: associations under adjustment, "
               "dynamics heavily confounded by indication (operator response). "
               "SEs are model-based (lifelines CoxTimeVaryingFitter has no cluster-robust "
               "option) -> p-values are screening signals, not confirmatory; verdict is D4.")

    # ── screen on train only (univariate → Spearman prune) ──────────────────
    uni = univariate_screen(train, cands, duration_col="tte_land", event_col="event_land")
    uni.to_csv(tbl / "screen_univariate.csv", index=False, encoding="utf-8-sig")
    survivors = uni.loc[uni["pass_screen"], "covariate"].tolist()
    cindex = uni.set_index("covariate")["c_index"].to_dict()
    kept, dropped = correlation_prune(train, survivors, cindex)
    log.append(f"[D2] {len(survivors)}/{len(cands)} passed univariate; "
               f"correlation prune kept {len(kept)} (dropped {len(dropped)})")
    if not kept:
        log.append("[D2] no dynamics survived screening — honest null at the inference layer.")
        (logs / "tv_cox.txt").write_text("\n".join(log), encoding="utf-8")
        print("\n".join(log)); return

    # ── all-cause + cause-specific time-varying Cox ─────────────────────────
    sig_by_mode: dict[str, set] = {}
    for label, event_source in (("all_cause", "event_land"),
                                ("hydraulic", "event_hydraulic"),
                                ("electro-thermal", "event_electro-thermal")):
        epi = build_episode_frame(train, kept, event_source=event_source)
        res = fit_time_varying_cox(epi, kept)
        if not res.success:
            log.append(f"[D2] {label}: fit failed ({res.message})")
            continue
        s = res.summary.copy()
        s["p"] = s["p"].astype(float)
        s = s.sort_values("p")
        s.to_csv(tbl / f"tvcox_{label}.csv", index=False, encoding="utf-8-sig")
        sig = set(s.loc[s["p"] < 0.05, "covariate"])
        if label != "all_cause":
            sig_by_mode[label] = sig
        log.append(f"[D2] {label}: {res.n_events} events, {res.n_intervals} intervals "
                   f"({res.message}); {len(sig)} covariates p<0.05")
        for _, r in s.head(6).iterrows():
            star = " *" if r["p"] < 0.05 else ""
            log.append(f"       {r['covariate']:<22} HR/SD={r['hr']:.3f} "
                       f"[{r['hr_ci_lo']:.2f},{r['hr_ci_hi']:.2f}] p={r['p']:.4f}{star}")

    # ── prior corroboration table ───────────────────────────────────────────
    rows = []
    for stem, expected in PRIORS.items():
        hits = []
        for mode, sig in sig_by_mode.items():
            if any(c.startswith(stem) for c in sig):
                hits.append(mode)
        verdict = ("none-significant" if not hits else
                   "match" if hits == [expected] else
                   "diffuse" if len(hits) >= 2 else f"mismatch->{hits[0]}")
        rows.append({"feature_stem": stem, "expected_mode": expected,
                     "significant_modes": "+".join(hits) if hits else "(none)",
                     "verdict": verdict})
    pd.DataFrame(rows).to_csv(tbl / "prior_check.csv", index=False, encoding="utf-8-sig")
    log.append("[D2] prior check (cycling/freq->electro-thermal; kprod/qliq->hydraulic):")
    for r in rows:
        log.append(f"       {r['feature_stem']:<16} expect {r['expected_mode']:<16} "
                   f"got {r['significant_modes']:<20} -> {r['verdict']}")

    (logs / "tv_cox.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"[D2] outputs -> {out}")


if __name__ == "__main__":
    main()
