"""Phase D — D0: build the landmark frame from the 1.25M-row daily table.

Materialises one row per run × landmark (operating ages 30, 60, 90, … while alive)
with leakage-proof trailing-window features (14 & 30 operating-day windows) and
per-(horizon × guard) targets, plus the op-day↔ttf_true reconciliation and the
per-landmark feature-coverage audit.

Outputs under ``results/phase_d_landmarks/<date>/``:
  tables/landmark_frame.parquet     — the frame every later Phase D task loads
  tables/reconciliation.csv         — per-run op-day (qliq) vs ttf_true discrepancy
  tables/coverage.csv               — per-landmark feature measured-coverage
  tables/summary.csv                — population / eligibility summary
  logs/build.txt                    — build log (counts, flags)

Run:
    python scripts/run/phase_d_landmarks.py
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
from analysis.data.run_covariates import WAREHOUSE_DB
from analysis.data.landmark_features import (
    build_from_warehouse, HORIZONS, GUARD_GAPS, DEFAULT_GUARD, LANDMARK_CADENCE,
    MIN_TELEMETRY_DAYS, WINDOWS,
)


def _write_frame(frame: pd.DataFrame, tbl: Path) -> Path:
    try:
        p = tbl / "landmark_frame.parquet"
        frame.to_parquet(p, index=False)
        return p
    except Exception as exc:
        print(f"[D0] parquet unavailable ({exc}); writing CSV")
        p = tbl / "landmark_frame.csv"
        frame.to_csv(p, index=False, encoding="utf-8-sig")
        return p


def main() -> None:
    out = results_dir("phase_d_landmarks")
    tbl, logs = out / "tables", out / "logs"
    log: list[str] = []

    def emit(msg: str) -> None:
        print(msg)
        log.append(msg)

    emit(f"[D0] cadence={LANDMARK_CADENCE}  horizons={HORIZONS}  guards={GUARD_GAPS} "
         f"(default {DEFAULT_GUARD})  windows={WINDOWS}  min_telemetry={MIN_TELEMETRY_DAYS}")
    emit("[D0] building landmark frame from warehouse (this pulls the daily table)...")
    res = build_from_warehouse(WAREHOUSE_DB)
    frame, recon, cov = res.frame, res.reconciliation, res.coverage

    p = _write_frame(frame, tbl)
    recon.to_csv(tbl / "reconciliation.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(tbl / "coverage.csv", index=False, encoding="utf-8-sig")

    n_runs = frame["row_id"].nunique() if not frame.empty else 0
    n_land = len(frame)
    n_recon_flag = int(recon["flag"].sum()) if not recon.empty else 0
    n_recon_scored = int(recon["discrepancy"].notna().sum()) if not recon.empty else 0

    emit(f"[D0] landmark frame: {n_land} landmarks over {n_runs} eligible runs -> {p.name}")
    emit(f"[D0] op-day/ttf_true reconciliation: {n_recon_flag}/{n_recon_scored} runs "
         f"flagged >{int(100*0.10)}% discrepancy")

    # Eligibility / population summary.
    if not frame.empty:
        per_run = frame.groupby("row_id")["op_age"].agg(["count", "max"])
        landmarks_per_run = per_run["count"]
        emit(f"[D0] landmarks/run: median {landmarks_per_run.median():.0f}, "
             f"mean {landmarks_per_run.mean():.1f}, max {int(landmarks_per_run.max())}")

        # Target positives at the primary (H, g).
        rows = []
        for H in HORIZONS:
            for g in GUARD_GAPS:
                y = frame[f"y_H{H}_g{g}"]
                elig = frame[f"eligible_H{H}_g{g}"] == 1
                n_elig = int(elig.sum())
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                n_unk = int(elig.sum() - (y == 1).sum() - (y == 0).sum())
                rows.append({"H": H, "g": g, "n_eligible": n_elig,
                             "n_pos": n_pos, "n_neg": n_neg, "n_censored_in_window": n_unk,
                             "pos_rate": round(n_pos / max(n_pos + n_neg, 1), 4),
                             "primary": (H == 60 and g == DEFAULT_GUARD)})
        summary = pd.DataFrame(rows)
        summary.to_csv(tbl / "summary.csv", index=False, encoding="utf-8-sig")
        emit("[D0] target grid (eligible / pos / neg / censored-in-window):")
        for _, r in summary.iterrows():
            star = " *primary" if r["primary"] else ""
            emit(f"       H={int(r['H'])} g={int(r['g'])}: elig={int(r['n_eligible'])} "
                 f"pos={int(r['n_pos'])} neg={int(r['n_neg'])} "
                 f"cens={int(r['n_censored_in_window'])} pos_rate={r['pos_rate']}{star}")

        # Feature coverage headline.
        worst = cov.sort_values("pct_measured").head(6)
        emit("[D0] lowest-coverage features:")
        for _, r in worst.iterrows():
            emit(f"       {r['feature']}: {r['pct_measured']}% measured")

    (logs / "build.txt").write_text("\n".join(log), encoding="utf-8")
    emit(f"[D0] outputs -> {out}")


if __name__ == "__main__":
    main()
