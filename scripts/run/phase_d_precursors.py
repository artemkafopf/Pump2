"""Phase D — D1: descriptive precursor atlas (the "is there anything here?" look).

Before any model, on the **train split only** (install cohort ≤ 2023-12-31, per the
C5 convention and leakage rule 3), compare each trailing-window feature between
pre-failure landmarks (target ``y=1`` at the primary H=60 / g=7) and survivor
landmarks (``y=0``), **matched on stratum × operating-age band** so the comparison is
not just an age effect.  This caps expectations honestly and ranks where signal, if
any, lives.

Outputs under ``results/phase_d_precursors/<date>/``:
  tables/standardized_diffs.csv   — matched standardized mean difference per feature
  figures/precursor_grid.png      — pos vs neg distributions for the top features
  logs/atlas.txt

Run (needs phase_d_landmarks.py first):
    python scripts/run/phase_d_precursors.py
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
    load_latest_landmark_frame, FEATURE_STEMS, WINDOWS, DEFAULT_GUARD,
)
from analysis.models.survival.landmark_cox import install_cohort_split

PRIMARY_H = 60
AGE_BANDS = [0, 60, 120, 240, 480, np.inf]


def matched_smd(train: pd.DataFrame, feat: str, y_col: str) -> dict:
    """Stratum × age-band matched standardized mean difference (pos − neg)/sd.

    Computes the SMD within each (stratum × age-band) cell that has ≥5 of each class,
    then pools cells weighted by cell size.  Positive → higher in pre-failure windows.
    """
    d = train[train[y_col].isin([0, 1])].copy()
    d = d[np.isfinite(d[feat])]
    if d.empty:
        return {"feature": feat, "smd": np.nan, "n_pos": 0, "n_neg": 0, "n_cells": 0}
    d["_band"] = pd.cut(d["op_age"], bins=AGE_BANDS)
    cell_smd, weights = [], []
    for _, cell in d.groupby(["stratum_key", "_band"], observed=True):
        pos = cell.loc[cell[y_col] == 1, feat].to_numpy(float)
        neg = cell.loc[cell[y_col] == 0, feat].to_numpy(float)
        if len(pos) >= 5 and len(neg) >= 5:
            sd = np.sqrt((np.var(pos) + np.var(neg)) / 2.0)
            if sd > 0:
                cell_smd.append((np.mean(pos) - np.mean(neg)) / sd)
                weights.append(len(pos) + len(neg))
    smd = float(np.average(cell_smd, weights=weights)) if cell_smd else np.nan
    return {"feature": feat, "smd": round(smd, 4) if np.isfinite(smd) else np.nan,
            "n_pos": int((d[y_col] == 1).sum()), "n_neg": int((d[y_col] == 0).sum()),
            "n_cells": len(cell_smd)}


def _grid(train: pd.DataFrame, feats: list[str], y_col: str, path: Path) -> None:
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    for i, f in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        pos = pd.to_numeric(train.loc[train[y_col] == 1, f], errors="coerce").dropna()
        neg = pd.to_numeric(train.loc[train[y_col] == 0, f], errors="coerce").dropna()
        if len(pos) < 5 or len(neg) < 5:
            ax.set_visible(False); continue
        lo, hi = np.nanpercentile(pd.concat([pos, neg]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(neg, bins=bins, density=True, alpha=0.55, color="#4C78A8", label="survivor")
        ax.hist(pos, bins=bins, density=True, alpha=0.55, color="#F58518", label="pre-failure")
        ax.set_title(f, fontsize=7.5); ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(feats), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(f"Phase D precursor atlas — pre-failure vs survivor landmarks "
                 f"(train, H={PRIMARY_H}/g={DEFAULT_GUARD})", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    out = results_dir("phase_d_precursors")
    tbl, figs, logs = out / "tables", out / "figures", out / "logs"
    log: list[str] = []

    frame = load_latest_landmark_frame()
    split = install_cohort_split(frame)
    train = split.train
    y_col = f"y_H{PRIMARY_H}_g{DEFAULT_GUARD}"
    log.append(f"[D1] train landmarks {len(train)} (installs <= {split.cutoff}); "
               f"pos={int((train[y_col]==1).sum())} neg={int((train[y_col]==0).sum())}")

    feats = [f"{s}_w{w}" for w in WINDOWS for s in FEATURE_STEMS]
    feats = [f for f in feats if f in train.columns]
    rows = [matched_smd(train, f, y_col) for f in feats]
    smd = pd.DataFrame(rows)
    smd["abs_smd"] = smd["smd"].abs()
    smd = smd.sort_values("abs_smd", ascending=False, na_position="last").reset_index(drop=True)
    smd.drop(columns="abs_smd").to_csv(tbl / "standardized_diffs.csv",
                                       index=False, encoding="utf-8-sig")

    log.append("[D1] top matched standardized differences (|SMD|, train only):")
    for _, r in smd.head(12).iterrows():
        log.append(f"       {r['feature']:<26} SMD={r['smd']:+.3f}  (cells={int(r['n_cells'])})")

    top = smd.dropna(subset=["smd"]).head(9)["feature"].tolist()
    if top:
        _grid(train, top, y_col, figs / "precursor_grid.png")

    (logs / "atlas.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"[D1] outputs -> {out}")


if __name__ == "__main__":
    main()
