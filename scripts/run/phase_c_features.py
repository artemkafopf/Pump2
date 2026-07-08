"""Phase C C0 — feature completion + registry/coverage refresh.

Rebuilds the run-level covariate registry after the C0 additions
(``freq_std_early``, ``load_std_early``, ``freq_above_55hz_pct_early`` and the
four mandatory cohort adjusters ``install_period`` / ``run_seq`` /
``days_since_prev_failure`` / ``has_telemetry``), and refreshes the coverage
audit that Phase C models plan around.

Outputs under ``results/phase_c_features/<date>/``:
  tables/run_covariates.parquet          — full run-level feature table (2,634 rows)
  tables/registry.csv                    — per-covariate measured/imputed coverage
  tables/registry_by_stratum.csv         — coverage per (covariate × stratum_key)
  tables/block2_coverage_summary.csv     — Block-2 telemetry coverage + n≥window flags
  tables/spearman_matrix.csv             — correlation across all Phase C candidates
  figures/coverage_heatmap.png           — measured coverage (covariate × stratum)
  figures/block2_distribution_grid.png   — histograms of the new Block-2 features

Run:
    python scripts/run/phase_c_features.py
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
from analysis.data.run_covariates import (
    build_run_covariates, coverage_report, coverage_by_stratum, _IMPUTE_COLS,
    EARLY_OP_DAYS,
)

# Phase C Block-2 covariates that were added / are the focus of C0.
BLOCK2_NEW = [
    "freq_mean", "freq_std_early", "freq_above_55hz_pct_early",
    "load_mean", "load_std_early", "kpod_mean", "kpod_freq_mean",
    "frac_kpod_below_0p7", "pzab_over_pbubble", "frac_pzab_below_1",
    "rpump_intake_mean", "n_freq_steps_per_100d", "n_restarts_per_100d",
    "idle_frac",
]

_SPEARMAN_COLS = list(dict.fromkeys(_IMPUTE_COLS + [
    "ca_so4_product", "frac_kpod_below_0p7", "frac_pzab_below_1",
    "glf_frac_days_gas", "n_freq_steps_per_100d", "n_restarts_per_100d",
    "idle_frac", "od_group_mm", "pump_q_design_m3d", "run_seq",
    "days_since_prev_failure", "install_year", "has_telemetry",
]))


def _coverage_heatmap(cov_stratum: pd.DataFrame, path: Path) -> None:
    piv = cov_stratum.pivot(index="covariate", columns="stratum_key", values="pct_measured")
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * piv.shape[1]), max(6, 0.32 * piv.shape[0])))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title("Measured coverage % (covariate × stratum) — Phase C", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="% measured")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _dist_grid(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() > 10]
    ncol = 4; nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        ax.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=30,
                color="#4C78A8", alpha=0.85)
        ax.set_title(col, fontsize=7); ax.tick_params(labelsize=6)
    for j in range(len(cols), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle("Block-2 covariate distributions (post-imputation)", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def block2_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per Block-2 covariate: measured coverage + runs clearing the early window,
    with n on telemetry (the population Block-2 models actually run on)."""
    rows = []
    for col in BLOCK2_NEW:
        src = f"{col}_imputed_src"
        n_meas = int((df[src] == "measured").sum()) if src in df.columns else int(df[col].notna().sum())
        rows.append({
            "covariate": col,
            "n_measured": n_meas,
            "pct_measured": round(100.0 * n_meas / len(df), 1),
            "n_full_early_window": int((df["n_early_op_days"] >= EARLY_OP_DAYS).sum()),
            "n_has_telemetry": int((df["has_telemetry"] == 1).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    out = results_dir("phase_c_features")
    tbl, fig_dir = out / "tables", out / "figures"

    print(f"[C0] Building run covariates (early window = first {EARLY_OP_DAYS} operating days)...")
    df = build_run_covariates(tte_col="ttf_mix")
    print(f"[C0] {len(df)} rows × {df.shape[1]} cols")

    try:
        df.to_parquet(tbl / "run_covariates.parquet", index=False)
    except Exception as exc:
        print(f"[C0] parquet unavailable ({exc}); CSV instead")
        df.to_csv(tbl / "run_covariates.csv", index=False, encoding="utf-8-sig")

    cov = coverage_report(df)
    cov.to_csv(tbl / "registry.csv", index=False, encoding="utf-8-sig")
    print("\n[C0] Coverage — newly added / Block-2 focus features:")
    print(cov[cov["covariate"].isin(BLOCK2_NEW + ["freq_std_early", "load_std_early",
          "freq_above_55hz_pct_early"])][["covariate", "n_measured", "pct_measured"]]
          .to_string(index=False))

    cov_str = coverage_by_stratum(df)
    cov_str.to_csv(tbl / "registry_by_stratum.csv", index=False, encoding="utf-8-sig")
    print(f"\n[C0] {int(cov_str['low_coverage_flag'].sum())} (covariate × stratum) cells <50% measured")

    b2 = block2_coverage_summary(df)
    b2.to_csv(tbl / "block2_coverage_summary.csv", index=False, encoding="utf-8-sig")

    spear_cols = [c for c in _SPEARMAN_COLS if c in df.columns]
    corr = df[spear_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr.to_csv(tbl / "spearman_matrix.csv", encoding="utf-8-sig")

    _coverage_heatmap(cov_str, fig_dir / "coverage_heatmap.png")
    _dist_grid(df, BLOCK2_NEW, fig_dir / "block2_distribution_grid.png")

    print(f"\n[C0] Outputs written to: {out}")


if __name__ == "__main__":
    main()
