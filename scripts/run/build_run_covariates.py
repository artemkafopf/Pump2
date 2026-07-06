"""Phase A T7 — build the run-level covariate registry and coverage audit.

Thin CLI wrapper around ``analysis.data.run_covariates.build_run_covariates``.
Produces, under ``results/phase_a_covariate_registry/<date>/``:

  tables/run_covariates.parquet     — the full run-level feature table (2,634 rows)
  tables/registry.csv               — per-covariate measured/imputed coverage
  tables/registry_by_stratum.csv    — coverage per (covariate × stratum_key)
  tables/pump_type_parse_report.csv — parse rate by series
  tables/pump_type_unparsed.csv     — top unparsed strings for user review
  tables/spearman_matrix.csv        — correlation across all candidates (Phase C VIF pre-screen)
  figures/coverage_heatmap.png      — measured coverage (covariate × stratum)
  figures/distribution_grid.png     — histogram per covariate
  reports/OPEN_QUESTIONS.md         — vg_m / curvature units, gabarit→OD mapping, tubing gap

Run:
    python scripts/run/build_run_covariates.py
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
    build_run_covariates,
    coverage_report,
    coverage_by_stratum,
    _IMPUTE_COLS,
)
from analysis.data.pump_type_parser import (
    parse_pump_type_series,
    parse_report,
    OPEN_QUESTIONS,
    GABARIT_OD_MM,
    REDA_SERIES_OD_INCH,
)

# Continuous candidates for the Spearman pre-screen (measured/imputed values).
_SPEARMAN_COLS = _IMPUTE_COLS + [
    "ca_so4_product", "kpod_freq_mean", "frac_kpod_below_0p7", "frac_pzab_below_1",
    "glf_frac_days_gas", "freq_above_55hz_pct", "freq_below_45hz_pct",
    "n_freq_steps_per_100d", "n_restarts_per_100d", "idle_frac",
    "od_group_mm", "pump_q_design_m3d", "run_seq", "days_since_prev_failure",
    "install_year",
]


def _coverage_heatmap(cov_stratum: pd.DataFrame, path: Path) -> None:
    piv = cov_stratum.pivot(index="covariate", columns="stratum_key", values="pct_measured")
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * piv.shape[1]), max(6, 0.32 * piv.shape[0])))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title("Measured coverage % (covariate × stratum)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="% measured")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _distribution_grid(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() > 10]
    n = len(cols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(vals, bins=30, color="steelblue", alpha=0.8)
        ax.set_title(col, fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle("Covariate distributions (post-imputation)", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_open_questions(path: Path, parse_rep: dict) -> None:
    lines = [
        "# Phase A T7 — Open questions for the user (NOT silently resolved)",
        "",
        "These require confirmation before the affected covariates are interpreted or",
        "used in Phase C model fitting.",
        "",
        "## 1. `vg_m` semantics/units",
        "",
        "- Range 0–2,740 m, mean ≈ 2,310. Labelled here as *setting depth?* — the true",
        "  meaning (pump setting depth vs. perforation depth vs. something else) is",
        "  **unconfirmed**. `submergence_depth_m` is 1% populated and unusable.",
        "- **Question:** what does `vg_m` measure, in what units, per the source workbook?",
        "",
        "## 2. `curvature` (curvature_flag) units",
        "",
        "- 56% populated; numeric values cluster ~0.31–0.51. `'-'` and NULL = missing.",
        "- **Question:** are these degrees per 10 m (dogleg severity), or another unit?",
        "  Direction of effect (↑ dogleg → wear/cable damage) assumed but not usable",
        "  quantitatively until units are known.",
        "",
        "## 3. gabarit → OD (mm) mapping confirmation",
        "",
        "- Parser maps ЭЦН gabarit → OD group. Confirmed from prompt: 5→92, 5А→103, 6→114.",
        "  All others in `GABARIT_OD_MM` are conventional ГОСТ sizes **pending confirmation**:",
        f"    {GABARIT_OD_MM}",
        "- REDA series → OD (inch) are approximate:",
        f"    {REDA_SERIES_OD_INCH}",
        f"- Parse rate achieved: {parse_rep['parse_rate']:.1%} "
        f"({parse_rep['n_parsed']}/{parse_rep['n_total']}); by series: {parse_rep['by_series']}.",
        "",
        "## 4. Tubing diameter — data request",
        "",
        "- Tubing (НКТ) diameter is **not in the warehouse**. It was user-requested as a",
        "  candidate completion covariate. **Raise as a data request to the data owner.**",
        "",
        "## 5. КВЧ measured coverage is lower than previously assumed",
        "",
        "- The registry measures ~14% *measured* КВЧ coverage at run level (683 non-null",
        "  КВЧ samples across 384 wells in lab.sqlite), vs the ~41% figure quoted in",
        "  `chemistry_cox.md`. The difference is imputation: the honest measured coverage is",
        "  ~14%. КВЧ coefficients will be heavily imputation-driven (see T1.4).",
        "",
        f"_Parser note: {OPEN_QUESTIONS}_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = results_dir("phase_a_covariate_registry")
    tbl, fig_dir, rep = out / "tables", out / "figures", out / "reports"

    print("[T7] Building run-level covariate registry (tte=ttf_mix)...")
    df = build_run_covariates(tte_col="ttf_mix")
    print(f"[T7] Built {len(df)} rows × {df.shape[1]} cols")

    # Feature table (parquet + a slim CSV preview)
    try:
        df.to_parquet(tbl / "run_covariates.parquet", index=False)
    except Exception as exc:  # pyarrow may be absent
        print(f"[T7] parquet unavailable ({exc}); writing CSV instead")
        df.to_csv(tbl / "run_covariates.csv", index=False, encoding="utf-8-sig")

    # Registry coverage
    cov = coverage_report(df)
    cov.to_csv(tbl / "registry.csv", index=False, encoding="utf-8-sig")
    print("\n[T7] Coverage (measured %):")
    print(cov[["covariate", "pct_measured", "n_measured"]].to_string(index=False))

    cov_str = coverage_by_stratum(df)
    cov_str.to_csv(tbl / "registry_by_stratum.csv", index=False, encoding="utf-8-sig")
    n_low = int(cov_str["low_coverage_flag"].sum())
    print(f"\n[T7] {n_low} (covariate × stratum) cells < 50% measured (flagged).")

    # pump_type parse report
    parsed = parse_pump_type_series(df["pump_type"])
    parse_rep = parse_report(df["pump_type"])
    pd.DataFrame([{
        "n_total": parse_rep["n_total"],
        "n_parsed": parse_rep["n_parsed"],
        "parse_rate": parse_rep["parse_rate"],
        "n_gabarit": parse_rep["n_gabarit"],
        "n_q_design": parse_rep["n_q_design"],
        **{f"series_{k}": v for k, v in parse_rep["by_series"].items()},
    }]).to_csv(tbl / "pump_type_parse_report.csv", index=False, encoding="utf-8-sig")

    unparsed = df.loc[~parsed["parsed"], "pump_type"].value_counts().head(30)
    unparsed.rename_axis("pump_type").reset_index(name="n").to_csv(
        tbl / "pump_type_unparsed.csv", index=False, encoding="utf-8-sig")
    print(f"[T7] pump_type parse rate: {parse_rep['parse_rate']:.1%}; "
          f"{len(unparsed)} distinct unparsed strings logged")

    # Spearman matrix (Phase C VIF pre-screen)
    spear_cols = [c for c in _SPEARMAN_COLS if c in df.columns]
    corr = df[spear_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr.to_csv(tbl / "spearman_matrix.csv", encoding="utf-8-sig")

    # Figures
    _coverage_heatmap(cov_str, fig_dir / "coverage_heatmap.png")
    _distribution_grid(df, spear_cols, fig_dir / "distribution_grid.png")

    # Open questions
    _write_open_questions(rep / "OPEN_QUESTIONS.md", parse_rep)

    print(f"\n[T7] Outputs written to: {out}")


if __name__ == "__main__":
    main()
