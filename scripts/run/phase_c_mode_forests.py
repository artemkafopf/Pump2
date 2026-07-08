"""Phase C C3 — mode-differentiation forest report (output: phase_c_mode_forests).

The confounding audit for the whole phase: for every Block-2/3 joint-model
survivor, one forest across the three cause groups (hydraulic / electro-thermal /
protector) plus the "expected mode?" verdict.  A covariate that lands on its
physically expected mode corroborates a mechanism; one elevated equally in every
mode (diffuse) flags residual confounding (field / vintage).

Consumes the survivor + expected-mode lists produced by C1 and C2, fits one
combined cause-specific sweep (survivors adjusted by the cohort + Block-1
adjusters), and assembles a consolidated forest grid + master verdict table.

Run C1 and C2 first (same date).  Then:
    python scripts/run/phase_c_mode_forests.py
"""
from __future__ import annotations

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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.common.cox_reporting import (
    cause_specific_sweep, forest_across_causes, GROUP_COLORS, COX_MODE_GROUPS,
)

TODAY = date.today().isoformat()
# Adjusters carried in the combined cause-specific models.
CS_ADJ = ["is_sour_flagged", "log_run_seq", "install_pre2020", "install_2023plus"]


def _read_verdicts(slug: str) -> pd.DataFrame:
    p = results_dir(slug, TODAY) / "tables" / (
        "c1_expected_mode_verdicts.csv" if "block3" in slug else "c2_expected_mode_verdicts.csv")
    if not p.exists():
        return pd.DataFrame(columns=["covariate", "expected_mode"])
    df = pd.read_csv(p)
    df["block"] = "Block 3 (completion)" if "block3" in slug else "Block 2 (operational)"
    return df[["covariate", "expected_mode", "block"]]


def _consolidated_grid(sweep: pd.DataFrame, survivors: list[str], path: Path) -> None:
    covs = [c for c in survivors if c in set(sweep["covariate"])]
    if not covs:
        return
    ncol = 2
    nrow = int(np.ceil(len(covs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.1 * nrow), squeeze=False)
    for i, cov in enumerate(covs):
        ax = axes[i // ncol][i % ncol]
        rows = sweep[sweep["covariate"] == cov]
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            c = GROUP_COLORS.get(r["cause_group"], "#555")
            ax.plot([r["hr_ci_lo"], r["hr_ci_hi"]], [yi, yi], color=c, lw=2)
            ax.plot(r["hr"], yi, "o", color=c, ms=6)
            if r["p"] < 0.05:
                ax.plot(r["hr"], yi, "o", color=c, ms=10, mfc="none")
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(rows["cause_group"], fontsize=7)
        ax.set_xscale("log"); ax.set_title(cov, fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(covs), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle("C3 — mode differentiation across cause groups (○ = p<0.05; log HR axis)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    out = results_dir("phase_c_mode_forests")
    tbl, figs = out / "tables", out / "figures"

    surv_meta = pd.concat([_read_verdicts("phase_c_block3_completion"),
                           _read_verdicts("phase_c_block2_operational")], ignore_index=True)
    survivors = surv_meta["covariate"].tolist()
    print(f"[C3] {len(survivors)} survivors across blocks: {survivors}")

    df = build_competing_risks_df(tte_col="ttf_mix")
    sweep = cause_specific_sweep(df, survivors + CS_ADJ)
    sweep.round(5).to_csv(tbl / "c3_cause_specific_coeffs.csv", index=False, encoding="utf-8-sig")

    rows = []
    exp_map = dict(zip(surv_meta["covariate"], surv_meta["expected_mode"]))
    for cov in survivors:
        em = None if exp_map.get(cov) in (None, "?", np.nan) else exp_map.get(cov)
        v = forest_across_causes(sweep, cov, figs / f"c3_forest_{cov}.png",
                                 expected_mode=em, title_prefix="C3 — ")
        # significant groups for the master table
        r = sweep[sweep["covariate"] == cov]
        sig = "+".join(r.loc[r["p"] < 0.05, "cause_group"]) if not r.empty else ""
        rows.append({"covariate": cov, "block": surv_meta.loc[
            surv_meta["covariate"] == cov, "block"].iloc[0],
            "expected_mode": exp_map.get(cov, "?"), "significant_modes": sig or "none",
            "verdict": v})
    master = pd.DataFrame(rows)
    master.to_csv(tbl / "c3_mode_verdicts_master.csv", index=False, encoding="utf-8-sig")
    print(master.to_string(index=False))

    _consolidated_grid(sweep, survivors, figs / "c3_consolidated_grid.png")

    # confounding-audit summary counts
    n_diffuse = int(master["verdict"].str.startswith("diffuse").sum())
    n_match = int(master["verdict"].str.startswith("match").sum())
    print(f"[C3] match={n_match}, diffuse(confounding-suspect)={n_diffuse}, "
          f"total survivors={len(master)}")
    print(f"[C3] Outputs written to: {out}")


if __name__ == "__main__":
    main()
