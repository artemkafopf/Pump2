"""Phase B B3 — Cause-specific Cox: does chemistry act on the pump?
(output slug: phase_b_cause_cox)

Refits the Block-1 chemistry covariates as **cause-specific** stratified Cox
models — one per cause group {hydraulic, electro-thermal, protector} — inheriting
the Phase A design (global β, ``strata=['stratum_key']``, cluster-robust by well,
clock=ttf_mix, early-window covariates).  §0.2 option (c): ``is_sour_flagged``
enters every model.

The money figure is one forest plot **per covariate across the three cause
groups**: the physical hypothesis says chemistry β should be nonzero for the
hydraulic hazard and ≈0 for electro-thermal.  A covariate significant in *all*
modes equally is a red flag for residual confounding (field / vintage) — the
report calls that out.

Sensitivity: hydraulic with vs without НКТ (§0.1); per-field heterogeneity of the
lead chemistry covariate.

Outputs under ``results/phase_b_cause_cox/<date>/``:
  tables/b3_cause_cox_coeffs.csv    — HR+CI+p per covariate × cause group
  tables/b3_nkt_sensitivity.csv     — hydraulic with vs without НКТ
  tables/b3_field_heterogeneity.csv — per-field HR of the lead chemistry covariate
  figures/b3_forest_<covariate>.png — forest plot per covariate across causes

Run:
    python scripts/run/phase_b_cause_cox.py
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
from analysis.data.competing_risks_loader import build_competing_risks_df, COX_MODE_GROUPS
from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox

# Block-1 chemistry / fluid covariates (early-window, from the Phase A registry).
# is_sour_flagged is the §0.2(c) covariate.  log_h2s_proxy is the continuous
# H₂S exposure; both are reported — collinearity between them is noted in B6.
COVARIATES = [
    "log_h2s_proxy_mg_l",
    "is_sour_flagged",
    "log_glf_mean_opdays",
    "watercut_daily",
    "log_mechanical_impurities_mg_l",   # КВЧ
    "log_total_mineralization_g_l",     # salt proxy
    "log_ca_so4_product",               # gypsum proxy
]
CHEM_LEAD = "log_h2s_proxy_mg_l"        # lead covariate for field heterogeneity
GROUP_COLORS = {
    "hydraulic": "#4C78A8", "electro-thermal": "#F58518", "protector": "#54A24B",
}


def _forest(cov: str, rows: pd.DataFrame, path: Path) -> None:
    """Forest plot: HR (log axis) + 95% CI for one covariate across cause groups."""
    rows = rows[rows["covariate"] == cov]
    if rows.empty:
        return
    groups = list(rows["cause_group"])
    y = np.arange(len(groups))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.6 + 0.35 * len(groups)))
    for yi, (_, r) in zip(y, rows.iterrows()):
        ax.plot([r["hr_ci_lo"], r["hr_ci_hi"]], [yi, yi], color=GROUP_COLORS.get(r["cause_group"], "#555"), lw=2)
        ax.plot(r["hr"], yi, "o", color=GROUP_COLORS.get(r["cause_group"], "#555"), ms=7)
        ax.text(r["hr_ci_hi"], yi + 0.12,
                f" HR={r['hr']:.2f} [{r['hr_ci_lo']:.2f},{r['hr_ci_hi']:.2f}] p={r['p']:.3f}",
                va="bottom", fontsize=7.5)
    ax.axvline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xscale("log")
    ax.set_xlabel("hazard ratio (log scale)")
    ax.set_title(f"B3 forest — {cov}  (cause-specific, clock=ttf_mix)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    out = results_dir("phase_b_cause_cox")
    tbl, figs = out / "tables", out / "figures"

    df = build_competing_risks_df(tte_col="ttf_mix")
    # hydraulic-without-НКТ event indicator for the §0.1 sensitivity.
    df["event_hydraulic_no_nkt"] = (
        (df["event_hydraulic"] == 1) & (df["failed_node_trimmed"] != "НКТ")
    ).astype(np.int8)

    # ── main sweep: each cause group × covariate set ─────────────────────────
    coeff_rows = []
    for group in COX_MODE_GROUPS:
        res = fit_cause_specific_cox(
            df, covariates=COVARIATES, event_col=f"event_{group}",
            duration_col="tte", strata=["stratum_key"], cluster_col="well_key",
        )
        print(f"[B3] {group:<16} n={res.n} events={res.n_events} "
              f"success={res.success} ({res.message})")
        if not res.success:
            continue
        s = res.summary.copy()
        s.insert(0, "cause_group", group)
        s["n_events"] = res.n_events
        coeff_rows.append(s)

    coeffs = pd.concat(coeff_rows, ignore_index=True)
    coeffs.to_csv(tbl / "b3_cause_cox_coeffs.csv", index=False, encoding="utf-8-sig")

    # ── forest plot per covariate across cause groups ────────────────────────
    for cov in COVARIATES:
        _forest(cov, coeffs, figs / f"b3_forest_{cov}.png")

    # ── sensitivity: hydraulic with vs without НКТ ───────────────────────────
    sens_rows = []
    for label, ev_col in (("hydraulic_with_nkt", "event_hydraulic"),
                          ("hydraulic_no_nkt", "event_hydraulic_no_nkt")):
        res = fit_cause_specific_cox(
            df, covariates=COVARIATES, event_col=ev_col,
            duration_col="tte", strata=["stratum_key"], cluster_col="well_key",
        )
        if res.success:
            s = res.summary.copy()
            s.insert(0, "variant", label)
            s["n_events"] = res.n_events
            sens_rows.append(s)
    if sens_rows:
        pd.concat(sens_rows, ignore_index=True).to_csv(
            tbl / "b3_nkt_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ── per-field heterogeneity of the lead chemistry covariate (hydraulic) ──
    het_rows = []
    for field, gdf in df.groupby("field", observed=True):
        if int(gdf["event_hydraulic"].sum()) < 30:
            continue
        res = fit_cause_specific_cox(
            gdf, covariates=[CHEM_LEAD, "is_sour_flagged"], event_col="event_hydraulic",
            duration_col="tte", strata=["stratum_key"], cluster_col="well_key",
            min_events=30,
        )
        if res.success:
            r = res.summary[res.summary["covariate"] == CHEM_LEAD]
            if not r.empty:
                het_rows.append({
                    "field": field, "covariate": CHEM_LEAD,
                    "hr": round(float(r["hr"].iloc[0]), 3),
                    "hr_ci_lo": round(float(r["hr_ci_lo"].iloc[0]), 3),
                    "hr_ci_hi": round(float(r["hr_ci_hi"].iloc[0]), 3),
                    "p": round(float(r["p"].iloc[0]), 4),
                    "n_events": res.n_events,
                })
    if het_rows:
        pd.DataFrame(het_rows).to_csv(
            tbl / "b3_field_heterogeneity.csv", index=False, encoding="utf-8-sig")

    print(f"[B3] forests: {len(list(figs.glob('b3_forest_*.png')))}  "
          f"outputs written to: {out}")


if __name__ == "__main__":
    main()
