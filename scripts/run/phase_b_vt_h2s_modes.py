"""Phase B B4 — What does H₂S attack in Vt? (output slug: phase_b_vt_h2s_modes)

Cause-specific dissection of the Vt H₂S effect by failure mode.  For each mode
group {hydraulic, electro-thermal, protector}:

* cause-specific Cox HR for ``sour`` within Vt (cluster-robust by well, stratified
  by contractor group);
* the time-interaction γ (does the sour disadvantage grow with time?) via the
  corrected event-time episode-split module (B0.2), with a well-bootstrap CI —
  event counts per mode are ~80–124, so the CIs are wide and reported honestly
  (no curve-regression p-values);
* the CIF sour-vs-nonsour comparison per mode (Gray-style bootstrap) restated as
  the operational sentence "by 90 d, X% of sour installs have lost the
  cable/motor vs Y% nonsour".

Outputs under ``results/phase_b_vt_h2s_modes/<date>/``:
  tables/b4_vt_sour_hr.csv      — per-mode cause-specific sour HR + CI
  tables/b4_vt_gamma.csv        — per-mode time-interaction γ + bootstrap CI
  tables/b4_vt_cif_compare.csv  — per-mode CIF sour/nonsour at 90/365 + boot diff
  figures/b4_vt_sour_hr.png     — forest of per-mode sour HR

Run:
    python scripts/run/phase_b_vt_h2s_modes.py [n_boot]
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
from analysis.models.survival.time_interaction_cox import (
    fit_time_interaction_cox, bootstrap_gamma_ci,
)
from analysis.models.survival.cif import event_code_series, bootstrap_cif_difference
from analysis.data.failure_modes import MODE_GROUPS

GROUP_COLORS = {
    "hydraulic": "#4C78A8", "electro-thermal": "#F58518", "protector": "#54A24B",
}


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = results_dir("phase_b_vt_h2s_modes")
    tbl, figs = out / "tables", out / "figures"

    df = build_competing_risks_df(tte_col="ttf_mix")
    vt = df[df["field"] == "Vt"].copy()
    vt["sour"] = (vt["h2s_class"] == "sour").astype(int)
    vt["event_code"] = event_code_series(vt["mode_group"], vt["event"], MODE_GROUPS)
    vt_end = float(vt["tte"].max())
    print(f"[B4] Vt runs={len(vt)}  sour={int(vt['sour'].sum())}  "
          f"failures={int(vt['event'].sum())}")

    # ── per-mode cause-specific sour HR ──────────────────────────────────────
    hr_rows = []
    for group in COX_MODE_GROUPS:
        res = fit_cause_specific_cox(
            vt, covariates=["sour"], event_col=f"event_{group}",
            duration_col="tte", strata=["contractor_group"], cluster_col="well_key",
            min_events=15,
        )
        if not res.success:
            # fall back to unstratified if a contractor stratum is too thin
            res = fit_cause_specific_cox(
                vt, covariates=["sour"], event_col=f"event_{group}",
                duration_col="tte", strata=["field"], cluster_col="well_key",
                min_events=15,
            )
        if res.success:
            r = res.summary.iloc[0]
            hr_rows.append({
                "mode_group": group, "hr": round(float(r["hr"]), 3),
                "hr_ci_lo": round(float(r["hr_ci_lo"]), 3),
                "hr_ci_hi": round(float(r["hr_ci_hi"]), 3),
                "p": round(float(r["p"]), 4), "n_events": res.n_events,
            })
            print(f"[B4] {group:<16} sour HR={r['hr']:.2f} "
                  f"[{r['hr_ci_lo']:.2f},{r['hr_ci_hi']:.2f}] p={r['p']:.3f} "
                  f"(events={res.n_events})")
    hr_df = pd.DataFrame(hr_rows)
    hr_df.to_csv(tbl / "b4_vt_sour_hr.csv", index=False, encoding="utf-8-sig")

    # ── per-mode time-interaction γ (event-time split) + bootstrap CI ────────
    gamma_rows = []
    for group in COX_MODE_GROUPS:
        subj = pd.DataFrame({
            "subject_id": np.arange(len(vt)),
            "well_key": vt["well_key"].astype(str).to_numpy(),
            "duration": vt["tte"].to_numpy(float),
            "event": vt[f"event_{group}"].to_numpy(int),
            "covariate": vt["sour"].to_numpy(float),
        })
        subj = subj[subj["duration"] > 0]
        try:
            fit = fit_time_interaction_cox(subj)
            ci = bootstrap_gamma_ci(subj, n_boot=n_boot, seed=42)
            gamma_rows.append({
                "mode_group": group,
                "beta_sour": round(fit.beta, 4), "gamma": round(fit.gamma, 4),
                "gamma_model_se": round(fit.gamma_se, 4),
                "gamma_boot_ci_lo": ci["gamma_ci_lo"], "gamma_boot_ci_hi": ci["gamma_ci_hi"],
                "hr_sour_at_90d": round(fit.hr_at(90.0), 3),
                "hr_sour_at_365d": round(fit.hr_at(365.0), 3),
                "n_events": fit.n_events,
            })
            print(f"[B4] {group:<16} γ={fit.gamma:+.3f} "
                  f"boot95%[{ci['gamma_ci_lo']},{ci['gamma_ci_hi']}] "
                  f"HR(90d)={fit.hr_at(90.0):.2f}→HR(365d)={fit.hr_at(365.0):.2f}")
        except Exception as exc:
            print(f"[B4] {group:<16} γ fit failed: {exc}")
    pd.DataFrame(gamma_rows).to_csv(tbl / "b4_vt_gamma.csv", index=False, encoding="utf-8-sig")

    # ── per-mode CIF sour vs nonsour (Gray-style bootstrap) ──────────────────
    cif_rows = []
    for i, group in enumerate(MODE_GROUPS, start=1):
        res = bootstrap_cif_difference(
            vt, group_col="sour", group_a=1, group_b=0,
            duration_col="tte", event_code_col="event_code", cause_code=i,
            at_times=(90.0, 365.0, vt_end), cluster_col="well_key",
            n_boot=n_boot, seed=7,
        )
        res.insert(0, "mode_group", group)
        res = res.rename(columns={"cif_1": "cif_sour", "cif_0": "cif_nonsour"})
        cif_rows.append(res)
    pd.concat(cif_rows, ignore_index=True).to_csv(
        tbl / "b4_vt_cif_compare.csv", index=False, encoding="utf-8-sig")

    # ── forest of per-mode sour HR ───────────────────────────────────────────
    if not hr_df.empty:
        y = np.arange(len(hr_df))[::-1]
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        for yi, (_, r) in zip(y, hr_df.iterrows()):
            col = GROUP_COLORS.get(r["mode_group"], "#555")
            ax.plot([r["hr_ci_lo"], r["hr_ci_hi"]], [yi, yi], color=col, lw=2)
            ax.plot(r["hr"], yi, "o", color=col, ms=8)
            ax.text(r["hr_ci_hi"], yi + 0.1,
                    f" HR={r['hr']:.2f} [{r['hr_ci_lo']:.2f},{r['hr_ci_hi']:.2f}] "
                    f"(n_ev={r['n_events']})", va="bottom", fontsize=8)
        ax.axvline(1.0, color="black", lw=0.9, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(hr_df["mode_group"])
        ax.set_xscale("log")
        ax.set_xlabel("hazard ratio (log scale)")
        ax.set_title("B4 — Vt cause-specific sour HR by mode (clock=ttf_mix)")
        fig.tight_layout()
        fig.savefig(figs / "b4_vt_sour_hr.png", dpi=130)
        plt.close(fig)

    print(f"[B4] outputs written to: {out}")


if __name__ == "__main__":
    main()
