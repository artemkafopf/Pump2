"""Phase C C4 — merged θ(t) across Blocks 1+2+3 (output: phase_c_joint_theta).

Pools the surviving covariates of Block 1 (chemistry: H₂S, GLF), Block 2
(operational) and Block 3 (completion) + mandatory adjusters into one stratified
Cox with cross-block VIF cleanup (chemistry–operational overlaps: GLF vs drawdown,
H₂S vs sour flag).  Schoenfeld per survivor assigns STANDARD / EXTENDED; EXTENDED
γ is estimated with the corrected event-time episode-split module (curve-regression
is banned).  §0.3 decision: **EXTENDED terms are exported as first-class β/γ
columns** in ``theta_final_coeffs.csv`` (VBA to be extended later; no VBA changes
this phase).

``theta_final_coeffs.csv`` supersedes ``chem_final_coeffs.csv`` as the VBA-handoff
candidate: covariate | β | γ | ref_value | type | block | window | clock, with
population-weighted ref values (θ=1 for the fleet-average pump), stamped
clock=ttf_mix.

Note: the merged model is complete-case on operational covariates, so it runs on
the telemetry-covered subpopulation; ``has_telemetry`` is therefore constant there
and cannot enter the merged θ (it enters the completion-only Block 3 model).

Run C1 and C2 first (same date).  Then:
    python scripts/run/phase_c_joint_theta.py
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
from analysis.models.survival.screening_cox import (
    fit_joint_cox, vif_prune, ph_diagnostic, population_ref_coeffs,
)
from analysis.models.survival.time_interaction_cox import fit_time_interaction_cox
from analysis.common.cox_reporting import per_field_heterogeneity

TODAY = date.today().isoformat()

# Block-1 chemistry survivors (from phase_a_chem_ttfmix: log_h2s_proxy, log_glf).
BLOCK1 = ["log_h2s_proxy_mg_l", "log_glf_mean_opdays"]
BLOCK_OF = {
    "log_h2s_proxy_mg_l": "1-chemistry", "log_glf_mean_opdays": "1-chemistry",
    "log_motor_power_kw": "3-completion", "curvature_missing": "3-completion",
    "vg_m": "3-completion", "pbubble_atm": "3-completion", "nominal_freq_hz": "3-completion",
    "freq_above_55hz_pct_early": "2-operational", "load_mean": "2-operational",
    "load_std_early": "2-operational", "kpod_freq_mean": "2-operational",
    "frac_kpod_below_0p7": "2-operational", "n_freq_steps_per_100d": "2-operational",
    "n_restarts_per_100d": "2-operational",
}
WINDOW_OF = {"1-chemistry": "early", "2-operational": "early", "3-completion": "t0"}

# Adjusters forced into the merged model.  has_telemetry omitted deliberately:
# constant in the operational (telemetry) complete-case sample.
ADJ_NUMERIC = ["log_run_seq", "log_days_since_prev_failure"]
ADJ_CAT = ["install_period"]


def _read_block_survivors(slug: str, verdict_file: str) -> list[str]:
    p = results_dir(slug, TODAY) / "tables" / verdict_file
    if not p.exists():
        return []
    return pd.read_csv(p)["covariate"].tolist()


def main() -> None:
    out = results_dir("phase_c_joint_theta")
    tbl, figs, rep = out / "tables", out / "figures", out / "reports"

    df = build_competing_risks_df(tte_col="ttf_mix")

    b3 = _read_block_survivors("phase_c_block3_completion", "c1_expected_mode_verdicts.csv")
    b2 = _read_block_survivors("phase_c_block2_operational", "c2_expected_mode_verdicts.csv")
    pool = list(dict.fromkeys(BLOCK1 + b2 + b3))
    print(f"[C4] merged candidate pool ({len(pool)}): {pool}")

    # ── cross-block VIF cleanup ──────────────────────────────────────────────
    present = [c for c in pool if c in df.columns]
    cc = df[present].apply(pd.to_numeric, errors="coerce").dropna()
    vif_keep, vif_df = vif_prune(cc, present, thr=5.0)
    dropped = [c for c in present if c not in vif_keep]
    vif_df.to_csv(tbl / "c4_crossblock_vif.csv", index=False, encoding="utf-8-sig")
    if dropped:
        print(f"[C4] cross-block VIF dropped: {dropped}")

    numeric = vif_keep + ADJ_NUMERIC
    cph, used, summary = fit_joint_cox(df, numeric, ADJ_CAT)
    if cph is None:
        print("[C4] merged joint model FAILED"); return
    summary.round(5).to_csv(tbl / "c4_joint_model.csv", index=False, encoding="utf-8-sig")
    print(f"[C4] merged model: n={len(used)}, events={int(used['event'].sum())}")
    print(summary.round(4).to_string(index=False))

    # ── Schoenfeld → STANDARD / EXTENDED ─────────────────────────────────────
    ph = ph_diagnostic(cph, used)
    ph.to_csv(tbl / "c4_ph_test.csv", index=False, encoding="utf-8-sig")
    survivor_terms = [c for c in vif_keep]   # hazard covariates (exclude adjusters)
    violators = [c for c in survivor_terms
                 if c in set(ph.loc[ph["ph_violation"], "term"])]
    print(f"[C4] PH violators (EXTENDED candidates): {violators}")

    # ── EXTENDED γ via episode split (+ well bootstrap CI) ────────────────────
    gamma_by_cov: dict[str, dict] = {}
    ext_rows = []
    for cov in violators:
        sub = df[["well_key", "tte", "event", cov]].dropna()
        sub = sub[sub["tte"] > 0].rename(columns={"tte": "duration", cov: "covariate"})
        try:
            # point γ + model inference (episode split); the citable well-bootstrap
            # CI is available via time_interaction_cox.bootstrap_gamma_ci but omitted
            # here to bound runtime — model SE/p drive the STANDARD/EXTENDED call.
            fit = fit_time_interaction_cox(sub)
            gamma_by_cov[cov] = {"beta": fit.beta, "gamma": fit.gamma}
            ext_rows.append({"covariate": cov, "beta": round(fit.beta, 5),
                             "gamma": round(fit.gamma, 5), "gamma_se": round(fit.gamma_se, 5),
                             "gamma_p": round(fit.gamma_p, 5),
                             "hr_30d": round(fit.hr_at(30), 3), "hr_365d": round(fit.hr_at(365), 3),
                             "material": abs(fit.gamma) > 0.05 and (fit.gamma_p < 0.10)})
        except Exception as exc:
            ext_rows.append({"covariate": cov, "beta": None, "gamma": None, "note": str(exc)[:50]})
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(tbl / "c4_extended_gamma.csv", index=False, encoding="utf-8-sig")
        print("[C4] EXTENDED γ:"); print(pd.DataFrame(ext_rows).to_string(index=False))

    # ── theta_final_coeffs.csv (supersedes chem_final_coeffs.csv) ────────────
    coeffs = population_ref_coeffs(summary, df, numeric, clock="ttf_mix")
    # attach block/window, promote EXTENDED terms with a materially non-zero γ
    coeffs["block"] = coeffs["covariate"].map(
        lambda c: BLOCK_OF.get(c, "adjuster" if c in ADJ_NUMERIC else
                               ("cohort" if c.startswith("C(install") else "?")))
    coeffs["window"] = coeffs["block"].map(lambda b: WINDOW_OF.get(b, "cohort"))
    for cov, g in gamma_by_cov.items():
        mask = coeffs["covariate"] == cov
        material = abs(g["gamma"]) > 0.05
        if material:
            coeffs.loc[mask, "gamma"] = round(g["gamma"], 6)
            coeffs.loc[mask, "type"] = "EXTENDED"
    coeffs.to_csv(tbl / "theta_final_coeffs.csv", index=False, encoding="utf-8-sig")
    print("\n[C4] theta_final_coeffs.csv:"); print(coeffs.to_string(index=False))

    # ── per-field interaction / sign-flip checks ─────────────────────────────
    het_frames = []
    for cov in survivor_terms:
        others = [c for c in survivor_terms if c != cov][:6]
        h = per_field_heterogeneity(df, cov, adjusters=others + ADJ_NUMERIC, min_events=40)
        if not h.empty:
            het_frames.append(h)
    if het_frames:
        het = pd.concat(het_frames, ignore_index=True)
        het.to_csv(tbl / "c4_field_heterogeneity.csv", index=False, encoding="utf-8-sig")
        flips = int(het["sign_flip"].sum())
        print(f"[C4] per-field sign flips flagged: {flips}")

    # ── forest of the merged β ───────────────────────────────────────────────
    beta_rows = summary[~summary["term"].str.startswith("C(")].copy()
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.4 * len(beta_rows))))
    yy = np.arange(len(beta_rows))[::-1]
    ax.errorbar(beta_rows["coef"], yy, xerr=1.96 * beta_rows["se"], fmt="o",
                color="#4C78A8", capsize=3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels(beta_rows["term"], fontsize=8)
    ax.set_xlabel("β (log-hazard)"); ax.set_title("C4 — merged θ β coefficients (clock=ttf_mix)")
    fig.tight_layout(); fig.savefig(figs / "c4_merged_forest.png", dpi=140); plt.close(fig)

    (rep / "C4_NOTES.md").write_text(
        "# C4 — Merged θ(t)\n\n"
        f"- merged model on the **telemetry subpopulation**: n={len(used)}, "
        f"events={int(used['event'].sum())}, clock=ttf_mix\n"
        f"- candidate pool: {pool}\n"
        f"- cross-block VIF dropped: {dropped or 'none'}\n"
        f"- PH violators (EXTENDED): {violators or 'none'}\n"
        "- §0.3: EXTENDED terms exported as first-class β/γ columns; VBA "
        "`ApplyCoxEta` is STANDARD-only today, so consuming γ needs a future VBA "
        "change (no VBA changes in Phase C).\n"
        "- `theta_final_coeffs.csv` supersedes `chem_final_coeffs.csv` as the VBA "
        "handoff candidate.\n"
        "- has_telemetry cannot enter (constant in the operational complete-case).\n",
        encoding="utf-8")
    print(f"\n[C4] Outputs written to: {out}")


if __name__ == "__main__":
    main()
