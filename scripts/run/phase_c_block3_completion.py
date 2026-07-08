"""Phase C C1 — Block 3 completion / design Cox (output: phase_c_block3_completion).

What the *installed pump* brings to the hazard — all covariates knowable at t=0,
so this block is free of the exposure-window / reverse-causation problems of Block
2.  Procedure:

  1. Collinearity triage of the "size axis" (motor_power ~ nominal_flow ~ stages ~
     head_per_stage ~ gabarit): Spearman + VIF, keep physically distinct residuals.
  2. Screening cascade (univariate → correlation → joint + VIF → Schoenfeld PH),
     pooled all-cause on ttf_mix, mandatory adjusters forced in.
  3. Categorical design factors (pump_series, pump_gabarit) added via ΔAIC.
  4. Cause-specific pass for joint survivors (hydraulic / electro-thermal /
     protector) + "expected mode?" verdict.
  5. Per-field heterogeneity for every survivor.
  6. EXTENDED γ (event-time episode split) for any PH-violating survivor.

§0.1 proxy flags stamped: vg_m = uninterpreted monotone depth proxy; curvature =
unitless dogleg proxy (+missing indicator); pump_gabarit categorical (never numeric OD).

Run:
    python scripts/run/phase_c_block3_completion.py
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
from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.models.survival.screening_cox import (
    screen_and_fit, fit_joint_cox, correlation_prune, vif_prune,
    MANDATORY_NUMERIC,
)
from analysis.models.survival.time_interaction_cox import fit_time_interaction_cox
from analysis.common.cox_reporting import (
    cause_specific_sweep, forest_across_causes, per_field_heterogeneity,
)

PROXY_FLAGS = (
    "§0.1 proxy flags: vg_m = uninterpreted monotone depth proxy (units unconfirmed); "
    "curvature = unitless dogleg proxy (+missing indicator); pump_gabarit categorical "
    "(never numeric OD). Effect directions reportable; physical-envelope claims are not."
)

# Numeric completion candidates (curvature carries its missing indicator).
NUMERIC_CANDIDATES = [
    "stages", "head_per_stage", "log_motor_power_kw", "log_nominal_flow_m3d",
    "curvature", "curvature_missing", "vg_m", "pbubble_atm", "nominal_freq_hz",
]
# The "size axis" whose collinearity is triaged before screening.
SIZE_AXIS = ["stages", "head_per_stage", "log_motor_power_kw",
             "log_nominal_flow_m3d", "vg_m", "pbubble_atm", "nominal_freq_hz"]

# Physically expected mode for the cause-specific "expected mode?" audit.
EXPECTED_MODE = {
    "curvature": "electro-thermal",       # dogleg → cable damage
    "vg_m": "electro-thermal",            # depth → temperature → insulation
    "head_per_stage": "hydraulic",        # stage aggressiveness
    "stages": "hydraulic",
    "log_motor_power_kw": "electro-thermal",
    "log_nominal_flow_m3d": "hydraulic",
    "pbubble_atm": "hydraulic",
    "nominal_freq_hz": "electro-thermal",
}

ADJ_CAT = ["install_period"]


def _collapse_gabarit(s: pd.Series) -> pd.Series:
    keep = {"5A", "4", "5", "3", "6"}
    return s.where(s.isin(keep), other="other").fillna("other")


def main() -> None:
    out = results_dir("phase_c_block3_completion")
    tbl, figs, rep = out / "tables", out / "figures", out / "reports"

    df = build_competing_risks_df(tte_col="ttf_mix")
    df["pump_gabarit_grp"] = _collapse_gabarit(df["pump_gabarit"])
    print(f"[C1] {len(df)} runs, {int(df.event.sum())} all-cause events")

    # ── 1. Size-axis collinearity triage ─────────────────────────────────────
    corr = df[SIZE_AXIS].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr.round(3).to_csv(tbl / "size_axis_spearman.csv", encoding="utf-8-sig")
    vif_keep, vif_df = vif_prune(df.dropna(subset=SIZE_AXIS), SIZE_AXIS, thr=5.0)
    vif_df.to_csv(tbl / "size_axis_vif.csv", index=False, encoding="utf-8-sig")
    print("[C1] size-axis Spearman (|ρ|>0.65 pairs):")
    for i, a in enumerate(SIZE_AXIS):
        for b in SIZE_AXIS[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.65:
                print(f"      {a} ~ {b}: ρ={r:.3f}")
    print(f"[C1] size-axis VIF-kept: {vif_keep}")

    # ── 2. Screening cascade (numeric candidates + mandatory adjusters) ──────
    res = screen_and_fit(df, NUMERIC_CANDIDATES, forced_categorical=ADJ_CAT)
    res.univariate.to_csv(tbl / "c1_univariate.csv", index=False, encoding="utf-8-sig")
    if res.joint_summary is not None:
        res.joint_summary.round(5).to_csv(tbl / "c1_joint_model.csv", index=False, encoding="utf-8-sig")
    if res.ph is not None:
        res.ph.to_csv(tbl / "c1_ph_test.csv", index=False, encoding="utf-8-sig")
    if not res.vif.empty:
        res.vif.to_csv(tbl / "c1_vif.csv", index=False, encoding="utf-8-sig")
    print("[C1] cascade notes:"); [print("      ", n) for n in res.notes]
    print(res.joint_summary.round(4).to_string(index=False) if res.joint_summary is not None else "JOINT FAILED")

    # Surviving numeric hazard covariates (exclude the forced adjusters).
    survivors = [c for c in res.final_numeric if c not in MANDATORY_NUMERIC]
    print(f"[C1] joint survivors (hazard covariates): {survivors}")

    # ── 3. Categorical design factors via ΔAIC ───────────────────────────────
    base_terms = res.final_numeric
    cph_base, used_base, _ = fit_joint_cox(df, base_terms, ADJ_CAT)
    aic_rows = []
    if cph_base is not None:
        aic_base = float(cph_base.AIC_partial_)
        for cat in ("pump_series", "pump_gabarit_grp"):
            cph_c, used_c, _ = fit_joint_cox(df, base_terms, ADJ_CAT + [cat])
            if cph_c is not None:
                # refit base on the same complete-case sample for a fair ΔAIC
                cph_b2, _, _ = fit_joint_cox(used_c, base_terms, ADJ_CAT)
                aic_b = float(cph_b2.AIC_partial_) if cph_b2 is not None else aic_base
                aic_rows.append({"factor": cat, "aic_without": round(aic_b, 2),
                                 "aic_with": round(float(cph_c.AIC_partial_), 2),
                                 "delta_aic": round(aic_b - float(cph_c.AIC_partial_), 2),
                                 "n": int(len(used_c)),
                                 "improves": aic_b - float(cph_c.AIC_partial_) > 2})
    pd.DataFrame(aic_rows).to_csv(tbl / "c1_design_factor_aic.csv", index=False, encoding="utf-8-sig")
    print("[C1] design-factor ΔAIC:"); print(pd.DataFrame(aic_rows).to_string(index=False) if aic_rows else "  none")

    # ── 4. Cause-specific pass + forests + expected-mode verdicts ────────────
    cs_covs = survivors + ["is_sour_flagged", "log_run_seq", "has_telemetry",
                           "install_pre2020", "install_2023plus"]
    sweep = cause_specific_sweep(df, cs_covs)
    sweep.round(5).to_csv(tbl / "c1_cause_specific_coeffs.csv", index=False, encoding="utf-8-sig")
    verdicts = []
    for cov in survivors:
        v = forest_across_causes(sweep, cov, figs / f"c1_forest_{cov}.png",
                                 expected_mode=EXPECTED_MODE.get(cov),
                                 title_prefix="C1 completion — ")
        verdicts.append({"covariate": cov, "expected_mode": EXPECTED_MODE.get(cov, "?"),
                         "verdict": v})
    pd.DataFrame(verdicts).to_csv(tbl / "c1_expected_mode_verdicts.csv", index=False, encoding="utf-8-sig")
    print("[C1] expected-mode verdicts:"); print(pd.DataFrame(verdicts).to_string(index=False))

    # ── 5. Per-field heterogeneity for every survivor ────────────────────────
    het_frames = []
    for cov in survivors:
        h = per_field_heterogeneity(df, cov, adjusters=["log_run_seq", "has_telemetry"])
        if not h.empty:
            het_frames.append(h)
    if het_frames:
        pd.concat(het_frames, ignore_index=True).to_csv(
            tbl / "c1_field_heterogeneity.csv", index=False, encoding="utf-8-sig")

    # ── 6. EXTENDED γ for PH-violating survivors ─────────────────────────────
    ext_rows = []
    if res.ph is not None:
        violators = [c for c in survivors
                     if c in set(res.ph.loc[res.ph["ph_violation"], "term"])]
        for cov in violators:
            sub = df[["well_key", "tte", "event", cov]].dropna()
            sub = sub[sub["tte"] > 0].rename(columns={"tte": "duration", cov: "covariate"})
            try:
                # point γ + model p (single episode-split fit); bootstrap CI is a
                # C4 concern for the final handoff covariates.
                fit = fit_time_interaction_cox(sub)
                ext_rows.append({"covariate": cov, "beta": round(fit.beta, 4),
                                 "gamma": round(fit.gamma, 4), "gamma_se": round(fit.gamma_se, 4),
                                 "gamma_p": round(fit.gamma_p, 4),
                                 "hr_30d": round(fit.hr_at(30), 3), "hr_365d": round(fit.hr_at(365), 3),
                                 "trend": "increasing" if fit.gamma > 0 else "decreasing"})
            except Exception as exc:
                ext_rows.append({"covariate": cov, "beta": None, "gamma": None,
                                 "note": str(exc)[:50]})
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(tbl / "c1_extended_gamma.csv", index=False, encoding="utf-8-sig")
        print("[C1] EXTENDED γ (episode-split):"); print(pd.DataFrame(ext_rows).to_string(index=False))

    # ── notes / provenance ───────────────────────────────────────────────────
    (rep / "C1_NOTES.md").write_text(
        "# C1 — Block 3 completion Cox\n\n"
        f"{PROXY_FLAGS}\n\n"
        f"- n runs = {res.n}, all-cause events = {res.n_events}, clock = ttf_mix\n"
        f"- joint survivors: {survivors}\n"
        f"- cascade: {res.notes}\n"
        "- EPV budget: 1,527 pooled events; protector cause-specific limited to survivors.\n",
        encoding="utf-8")
    print(f"\n[C1] Outputs written to: {out}")


if __name__ == "__main__":
    main()
