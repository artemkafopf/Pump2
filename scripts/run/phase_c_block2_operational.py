"""Phase C C2 — Block 2 operational Cox (output: phase_c_block2_operational).

>>> ASSOCIATIONS UNDER ADJUSTMENT, NOT CAUSATION. <<<
Operational covariates are *chosen responses* to well conditions (confounding by
indication: troubled wells get run differently). This block reports associations
under adjustment for Block-1 chemistry and cohort/vintage; it never claims that
changing X will cause Y. Causal language is reserved for a future design-based
(within-well) analysis. (§0.2 disclaimer — carried verbatim.)

Same cascade as C1 on the C0 early-window covariates (first 30 operating days) +
the four mandatory adjusters + the Block-1 survivors as adjusters (is_sour_flagged
and log_glf; the B3 clean-refit lesson — H2S enters once, not both proxies).

Five named hypotheses each get a yes/no/inconclusive verdict:
  H-GLF     is the protective GLF→hydraulic effect an operating-point proxy?
  H-FREQ    level vs instability of frequency (electro-thermal channel)
  H-KPOD    off-BEP U-shape (<0.7 / 0.7-1.2 ref / >1.2), hydraulic-dominant
  H-CYCLING restarts / idle → all-cause + electro-thermal (start-up inrush)
  H-LOAD    load level (linear+quadratic) + instability → electro-thermal

Run:
    python scripts/run/phase_c_block2_operational.py
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
from analysis.models.survival.screening_cox import screen_and_fit, MANDATORY_NUMERIC
from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox
from analysis.common.cox_reporting import (
    cause_specific_sweep, forest_across_causes, per_field_heterogeneity,
)

DISCLAIMER = (
    "ASSOCIATIONS UNDER ADJUSTMENT, NOT CAUSATION. Operational covariates are chosen "
    "responses to well conditions (confounding by indication). Phase C reports "
    "associations under adjustment; it never claims changing X causes Y. (§0.2)"
)

# Block-2 early-window numeric candidates.
# NOTE: idle_frac is deliberately EXCLUDED here — it is a whole_run covariate
# (1 − ttf_true/run_days), a near-deterministic function of the outcome clock, so
# it is barred from any baseline Cox that feeds VBA (reverse causation; registry
# rule). It is examined only in the H-CYCLING diagnostic, with its caveat stamped.
NUMERIC_CANDIDATES = [
    "freq_mean", "freq_std_early", "freq_above_55hz_pct_early",
    "load_mean", "load_std_early",
    "kpod_mean", "kpod_freq_mean", "frac_kpod_below_0p7",
    "pzab_over_pbubble", "frac_pzab_below_1", "rpump_intake_mean",
    "n_freq_steps_per_100d", "n_restarts_per_100d",
    "log_glf_mean_opdays",
]
# Block-1 survivors carried as adjusters (H2S once as the flag; log_glf is a candidate).
BLOCK1_ADJ = ["is_sour_flagged"]
ADJ_NUMERIC = MANDATORY_NUMERIC + BLOCK1_ADJ
ADJ_CAT = ["install_period"]

EXPECTED_MODE = {
    "freq_mean": "electro-thermal", "freq_std_early": "electro-thermal",
    "freq_above_55hz_pct_early": "electro-thermal",
    "load_mean": "electro-thermal", "load_std_early": "electro-thermal",
    "n_freq_steps_per_100d": "electro-thermal", "n_restarts_per_100d": "electro-thermal",
    "idle_frac": "electro-thermal",
    "kpod_mean": "hydraulic", "frac_kpod_below_0p7": "hydraulic",
    "pzab_over_pbubble": "hydraulic", "frac_pzab_below_1": "hydraulic",
    "rpump_intake_mean": "hydraulic", "log_glf_mean_opdays": "hydraulic",
}

CS_ADJ = ["is_sour_flagged", "log_run_seq", "has_telemetry",
          "install_pre2020", "install_2023plus"]


def _cs_fit(df, covs, group, min_events=15):
    return fit_cause_specific_cox(
        df, covariates=covs, event_col=f"event_{group}", duration_col="tte",
        strata=["stratum_key"], cluster_col="well_key", min_events=min_events)


def _hr_row(res, cov):
    if not res.success:
        return {"hr": None, "p": None, "n_events": res.n_events, "note": res.message[:40]}
    r = res.summary[res.summary["covariate"] == cov]
    if r.empty:
        return {"hr": None, "p": None, "n_events": res.n_events, "note": "term absent"}
    return {"hr": round(float(r["hr"].iloc[0]), 4), "p": round(float(r["p"].iloc[0]), 4),
            "n_events": res.n_events, "note": ""}


# ---------------------------------------------------------------------------
# Named hypotheses
# ---------------------------------------------------------------------------

def h_glf(df) -> tuple[pd.DataFrame, str]:
    """Is protective GLF→hydraulic an operating-point proxy? Add pzab/kpod/rpump."""
    base = ["log_glf_mean_opdays"] + CS_ADJ
    plus = base + ["pzab_over_pbubble", "kpod_mean", "rpump_intake_mean"]
    r_base = _cs_fit(df, base, "hydraulic")
    r_plus = _cs_fit(df, plus, "hydraulic")
    rb, rp = _hr_row(r_base, "log_glf_mean_opdays"), _hr_row(r_plus, "log_glf_mean_opdays")
    tab = pd.DataFrame([{"model": "glf+adjusters", **rb},
                        {"model": "glf+adjusters+operating_point", **rp}])
    tab.insert(1, "covariate", "log_glf_mean_opdays")
    if rb["hr"] is None or rp["hr"] is None:
        verdict = "inconclusive (fit failed)"
    elif rp["p"] is not None and rp["p"] > 0.05 and abs(rp["hr"] - 1) < abs(rb["hr"] - 1) * 0.5:
        verdict = (f"ABSORBED — GLF HR {rb['hr']}→{rp['hr']} (p {rb['p']}→{rp['p']}); "
                   "operating-point proxy, not an independent gas-interference effect")
    elif rp["p"] is not None and rp["p"] < 0.05:
        verdict = (f"SURVIVES — GLF HR {rb['hr']}→{rp['hr']} (p {rp['p']}) after operating-point "
                   "adjustment; treat as a real gas-interference/regime effect")
    else:
        verdict = f"WEAKENED but not absorbed — GLF HR {rb['hr']}→{rp['hr']} (p {rp['p']})"
    return tab, verdict


def h_freq(df) -> tuple[pd.DataFrame, str]:
    """Level vs instability of frequency, all-cause + electro-thermal."""
    rows = []
    for cov in ["freq_mean", "freq_above_55hz_pct_early", "freq_std_early",
                "n_freq_steps_per_100d"]:
        r_all = fit_cause_specific_cox(df, covariates=[cov] + CS_ADJ, event_col="event",
                                       duration_col="tte", strata=["stratum_key"],
                                       cluster_col="well_key")
        r_et = _cs_fit(df, [cov] + CS_ADJ, "electro-thermal")
        rows.append({"covariate": cov, "kind": "level" if cov in
                     ("freq_mean", "freq_above_55hz_pct_early") else "instability",
                     **{f"allcause_{k}": v for k, v in _hr_row(r_all, cov).items()},
                     **{f"et_{k}": v for k, v in _hr_row(r_et, cov).items()}})
    tab = pd.DataFrame(rows)
    inst = tab[tab["kind"] == "instability"]
    lvl = tab[tab["kind"] == "level"]
    inst_sig = ((inst["et_p"] < 0.05) & inst["et_p"].notna()).any()
    lvl_sig = ((lvl["et_p"] < 0.05) & lvl["et_p"].notna()).any()
    if inst_sig and not lvl_sig:
        verdict = "YES to instability — freq instability (std/steps) hits electro-thermal; level does not (matches the Vt 60 Hz prior)"
    elif lvl_sig and not inst_sig:
        verdict = "level, not instability — freq level hits electro-thermal (contra the Vt prior)"
    elif inst_sig and lvl_sig:
        verdict = "both level and instability significant in electro-thermal"
    else:
        verdict = "inconclusive — neither freq level nor instability significant in electro-thermal"
    return tab, verdict


def h_kpod(df) -> tuple[pd.DataFrame, str]:
    """Off-BEP U-shape: kpod categories <0.7 / 0.7-1.2 (ref) / >1.2, hydraulic."""
    d = df.copy()
    d["kpod_lt07"] = (d["kpod_mean"] < 0.7).astype("float")
    d["kpod_gt12"] = (d["kpod_mean"] > 1.2).astype("float")
    d.loc[d["kpod_mean"].isna(), ["kpod_lt07", "kpod_gt12"]] = np.nan
    rows = []
    for grp in ("hydraulic", "electro-thermal", "protector"):
        res = _cs_fit(d, ["kpod_lt07", "kpod_gt12"] + CS_ADJ, grp)
        for cov in ("kpod_lt07", "kpod_gt12"):
            rows.append({"cause_group": grp, "covariate": cov, **_hr_row(res, cov)})
    # frac_kpod_below_0p7 continuous, hydraulic
    res_frac = _cs_fit(d, ["frac_kpod_below_0p7"] + CS_ADJ, "hydraulic")
    rows.append({"cause_group": "hydraulic", "covariate": "frac_kpod_below_0p7",
                 **_hr_row(res_frac, "frac_kpod_below_0p7")})
    tab = pd.DataFrame(rows)
    hyd = tab[tab["cause_group"] == "hydraulic"]
    lo = hyd[hyd["covariate"] == "kpod_lt07"]
    hi = hyd[hyd["covariate"] == "kpod_gt12"]
    lo_sig = not lo.empty and lo["p"].iloc[0] is not None and lo["p"].iloc[0] < 0.05 and lo["hr"].iloc[0] > 1
    hi_sig = not hi.empty and hi["p"].iloc[0] is not None and hi["p"].iloc[0] < 0.05 and hi["hr"].iloc[0] > 1
    if lo_sig and hi_sig:
        verdict = "YES U-shape — both underload (<0.7) and overload (>1.2) elevate hydraulic hazard"
    elif lo_sig:
        verdict = "underload only — kpod<0.7 elevates hydraulic hazard; overload side not significant"
    elif hi_sig:
        verdict = "overload only — kpod>1.2 elevates hydraulic hazard; underload side not significant"
    else:
        verdict = "inconclusive — no significant off-BEP hydraulic effect either side"
    return tab, verdict


def h_cycling(df) -> tuple[pd.DataFrame, str]:
    """Restarts (early-window, clean) + idle_frac (whole-run caveat) → allcause + ET."""
    rows = []
    for cov in ("n_restarts_per_100d", "idle_frac"):
        r_all = fit_cause_specific_cox(df, covariates=[cov] + CS_ADJ, event_col="event",
                                       duration_col="tte", strata=["stratum_key"],
                                       cluster_col="well_key")
        r_et = _cs_fit(df, [cov] + CS_ADJ, "electro-thermal")
        rows.append({"covariate": cov,
                     "window": "early" if cov == "n_restarts_per_100d" else "whole_run (reverse-causation caveat)",
                     **{f"allcause_{k}": v for k, v in _hr_row(r_all, cov).items()},
                     **{f"et_{k}": v for k, v in _hr_row(r_et, cov).items()}})
    tab = pd.DataFrame(rows)
    rst = tab[tab["covariate"] == "n_restarts_per_100d"].iloc[0]
    if rst["et_p"] is not None and rst["et_p"] < 0.05 and rst["et_hr"] > 1:
        verdict = "YES — early-window restarts elevate electro-thermal hazard (start-up inrush)"
    elif rst["allcause_p"] is not None and rst["allcause_p"] < 0.05 and rst["allcause_hr"] > 1:
        verdict = "partial — restarts elevate all-cause but not specifically electro-thermal"
    else:
        verdict = "inconclusive — early-window restart count not significant (idle_frac carries reverse-causation)"
    return tab, verdict


def h_load(df) -> tuple[pd.DataFrame, str]:
    """load_mean linear+quadratic and load_std_early → electro-thermal."""
    d = df.copy()
    m = d["load_mean"].mean()
    d["load_c"] = d["load_mean"] - m
    d["load_sq"] = d["load_c"] ** 2
    rows = []
    r_lin = _cs_fit(d, ["load_c", "load_sq"] + CS_ADJ, "electro-thermal")
    for cov in ("load_c", "load_sq"):
        rows.append({"covariate": cov, "form": "quadratic", **_hr_row(r_lin, cov)})
    r_std = _cs_fit(d, ["load_std_early"] + CS_ADJ, "electro-thermal")
    rows.append({"covariate": "load_std_early", "form": "instability", **_hr_row(r_std, "load_std_early")})
    tab = pd.DataFrame(rows)
    std_row = tab[tab["covariate"] == "load_std_early"].iloc[0]
    sq_row = tab[tab["covariate"] == "load_sq"].iloc[0]
    sig = []
    if std_row["p"] is not None and std_row["p"] < 0.05:
        sig.append("instability")
    if sq_row["p"] is not None and sq_row["p"] < 0.05:
        sig.append("quadratic(level)")
    verdict = ("YES — load " + "+".join(sig) + " elevates electro-thermal hazard") if sig \
        else "inconclusive — neither load level (quadratic) nor instability significant in electro-thermal"
    return tab, verdict


def main() -> None:
    out = results_dir("phase_c_block2_operational")
    tbl, figs, rep = out / "tables", out / "figures", out / "reports"

    df = build_competing_risks_df(tte_col="ttf_mix")
    print(f"[C2] {len(df)} runs, {int(df.event.sum())} all-cause events")
    print(f"[C2] {DISCLAIMER}")

    # ── main cascade for the merged-θ feed ───────────────────────────────────
    res = screen_and_fit(df, NUMERIC_CANDIDATES,
                         forced_numeric=ADJ_NUMERIC, forced_categorical=ADJ_CAT)
    res.univariate.to_csv(tbl / "c2_univariate.csv", index=False, encoding="utf-8-sig")
    if res.joint_summary is not None:
        res.joint_summary.round(5).to_csv(tbl / "c2_joint_model.csv", index=False, encoding="utf-8-sig")
    if res.ph is not None:
        res.ph.to_csv(tbl / "c2_ph_test.csv", index=False, encoding="utf-8-sig")
    if not res.vif.empty:
        res.vif.to_csv(tbl / "c2_vif.csv", index=False, encoding="utf-8-sig")
    survivors = [c for c in res.final_numeric if c not in ADJ_NUMERIC]
    print("[C2] cascade notes:"); [print("      ", n) for n in res.notes]
    print(f"[C2] joint survivors: {survivors}")
    print(res.joint_summary.round(4).to_string(index=False) if res.joint_summary is not None else "JOINT FAILED")

    # ── named hypotheses ─────────────────────────────────────────────────────
    verdicts = {}
    for name, fn in (("H-GLF", h_glf), ("H-FREQ", h_freq), ("H-KPOD", h_kpod),
                     ("H-CYCLING", h_cycling), ("H-LOAD", h_load)):
        tab, verdict = fn(df)
        tab.to_csv(tbl / f"c2_{name.lower().replace('-', '_')}.csv", index=False, encoding="utf-8-sig")
        verdicts[name] = verdict
        print(f"[C2] {name}: {verdict}")
    pd.DataFrame([{"hypothesis": k, "verdict": v} for k, v in verdicts.items()]).to_csv(
        tbl / "c2_hypothesis_verdicts.csv", index=False, encoding="utf-8-sig")

    # ── cause-specific sweep + forests for survivors ─────────────────────────
    if survivors:
        sweep = cause_specific_sweep(df, survivors + CS_ADJ)
        sweep.round(5).to_csv(tbl / "c2_cause_specific_coeffs.csv", index=False, encoding="utf-8-sig")
        mode_rows = []
        for cov in survivors:
            v = forest_across_causes(sweep, cov, figs / f"c2_forest_{cov}.png",
                                     expected_mode=EXPECTED_MODE.get(cov),
                                     title_prefix="C2 operational — ")
            mode_rows.append({"covariate": cov, "expected_mode": EXPECTED_MODE.get(cov, "?"),
                              "verdict": v})
        pd.DataFrame(mode_rows).to_csv(tbl / "c2_expected_mode_verdicts.csv", index=False, encoding="utf-8-sig")

        het_frames = [per_field_heterogeneity(df, cov, adjusters=["log_run_seq", "has_telemetry"])
                      for cov in survivors]
        het_frames = [h for h in het_frames if not h.empty]
        if het_frames:
            pd.concat(het_frames, ignore_index=True).to_csv(
                tbl / "c2_field_heterogeneity.csv", index=False, encoding="utf-8-sig")

    (rep / "C2_NOTES.md").write_text(
        "# C2 — Block 2 operational Cox\n\n"
        f"{DISCLAIMER}\n\n"
        f"- window = first 30 operating days (early); clock = ttf_mix\n"
        f"- n runs = {res.n}, events = {res.n_events} (telemetry-covered subpopulation)\n"
        f"- joint survivors: {survivors}\n\n## Hypothesis verdicts\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in verdicts.items()) + "\n",
        encoding="utf-8")
    print(f"\n[C2] Outputs written to: {out}")


if __name__ == "__main__":
    main()
