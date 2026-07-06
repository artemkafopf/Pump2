"""Phase B follow-up — tighten the two inference gaps flagged in review.

1. **B4 mode contrast (Lunn–McNeil).** The report's flat-HR headline ("H₂S hits
   cable/motor harder than the pump", 2.72 vs 1.93) rested on overlapping CIs with
   no formal test.  Here the Vt data are stacked per cause (one copy per mode
   group, cause-specific baselines via ``strata = contractor|mode``) and fitted
   with mode-specific ``sour`` terms in one partial likelihood, cluster-robust by
   well — the covariance between coefficients then gives a proper z-test for
   HR_electro-thermal / HR_hydraulic.
     → tables/b4_mode_contrast.csv (in phase_b_vt_h2s_modes)

2. **B4 Δγ paired well bootstrap.** The γ contrast (+0.65 vs −0.21) compared two
   bootstrap CIs computed independently; modes share wells, so the honest test is
   the *paired* difference: resample wells once per draw, fit both per-mode
   time-interaction Cox models on the same resample, take the percentile CI of
   γ_et − γ_hyd.
     → tables/b4_gamma_contrast_boot.csv (in phase_b_vt_h2s_modes)

3. **B3 clean sour-flag refit.** ``is_sour_flagged`` was fitted alongside its
   collinear parent ``log_h2s_proxy_mg_l`` (both derive from ``h2s_proxy``), which
   inflates its variance and muddies the §0.2 stratum decision.  Refit each
   cause-specific model without the continuous term for a citable flag HR.
     → tables/b3_sour_flag_clean.csv (in phase_b_cause_cox)

Run:
    python scripts/run/phase_b_followup.py [n_boot]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lifelines import CoxPHFitter

from analysis.paths import results_dir
from analysis.data.competing_risks_loader import build_competing_risks_df, COX_MODE_GROUPS
from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox
from analysis.models.survival.time_interaction_cox import fit_time_interaction_cox

# B3 covariate set minus the collinear continuous H₂S term.
B3_COVARIATES_CLEAN = [
    "is_sour_flagged",
    "log_glf_mean_opdays",
    "watercut_daily",
    "log_mechanical_impurities_mg_l",
    "log_total_mineralization_g_l",
    "log_ca_so4_product",
]

_SLUG = {"hydraulic": "hyd", "electro-thermal": "et", "protector": "prot"}


def _vt_frame() -> pd.DataFrame:
    df = build_competing_risks_df(tte_col="ttf_mix")
    vt = df[df["field"] == "Vt"].copy()
    vt["sour"] = (vt["h2s_class"] == "sour").astype(int)
    return vt.reset_index(drop=True)


# ── 1. Lunn–McNeil stacked contrast ──────────────────────────────────────────

def lunn_mcneil_contrast(vt: pd.DataFrame, tbl: Path) -> None:
    parts = []
    for group in COX_MODE_GROUPS:
        part = pd.DataFrame({
            "tte": vt["tte"].astype(float),
            "event": vt[f"event_{group}"].astype(int),
            "well_key": vt["well_key"].astype(str),
            "stratum_lm": vt["contractor_group"].astype(str) + "|" + group,
        })
        for g2 in COX_MODE_GROUPS:
            part[f"sour_{_SLUG[g2]}"] = (vt["sour"] * (g2 == group)).astype(float)
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True)
    stacked = stacked[stacked["tte"] > 0]

    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(stacked, duration_col="tte", event_col="event",
            strata=["stratum_lm"], cluster_col="well_key", robust=True)
    b = cph.params_
    V = cph.variance_matrix_

    rows = []
    for name in b.index:
        rows.append({
            "term": name, "kind": "coefficient",
            "estimate": round(float(b[name]), 4),
            "hr_or_ratio": round(float(np.exp(b[name])), 3),
            "se": round(float(np.sqrt(V.loc[name, name])), 4),
            "z": round(float(b[name] / np.sqrt(V.loc[name, name])), 3),
            "p": round(float(2 * (1 - stats.norm.cdf(abs(b[name] / np.sqrt(V.loc[name, name]))))), 4),
        })
    for a, c in (("sour_et", "sour_hyd"), ("sour_prot", "sour_hyd"), ("sour_prot", "sour_et")):
        d = float(b[a] - b[c])
        se = float(np.sqrt(V.loc[a, a] + V.loc[c, c] - 2 * V.loc[a, c]))
        z = d / se if se > 0 else 0.0
        rows.append({
            "term": f"{a} - {c}", "kind": "contrast",
            "estimate": round(d, 4),
            "hr_or_ratio": round(float(np.exp(d)), 3),
            "se": round(se, 4), "z": round(z, 3),
            "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(tbl / "b4_mode_contrast.csv", index=False, encoding="utf-8-sig")
    print("[followup] Lunn–McNeil stacked model (cluster-robust by well):")
    print(out.to_string(index=False))


# ── 2. Paired Δγ well bootstrap ──────────────────────────────────────────────

def paired_gamma_bootstrap(vt: pd.DataFrame, tbl: Path, n_boot: int, seed: int = 42) -> None:
    def _gamma(frame: pd.DataFrame, group: str) -> float:
        subj = pd.DataFrame({
            "well_key": frame["well_key"].astype(str).to_numpy(),
            "duration": frame["tte"].to_numpy(float),
            "event": frame[f"event_{group}"].to_numpy(int),
            "covariate": frame["sour"].to_numpy(float),
        })
        subj = subj[subj["duration"] > 0].reset_index(drop=True)
        return float(fit_time_interaction_cox(subj).gamma)

    g_hyd_pt = _gamma(vt, "hydraulic")
    g_et_pt = _gamma(vt, "electro-thermal")

    wells = vt["well_key"].astype(str).unique()
    by_well = {w: sub for w, sub in vt.groupby(vt["well_key"].astype(str))}
    rng = np.random.default_rng(seed)

    diffs, oks = [], 0
    for i in range(n_boot):
        drawn = rng.choice(wells, size=len(wells), replace=True)
        boot = pd.concat([by_well[w] for w in drawn], ignore_index=True)
        try:
            diffs.append(_gamma(boot, "electro-thermal") - _gamma(boot, "hydraulic"))
            oks += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"[followup]   Δγ bootstrap {i + 1}/{n_boot} (ok={oks})")

    d = np.asarray(diffs)
    lo, hi = (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) >= 20 else (np.nan, np.nan)
    row = pd.DataFrame([{
        "gamma_hydraulic_point": round(g_hyd_pt, 4),
        "gamma_electro_thermal_point": round(g_et_pt, 4),
        "delta_gamma_point": round(g_et_pt - g_hyd_pt, 4),
        "delta_gamma_boot_median": round(float(np.median(d)), 4) if len(d) else None,
        "delta_gamma_ci_lo": round(float(lo), 4) if np.isfinite(lo) else None,
        "delta_gamma_ci_hi": round(float(hi), 4) if np.isfinite(hi) else None,
        "n_boot_ok": int(oks),
        "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
    }])
    row.to_csv(tbl / "b4_gamma_contrast_boot.csv", index=False, encoding="utf-8-sig")
    print("[followup] Paired Δγ (electro-thermal − hydraulic):")
    print(row.to_string(index=False))


# ── 3. Clean sour-flag refit (B3 minus log_h2s_proxy) ────────────────────────

def clean_flag_refit(tbl: Path) -> None:
    df = build_competing_risks_df(tte_col="ttf_mix")
    rows = []
    for group in COX_MODE_GROUPS:
        res = fit_cause_specific_cox(
            df, covariates=B3_COVARIATES_CLEAN, event_col=f"event_{group}",
            duration_col="tte", strata=["stratum_key"], cluster_col="well_key",
        )
        if not res.success:
            print(f"[followup] {group}: {res.message}")
            continue
        r = res.summary[res.summary["covariate"] == "is_sour_flagged"].iloc[0]
        rows.append({
            "cause_group": group,
            "hr": round(float(r["hr"]), 3),
            "hr_ci_lo": round(float(r["hr_ci_lo"]), 3),
            "hr_ci_hi": round(float(r["hr_ci_hi"]), 3),
            "p": round(float(r["p"]), 4),
            "n_events": res.n_events,
            "model": "B3 covariates minus log_h2s_proxy_mg_l",
        })
    out = pd.DataFrame(rows)
    out.to_csv(tbl / "b3_sour_flag_clean.csv", index=False, encoding="utf-8-sig")
    print("[followup] is_sour_flagged, clean refit (no collinear log_h2s_proxy):")
    print(out.to_string(index=False))


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tbl_b4 = results_dir("phase_b_vt_h2s_modes") / "tables"
    tbl_b3 = results_dir("phase_b_cause_cox") / "tables"

    vt = _vt_frame()
    print(f"[followup] Vt runs={len(vt)} sour={int(vt['sour'].sum())}")

    lunn_mcneil_contrast(vt, tbl_b4)
    clean_flag_refit(tbl_b3)
    paired_gamma_bootstrap(vt, tbl_b4, n_boot=n_boot)
    print("[followup] done.")


if __name__ == "__main__":
    main()
