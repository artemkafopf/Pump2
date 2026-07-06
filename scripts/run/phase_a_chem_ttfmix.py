"""Phase A T1 — chemistry Cox refit on ttf_mix + GLF fix + КВЧ + early window.

Produces, under ``results/phase_a_chem_ttfmix/<date>/``:

  tables/clock_compare.csv       — univariate β/HR/p on calendar vs ttf_mix (side-by-side)
  tables/glf_rescreen.csv        — GLF re-screened with fixed aggregation (both new covariates)
  tables/glf_per_field.csv       — per-field β̂ for glf_mean_opdays (heterogeneity; Mc flip check)
  tables/kvch_sensitivity.csv    — imputed-fit β vs complete-case β + missing-indicator significance
  tables/kvch_impute_source.csv  — imputation-source share (measured/well/pad/stratum) per stratum
  tables/window_compare.csv      — β_early vs β_whole_run for the final survivors
  tables/chem_final_coeffs.csv   — VBA handoff, stamped clock=ttf_mix window=early
  figures/chem_forest_beta.png   — forest plot of the final (ttf_mix, early) model

Run:
    python scripts/run/phase_a_chem_ttfmix.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.data.chemistry_run_features import build_chemistry_df
from analysis.workflows.chemistry_cox import phase_chem as pc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Convergence.*")

MIN_STRATUM_EVENTS = 5

# Candidate list for T1: chemistry + the two fixed GLF covariates + КВЧ.
CANDIDATES = [
    "log_chloride_mg_l", "log_sulfate_mg_l", "log_calcium_mg_l",
    "log_bicarbonate_mg_l", "log_total_mineralization_g_l",
    "log_mechanical_impurities_mg_l", "log_h2s_proxy_mg_l",
    "log_glf_mean_opdays", "glf_frac_days_gas", "log_ca_so4",
    "ph", "watercut_percent",
]


def _fit_uni(df: pd.DataFrame, col: str) -> dict | None:
    """Univariate stratified cluster-robust Cox on df['tte']. Returns HR/β/p or None."""
    sub = df[["well_key", "tte", "event", "stratum_key", col]].dropna()
    sub = sub[sub["tte"] > 0]
    counts = sub.groupby("stratum_key")["event"].sum()
    sub = sub[sub["stratum_key"].isin(counts[counts >= MIN_STRATUM_EVENTS].index)]
    if int(sub["event"].sum()) < 15:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="tte", event_col="event", strata=["stratum_key"],
                cluster_col="well_key", formula=col, robust=True)
        s = cph.summary
        return {
            "beta": float(s.loc[col, "coef"]),
            "HR": float(np.exp(s.loc[col, "coef"])),
            "p": float(s.loc[col, "p"]),
            "c_index": float(cph.concordance_index_),
        }
    except Exception:
        return None


def clock_compare(df_mix: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CANDIDATES:
        if col not in df_mix.columns:
            continue
        m = _fit_uni(df_mix, col)
        c = _fit_uni(df_cal, col)
        rows.append({
            "covariate": col,
            "beta_cal": round(c["beta"], 5) if c else None,
            "beta_mix": round(m["beta"], 5) if m else None,
            "HR_cal": round(c["HR"], 4) if c else None,
            "HR_mix": round(m["HR"], 4) if m else None,
            "p_cal": round(c["p"], 5) if c else None,
            "p_mix": round(m["p"], 5) if m else None,
        })
    return pd.DataFrame(rows)


def glf_per_field(df: pd.DataFrame, col: str = "log_glf_mean_opdays") -> pd.DataFrame:
    """Refit the GLF covariate within each field subset (heterogeneity / sign-flip)."""
    rows = []
    for field, g in df.groupby("field"):
        sub = g[["well_key", "tte", "event", "stratum_key", col]].dropna()
        sub = sub[sub["tte"] > 0]
        counts = sub.groupby("stratum_key")["event"].sum()
        sub = sub[sub["stratum_key"].isin(counts[counts >= MIN_STRATUM_EVENTS].index)]
        n_ev = int(sub["event"].sum())
        if n_ev < 15 or sub["stratum_key"].nunique() < 1:
            rows.append({"field": field, "n_events": n_ev, "beta": None, "HR": None,
                         "p": None, "note": "insufficient events"})
            continue
        try:
            cph = CoxPHFitter()
            strata = ["stratum_key"] if sub["stratum_key"].nunique() > 1 else None
            cph.fit(sub, duration_col="tte", event_col="event", strata=strata,
                    cluster_col="well_key", formula=col, robust=True)
            b = float(cph.summary.loc[col, "coef"])
            rows.append({"field": field, "n_events": n_ev, "beta": round(b, 5),
                         "HR": round(float(np.exp(b)), 4),
                         "p": round(float(cph.summary.loc[col, "p"]), 5),
                         "note": "sign-flip" if b > 0 else ""})
        except Exception as exc:
            rows.append({"field": field, "n_events": n_ev, "beta": None, "HR": None,
                         "p": None, "note": str(exc)[:40]})
    out = pd.DataFrame(rows).sort_values("n_events", ascending=False)
    return out


def kvch_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputed vs complete-case КВЧ β + missing-indicator significance in a joint fit.

    Returns (sensitivity_table, imputation_source_share_by_stratum).
    """
    col = "log_mechanical_impurities_mg_l"
    miss = "mechanical_impurities_mg_l_missing"

    # (a) imputed fit with the missing indicator alongside the covariate
    sub = df[["well_key", "tte", "event", "stratum_key", col, miss]].dropna(subset=["tte", col])
    sub = sub[sub["tte"] > 0]
    counts = sub.groupby("stratum_key")["event"].sum()
    sub = sub[sub["stratum_key"].isin(counts[counts >= MIN_STRATUM_EVENTS].index)]
    rows = []
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="tte", event_col="event", strata=["stratum_key"],
                cluster_col="well_key", formula=f"{col} + {miss}", robust=True)
        rows.append({"fit": "imputed+missing_indicator", "term": col,
                     "beta": round(float(cph.summary.loc[col, "coef"]), 5),
                     "p": round(float(cph.summary.loc[col, "p"]), 5),
                     "n_events": int(sub["event"].sum())})
        rows.append({"fit": "imputed+missing_indicator", "term": miss,
                     "beta": round(float(cph.summary.loc[miss, "coef"]), 5),
                     "p": round(float(cph.summary.loc[miss, "p"]), 5),
                     "n_events": int(sub["event"].sum())})
    except Exception as exc:
        rows.append({"fit": "imputed+missing_indicator", "term": col, "beta": None,
                     "p": None, "n_events": None, "note": str(exc)[:50]})

    # (b) complete-case fit: real КВЧ measurements only
    cc = df[df[miss] == 0][["well_key", "tte", "event", "stratum_key", col]].dropna()
    cc = cc[cc["tte"] > 0]
    counts = cc.groupby("stratum_key")["event"].sum()
    cc = cc[cc["stratum_key"].isin(counts[counts >= MIN_STRATUM_EVENTS].index)]
    try:
        cph = CoxPHFitter()
        cph.fit(cc, duration_col="tte", event_col="event", strata=["stratum_key"],
                cluster_col="well_key", formula=col, robust=True)
        rows.append({"fit": "complete_case", "term": col,
                     "beta": round(float(cph.summary.loc[col, "coef"]), 5),
                     "p": round(float(cph.summary.loc[col, "p"]), 5),
                     "n_events": int(cc["event"].sum())})
    except Exception as exc:
        rows.append({"fit": "complete_case", "term": col, "beta": None, "p": None,
                     "n_events": int(cc["event"].sum()), "note": str(exc)[:50]})

    # imputation-source share by stratum (measured vs imputed)
    share_rows = []
    for stratum, g in df.groupby("stratum_key"):
        n = len(g)
        n_meas = int((g[miss] == 0).sum())
        share_rows.append({"stratum_key": stratum, "n_runs": n,
                           "n_measured": n_meas,
                           "pct_measured": round(100 * n_meas / n, 1) if n else 0.0})
    share = pd.DataFrame(share_rows).sort_values("n_runs", ascending=False)
    return pd.DataFrame(rows), share


def run_final_model(df: pd.DataFrame):
    """Run the phase_chem adaptive cascade on the given df, return (cph, std, ext)."""
    df = pc._add_logt_terms(df, [c for c in pc.CANDIDATES if c in df.columns])
    uni = pc.run_univariate(df)
    survivors = uni.loc[uni["pass_screen"], "covariate"].tolist()
    if not survivors:
        return None, [], [], uni
    ph = pc.run_ph_tests(df, survivors)
    std = ph.loc[ph["assignment"] == "STANDARD", "covariate"].tolist()
    ext = ph.loc[ph["assignment"] == "EXTENDED", "covariate"].tolist()
    kept = pc.correlation_screen(df, std + ext, uni)
    std = [c for c in kept if c in std]
    ext = [c for c in kept if c in ext]
    cph = pc.run_joint_model(df, std, ext)
    if cph is None:
        return None, std, ext, uni
    std2, ext2, _ = pc.vif_check(df, std, ext)
    if set(std2) != set(std) or set(ext2) != set(ext):
        cph2 = pc.run_joint_model(df, std2, ext2)
        if cph2 is not None:
            cph, std, ext = cph2, std2, ext2
    return cph, std, ext, uni


def window_compare(df_early: pd.DataFrame, df_whole: pd.DataFrame,
                   covs: list[str]) -> pd.DataFrame:
    rows = []
    for col in covs:
        e = _fit_uni(df_early, col) if col in df_early.columns else None
        w = _fit_uni(df_whole, col) if col in df_whole.columns else None
        rows.append({
            "covariate": col,
            "beta_early": round(e["beta"], 5) if e else None,
            "beta_whole_run": round(w["beta"], 5) if w else None,
            "HR_early": round(e["HR"], 4) if e else None,
            "HR_whole_run": round(w["HR"], 4) if w else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    out = results_dir("phase_a_chem_ttfmix")
    tbl, fig_dir = out / "tables", out / "figures"

    print("[T1] Building chemistry frames (ttf_mix/run_days × early/whole_run)...")
    df_mix_early = build_chemistry_df(tte_col="ttf_mix", window="early")
    df_mix_whole = build_chemistry_df(tte_col="ttf_mix", window="whole_run")
    df_cal_whole = build_chemistry_df(tte_col="run_days", window="whole_run")

    # 1) Side-by-side clock comparison (univariate)
    print("[T1] Clock comparison (calendar vs ttf_mix)...")
    cc = clock_compare(df_mix_whole, df_cal_whole)
    cc.to_csv(tbl / "clock_compare.csv", index=False, encoding="utf-8-sig")
    print(cc.to_string(index=False))

    # 2) GLF re-screen (fixed aggregation) + per-field β̂
    print("\n[T1] GLF re-screen + per-field β̂...")
    glf_rows = []
    for col in ("log_glf_mean_opdays", "glf_frac_days_gas"):
        r = _fit_uni(df_mix_whole, col)
        glf_rows.append({"covariate": col, **({"beta": round(r["beta"], 5), "HR": round(r["HR"], 4),
                        "p": round(r["p"], 5)} if r else {"beta": None, "HR": None, "p": None})})
    pd.DataFrame(glf_rows).to_csv(tbl / "glf_rescreen.csv", index=False, encoding="utf-8-sig")
    gpf = glf_per_field(df_mix_whole)
    gpf.to_csv(tbl / "glf_per_field.csv", index=False, encoding="utf-8-sig")
    print(gpf.to_string(index=False))

    # 3) КВЧ sensitivity + imputation-source share
    print("\n[T1] КВЧ complete-case + missing-indicator sensitivity...")
    kv, share = kvch_sensitivity(df_mix_whole)
    kv.to_csv(tbl / "kvch_sensitivity.csv", index=False, encoding="utf-8-sig")
    share.to_csv(tbl / "kvch_impute_source.csv", index=False, encoding="utf-8-sig")
    print(kv.to_string(index=False))

    # 4) Final (ttf_mix, early) model → stamped chem_final_coeffs.csv
    print("\n[T1] Final adaptive model on (ttf_mix, early)...")
    cph, std, ext, uni = run_final_model(df_mix_early)
    uni.to_csv(tbl / "chem_univariate_ttfmix_early.csv", index=False, encoding="utf-8-sig")
    if cph is not None:
        coeffs = pc.build_final_coeffs(cph, df_mix_early, std, ext)
        coeffs["clock"] = "ttf_mix"
        coeffs["window"] = "early"
        coeffs.to_csv(tbl / "chem_final_coeffs.csv", index=False, encoding="utf-8-sig")
        print(coeffs.to_string(index=False))
        summ = cph.summary.reset_index().rename(columns={"covariate": "term"})
        beta_rows = summ[~summ["term"].str.endswith("_x_logt")]
        pc._forest_plot(beta_rows, "Chemistry Cox — ttf_mix / early window",
                        fig_dir / "chem_forest_beta.png")

        # 5) early vs whole-run β for the survivors
        survivors = std + ext
        wc = window_compare(df_mix_early, df_mix_whole, survivors)
        wc.to_csv(tbl / "window_compare.csv", index=False, encoding="utf-8-sig")
        print("\n[T1] Early vs whole-run β:")
        print(wc.to_string(index=False))
    else:
        print("[T1] No covariates survived — no final coeffs written.")

    print(f"\n[T1] Outputs written to: {out}")


if __name__ == "__main__":
    main()
