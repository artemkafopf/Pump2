"""Liquid / water / ion-load proxy Cox screen.

This is a companion to results/chemistry_cox/2026-07-03.  It keeps the same
stratified, cluster-robust Cox machinery, but replaces concentration means with
cumulative and mean-rate exposure proxies:

  Ql       = sum daily qliq, m3
  Qw       = sum daily qliq * watercut / 100, m3
  QC_i     = sum daily qliq * watercut / 100 * C_i / 1000, kg

  Ql_mean   = mean daily qliq, m3/d
  Qw_mean   = mean daily qliq * watercut / 100, m3/d
  QC_i_mean = mean daily qliq * watercut / 100 * C_i / 1000, kg/d

where i is Ca, SO4, Cl and C_i comes from proc__daily_lab.  proc__daily_lab is
already a daily step-function expansion of sparse lab samples.

Run:
    python scripts/run/chem_cumulative_ion_proxies.py
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.chemistry_run_features import CONTRACTOR_MAP, H2S_SOUR_THRESHOLD
from analysis.paths import WAREHOUSE_DIR, results_dir
from analysis.workflows.chemistry_cox import phase_chem as pc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Convergence.*")

MIN_STRATUM_EVENTS = 5
MIN_TOTAL_EVENTS = 15
P_UNIVARIATE = 0.10

CUMULATIVE_CANDIDATES = [
    "log_cum_qliq_m3",
    "log_cum_qw_m3",
    "log_cum_qw_ca_kg",
    "log_cum_qw_so4_kg",
    "log_cum_qw_cl_kg",
]

MEAN_RATE_CANDIDATES = [
    "log_mean_qliq_m3d",
    "log_mean_qw_m3d",
    "log_mean_qw_ca_kgd",
    "log_mean_qw_so4_kgd",
    "log_mean_qw_cl_kgd",
]


def _map_contractor(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Pooled"
    return CONTRACTOR_MAP.get(str(raw).strip(), "oth")


def _derive_h2s_class(field: str, h2s_mg_l: object) -> str:
    if field == "Vt":
        try:
            if float(h2s_mg_l) > H2S_SOUR_THRESHOLD:
                return "sour"
        except (TypeError, ValueError):
            pass
    return "nonsour"


def _load_base() -> pd.DataFrame:
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    try:
        return pd.read_sql(
            """
            SELECT
              row_id, well_key, field, pad, contractor,
              install_date, stop_date, event, run_days, ttf_mix,
              h2s_proxy_mg_l
            FROM mart__weibull_input
            """,
            con,
        )
    finally:
        con.close()


def _load_cumulative_proxies() -> pd.DataFrame:
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    try:
        return pd.read_sql(
            """
            WITH d AS (
              SELECT
                m.row_id,
                dm.dt,
                CASE WHEN dm.qliq >= 0 THEN dm.qliq END AS qliq,
                CASE WHEN dm.watercut BETWEEN 0 AND 100 THEN dm.watercut END AS watercut,
                dl.calcium_mg_l,
                dl.sulfate_mg_l,
                dl.chloride_mg_l
              FROM mart__weibull_input m
              JOIN proc__daily_merged dm
                ON dm.well_key = m.well_key
               AND dm.dt BETWEEN m.install_date AND m.stop_date
              LEFT JOIN proc__daily_lab dl
                ON dl.well_key = dm.well_key
               AND dl.dt = dm.dt
            )
            SELECT
              row_id,
              COUNT(*) AS n_daily_rows,
              SUM(CASE WHEN qliq IS NOT NULL THEN 1 ELSE 0 END) AS n_qliq_days,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL THEN 1 ELSE 0 END) AS n_qw_days,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                        AND calcium_mg_l IS NOT NULL THEN 1 ELSE 0 END) AS n_ca_days,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                        AND sulfate_mg_l IS NOT NULL THEN 1 ELSE 0 END) AS n_so4_days,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                        AND chloride_mg_l IS NOT NULL THEN 1 ELSE 0 END) AS n_cl_days,
              SUM(qliq) AS cum_qliq_m3,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                       THEN qliq * watercut / 100.0 END) AS cum_qw_m3,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND calcium_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * calcium_mg_l / 1000.0 END) AS cum_qw_ca_kg,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND sulfate_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * sulfate_mg_l / 1000.0 END) AS cum_qw_so4_kg,
              SUM(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND chloride_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * chloride_mg_l / 1000.0 END) AS cum_qw_cl_kg
              ,
              AVG(qliq) AS mean_qliq_m3d,
              AVG(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                       THEN qliq * watercut / 100.0 END) AS mean_qw_m3d,
              AVG(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND calcium_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * calcium_mg_l / 1000.0 END) AS mean_qw_ca_kgd,
              AVG(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND sulfate_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * sulfate_mg_l / 1000.0 END) AS mean_qw_so4_kgd,
              AVG(CASE WHEN qliq IS NOT NULL AND watercut IS NOT NULL
                         AND chloride_mg_l IS NOT NULL
                       THEN qliq * watercut / 100.0 * chloride_mg_l / 1000.0 END) AS mean_qw_cl_kgd
            FROM d
            GROUP BY row_id
            """,
            con,
        )
    finally:
        con.close()


def _impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        src = pd.Series("measured", index=df.index)
        src[df[col].isna()] = "none"
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)
        for level, key in (("well", "well_key"), ("pad", "pad"), ("stratum", "stratum_key")):
            missing = df[col].isna()
            if not missing.any():
                break
            fill = df.groupby(key)[col].transform("mean")
            newly = missing & fill.notna()
            df.loc[newly, col] = fill[newly]
            src[newly] = level
        df[f"{col}_imputed_src"] = src
    return df


def build_df() -> pd.DataFrame:
    base = _load_base()
    base["tte"] = pd.to_numeric(base["ttf_mix"], errors="coerce")
    base["event"] = pd.to_numeric(base["event"], errors="coerce").fillna(0).astype(int)
    base["contractor_group"] = base["contractor"].apply(_map_contractor)
    base["h2s_class"] = base.apply(
        lambda r: _derive_h2s_class(r["field"], r["h2s_proxy_mg_l"]), axis=1
    )
    base["stratum_key"] = base["field"] + "_" + base["h2s_class"] + "_" + base["contractor_group"]

    df = base.merge(_load_cumulative_proxies(), on="row_id", how="left")
    raw_cols = [
        "cum_qliq_m3",
        "cum_qw_m3",
        "cum_qw_ca_kg",
        "cum_qw_so4_kg",
        "cum_qw_cl_kg",
        "mean_qliq_m3d",
        "mean_qw_m3d",
        "mean_qw_ca_kgd",
        "mean_qw_so4_kgd",
        "mean_qw_cl_kgd",
    ]
    for col in raw_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    df = _impute(df, raw_cols)
    for col in raw_cols:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "cum_qliq_m3", "cum_qw_m3", "cum_qw_ca_kg", "cum_qw_so4_kg", "cum_qw_cl_kg",
        "mean_qliq_m3d", "mean_qw_m3d", "mean_qw_ca_kgd", "mean_qw_so4_kgd", "mean_qw_cl_kgd",
    ]:
        src = df[f"{col}_imputed_src"].value_counts()
        rows.append(
            {
                "covariate": col,
                "n_total": len(df),
                "n_measured": int(src.get("measured", 0)),
                "n_imputed_well": int(src.get("well", 0)),
                "n_imputed_pad": int(src.get("pad", 0)),
                "n_imputed_stratum": int(src.get("stratum", 0)),
                "n_still_missing": int(src.get("none", 0)),
                "pct_measured": round(100.0 * int(src.get("measured", 0)) / len(df), 1),
            }
        )
    return pd.DataFrame(rows)


def run_univariate(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for col in candidates:
        sub = df[["well_key", "tte", "event", "stratum_key", col]].dropna()
        sub = sub[sub["tte"] > 0].copy()
        counts = sub.groupby("stratum_key")["event"].sum()
        valid = counts[counts >= MIN_STRATUM_EVENTS].index
        sub = sub[sub["stratum_key"].isin(valid)]
        n_events = int(sub["event"].sum())
        if n_events < MIN_TOTAL_EVENTS:
            rows.append({"covariate": col, "n_runs": len(sub), "n_events": n_events, "note": "insufficient events"})
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(
                sub,
                duration_col="tte",
                event_col="event",
                strata=["stratum_key"],
                cluster_col="well_key",
                formula=col,
                robust=True,
            )
            s = cph.summary
            rows.append(
                {
                    "covariate": col,
                    "n_runs": len(sub),
                    "n_events": n_events,
                    "beta": round(float(s.loc[col, "coef"]), 6),
                    "HR": round(float(np.exp(s.loc[col, "coef"])), 4),
                    "HR_lo": round(float(np.exp(s.loc[col, "coef lower 95%"])), 4),
                    "HR_hi": round(float(np.exp(s.loc[col, "coef upper 95%"])), 4),
                    "p": round(float(s.loc[col, "p"]), 6),
                    "c_index": round(float(cph.concordance_index_), 4),
                    "pass_screen": bool(float(s.loc[col, "p"]) < P_UNIVARIATE),
                }
            )
        except Exception as exc:
            rows.append({"covariate": col, "n_runs": len(sub), "n_events": n_events, "note": str(exc)[:120]})
    return pd.DataFrame(rows)


def _run_candidate_set(df: pd.DataFrame, candidates: list[str], stem: str, tbl: Path, fig: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pc._add_logt_terms(df, candidates)
    uni = run_univariate(df, candidates)
    uni.to_csv(tbl / f"{stem}_univariate.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{stem}]")
    print(uni.to_string(index=False))

    survivors = uni.loc[uni.get("pass_screen", False) == True, "covariate"].tolist()
    if not survivors:
        return uni, pd.DataFrame(), pd.DataFrame()

    ph = pc.run_ph_tests(df, survivors)
    ph.to_csv(tbl / f"{stem}_ph_test.csv", index=False, encoding="utf-8-sig")
    std = ph.loc[ph["assignment"] == "STANDARD", "covariate"].tolist()
    ext = ph.loc[ph["assignment"] == "EXTENDED", "covariate"].tolist()
    kept = pc.correlation_screen(df, std + ext, uni)
    std = [c for c in kept if c in std]
    ext = [c for c in kept if c in ext]
    cph = pc.run_joint_model(df, std, ext)
    coeffs = pd.DataFrame()
    vif = pd.DataFrame()
    if cph is not None:
        std2, ext2, vif = pc.vif_check(df, std, ext)
        if set(std2) != set(std) or set(ext2) != set(ext):
            cph2 = pc.run_joint_model(df, std2, ext2)
            if cph2 is not None:
                cph, std, ext = cph2, std2, ext2
        joint = cph.summary.reset_index().rename(columns={"covariate": "term"})
        joint.to_csv(tbl / f"{stem}_joint_model.csv", index=False, encoding="utf-8-sig")
        coeffs = pc.build_final_coeffs(cph, df, std, ext)
        coeffs.to_csv(tbl / f"{stem}_final_coeffs.csv", index=False, encoding="utf-8-sig")
        beta_rows = joint[~joint["term"].str.endswith("_x_logt")]
        pc._forest_plot(beta_rows, f"{stem} Cox - beta", fig / f"{stem}_forest_beta.png")
    vif.to_csv(tbl / f"{stem}_vif.csv", index=False, encoding="utf-8-sig")
    return uni, ph, coeffs


def run() -> dict[str, object]:
    out = results_dir("chem_ion_rate_proxies")
    tbl = out / "tables"
    fig = out / "figures"

    print("[cum-ion] Building cumulative exposure frame...")
    df = build_df()
    print(f"[cum-ion] rows={len(df)} events={int(df['event'].sum())}")

    coverage = coverage_report(df)
    coverage.to_csv(tbl / "coverage.csv", index=False, encoding="utf-8-sig")

    cum_uni, cum_ph, cum_coeffs = _run_candidate_set(df, CUMULATIVE_CANDIDATES, "cumulative", tbl, fig)
    mean_uni, mean_ph, mean_coeffs = _run_candidate_set(df, MEAN_RATE_CANDIDATES, "mean_rate", tbl, fig)

    readme = out / "README.md"
    lines = [
        "# Cumulative Salt / Ion Proxy Cox Screen",
        "",
        "**Clock:** `ttf_mix` operating-time axis.  **Strata:** `field_h2sClass_contractorGroup`, same chemistry block convention.",
        "",
        "Cumulative totals are observed over the run and therefore are not deployable baseline covariates; use this as an exposure/proxy screen.",
        "",
        "## Definitions",
        "",
        "- `cum_qliq_m3 = sum(qliq)`",
        "- `cum_qw_m3 = sum(qliq * watercut / 100)`",
        "- `cum_qw_ca_kg = sum(qliq * watercut / 100 * calcium_mg_l / 1000)`",
        "- `cum_qw_so4_kg = sum(qliq * watercut / 100 * sulfate_mg_l / 1000)`",
        "- `cum_qw_cl_kg = sum(qliq * watercut / 100 * chloride_mg_l / 1000)`",
        "- `mean_qliq_m3d = avg(qliq)`",
        "- `mean_qw_m3d = avg(qliq * watercut / 100)`",
        "- `mean_qw_ca_kgd = avg(qliq * watercut / 100 * calcium_mg_l / 1000)`",
        "- `mean_qw_so4_kgd = avg(qliq * watercut / 100 * sulfate_mg_l / 1000)`",
        "- `mean_qw_cl_kgd = avg(qliq * watercut / 100 * chloride_mg_l / 1000)`",
        "",
        "All Cox candidates use `log1p(...)` transforms.",
        "",
        "## Outputs",
        "",
        "- `tables/coverage.csv`",
        "- `tables/cumulative_*.csv`",
        "- `tables/mean_rate_*.csv`",
        "- `figures/cumulative_forest_beta.png`",
        "- `figures/mean_rate_forest_beta.png`",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cum-ion] Outputs written to: {out}")
    return {
        "df": df,
        "coverage": coverage,
        "cumulative_univariate": cum_uni,
        "cumulative_ph": cum_ph,
        "cumulative_coeffs": cum_coeffs,
        "mean_rate_univariate": mean_uni,
        "mean_rate_ph": mean_ph,
        "mean_rate_coeffs": mean_coeffs,
        "out_dir": out,
    }


if __name__ == "__main__":
    run()
