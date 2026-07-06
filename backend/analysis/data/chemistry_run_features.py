"""Run-level chemistry feature builder for Cox Block 1.

Sources:
  pump2.db / mart__weibull_input  — survival data, H2S proxy, GLF (2,634 runs)
  pump2.db / proc__daily_lab      — daily ion chemistry per well (883,959 rows)
  lab.sqlite / lab_samples        — КВЧ, watercut (8,981 rows)

Output: one row per run. Columns include survival fields, all chemistry
covariates (imputed), log transforms, and:
    stratum_key = {field}_{h2s_class}_{contractor_group}
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import WAREHOUSE_DIR, resolve_lab_db_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches ContractorGroup() in mdlModelRegistry.bas
CONTRACTOR_MAP: dict[str, str] = {
    "Борец": "brt",
    "Шлюмберже": "slb",
    "Новомет": "oth",
    "ИНК": "oth",
    "Новые технологии": "oth",
}

# H2S threshold for sour classification (mg/l).
# Only Vt has sour strata in the baseline Weibull model. Threshold 10 mg/l
# yields ~89 Vt failures classified as sour, matching the VBA model count.
H2S_SOUR_THRESHOLD = 10.0

# Columns that come from proc__daily_lab (in-run mean)
DAILY_LAB_COLS: list[str] = [
    "chloride_mg_l",
    "sulfate_mg_l",
    "calcium_mg_l",
    "bicarbonate_mg_l",
    "total_mineralization_g_l",
    "ph",
]

# Columns that come from lab_samples (in-run mean, with 180-day fallback)
LAB_SAMPLE_COLS: list[str] = [
    "mechanical_impurities_mg_l",
    "watercut_percent",
]

# Columns that get log1p-transformed (right-skewed, clip to 0)
LOG_COLS: list[str] = [
    "chloride_mg_l",
    "sulfate_mg_l",
    "calcium_mg_l",
    "bicarbonate_mg_l",
    "total_mineralization_g_l",
    "mechanical_impurities_mg_l",
    "h2s_proxy_mg_l",
    # glf_mean handled separately after merge
]

# All columns subject to three-level imputation
ALL_IMPUTE_COLS: list[str] = DAILY_LAB_COLS + LAB_SAMPLE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_contractor(raw: object) -> str:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return "Pooled"
    return CONTRACTOR_MAP.get(str(raw).strip(), "oth")


def _derive_h2s_class(field: str, h2s_mg_l: object) -> str:
    """Return 'sour' / 'nonsour' consistent with baseline Weibull model strata."""
    if field == "Vt":
        try:
            if float(h2s_mg_l) > H2S_SOUR_THRESHOLD:
                return "sour"
        except (TypeError, ValueError):
            pass
    return "nonsour"


def _impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Three-level imputation: well → pad → stratum. Adds *_missing indicator."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)
        df[col] = df[col].fillna(df.groupby("well_key")[col].transform("mean"))
        df[col] = df[col].fillna(df.groupby("pad")[col].transform("mean"))
        df[col] = df[col].fillna(df.groupby("stratum_key")[col].transform("mean"))
    return df


# ---------------------------------------------------------------------------
# SQL loaders
# ---------------------------------------------------------------------------

def _load_mart() -> pd.DataFrame:
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    df = pd.read_sql(
        """
        SELECT
            well_key,
            field,
            pad,
            contractor,
            install_date,
            stop_date,
            event,
            run_days,
            ttf_mix,
            ttf_true_source,
            h2s_proxy_mg_l,
            h2s_proxy_source,
            glf_m_mean,
            nominal_flow_m3d
        FROM mart__weibull_input
        """,
        con,
    )
    con.close()
    return df


def _load_glf_from_daily(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Per-run GLF exposure from proc__daily_merged (full run window).

    **Aggregation fix (Phase A T1.3).** The previous version averaged gas_factor over
    days ``WHERE gas_factor > 0`` — a selection bias: a well gassy only 10% of days
    got the mean of its gassy days only, so radically different exposure regimes
    mapped to the same "mean GLF".  Replaced with two orthogonal covariates over
    **all operating days (qliq > 0)**:

      glf_mean_opdays    — mean gas_factor including zero-gas op-days (NULL→0);
      glf_frac_days_gas  — fraction of op-days with gas_factor > 0 (exposure frequency).

    gas_factor = qgas / qliq (pre-computed daily GLF, m³/m³).
    """
    owns_con = con is None
    if owns_con:
        con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    df = pd.read_sql(
        """
        SELECT
            m.well_key,
            m.install_date,
            AVG(COALESCE(dm.gas_factor, 0.0))                    AS glf_mean_opdays,
            AVG(CASE WHEN dm.gas_factor > 0 THEN 1.0 ELSE 0.0 END) AS glf_frac_days_gas,
            COUNT(*)                                            AS glf_n_opdays
        FROM mart__weibull_input m
        JOIN proc__daily_merged dm
          ON dm.well_key = m.well_key
         AND dm.dt BETWEEN m.install_date AND m.stop_date
         AND dm.qliq > 0
        GROUP BY m.well_key, m.install_date
        """,
        con,
    )
    if owns_con:
        con.close()
    return df


def _load_daily_lab_means(window: str = "whole_run", n_early: int = 30) -> pd.DataFrame:
    """Per-run mean of ion chemistry from proc__daily_lab.

    window="whole_run" — mean over [install, stop] (legacy default).
    window="early"      — mean over the first ``n_early`` lab-bearing days in-run
                          (Phase A T1.5: approximates a t0 covariate, escaping the
                          reverse-causation contamination of whole-run averaging).
    """
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    if window == "early":
        df = pd.read_sql(
            f"""
            WITH lab AS (
              SELECT m.well_key, m.install_date, dl.dt,
                     dl.chloride_mg_l, dl.sulfate_mg_l, dl.calcium_mg_l,
                     dl.bicarbonate_mg_l, dl.total_mineralization_g_l,
                     CASE WHEN dl.ph BETWEEN 0 AND 14 THEN dl.ph END AS ph,
                     ROW_NUMBER() OVER (PARTITION BY m.well_key, m.install_date ORDER BY dl.dt) AS rk
              FROM mart__weibull_input m
              JOIN proc__daily_lab dl
                ON dl.well_key = m.well_key
               AND dl.dt BETWEEN m.install_date AND m.stop_date
            )
            SELECT well_key, install_date,
                   AVG(chloride_mg_l) chloride_mg_l, AVG(sulfate_mg_l) sulfate_mg_l,
                   AVG(calcium_mg_l) calcium_mg_l, AVG(bicarbonate_mg_l) bicarbonate_mg_l,
                   AVG(total_mineralization_g_l) total_mineralization_g_l, AVG(ph) ph
            FROM lab WHERE rk <= {int(n_early)}
            GROUP BY well_key, install_date
            """,
            con,
        )
    else:
        df = pd.read_sql(
            """
            SELECT
                m.well_key,
                m.install_date,
                AVG(dl.chloride_mg_l)            AS chloride_mg_l,
                AVG(dl.sulfate_mg_l)             AS sulfate_mg_l,
                AVG(dl.calcium_mg_l)             AS calcium_mg_l,
                AVG(dl.bicarbonate_mg_l)         AS bicarbonate_mg_l,
                AVG(dl.total_mineralization_g_l) AS total_mineralization_g_l,
                AVG(CASE WHEN dl.ph BETWEEN 0 AND 14 THEN dl.ph END) AS ph
            FROM mart__weibull_input m
            JOIN proc__daily_lab dl
              ON dl.well_key = m.well_key
             AND dl.dt BETWEEN m.install_date AND m.stop_date
            GROUP BY m.well_key, m.install_date
            """,
            con,
        )
    con.close()
    return df


def _load_lab_sample_means(lab_db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (in_run, pre_install) DataFrames for КВЧ and watercut.

    in_run: samples whose sample_date falls within [install_date, stop_date].
    pre_install: samples within 180 days before install_date (fallback).
    """
    wh_db = str(WAREHOUSE_DIR / "pump2.db")
    con = sqlite3.connect(wh_db)
    con.execute(f"ATTACH DATABASE '{lab_db}' AS lab_db")

    in_run = pd.read_sql(
        """
        SELECT
            m.well_key,
            m.install_date,
            AVG(ls.mechanical_impurities_mg_l) AS mechanical_impurities_mg_l,
            AVG(ls.watercut_percent)            AS watercut_percent
        FROM mart__weibull_input m
        JOIN lab_db.lab_samples ls
          ON ls.well_key = m.well_key
         AND ls.sample_date BETWEEN m.install_date AND m.stop_date
        GROUP BY m.well_key, m.install_date
        """,
        con,
    )

    pre_install = pd.read_sql(
        """
        SELECT
            m.well_key,
            m.install_date,
            AVG(ls.mechanical_impurities_mg_l) AS mechanical_impurities_mg_l_pre,
            AVG(ls.watercut_percent)            AS watercut_percent_pre
        FROM mart__weibull_input m
        JOIN lab_db.lab_samples ls
          ON ls.well_key = m.well_key
         AND ls.sample_date
               BETWEEN date(m.install_date, '-180 days')
                   AND date(m.install_date, '-1 day')
        GROUP BY m.well_key, m.install_date
        """,
        con,
    )

    con.close()
    return in_run, pre_install


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_chemistry_df(
    tte_col: str = "ttf_mix",
    window: str = "whole_run",
    n_early: int = 30,
) -> pd.DataFrame:
    """Build run-level DataFrame with chemistry covariates.

    Parameters
    ----------
    tte_col : str
        Duration column exposed as ``tte`` (Phase A T1.1). Default ``ttf_mix``
        (operating time, consistent with the baseline Weibulls the coefficients
        multiply in VBA). ``run_days`` remains available for the calendar-clock
        side-by-side comparison.
    window : str
        ``whole_run`` (legacy) or ``early`` (first ``n_early`` lab-bearing in-run
        days — a t0 approximation that escapes reverse-causation contamination, T1.5).
    n_early : int
        Number of in-run lab days defining the early window.

    Returns one row per run (2,634 rows) with survival fields, stratum key,
    imputed chemistry covariates, log transforms, ``{col}_missing`` indicators,
    the fixed GLF pair (``glf_mean_opdays``, ``glf_frac_days_gas``), and a
    ``clock`` / ``window`` stamp.
    """
    lab_db = resolve_lab_db_path()

    # --- Base mart data ---
    df = _load_mart()
    if tte_col not in df.columns:
        raise ValueError(f"tte_col={tte_col!r} not in mart columns: {list(df.columns)}")
    df["tte"] = pd.to_numeric(df[tte_col], errors="coerce")
    df["clock"] = tte_col
    df["window"] = window

    # --- Stratum classification ---
    df["contractor_group"] = df["contractor"].apply(_map_contractor)
    df["h2s_class"] = df.apply(
        lambda r: _derive_h2s_class(r["field"], r["h2s_proxy_mg_l"]), axis=1
    )
    df["stratum_key"] = (
        df["field"] + "_" + df["h2s_class"] + "_" + df["contractor_group"]
    )

    # --- Daily lab means (in-run; whole_run or early window) ---
    daily_means = _load_daily_lab_means(window=window, n_early=n_early)
    df = df.merge(daily_means, on=["well_key", "install_date"], how="left")

    # --- Lab sample КВЧ / watercut: in-run then 180-day pre-install fallback ---
    in_run, pre_install = _load_lab_sample_means(lab_db)
    df = df.merge(in_run, on=["well_key", "install_date"], how="left")
    df = df.merge(pre_install, on=["well_key", "install_date"], how="left")
    for col in LAB_SAMPLE_COLS:
        df[col] = df[col].fillna(df[f"{col}_pre"])
    df = df.drop(columns=[f"{col}_pre" for col in LAB_SAMPLE_COLS], errors="ignore")

    # --- GLF: fixed aggregation over all operating days (T1.3) ---
    # glf_mean_opdays = mean gas_factor incl zero-gas op-days (removes the old
    # WHERE gas_factor>0 selection bias); glf_frac_days_gas = exposure frequency.
    # Fall back to mature-phase glf_m_mean for the ~8% with no daily op-days.
    glf_daily = _load_glf_from_daily()
    df = df.merge(glf_daily, on=["well_key", "install_date"], how="left")
    df["glf_mean_opdays"] = df["glf_mean_opdays"].fillna(df["glf_m_mean"])
    df["glf_frac_days_gas"] = df["glf_frac_days_gas"].fillna(0.0)
    # Back-compat alias used by existing phase_chem CANDIDATES.
    df["glf_mean"] = df["glf_mean_opdays"]
    df = df.drop(columns=["glf_n_opdays", "glf_m_mean"], errors="ignore")

    # --- Imputation: well → pad → stratum ---
    df = _impute(df, ALL_IMPUTE_COLS)

    # --- Log transforms ---
    for col in LOG_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    for col in ("glf_mean", "glf_mean_opdays"):
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # --- Ca × SO₄ gypsum saturation index ---
    df["ca_so4_product"] = df["calcium_mg_l"] * df["sulfate_mg_l"]
    df["log_ca_so4"] = np.log1p(df["ca_so4_product"].clip(lower=0))

    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness summary after imputation."""
    cov_cols = ALL_IMPUTE_COLS + ["h2s_proxy_mg_l", "glf_mean"]
    rows = []
    for col in cov_cols:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        rows.append(
            {
                "covariate": col,
                "n_total": len(df),
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / len(df), 1),
                "n_present": len(df) - n_miss,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "build_chemistry_df",
    "coverage_report",
    "CONTRACTOR_MAP",
    "H2S_SOUR_THRESHOLD",
    "LOG_COLS",
    "ALL_IMPUTE_COLS",
    "DAILY_LAB_COLS",
    "LAB_SAMPLE_COLS",
]
