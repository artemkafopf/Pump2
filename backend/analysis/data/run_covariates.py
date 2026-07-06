"""Run-level covariate registry builder for the Cox extensions (Phase A T7).

Produces **one row per run** (2,634 rows) carrying every candidate covariate for
the three planned Cox blocks, plus ``*_missing`` indicators, log transforms and the
``stratum_key`` used across the survival stack.  **No model fitting happens here** —
this is the data-wrangling layer that Phase C (Blocks 2+3 fitting) starts from.

Registry blocks (see ``agents/analyses/phase_a_cox_foundation.md`` Part 2):

* Block 1 — environment / fluid (ion chemistry, H2S, GLF, watercut, КВЧ)
* Block 2 — operational (frequency, load, kpod, pzab/pbubble, restarts, idle)
* Block 3 — completion / design (stages, motor, pump gabarit/series, curvature …)
* Block 4 — cohort / history (install vintage, run sequence, gap since prev failure)

Exposure windows (recorded in the registry):
  ``t0``        — known at install (design, completion, pre-install chemistry)
  ``early``     — first 30 **operating** days (qliq>0); baseline-Cox-safe approximation of t0
  ``whole_run`` — mean over [install, stop]; diagnostics / time-varying only

The builder is deliberately explicit about imputation provenance: every imputed
column carries a ``<col>_imputed_src`` label (measured / well / pad / stratum / none)
so the coverage audit reports *measured* coverage, not post-imputation coverage.
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

from analysis.paths import WAREHOUSE_DIR
from analysis.data.pump_type_parser import parse_pump_type_series
from analysis.data.chemistry_run_features import CONTRACTOR_MAP, H2S_SOUR_THRESHOLD

WAREHOUSE_DB = WAREHOUSE_DIR / "pump2.db"

# First N operating days define the ``early`` window.
EARLY_OP_DAYS = 30
# Extra calendar days pulled to compute sequence features (freq steps, restarts).
SEQ_WINDOW_CALENDAR_DAYS = 60

# Data-validity guards (raw daily telemetry carries out-of-range values).
_FREQ_RANGE = (20.0, 75.0)
_LOAD_RANGE = (0.0, 150.0)
_WC_RANGE = (0.0, 100.0)


# ---------------------------------------------------------------------------
# Stratum classification (mirrors chemistry_run_features / VBA ContractorGroup)
# ---------------------------------------------------------------------------

def _map_contractor(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Pooled"
    return CONTRACTOR_MAP.get(str(raw).strip(), "oth")


def _derive_h2s_class(field: str, h2s_mg_l: object) -> str:
    """sour/nonsour — Vt-only active behaviour, consistent with baseline strata."""
    if field == "Vt":
        try:
            if float(h2s_mg_l) > H2S_SOUR_THRESHOLD:
                return "sour"
        except (TypeError, ValueError):
            pass
    return "nonsour"


# ---------------------------------------------------------------------------
# SQL loaders
# ---------------------------------------------------------------------------

def _load_runs(con: sqlite3.Connection) -> pd.DataFrame:
    """Base run table: mart survival fields + raw completion/design fields.

    Joined on ``row_id`` (verified aligned across mart / raw / feat tables).
    """
    return pd.read_sql(
        """
        SELECT
            m.row_id, m.well_key, m.field, m.pad, m.contractor,
            m.install_date, m.stop_date, m.event, m.run_days, m.ttf_mix,
            m.ttf_true_best_days, m.ttf_true_source,
            m.h2s_proxy_mg_l, m.h2s_proxy_source,
            m.nominal_flow_m3d, m.nominal_freq_hz, m.pbubble_atm,
            r.pump_type, r.nominal_head_50hz_m, r.motor_power_kw, r.stages,
            r.submergence_depth_m, r.curvature_flag, r.vg_m, r.nominal_current_a,
            r.failed_node
        FROM mart__weibull_input m
        JOIN raw__v03_runs r ON r.row_id = m.row_id
        """,
        con,
    )


def _load_freq_exposure(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT row_id,
               freq_above_55hz_pct, freq_below_45hz_pct,
               freq_signed_exposure, freq_w_mean,
               n_freq_valid_days, n_freq_op_days
        FROM feat__run_freq_exposure
        """,
        con,
    )


def _load_early_window(con: sqlite3.Connection, n_days: int = EARLY_OP_DAYS) -> pd.DataFrame:
    """Early-window (first ``n_days`` operating days) means/fractions per run.

    Operating day := qliq > 0.  Range guards drop physically impossible telemetry
    before averaging.  Returns one row per run that has ≥1 operating day.
    """
    lo_f, hi_f = _FREQ_RANGE
    lo_l, hi_l = _LOAD_RANGE
    lo_w, hi_w = _WC_RANGE
    q = f"""
    WITH op AS (
      SELECT m.row_id, m.nominal_flow_m3d, m.pbubble_atm,
             dm.dt,
             CASE WHEN dm.freq BETWEEN {lo_f} AND {hi_f} THEN dm.freq END AS freq,
             CASE WHEN dm.load BETWEEN {lo_l} AND {hi_l} THEN dm.load END AS load,
             dm.qliq,
             CASE WHEN dm.watercut BETWEEN {lo_w} AND {hi_w} THEN dm.watercut END AS watercut,
             CASE WHEN dm.gas_factor >= 0 THEN dm.gas_factor END AS gas_factor,
             CASE WHEN dm.rzab > 0 THEN dm.rzab END AS rzab,
             CASE WHEN dm.rpump_intake > 0 THEN dm.rpump_intake END AS rpump_intake,
             ROW_NUMBER() OVER (PARTITION BY m.row_id ORDER BY dm.dt) AS oprank
      FROM mart__weibull_input m
      JOIN proc__daily_merged dm
        ON dm.well_key = m.well_key
       AND dm.dt BETWEEN m.install_date AND m.stop_date
       AND dm.qliq > 0
    )
    SELECT row_id,
      AVG(freq)                                   AS freq_mean,
      AVG(load)                                   AS load_mean,
      AVG(watercut)                               AS watercut_daily,
      AVG(gas_factor)                             AS glf_mean_opdays,
      AVG(CASE WHEN gas_factor > 0 THEN 1.0 ELSE 0.0 END) AS glf_frac_days_gas,
      AVG(qliq / NULLIF(nominal_flow_m3d, 0))     AS kpod_mean,
      AVG(CASE WHEN freq > 0
               THEN qliq / NULLIF(nominal_flow_m3d * freq / 50.0, 0) END) AS kpod_freq_mean,
      AVG(CASE WHEN qliq / NULLIF(nominal_flow_m3d, 0) < 0.7 THEN 1.0 ELSE 0.0 END) AS frac_kpod_below_0p7,
      AVG(rzab / NULLIF(pbubble_atm, 0))          AS pzab_over_pbubble,
      AVG(CASE WHEN rzab / NULLIF(pbubble_atm, 0) < 1.0 THEN 1.0 ELSE 0.0 END) AS frac_pzab_below_1,
      AVG(rpump_intake)                           AS rpump_intake_mean,
      COUNT(*)                                    AS n_early_op_days
    FROM op
    WHERE oprank <= {n_days}
    GROUP BY row_id
    """
    return pd.read_sql(q, con)


def _load_early_chemistry(con: sqlite3.Connection, n_days: int = EARLY_OP_DAYS) -> pd.DataFrame:
    """Early-window ion chemistry means (first ``n_days`` lab-bearing days), with
    a 180-day pre-install fallback merged in for runs with no in-run lab days."""
    q_in = f"""
    WITH lab AS (
      SELECT m.row_id, dl.dt,
             dl.chloride_mg_l, dl.sulfate_mg_l, dl.calcium_mg_l, dl.bicarbonate_mg_l,
             dl.magnesium_mg_l, dl.sodium_potassium_mg_l, dl.total_mineralization_g_l,
             CASE WHEN dl.ph BETWEEN 0 AND 14 THEN dl.ph END AS ph,
             ROW_NUMBER() OVER (PARTITION BY m.row_id ORDER BY dl.dt) AS rk
      FROM mart__weibull_input m
      JOIN proc__daily_lab dl
        ON dl.well_key = m.well_key
       AND dl.dt BETWEEN m.install_date AND m.stop_date
    )
    SELECT row_id,
       AVG(chloride_mg_l) chloride_mg_l, AVG(sulfate_mg_l) sulfate_mg_l,
       AVG(calcium_mg_l) calcium_mg_l, AVG(bicarbonate_mg_l) bicarbonate_mg_l,
       AVG(magnesium_mg_l) magnesium_mg_l, AVG(sodium_potassium_mg_l) sodium_potassium_mg_l,
       AVG(total_mineralization_g_l) total_mineralization_g_l, AVG(ph) ph
    FROM lab WHERE rk <= {n_days} GROUP BY row_id
    """
    df_in = pd.read_sql(q_in, con)

    q_pre = """
    SELECT m.row_id,
       AVG(dl.chloride_mg_l) chloride_mg_l, AVG(dl.sulfate_mg_l) sulfate_mg_l,
       AVG(dl.calcium_mg_l) calcium_mg_l, AVG(dl.bicarbonate_mg_l) bicarbonate_mg_l,
       AVG(dl.magnesium_mg_l) magnesium_mg_l, AVG(dl.sodium_potassium_mg_l) sodium_potassium_mg_l,
       AVG(dl.total_mineralization_g_l) total_mineralization_g_l,
       AVG(CASE WHEN dl.ph BETWEEN 0 AND 14 THEN dl.ph END) ph
    FROM mart__weibull_input m
    JOIN proc__daily_lab dl
      ON dl.well_key = m.well_key
     AND dl.dt BETWEEN date(m.install_date, '-180 days') AND date(m.install_date, '-1 day')
    GROUP BY m.row_id
    """
    df_pre = pd.read_sql(q_pre, con).set_index("row_id")
    df_in = df_in.set_index("row_id")
    chem_cols = [c for c in df_in.columns]
    merged = df_in.reindex(df_in.index.union(df_pre.index))
    for c in chem_cols:
        merged[c] = merged[c].fillna(df_pre.get(c))
    return merged.reset_index()


def _load_kvch(con: sqlite3.Connection) -> pd.DataFrame:
    """КВЧ (mechanical impurities) — in-run then 180d pre-install, from lab.sqlite."""
    from analysis.paths import resolve_lab_db_path
    lab_db = resolve_lab_db_path()
    con.execute(f"ATTACH DATABASE '{lab_db}' AS lab_db")
    in_run = pd.read_sql(
        """
        SELECT m.row_id, AVG(ls.mechanical_impurities_mg_l) mechanical_impurities_mg_l
        FROM mart__weibull_input m
        JOIN lab_db.lab_samples ls
          ON ls.well_key = m.well_key
         AND ls.sample_date BETWEEN m.install_date AND m.stop_date
        GROUP BY m.row_id
        """, con).set_index("row_id")
    pre = pd.read_sql(
        """
        SELECT m.row_id, AVG(ls.mechanical_impurities_mg_l) mechanical_impurities_mg_l
        FROM mart__weibull_input m
        JOIN lab_db.lab_samples ls
          ON ls.well_key = m.well_key
         AND ls.sample_date BETWEEN date(m.install_date,'-180 days') AND date(m.install_date,'-1 day')
        GROUP BY m.row_id
        """, con).set_index("row_id")
    con.execute("DETACH DATABASE lab_db")
    merged = in_run.reindex(in_run.index.union(pre.index))
    merged["mechanical_impurities_mg_l"] = merged["mechanical_impurities_mg_l"].fillna(
        pre["mechanical_impurities_mg_l"])
    return merged.reset_index()


def _load_sequence_features(con: sqlite3.Connection) -> pd.DataFrame:
    """freq_std, n_freq_steps_per_100d, n_restarts_per_100d over the early calendar
    window ([install, install+SEQ_WINDOW_CALENDAR_DAYS])."""
    lo_f, hi_f = _FREQ_RANGE
    df = pd.read_sql(
        f"""
        SELECT m.row_id, dm.dt, dm.qliq,
               CASE WHEN dm.freq BETWEEN {lo_f} AND {hi_f} THEN dm.freq END AS freq
        FROM mart__weibull_input m
        JOIN proc__daily_merged dm
          ON dm.well_key = m.well_key
         AND dm.dt BETWEEN m.install_date
                       AND date(m.install_date, '+{SEQ_WINDOW_CALENDAR_DAYS} days')
        ORDER BY m.row_id, dm.dt
        """,
        con,
    )
    rows = []
    for row_id, g in df.groupby("row_id", sort=False):
        span = max(len(g), 1)
        freq = g["freq"].to_numpy(dtype=float)
        freq_valid = freq[~np.isnan(freq)]
        freq_std = float(np.std(freq_valid)) if len(freq_valid) >= 3 else np.nan
        dfreq = np.abs(np.diff(freq))
        n_steps = int(np.nansum(dfreq > 1.0))
        q = g["qliq"].fillna(0).to_numpy(dtype=float)
        on = (q > 0).astype(int)
        restarts = int(np.sum((np.diff(on) == 1)))
        rows.append({
            "row_id": row_id,
            "freq_std": round(freq_std, 3) if np.isfinite(freq_std) else np.nan,
            "n_freq_steps_per_100d": round(100.0 * n_steps / span, 3),
            "n_restarts_per_100d": round(100.0 * restarts / span, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Imputation with provenance
# ---------------------------------------------------------------------------

def _impute_with_source(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """well → pad → stratum mean imputation; records ``<col>_imputed_src`` per cell
    (measured / well / pad / stratum / none) and a ``<col>_missing`` indicator."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        src = pd.Series("measured", index=df.index)
        src[df[col].isna()] = "none"
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)

        for level, key in (("well", "well_key"), ("pad", "pad"), ("stratum", "stratum_key")):
            miss = df[col].isna()
            if not miss.any():
                break
            fill = df.groupby(key)[col].transform("mean")
            newly = miss & fill.notna()
            df.loc[newly, col] = fill[newly]
            src[newly] = level
        df[f"{col}_imputed_src"] = src
    return df


# Columns subject to imputation (fluid + operational continuous covariates).
_IMPUTE_COLS = [
    # Block 1 fluid
    "chloride_mg_l", "sulfate_mg_l", "calcium_mg_l", "bicarbonate_mg_l",
    "magnesium_mg_l", "sodium_potassium_mg_l", "total_mineralization_g_l", "ph",
    "h2s_proxy_mg_l", "mechanical_impurities_mg_l", "watercut_daily",
    "glf_mean_opdays",
    # Block 2 operational
    "freq_mean", "load_mean", "kpod_mean", "kpod_freq_mean",
    "pzab_over_pbubble", "rpump_intake_mean", "freq_std",
    # Block 3 completion
    "stages", "head_per_stage", "motor_power_kw", "nominal_current_a",
    "nominal_flow_m3d", "curvature", "vg_m", "pbubble_atm", "nominal_freq_hz",
]

_LOG_COLS = [
    "chloride_mg_l", "sulfate_mg_l", "calcium_mg_l", "bicarbonate_mg_l",
    "magnesium_mg_l", "sodium_potassium_mg_l", "total_mineralization_g_l",
    "h2s_proxy_mg_l", "mechanical_impurities_mg_l", "glf_mean_opdays",
    "ca_so4_product", "motor_power_kw", "nominal_current_a", "nominal_flow_m3d",
    "n_restarts_per_100d",
]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_run_covariates(tte_col: str = "ttf_mix") -> pd.DataFrame:
    """Build the run-level covariate registry (one row per run).

    Parameters
    ----------
    tte_col : str
        Duration column exposed as ``tte`` (default ``ttf_mix``).
    """
    con = sqlite3.connect(WAREHOUSE_DB)
    try:
        runs = _load_runs(con)
        freq_exp = _load_freq_exposure(con)
        early = _load_early_window(con)
        chem = _load_early_chemistry(con)
        kvch = _load_kvch(con)
        seq = _load_sequence_features(con)
    finally:
        con.close()

    df = runs.copy()
    if tte_col not in df.columns:
        raise ValueError(f"tte_col={tte_col!r} not in run columns")
    df["tte"] = pd.to_numeric(df[tte_col], errors="coerce")

    # ── Stratum ───────────────────────────────────────────────────────────────
    df["contractor_group"] = df["contractor"].apply(_map_contractor)
    df["h2s_class"] = df.apply(
        lambda r: _derive_h2s_class(r["field"], r["h2s_proxy_mg_l"]), axis=1)
    df["stratum_key"] = df["field"] + "_" + df["h2s_class"] + "_" + df["contractor_group"]

    # ── Merge feature blocks on row_id ────────────────────────────────────────
    for block in (freq_exp, early, chem, kvch, seq):
        df = df.merge(block, on="row_id", how="left")

    # ── Block 3 derived completion features ───────────────────────────────────
    df["curvature"] = pd.to_numeric(df["curvature_flag"], errors="coerce")  # '-' → NaN
    df["head_per_stage"] = df["nominal_head_50hz_m"] / df["stages"].replace(0, np.nan)
    parsed = parse_pump_type_series(df["pump_type"])
    df["pump_series"] = parsed["pump_series"]
    df["pump_series_detail"] = parsed["pump_series_detail"]
    df["pump_gabarit"] = parsed["pump_gabarit"]
    df["od_group_mm"] = parsed["od_group_mm"]
    df["pump_q_design_m3d"] = parsed["q_design_m3d"]

    # ── Block 4 cohort / history ──────────────────────────────────────────────
    df["install_dt"] = pd.to_datetime(df["install_date"], errors="coerce")
    df["stop_dt"] = pd.to_datetime(df["stop_date"], errors="coerce")
    df["install_year"] = df["install_dt"].dt.year
    df = df.sort_values(["well_key", "install_dt"]).reset_index(drop=True)
    df["run_seq"] = df.groupby("well_key").cumcount() + 1
    prev_stop = df.groupby("well_key")["stop_dt"].shift(1)
    df["days_since_prev_failure"] = (df["install_dt"] - prev_stop).dt.days

    # ── idle_frac (whole_run; unavoidable) ────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        df["idle_frac"] = 1.0 - (
            pd.to_numeric(df["ttf_true_best_days"], errors="coerce")
            / pd.to_numeric(df["run_days"], errors="coerce").replace(0, np.nan)
        )
    df["idle_frac"] = df["idle_frac"].clip(lower=0.0, upper=1.0)
    df["pct_mixed_clock"] = (df["ttf_true_source"] == "missing").astype(np.int8)

    # ── Ca×SO4 gypsum index ───────────────────────────────────────────────────
    df["ca_so4_product"] = df["calcium_mg_l"] * df["sulfate_mg_l"]

    # ── Imputation (with provenance) ──────────────────────────────────────────
    df = _impute_with_source(df, _IMPUTE_COLS)

    # ── Log transforms ────────────────────────────────────────────────────────
    for col in _LOG_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# ---------------------------------------------------------------------------
# Coverage audit helpers
# ---------------------------------------------------------------------------

def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-covariate measured vs imputed-source counts (measured coverage, not
    post-imputation coverage)."""
    rows = []
    for col in _IMPUTE_COLS:
        src_col = f"{col}_imputed_src"
        if src_col not in df.columns:
            continue
        vc = df[src_col].value_counts()
        n = len(df)
        n_meas = int(vc.get("measured", 0))
        rows.append({
            "covariate": col,
            "n_total": n,
            "n_measured": n_meas,
            "n_imputed_well": int(vc.get("well", 0)),
            "n_imputed_pad": int(vc.get("pad", 0)),
            "n_imputed_stratum": int(vc.get("stratum", 0)),
            "n_still_missing": int(vc.get("none", 0)),
            "pct_measured": round(100.0 * n_meas / n, 1),
        })
    return pd.DataFrame(rows)


def coverage_by_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Measured coverage per (covariate × stratum_key); flags cells < 50% measured."""
    rows = []
    for col in _IMPUTE_COLS:
        src_col = f"{col}_imputed_src"
        if src_col not in df.columns:
            continue
        for stratum, g in df.groupby("stratum_key"):
            n = len(g)
            n_meas = int((g[src_col] == "measured").sum())
            pct = 100.0 * n_meas / n if n else 0.0
            rows.append({
                "covariate": col,
                "stratum_key": stratum,
                "n_runs": n,
                "n_measured": n_meas,
                "pct_measured": round(pct, 1),
                "low_coverage_flag": pct < 50.0,
            })
    return pd.DataFrame(rows)


__all__ = [
    "build_run_covariates",
    "coverage_report",
    "coverage_by_stratum",
    "EARLY_OP_DAYS",
]
