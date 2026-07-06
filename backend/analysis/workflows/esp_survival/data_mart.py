"""Data loading for ESP survival analysis using mart__weibull_input.

Reads from the SQLite warehouse mart instead of the Excel workbook.
Uses ``ttf_mix`` as the time-to-event column:
    ttf_mix = ttf_true_best_days  (if telemetry/techregime data available)
              run_days             (calendar fallback when source="missing")

H2S class is derived from h2s_proxy_mg_l: Vt field wells with proxy > 50 mg/L
are classified as "sour"; all others as "nonsour".  This is consistent with the
Excel-based classification where only Vt field has confirmed sour wells.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import WAREHOUSE_DIR

WAREHOUSE_DB = WAREHOUSE_DIR / "pump2.db"

_CTR_MAP: dict[str, str] = {
    "Борец": "brt",    # Борец
    "Шлюмберже": "slb",  # Шлюмберже
}

MIN_FAILURES_INDEPENDENT = 40
MIN_FAILURES_TWO_STAGE = 20
SMALL_FIELD_FAILURE_THRESHOLD = 10
H2S_SOUR_THRESHOLD_MG_L = 50.0


def _contractor_group(name: object) -> str:
    s = str(name).strip() if pd.notna(name) else ""
    return _CTR_MAP.get(s, "oth")


def _h2s_class(field: str, h2s_proxy: float | None) -> str:
    if field == "Vt" and h2s_proxy is not None and not np.isnan(h2s_proxy) and h2s_proxy > H2S_SOUR_THRESHOLD_MG_L:
        return "sour"
    return "nonsour"


def load_mart_df(tte_col: str = "ttf_mix") -> pd.DataFrame:
    """Load mart__weibull_input and return a DataFrame ready for Phase 1 / Phase 2.

    Parameters
    ----------
    tte_col : str
        Column to use as time-to-event. Options:
          "ttf_mix"   — true operating days (capped at run_days); default
          "run_days"  — calendar days (for apples-to-apples comparison)

    Columns returned:
        tte           — values from tte_col; rows with tte <= 0 or NaN are dropped
        event         — 1=failure, 0=censored
        field         — raw field name from mart
        field_clean   — field with small fields pooled to "Other"
        h2s_class     — "sour" or "nonsour" (derived from h2s_proxy_mg_l + field)
        stratum       — "{field_clean}_{h2s_class}"
        contractor    — raw Russian name from mart
        contractor_group — "brt", "slb", "oth"
        well_key      — well identifier
        pad           — pad identifier
        run_days      — calendar TTF (for comparison)
        ttf_true_best_days — actual operating days (NaN when missing)
        ttf_true_source    — source label
        h2s_proxy_mg_l     — raw H2S proxy value
    """
    conn = sqlite3.connect(WAREHOUSE_DB)
    df = pd.read_sql_query(
        """
        SELECT
            well_key,
            field,
            pad,
            contractor,
            event,
            run_days,
            ttf_true_best_days,
            ttf_true_source,
            h2s_proxy_mg_l,
            ttf_mix
        FROM mart__weibull_input
        """,
        conn,
    )
    conn.close()

    # ── TTE from selected column ───────────────────────────────────────────────
    if tte_col not in df.columns:
        raise ValueError(f"tte_col={tte_col!r} not in mart columns: {list(df.columns)}")
    df["tte"] = pd.to_numeric(df[tte_col], errors="coerce")
    df = df[df["tte"].notna() & (df["tte"] > 0)].copy()

    # ── H2S class ─────────────────────────────────────────────────────────────
    h2s_num = pd.to_numeric(df["h2s_proxy_mg_l"], errors="coerce")
    df["h2s_class"] = [
        _h2s_class(f, h) for f, h in zip(df["field"], h2s_num)
    ]

    # ── Small-field pooling ───────────────────────────────────────────────────
    all_fields = set(df["field"].unique())
    field_failures = df[df["event"] == 1].groupby("field")["event"].count()
    # Fields with 0 failures are absent from field_failures index → also pool them
    small_fields = (
        set(field_failures[field_failures < SMALL_FIELD_FAILURE_THRESHOLD].index)
        | (all_fields - set(field_failures.index))
    )
    df["field_clean"] = df["field"].apply(
        lambda f: "Other" if f in small_fields else str(f)
    )

    # ── Stratum key ───────────────────────────────────────────────────────────
    df["stratum"] = df["field_clean"] + "_" + df["h2s_class"]

    # ── Contractor group ──────────────────────────────────────────────────────
    df["contractor_group"] = df["contractor"].map(_contractor_group)

    df["event"] = df["event"].astype(int)
    df["run_days"] = pd.to_numeric(df["run_days"], errors="coerce")
    df["ttf_true_best_days"] = pd.to_numeric(df["ttf_true_best_days"], errors="coerce")

    return df.reset_index(drop=True)


def stratum_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-stratum size summary with fit_mode classification."""
    rows = []
    for stratum, g in df.groupby("stratum"):
        n_total = len(g)
        n_failures = int(g["event"].sum())
        b50 = g.loc[g["event"] == 1, "tte"].median()
        if n_failures >= MIN_FAILURES_INDEPENDENT:
            fit_mode = "independent_K2"
        elif n_failures >= MIN_FAILURES_TWO_STAGE:
            fit_mode = "two_stage_K2"
        else:
            fit_mode = "single_weibull_only"
        rows.append({
            "stratum": stratum,
            "n_total": n_total,
            "n_failures": n_failures,
            "n_censored": n_total - n_failures,
            "empirical_B50_days": round(b50, 1) if np.isfinite(b50) else None,
            "fit_mode": fit_mode,
        })
    return pd.DataFrame(rows).sort_values("n_failures", ascending=False).reset_index(drop=True)


__all__ = [
    "load_mart_df",
    "stratum_summary",
    "MIN_FAILURES_INDEPENDENT",
    "MIN_FAILURES_TWO_STAGE",
    "H2S_SOUR_THRESHOLD_MG_L",
]
