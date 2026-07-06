"""Data loading for ESP survival analysis.

Loads Отказы свод с анализом_БДА_В03_failures.xlsx, maps Russian column names
to clean English identifiers, applies pad-level imputation for chemistry and
operational covariates, and constructs the analysis DataFrame with derived features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import resolve_v03_failures_path


# ---------------------------------------------------------------------------
# Column mapping: Russian Excel → clean English names
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    "Месторождение": "field",
    "Куст": "pad_key",
    "Скв.": "well_key",
    "Принадлежность": "contractor",
    "Наработка (сут)": "tte",
    "Failure Flag": "failure_flag",
    "Кислый/Некислый": "h2s_class_raw",
    "Признак отказа": "failure_sign",
    "Причина отказа УЭЦН": "failure_cause",
    "Отказавший узел": "failure_node",
    "Отказавший элемент": "failure_element",
    "Характер неисправности": "failure_char",
    # Operational
    "Частота": "frequency",
    "Загр, Двиг,": "motor_load",
    "Обводненность": "water_cut",
    "ГЖФ": "glr",
    "Рзаб": "p_bot",
    "Дав. Нас": "p_bubble",
    "Дебит жидк.": "q_actual",
    "Ном. Произв. м₃/сут": "q_nominal",
    # Chemistry
    "pH": "ph",
    "Механические примеси (КВЧ), мг/дм³": "kvch",
    "Массовая доля сероводорода, мг/дм³": "h2s_conc",
    "Cl⁻, мг/л": "cl",
    "SO₄²⁻, мг/л": "so4",
    "Ca₂⁺, мг/л": "ca",
    "HCO₃⁻, мг/л": "hco3",
    "Общая минерализация, г/л": "mineralization",
    # Equipment and completion
    "Группа исполнения УЭЦН": "execution_group",
    "Коррозионная стойкость": "corrosion_resistance",
    "Габарит УЭЦН": "esp_size",
    "Диаметр НКТ": "tubing_id",
    "Марка НКТ": "tubing_grade",
    "Работа в кривизне": "curvature",
    "Нспуска": "setting_depth",
    "Тип ствола скв": "well_type",
    "Кол.ступеней": "n_stages",
    "Ток x.x": "current_noload",
    "Ном. ток/ A": "current_nominal",
    "2-х этапка": "two_stage_sep",
    # Treatment history
    "ОПЗ": "opz_count",
    "СКО УЭЦН": "sko_esp_count",
}

CHEMISTRY_COLS = ["cl", "so4", "ca", "hco3", "mineralization", "ph", "kvch", "h2s_conc"]
OPERATIONAL_IMPUTE_COLS = ["frequency", "motor_load"]

# Strata with ≥ 40 observed failures get independent K=2 mixture fits.
# Strata with 20–39 failures get two-stage (borrow global shapes).
# Strata below 20 failures are excluded from mixture fitting.
MIN_FAILURES_INDEPENDENT = 40
MIN_FAILURES_TWO_STAGE = 20
# Fields with < 10 failures are pooled into "Other"
SMALL_FIELD_FAILURE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _classify_failure_sign(sign: object) -> int:
    """Map Признак отказа to early/mature flag: 1=early, 0=mature, -1=unknown."""
    if pd.isna(sign):
        return -1
    s = str(sign).strip().lower()
    if s in ("ранний", "преждевременный", "преждевременный "):
        return 1
    if s in ("многосуточный", "многосуточный ", "затянувшийся", "затянувшийся "):
        return 0
    return -1


def _is_h2s_cause(cause: object) -> int:
    """1 if cause text mentions сероводород, else 0."""
    if pd.isna(cause):
        return 0
    return int("сероводород" in str(cause).lower())


# ---------------------------------------------------------------------------
# Pad-level imputation
# ---------------------------------------------------------------------------

def pad_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Three-level imputation: well-mean → pad-mean → stratum-mean.

    Adds a binary ``<col>_imputed`` indicator for each imputed column.
    Chemistry values come from pad-level lab samples applied to all wells
    on the same pad — this is the primary missingness mechanism for Cl⁻, pH, etc.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_imputed"] = df[col].isna().astype(int)

        # Level 1: well-level mean across all runs for the same well
        df[col] = df[col].fillna(df.groupby("well_key")[col].transform("mean"))
        # Level 2: pad-level mean (primary for lab chemistry)
        df[col] = df[col].fillna(df.groupby("pad_key")[col].transform("mean"))
        # Level 3: stratum-level mean fallback
        df[col] = df[col].fillna(df.groupby("stratum")[col].transform("mean"))

    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_failures_df() -> pd.DataFrame:
    """Load, clean, and enrich the ESP failures dataset.

    Returns one row per run with columns:
      tte, event, field, field_clean, h2s_class, stratum, contractor,
      well_key, pad_key, is_early_failure, is_h2s_cause,
      plus all operational / chemistry / equipment covariates.
    """
    path = resolve_v03_failures_path()
    raw = pd.read_excel(path, sheet_name="Свод", header=0)

    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
    df = raw.rename(columns=rename_map)

    # ── Survival columns ────────────────────────────────────────────────────
    df["tte"] = _numeric(df["tte"])
    # Failure Flag: 1 = failure, -1 = censored
    df["event"] = (df["failure_flag"] == 1).astype(int)

    # Drop rows with unusable TTE
    df = df[df["tte"].notna() & (df["tte"] > 0)].copy()

    # ── H2S classification ──────────────────────────────────────────────────
    df["h2s_class"] = (
        df["h2s_class_raw"].astype(str).str.strip()
        .map({"Кислый": "sour", "Некислый": "nonsour"})
        .fillna("nonsour")
    )

    # ── Field normalisation: pool tiny fields into "Other" ──────────────────
    field_failures = (
        df[df["event"] == 1].groupby("field")["event"].count()
    )
    small_fields = set(
        field_failures[field_failures < SMALL_FIELD_FAILURE_THRESHOLD].index
    )
    df["field_clean"] = df["field"].apply(
        lambda f: "Other" if f in small_fields else str(f)
    )

    # ── Stratum key ─────────────────────────────────────────────────────────
    # Sour wells exist only in Vt; all others are non-sour by definition.
    df["stratum"] = (
        df["field_clean"] + "_" + df["h2s_class"]
    )

    # ── Numeric coerce for covariates ────────────────────────────────────────
    for col in CHEMISTRY_COLS + OPERATIONAL_IMPUTE_COLS + [
        "water_cut", "glr", "p_bot", "p_bubble", "q_actual", "q_nominal",
        "curvature", "setting_depth", "n_stages",
        "current_noload", "current_nominal", "two_stage_sep",
        "opz_count", "sko_esp_count",
    ]:
        if col in df.columns:
            df[col] = _numeric(df[col])

    # ── Pad-level imputation ─────────────────────────────────────────────────
    df["pad_key"] = df["pad_key"].astype(str).str.strip()
    df["well_key"] = df["well_key"].astype(str).str.strip()
    df = pad_impute(df, CHEMISTRY_COLS + OPERATIONAL_IMPUTE_COLS)

    # ── Derived features ─────────────────────────────────────────────────────
    # Deviation from Best Efficiency Point
    q_nom = df.get("q_nominal", pd.Series(dtype=float))
    q_act = df.get("q_actual", pd.Series(dtype=float))
    df["delta_bep"] = np.where(
        (_numeric(q_nom) > 0) & _numeric(q_act).notna(),
        _numeric(q_act) / _numeric(q_nom).replace(0.0, np.nan) - 1.0,
        np.nan,
    )

    # Log transforms for right-skewed chemistry
    for col in ["cl", "so4", "ca", "hco3", "mineralization", "kvch", "h2s_conc"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # CaSO₄ saturation index
    if "ca" in df.columns and "so4" in df.columns:
        df["ca_so4_product"] = df["ca"] * df["so4"]
        df["log_ca_so4"] = np.log1p(df["ca_so4_product"].clip(lower=0))

    # Time-interaction terms for Extended Cox (X · log t)
    log_tte = np.log(df["tte"].clip(lower=1e-6))
    for col in ["frequency", "motor_load"]:
        if col in df.columns:
            df[f"{col}_x_logt"] = df[col] * log_tte

    # No-load / nominal current ratio (assembly quality proxy)
    if "current_noload" in df.columns and "current_nominal" in df.columns:
        df["current_ratio"] = df["current_noload"] / df["current_nominal"].replace(0.0, np.nan)

    # Number of stages normalised by nominal flowrate (shaft bending proxy)
    if "n_stages" in df.columns and "q_nominal" in df.columns:
        df["n_stages_ratio"] = df["n_stages"] / df["q_nominal"].replace(0.0, np.nan)

    # ── Failure sign classification ───────────────────────────────────────────
    df["is_early_failure"] = df["failure_sign"].map(_classify_failure_sign)
    df["is_h2s_cause"] = df["failure_cause"].map(_is_h2s_cause)

    # ── Contractor normalisation ─────────────────────────────────────────────
    df["contractor"] = df["contractor"].astype(str).str.strip().fillna("Unknown")

    return df


def stratum_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of stratum sizes for quick inspection."""
    rows = []
    for stratum, g in df.groupby("stratum"):
        n_total = len(g)
        n_failures = int(g["event"].sum())
        n_censored = n_total - n_failures
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
            "n_censored": n_censored,
            "empirical_B50_days": round(b50, 1) if not np.isnan(b50) else None,
            "fit_mode": fit_mode,
        })
    return pd.DataFrame(rows).sort_values("n_failures", ascending=False).reset_index(drop=True)


__all__ = [
    "load_failures_df",
    "stratum_summary",
    "pad_impute",
    "CHEMISTRY_COLS",
    "OPERATIONAL_IMPUTE_COLS",
    "MIN_FAILURES_INDEPENDENT",
    "MIN_FAILURES_TWO_STAGE",
]
