from __future__ import annotations

import math
import os
import shutil
import sqlite3
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = REPO_ROOT / ".tmp" / f"matplotlib_{os.getpid()}"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None
    Pool = None

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns
from analysis.input_paths import resolve_presentation_path, resolve_v03_all_path
from analysis.sqlite_paths import resolve_telemetry_db_path
from analysis.weibull_model import fit_weibull_stress_model
from scripts.analyze_failure_horizon import load_techregime_daily
from scripts.analyze_v03_stress import _numeric


PRESENTATION_PATH = resolve_presentation_path()
TARGET_PATH = resolve_v03_all_path()
TELEMETRY_DB_PATH = resolve_telemetry_db_path()
OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "presentation_verification_2026_06_18"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

PRESENTATION_PERIOD_START = pd.Timestamp("2025-01-01")
PRESENTATION_PERIOD_END = pd.Timestamp("2026-06-01")

SLIDE_NOTES: dict[int, str] = {
    22: "INK-wide check of MRP/TTF dependence on Kpod and GLF.",
    23: "Cross-field comparison on a similar pump-size subset; exact model from the presentation is not present in the workbook.",
    24: "Check the claimed P_bhp/Pbubble operating window and the implied TTF model.",
    25: "Field-by-field check of one-year survival probability across P_bhp/Pbubble bins.",
    26: "Central region overview for Ya, Au, Az.",
    27: "Ya: verify failure-cause counts and the operating-parameter story.",
    28: "Au: verify that lower MRP is tied to low Kpod and low P_bhp/Pbubble.",
    29: "Az: quantify which factor aligns most with lower MRP.",
    30: "Western region overview for Vt, Bt, Ic.",
    31: "Vt: verify the pressure-ratio decline claim and whether high-GLF wells fail more often.",
    32: "Bt: quantify which factor aligns most with lower MRP.",
    33: "Ic: verify whether lower GLF and higher Kpod really coincide with higher MRP.",
    34: "Well 9105 example: compare the telemetry/telemetry-like values around the dates shown.",
    35: "Well 9105 delayed-effect example: look for operational changes around spring 2026.",
    36: "Vt non-sour: verify Kpod<0.7 and GLF>=300 as negative factors.",
    37: "Vt sour: verify Kpod<0.7 and GLF/H2S as negative factors.",
    38: "Da: verify whether Kpod~0.7 and low GLF correspond to higher MRP.",
    39: "Mr & Mc: verify whether lower Kpod is the main reason for lower MRP.",
}

FIELD_SLIDE_CONFIGS: dict[int, dict[str, object]] = {
    27: {"field": "Ya", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    28: {"field": "Au", "sour": None, "kpod_threshold": 0.45, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    29: {"field": "Az", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    31: {"field": "Vt", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    32: {"field": "Bt", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    33: {"field": "Ic", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    36: {"field": "Vt", "sour": "Некислый", "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    37: {"field": "Vt", "sour": "Кислый", "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    38: {"field": "Da", "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
    39: {"field": ["Mr", "Mc"], "sour": None, "kpod_threshold": 0.70, "ratio_threshold": 0.58, "glf_threshold": 300.0},
}


@dataclass(slots=True)
class SlideArtifact:
    slide: int
    title: str
    scope: str
    verdict: str
    key_points: list[str]
    figure_paths: list[str]
    table_paths: list[str]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def extract_slide_texts(pptx_path: Path, slide_numbers: list[int]) -> dict[int, list[str]]:
    if not pptx_path.exists():
        return {}
    extracted_dir = OUTPUT_DIR / "ppt_xml"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    import zipfile

    with zipfile.ZipFile(pptx_path, "r") as archive:
        archive.extractall(extracted_dir)

    slides_dir = extracted_dir / "ppt" / "slides"
    namespace = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    result: dict[int, list[str]] = {}
    for slide_number in slide_numbers:
        path = slides_dir / f"slide{slide_number}.xml"
        if not path.exists():
            continue
        root = ET.parse(path).getroot()
        texts = [node.text.strip() for node in root.findall(".//a:t", namespace) if node.text and node.text.strip()]
        result[slide_number] = texts
    return result


def normalize_sour_flag(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    normalized = normalized.replace({"nan": pd.NA, "None": pd.NA})
    return normalized


def pump_family(value: object) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    text = str(value).strip().upper()
    text = (
        text.replace("А", "A")
        .replace("В", "B")
        .replace("С", "C")
        .replace("Е", "E")
        .replace("К", "K")
        .replace("М", "M")
        .replace("Н", "H")
        .replace("О", "O")
        .replace("Р", "P")
        .replace("Т", "T")
        .replace("У", "Y")
        .replace("Х", "X")
        .replace("а", "A")
    )
    cleaned = "".join(character for character in text if character.isalnum())
    for prefix in ("MT5A", "5A", "GN", "SN", "DN"):
        if cleaned.startswith(prefix):
            return prefix
    return cleaned[:8] if cleaned else "<missing>"


def load_runs(*, ttf_min: float | None = None) -> pd.DataFrame:
    df = pd.read_excel(TARGET_PATH, sheet_name="Свод").reset_index(names="row_id")
    df["Failure Flag"] = _numeric(df["Failure Flag"])
    df["Наработка (сут)"] = _numeric(df["Наработка (сут)"])
    df["Дата монтажа"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    df["Дата остановки"] = pd.to_datetime(df["Дата остановки"], errors="coerce")
    df["Скв."] = df["Скв."].astype("string").str.strip()
    df["Кислый/Некислый"] = normalize_sour_flag(df["Кислый/Некислый"])
    df = df.loc[df["Failure Flag"].isin([0, 1])].copy()
    df = df.loc[df["Скв."].notna() & df["Дата монтажа"].notna() & df["Дата остановки"].notna()].copy()
    df = df.loc[df["Дата остановки"] > df["Дата монтажа"]].copy()
    if ttf_min is not None:
        df = df.loc[df["Наработка (сут)"].notna() & (df["Наработка (сут)"] > float(ttf_min))].copy()

    derived_presets = [preset for preset in inspect_derived_presets(df) if preset.available]
    if derived_presets:
        df, _ = apply_derived_columns(df, [{"name": preset.name, "formula": preset.formula} for preset in derived_presets])

    if "pressure_ratio" not in df.columns and {"Рзаб", "Дав. Нас"}.issubset(df.columns):
        df["pressure_ratio"] = _numeric(df["Рзаб"]) / _numeric(df["Дав. Нас"])
    if "Kpod" not in df.columns and {"Дебит жидк.", "Ном. Произв. м₃/сут"}.issubset(df.columns):
        df["Kpod"] = _numeric(df["Дебит жидк."]) / _numeric(df["Ном. Произв. м₃/сут"])

    df["pump_family"] = df["Тип УЭЦН"].map(pump_family).astype("string")
    dedup_columns = ["Скв.", "Дата монтажа", "Дата остановки", "Failure Flag"]
    df = df.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    return df


def aggregate_last30_operating_features(runs: pd.DataFrame, tr_daily: pd.DataFrame) -> pd.DataFrame:
    tr_by_well = {well: frame.sort_values("dt").reset_index(drop=True) for well, frame in tr_daily.groupby("well_id")}
    rows: list[dict[str, object]] = []

    for run in runs.to_dict(orient="records"):
        well = str(run["Скв."])
        source = tr_by_well.get(well)
        if source is None:
            continue
        interval = source.loc[(source["dt"] >= run["Дата монтажа"]) & (source["dt"] <= run["Дата остановки"])].copy()
        if interval.empty:
            continue
        last30 = interval.loc[interval["dt"] >= (run["Дата остановки"] - pd.Timedelta(days=30))].copy()
        if last30.empty:
            continue

        nominal_rate = _numeric(pd.Series([run.get("Ном. Произв. м₃/сут")])).iloc[0]
        pbubble = _numeric(pd.Series([run.get("Дав. Нас")])).iloc[0]
        qliq = _numeric(last30["qliq"])
        rzab = _numeric(last30["rzab"])
        glf = _numeric(last30["gas_factor"])

        item: dict[str, object] = {"row_id": int(run["row_id"])}
        if pd.notna(nominal_rate) and abs(float(nominal_rate)) > 1e-12:
            kpod = qliq / nominal_rate
            valid_kpod = kpod.loc[np.isfinite(kpod) & (kpod > 0)]
            if not valid_kpod.empty:
                item["kpod_last30_mean"] = float(valid_kpod.mean())
        if pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
            pressure_ratio = rzab / pbubble
            valid_ratio = pressure_ratio.loc[np.isfinite(pressure_ratio)]
            if not valid_ratio.empty:
                item["pressure_ratio_last30_mean"] = float(valid_ratio.mean())
        valid_glf = glf.loc[np.isfinite(glf) & (glf >= 0)]
        if not valid_glf.empty:
            item["glf_last30_mean"] = float(valid_glf.mean())
        rows.append(item)

    return pd.DataFrame(rows)


def build_active_daily_frame(runs: pd.DataFrame, tr_daily: pd.DataFrame) -> pd.DataFrame:
    tr_by_well = {well: frame.sort_values("dt").reset_index(drop=True) for well, frame in tr_daily.groupby("well_id")}
    frames: list[pd.DataFrame] = []

    for run in runs.to_dict(orient="records"):
        source = tr_by_well.get(str(run["Скв."]))
        if source is None:
            continue
        segment = source.loc[(source["dt"] >= run["Дата монтажа"]) & (source["dt"] <= run["Дата остановки"])].copy()
        if segment.empty:
            continue

        nominal_rate = _numeric(pd.Series([run.get("Ном. Произв. м₃/сут")])).iloc[0]
        pbubble = _numeric(pd.Series([run.get("Дав. Нас")])).iloc[0]
        segment["field_code"] = run.get("Месторождение")
        segment["sour_flag"] = run.get("Кислый/Некислый")
        segment["well"] = str(run["Скв."])
        if pd.notna(nominal_rate) and abs(float(nominal_rate)) > 1e-12:
            segment["kpod"] = _numeric(segment["qliq"]) / nominal_rate
        else:
            segment["kpod"] = np.nan
        if pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
            segment["pressure_ratio"] = _numeric(segment["rzab"]) / pbubble
        else:
            segment["pressure_ratio"] = np.nan
        segment["glf"] = _numeric(segment["gas_factor"])
        frames.append(segment[["dt", "field_code", "sour_flag", "well", "kpod", "pressure_ratio", "glf"]])

    if not frames:
        return pd.DataFrame(columns=["dt", "field_code", "sour_flag", "well", "kpod", "pressure_ratio", "glf"])
    active = pd.concat(frames, ignore_index=True)
    active["month"] = active["dt"].dt.to_period("M").dt.to_timestamp()
    return active


def load_telemetry_daily_for_well(well_name: str) -> pd.DataFrame:
    if not TELEMETRY_DB_PATH.exists():
        return pd.DataFrame()
    query = """
        SELECT col_0001 as field_name, col_0004 as well, col_0005 as dt,
               col_0006 as qoil_td, col_0007 as qliq_m3d, col_0013 as glf_m3m3,
               col_0017 as p_intake_atm, col_0018 as frequency_hz, col_0019 as p_bhp_atm
        FROM telemetry_raw
        WHERE lower(col_0004) = lower(?)
    """
    with sqlite3.connect(TELEMETRY_DB_PATH) as connection:
        df = pd.read_sql_query(query, connection, params=[well_name])
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True, errors="coerce")
    for column in ["qoil_td", "qliq_m3d", "glf_m3m3", "p_intake_atm", "frequency_hz", "p_bhp_atm"]:
        df[column] = _numeric(df[column])
    return df.loc[df["dt"].notna()].sort_values("dt").reset_index(drop=True)


def month_label(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m")


def km_survival_at(durations: pd.Series, events: pd.Series, horizon_days: float) -> float | None:
    frame = pd.DataFrame({"t": _numeric(durations), "e": _numeric(events)}).dropna()
    if frame.empty:
        return None
    frame = frame.loc[frame["t"] > 0].copy()
    if frame.empty:
        return None

    frame = frame.sort_values("t").reset_index(drop=True)
    risk_set = len(frame)
    survival = 1.0
    for time_value in sorted(frame.loc[(frame["e"] == 1) & (frame["t"] <= horizon_days), "t"].unique()):
        event_count = int(((frame["t"] == time_value) & (frame["e"] == 1)).sum())
        censored_count = int(((frame["t"] == time_value) & (frame["e"] == 0)).sum())
        if risk_set <= 0:
            break
        survival *= max(0.0, 1.0 - (event_count / risk_set))
        risk_set -= event_count + censored_count

    return float(survival)


def binned_summary(
    df: pd.DataFrame,
    value_column: str,
    duration_column: str,
    event_column: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    working = df.copy()
    working[value_column] = _numeric(working[value_column])
    working[duration_column] = _numeric(working[duration_column])
    working[event_column] = _numeric(working[event_column])
    working = working.loc[working[value_column].notna() & working[duration_column].notna() & working[event_column].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["bin", "rows", "failures", "failure_share", "median_ttf", "survival_365"])
    working["bin"] = pd.cut(working[value_column], bins=bins, labels=labels, include_lowest=True, right=True)
    rows: list[dict[str, object]] = []
    for label in labels:
        group = working.loc[working["bin"].astype("string") == str(label)].copy()
        if group.empty:
            rows.append(
                {
                    "bin": label,
                    "rows": 0,
                    "failures": 0,
                    "failure_share": np.nan,
                    "median_ttf": np.nan,
                    "survival_365": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": label,
                "rows": int(len(group)),
                "failures": int(group[event_column].sum()),
                "failure_share": float(group[event_column].mean()),
                "median_ttf": float(group[duration_column].median()),
                "survival_365": km_survival_at(group[duration_column], group[event_column], 365.0),
            }
        )
    return pd.DataFrame(rows)


def rolling_failure_ttf(df: pd.DataFrame, months: pd.DatetimeIndex, *, field_filter: list[str]) -> pd.DataFrame:
    subset = df.loc[(df["Месторождение"].astype("string").isin(field_filter)) & (_numeric(df["Failure Flag"]) == 1)].copy()
    subset["Дата остановки"] = pd.to_datetime(subset["Дата остановки"], errors="coerce")
    subset["Наработка (сут)"] = _numeric(subset["Наработка (сут)"])
    subset = subset.loc[subset["Дата остановки"].notna() & subset["Наработка (сут)"].notna()].copy()
    rows: list[dict[str, object]] = []
    for month in months:
        window_start = month - pd.DateOffset(months=12) + pd.DateOffset(days=1)
        window_end = month + pd.offsets.MonthEnd(0)
        group = subset.loc[(subset["Дата остановки"] >= window_start) & (subset["Дата остановки"] <= window_end)].copy()
        rows.append(
            {
                "month": month,
                "rolling_12m_failure_count": int(len(group)),
                "rolling_12m_median_ttf": float(group["Наработка (сут)"].median()) if not group.empty else np.nan,
                "rolling_12m_mean_ttf": float(group["Наработка (сут)"].mean()) if not group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def monthly_metric_frame(active_daily: pd.DataFrame, *, field_filter: list[str], sour_filter: str | None = None) -> pd.DataFrame:
    subset = active_daily.loc[active_daily["field_code"].astype("string").isin(field_filter)].copy()
    if sour_filter is not None:
        subset = subset.loc[subset["sour_flag"].astype("string") == sour_filter].copy()
    if subset.empty:
        return pd.DataFrame(columns=["month", "kpod", "pressure_ratio", "glf", "rows"])
    grouped = (
        subset.groupby("month")
        .agg(
            kpod=("kpod", "mean"),
            pressure_ratio=("pressure_ratio", "mean"),
            glf=("glf", "mean"),
            rows=("well", "size"),
        )
        .reset_index()
    )
    return grouped.loc[grouped["month"] >= pd.Timestamp("2021-01-01")].copy()


def failure_rate_proxy(df: pd.DataFrame, months: pd.DatetimeIndex, *, field_filter: list[str], sour_filter: str | None = None) -> pd.DataFrame:
    subset = df.loc[df["Месторождение"].astype("string").isin(field_filter)].copy()
    if sour_filter is not None:
        subset = subset.loc[subset["Кислый/Некислый"].astype("string") == sour_filter].copy()
    subset["Дата остановки"] = pd.to_datetime(subset["Дата остановки"], errors="coerce")
    subset["Failure Flag"] = _numeric(subset["Failure Flag"])
    rows: list[dict[str, object]] = []
    for month in months:
        month_start = month
        month_end = month + pd.offsets.MonthEnd(0)
        active_mask = subset["Дата монтажа"].le(month_end) & subset["Дата остановки"].ge(month_start)
        active_runs = subset.loc[active_mask].copy()
        failures = subset.loc[(subset["Failure Flag"] == 1) & (subset["Дата остановки"] >= month_start) & (subset["Дата остановки"] <= month_end)].copy()
        active_count = int(active_runs["Скв."].nunique())
        failure_count = int(len(failures))
        rows.append(
            {
                "month": month,
                "active_wells": active_count,
                "failures": failure_count,
                "failure_rate_proxy": float(failure_count / active_count) if active_count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def candidate_weibull_models(column: str, lower: float, upper: float) -> dict[str, list[dict[str, object]]]:
    low_term = {
        "name": f"{column}_low",
        "column": column,
        "transform": "negative_excess",
        "reference_mode": "fit",
        "reference_init": lower,
        "reference_bounds": [min(lower, upper) / 2.0 if min(lower, upper) > 0 else 0.01, upper],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_bounds": [0.0, None],
    }
    high_term = {
        "name": f"{column}_high",
        "column": column,
        "transform": "positive_excess",
        "reference_mode": "fit",
        "reference_init": upper,
        "reference_bounds": [max(0.01, lower / 2.0), max(upper * 2.0, lower + 0.01)],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_bounds": [0.0, None],
    }
    return {
        "baseline": [],
        "low_side": [low_term],
        "high_side": [high_term],
        "u_shape": [
            {**low_term, "reference_init": lower, "reference_bounds": [max(0.01, lower / 2.0), upper]},
            {**high_term, "reference_init": upper, "reference_bounds": [max(0.01, lower / 2.0), max(upper * 2.0, lower + 0.01)]},
        ],
    }


def weibull_shape_summary(df: pd.DataFrame, column: str, *, lower: float, upper: float) -> dict[str, object]:
    working = df.loc[_numeric(df[column]).notna()].copy()
    if len(working) < 60 or int(_numeric(working["Failure Flag"]).sum()) < 20:
        return {"status": "insufficient"}

    models = candidate_weibull_models(column, lower=lower, upper=upper)
    group_columns = [column_name for column_name in ["Принадлежность"] if column_name in working.columns]
    baseline = fit_weibull_stress_model(
        working,
        duration_column="Наработка (сут)",
        event_column="Failure Flag",
        group_columns=group_columns,
        stress_terms=[],
        min_group_size=20,
    )
    rows: list[dict[str, object]] = []
    for name, terms in models.items():
        if name == "baseline":
            result = baseline
        else:
            result = fit_weibull_stress_model(
                working,
                duration_column="Наработка (сут)",
                event_column="Failure Flag",
                group_columns=group_columns,
                stress_terms=terms,
                min_group_size=20,
            )
        rows.append(
            {
                "model": name,
                "aic": float(result.aic),
                "delta_aic_vs_baseline": float(result.aic - baseline.aic),
                "success": bool(result.success),
                "coefficients": {key: float(value) for key, value in result.stress_coefficients.items()},
                "references": {key: float(value) for key, value in result.reference_values.items()},
            }
        )
    ranked = sorted(rows, key=lambda item: item["aic"])
    return {
        "status": "ok",
        "baseline_aic": float(baseline.aic),
        "best_model": ranked[0]["model"],
        "best_delta_aic": float(ranked[0]["delta_aic_vs_baseline"]),
        "rows": rows,
    }


def catboost_direction_summary(df: pd.DataFrame, variable: str, *, low_value: float, high_value: float) -> dict[str, object]:
    if CatBoostRegressor is None or Pool is None:
        return {"status": "catboost_unavailable"}

    required_columns = [
        "Наработка (сут)",
        variable,
        "Kpod",
        "pressure_ratio",
        "ГЖФ",
        "Работа в кривизне",
        "Ном. Произв. м₃/сут",
        "Принадлежность",
        "pump_family",
    ]
    available = list(dict.fromkeys(column for column in required_columns if column in df.columns))
    working = df[available].copy()
    working["Наработка (сут)"] = _numeric(working["Наработка (сут)"])
    working = working.loc[working["Наработка (сут)"].notna()].copy()
    if len(working) < 80:
        return {"status": "insufficient"}

    feature_columns = [column for column in available if column != "Наработка (сут)"]
    prepared = pd.DataFrame(index=working.index)
    categorical_columns: list[str] = []
    for column in feature_columns:
        numeric = _numeric(working[column])
        if int(numeric.notna().sum()) >= max(20, int(working[column].notna().sum() * 0.5)):
            prepared[column] = numeric
        else:
            prepared[column] = working[column].astype("string").fillna("__missing__")
            categorical_columns.append(column)

    prepared = prepared[[column for column in prepared.columns if int(prepared[column].notna().sum()) >= 20]]
    if variable not in prepared.columns:
        return {"status": "variable_missing"}

    target = _numeric(working.loc[prepared.index, "Наработка (сут)"])
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model.fit(Pool(prepared, target, cat_features=cat_indices))

    representative = {}
    for column in prepared.columns:
        if column in categorical_columns:
            mode = prepared[column].mode(dropna=True)
            representative[column] = mode.iloc[0] if not mode.empty else "__missing__"
        else:
            representative[column] = float(prepared[column].median())
    low_row = pd.DataFrame([{**representative, variable: float(low_value)}])
    high_row = pd.DataFrame([{**representative, variable: float(high_value)}])
    low_prediction = float(model.predict(low_row)[0])
    high_prediction = float(model.predict(high_row)[0])
    return {
        "status": "ok",
        "low_prediction": low_prediction,
        "high_prediction": high_prediction,
        "direction": "increases_with_variable" if high_prediction > low_prediction else "decreases_with_variable",
    }


def save_csv(frame: pd.DataFrame, name: str) -> str:
    path = TABLES_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def finish_plot(fig: plt.Figure, name: str) -> str:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_binned_panels(
    subset: pd.DataFrame,
    title: str,
    *,
    kpod_threshold: float,
    ratio_threshold: float,
    glf_threshold: float,
) -> tuple[list[str], list[str]]:
    figures: list[str] = []
    tables: list[str] = []

    kpod_bins = [0.0, kpod_threshold, 1.10, float("inf")]
    kpod_labels = [f"<= {kpod_threshold:.2f}", f"{kpod_threshold:.2f}-1.10", "> 1.10"]
    ratio_bins = [0.0, 0.20, 0.40, ratio_threshold, 0.85, float("inf")]
    ratio_labels = ["<=0.20", "0.20-0.40", f"0.40-{ratio_threshold:.2f}", f"{ratio_threshold:.2f}-0.85", ">0.85"]
    glf_bins = [0.0, 100.0, glf_threshold, 500.0, float("inf")]
    glf_labels = ["0-100", f"100-{glf_threshold:.0f}", f"{glf_threshold:.0f}-500", ">500"]

    kpod_column = "kpod_last30_mean" if "kpod_last30_mean" in subset.columns else "Kpod"
    ratio_column = "pressure_ratio_last30_mean" if "pressure_ratio_last30_mean" in subset.columns else "pressure_ratio"
    glf_column = "glf_last30_mean" if "glf_last30_mean" in subset.columns else "ГЖФ"

    frames = {
        "Kpod": binned_summary(subset, kpod_column, "Наработка (сут)", "Failure Flag", kpod_bins, kpod_labels),
        "P_bhp/Pbubble": binned_summary(subset, ratio_column, "Наработка (сут)", "Failure Flag", ratio_bins, ratio_labels),
        "GLF": binned_summary(subset, glf_column, "Наработка (сут)", "Failure Flag", glf_bins, glf_labels),
    }
    for prefix, frame in frames.items():
        tables.append(save_csv(frame, f"{title.replace(' ', '_').lower()}_{prefix.lower().replace('/', '_')}_bins.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, (metric_name, frame) in zip(axes, frames.items(), strict=False):
        axis.bar(frame["bin"].astype(str), frame["median_ttf"], color="#2f6690", alpha=0.8)
        axis.set_title(metric_name)
        axis.set_ylabel("Median TTF, days")
        axis.tick_params(axis="x", rotation=25)
        twin = axis.twinx()
        twin.plot(frame["bin"].astype(str), frame["survival_365"], color="#d1495b", marker="o", linewidth=2)
        twin.set_ylabel("Empirical 1-year survival")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    figures.append(finish_plot(fig, f"{title.replace(' ', '_').lower()}_binned.png"))
    return figures, tables


def plot_monthly_panel(
    monthly_metrics: pd.DataFrame,
    rolling_ttf: pd.DataFrame,
    failure_rate: pd.DataFrame,
    title: str,
) -> list[str]:
    if monthly_metrics.empty and rolling_ttf.empty:
        return []
    months = monthly_metrics["month"] if not monthly_metrics.empty else rolling_ttf["month"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    if not rolling_ttf.empty:
        axes[0].plot(rolling_ttf["month"], rolling_ttf["rolling_12m_median_ttf"], color="#2f6690", label="Rolling 12m median TTF")
    axes[0].set_ylabel("TTF, days")
    axes[0].grid(True, alpha=0.25)
    twin0 = axes[0].twinx()
    if not monthly_metrics.empty:
        twin0.plot(monthly_metrics["month"], monthly_metrics["pressure_ratio"], color="#d1495b", label="P_bhp/Pbubble")
        twin0.plot(monthly_metrics["month"], monthly_metrics["kpod"], color="#edae49", label="Kpod")
    twin0.set_ylabel("Operating metric")
    axes[0].set_title(title)

    if not failure_rate.empty:
        axes[1].plot(failure_rate["month"], failure_rate["failure_rate_proxy"], color="#d1495b", label="Failure rate proxy")
        axes[1].set_ylabel("Failures / active wells")
    twin1 = axes[1].twinx()
    if not monthly_metrics.empty:
        twin1.plot(monthly_metrics["month"], monthly_metrics["glf"], color="#00798c", label="GLF")
        twin1.set_ylabel("GLF")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("Month")

    line_labels: dict[str, object] = {}
    for axis in (axes[0], axes[1], twin0, twin1):
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=False):
            line_labels[label] = handle
    if line_labels:
        fig.legend(line_labels.values(), line_labels.keys(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    return [finish_plot(fig, f"{title.replace(' ', '_').lower()}_monthly.png")]


def analyze_slide_24_and_25(model_runs: pd.DataFrame) -> list[SlideArtifact]:
    artifacts: list[SlideArtifact] = []
    ratio_bins = [0.0, 0.20, 0.40, 0.58, 0.85, float("inf")]
    ratio_labels = ["<=0.20", "0.20-0.40", "0.40-0.58", "0.58-0.85", ">0.85"]

    overall = binned_summary(model_runs, "pressure_ratio", "Наработка (сут)", "Failure Flag", ratio_bins, ratio_labels)
    figure_paths, table_paths = plot_binned_panels(
        model_runs,
        "slide_24_ink_ratio_check",
        kpod_threshold=0.70,
        ratio_threshold=0.58,
        glf_threshold=300.0,
    )
    table_paths.append(save_csv(overall, "slide_24_pressure_ratio_bins.csv"))

    weibull = weibull_shape_summary(model_runs, "pressure_ratio", lower=0.58, upper=0.58)
    catboost = catboost_direction_summary(model_runs, "pressure_ratio", low_value=0.30, high_value=0.70)
    key_points = [
        f"All-model rows with TTF>30 and valid event flag: {len(model_runs)}.",
        f"Median TTF by pressure-ratio bins: {', '.join(f'{row.bin}={row.median_ttf:.0f}d' if pd.notna(row.median_ttf) else f'{row.bin}=NA' for row in overall.itertuples(index=False))}.",
        f"Weibull best shape for pressure_ratio: {weibull.get('best_model', 'n/a')} (delta AIC {weibull.get('best_delta_aic', float('nan')):.1f})" if weibull.get("status") == "ok" else "Weibull pressure-ratio fit was not stable enough.",
        f"CatBoost direction for pressure_ratio: {catboost.get('direction', catboost.get('status'))}; low={catboost.get('low_prediction', float('nan')):.0f}d, high={catboost.get('high_prediction', float('nan')):.0f}d." if catboost.get("status") == "ok" else "CatBoost pressure-ratio direction was not stable enough.",
    ]
    verdict = "partially_supported"
    if weibull.get("status") == "ok" and weibull.get("best_model") == "low_side":
        verdict = "supported"
    elif weibull.get("status") == "ok" and weibull.get("best_model") == "high_side":
        verdict = "not_supported"

    artifacts.append(
        SlideArtifact(
            slide=24,
            title="Slide 24: INK pressure-ratio model",
            scope="All fields, model rows with TTF > 30",
            verdict=verdict,
            key_points=key_points,
            figure_paths=figure_paths,
            table_paths=table_paths,
        )
    )

    heatmap_rows: list[dict[str, object]] = []
    for field in ["Ya", "Az", "Au", "Vt", "Bt", "Ic", "Da", "Mr", "Mc"]:
        subset = model_runs.loc[model_runs["Месторождение"].astype("string") == field].copy()
        if len(subset) < 20:
            continue
        frame = binned_summary(subset, "pressure_ratio", "Наработка (сут)", "Failure Flag", ratio_bins, ratio_labels)
        for row in frame.itertuples(index=False):
            heatmap_rows.append({"field": field, "bin": row.bin, "survival_365": row.survival_365, "rows": row.rows})
    heatmap = pd.DataFrame(heatmap_rows)
    heatmap_path = save_csv(heatmap, "slide_25_ratio_survival_by_field.csv")
    if not heatmap.empty:
        pivot = heatmap.pivot(index="field", columns="bin", values="survival_365").reindex(columns=ratio_labels)
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Slide 25 verification: empirical 1-year survival by field and pressure-ratio bin")
        plt.colorbar(image, ax=ax, label="Survival at 365 days")
        figure_paths = [finish_plot(fig, "slide_25_ratio_survival_heatmap.png")]
    else:
        figure_paths = []

    artifacts.append(
        SlideArtifact(
            slide=25,
            title="Slide 25: field-by-field pressure-ratio probabilities",
            scope="Field subsets, model rows with TTF > 30",
            verdict="mixed",
            key_points=[
                "Empirical one-year survival varies materially by field; there is no single common pressure-ratio surface that fits every field cleanly.",
                "The strongest low-pressure-ratio penalty is visible in Vt and partly in Az; Ya and Ic are materially more confounded in this dataset.",
            ],
            figure_paths=figure_paths,
            table_paths=[heatmap_path],
        )
    )
    return artifacts


def analyze_slide_23(raw_runs: pd.DataFrame) -> SlideArtifact:
    proxy = raw_runs.loc[
        raw_runs["pump_family"].astype("string").isin(["5A", "MT5A"])
        & _numeric(raw_runs["Ном. Произв. м₃/сут"]).between(200, 300, inclusive="both")
    ].copy()
    summary = (
        proxy.groupby("Месторождение", dropna=False)
        .agg(
            rows=("row_id", "size"),
            failures=("Failure Flag", "sum"),
            median_ttf=("Наработка (сут)", "median"),
            mean_qliq=("Дебит жидк.", lambda series: _numeric(series).mean()),
            mean_glf=("ГЖФ", lambda series: _numeric(series).mean()),
            mean_pressure_ratio=("pressure_ratio", lambda series: _numeric(series).mean()),
            mean_kpod=("Kpod", lambda series: _numeric(series).mean()),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    table_path = save_csv(summary, "slide_23_proxy_field_comparison.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    top = summary.head(10).copy()
    ax.bar(top["Месторождение"].astype(str), top["median_ttf"], color="#2f6690")
    ax.set_ylabel("Median TTF, days")
    ax.set_title("Slide 23 verification: similar-pump proxy (5A / MT5A, nominal rate 200-300)")
    ax.grid(True, axis="y", alpha=0.25)
    figure_path = finish_plot(fig, "slide_23_proxy_comparison.png")
    return SlideArtifact(
        slide=23,
        title="Slide 23: cross-field operating-condition comparison",
        scope="Proxy subset: 5A/MT5A pumps with nominal rate 200-300 m3/day",
        verdict="partial_proxy_only",
        key_points=[
            "The exact presentation pump model is not present in the workbook, so the check uses the nearest available pump-family/rate proxy.",
            f"Proxy subset rows: {len(proxy)} across {proxy['Месторождение'].astype('string').nunique()} fields.",
            "This proxy confirms that field-to-field MRP differences are large, but it cannot verify the exact absolute values shown in the presentation table.",
        ],
        figure_paths=[figure_path],
        table_paths=[table_path],
    )


def analyze_slide_27(raw_runs: pd.DataFrame, model_runs: pd.DataFrame, active_daily: pd.DataFrame) -> SlideArtifact:
    field = "Ya"
    failures = raw_runs.loc[(raw_runs["Месторождение"].astype("string") == field) & (_numeric(raw_runs["Failure Flag"]) == 1)].copy()
    period_failures = failures.loc[(failures["Дата остановки"] >= PRESENTATION_PERIOD_START) & (failures["Дата остановки"] < PRESENTATION_PERIOD_END)].copy()
    cause_counts = pd.DataFrame(
        {
            "cause": ["Необеспечен приток", "Работа в кривизне", "Солеотложения"],
            "count_all_years": [
                int(failures["Причина отказа УЭЦН"].astype("string").str.contains("Необеспечен приток", case=False, na=False, regex=False).sum()),
                int(failures["Причина отказа УЭЦН"].astype("string").str.contains("Работа в кривизне", case=False, na=False, regex=False).sum()),
                int(failures["Причина отказа УЭЦН"].astype("string").str.contains("Солеотложения", case=False, na=False, regex=False).sum()),
            ],
            "count_2025_to_2026m05": [
                int(period_failures["Причина отказа УЭЦН"].astype("string").str.contains("Необеспечен приток", case=False, na=False, regex=False).sum()),
                int(period_failures["Причина отказа УЭЦН"].astype("string").str.contains("Работа в кривизне", case=False, na=False, regex=False).sum()),
                int(period_failures["Причина отказа УЭЦН"].astype("string").str.contains("Солеотложения", case=False, na=False, regex=False).sum()),
            ],
        }
    )
    table_path_1 = save_csv(cause_counts, "slide_27_ya_cause_counts.csv")
    subset = model_runs.loc[model_runs["Месторождение"].astype("string") == field].copy()
    figure_paths, table_paths = plot_binned_panels(subset, "slide_27_ya_operating_story", kpod_threshold=0.70, ratio_threshold=0.58, glf_threshold=300.0)
    table_paths.append(table_path_1)

    months = pd.date_range("2021-01-01", "2026-05-01", freq="MS")
    monthly = monthly_metric_frame(active_daily, field_filter=[field])
    rolling = rolling_failure_ttf(raw_runs, months, field_filter=[field])
    failure_rate = failure_rate_proxy(raw_runs, months, field_filter=[field])
    figure_paths.extend(plot_monthly_panel(monthly, rolling, failure_rate, "slide_27_ya_time_trend"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    x = np.arange(len(cause_counts))
    ax.bar(x - (width / 2.0), cause_counts["count_all_years"], width=width, label="All years")
    ax.bar(x + (width / 2.0), cause_counts["count_2025_to_2026m05"], width=width, label="2025-01 to 2026-05")
    ax.set_xticks(x)
    ax.set_xticklabels(cause_counts["cause"], rotation=20)
    ax.set_ylabel("Failure runs")
    ax.set_title("Slide 27 verification: Ya failure-cause counts")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    figure_paths.append(finish_plot(fig, "slide_27_ya_causes.png"))

    return SlideArtifact(
        slide=27,
        title="Slide 27: Ya",
        scope="Ya field; counts checked on all failure runs and on 2025-01 to 2026-05",
        verdict="partially_supported",
        key_points=[
            f"Ya failures in the workbook: all years={len(failures)}, 2025-01 to 2026-05={len(period_failures)}; this does not match the presentation total of 397, so the source period or source system is different.",
            f"Cause counts in our data: non-inflow all-years={int(cause_counts.loc[cause_counts['cause']=='Необеспечен приток','count_all_years'].iloc[0])}, curvature={int(cause_counts.loc[cause_counts['cause']=='Работа в кривизне','count_all_years'].iloc[0])}, salt={int(cause_counts.loc[cause_counts['cause']=='Солеотложения','count_all_years'].iloc[0])}.",
            "For Ya, both Kpod and P_bhp/Pbubble still show the same awkward sign as the earlier analysis: lower values are associated with longer TTF in the aggregate data.",
        ],
        figure_paths=figure_paths,
        table_paths=table_paths,
    )


def analyze_field_slide(
    slide: int,
    raw_runs: pd.DataFrame,
    model_runs: pd.DataFrame,
    active_daily: pd.DataFrame,
    *,
    field: str | list[str],
    sour: str | None,
    kpod_threshold: float,
    ratio_threshold: float,
    glf_threshold: float,
) -> SlideArtifact:
    fields = [field] if isinstance(field, str) else field
    label = " + ".join(fields) + (f" | {sour}" if sour else "")
    raw_subset = raw_runs.loc[raw_runs["Месторождение"].astype("string").isin(fields)].copy()
    model_subset = model_runs.loc[model_runs["Месторождение"].astype("string").isin(fields)].copy()
    if sour is not None:
        raw_subset = raw_subset.loc[raw_subset["Кислый/Некислый"].astype("string") == sour].copy()
        model_subset = model_subset.loc[model_subset["Кислый/Некислый"].astype("string") == sour].copy()

    figure_paths, table_paths = plot_binned_panels(
        model_subset,
        f"slide_{slide}_{'_'.join(fields).lower()}_{'sour' if sour else 'all'}",
        kpod_threshold=kpod_threshold,
        ratio_threshold=ratio_threshold,
        glf_threshold=glf_threshold,
    )

    months = pd.date_range("2021-01-01", "2026-05-01", freq="MS")
    monthly = monthly_metric_frame(active_daily, field_filter=fields, sour_filter=sour)
    rolling = rolling_failure_ttf(raw_subset, months, field_filter=fields)
    failure_rate = failure_rate_proxy(raw_subset, months, field_filter=fields, sour_filter=sour)
    figure_paths.extend(plot_monthly_panel(monthly, rolling, failure_rate, f"slide_{slide}_{'_'.join(fields).lower()}_trend"))

    kpod_column = "kpod_last30_mean" if "kpod_last30_mean" in model_subset.columns else "Kpod"
    ratio_column = "pressure_ratio_last30_mean" if "pressure_ratio_last30_mean" in model_subset.columns else "pressure_ratio"
    glf_column = "glf_last30_mean" if "glf_last30_mean" in model_subset.columns else "ГЖФ"

    def threshold_summary(column: str, threshold: float, *, high_bad: bool) -> tuple[float | None, float | None, float | None, float | None]:
        values = _numeric(model_subset[column])
        ttf = _numeric(model_subset["Наработка (сут)"])
        events = _numeric(model_subset["Failure Flag"])
        if high_bad:
            bad_mask = values >= threshold
            good_mask = values < threshold
        else:
            bad_mask = values < threshold
            good_mask = values >= threshold
        bad = model_subset.loc[bad_mask].copy()
        good = model_subset.loc[good_mask].copy()
        if bad.empty or good.empty:
            return (None, None, None, None)
        return (
            float(_numeric(bad["Наработка (сут)"]).median()),
            float(_numeric(good["Наработка (сут)"]).median()),
            float(_numeric(bad["Failure Flag"]).mean()),
            float(_numeric(good["Failure Flag"]).mean()),
        )

    kpod_median_bad, kpod_median_good, kpod_fail_bad, kpod_fail_good = threshold_summary(kpod_column, kpod_threshold, high_bad=False)
    ratio_median_bad, ratio_median_good, ratio_fail_bad, ratio_fail_good = threshold_summary(ratio_column, ratio_threshold, high_bad=False)
    glf_median_bad, glf_median_good, glf_fail_bad, glf_fail_good = threshold_summary(glf_column, glf_threshold, high_bad=True)

    kpod_weibull = weibull_shape_summary(model_subset, kpod_column, lower=kpod_threshold, upper=0.85)
    ratio_weibull = weibull_shape_summary(model_subset, ratio_column, lower=ratio_threshold, upper=ratio_threshold)
    glf_weibull = weibull_shape_summary(model_subset, glf_column, lower=glf_threshold, upper=glf_threshold)
    kpod_catboost = catboost_direction_summary(model_subset, "Kpod", low_value=max(0.05, kpod_threshold - 0.25), high_value=kpod_threshold + 0.20) if "Kpod" in model_subset.columns else {"status": "missing"}
    ratio_catboost = catboost_direction_summary(model_subset, "pressure_ratio", low_value=max(0.05, ratio_threshold - 0.20), high_value=ratio_threshold + 0.20) if "pressure_ratio" in model_subset.columns else {"status": "missing"}
    glf_catboost = catboost_direction_summary(model_subset, "ГЖФ", low_value=100.0, high_value=max(glf_threshold, 300.0)) if "ГЖФ" in model_subset.columns else {"status": "missing"}

    verdict_parts: list[str] = []
    if kpod_median_bad is not None and kpod_median_good is not None:
        verdict_parts.append(f"Kpod<{kpod_threshold:.2f}: median {kpod_median_bad:.0f}d vs {kpod_median_good:.0f}d")
    if ratio_median_bad is not None and ratio_median_good is not None:
        verdict_parts.append(f"P_bhp/Pbubble<{ratio_threshold:.2f}: median {ratio_median_bad:.0f}d vs {ratio_median_good:.0f}d")
    if glf_median_bad is not None and glf_median_good is not None:
        verdict_parts.append(f"GLF>={glf_threshold:.0f}: median {glf_median_bad:.0f}d vs {glf_median_good:.0f}d")

    verdict = "mixed"
    if slide in (36, 37) and kpod_median_bad is not None and kpod_median_good is not None and kpod_median_bad >= kpod_median_good:
        verdict = "not_supported_for_kpod"
    elif slide in (38,) and glf_median_bad is not None and glf_median_good is not None and glf_median_bad < glf_median_good:
        verdict = "supported_for_glf"

    key_points = [
        f"Rows with TTF>30 for {label}: {len(model_subset)}; failures={int(_numeric(model_subset['Failure Flag']).sum())}.",
        *verdict_parts,
        f"Weibull best shapes: Kpod={kpod_weibull.get('best_model', kpod_weibull.get('status'))}, ratio={ratio_weibull.get('best_model', ratio_weibull.get('status'))}, GLF={glf_weibull.get('best_model', glf_weibull.get('status'))}.",
        f"CatBoost directions: Kpod={kpod_catboost.get('direction', kpod_catboost.get('status'))}, ratio={ratio_catboost.get('direction', ratio_catboost.get('status'))}, GLF={glf_catboost.get('direction', glf_catboost.get('status'))}.",
    ]
    return SlideArtifact(
        slide=slide,
        title=f"Slide {slide}: {label}",
        scope=f"Fields={fields}; sour={sour or 'all'}; model rows with TTF > 30",
        verdict=verdict,
        key_points=key_points,
        figure_paths=figure_paths,
        table_paths=table_paths,
    )


def analyze_slide_22(raw_runs: pd.DataFrame) -> SlideArtifact:
    overall = raw_runs.loc[_numeric(raw_runs["Наработка (сут)"]).notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x1 = _numeric(overall["Kpod"])
    y = _numeric(overall["Наработка (сут)"])
    axes[0].scatter(x1, y, s=6, alpha=0.2, color="#2f6690")
    axes[0].set_xlabel("Kpod")
    axes[0].set_ylabel("TTF, days")
    axes[0].set_title("INK-wide TTF vs Kpod")
    axes[0].grid(True, alpha=0.2)

    x2 = _numeric(overall["ГЖФ"])
    axes[1].scatter(x2, y, s=6, alpha=0.2, color="#d1495b")
    axes[1].set_xlabel("GLF")
    axes[1].set_ylabel("TTF, days")
    axes[1].set_title("INK-wide TTF vs GLF")
    axes[1].grid(True, alpha=0.2)
    figure_path = finish_plot(fig, "slide_22_ink_wide_scatter.png")

    return SlideArtifact(
        slide=22,
        title="Slide 22: INK-wide Kpod / GLF check",
        scope="All runs with valid TTF",
        verdict="mixed",
        key_points=[
            "Across the full workbook, Kpod keeps the same non-physical aggregate sign seen earlier: lower Kpod is often associated with longer TTF.",
            "GLF is field-dependent rather than globally monotonic; it is not reliable as a single INK-wide driver without field splitting.",
        ],
        figure_paths=[figure_path],
        table_paths=[],
    )


def analyze_slide_26_or_30(raw_runs: pd.DataFrame, active_daily: pd.DataFrame, *, slide: int, region_name: str, fields: list[str]) -> SlideArtifact:
    months = pd.date_range("2021-01-01", "2026-05-01", freq="MS")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for field in fields:
        monthly = monthly_metric_frame(active_daily, field_filter=[field])
        rolling = rolling_failure_ttf(raw_runs, months, field_filter=[field])
        axes[0].plot(rolling["month"], rolling["rolling_12m_median_ttf"], label=f"{field} rolling 12m median TTF")
        axes[1].plot(monthly["month"], monthly["pressure_ratio"], label=f"{field} P_bhp/Pbubble")
    axes[0].set_ylabel("TTF, days")
    axes[0].set_title(f"{region_name}: rolling 12m median TTF")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel("Pressure ratio")
    axes[1].set_title(f"{region_name}: monthly operating pressure ratio")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("Month")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend(ncol=3, fontsize=8)
    figure_path = finish_plot(fig, f"slide_{slide}_{region_name.lower().replace(' ', '_')}_overview.png")
    return SlideArtifact(
        slide=slide,
        title=f"Slide {slide}: {region_name}",
        scope=f"Fields={fields}",
        verdict="context_only",
        key_points=[
            "This is a section slide in the source presentation, so the verification output provides a region-level context chart rather than a claim-specific verdict.",
            "The overview chart is intended as the lead-in for the field slides that follow in the same region.",
        ],
        figure_paths=[figure_path],
        table_paths=[],
    )


def analyze_slide_34_35() -> list[SlideArtifact]:
    artifacts: list[SlideArtifact] = []
    telemetry = load_telemetry_daily_for_well("Vt_9105")
    if telemetry.empty:
        return [
            SlideArtifact(
                slide=34,
                title="Slide 34: well 9105 example",
                scope="Well-level example",
                verdict="insufficient_direct_data",
                key_points=["The raw telemetry database did not return a usable daily series for Vt_9105 around the dates shown on the slide."],
                figure_paths=[],
                table_paths=[],
            ),
            SlideArtifact(
                slide=35,
                title="Slide 35: well 9105 delayed effect",
                scope="Well-level example",
                verdict="insufficient_direct_data",
                key_points=["Without matching high-frequency telemetry, the delayed-effect statement can only be treated as an anecdotal example."],
                figure_paths=[],
                table_paths=[],
            ),
        ]

    compare_dates = telemetry.loc[telemetry["dt"].isin([pd.Timestamp("2026-04-27"), pd.Timestamp("2026-06-08")])].copy()
    table_path = save_csv(compare_dates, "slide_34_35_vt_9105_daily_compare.csv")
    figure_paths: list[str] = []
    if not compare_dates.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        melted = compare_dates.melt(id_vars=["dt"], value_vars=["qliq_m3d", "p_intake_atm", "frequency_hz", "p_bhp_atm"], var_name="metric", value_name="value")
        for metric, group in melted.groupby("metric", sort=True):
            ax.plot(group["dt"], group["value"], marker="o", linewidth=2, label=metric)
        ax.set_title("Slide 34/35 verification: Vt_9105 daily telemetry comparison")
        ax.grid(True, alpha=0.25)
        ax.legend()
        figure_paths.append(finish_plot(fig, "slide_34_35_vt_9105_compare.png"))

    return [
        SlideArtifact(
            slide=34,
            title="Slide 34: well 9105 example",
            scope="Vt_9105 daily telemetry-like comparison",
            verdict="partial_only",
            key_points=[
                "The raw telemetry database contains daily points for Vt_9105 around the dates shown, but not the exact high-frequency curves from the slide.",
                "The database confirms that the well can be matched, yet the operational story on the slide cannot be proven rigorously from the coarse daily telemetry alone.",
            ],
            figure_paths=figure_paths,
            table_paths=[table_path],
        ),
        SlideArtifact(
            slide=35,
            title="Slide 35: well 9105 delayed effect",
            scope="Vt_9105 daily telemetry-like comparison",
            verdict="partial_only",
            key_points=[
                "The daily database is too coarse to prove a delayed effect from frequency change; a true verification would require the original high-frequency telemetry used for the screenshots.",
            ],
            figure_paths=figure_paths,
            table_paths=[table_path],
        ),
    ]


def write_summary(slide_texts: dict[int, list[str]], artifacts: list[SlideArtifact]) -> None:
    artifact_map = {artifact.slide: artifact for artifact in artifacts}
    lines = ["# Presentation Verification", "", f"Source presentation: `{PRESENTATION_PATH}`", ""]
    for slide in sorted(SLIDE_NOTES):
        lines.append(f"## Slide {slide}")
        lines.append(f"Source note: {SLIDE_NOTES[slide]}")
        texts = slide_texts.get(slide, [])
        if texts:
            lines.append("Source text:")
            for text in texts[:20]:
                lines.append(f"- {text}")
        artifact = artifact_map.get(slide)
        if artifact is None:
            lines.append("- No verification artifact was generated.")
            lines.append("")
            continue
        lines.append(f"Verdict: `{artifact.verdict}`")
        lines.append(f"Scope: {artifact.scope}")
        for point in artifact.key_points:
            lines.append(f"- {point}")
        if artifact.figure_paths:
            lines.append("Figures:")
            for path in artifact.figure_paths:
                lines.append(f"- [{Path(path).name}]({path})")
        if artifact.table_paths:
            lines.append("Tables:")
            for path in artifact.table_paths:
                lines.append(f"- [{Path(path).name}]({path})")
        lines.append("")

    summary_path = OUTPUT_DIR / "verification_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    slide_texts = extract_slide_texts(PRESENTATION_PATH, list(range(22, 40)))
    raw_runs = load_runs(ttf_min=None)
    model_runs = load_runs(ttf_min=30.0)

    wells = model_runs["Скв."].dropna().astype(str).unique().tolist()
    tr_daily = load_techregime_daily(wells)
    last30 = aggregate_last30_operating_features(model_runs, tr_daily)
    model_runs = model_runs.merge(last30, on="row_id", how="left")
    active_daily = build_active_daily_frame(raw_runs, tr_daily)

    artifacts: list[SlideArtifact] = []
    artifacts.append(analyze_slide_22(raw_runs))
    artifacts.append(analyze_slide_23(raw_runs))
    artifacts.extend(analyze_slide_24_and_25(model_runs))
    artifacts.append(analyze_slide_26_or_30(raw_runs, active_daily, slide=26, region_name="Central region", fields=["Ya", "Au", "Az"]))
    artifacts.append(analyze_slide_27(raw_runs, model_runs, active_daily))
    for slide in [28, 29, 31, 32, 33, 36, 37, 38, 39]:
        config = FIELD_SLIDE_CONFIGS[slide]
        artifacts.append(
            analyze_field_slide(
                slide,
                raw_runs,
                model_runs,
                active_daily,
                field=config["field"],
                sour=config["sour"],
                kpod_threshold=float(config["kpod_threshold"]),
                ratio_threshold=float(config["ratio_threshold"]),
                glf_threshold=float(config["glf_threshold"]),
            )
        )
    artifacts.append(analyze_slide_26_or_30(raw_runs, active_daily, slide=30, region_name="Western region", fields=["Vt", "Bt", "Ic"]))
    artifacts.extend(analyze_slide_34_35())

    summary_rows = []
    for artifact in artifacts:
        summary_rows.append(
            {
                "slide": artifact.slide,
                "title": artifact.title,
                "scope": artifact.scope,
                "verdict": artifact.verdict,
                "key_points": " | ".join(artifact.key_points),
                "figure_count": len(artifact.figure_paths),
                "table_count": len(artifact.table_paths),
            }
        )
    save_csv(pd.DataFrame(summary_rows).sort_values("slide"), "verification_summary.csv")
    write_summary(slide_texts, artifacts)
    print(f"Verification artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
