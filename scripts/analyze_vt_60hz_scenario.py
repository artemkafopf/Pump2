from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ModuleNotFoundError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None

from analysis.transform_selection import rank_transform_candidates
from analysis.weibull_model import fit_weibull_stress_model
from analysis.sqlite_paths import resolve_telemetry_db_path
from scripts.data_utils import split_by_run_id
from scripts.analyze_failure_horizon import (
    ALL_PATH,
    DatasetSpec,
    TR_FEATURE_COLUMNS,
    build_dataset,
    evaluate_catboost_cv,
    load_runs as load_target_runs,
    load_techregime_daily,
    prepare_feature_matrix,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_scenario_2026_06_18"
TELEMETRY_DB_PATH = resolve_telemetry_db_path()

TELEMETRY_RAW_QUERY_COLUMNS = {
    "qliq": "col_0007",
    "watercut": "col_0008",
    "gas_factor": "col_0013",
    "load": "col_0016",
    "rpump_intake": "col_0017",
    "freq": "col_0018",
    "rzab": "col_0019",
}

RISK_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "Kpod_30d",
    "Kpod_freq_30d",
    "pressure_ratio_bhp_30d",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_std",
    "Работа в кривизне",
]

QLIQ_RESPONSE_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "pressure_ratio_bhp_30d",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_mean",
    "load_30d_std",
    "Работа в кривизне",
]

LOAD_MEAN_RESPONSE_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "qliq_30d_mean",
    "pressure_ratio_bhp_30d",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_std",
    "Работа в кривизне",
]

LOAD_STD_RESPONSE_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "qliq_30d_mean",
    "pressure_ratio_bhp_30d",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_mean",
    "Работа в кривизне",
]

PRESSURE_RESPONSE_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "qliq_30d_mean",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_mean",
    "load_30d_std",
    "Работа в кривизне",
]

MODE_FEATURES = [
    "Принадлежность",
    "pump_family",
    "days_since_install",
    "freq_30d_mean",
    "Kpod_30d",
    "Kpod_freq_30d",
    "pressure_ratio_bhp_30d",
    "gas_factor_30d_mean",
    "kprod_30d_mean",
    "load_30d_mean",
    "Работа в кривизне",
]

STRESS_CANDIDATES = [
    {"column": "frequency_to_reference_ratio_30d", "scale": 0.03},
    {"column": "Kpod_30d", "scale": 0.10},
    {"column": "Kpod_freq_30d", "scale": 0.10},
    {"column": "pressure_ratio_bhp_30d", "scale": 0.05},
    {"column": "gas_factor_30d_mean", "scale": 100.0},
    {"column": "load_30d_std", "scale": 2.0},
    {"column": "Работа в кривизне", "scale": 1.0},
]

STRESS_TRANSFORMS = [
    "negative_excess",
    "positive_excess",
    "relative_abs_deviation",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


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


def normalize_failure_node(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    lowered = text.casefold()
    alias_map = {
        "нкт ": "НКТ",
        "нкт": "НКТ",
        "пэд": "ПЭД",
        "эцн": "ЭЦН",
        "гидрозащита": "Гидрозащита",
        "кабельная линия": "Кабельная линия",
        "диспергатор": "Диспергатор",
        "клапан сливной": "Клапан сливной",
        "газосепаратор": "Газосепаратор",
        "тмс": "ТМС",
        "подвесной патрубок": "Подвесной патрубок",
        "межфланцевое соединение": "Межфланцевое соединение",
    }
    return alias_map.get(lowered, text)


def normalize_failure_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def watercut_fraction(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    fraction = numeric.copy()
    fraction.loc[fraction > 1.0] = fraction.loc[fraction > 1.0] / 100.0
    return fraction.clip(lower=0.0, upper=1.0)


def normalize_well_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def stable_group_test_mask(series: pd.Series, test_fraction: float = 0.2) -> np.ndarray:
    modulus = max(2, int(round(1.0 / max(min(test_fraction, 0.5), 0.05))))

    def is_test(value: object) -> bool:
        key = "<missing>" if pd.isna(value) else str(value)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % modulus == 0

    return series.map(is_test).to_numpy(dtype=bool)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = truth - pred
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    truth_mean = float(np.mean(truth))
    ss_tot = float(np.sum(np.square(truth - truth_mean)))
    ss_res = float(np.sum(np.square(residual)))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def prepare_frequency_response_rows(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    required = ["well_id", "Принадлежность", "pump_family", "freq_30d_mean", "freq_prev30_mean", "qliq_30d_mean", "qliq_prev30_mean"]
    for column in required:
        if column not in working.columns:
            raise ValueError(f"Frequency response fitting requires column '{column}'.")
    working["freq_30d_mean"] = _numeric(working["freq_30d_mean"])
    working["freq_prev30_mean"] = _numeric(working["freq_prev30_mean"])
    working["qliq_30d_mean"] = _numeric(working["qliq_30d_mean"])
    working["qliq_prev30_mean"] = _numeric(working["qliq_prev30_mean"])
    working = working.loc[
        working["freq_30d_mean"].between(20.0, 70.0)
        & working["freq_prev30_mean"].between(20.0, 70.0)
        & (working["qliq_30d_mean"] > 0.0)
        & (working["qliq_prev30_mean"] > 0.0)
    ].copy()
    working["delta_freq_hz"] = working["freq_30d_mean"] - working["freq_prev30_mean"]
    working = working.loc[working["delta_freq_hz"].abs() >= 1.0].copy()
    working["log_freq_ratio"] = np.log(working["freq_30d_mean"] / working["freq_prev30_mean"])
    working["log_qliq_ratio"] = np.log(working["qliq_30d_mean"] / working["qliq_prev30_mean"])
    working = working.loc[working["log_freq_ratio"].replace([np.inf, -np.inf], np.nan).notna()].copy()
    working = working.loc[working["log_qliq_ratio"].replace([np.inf, -np.inf], np.nan).notna()].copy()
    working["Принадлежность_key"] = working["Принадлежность"].astype("string").fillna("<missing>")
    working["pump_family_key"] = working["pump_family"].astype("string").fillna("<missing>")
    working["weight"] = np.clip(working["delta_freq_hz"].abs(), 1.0, 8.0)
    return working


def build_frequency_response_model(working: pd.DataFrame, *, group_column: str) -> FrequencyResponseModel:
    def fit_slope(frame: pd.DataFrame) -> tuple[float, int, int]:
        x = frame["log_freq_ratio"].to_numpy(dtype=float)
        y = frame["log_qliq_ratio"].to_numpy(dtype=float)
        w = frame["weight"].to_numpy(dtype=float)
        denom = float(np.sum(w * np.square(x)))
        if denom <= 1e-12:
            return 1.0, int(len(frame)), int(frame[group_column].nunique())
        slope = float(np.sum(w * x * y) / denom)
        return float(np.clip(slope, 0.0, 1.5)), int(len(frame)), int(frame[group_column].nunique())

    coefficient_rows: list[dict[str, object]] = []
    global_elasticity, global_rows, global_wells = fit_slope(working)
    coefficient_rows.append(
        {
            "level": "global",
            "Принадлежность_key": "<all>",
            "pump_family_key": "<all>",
            "elasticity": global_elasticity,
            "rows": global_rows,
            "wells": global_wells,
        }
    )

    for keys, level, min_group_rows, min_group_wells in [
        (["pump_family_key"], "pump", 25, 6),
        (["Принадлежность_key"], "contractor", 25, 6),
        (["Принадлежность_key", "pump_family_key"], "contractor_pump", 16, 4),
    ]:
        for group_values, frame in working.groupby(keys, dropna=False):
            elasticity, rows_count, wells_count = fit_slope(frame)
            if rows_count < min_group_rows or wells_count < min_group_wells:
                continue
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            payload = {
                "level": level,
                "Принадлежность_key": "<all>",
                "pump_family_key": "<all>",
                "elasticity": elasticity,
                "rows": rows_count,
                "wells": wells_count,
            }
            for key_name, key_value in zip(keys, group_values):
                payload[key_name] = str(key_value)
            coefficient_rows.append(payload)

    coefficients = pd.DataFrame(coefficient_rows)
    return FrequencyResponseModel(metrics={}, coefficients=coefficients, global_elasticity=float(global_elasticity))


def fit_frequency_response_model(
    df: pd.DataFrame,
    *,
    group_column: str = "row_id",
    min_rows: int = 120,
) -> FrequencyResponseModel:
    working = prepare_frequency_response_rows(df)
    if len(working) < min_rows:
        raise ValueError(f"Not enough rows to fit frequency response model: {len(working)} < {min_rows}.")

    fitted = build_frequency_response_model(working, group_column=group_column)

    train, test = split_by_run_id(working, run_id_column=group_column, test_fraction=0.2)
    if len(test) < max(30, int(len(working) * 0.1)) or len(train) < max(80, int(len(working) * 0.4)):
        fallback_mask = np.zeros(len(working), dtype=bool)
        fallback_mask[::5] = True
        train = working.loc[~fallback_mask].copy()
        test = working.loc[fallback_mask].copy()
    train_model = None
    metrics: dict[str, object] = {
        "rows": int(len(working)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "unique_wells": int(working[group_column].nunique()),
    }
    if len(train) >= min_rows // 2 and len(test) > 0:
        train_model = build_frequency_response_model(train, group_column=group_column)
        elasticity = train_model.elasticity_for_rows(test)
        predicted = test["qliq_prev30_mean"] * np.power(test["freq_30d_mean"] / test["freq_prev30_mean"], elasticity)
        metrics.update(regression_metrics(test["qliq_30d_mean"].to_numpy(dtype=float), predicted.to_numpy(dtype=float)))
        metrics["mean_elasticity_test"] = float(elasticity.mean())

    metrics["global_elasticity"] = float(fitted.global_elasticity)
    metrics["group_count"] = int(len(fitted.coefficients))
    fitted.metrics = metrics
    return fitted


def prediction_frame(rows: pd.DataFrame, prepared_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    categorical_set = set(categorical_columns)
    prepared = pd.DataFrame(index=rows.index)
    for column in prepared_columns:
        if column in rows.columns:
            if column in categorical_set:
                prepared[column] = rows[column].astype("string").fillna("__missing__")
            else:
                prepared[column] = _numeric(rows[column])
        else:
            prepared[column] = "__missing__" if column in categorical_set else np.nan
    return prepared


@dataclass(slots=True)
class TrainedRegressor:
    target_column: str
    features: list[str]
    prepared_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, object]
    model: CatBoostRegressor


@dataclass(slots=True)
class TrainedBinaryClassifier:
    target_column: str
    features: list[str]
    prepared_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, object]
    model: CatBoostClassifier


@dataclass(slots=True)
class TrainedMulticlassClassifier:
    target_column: str
    features: list[str]
    prepared_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, object]
    model: CatBoostClassifier
    classes: list[str]


@dataclass(slots=True)
class FrequencyResponseModel:
    metrics: dict[str, object]
    coefficients: pd.DataFrame
    global_elasticity: float

    def elasticity_for_rows(self, rows: pd.DataFrame) -> pd.Series:
        frame = rows.copy()
        frame["Принадлежность_key"] = frame.get("Принадлежность", pd.Series(index=frame.index, dtype="object")).astype("string").fillna("<missing>")
        frame["pump_family_key"] = frame.get("pump_family", pd.Series(index=frame.index, dtype="object")).astype("string").fillna("<missing>")

        coefficient_frame = self.coefficients.copy()
        coefficient_frame["Принадлежность_key"] = coefficient_frame["Принадлежность_key"].astype("string")
        coefficient_frame["pump_family_key"] = coefficient_frame["pump_family_key"].astype("string")

        result = pd.Series(np.nan, index=frame.index, dtype=float)
        for level in ["contractor_pump", "pump", "contractor", "global"]:
            subset = coefficient_frame.loc[coefficient_frame["level"] == level].copy()
            if subset.empty:
                continue
            if level == "contractor_pump":
                merged = frame[["Принадлежность_key", "pump_family_key"]].reset_index().merge(
                    subset[["Принадлежность_key", "pump_family_key", "elasticity"]],
                    on=["Принадлежность_key", "pump_family_key"],
                    how="left",
                )
                values = _numeric(merged.set_index("index")["elasticity"]).reindex(frame.index)
            elif level == "pump":
                merged = frame[["pump_family_key"]].reset_index().merge(
                    subset[["pump_family_key", "elasticity"]],
                    on="pump_family_key",
                    how="left",
                )
                values = _numeric(merged.set_index("index")["elasticity"]).reindex(frame.index)
            elif level == "contractor":
                merged = frame[["Принадлежность_key"]].reset_index().merge(
                    subset[["Принадлежность_key", "elasticity"]],
                    on="Принадлежность_key",
                    how="left",
                )
                values = _numeric(merged.set_index("index")["elasticity"]).reindex(frame.index)
            else:
                values = pd.Series(np.repeat(float(self.global_elasticity), len(frame)), index=frame.index, dtype=float)
            result.loc[result.isna() & values.notna()] = values.loc[result.isna() & values.notna()]
        return result.fillna(float(self.global_elasticity))


def fit_regressor(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    group_column: str = "row_id",
    min_rows: int = 160,
) -> TrainedRegressor:
    if CatBoostRegressor is None or Pool is None:
        raise RuntimeError("CatBoost is not available in the current environment.")

    working = df.copy()
    working[target_column] = _numeric(working[target_column])
    working = working.loc[working[target_column].notna()].copy()
    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    if len(prepared) < min_rows:
        raise ValueError(f"Not enough rows to fit regressor for '{target_column}'.")
    aligned = working.loc[prepared.index].copy()
    train_rows, test_rows = split_by_run_id(aligned, run_id_column=group_column, test_fraction=0.2)
    train_index = train_rows.index
    test_index = test_rows.index
    if len(test_index) < max(30, int(len(aligned) * 0.1)) or len(train_index) < max(80, int(len(aligned) * 0.4)):
        fallback_mask = np.zeros(len(aligned), dtype=bool)
        fallback_mask[::5] = True
        train_index = aligned.index[~fallback_mask]
        test_index = aligned.index[fallback_mask]

    train_x = prepared.loc[train_index]
    test_x = prepared.loc[test_index]
    train_y = aligned.loc[train_index, target_column].to_numpy(dtype=float)
    test_y = aligned.loc[test_index, target_column].to_numpy(dtype=float)
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]

    eval_model = CatBoostRegressor(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    eval_model.fit(Pool(train_x, train_y, cat_features=cat_indices))
    metrics: dict[str, object] = {
        "rows": int(len(aligned)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "feature_count": int(prepared.shape[1]),
        "categorical_feature_count": int(len(categorical_columns)),
    }
    if len(test_x) > 0:
        metrics.update(regression_metrics(test_y, eval_model.predict(test_x)))

    final_model = CatBoostRegressor(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    final_model.fit(Pool(prepared, aligned[target_column].to_numpy(dtype=float), cat_features=cat_indices))
    return TrainedRegressor(
        target_column=target_column,
        features=[column for column in feature_columns if column in df.columns],
        prepared_columns=prepared.columns.tolist(),
        categorical_columns=categorical_columns,
        metrics=metrics,
        model=final_model,
    )


def fit_binary_classifier(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    group_column: str = "row_id",
    min_rows: int = 200,
) -> TrainedBinaryClassifier:
    if CatBoostClassifier is None or Pool is None:
        raise RuntimeError("CatBoost is not available in the current environment.")

    working = df.loc[df[target_column].isin([0, 1])].copy()
    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    if len(prepared) < min_rows:
        raise ValueError(f"Not enough rows to fit classifier for '{target_column}'.")
    aligned = working.loc[prepared.index].copy()
    metrics = evaluate_catboost_cv(aligned, prepared.columns.tolist(), group_column=group_column, label_column=target_column)
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, max(1.0, float((aligned[target_column] == 0).sum()) / max(float((aligned[target_column] == 1).sum()), 1.0))],
    )
    model.fit(Pool(prepared, aligned[target_column].astype(int), cat_features=cat_indices))
    return TrainedBinaryClassifier(
        target_column=target_column,
        features=[column for column in feature_columns if column in df.columns],
        prepared_columns=prepared.columns.tolist(),
        categorical_columns=categorical_columns,
        metrics=metrics,
        model=model,
    )


def fit_multiclass_classifier(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    group_column: str = "row_id",
    min_rows: int = 60,
) -> TrainedMulticlassClassifier:
    if CatBoostClassifier is None or Pool is None:
        raise RuntimeError("CatBoost is not available in the current environment.")

    working = df.copy()
    working[target_column] = working[target_column].astype("string")
    working = working.loc[working[target_column].notna()].copy()
    if len(working) < min_rows or int(working[target_column].nunique(dropna=True)) < 2:
        raise ValueError(f"Not enough labeled rows to fit multiclass classifier for '{target_column}'.")

    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    aligned = working.loc[prepared.index].copy()
    train_rows, test_rows = split_by_run_id(aligned, run_id_column=group_column, test_fraction=0.2)
    train_index = train_rows.index
    test_index = test_rows.index
    if len(test_index) < max(20, int(len(aligned) * 0.1)) or len(train_index) < max(60, int(len(aligned) * 0.4)):
        fallback_mask = np.zeros(len(aligned), dtype=bool)
        fallback_mask[::5] = True
        train_index = aligned.index[~fallback_mask]
        test_index = aligned.index[fallback_mask]

    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    eval_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    eval_model.fit(Pool(prepared.loc[train_index], aligned.loc[train_index, target_column], cat_features=cat_indices))
    metrics: dict[str, object] = {
        "rows": int(len(aligned)),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "feature_count": int(prepared.shape[1]),
        "categorical_feature_count": int(len(categorical_columns)),
    }
    if int(len(test_index)) > 0:
        predictions = eval_model.predict(prepared.loc[test_index]).reshape(-1)
        truth = aligned.loc[test_index, target_column].to_numpy(dtype=object)
        metrics["accuracy"] = float(np.mean(predictions == truth))

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(prepared, aligned[target_column], cat_features=cat_indices))
    class_names = [str(item) for item in model.classes_]
    return TrainedMulticlassClassifier(
        target_column=target_column,
        features=[column for column in feature_columns if column in df.columns],
        prepared_columns=prepared.columns.tolist(),
        categorical_columns=categorical_columns,
        metrics=metrics,
        model=model,
        classes=class_names,
    )


def predict_regressor(model: TrainedRegressor, rows: pd.DataFrame) -> np.ndarray:
    prepared = prediction_frame(rows, model.prepared_columns, model.categorical_columns)
    return model.model.predict(prepared)


def predict_binary_proba(model: TrainedBinaryClassifier, rows: pd.DataFrame) -> np.ndarray:
    prepared = prediction_frame(rows, model.prepared_columns, model.categorical_columns)
    return model.model.predict_proba(prepared)[:, 1]


def predict_multiclass_proba(model: TrainedMulticlassClassifier, rows: pd.DataFrame) -> pd.DataFrame:
    prepared = prediction_frame(rows, model.prepared_columns, model.categorical_columns)
    probabilities = model.model.predict_proba(prepared)
    return pd.DataFrame(probabilities, columns=model.classes, index=rows.index)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_telemetry_daily(wells: list[str]) -> pd.DataFrame:
    well_map = {normalize_well_key(well): str(well).strip() for well in wells if normalize_well_key(well)}
    normalized_wells = sorted(key for key in well_map if key)
    if not normalized_wells:
        return pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])

    query_columns = {
        "qliq": "Qliq_m3d",
        "qgas": "Qgas_m3d",
        "watercut": "watercut_percent",
        "gas_factor": "GLF_m3m3",
        "load": "motor_load_percent",
        "rpump_intake": "P_intake_atm",
        "freq": "frequency_hz",
        "rzab": "P_bhp_atm",
    }
    query_select = ", ".join([f"{storage} as {alias}" for alias, storage in query_columns.items()])
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TELEMETRY_DB_PATH) as connection:
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT _meta_normalized_well as well_key, _meta_record_date as dt, {query_select}
                FROM telemetry_daily
                WHERE _meta_normalized_well IN ({placeholders})
            """
            frames.append(pd.read_sql_query(query, connection, params=chunk))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["well_key", "dt", *query_columns])
    if df.empty:
        return pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for column in query_columns:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_key"].notna() & df["dt"].notna()].copy()
    df["well_id"] = df["well_key"].map(well_map).fillna(df["well_key"])
    df["rpl"] = np.nan
    df["kprod"] = np.nan
    df["qgas"] = _numeric(df["qgas"])
    missing_qgas = df["qgas"].isna() & df["qliq"].notna() & df["gas_factor"].notna()
    df.loc[missing_qgas, "qgas"] = df.loc[missing_qgas, "qliq"] * df.loc[missing_qgas, "gas_factor"]

    telemetry = (
        df.groupby(["well_id", "dt"], as_index=False)[["freq", "load", "rpl", "rpump_intake", "rzab", "qliq", "watercut", "gas_factor", "qgas", "kprod"]]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return telemetry


def load_dynamic_daily_telemetry_first(wells: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    telemetry = load_telemetry_daily(wells)
    techregime = load_techregime_daily(wells)

    telemetry = telemetry.copy()
    techregime = techregime.copy()
    telemetry["well_key"] = telemetry["well_id"].map(normalize_well_key)
    techregime["well_key"] = techregime["well_id"].map(normalize_well_key)

    merged = telemetry.merge(
        techregime,
        on=["well_key", "dt"],
        how="outer",
        suffixes=("_tel", "_tr"),
    )

    well_id = merged.get("well_id_tel")
    if well_id is None:
        well_id = pd.Series(index=merged.index, dtype="string")
    merged["well_id"] = well_id.astype("string").fillna(merged.get("well_id_tr", pd.Series(index=merged.index, dtype="string")).astype("string"))

    source_rows: list[dict[str, object]] = []
    result = pd.DataFrame({"well_id": merged["well_id"], "dt": merged["dt"]})
    for column in TR_FEATURE_COLUMNS:
        tel_name = f"{column}_tel"
        tr_name = f"{column}_tr"
        tel_series = _numeric(merged[tel_name]) if tel_name in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
        tr_series = _numeric(merged[tr_name]) if tr_name in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
        result[column] = tel_series.combine_first(tr_series)
        source = pd.Series(pd.NA, index=merged.index, dtype="string")
        source.loc[tel_series.notna()] = "telemetry"
        source.loc[tel_series.isna() & tr_series.notna()] = "techregime"
        result[f"{column}_source"] = source
        source_rows.append(
            {
                "column": column,
                "telemetry_rows": int(tel_series.notna().sum()),
                "techregime_rows": int(tr_series.notna().sum()),
                "merged_rows": int(result[column].notna().sum()),
                "telemetry_share_of_merged": float(tel_series.notna().sum() / max(int(result[column].notna().sum()), 1)),
            }
        )

    result = result.loc[result["well_id"].notna() & result["dt"].notna()].copy()
    result = result.sort_values(["well_id", "dt"]).reset_index(drop=True)
    source_summary = pd.DataFrame(source_rows).sort_values("column").reset_index(drop=True)
    return result, source_summary
def load_rolling_dataset(field_code: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = load_target_runs(ALL_PATH)
    if field_code is not None:
        runs = runs.loc[runs["Месторождение"].astype("string") == field_code].copy()
    runs["pump_family"] = runs["Тип УЭЦН"].map(pump_family).astype("string")
    runs["failure_node"] = runs["Отказавший узел"].map(normalize_failure_node).astype("string")
    runs["failure_element"] = runs["Отказавший элемент"].map(normalize_failure_text).astype("string")
    dynamic_daily, source_summary = load_dynamic_daily_telemetry_first(runs["Скв."].dropna().astype(str).unique().tolist())
    dataset = build_dataset(DatasetSpec(name="rolling_h30", mode="rolling", horizon_days=30), runs, dynamic_daily)
    run_meta = runs[
        [
            "row_id",
            "Скв.",
            "Тип УЭЦН",
            "pump_family",
            "failure_node",
            "failure_element",
            "Отказавший элемент",
            "Причина отказа УЭЦН",
            "Кислый/Некислый",
        ]
    ].copy()
    dataset = dataset.merge(run_meta, on="row_id", how="left", suffixes=("", "_run"))
    dataset["anchor_date"] = pd.to_datetime(dataset["anchor_date"], errors="coerce")
    dataset["pump_family"] = dataset["pump_family"].astype("string").fillna("<missing>")
    dataset["watercut_fraction"] = watercut_fraction(dataset["watercut_30d_mean"])
    dataset["oil_rate_30d_mean"] = _numeric(dataset["qliq_30d_mean"]) * (1.0 - dataset["watercut_fraction"])
    dataset["analysis_row_id"] = np.arange(len(dataset), dtype=int)
    return runs, dynamic_daily, dataset, source_summary


def load_vt_rolling_dataset(field_code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_rolling_dataset(field_code)


def apply_quality_filters(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    working = dataset.copy()
    notes: list[str] = []

    masks = {
        "freq_30d_mean in [20, 70]": _numeric(working["freq_30d_mean"]).between(20.0, 70.0),
        "qliq_30d_mean > 0": _numeric(working["qliq_30d_mean"]) > 0.0,
        "pressure_ratio_bhp_30d in [0.02, 1.20]": _numeric(working["pressure_ratio_bhp_30d"]).between(0.02, 1.20),
        "Kpod_30d in [0.05, 2.50]": _numeric(working["Kpod_30d"]).between(0.05, 2.50),
        "gas_factor_30d_mean in [0, 5000]": _numeric(working["gas_factor_30d_mean"]).between(0.0, 5000.0),
        "kprod_30d_mean in [0, 100]": _numeric(working["kprod_30d_mean"]).between(0.0, 100.0),
        "load_30d_mean in [1, 100]": _numeric(working["load_30d_mean"]).between(1.0, 100.0),
        "load_30d_std in [0, 60]": _numeric(working["load_30d_std"]).between(0.0, 60.0),
        "watercut_30d_mean in [0, 100]": _numeric(working["watercut_30d_mean"]).between(0.0, 100.0),
    }

    combined = pd.Series(True, index=working.index)
    for label, mask in masks.items():
        before = int(combined.sum())
        combined &= mask.fillna(False)
        after = int(combined.sum())
        notes.append(f"{label}: {before} -> {after} rows")

    filtered = working.loc[combined].copy().reset_index(drop=True)
    return filtered, notes


def latest_portfolio_rows(dataset: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    latest = dataset.sort_values(["well_id", "anchor_date"]).groupby("well_id", as_index=False).tail(1).copy()
    latest = latest.loc[latest["anchor_date"] >= cutoff_date].copy()
    return latest.sort_values(["anchor_date", "well_id"]).reset_index(drop=True)


def build_term_payload(column: str, transform: str, scale: float, reference_init: float, reference_lower: float, reference_upper: float) -> dict[str, object]:
    return {
        "name": f"stress_{column}_{transform}",
        "column": column,
        "transform": transform,
        "reference_mode": "fit",
        "reference_init": float(reference_init),
        "reference_bounds": [float(reference_lower), float(reference_upper)],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_bounds": [0.0, None],
        "scale": float(scale),
    }


def select_stress_terms(
    dataset: pd.DataFrame,
    *,
    duration_column: str,
    event_column: str,
    group_columns: list[str],
    min_group_size: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for candidate in STRESS_CANDIDATES:
        column = str(candidate["column"])
        if column not in dataset.columns:
            continue
        numeric_series = _numeric(dataset[column])
        coverage = float(numeric_series.notna().mean())
        numeric = numeric_series.dropna()
        if len(numeric) < 120 or coverage < 0.75:
            continue
        ranked = rank_transform_candidates(
            dataset,
            duration_column=duration_column,
            event_column=event_column,
            column=column,
            group_columns=group_columns,
            min_group_size=min_group_size,
            candidate_transforms=STRESS_TRANSFORMS,
            coefficient_non_negative=True,
            scale=float(candidate["scale"]),
        )
        if not ranked:
            continue
        best = ranked[0]
        rows.append(
            {
                "column": column,
                "scale": float(candidate["scale"]),
                "transform": best.transform,
                "aic": float(best.aic),
                "delta_aic_vs_baseline": float(best.delta_aic_vs_baseline),
                "coefficient": None if best.coefficient is None else float(best.coefficient),
                "reference_value": None if best.reference_value is None else float(best.reference_value),
                "coverage": coverage,
                "success": bool(best.success),
                "message": str(best.message),
                "reference_init": float(numeric.median()),
                "reference_lower": float(numeric.min()),
                "reference_upper": float(numeric.max()),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return [], ranking

    ranking = ranking.sort_values(["success", "delta_aic_vs_baseline", "column"], ascending=[False, True, True]).reset_index(drop=True)
    selected_rows = ranking.loc[(ranking["success"]) & (ranking["delta_aic_vs_baseline"] <= -2.0)].head(4).copy()
    terms = [
        build_term_payload(
            str(row["column"]),
            str(row["transform"]),
            float(row["scale"]),
            float(row["reference_init"]),
            float(row["reference_lower"]),
            float(row["reference_upper"]),
        )
        for _, row in selected_rows.iterrows()
    ]
    return terms, ranking


def fit_weibull_with_fallback(dataset: pd.DataFrame, stress_terms: list[dict[str, object]]) -> tuple[Any, list[str]]:
    attempts = [
        (["Принадлежность", "pump_family"], 20),
        (["Принадлежность"], 20),
        (["pump_family"], 20),
        ([], 1),
    ]
    errors: list[str] = []
    for group_columns, min_group_size in attempts:
        try:
            result = fit_weibull_stress_model(
                dataset,
                duration_column="days_to_stop",
                event_column="event",
                group_columns=group_columns,
                stress_terms=stress_terms,
                min_group_size=min_group_size,
            )
            return result, errors
        except Exception as exc:  # pragma: no cover - depends on data profile
            errors.append(f"group_columns={group_columns or ['GLOBAL']}: {exc}")
    raise RuntimeError("Weibull fit failed for every fallback grouping option.")


def scenario_frequency_update(rows: pd.DataFrame, frequency_hz: float) -> pd.DataFrame:
    scenario = rows.copy()
    scenario["freq_30d_mean"] = float(frequency_hz)
    scenario["freq_30d_last"] = float(frequency_hz)
    nominal_frequency = _numeric(scenario["Номинальная частота, Гц"]).replace(0.0, np.nan)
    scenario["frequency_to_reference_ratio_30d"] = float(frequency_hz) / nominal_frequency
    scenario["frequency_over_reference_hz_30d"] = float(frequency_hz) - nominal_frequency
    return scenario


def clip_like(series: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    result = _numeric(series)
    if lower is not None:
        result = result.clip(lower=lower)
    if upper is not None:
        result = result.clip(upper=upper)
    return result


def propagate_scenario_state(
    baseline: pd.DataFrame,
    *,
    scenario_frequency_hz: float,
    qliq_model: FrequencyResponseModel,
    load_mean_model: TrainedRegressor,
    load_std_model: TrainedRegressor,
    pressure_model: TrainedRegressor,
) -> pd.DataFrame:
    # Phase 1 caveat:
    # The response models below and the failure model are trained on the same rolling-window dataset.
    # This means the propagated counterfactual state is not a fully causal simulation.
    # We keep this working approximation in Phase 1 and surface the limitation in outputs.
    scenario = scenario_frequency_update(baseline, scenario_frequency_hz)

    baseline_freq = _numeric(baseline["freq_30d_mean"]).replace(0.0, np.nan)
    elasticity = qliq_model.elasticity_for_rows(baseline)
    scenario["qliq_frequency_elasticity"] = elasticity
    scenario["qliq_mechanistic_linear_hz"] = _numeric(baseline["qliq_30d_mean"]) * (float(scenario_frequency_hz) / baseline_freq)
    scenario["qliq_30d_mean"] = clip_like(
        _numeric(baseline["qliq_30d_mean"]) * np.power(float(scenario_frequency_hz) / baseline_freq, elasticity),
        lower=1.0,
    )
    nominal_rate = _numeric(scenario["Ном. Произв. м₃/сут"]).replace(0.0, np.nan)
    nominal_frequency = _numeric(scenario["Номинальная частота, Гц"]).replace(0.0, np.nan)
    scenario["Kpod_30d"] = scenario["qliq_30d_mean"] / nominal_rate
    scenario["Kpod_freq_30d"] = scenario["Kpod_30d"] * (nominal_frequency / float(scenario_frequency_hz))

    scenario["load_30d_mean"] = clip_like(pd.Series(predict_regressor(load_mean_model, scenario), index=scenario.index), lower=0.0, upper=100.0)
    scenario["load_30d_std"] = clip_like(pd.Series(predict_regressor(load_std_model, scenario), index=scenario.index), lower=0.0, upper=100.0)
    scenario["pressure_ratio_bhp_30d"] = clip_like(
        pd.Series(predict_regressor(pressure_model, scenario), index=scenario.index),
        lower=0.01,
        upper=1.50,
    )
    pbubble = _numeric(scenario["Дав. Нас"])
    scenario["pressure_margin_to_bubble_30d"] = (scenario["pressure_ratio_bhp_30d"] - 1.0) * pbubble
    scenario["qliq_per_hz_30d"] = scenario["qliq_30d_mean"] / float(scenario_frequency_hz)
    power = _numeric(scenario["Мощность, кВт"]).replace(0.0, np.nan)
    scenario["qliq_per_kw_30d"] = scenario["qliq_30d_mean"] / power
    scenario["motor_load_per_hz_30d"] = scenario["load_30d_mean"] / float(scenario_frequency_hz)
    scenario["oil_rate_30d_mean"] = scenario["qliq_30d_mean"] * (1.0 - watercut_fraction(scenario["watercut_30d_mean"]))
    return scenario


def weibull_group_lookup(result: Any) -> dict[int, str]:
    prepared = result.prepared_df.copy()
    if "analysis_row_id" not in prepared.columns:
        return {}
    series = prepared[["analysis_row_id", "analysis_original_group"]].drop_duplicates()
    return {int(row.analysis_row_id): str(row.analysis_original_group) for row in series.itertuples(index=False)}


def weibull_row_summary(result: Any, group_key: str, row: pd.Series) -> dict[str, float]:
    beta = float(result.beta_for_group(group_key))
    eta = float(result.adjusted_eta(group_key, row=row))
    median = float(eta * math.pow(math.log(2.0), 1.0 / beta))
    mean = float(eta * math.gamma(1.0 + (1.0 / beta)))

    def fail_probability(days: float) -> float:
        return float(1.0 - math.exp(-math.pow(max(days, 1e-9) / eta, beta)))

    return {
        "beta": beta,
        "eta": eta,
        "median_ttf_days": median,
        "mean_ttf_days": mean,
        "fail_prob_30d": fail_probability(30.0),
        "fail_prob_90d": fail_probability(90.0),
        "fail_prob_365d": fail_probability(365.0),
    }


def summarize_mode_expectations(
    fail_probability: pd.Series,
    mode_probabilities: pd.DataFrame,
    *,
    top_n: int = 8,
    label: str = "failure_mode",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in mode_probabilities.columns:
        expected = float((fail_probability * mode_probabilities[column]).sum())
        conditional = float(mode_probabilities[column].mean())
        rows.append(
            {
                label: column,
                "expected_failures_30d": expected,
                "mean_conditional_probability": conditional,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("expected_failures_30d", ascending=False).head(top_n).reset_index(drop=True)


def summarize_elasticity_levels(
    model: FrequencyResponseModel,
    *,
    field_code: str,
) -> dict[str, object]:
    coefficients = model.coefficients.copy()
    global_row = coefficients.loc[coefficients["level"] == "global"].head(1)
    field_row = coefficients.loc[
        (coefficients["level"] == "contractor") | (coefficients["level"] == "global")
    ].head(0)
    pump_rows = coefficients.loc[coefficients["level"] == "pump"].copy()
    contractor_pump_rows = coefficients.loc[coefficients["level"] == "contractor_pump"].copy()
    summary: dict[str, object] = {
        "global_a": None if global_row.empty else float(global_row["elasticity"].iloc[0]),
        "field_code": field_code,
        "field_a": float(model.global_elasticity),
        "pump_count": int(len(pump_rows)),
        "pump_a_min": None if pump_rows.empty else float(pump_rows["elasticity"].min()),
        "pump_a_max": None if pump_rows.empty else float(pump_rows["elasticity"].max()),
        "pump_a_std": None if pump_rows.empty else float(pump_rows["elasticity"].std(ddof=0)),
        "contractor_pump_count": int(len(contractor_pump_rows)),
        "contractor_pump_a_min": None if contractor_pump_rows.empty else float(contractor_pump_rows["elasticity"].min()),
        "contractor_pump_a_max": None if contractor_pump_rows.empty else float(contractor_pump_rows["elasticity"].max()),
        "contractor_pump_a_std": None if contractor_pump_rows.empty else float(contractor_pump_rows["elasticity"].std(ddof=0)),
    }
    return summary


def feature_usage_table(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, columns in feature_sets.items():
        for order, column in enumerate(columns, start=1):
            rows.append({"model": model_name, "order": order, "feature": column})
    return pd.DataFrame(rows)


def render_report(
    *,
    output_dir: Path,
    field_code: str,
    scenario_frequency_hz: float,
    cutoff_date: pd.Timestamp,
    source_rows: int,
    filtered_rows: int,
    portfolio_rows: int,
    portfolio_wells: int,
    source_summary: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    global_qliq_model: FrequencyResponseModel,
    qliq_model: FrequencyResponseModel,
    load_mean_model: TrainedRegressor,
    load_std_model: TrainedRegressor,
    pressure_model: TrainedRegressor,
    ttf_model: TrainedRegressor,
    failure_model: TrainedBinaryClassifier,
    mode_model: TrainedMulticlassClassifier,
    element_model: TrainedMulticlassClassifier,
    weibull_result: Any,
    weibull_errors: list[str],
    stress_ranking: pd.DataFrame,
    selected_stress_terms: list[dict[str, object]],
    summary: dict[str, float | int | str],
    elasticity_summary: dict[str, object],
    top_nodes_baseline: pd.DataFrame,
    top_nodes_scenario: pd.DataFrame,
    top_elements_baseline: pd.DataFrame,
    top_elements_scenario: pd.DataFrame,
) -> None:
    report_path = output_dir / "vt_60hz_scenario_report.md"
    lines = [
        "# Vt 60 Hz Scenario Analysis",
        "",
        f"- Field: `{field_code}`",
        f"- Scenario: set `freq_30d_mean = {scenario_frequency_hz:.1f} Hz` for the latest portfolio windows",
        f"- Portfolio cutoff: `{cutoff_date.date()}`",
        f"- Rolling window horizon for failure classifier: `30 days`",
        "",
        "## Data",
        "",
        f"- Source rolling rows before quality filters: `{source_rows}`",
        f"- Rows after quality filters: `{filtered_rows}`",
        f"- Scenario portfolio rows: `{portfolio_rows}` across `{portfolio_wells}` wells",
        "",
        "Dynamic source priority was `telemetry daily -> techregime fallback`.",
        "Merged daily coverage by column:",
        "",
    ]
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"- `{row.column}`: telemetry `{row.telemetry_rows}`, techregime `{row.techregime_rows}`, merged `{row.merged_rows}`, telemetry share `{100.0 * float(row.telemetry_share_of_merged):.1f}%`"
        )

    lines.extend(["", "## Compact Feature Sets", ""])
    for model_name, columns in feature_sets.items():
        lines.append(f"- `{model_name}`: {', '.join(columns)}")

    lines.extend(
        [
            "",
            "## Model Quality",
            "",
            f"- `qliq` frequency response all-fields global: `{json.dumps(global_qliq_model.metrics, ensure_ascii=False)}`",
            f"- `qliq` frequency response: `{json.dumps(qliq_model.metrics, ensure_ascii=False)}`",
            f"- `load_mean` response: `{json.dumps(load_mean_model.metrics, ensure_ascii=False)}`",
            f"- `load_std` response: `{json.dumps(load_std_model.metrics, ensure_ascii=False)}`",
            f"- `pressure_ratio_bhp` response: `{json.dumps(pressure_model.metrics, ensure_ascii=False)}`",
            f"- `CatBoost TTF` regression: `{json.dumps(ttf_model.metrics, ensure_ascii=False)}`",
            f"- `CatBoost 30d failure` classifier: `{json.dumps(failure_model.metrics, ensure_ascii=False)}`",
            f"- `CatBoost failure node` classifier: `{json.dumps(mode_model.metrics, ensure_ascii=False)}`",
            f"- `CatBoost failure element` classifier: `{json.dumps(element_model.metrics, ensure_ascii=False)}`",
            "",
            "## Liquid Response Elasticity",
            "",
            f"- All-fields global `a`: `{elasticity_summary['global_a']:.3f}`",
            f"- Vt field-level `a`: `{elasticity_summary['field_a']:.3f}`",
            f"- Vt pump-family count with fitted `a`: `{elasticity_summary['pump_count']}`",
            f"- Vt pump-family `a` range: `{elasticity_summary['pump_a_min']:.3f}` to `{elasticity_summary['pump_a_max']:.3f}`",
            f"- Vt pump-family `a` std: `{elasticity_summary['pump_a_std']:.3f}`",
            f"- Vt contractor+pump count with fitted `a`: `{elasticity_summary['contractor_pump_count']}`",
            f"- Vt contractor+pump `a` range: `{elasticity_summary['contractor_pump_a_min']:.3f}` to `{elasticity_summary['contractor_pump_a_max']:.3f}`",
            f"- Vt contractor+pump `a` std: `{elasticity_summary['contractor_pump_a_std']:.3f}`",
            "",
            "## Weibull Stress Selection",
            "",
        ]
    )

    if stress_ranking.empty:
        lines.append("- No stress candidates produced a valid ranking.")
    else:
        for row in stress_ranking.itertuples(index=False):
            lines.append(
                f"- `{row.column}` -> `{row.transform}` with delta AIC `{row.delta_aic_vs_baseline:.2f}`"
            )

    lines.extend(["", "## Selected Weibull Stress Terms", ""])
    if not selected_stress_terms:
        lines.append("- No stress term met the selection threshold; Weibull fell back to category-only survival.")
    else:
        for term in selected_stress_terms:
            lines.append(
                f"- `{term['name']}`: column `{term['column']}`, transform `{term['transform']}`, scale `{term['scale']}`"
            )
    if weibull_errors:
        lines.extend(["", "## Weibull Fallback Notes", ""])
        for error in weibull_errors:
            lines.append(f"- {error}")

    lines.extend(
        [
            "",
            "## Portfolio Scenario Summary",
            "",
            f"- Baseline total liquid rate: `{summary['baseline_total_qliq']:.1f} m3/d`",
            f"- Scenario total liquid rate: `{summary['scenario_total_qliq']:.1f} m3/d` ({summary['delta_total_qliq_pct']:+.1f}%)",
            f"- Baseline total oil rate: `{summary['baseline_total_oil']:.1f} m3/d`",
            f"- Scenario total oil rate: `{summary['scenario_total_oil']:.1f} m3/d` ({summary['delta_total_oil_pct']:+.1f}%)",
            f"- CatBoost mean remaining TTF: `{summary['catboost_mean_ttf_base']:.1f}` -> `{summary['catboost_mean_ttf_scenario']:.1f}` days ({summary['catboost_mean_ttf_delta_pct']:+.1f}%)",
            f"- CatBoost median remaining TTF: `{summary['catboost_median_ttf_base']:.1f}` -> `{summary['catboost_median_ttf_scenario']:.1f}` days ({summary['catboost_median_ttf_delta_pct']:+.1f}%)",
            f"- CatBoost expected 30d failures: `{summary['catboost_expected_failures_30d_base']:.2f}` -> `{summary['catboost_expected_failures_30d_scenario']:.2f}` ({summary['catboost_expected_failures_30d_delta_pct']:+.1f}%)",
            f"- Weibull mean median TTF: `{summary['weibull_mean_median_ttf_base']:.1f}` -> `{summary['weibull_mean_median_ttf_scenario']:.1f}` days ({summary['weibull_mean_median_ttf_delta_pct']:+.1f}%)",
            f"- Weibull expected 30d failures: `{summary['weibull_expected_failures_30d_base']:.2f}` -> `{summary['weibull_expected_failures_30d_scenario']:.2f}` ({summary['weibull_expected_failures_30d_delta_pct']:+.1f}%)",
            f"- Weibull expected 90d failures: `{summary['weibull_expected_failures_90d_base']:.2f}` -> `{summary['weibull_expected_failures_90d_scenario']:.2f}` ({summary['weibull_expected_failures_90d_delta_pct']:+.1f}%)",
            f"- Weibull expected 365d failures: `{summary['weibull_expected_failures_365d_base']:.2f}` -> `{summary['weibull_expected_failures_365d_scenario']:.2f}` ({summary['weibull_expected_failures_365d_delta_pct']:+.1f}%)",
            "",
            "## Likely Failure Nodes In The Next 30 Days",
            "",
            "Baseline:",
        ]
    )
    for row in top_nodes_baseline.itertuples(index=False):
        lines.append(f"- `{row.failure_node}`: expected `{row.expected_failures_30d:.2f}` failures")
    lines.extend(["", "Scenario:"])
    for row in top_nodes_scenario.itertuples(index=False):
        lines.append(f"- `{row.failure_node}`: expected `{row.expected_failures_30d:.2f}` failures")
    lines.extend(["", "## Likely Failure Elements In The Next 30 Days", "", "Baseline:"])
    for row in top_elements_baseline.itertuples(index=False):
        lines.append(f"- `{row.failure_element}`: expected `{row.expected_failures_30d:.2f}` failures")
    lines.extend(["", "Scenario:"])
    for row in top_elements_scenario.itertuples(index=False):
        lines.append(f"- `{row.failure_element}`: expected `{row.expected_failures_30d:.2f}` failures")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is an observational scenario model, not a causal field test.",
            "- The 60 Hz case updates frequency directly and rescales liquid rate with a monotone within-well frequency elasticity, then recomputes dependent variables such as Kpod, load, and pressure ratio.",
            "- Watercut and gas factor are kept at their latest observed values in this first version, so oil-rate and gas-risk effects are conservative.",
            "- The Weibull model uses rolling windows with residual days-to-stop, so it should be interpreted as a state-conditioned survival model rather than a classic one-row-per-run fit.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a Vt 60 Hz scenario with CatBoost and Weibull survival.")
    parser.add_argument("--field", default="Vt", help="Field code to analyze.")
    parser.add_argument("--scenario-frequency", type=float, default=60.0, help="Scenario frequency in Hz.")
    parser.add_argument("--portfolio-cutoff", default="2025-01-01", help="Keep the latest portfolio window per well on or after this date.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for markdown and CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    cutoff_date = pd.Timestamp(args.portfolio_cutoff)

    _, _, rolling_all, _ = load_rolling_dataset(None)
    filtered_all, _ = apply_quality_filters(rolling_all)
    runs, dynamic_daily, rolling, source_summary = load_vt_rolling_dataset(args.field)
    filtered, filter_notes = apply_quality_filters(rolling)
    portfolio = latest_portfolio_rows(filtered, cutoff_date)
    if portfolio.empty:
        raise RuntimeError("The selected portfolio cutoff produced no latest windows.")

    feature_sets = {
        "ttf_and_failure": RISK_FEATURES,
        "qliq_frequency_response": ["freq_prev30_mean", "freq_30d_mean", "qliq_prev30_mean", "Принадлежность", "pump_family"],
        "load_mean_response": LOAD_MEAN_RESPONSE_FEATURES,
        "load_std_response": LOAD_STD_RESPONSE_FEATURES,
        "pressure_response": PRESSURE_RESPONSE_FEATURES,
        "failure_mode": MODE_FEATURES,
    }

    global_qliq_model = fit_frequency_response_model(filtered_all)
    qliq_model = fit_frequency_response_model(filtered)
    load_mean_model = fit_regressor(filtered, target_column="load_30d_mean", feature_columns=LOAD_MEAN_RESPONSE_FEATURES)
    load_std_model = fit_regressor(filtered, target_column="load_30d_std", feature_columns=LOAD_STD_RESPONSE_FEATURES)
    pressure_model = fit_regressor(filtered, target_column="pressure_ratio_bhp_30d", feature_columns=PRESSURE_RESPONSE_FEATURES)
    ttf_training = filtered.loc[filtered["event"].eq(1)].copy()
    ttf_model = fit_regressor(ttf_training, target_column="days_to_stop", feature_columns=RISK_FEATURES, min_rows=120)
    failure_model = fit_binary_classifier(filtered, target_column="label", feature_columns=RISK_FEATURES)

    mode_training = filtered.loc[filtered["label"].eq(1) & filtered["failure_node"].notna()].copy()
    try:
        mode_model = fit_multiclass_classifier(mode_training, target_column="failure_node", feature_columns=MODE_FEATURES)
    except ValueError:
        # Field-specific failure-node rows are too sparse; fall back to all-fields training.
        # Failure mode patterns are relatively universal, so cross-field training is acceptable here.
        print(f"  [warn] Insufficient field failure-node rows ({len(mode_training)}); falling back to all-fields training.")
        mode_training_all = filtered_all.loc[filtered_all["label"].eq(1) & filtered_all["failure_node"].notna()].copy()
        mode_model = fit_multiclass_classifier(mode_training_all, target_column="failure_node", feature_columns=MODE_FEATURES)

    element_training = filtered.loc[filtered["label"].eq(1) & filtered["failure_element"].notna()].copy()
    try:
        element_model = fit_multiclass_classifier(element_training, target_column="failure_element", feature_columns=MODE_FEATURES)
    except ValueError:
        print(f"  [warn] Insufficient field failure-element rows ({len(element_training)}); falling back to all-fields training.")
        element_training_all = filtered_all.loc[filtered_all["label"].eq(1) & filtered_all["failure_element"].notna()].copy()
        element_model = fit_multiclass_classifier(element_training_all, target_column="failure_element", feature_columns=MODE_FEATURES)

    selected_stress_terms, stress_ranking = select_stress_terms(
        filtered,
        duration_column="days_to_stop",
        event_column="event",
        group_columns=["Принадлежность", "pump_family"],
        min_group_size=20,
    )
    weibull_result, weibull_errors = fit_weibull_with_fallback(filtered, selected_stress_terms)

    baseline_portfolio = portfolio.copy()
    scenario_portfolio = propagate_scenario_state(
        baseline_portfolio,
        scenario_frequency_hz=float(args.scenario_frequency),
        qliq_model=qliq_model,
        load_mean_model=load_mean_model,
        load_std_model=load_std_model,
        pressure_model=pressure_model,
    )

    baseline_portfolio["catboost_ttf_days"] = predict_regressor(ttf_model, baseline_portfolio)
    scenario_portfolio["catboost_ttf_days"] = predict_regressor(ttf_model, scenario_portfolio)
    baseline_portfolio["catboost_fail_prob_30d"] = predict_binary_proba(failure_model, baseline_portfolio)
    scenario_portfolio["catboost_fail_prob_30d"] = predict_binary_proba(failure_model, scenario_portfolio)

    baseline_mode_probs = predict_multiclass_proba(mode_model, baseline_portfolio)
    scenario_mode_probs = predict_multiclass_proba(mode_model, scenario_portfolio)
    baseline_element_probs = predict_multiclass_proba(element_model, baseline_portfolio)
    scenario_element_probs = predict_multiclass_proba(element_model, scenario_portfolio)

    group_lookup = weibull_group_lookup(weibull_result)
    baseline_weibull_rows: list[dict[str, float]] = []
    scenario_weibull_rows: list[dict[str, float]] = []
    kept_indices: list[int] = []

    for row_index, row in baseline_portfolio.iterrows():
        analysis_row_id = int(row["analysis_row_id"])
        group_key = group_lookup.get(analysis_row_id)
        if group_key is None:
            continue
        kept_indices.append(row_index)
        baseline_weibull_rows.append(weibull_row_summary(weibull_result, group_key, baseline_portfolio.loc[row_index]))
        scenario_weibull_rows.append(weibull_row_summary(weibull_result, group_key, scenario_portfolio.loc[row_index]))

    baseline_portfolio = baseline_portfolio.loc[kept_indices].copy()
    scenario_portfolio = scenario_portfolio.loc[kept_indices].copy()
    baseline_mode_probs = baseline_mode_probs.loc[baseline_portfolio.index].copy()
    scenario_mode_probs = scenario_mode_probs.loc[scenario_portfolio.index].copy()

    baseline_weibull = pd.DataFrame(baseline_weibull_rows, index=baseline_portfolio.index)
    scenario_weibull = pd.DataFrame(scenario_weibull_rows, index=scenario_portfolio.index)
    baseline_portfolio = pd.concat([baseline_portfolio, baseline_weibull.add_prefix("weibull_")], axis=1)
    scenario_portfolio = pd.concat([scenario_portfolio, scenario_weibull.add_prefix("weibull_")], axis=1)

    baseline_portfolio["top_failure_node"] = baseline_mode_probs.idxmax(axis=1)
    scenario_portfolio["top_failure_node"] = scenario_mode_probs.idxmax(axis=1)
    baseline_portfolio["top_failure_element"] = baseline_element_probs.idxmax(axis=1)
    scenario_portfolio["top_failure_element"] = scenario_element_probs.idxmax(axis=1)

    portfolio_report = baseline_portfolio[
        [
            "well_id",
            "anchor_date",
            "Принадлежность",
            "pump_family",
            "freq_30d_mean",
            "qliq_30d_mean",
            "oil_rate_30d_mean",
            "pressure_ratio_bhp_30d",
            "Kpod_30d",
            "load_30d_std",
            "catboost_ttf_days",
            "catboost_fail_prob_30d",
            "weibull_median_ttf_days",
            "weibull_fail_prob_30d",
            "weibull_fail_prob_90d",
            "weibull_fail_prob_365d",
            "top_failure_node",
            "top_failure_element",
        ]
    ].copy()
    portfolio_report = portfolio_report.rename(
        columns={
            "freq_30d_mean": "baseline_freq_30d_mean",
            "qliq_30d_mean": "baseline_qliq_30d_mean",
            "oil_rate_30d_mean": "baseline_oil_rate_30d_mean",
            "pressure_ratio_bhp_30d": "baseline_pressure_ratio_bhp_30d",
            "Kpod_30d": "baseline_Kpod_30d",
            "load_30d_std": "baseline_load_30d_std",
            "catboost_ttf_days": "baseline_catboost_ttf_days",
            "catboost_fail_prob_30d": "baseline_catboost_fail_prob_30d",
            "weibull_median_ttf_days": "baseline_weibull_median_ttf_days",
            "weibull_fail_prob_30d": "baseline_weibull_fail_prob_30d",
            "weibull_fail_prob_90d": "baseline_weibull_fail_prob_90d",
            "weibull_fail_prob_365d": "baseline_weibull_fail_prob_365d",
            "top_failure_node": "baseline_top_failure_node",
            "top_failure_element": "baseline_top_failure_element",
        }
    )
    portfolio_report["scenario_freq_30d_mean"] = scenario_portfolio["freq_30d_mean"].to_numpy()
    portfolio_report["scenario_qliq_frequency_elasticity"] = scenario_portfolio["qliq_frequency_elasticity"].to_numpy()
    portfolio_report["scenario_qliq_mechanistic_linear_hz"] = scenario_portfolio["qliq_mechanistic_linear_hz"].to_numpy()
    portfolio_report["scenario_qliq_30d_mean"] = scenario_portfolio["qliq_30d_mean"].to_numpy()
    portfolio_report["scenario_oil_rate_30d_mean"] = scenario_portfolio["oil_rate_30d_mean"].to_numpy()
    portfolio_report["scenario_pressure_ratio_bhp_30d"] = scenario_portfolio["pressure_ratio_bhp_30d"].to_numpy()
    portfolio_report["scenario_Kpod_30d"] = scenario_portfolio["Kpod_30d"].to_numpy()
    portfolio_report["scenario_load_30d_std"] = scenario_portfolio["load_30d_std"].to_numpy()
    portfolio_report["scenario_catboost_ttf_days"] = scenario_portfolio["catboost_ttf_days"].to_numpy()
    portfolio_report["scenario_catboost_fail_prob_30d"] = scenario_portfolio["catboost_fail_prob_30d"].to_numpy()
    portfolio_report["scenario_weibull_median_ttf_days"] = scenario_portfolio["weibull_median_ttf_days"].to_numpy()
    portfolio_report["scenario_weibull_fail_prob_30d"] = scenario_portfolio["weibull_fail_prob_30d"].to_numpy()
    portfolio_report["scenario_weibull_fail_prob_90d"] = scenario_portfolio["weibull_fail_prob_90d"].to_numpy()
    portfolio_report["scenario_weibull_fail_prob_365d"] = scenario_portfolio["weibull_fail_prob_365d"].to_numpy()
    portfolio_report["scenario_top_failure_node"] = scenario_portfolio["top_failure_node"].to_numpy()
    portfolio_report["scenario_top_failure_element"] = scenario_portfolio["top_failure_element"].to_numpy()

    portfolio_report["delta_qliq_pct"] = 100.0 * (
        (portfolio_report["scenario_qliq_30d_mean"] - portfolio_report["baseline_qliq_30d_mean"])
        / portfolio_report["baseline_qliq_30d_mean"].replace(0.0, np.nan)
    )
    portfolio_report["delta_oil_pct"] = 100.0 * (
        (portfolio_report["scenario_oil_rate_30d_mean"] - portfolio_report["baseline_oil_rate_30d_mean"])
        / portfolio_report["baseline_oil_rate_30d_mean"].replace(0.0, np.nan)
    )
    portfolio_report["delta_catboost_ttf_pct"] = 100.0 * (
        (portfolio_report["scenario_catboost_ttf_days"] - portfolio_report["baseline_catboost_ttf_days"])
        / portfolio_report["baseline_catboost_ttf_days"].replace(0.0, np.nan)
    )
    portfolio_report["delta_catboost_fail_prob_30d_pct"] = 100.0 * (
        (portfolio_report["scenario_catboost_fail_prob_30d"] - portfolio_report["baseline_catboost_fail_prob_30d"])
        / portfolio_report["baseline_catboost_fail_prob_30d"].replace(0.0, np.nan)
    )
    portfolio_report["delta_weibull_median_ttf_pct"] = 100.0 * (
        (portfolio_report["scenario_weibull_median_ttf_days"] - portfolio_report["baseline_weibull_median_ttf_days"])
        / portfolio_report["baseline_weibull_median_ttf_days"].replace(0.0, np.nan)
    )
    portfolio_report["delta_weibull_fail_prob_30d_pct"] = 100.0 * (
        (portfolio_report["scenario_weibull_fail_prob_30d"] - portfolio_report["baseline_weibull_fail_prob_30d"])
        / portfolio_report["baseline_weibull_fail_prob_30d"].replace(0.0, np.nan)
    )

    top_nodes_baseline = summarize_mode_expectations(
        baseline_portfolio["catboost_fail_prob_30d"],
        baseline_mode_probs,
        label="failure_node",
    )
    top_nodes_scenario = summarize_mode_expectations(
        scenario_portfolio["catboost_fail_prob_30d"],
        scenario_mode_probs,
        label="failure_node",
    )
    top_elements_baseline = summarize_mode_expectations(
        baseline_portfolio["catboost_fail_prob_30d"],
        baseline_element_probs,
        label="failure_element",
    )
    top_elements_scenario = summarize_mode_expectations(
        scenario_portfolio["catboost_fail_prob_30d"],
        scenario_element_probs,
        label="failure_element",
    )
    elasticity_summary = summarize_elasticity_levels(qliq_model, field_code=args.field)
    elasticity_summary["global_a"] = float(global_qliq_model.global_elasticity)

    summary = {
        "baseline_total_qliq": float(baseline_portfolio["qliq_30d_mean"].sum()),
        "scenario_total_qliq": float(scenario_portfolio["qliq_30d_mean"].sum()),
        "baseline_total_oil": float(baseline_portfolio["oil_rate_30d_mean"].sum()),
        "scenario_total_oil": float(scenario_portfolio["oil_rate_30d_mean"].sum()),
        "catboost_mean_ttf_base": float(baseline_portfolio["catboost_ttf_days"].mean()),
        "catboost_mean_ttf_scenario": float(scenario_portfolio["catboost_ttf_days"].mean()),
        "catboost_median_ttf_base": float(baseline_portfolio["catboost_ttf_days"].median()),
        "catboost_median_ttf_scenario": float(scenario_portfolio["catboost_ttf_days"].median()),
        "catboost_expected_failures_30d_base": float(baseline_portfolio["catboost_fail_prob_30d"].sum()),
        "catboost_expected_failures_30d_scenario": float(scenario_portfolio["catboost_fail_prob_30d"].sum()),
        "weibull_mean_median_ttf_base": float(baseline_portfolio["weibull_median_ttf_days"].mean()),
        "weibull_mean_median_ttf_scenario": float(scenario_portfolio["weibull_median_ttf_days"].mean()),
        "weibull_expected_failures_30d_base": float(baseline_portfolio["weibull_fail_prob_30d"].sum()),
        "weibull_expected_failures_30d_scenario": float(scenario_portfolio["weibull_fail_prob_30d"].sum()),
        "weibull_expected_failures_90d_base": float(baseline_portfolio["weibull_fail_prob_90d"].sum()),
        "weibull_expected_failures_90d_scenario": float(scenario_portfolio["weibull_fail_prob_90d"].sum()),
        "weibull_expected_failures_365d_base": float(baseline_portfolio["weibull_fail_prob_365d"].sum()),
        "weibull_expected_failures_365d_scenario": float(scenario_portfolio["weibull_fail_prob_365d"].sum()),
    }
    summary["delta_total_qliq_pct"] = 100.0 * (summary["scenario_total_qliq"] - summary["baseline_total_qliq"]) / max(summary["baseline_total_qliq"], 1e-9)
    summary["delta_total_oil_pct"] = 100.0 * (summary["scenario_total_oil"] - summary["baseline_total_oil"]) / max(summary["baseline_total_oil"], 1e-9)
    summary["catboost_mean_ttf_delta_pct"] = 100.0 * (summary["catboost_mean_ttf_scenario"] - summary["catboost_mean_ttf_base"]) / max(summary["catboost_mean_ttf_base"], 1e-9)
    summary["catboost_median_ttf_delta_pct"] = 100.0 * (summary["catboost_median_ttf_scenario"] - summary["catboost_median_ttf_base"]) / max(summary["catboost_median_ttf_base"], 1e-9)
    summary["catboost_expected_failures_30d_delta_pct"] = 100.0 * (summary["catboost_expected_failures_30d_scenario"] - summary["catboost_expected_failures_30d_base"]) / max(summary["catboost_expected_failures_30d_base"], 1e-9)
    summary["weibull_mean_median_ttf_delta_pct"] = 100.0 * (summary["weibull_mean_median_ttf_scenario"] - summary["weibull_mean_median_ttf_base"]) / max(summary["weibull_mean_median_ttf_base"], 1e-9)
    summary["weibull_expected_failures_30d_delta_pct"] = 100.0 * (summary["weibull_expected_failures_30d_scenario"] - summary["weibull_expected_failures_30d_base"]) / max(summary["weibull_expected_failures_30d_base"], 1e-9)
    summary["weibull_expected_failures_90d_delta_pct"] = 100.0 * (summary["weibull_expected_failures_90d_scenario"] - summary["weibull_expected_failures_90d_base"]) / max(summary["weibull_expected_failures_90d_base"], 1e-9)
    summary["weibull_expected_failures_365d_delta_pct"] = 100.0 * (summary["weibull_expected_failures_365d_scenario"] - summary["weibull_expected_failures_365d_base"]) / max(summary["weibull_expected_failures_365d_base"], 1e-9)

    pd.DataFrame({"filter_step": filter_notes}).to_csv(output_dir / "quality_filter_steps.csv", index=False, encoding="utf-8-sig")
    feature_usage_table(feature_sets).to_csv(output_dir / "feature_sets.csv", index=False, encoding="utf-8-sig")
    source_summary.to_csv(output_dir / "dynamic_source_coverage.csv", index=False, encoding="utf-8-sig")
    qliq_model.coefficients.to_csv(output_dir / "qliq_frequency_response_coefficients.csv", index=False, encoding="utf-8-sig")
    portfolio_report.to_csv(output_dir / "portfolio_scenario_predictions.csv", index=False, encoding="utf-8-sig")
    top_nodes_baseline.to_csv(output_dir / "top_failure_nodes_baseline.csv", index=False, encoding="utf-8-sig")
    top_nodes_scenario.to_csv(output_dir / "top_failure_nodes_scenario.csv", index=False, encoding="utf-8-sig")
    top_elements_baseline.to_csv(output_dir / "top_failure_elements_baseline.csv", index=False, encoding="utf-8-sig")
    top_elements_scenario.to_csv(output_dir / "top_failure_elements_scenario.csv", index=False, encoding="utf-8-sig")
    if not stress_ranking.empty:
        stress_ranking.to_csv(output_dir / "weibull_stress_ranking.csv", index=False, encoding="utf-8-sig")
    if selected_stress_terms:
        pd.DataFrame(selected_stress_terms).to_csv(output_dir / "weibull_selected_stress_terms.csv", index=False, encoding="utf-8-sig")

    metrics_payload = {
        "qliq_frequency_response": qliq_model.metrics,
        "qliq_frequency_response_all_fields": global_qliq_model.metrics,
        "load_mean_response": load_mean_model.metrics,
        "load_std_response": load_std_model.metrics,
        "pressure_response": pressure_model.metrics,
        "catboost_ttf": ttf_model.metrics,
        "catboost_failure_30d": failure_model.metrics,
        "catboost_failure_mode": mode_model.metrics,
        "catboost_failure_element": element_model.metrics,
        "elasticity_summary": elasticity_summary,
        "weibull_group_columns": weibull_result.group_columns,
        "weibull_stage_summaries": weibull_result.to_dict()["stage_summaries"],
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scenario_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    render_report(
        output_dir=output_dir,
        field_code=args.field,
        scenario_frequency_hz=float(args.scenario_frequency),
        cutoff_date=cutoff_date,
        source_rows=int(len(rolling)),
        filtered_rows=int(len(filtered)),
        portfolio_rows=int(len(portfolio_report)),
        portfolio_wells=int(portfolio_report["well_id"].nunique()),
        source_summary=source_summary,
        feature_sets=feature_sets,
        global_qliq_model=global_qliq_model,
        qliq_model=qliq_model,
        load_mean_model=load_mean_model,
        load_std_model=load_std_model,
        pressure_model=pressure_model,
        ttf_model=ttf_model,
        failure_model=failure_model,
        mode_model=mode_model,
        element_model=element_model,
        weibull_result=weibull_result,
        weibull_errors=weibull_errors,
        stress_ranking=stress_ranking,
        selected_stress_terms=selected_stress_terms,
        summary=summary,
        elasticity_summary=elasticity_summary,
        top_nodes_baseline=top_nodes_baseline,
        top_nodes_scenario=top_nodes_scenario,
        top_elements_baseline=top_elements_baseline,
        top_elements_scenario=top_elements_scenario,
    )

    print(f"Scenario analysis written to: {output_dir}")
    print(f"Filtered rolling rows: {len(filtered)}")
    print(f"Scenario portfolio wells: {portfolio_report['well_id'].nunique()}")
    print(f"CatBoost expected 30d failures: {summary['catboost_expected_failures_30d_base']:.2f} -> {summary['catboost_expected_failures_30d_scenario']:.2f}")
    print(f"Weibull expected 30d failures: {summary['weibull_expected_failures_30d_base']:.2f} -> {summary['weibull_expected_failures_30d_scenario']:.2f}")
    print(f"Total liquid rate: {summary['baseline_total_qliq']:.1f} -> {summary['scenario_total_qliq']:.1f} m3/d")


if __name__ == "__main__":
    main()
