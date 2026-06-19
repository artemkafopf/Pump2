from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

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

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns
from analysis.input_paths import resolve_v03_all_path
from analysis.modeling_config import (
    DEFAULT_GLF_THRESHOLD,
    DEFAULT_INFANT_MORTALITY_DAYS,
    stable_hash_test_mask,
)
from analysis.sqlite_paths import resolve_techregime_db_path
from scripts.data_utils import (
    CANONICAL_COLUMNS,
    _numeric as _shared_numeric,
    load_daily_merged,
    source_coverage_report,
    split_by_run_id,
)

from catboost import CatBoostClassifier, Pool


ALL_PATH = resolve_v03_all_path()
TR_DB_PATH = resolve_techregime_db_path()

TR_QUERY_COLUMNS = {
    "freq": "col_0047",
    "load": "col_0048",
    "rpl": "col_0061",
    "rpump_intake": "col_0062",
    "rzab": "col_0063",
    "qliq": "col_0064",
    "watercut": "col_0065",
    "gas_factor": "col_0067",
    "qgas": "col_0068",
    "kprod": "col_0069",
}
TR_FEATURE_COLUMNS = list(TR_QUERY_COLUMNS)

STATIC_CATEGORICAL_COLUMNS = [
    "Месторождение",
    "Принадлежность",
    "Тип УЭЦН",
]

STATIC_NUMERIC_COLUMNS = [
    "Ном. Произв. м₃/сут",
    "Ном.напор (50Гц)",
    "Номинальная частота, Гц",
    "Мощность, кВт",
    "Ном. ток/ A",
    "Глубина спуска УЭЦН, по НКТ",
    "ВГ, м",
    "Дав. Нас",
    "Кол.ступеней",
    "Работа в кривизне",
]

STATIC_DERIVED_COLUMNS = [
    "submergence_margin_m",
    "submergence_margin_ratio",
    "nominal_head_per_stage",
    "curve_work_per_meter",
]


@dataclass(slots=True)
class DatasetSpec:
    name: str
    mode: str
    horizon_days: int
    lookback_days: int = 30
    step_days: int = 30
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS


def _numeric(series: pd.Series) -> pd.Series:
    return _shared_numeric(series)


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Свод").reset_index(names="row_id")
    df["event"] = _numeric(df["Failure Flag"])
    df["Дата монтажа"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    df["Дата остановки"] = pd.to_datetime(df["Дата остановки"], errors="coerce")
    df["Скв."] = df["Скв."].astype("string").str.strip()
    df = df.loc[df["event"].isin([0, 1])].copy()
    df = df.loc[df["Скв."].notna() & df["Дата монтажа"].notna() & df["Дата остановки"].notna()].copy()
    df = df.loc[df["Дата остановки"] > df["Дата монтажа"]].copy()
    df["run_days"] = (df["Дата остановки"] - df["Дата монтажа"]).dt.days.astype(float)

    derived_presets = [preset for preset in inspect_derived_presets(df) if preset.available]
    derived_specs = [{"name": preset.name, "formula": preset.formula} for preset in derived_presets]
    if derived_specs:
        df, _ = apply_derived_columns(df, derived_specs)

    dedup_subset = ["Скв.", "Дата монтажа", "Дата остановки", "event"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
    if before != len(df):
        print(f"Dropped duplicate usable runs: {before} -> {len(df)}")
    return df


def load_techregime_daily(wells: list[str]) -> pd.DataFrame:
    if not wells:
        return pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])

    query_select = ", ".join([f"{storage} as {alias}" for alias, storage in TR_QUERY_COLUMNS.items()])
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TR_DB_PATH) as connection:
        for start in range(0, len(wells), 400):
            chunk = wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT col_0003 as well_id, col_0010 as dt, {query_select}
                FROM techregime_records
                WHERE col_0003 IN ({placeholders})
            """
            frames.append(pd.read_sql_query(query, connection, params=chunk))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True, errors="coerce")
    df["well_id"] = df["well_id"].astype("string").str.strip()
    for column in TR_FEATURE_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_id"].notna() & df["dt"].notna()].copy()
    # Use unique-date means so duplicate rows do not overweight the same day.
    df = (
        df.groupby(["well_id", "dt"], as_index=False)[TR_FEATURE_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return df


def derive_daily_features(daily_df: pd.DataFrame, run_row: dict[str, object]) -> pd.DataFrame:
    derived = daily_df.copy()
    nominal_rate = _numeric(pd.Series([run_row.get("Ном. Произв. м₃/сут")])).iloc[0]
    nominal_freq = _numeric(pd.Series([run_row.get("Номинальная частота, Гц")])).iloc[0]
    pbubble = _numeric(pd.Series([run_row.get("Дав. Нас")])).iloc[0]

    derived["kpod_daily"] = np.nan
    if pd.notna(nominal_rate) and abs(float(nominal_rate)) > 1e-12:
        derived["kpod_daily"] = _numeric(derived["qliq"]) / float(nominal_rate)

    derived["kpod_freq_daily"] = np.nan
    freq = _numeric(derived["freq"])
    valid_kpod_freq = derived["kpod_daily"].notna() & freq.notna() & (freq.abs() > 1e-12) & pd.notna(nominal_freq)
    if pd.notna(nominal_freq):
        derived.loc[valid_kpod_freq, "kpod_freq_daily"] = derived.loc[valid_kpod_freq, "kpod_daily"] * (float(nominal_freq) / freq.loc[valid_kpod_freq])

    derived["pressure_ratio_daily"] = np.nan
    if pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
        derived["pressure_ratio_daily"] = _numeric(derived["rzab"]) / float(pbubble)
    derived["neg_excess_kpod_daily"] = np.maximum(0.8 - _numeric(derived["kpod_daily"]), 0.0)
    derived["high_glf_low_pzab_daily"] = (
        (_numeric(derived["gas_factor"]) > DEFAULT_GLF_THRESHOLD)
        & (_numeric(derived["pressure_ratio_daily"]) < 1.0)
    ).astype(float)
    return derived


def summarize_threshold_features(
    source_df: pd.DataFrame,
    *,
    prefix: str,
    load_p90: float | None = None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if source_df.empty:
        return result

    kpod_daily = _numeric(source_df.get("kpod_daily", pd.Series(dtype=float)))
    kpod_freq_daily = _numeric(source_df.get("kpod_freq_daily", pd.Series(dtype=float)))
    pressure_ratio_daily = _numeric(source_df.get("pressure_ratio_daily", pd.Series(dtype=float)))
    gas_factor = _numeric(source_df.get("gas_factor", pd.Series(dtype=float)))
    load = _numeric(source_df.get("load", pd.Series(dtype=float)))
    neg_excess = _numeric(source_df.get("neg_excess_kpod_daily", pd.Series(dtype=float)))

    if kpod_freq_daily.notna().any():
        result[f"kpod_freq_{prefix}_mean"] = float(kpod_freq_daily.mean())
    if kpod_daily.notna().any():
        result[f"frac_kpod_below_0p7_{prefix}"] = float((kpod_daily < 0.7).mean())
    if pressure_ratio_daily.notna().any():
        result[f"frac_pzab_below_1_{prefix}"] = float((pressure_ratio_daily < 1.0).mean())
    if gas_factor.notna().any():
        result[f"frac_glf_above_thr_{prefix}"] = float((gas_factor > DEFAULT_GLF_THRESHOLD).mean())
    if neg_excess.notna().any():
        result[f"mean_neg_excess_kpod_{prefix}"] = float(neg_excess.mean())

    high_glf_low_pzab = _numeric(source_df.get("high_glf_low_pzab_daily", pd.Series(dtype=float)))
    if high_glf_low_pzab.notna().any() and prefix == "30d":
        result["frac_high_glf_low_pzab_30d"] = float(high_glf_low_pzab.mean())

    if prefix == "30d" and load_p90 is not None and np.isfinite(load_p90) and kpod_freq_daily.notna().any() and load.notna().any():
        valid = load.notna() & kpod_freq_daily.notna()
        if valid.any():
            result["frac_high_load_low_kpod_30d"] = float(((load.loc[valid] > load_p90) & (kpod_freq_daily.loc[valid] < 0.7)).mean())
    return result


def build_window_feature_entry(
    window_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    run_to_date_df: pd.DataFrame,
    run_row: dict[str, object],
    anchor: pd.Timestamp,
    horizon_days: int,
    label: int,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "row_id": int(run_row["row_id"]),
        "well_id": str(run_row["Скв."]),
        "anchor_date": anchor,
        "label": int(label),
        "event": int(run_row["event"]),
        "horizon_days": int(horizon_days),
        "days_since_install": int((anchor - run_row["Дата монтажа"]).days),
        "days_to_stop": int((run_row["Дата остановки"] - anchor).days),
        "tr_days_30d": int(window_df["dt"].nunique()),
        "tr_prev30_days": int(previous_df["dt"].nunique()) if not previous_df.empty else 0,
    }

    for column in STATIC_CATEGORICAL_COLUMNS:
        if column in run_row:
            entry[column] = run_row[column]
    for column in STATIC_NUMERIC_COLUMNS + STATIC_DERIVED_COLUMNS:
        if column in run_row:
            entry[column] = run_row[column]

    for feature in TR_FEATURE_COLUMNS:
        series = _numeric(window_df[feature]).dropna()
        if series.empty:
            continue
        entry[f"{feature}_30d_mean"] = float(series.mean())
        entry[f"{feature}_30d_std"] = float(series.std(ddof=0)) if len(series) > 1 else 0.0
        entry[f"{feature}_30d_min"] = float(series.min())
        entry[f"{feature}_30d_max"] = float(series.max())
        entry[f"{feature}_30d_range"] = float(series.max() - series.min())
        entry[f"{feature}_30d_last"] = float(series.iloc[-1])

        prev_series = _numeric(previous_df[feature]).dropna()
        if not prev_series.empty:
            prev_mean = float(prev_series.mean())
            entry[f"{feature}_prev30_mean"] = prev_mean
            if abs(prev_mean) > 1e-12:
                entry[f"{feature}_30d_to_prev30_ratio"] = float(series.mean() / prev_mean)
            entry[f"{feature}_30d_minus_prev30"] = float(series.mean() - prev_mean)

    load_p90 = None
    run_to_date_load = _numeric(run_to_date_df["load"]).dropna() if "load" in run_to_date_df.columns else pd.Series(dtype=float)
    if not run_to_date_load.empty:
        load_p90 = float(run_to_date_load.quantile(0.9))

    entry.update(summarize_threshold_features(window_df, prefix="30d", load_p90=load_p90))
    entry.update(summarize_threshold_features(run_to_date_df, prefix="rtd"))

    qliq = entry.get("qliq_30d_mean")
    freq = entry.get("freq_30d_mean")
    load = entry.get("load_30d_mean")
    intake = entry.get("rpump_intake_30d_mean")
    rzab = entry.get("rzab_30d_mean")
    pbubble = _numeric(pd.Series([run_row.get("Дав. Нас")])).iloc[0]
    nominal_rate = _numeric(pd.Series([run_row.get("Ном. Произв. м₃/сут")])).iloc[0]
    nominal_freq = _numeric(pd.Series([run_row.get("Номинальная частота, Гц")])).iloc[0]
    power = _numeric(pd.Series([run_row.get("Мощность, кВт")])).iloc[0]

    if pd.notna(qliq) and pd.notna(nominal_rate) and abs(float(nominal_rate)) > 1e-12:
        entry["Kpod_30d"] = float(qliq / nominal_rate)
    if pd.notna(qliq) and pd.notna(nominal_rate) and pd.notna(freq) and pd.notna(nominal_freq):
        if abs(float(nominal_rate)) > 1e-12 and abs(float(freq)) > 1e-12:
            entry["Kpod_freq_30d"] = float((qliq / nominal_rate) * (nominal_freq / freq))
    if pd.notna(intake) and pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
        entry["pressure_ratio_intake_30d"] = float(intake / pbubble)
    if pd.notna(rzab) and pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
        entry["pressure_ratio_bhp_30d"] = float(rzab / pbubble)
        entry["pressure_margin_to_bubble_30d"] = float(rzab - pbubble)
    if pd.notna(qliq) and pd.notna(power) and abs(float(power)) > 1e-12:
        entry["qliq_per_kw_30d"] = float(qliq / power)
    if pd.notna(qliq) and pd.notna(freq) and abs(float(freq)) > 1e-12:
        entry["qliq_per_hz_30d"] = float(qliq / freq)
    if pd.notna(load) and pd.notna(freq) and abs(float(freq)) > 1e-12:
        entry["motor_load_per_hz_30d"] = float(load / freq)
    if pd.notna(freq) and pd.notna(nominal_freq) and abs(float(nominal_freq)) > 1e-12:
        entry["frequency_to_reference_ratio_30d"] = float(freq / nominal_freq)
        entry["frequency_over_reference_hz_30d"] = float(freq - nominal_freq)

    kpod_prev = _numeric(previous_df.get("kpod_daily", pd.Series(dtype=float))).dropna()
    if "Kpod_30d" in entry and not kpod_prev.empty and abs(float(kpod_prev.mean())) > 1e-12:
        entry["kpod_30d_to_prev30_ratio"] = float(entry["Kpod_30d"] / float(kpod_prev.mean()))

    load_prev = _numeric(previous_df.get("load", pd.Series(dtype=float))).dropna()
    if "load_30d_mean" in entry and not load_prev.empty and abs(float(load_prev.mean())) > 1e-12:
        entry["load_30d_to_prev30_ratio"] = float(entry["load_30d_mean"] / float(load_prev.mean()))

    pressure_prev = _numeric(previous_df.get("pressure_ratio_daily", pd.Series(dtype=float))).dropna()
    if "pressure_ratio_bhp_30d" in entry and not pressure_prev.empty and abs(float(pressure_prev.mean())) > 1e-12:
        entry["pzab_30d_to_prev30_ratio"] = float(entry["pressure_ratio_bhp_30d"] / float(pressure_prev.mean()))
    return entry


def build_dataset(spec: DatasetSpec, runs: pd.DataFrame, tr_daily: pd.DataFrame) -> pd.DataFrame:
    tr_by_well = {well: frame.sort_values("dt").reset_index(drop=True) for well, frame in tr_daily.groupby("well_id")}
    records: list[dict[str, object]] = []
    lookback = pd.Timedelta(days=spec.lookback_days)
    step = pd.Timedelta(days=spec.step_days)
    horizon = pd.Timedelta(days=spec.horizon_days)

    for run_row in runs.to_dict(orient="records"):
        well = str(run_row["Скв."])
        source = tr_by_well.get(well)
        if source is None:
            continue

        run_df = source.loc[
            (source["dt"] >= run_row["Дата монтажа"])
            & (source["dt"] <= run_row["Дата остановки"])
        ].copy()
        if run_df.empty:
            continue
        run_df = derive_daily_features(run_df, run_row)

        if spec.mode == "endpoint":
            anchors = [run_row["Дата остановки"] - horizon]
        else:
            first_anchor = run_row["Дата монтажа"] + lookback - pd.Timedelta(days=1)
            last_anchor = run_row["Дата остановки"] - pd.Timedelta(days=1)
            anchors = []
            current = first_anchor
            while current <= last_anchor:
                anchors.append(current)
                current += step

        for anchor in anchors:
            infant_cutoff = run_row["Дата монтажа"] + pd.Timedelta(days=spec.infant_mortality_days)
            if anchor < infant_cutoff:
                continue
            window_start = anchor - lookback + pd.Timedelta(days=1)
            window_df = run_df.loc[(run_df["dt"] >= window_start) & (run_df["dt"] <= anchor)].copy()
            if window_df["dt"].nunique() < 10:
                continue

            future_days = int((run_row["Дата остановки"] - anchor).days)
            if future_days <= 0:
                continue

            if int(run_row["event"]) == 1:
                label = int(future_days <= spec.horizon_days)
            else:
                if future_days < spec.horizon_days:
                    continue
                label = 0

            if spec.mode == "endpoint" and int(run_row["event"]) == 1:
                label = 1

            previous_end = window_start - pd.Timedelta(days=1)
            previous_start = previous_end - lookback + pd.Timedelta(days=1)
            previous_df = run_df.loc[(run_df["dt"] >= previous_start) & (run_df["dt"] <= previous_end)].copy()
            run_to_date_start = infant_cutoff
            run_to_date_df = run_df.loc[(run_df["dt"] >= run_to_date_start) & (run_df["dt"] <= anchor)].copy()

            records.append(build_window_feature_entry(window_df, previous_df, run_to_date_df, run_row, anchor, spec.horizon_days, label))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    prepared = pd.DataFrame(index=df.index)
    categorical_columns: list[str] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        series = df[column]
        numeric = _numeric(series)
        if int(numeric.notna().sum()) >= max(20, int(series.notna().sum() * 0.6)):
            prepared[column] = numeric
        else:
            prepared[column] = series.astype("string").fillna("__missing__")
            categorical_columns.append(column)
    usable_columns = []
    usable_cats: list[str] = []
    for column in prepared.columns:
        non_missing = int(prepared[column].notna().sum())
        if non_missing < 20:
            continue
        if prepared[column].dtype.name == "string":
            usable_cats.append(column)
        usable_columns.append(column)
    prepared = prepared[usable_columns]
    return prepared, usable_cats


def make_stratified_group_folds(
    df: pd.DataFrame,
    group_column: str,
    label_column: str,
    n_splits: int = 5,
) -> list[np.ndarray]:
    group_frame = (
        df.groupby(group_column, dropna=False)[label_column]
        .agg(["sum", "count"])
        .rename(columns={"sum": "positives", "count": "rows"})
        .reset_index()
    )
    group_frame["group_key"] = group_frame[group_column].astype("string").fillna("__missing__")
    group_frame = group_frame.sort_values(["positives", "rows", "group_key"], ascending=[False, False, True]).reset_index(drop=True)

    fold_rows = [0 for _ in range(n_splits)]
    fold_positives = [0.0 for _ in range(n_splits)]
    assignments: dict[str, int] = {}

    total_rows = float(group_frame["rows"].sum())
    total_positives = float(group_frame["positives"].sum())
    target_rows = total_rows / n_splits if n_splits else total_rows
    target_positives = total_positives / n_splits if n_splits else total_positives

    for row in group_frame.itertuples(index=False):
        best_fold = 0
        best_score: float | None = None
        for fold_index in range(n_splits):
            new_rows = fold_rows[fold_index] + int(row.rows)
            new_positives = fold_positives[fold_index] + float(row.positives)
            row_penalty = ((new_rows - target_rows) / max(target_rows, 1.0)) ** 2
            if target_positives > 0:
                pos_penalty = ((new_positives - target_positives) / target_positives) ** 2
            else:
                pos_penalty = 0.0
            score = row_penalty + (2.5 * pos_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_index
        assignments[str(row.group_key)] = best_fold
        fold_rows[best_fold] += int(row.rows)
        fold_positives[best_fold] += float(row.positives)

    group_keys = df[group_column].astype("string").fillna("__missing__")
    folds: list[np.ndarray] = []
    for fold_index in range(n_splits):
        mask = group_keys.map(lambda value: assignments.get(str(value), -1) == fold_index).to_numpy(dtype=bool)
        folds.append(df.index.to_numpy()[mask])
    return folds


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_true.sum()
    negatives = len(y_true) - positives
    if positives <= 0 or negatives <= 0:
        return None
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    ranks = np.empty(len(y_score), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end
    positive_rank_sum = ranks[sorted_true > 0.5].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = int(y_true.sum())
    if positives <= 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    cumulative_true = np.cumsum(y_true)
    ranks = np.arange(1, len(y_true) + 1)
    precision = cumulative_true / ranks
    ap = float((precision * y_true).sum() / positives)
    return ap


def logloss_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    eps = 1e-12
    probs = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    truth = np.asarray(y_true, dtype=float)
    return float(-(truth * np.log(probs) + (1.0 - truth) * np.log(1.0 - probs)).mean())


def brier_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    probs = np.asarray(y_score, dtype=float)
    return float(np.mean((probs - truth) ** 2))


def balanced_accuracy_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float | None:
    truth = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    pos_mask = truth == 1
    neg_mask = truth == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    tpr = float((pred[pos_mask] == 1).mean())
    tnr = float((pred[neg_mask] == 0).mean())
    return 0.5 * (tpr + tnr)


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    truth = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_score, dtype=float)
    base_rate = float(truth.mean())
    if len(truth) == 0 or base_rate <= 0:
        return None
    cutoff = max(1, int(math.ceil(len(truth) * 0.10)))
    order = np.argsort(-probs)
    top_rate = float(truth[order][:cutoff].mean())
    return float(top_rate / base_rate) if base_rate > 0 else None


def evaluate_catboost_cv(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    group_column: str = "row_id",
    label_column: str = "label",
    n_splits: int = 5,
) -> dict[str, object]:
    working = dataset.loc[dataset[label_column].isin([0, 1])].copy()
    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    if prepared.empty:
        return {"error": "No usable features after preparation."}

    aligned = working.loc[prepared.index].copy()
    y = aligned[label_column].astype(int).to_numpy()
    folds = make_stratified_group_folds(aligned, group_column, label_column, n_splits=n_splits)

    probabilities = pd.Series(index=aligned.index, dtype=float)
    fold_metrics: list[dict[str, float | int | None]] = []
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]

    for fold_number, test_index in enumerate(folds, start=1):
        if len(test_index) == 0:
            continue
        test_mask = aligned.index.isin(test_index)
        train_mask = ~test_mask
        train_x = prepared.loc[train_mask]
        train_y = aligned.loc[train_mask, label_column].astype(int)
        test_x = prepared.loc[test_mask]
        test_y = aligned.loc[test_mask, label_column].astype(int)

        if train_y.nunique(dropna=True) < 2 or test_y.nunique(dropna=True) < 2:
            continue

        scale_pos_weight = max(1.0, float((train_y == 0).sum()) / max(float((train_y == 1).sum()), 1.0))
        model = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42 + fold_number,
            verbose=False,
            allow_writing_files=False,
            class_weights=[1.0, scale_pos_weight],
        )
        model.fit(Pool(train_x, train_y, cat_features=cat_indices))
        proba = model.predict_proba(test_x)[:, 1]
        probabilities.loc[test_x.index] = proba

        fold_metrics.append(
            {
                "fold": fold_number,
                "rows": int(len(test_y)),
                "positives": int(test_y.sum()),
                "base_rate": float(test_y.mean()),
                "roc_auc": roc_auc_score_binary(test_y.to_numpy(), proba),
                "average_precision": average_precision_score_binary(test_y.to_numpy(), proba),
                "logloss": logloss_binary(test_y.to_numpy(), proba),
                "brier": brier_score_binary(test_y.to_numpy(), proba),
                "balanced_accuracy": balanced_accuracy_binary(test_y.to_numpy(), proba),
                "lift_top_decile": top_decile_lift(test_y.to_numpy(), proba),
            }
        )

    valid_mask = probabilities.notna()
    if not valid_mask.any():
        return {"error": "Cross-validation produced no valid holdout predictions."}
    holdout_y = aligned.loc[valid_mask, label_column].astype(int).to_numpy()
    holdout_p = probabilities.loc[valid_mask].to_numpy(dtype=float)

    return {
        "rows": int(len(aligned)),
        "positives": int(aligned[label_column].sum()),
        "base_rate": float(aligned[label_column].mean()),
        "feature_count": int(prepared.shape[1]),
        "categorical_feature_count": int(len(categorical_columns)),
        "holdout_rows": int(valid_mask.sum()),
        "roc_auc": roc_auc_score_binary(holdout_y, holdout_p),
        "average_precision": average_precision_score_binary(holdout_y, holdout_p),
        "logloss": logloss_binary(holdout_y, holdout_p),
        "brier": brier_score_binary(holdout_y, holdout_p),
        "balanced_accuracy": balanced_accuracy_binary(holdout_y, holdout_p),
        "lift_top_decile": top_decile_lift(holdout_y, holdout_p),
        "fold_metrics": fold_metrics,
        "prepared_columns": prepared.columns.tolist(),
        "categorical_columns": categorical_columns,
    }


def evaluate_time_holdout(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split_date: pd.Timestamp,
) -> dict[str, object]:
    working = dataset.loc[dataset["label"].isin([0, 1])].copy()
    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    if prepared.empty:
        return {"error": "No usable features after preparation."}
    aligned = working.loc[prepared.index].copy()

    train_mask = aligned["anchor_date"] < split_date
    test_mask = aligned["anchor_date"] >= split_date
    if int(train_mask.sum()) < 200 or int(test_mask.sum()) < 100:
        return {"error": f"Time split too small at {split_date.date()}."}

    train_y = aligned.loc[train_mask, "label"].astype(int)
    test_y = aligned.loc[test_mask, "label"].astype(int)
    if train_y.nunique(dropna=True) < 2 or test_y.nunique(dropna=True) < 2:
        return {"error": f"Time split at {split_date.date()} lacks both classes."}

    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model = CatBoostClassifier(
        iterations=250,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, max(1.0, float((train_y == 0).sum()) / max(float((train_y == 1).sum()), 1.0))],
    )
    model.fit(Pool(prepared.loc[train_mask], train_y, cat_features=cat_indices))
    proba = model.predict_proba(prepared.loc[test_mask])[:, 1]
    y_true = test_y.to_numpy()
    return {
        "split_date": str(split_date.date()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_base_rate": float(train_y.mean()),
        "test_base_rate": float(test_y.mean()),
        "roc_auc": roc_auc_score_binary(y_true, proba),
        "average_precision": average_precision_score_binary(y_true, proba),
        "logloss": logloss_binary(y_true, proba),
        "brier": brier_score_binary(y_true, proba),
        "balanced_accuracy": balanced_accuracy_binary(y_true, proba),
        "lift_top_decile": top_decile_lift(y_true, proba),
        "calibration_note": "Probability calibration is deferred in Phase 1; grouped CV and calibration diagnostics should be added in Phase 2.",
    }


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    static_features = [column for column in [*STATIC_CATEGORICAL_COLUMNS, *STATIC_NUMERIC_COLUMNS, *STATIC_DERIVED_COLUMNS] if column in df.columns]
    mean_features = sorted([column for column in df.columns if column.endswith("_30d_mean") or column.endswith("_30d_last")])
    variability_features = sorted(
        [column for column in df.columns if column.endswith("_30d_std") or column.endswith("_30d_range")]
    )
    trend_features = sorted(
        [column for column in df.columns if column.endswith("_30d_to_prev30_ratio") or column.endswith("_30d_minus_prev30")]
    )
    extra_window_derived = sorted(
        [
            column
            for column in df.columns
            if column
            in {
                "Kpod_30d",
                "Kpod_freq_30d",
                "pressure_ratio_intake_30d",
                "pressure_ratio_bhp_30d",
                "pressure_margin_to_bubble_30d",
                "qliq_per_kw_30d",
                "qliq_per_hz_30d",
                "motor_load_per_hz_30d",
                "frequency_to_reference_ratio_30d",
                "frequency_over_reference_hz_30d",
            }
        ]
    )
    return {
        "static_only": static_features,
        "static_plus_mean": [*static_features, *mean_features, *extra_window_derived],
        "static_mean_var": [*static_features, *mean_features, *variability_features, *extra_window_derived],
        "static_mean_var_trend": [*static_features, *mean_features, *variability_features, *trend_features, *extra_window_derived],
    }


def top_binary_correlations(df: pd.DataFrame, target_column: str, feature_columns: list[str], top_n: int = 15) -> list[dict[str, object]]:
    target = _numeric(df[target_column])
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        series = _numeric(df[column])
        valid = target.notna() & series.notna()
        if int(valid.sum()) < 80:
            continue
        corr = series[valid].corr(target[valid])
        if pd.notna(corr):
            rows.append({"feature": column, "rows": int(valid.sum()), "correlation": float(corr)})
    rows.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return rows[:top_n]


def fit_feature_importance(df: pd.DataFrame, target_column: str, feature_columns: list[str], top_n: int = 15) -> list[dict[str, object]]:
    working = df.loc[df[target_column].isin([0, 1])].copy()
    prepared, categorical_columns = prepare_feature_matrix(working, feature_columns)
    if prepared.empty:
        return []
    aligned = working.loc[prepared.index].copy()
    y = aligned[target_column].astype(int)
    if y.nunique(dropna=True) < 2:
        return []
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model = CatBoostClassifier(
        iterations=250,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, max(1.0, float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))],
    )
    model.fit(Pool(prepared, y, cat_features=cat_indices))
    frame = pd.DataFrame({"feature": prepared.columns.tolist(), "importance": model.get_feature_importance()})
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    return [{"feature": str(row.feature), "importance": float(row.importance)} for row in frame.itertuples(index=False)]


def empirical_risk_bins(df: pd.DataFrame, feature: str, target_column: str = "label", bins: int = 5) -> list[dict[str, object]]:
    if feature not in df.columns:
        return []
    x = _numeric(df[feature])
    y = _numeric(df[target_column])
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 100:
        return []
    working = pd.DataFrame({"x": x[valid], "y": y[valid]})
    working = working.sort_values("x").reset_index(drop=True)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = sorted(set(float(working["x"].quantile(q)) for q in quantiles))
    if len(edges) < 3:
        return []
    rows: list[dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if left == right:
            continue
        if right == edges[-1]:
            mask = (working["x"] >= left) & (working["x"] <= right)
        else:
            mask = (working["x"] >= left) & (working["x"] < right)
        bucket = working.loc[mask]
        if len(bucket) < 20:
            continue
        rows.append(
            {
                "feature": feature,
                "left": left,
                "right": right,
                "rows": int(len(bucket)),
                "positive_rate": float(bucket["y"].mean()),
                "median_x": float(bucket["x"].median()),
            }
        )
    return rows


def main() -> None:
    runs = load_runs(ALL_PATH)
    print("## Input")
    print(f"Workbook: {ALL_PATH}")
    print(f"Usable runs with event 0/1 and dates: {len(runs)}")
    print(f"Failure runs: {int(runs['event'].sum())}")
    print(f"Non-failure/censored runs: {int((runs['event'] == 0).sum())}")
    print(f"Unique wells: {runs['Скв.'].nunique()}")
    print(f"Install range: {runs['Дата монтажа'].min().date()} -> {runs['Дата монтажа'].max().date()}")
    print(f"Stop range: {runs['Дата остановки'].min().date()} -> {runs['Дата остановки'].max().date()}")

    tr_daily = load_daily_merged(runs["Скв."].dropna().astype(str).unique().tolist())
    print("\n## Dynamic Daily")
    print(f"Merged telemetry-first rows: {len(tr_daily)}")
    print(f"Covered wells: {tr_daily['well_id'].nunique()}")
    print(f"Date range: {tr_daily['dt'].min().date()} -> {tr_daily['dt'].max().date()}")
    coverage = source_coverage_report(tr_daily)
    if not coverage.empty:
        print(f"Mean telemetry fraction: {float(coverage['telemetry_fraction'].mean()):.3f}")

    specs = [
        DatasetSpec(name="endpoint_h30", mode="endpoint", horizon_days=30),
        DatasetSpec(name="rolling_h30", mode="rolling", horizon_days=30),
        DatasetSpec(name="rolling_h60", mode="rolling", horizon_days=60),
        DatasetSpec(name="rolling_h90", mode="rolling", horizon_days=90),
    ]

    built_datasets: dict[str, pd.DataFrame] = {}
    for spec in specs:
        dataset = build_dataset(spec, runs, tr_daily)
        built_datasets[spec.name] = dataset
        print(f"\n## Dataset {spec.name}")
        if dataset.empty:
            print("(empty)")
            continue
        print(f"Rows: {len(dataset)}")
        print(f"Positive rows: {int(dataset['label'].sum())}")
        print(f"Base rate: {float(dataset['label'].mean()):.4f}")
        print(f"Unique wells: {dataset['well_id'].nunique()}")
        print(f"Unique runs: {dataset['row_id'].nunique()}")
        print(f"Anchor range: {dataset['anchor_date'].min().date()} -> {dataset['anchor_date'].max().date()}")

    results: dict[str, object] = {"datasets": {}, "recommendations": {}}

    for dataset_name in ["endpoint_h30", "rolling_h30"]:
        dataset = built_datasets.get(dataset_name)
        if dataset is None or dataset.empty:
            continue
        sets = feature_sets(dataset)
        dataset_results: dict[str, object] = {"feature_sets": {}}
        for feature_set_name, columns in sets.items():
            metrics = evaluate_catboost_cv(dataset, columns)
            dataset_results["feature_sets"][feature_set_name] = metrics
            print(f"\n## CV {dataset_name} :: {feature_set_name}")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        results["datasets"][dataset_name] = dataset_results

    rolling_full_features = feature_sets(built_datasets["rolling_h30"]).get("static_mean_var_trend", [])
    for dataset_name in ["rolling_h60", "rolling_h90"]:
        dataset = built_datasets.get(dataset_name)
        if dataset is None or dataset.empty:
            continue
        metrics = evaluate_catboost_cv(dataset, rolling_full_features)
        results["datasets"][dataset_name] = {"feature_sets": {"static_mean_var_trend": metrics}}
        print(f"\n## CV {dataset_name} :: static_mean_var_trend")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    rolling_h30 = built_datasets.get("rolling_h30")
    if rolling_h30 is not None and not rolling_h30.empty:
        split_date = pd.Timestamp("2024-01-01")
        holdout = evaluate_time_holdout(rolling_h30, rolling_full_features, split_date=split_date)
        results["datasets"].setdefault("rolling_h30", {})["time_holdout_2024"] = holdout
        print("\n## Time Holdout rolling_h30")
        print(json.dumps(holdout, ensure_ascii=False, indent=2))

        correlations = top_binary_correlations(rolling_h30, "label", rolling_full_features, top_n=15)
        importance = fit_feature_importance(rolling_h30, "label", rolling_full_features, top_n=15)
        nomogram_features = [item["feature"] for item in importance[:6]]
        bins = {feature: empirical_risk_bins(rolling_h30, feature) for feature in nomogram_features}
        results["datasets"]["rolling_h30"]["correlations"] = correlations
        results["datasets"]["rolling_h30"]["feature_importance"] = importance
        results["datasets"]["rolling_h30"]["empirical_risk_bins"] = bins
        print("\n## rolling_h30 correlations")
        print(json.dumps(correlations, ensure_ascii=False, indent=2))
        print("\n## rolling_h30 feature importance")
        print(json.dumps(importance, ensure_ascii=False, indent=2))
        print("\n## rolling_h30 empirical risk bins")
        print(json.dumps(bins, ensure_ascii=False, indent=2))

    print("\n## Notes")
    print("1. This classifier uses only static run descriptors plus trailing 30-day telemetry-first merged windows.")
    print("2. Full-run workbook averages such as run-level Qliq/Frequency/Rzab are intentionally excluded to avoid future leakage.")
    print("3. Endpoint classification answers 'does the last observed month before stop look failed vs censored?', while rolling classification answers 'from this anchor, is failure due within H days?'.")


if __name__ == "__main__":
    main()
