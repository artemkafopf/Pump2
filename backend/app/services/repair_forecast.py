from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Dataset, DatasetColumnMatch, TrainedModel
from app.schemas.analysis import (
    RepairForecastMonthlySummary,
    RepairForecastResponse,
    RepairForecastRow,
    RepairForecastSourceDataset,
)
from app.services.analysis import (
    build_dataset_detail,
    coerce_numeric_series,
    load_saved_forecast_model,
    predict_forecast_rows,
    resolve_forecast_columns,
    train_forecast_model,
)

FIELD_PATTERNS = ("месторожд", "field", "field_name")
CLUSTER_PATTERNS = ("куст", "cluster", "pad", "cluster_name")
WELL_PATTERNS = ("скваж", "well", "well id", "well_id", "id скв", "well_name", "скв.№", "скв №")
IDENTIFIER_PATTERNS = ("ун", "well_id", "well id", "id скв", "id well")
NNO_PATTERNS = ("нно", "mtbf", "наработка на отказ", "mean time")
RUNTIME_PATTERNS = ("наработ", "runtime", "отработ", "worked")
DATE_PATTERNS = ("дата", "date", "day", "period")
LIQUID_RATE_PATTERNS = ("дебж", "дебит жид", "liq", "liquid")


@dataclass
class PreparedDataset:
    dataset: Dataset
    rows: list[dict]
    canonical_map: dict[str, str]


def normalize_text(value: object) -> str:
    return str(value or "").strip().casefold().replace("ё", "е")


def find_best_column(columns: Iterable[str], patterns: tuple[str, ...]) -> str | None:
    scored: list[tuple[int, str]] = []
    for column in columns:
        normalized = normalize_text(column)
        score = sum(len(pattern) for pattern in patterns if pattern in normalized)
        if score:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def parse_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        numeric_value = float(numeric)
        if 20000 <= numeric_value <= 80000:
            parsed = pd.to_datetime(numeric_value, errors="coerce", unit="D", origin="1899-12-30")
            if pd.notna(parsed):
                return parsed.date()
        if 1900 <= numeric_value <= 2500 and numeric_value.is_integer():
            return date(int(numeric_value), 1, 1)

    parsed = pd.to_datetime(pd.Series([value], dtype="object"), errors="coerce", dayfirst=True).iloc[0]
    if pd.notna(parsed):
        return parsed.date()
    return None


def to_float(value: object) -> float | None:
    numeric = coerce_numeric_series(pd.Series([value], dtype="object")).iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def stringify(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_canonical_map(matches: list[DatasetColumnMatch]) -> dict[str, str]:
    return {item.source_column: item.canonical_name for item in matches if item.canonical_name}


def canonicalize_row(row: dict, canonical_map: dict[str, str]) -> dict:
    prepared = dict(row)
    for source, canonical in canonical_map.items():
        if source in row and canonical:
            prepared.setdefault(canonical, row[source])
    return prepared


def prepare_dataset(dataset: Dataset) -> PreparedDataset:
    rows = [dict(record.payload) for record in dataset.records]
    canonical_map = build_canonical_map(list(dataset.column_matches))
    prepared_rows = [canonicalize_row(row, canonical_map) for row in rows]
    return PreparedDataset(dataset=dataset, rows=prepared_rows, canonical_map=canonical_map)


def resolve_model_bundle(
    dataset: Dataset,
    model_id: int | None,
    base_feature_columns: list[str],
) -> tuple[dict | None, TrainedModel | None]:
    selected_model = None
    if model_id is not None:
        selected_model = next((model for model in dataset.trained_models if model.id == model_id), None)
    else:
        selected_model = next((model for model in dataset.trained_models if model.is_active), None)

    if selected_model is not None:
        loaded = load_saved_forecast_model(selected_model, dataset, dataset.records)
        if loaded is not None:
            return loaded, selected_model

    trained = train_forecast_model(dataset, dataset.records, base_feature_columns)
    return trained, selected_model


def identify_key_columns(rows: list[dict]) -> dict[str, str | None]:
    columns = list({column for row in rows for column in row.keys()})
    return {
        "field": find_best_column(columns, FIELD_PATTERNS),
        "cluster": find_best_column(columns, CLUSTER_PATTERNS),
        "well": find_best_column(columns, WELL_PATTERNS),
        "identifier": find_best_column(columns, IDENTIFIER_PATTERNS),
        "nno": find_best_column(columns, NNO_PATTERNS),
        "runtime": find_best_column(columns, RUNTIME_PATTERNS),
        "date": find_best_column(columns, DATE_PATTERNS),
        "liquid_rate": find_best_column(columns, LIQUID_RATE_PATTERNS),
    }


def build_lookup_keys(row: dict, key_columns: dict[str, str | None]) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []

    identifier_column = key_columns.get("identifier")
    if identifier_column:
        identifier = normalize_text(row.get(identifier_column))
        if identifier:
            keys.append(("identifier", identifier))

    well = normalize_text(row.get(key_columns["well"])) if key_columns.get("well") else ""
    cluster = normalize_text(row.get(key_columns["cluster"])) if key_columns.get("cluster") else ""

    if well and cluster:
        keys.append(("well_cluster", well, cluster))
    if well:
        keys.append(("well", well))

    return keys


def primary_group_key(row: dict, key_columns: dict[str, str | None]) -> tuple[str, ...]:
    keys = build_lookup_keys(row, key_columns)
    if keys:
        return keys[0]
    return ("row", normalize_text(row))


def best_latest_rows(rows: list[dict], key_columns: dict[str, str | None]) -> dict[tuple[str, ...], dict]:
    dated_rows: dict[tuple[str, ...], tuple[date | None, int, dict]] = {}
    for index, row in enumerate(rows):
        row_date = parse_date(row.get(key_columns["date"])) if key_columns.get("date") else None
        for key in build_lookup_keys(row, key_columns):
            previous = dated_rows.get(key)
            if previous is None or ((row_date or date.min), index) >= ((previous[0] or date.min), previous[1]):
                dated_rows[key] = (row_date, index, row)
    return {key: item[2] for key, item in dated_rows.items()}


def find_matching_fact_row(
    source_row: dict,
    source_keys: dict[str, str | None],
    fact_latest_by_key: dict[tuple[str, ...], dict],
) -> dict | None:
    for key in build_lookup_keys(source_row, source_keys):
        if key in fact_latest_by_key:
            return fact_latest_by_key[key]
    return None


def build_identifier_stats(rows: list[dict], key_columns: dict[str, str | None], feature_columns: list[str]) -> dict[str, dict[str, float]]:
    identifier_column = key_columns.get("identifier")
    if not identifier_column:
        return {}

    frame = pd.DataFrame(rows)
    if frame.empty or identifier_column not in frame.columns:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for identifier_value, group in frame.groupby(identifier_column, dropna=True):
        metrics: dict[str, float] = {}
        for column in feature_columns:
            if column not in group.columns:
                continue
            numeric = coerce_numeric_series(group[column])
            if numeric.notna().any():
                metrics[column] = float(numeric.mean())
        stats[normalize_text(identifier_value)] = metrics
    return stats


def apply_feature_defaults(
    row: dict,
    feature_columns: list[str],
    fact_row: dict | None,
    identifier_stats: dict[str, dict[str, float]],
    key_columns: dict[str, str | None],
    manual_values: dict[str, object],
    baseline_values: dict[str, object],
) -> dict:
    prepared = dict(row)
    identifier_key = normalize_text(prepared.get(key_columns["identifier"])) if key_columns.get("identifier") else ""
    identifier_defaults = identifier_stats.get(identifier_key, {})

    for column in feature_columns:
        value = prepared.get(column)
        if value not in (None, ""):
            continue
        if column in manual_values and manual_values[column] not in (None, ""):
            prepared[column] = manual_values[column]
        elif fact_row and fact_row.get(column) not in (None, ""):
            prepared[column] = fact_row.get(column)
        elif column in identifier_defaults:
            prepared[column] = identifier_defaults[column]
        else:
            prepared[column] = baseline_values.get(column)
    return prepared


def predict_single_row(trained: dict, row: dict) -> float | None:
    prediction = predict_forecast_rows(trained, [row])
    if not prediction or prediction[0] is None:
        return None
    return max(float(prediction[0]), 5.0)


def calculate_first_failure_offset(
    predicted_nno: float | None,
    actual_nno: float | None,
    runtime_days: float | None,
    base_failure_coefficient: float,
) -> int:
    if predicted_nno is None:
        return 1
    if actual_nno is not None and predicted_nno > actual_nno:
        return max(int(round(predicted_nno * base_failure_coefficient)), 1)
    if runtime_days is not None:
        return max(int(round(predicted_nno - runtime_days)), 1)
    return max(int(round(predicted_nno * base_failure_coefficient)), 1)


def align_to_forecast_date(target: date, forecast_dates: list[date]) -> date | None:
    for item in forecast_dates:
        if item >= target:
            return item
    return None


def prediction_for_date(predictions: list[tuple[date, float | None]], target_date: date) -> float | None:
    current = None
    for row_date, prediction in predictions:
        if row_date <= target_date:
            current = prediction
        else:
            break
    if current is not None:
        return current
    return predictions[0][1] if predictions else None


def build_statuses_for_well(
    forecast_dates: list[date],
    prediction_series: list[tuple[date, float | None]],
    actual_nno: float | None,
    runtime_days: float | None,
    base_failure_coefficient: float,
) -> tuple[list[int], list[str], list[tuple[date, float]]]:
    if not forecast_dates:
        return [], [], []

    statuses = [1] * len(forecast_dates)
    if not prediction_series:
        return statuses, [], []

    first_prediction = prediction_series[0][1]
    if first_prediction is None:
        return statuses, [], []

    event_dates: list[str] = []
    event_nnos: list[tuple[date, float]] = []
    date_index = {item: index for index, item in enumerate(forecast_dates)}
    current_anchor = forecast_dates[0]
    next_failure = current_anchor + timedelta(
        days=calculate_first_failure_offset(
            first_prediction,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        )
    )

    while next_failure <= forecast_dates[-1]:
        aligned = align_to_forecast_date(next_failure, forecast_dates)
        if aligned is None:
            break

        index = date_index[aligned]
        statuses[index] = 0
        event_dates.append(aligned.isoformat())

        next_prediction = prediction_for_date(prediction_series, aligned)
        if next_prediction is None:
            break
        event_nnos.append((aligned, next_prediction))

        current_anchor = aligned
        next_failure = current_anchor + timedelta(days=max(int(round(next_prediction)), 1))
        actual_nno = None
        runtime_days = None

    return statuses, event_dates, event_nnos


def build_repair_forecast(
    db: Session,
    fact_dataset: Dataset,
    source_dataset: Dataset,
    model_id: int | None,
    base_failure_coefficient: float,
    nominal_gap_coefficient: float,
    manual_feature_values: dict[str, object],
) -> RepairForecastResponse:
    del db, nominal_gap_coefficient

    fact_prepared = prepare_dataset(fact_dataset)
    source_prepared = prepare_dataset(source_dataset)
    if not source_prepared.rows:
        raise ValueError("В сводпрогнозе нет строк для расчета.")

    feature_columns = resolve_forecast_columns(
        fact_dataset,
        list(fact_dataset.selected_features_json or []),
        pd.DataFrame(fact_prepared.rows, columns=list(fact_dataset.columns_json)),
    )
    trained, selected_model = resolve_model_bundle(fact_dataset, model_id, feature_columns)
    if trained is None:
        raise ValueError("Не удалось подготовить модель CatBoost для прогноза ремонтов.")

    fact_keys = identify_key_columns(fact_prepared.rows)
    source_keys = identify_key_columns(source_prepared.rows)
    date_column = source_keys.get("date")
    liquid_rate_column = source_keys.get("liquid_rate")
    identifier_column = source_keys.get("identifier")
    well_column = source_keys.get("well")

    if not date_column:
        raise ValueError("В сводпрогнозе не найдена колонка с датой.")
    if not identifier_column:
        raise ValueError("В сводпрогнозе не найдена колонка УН.")
    if not well_column:
        raise ValueError("В сводпрогнозе не найдена колонка со скважиной.")

    source_rows_with_dates: list[tuple[date, dict]] = []
    for row in source_prepared.rows:
        row_date = parse_date(row.get(date_column))
        if row_date is not None:
            source_rows_with_dates.append((row_date, row))
    if not source_rows_with_dates:
        raise ValueError("В сводпрогнозе не удалось распознать даты для расчета.")

    source_rows_with_dates.sort(key=lambda item: item[0])
    today = date.today()
    end_date = date(today.year + 1, 12, 31)
    forecast_dates = [today + timedelta(days=offset) for offset in range((end_date - today).days + 1)]

    fact_latest_by_key = best_latest_rows(fact_prepared.rows, fact_keys)
    identifier_stats = build_identifier_stats(fact_prepared.rows, fact_keys, trained["feature_columns"])

    available_columns = {column for row in source_prepared.rows for column in row.keys()}
    missing_feature_columns = [column for column in trained["feature_columns"] if column not in available_columns]

    wells: dict[tuple[str, ...], list[tuple[date, dict]]] = {}
    for row_date, row in source_rows_with_dates:
        wells.setdefault(primary_group_key(row, source_keys), []).append((row_date, row))

    result_rows: list[RepairForecastRow] = []
    notes: list[str] = []
    monthly_summary_map: dict[str, dict[str, float | int]] = {}

    for _, items in sorted(wells.items(), key=lambda item: item[0]):
        first_row = items[0][1]
        fact_row = find_matching_fact_row(first_row, source_keys, fact_latest_by_key)
        prediction_series: list[tuple[date, float | None]] = []
        actual_nno = None
        runtime_days = None

        for row_date, source_row in items:
            prepared_row = apply_feature_defaults(
                source_row,
                trained["feature_columns"],
                fact_row,
                identifier_stats,
                source_keys,
                manual_feature_values,
                trained["baseline_values"],
            )
            prediction_series.append((row_date, predict_single_row(trained, prepared_row)))

            if actual_nno is None and source_keys.get("nno"):
                actual_nno = to_float(prepared_row.get(source_keys["nno"]))
            if runtime_days is None and source_keys.get("runtime"):
                runtime_days = to_float(prepared_row.get(source_keys["runtime"]))

        if actual_nno is None and fact_row and fact_keys.get("nno"):
            actual_nno = to_float(fact_row.get(fact_keys["nno"]))
        if runtime_days is None and fact_row and fact_keys.get("runtime"):
            runtime_days = to_float(fact_row.get(fact_keys["runtime"]))

        statuses, event_dates, event_nnos = build_statuses_for_well(
            forecast_dates,
            prediction_series,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        )

        for event_date, event_nno in event_nnos:
            month_key = f"{event_date.year}-{event_date.month:02d}"
            bucket = monthly_summary_map.setdefault(month_key, {"total_nno": 0.0, "failure_count": 0})
            bucket["total_nno"] = float(bucket["total_nno"]) + float(event_nno)
            bucket["failure_count"] = int(bucket["failure_count"]) + 1

        predicted_values = [value for _, value in prediction_series if value is not None]
        average_prediction = float(sum(predicted_values) / len(predicted_values)) if predicted_values else None

        first_liquid_rate = to_float(first_row.get(liquid_rate_column)) if liquid_rate_column else None
        category = "База" if first_liquid_rate is not None and first_liquid_rate > 0 else "ВНС"

        result_rows.append(
            RepairForecastRow(
                category=category,
                field_name=stringify(first_row.get(source_keys["field"])) if source_keys.get("field") else None,
                license_area=stringify(first_row.get(identifier_column)),
                cluster_name=stringify(first_row.get(source_keys["cluster"])) if source_keys.get("cluster") else None,
                well_name=stringify(first_row.get(well_column)) or "Неизвестная скважина",
                predicted_nno=average_prediction,
                actual_nno=actual_nno,
                runtime_days=runtime_days,
                event_dates=event_dates,
                statuses=statuses,
            )
        )

    if selected_model is None:
        notes.append("Сохраненная модель не выбрана, используется текущая конфигурация CatBoost.")
    if missing_feature_columns:
        notes.append(
            "Часть признаков модели отсутствовала в сводпрогнозе и была заполнена из Факт ЭПУ, средними значениями по УН, вручную или базовыми значениями модели."
        )
    if not monthly_summary_map:
        notes.append("В расчетном горизонте не возникло прогнозных отказов, поэтому месячная диаграмма показывает ноль.")

    result_rows.sort(
        key=lambda item: (
            item.category,
            item.license_area or "",
            item.cluster_name or "",
            item.well_name,
        )
    )

    monthly_summary = [
        RepairForecastMonthlySummary(
            month=month,
            total_nno=round(float(values["total_nno"]), 1),
            failure_count=int(values["failure_count"]),
        )
        for month, values in sorted(monthly_summary_map.items())
    ]

    return RepairForecastResponse(
        start_date=today.isoformat(),
        end_date=end_date.isoformat(),
        dates=[item.isoformat() for item in forecast_dates],
        used_feature_columns=list(trained["feature_columns"]),
        missing_feature_columns=missing_feature_columns,
        source_datasets=[
            RepairForecastSourceDataset(
                label="Сводпрогноз",
                dataset=build_dataset_detail(source_dataset, source_dataset.records).dataset,
            )
        ],
        monthly_summary=monthly_summary,
        rows=result_rows,
        notes=notes,
    )
