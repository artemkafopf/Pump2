from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Dataset, DatasetColumnMatch, Record, TrainedModel
from app.schemas.analysis import (
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


PRODUCTION_SECTIONS = ("production",)
GTM_SECTIONS = ("gtm", "plan_gtm")

FIELD_PATTERNS = ("месторожд", "field")
LICENSE_PATTERNS = ("участ", "недр", "license", "area")
CLUSTER_PATTERNS = ("куст", "cluster", "pad")
WELL_PATTERNS = ("скваж", "well", "well id", "id скваж", "well_id")
NNO_PATTERNS = ("нно", "mtbf", "наработка на отказ", "mean time")
RUNTIME_PATTERNS = ("наработ", "runtime", "отработ", "worked")
DATE_PATTERNS = ("дата", "date")
GTM_TYPE_PATTERNS = ("вид гтм", "тип гтм", "gtm", "мероприят")
LIQUID_INCREMENT_PATTERNS = (
    "прирост дебита жидкости",
    "жидк",
    "liquid",
    "rate increase",
)
RATE_PATTERNS = ("дебит", "приемист", "закач", "rate", "режим")
TRIGGER_GTM_TYPES = ("оптимизац", "пмд", "смена эцн")


@dataclass
class PreparedDataset:
    dataset: Dataset
    rows: list[dict]
    canonical_map: dict[str, str]


def normalize_text(value: object) -> str:
    return str(value or "").strip().casefold()


def find_best_column(columns: Iterable[str], patterns: tuple[str, ...]) -> str | None:
    columns = list(columns)
    scored: list[tuple[int, str]] = []
    for column in columns:
        normalized = normalize_text(column)
        score = 0
        for pattern in patterns:
            if pattern in normalized:
                score += len(pattern)
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
        if 20000 <= float(numeric) <= 80000:
            parsed = pd.to_datetime(float(numeric), errors="coerce", unit="D", origin="1899-12-30")
            if pd.notna(parsed):
                return parsed.date()
        if 1900 <= float(numeric) <= 2500 and float(numeric).is_integer():
            return date(int(numeric), 1, 1)

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
    return {
        item.source_column: item.canonical_name
        for item in matches
        if item.canonical_name
    }


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


def get_latest_dataset_by_sections(db: Session, sections: tuple[str, ...]) -> Dataset | None:
    return db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.records), selectinload(Dataset.column_matches))
        .where(Dataset.storage_section.in_(sections))
        .order_by(Dataset.storage_version.desc(), Dataset.created_at.desc(), Dataset.id.desc())
    )


def resolve_model_bundle(
    dataset: Dataset,
    model_id: int | None,
    base_failure_feature_columns: list[str],
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

    trained = train_forecast_model(dataset, dataset.records, base_failure_feature_columns)
    return trained, selected_model


def identify_key_columns(rows: list[dict]) -> dict[str, str | None]:
    columns = list({column for row in rows for column in row.keys()})
    return {
        "field": find_best_column(columns, FIELD_PATTERNS),
        "license": find_best_column(columns, LICENSE_PATTERNS),
        "cluster": find_best_column(columns, CLUSTER_PATTERNS),
        "well": find_best_column(columns, WELL_PATTERNS),
        "nno": find_best_column(columns, NNO_PATTERNS),
        "runtime": find_best_column(columns, RUNTIME_PATTERNS),
        "date": find_best_column(columns, DATE_PATTERNS),
        "gtm_type": find_best_column(columns, GTM_TYPE_PATTERNS),
        "liquid_increment": find_best_column(columns, LIQUID_INCREMENT_PATTERNS),
    }


def build_well_key(row: dict, key_columns: dict[str, str | None]) -> tuple[str, str, str, str]:
    return (
        normalize_text(row.get(key_columns["field"])) if key_columns.get("field") else "",
        normalize_text(row.get(key_columns["license"])) if key_columns.get("license") else "",
        normalize_text(row.get(key_columns["cluster"])) if key_columns.get("cluster") else "",
        normalize_text(row.get(key_columns["well"])) if key_columns.get("well") else "",
    )


def best_latest_rows(rows: list[dict], key_columns: dict[str, str | None]) -> dict[tuple[str, str, str, str], dict]:
    dated_rows: dict[tuple[str, str, str, str], tuple[date | None, int, dict]] = {}
    for index, row in enumerate(rows):
        key = build_well_key(row, key_columns)
        row_date = parse_date(row.get(key_columns["date"])) if key_columns.get("date") else None
        previous = dated_rows.get(key)
        if previous is None or ((row_date or date.min), index) >= ((previous[0] or date.min), previous[1]):
            dated_rows[key] = (row_date, index, row)
    return {key: item[2] for key, item in dated_rows.items()}


def build_section_numeric_stats(
    rows: list[dict],
    key_columns: dict[str, str | None],
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    license_column = key_columns.get("license")
    if not license_column:
        return {}
    frame = pd.DataFrame(rows)
    if frame.empty or license_column not in frame.columns:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for license_value, group in frame.groupby(license_column, dropna=True):
        metrics: dict[str, float] = {}
        for column in feature_columns:
            if column not in group.columns:
                continue
            numeric = coerce_numeric_series(group[column])
            if numeric.notna().any():
                metrics[column] = float(numeric.mean())
        stats[normalize_text(license_value)] = metrics
    return stats


def apply_feature_defaults(
    row: dict,
    feature_columns: list[str],
    fact_row: dict | None,
    section_stats: dict[str, dict[str, float]],
    key_columns: dict[str, str | None],
    manual_values: dict[str, object],
    baseline_values: dict[str, object],
) -> dict:
    prepared = dict(row)
    section_key = normalize_text(prepared.get(key_columns["license"])) if key_columns.get("license") else ""
    section_defaults = section_stats.get(section_key, {})
    for column in feature_columns:
        value = prepared.get(column)
        if value not in (None, ""):
            continue
        if column in manual_values and manual_values[column] not in (None, ""):
            prepared[column] = manual_values[column]
        elif fact_row and fact_row.get(column) not in (None, ""):
            prepared[column] = fact_row.get(column)
        elif column in section_defaults:
            prepared[column] = section_defaults[column]
        else:
            prepared[column] = baseline_values.get(column)
    return prepared


def update_gtm_features(
    row: dict,
    gtm_row: dict,
    feature_columns: list[str],
    liquid_increment_column: str | None,
    nominal_gap_coefficient: float,
) -> dict:
    prepared = dict(row)
    increment = to_float(gtm_row.get(liquid_increment_column)) if liquid_increment_column else None
    for column in feature_columns:
        normalized = normalize_text(column)
        current_value = to_float(prepared.get(column))
        if increment is not None and current_value is not None and any(token in normalized for token in RATE_PATTERNS):
            prepared[column] = current_value + increment
        elif current_value is not None and any(token in normalized for token in RATE_PATTERNS):
            prepared[column] = current_value * (1 - nominal_gap_coefficient)
    return prepared


def predict_single_row(trained: dict, row: dict) -> float | None:
    return predict_forecast_rows(trained, [row])[0]


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


def build_period_statuses(
    dates: list[date],
    start_at: date,
    stop_before: date,
    predicted_nno: float | None,
    actual_nno: float | None,
    runtime_days: float | None,
    base_failure_coefficient: float,
) -> list[tuple[date, int]]:
    if predicted_nno is None:
        return []

    events: list[tuple[date, int]] = []
    interval_days = max(int(round(predicted_nno)), 1)
    next_failure = start_at + timedelta(
        days=calculate_first_failure_offset(
            predicted_nno,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        )
    )
    while next_failure < stop_before and next_failure <= dates[-1]:
        events.append((next_failure, 0))
        next_failure = next_failure + timedelta(days=interval_days)
        actual_nno = None
        runtime_days = None
    return events


def build_repair_forecast(
    db: Session,
    fact_dataset: Dataset,
    model_id: int | None,
    base_failure_coefficient: float,
    nominal_gap_coefficient: float,
    manual_feature_values: dict[str, object],
) -> RepairForecastResponse:
    production_dataset = get_latest_dataset_by_sections(db, PRODUCTION_SECTIONS)
    gtm_dataset = get_latest_dataset_by_sections(db, GTM_SECTIONS)
    if production_dataset is None:
        raise ValueError("Для расчета нужен датасет в разделе 'Добыча'.")

    fact_prepared = prepare_dataset(fact_dataset)
    production_prepared = prepare_dataset(production_dataset)
    gtm_prepared = prepare_dataset(gtm_dataset) if gtm_dataset is not None else None

    feature_columns = resolve_forecast_columns(
        fact_dataset,
        list(fact_dataset.selected_features_json or []),
        pd.DataFrame(fact_prepared.rows, columns=list(fact_dataset.columns_json)),
    )
    trained, selected_model = resolve_model_bundle(fact_dataset, model_id, feature_columns)
    if trained is None:
        raise ValueError("Не удалось подготовить модель CatBoost для прогноза ремонтов.")

    today = date.today()
    end_date = date(today.year + 1, 12, 31)
    dates = [today + timedelta(days=offset) for offset in range((end_date - today).days + 1)]

    fact_keys = identify_key_columns(fact_prepared.rows)
    production_keys = identify_key_columns(production_prepared.rows)
    gtm_keys = identify_key_columns(gtm_prepared.rows if gtm_prepared else [])

    fact_latest_by_well = best_latest_rows(fact_prepared.rows, fact_keys)
    production_latest_by_well = best_latest_rows(production_prepared.rows, production_keys)
    section_stats = build_section_numeric_stats(fact_prepared.rows, fact_keys, trained["feature_columns"])

    missing_feature_columns = [
        column for column in trained["feature_columns"]
        if column not in {key for row in production_prepared.rows for key in row.keys()}
    ]

    gtm_by_well: dict[tuple[str, str, str, str], list[dict]] = {}
    if gtm_prepared is not None:
        for row in gtm_prepared.rows:
            key = build_well_key(row, gtm_keys)
            gtm_by_well.setdefault(key, []).append(row)
        if gtm_keys.get("date"):
            for key, items in gtm_by_well.items():
                gtm_by_well[key] = sorted(
                    items,
                    key=lambda item: parse_date(item.get(gtm_keys["date"])) or today,
                )

    all_keys = set(production_latest_by_well) | set(gtm_by_well)
    category_order = {"БАЗА": 0, "ГТМ БАЗА": 1, "ГТМ Развитие": 2}
    result_rows: list[RepairForecastRow] = []
    notes: list[str] = []

    for key in sorted(all_keys):
        production_row = production_latest_by_well.get(key)
        fact_row = fact_latest_by_well.get(key)
        gtm_events = gtm_by_well.get(key, [])

        if production_row is not None and gtm_events:
            category = "ГТМ БАЗА"
        elif production_row is not None:
            category = "БАЗА"
        else:
            category = "ГТМ Развитие"

        base_row = dict(production_row or (gtm_events[0] if gtm_events else {}))
        if category == "БАЗА":
            base_row["категория фонда"] = "БАЗА"
        elif category == "ГТМ БАЗА":
            base_row["категория фонда"] = "ГТМ БАЗА"
        else:
            base_row["категория фонда"] = "ГТМ Развитие"

        base_row = apply_feature_defaults(
            base_row,
            trained["feature_columns"],
            fact_row,
            section_stats,
            production_keys if production_row is not None else gtm_keys,
            manual_feature_values,
            trained["baseline_values"],
        )

        current_predicted_nno = predict_single_row(trained, base_row)
        actual_nno = to_float(base_row.get(production_keys.get("nno"))) if production_keys.get("nno") else None
        runtime_days = to_float(base_row.get(production_keys.get("runtime"))) if production_keys.get("runtime") else None
        statuses = [1] * len(dates)
        event_dates: list[str] = []
        segment_start = today

        event_boundaries: list[tuple[date, dict]] = []
        for gtm_row in gtm_events:
            event_date = parse_date(gtm_row.get(gtm_keys.get("date"))) if gtm_keys.get("date") else None
            if event_date is not None:
                event_boundaries.append((event_date, gtm_row))

        event_boundaries.sort(key=lambda item: item[0])
        for event_date, gtm_row in event_boundaries:
            if event_date > end_date:
                continue
            stop_before = event_date
            for failure_date, code in build_period_statuses(
                dates,
                segment_start,
                stop_before,
                current_predicted_nno,
                actual_nno,
                runtime_days,
                base_failure_coefficient,
            ):
                if failure_date in dates:
                    statuses[(failure_date - today).days] = code

            gtm_type_value = stringify(gtm_row.get(gtm_keys.get("gtm_type"))) if gtm_keys.get("gtm_type") else None
            if gtm_type_value and any(token in normalize_text(gtm_type_value) for token in TRIGGER_GTM_TYPES):
                statuses[(event_date - today).days] = 2
            event_dates.append(event_date.isoformat())

            base_row = update_gtm_features(
                base_row,
                gtm_row,
                trained["feature_columns"],
                gtm_keys.get("liquid_increment"),
                nominal_gap_coefficient,
            )
            base_row = apply_feature_defaults(
                base_row,
                trained["feature_columns"],
                fact_row,
                section_stats,
                production_keys if production_row is not None else gtm_keys,
                manual_feature_values,
                trained["baseline_values"],
            )
            current_predicted_nno = predict_single_row(trained, base_row)
            actual_nno = None
            runtime_days = None
            segment_start = event_date

        for failure_date, code in build_period_statuses(
            dates,
            segment_start,
            end_date + timedelta(days=1),
            current_predicted_nno,
            actual_nno,
            runtime_days,
            base_failure_coefficient,
        ):
            if failure_date in dates:
                statuses[(failure_date - today).days] = code

        keys_source = production_keys if production_row is not None else gtm_keys
        field_name = stringify(base_row.get(keys_source.get("field"))) if keys_source.get("field") else None
        license_area = stringify(base_row.get(keys_source.get("license"))) if keys_source.get("license") else None
        cluster_name = stringify(base_row.get(keys_source.get("cluster"))) if keys_source.get("cluster") else None
        well_name = (
            stringify(base_row.get(keys_source.get("well")))
            or "Неизвестная скважина"
        )

        result_rows.append(
            RepairForecastRow(
                category=category,
                field_name=field_name,
                license_area=license_area,
                cluster_name=cluster_name,
                well_name=well_name,
                predicted_nno=current_predicted_nno,
                actual_nno=to_float(base_row.get(production_keys.get("nno"))) if production_keys.get("nno") else None,
                runtime_days=to_float(base_row.get(production_keys.get("runtime"))) if production_keys.get("runtime") else None,
                event_dates=event_dates,
                statuses=statuses,
            )
        )

    if selected_model is None:
        notes.append("Сохраненная модель не была выбрана, использована текущая обучаемая конфигурация CatBoost.")
    if gtm_dataset is None:
        notes.append("Датасет ГТМ не найден. Для расчета использована только категория БАЗА.")
    if missing_feature_columns:
        notes.append("Часть признаков модели не была найдена в наборе Добыча и была заполнена из Факт ЭПУ, статистики по участку недр или базовых значений модели.")

    result_rows.sort(
        key=lambda item: (
            category_order.get(item.category, 99),
            item.field_name or "",
            item.license_area or "",
            item.cluster_name or "",
            item.well_name,
        )
    )

    return RepairForecastResponse(
        start_date=today.isoformat(),
        end_date=end_date.isoformat(),
        dates=[item.isoformat() for item in dates],
        used_feature_columns=list(trained["feature_columns"]),
        missing_feature_columns=missing_feature_columns,
        source_datasets=[
            RepairForecastSourceDataset(label="Факт ЭПУ", dataset=build_dataset_detail(fact_dataset, fact_dataset.records).dataset),
            RepairForecastSourceDataset(label="Добыча", dataset=build_dataset_detail(production_dataset, production_dataset.records).dataset),
            RepairForecastSourceDataset(
                label="ГТМ",
                dataset=build_dataset_detail(gtm_dataset, gtm_dataset.records).dataset if gtm_dataset is not None else None,
            ),
        ],
        rows=result_rows,
        notes=notes,
    )
