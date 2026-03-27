from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.db.models import Dataset, Record
from app.schemas.analysis import (
    AnalysisResponse,
    CorrelationItem,
    DatasetDetail,
    DatasetSummary,
    ForecastContourResponse,
    ForecastMetrics,
    ForecastModelResponse,
    ForecastPredictResponse,
    FeatureImportanceItem,
    Overview,
    UploadResponse,
)


NUMERIC_PARSE_THRESHOLD = 0.8
DATETIME_PARSE_THRESHOLD = 0.8
MIN_NON_NULL_FOR_ANALYSIS = 5
DATETIME_NAME_TOKENS = (
    "date",
    "datetime",
    "time",
    "timestamp",
    "year",
    "month",
    "day",
    "дата",
    "время",
    "год",
    "месяц",
    "день",
)


def read_excel_to_dataframe(contents: bytes, filename: str) -> pd.DataFrame:
    lower_name = filename.lower()
    engine = "openpyxl" if lower_name.endswith(".xlsx") else "xlrd"
    return pd.read_excel(BytesIO(contents), engine=engine)


def to_json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def normalize_lookup_value(value: str) -> str:
    return str(value).strip().casefold()


def is_datetime_like_column_name(column_name: str) -> bool:
    normalized = normalize_lookup_value(column_name)
    return any(token in normalized for token in DATETIME_NAME_TOKENS)


def resolve_target_column(requested_target: str | None, columns: list[str]) -> str | None:
    if not requested_target:
        return None
    if requested_target in columns:
        return requested_target

    normalized_to_original = {normalize_lookup_value(column): column for column in columns}
    return normalized_to_original.get(normalize_lookup_value(requested_target))


def dataset_to_summary(dataset: Dataset) -> DatasetSummary:
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        storage_section=dataset.storage_section,
        storage_version=dataset.storage_version,
        target_column=dataset.target_column,
        selected_features=list(dataset.selected_features_json),
        row_count=dataset.row_count,
        columns=list(dataset.columns_json),
        created_at=dataset.created_at,
    )


def build_upload_response(dataset: Dataset) -> UploadResponse:
    return UploadResponse(dataset=dataset_to_summary(dataset))


def records_to_rows(records: list[Record]) -> list[dict]:
    return [dict(record.payload) for record in records]


def build_dataset_detail(dataset: Dataset, records: list[Record]) -> DatasetDetail:
    return DatasetDetail(dataset=dataset_to_summary(dataset), rows=records_to_rows(records))


def records_to_dataframe(records: list[Record], columns: list[str]) -> pd.DataFrame:
    rows = records_to_rows(records)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def get_analysis_columns(dataset: Dataset, df: pd.DataFrame) -> list[str]:
    available_columns = [column for column in list(dataset.columns_json) if column in df.columns]
    selected_features = [
        column
        for column in list(dataset.selected_features_json)
        if column in available_columns and column != dataset.target_column
    ]
    if selected_features:
        return selected_features
    return [column for column in available_columns if column != dataset.target_column]


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    normalized = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")

    comma_mask = (
        normalized.notna()
        & normalized.str.contains(",", regex=False)
        & ~normalized.str.contains(".", regex=False)
    )
    if comma_mask.any():
        numeric.loc[comma_mask] = pd.to_numeric(
            normalized.loc[comma_mask].str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return numeric


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, errors="coerce", unit="D", origin="1899-12-30")

    string_series = series.astype("string").str.strip()
    parsed = pd.to_datetime(string_series, errors="coerce", dayfirst=True)

    numeric_series = coerce_numeric_series(series)
    numeric_mask = parsed.isna() & numeric_series.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric_series.loc[numeric_mask],
            errors="coerce",
            unit="D",
            origin="1899-12-30",
        )
    return parsed


def classify_columns(
    df: pd.DataFrame,
    target_column: str | None,
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    feature_df = df.drop(columns=[target_column], errors="ignore").copy()
    if selected_columns is not None:
        filtered = [column for column in selected_columns if column in feature_df.columns]
        feature_df = feature_df[filtered]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for column in feature_df.columns:
        raw_series = feature_df[column]
        non_null_count = int(raw_series.notna().sum())
        if non_null_count == 0:
            categorical_columns.append(column)
            feature_df[column] = raw_series.astype("string")
            continue

        if is_datetime_like_column_name(column):
            datetime_series = coerce_datetime_series(raw_series)
            datetime_count = int(datetime_series.notna().sum())
            if datetime_count / non_null_count >= min(DATETIME_PARSE_THRESHOLD, 0.5):
                feature_df[column] = datetime_series
                datetime_columns.append(column)
                continue

        numeric_series = coerce_numeric_series(raw_series)
        numeric_count = int(numeric_series.notna().sum())
        if non_null_count >= MIN_NON_NULL_FOR_ANALYSIS and numeric_count / non_null_count >= NUMERIC_PARSE_THRESHOLD:
            feature_df[column] = numeric_series
            numeric_columns.append(column)
            continue

        datetime_series = coerce_datetime_series(raw_series)
        datetime_count = int(datetime_series.notna().sum())
        if non_null_count >= MIN_NON_NULL_FOR_ANALYSIS and datetime_count / non_null_count >= DATETIME_PARSE_THRESHOLD:
            feature_df[column] = datetime_series
            datetime_columns.append(column)
            continue

        feature_df[column] = raw_series.astype("string").str.strip()
        categorical_columns.append(column)

    return feature_df, numeric_columns, categorical_columns, datetime_columns


def build_target_series(df: pd.DataFrame, target_column: str | None) -> pd.Series:
    if not target_column or target_column not in df.columns:
        return pd.Series(dtype=float)
    return coerce_numeric_series(df[target_column])


def build_overview(df: pd.DataFrame, target_column: str | None, selected_columns: list[str] | None = None) -> Overview:
    target = build_target_series(df, target_column)
    _, numeric_columns, categorical_columns, datetime_columns = classify_columns(df, target_column, selected_columns)
    denominator = df.shape[0] * max(df.shape[1], 1)
    missing_share = float(df.isna().sum().sum() / denominator) if denominator else 0.0
    valid_target = target.dropna()

    return Overview(
        total_rows=int(df.shape[0]),
        total_columns=int(df.shape[1]),
        valid_target_rows=int(valid_target.shape[0]),
        missing_values_share=missing_share,
        numeric_column_count=len(numeric_columns),
        categorical_column_count=len(categorical_columns),
        datetime_column_count=len(datetime_columns),
        target_mean=float(valid_target.mean()) if not valid_target.empty else None,
        target_median=float(valid_target.median()) if not valid_target.empty else None,
        target_min=float(valid_target.min()) if not valid_target.empty else None,
        target_max=float(valid_target.max()) if not valid_target.empty else None,
    )


def build_correlations(df: pd.DataFrame, target_column: str | None) -> tuple[list[CorrelationItem], list[str], list[str], list[str]]:
    target = build_target_series(df, target_column)
    if target.empty:
        return [], [], [], []

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty:
        return [], [], [], []

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(clean, target_column)
    correlations: list[CorrelationItem] = []

    for column in numeric_columns:
        series = coerce_numeric_series(feature_df[column])
        if int(series.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = series.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlations.append(CorrelationItem(feature=column, correlation=float(corr)))

    for column in datetime_columns:
        series = pd.to_datetime(feature_df[column], errors="coerce")
        numeric_time = series.map(lambda value: value.timestamp() if pd.notna(value) else pd.NA)
        numeric_time = pd.to_numeric(numeric_time, errors="coerce")
        if int(numeric_time.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = numeric_time.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlations.append(CorrelationItem(feature=column, correlation=float(corr)))

    correlations.sort(key=lambda item: abs(item.correlation), reverse=True)
    return correlations[:25], numeric_columns, categorical_columns, datetime_columns


def build_feature_importance(df: pd.DataFrame, target_column: str | None) -> list[FeatureImportanceItem]:
    target = build_target_series(df, target_column)
    if target.empty:
        return []

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty or clean["_target_numeric"].nunique() < 2:
        return []

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(clean, target_column)
    if feature_df.empty:
        return []

    prepared = pd.DataFrame(index=feature_df.index)
    for column in numeric_columns:
        prepared[column] = coerce_numeric_series(feature_df[column])
    for column in datetime_columns:
        dt_series = pd.to_datetime(feature_df[column], errors="coerce")
        prepared[column] = dt_series.map(lambda value: value.timestamp() if pd.notna(value) else None)
    for column in categorical_columns:
        prepared[column] = feature_df[column].astype("string").fillna("__missing__")

    if prepared.empty:
        return []

    categorical_feature_indices = [prepared.columns.get_loc(column) for column in categorical_columns]
    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(prepared, clean["_target_numeric"], cat_features=categorical_feature_indices))

    items = [
        FeatureImportanceItem(feature=feature, importance=float(importance))
        for feature, importance in zip(prepared.columns.tolist(), model.get_feature_importance(), strict=False)
    ]
    items.sort(key=lambda item: item.importance, reverse=True)
    return items[:25]


def build_analysis_response(dataset: Dataset, records: list[Record]) -> AnalysisResponse:
    df = records_to_dataframe(records, list(dataset.columns_json))
    selected_columns = get_analysis_columns(dataset, df)
    overview = build_overview(df, dataset.target_column, selected_columns)
    correlations, numeric_columns, categorical_columns, datetime_columns = build_correlations_with_selection(
        df,
        dataset.target_column,
        selected_columns,
    )
    notes: list[str] = []

    if dataset.target_column is None:
        notes.append("Target column was not set.")
    elif overview.valid_target_rows == 0:
        notes.append("Target column does not contain enough numeric values for CatBoost analysis.")
    if not selected_columns:
        notes.append("Dependent variables were not selected.")
    if overview.total_rows < 20:
        notes.append("Dataset is small, so relationships may be unstable.")

    return AnalysisResponse(
        dataset=dataset_to_summary(dataset),
        overview=overview,
        correlations=correlations,
        feature_importance=build_feature_importance_with_selection(df, dataset.target_column, selected_columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        notes=notes,
    )


def build_correlations_with_selection(
    df: pd.DataFrame,
    target_column: str | None,
    selected_columns: list[str],
) -> tuple[list[CorrelationItem], list[str], list[str], list[str]]:
    target = build_target_series(df, target_column)
    if target.empty:
        return [], [], [], []

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty:
        return [], [], [], []

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(
        clean,
        target_column,
        selected_columns,
    )
    correlation_map: dict[str, float | None] = {column: None for column in feature_df.columns}

    for column in numeric_columns:
        series = coerce_numeric_series(feature_df[column])
        if int(series.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = series.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlation_map[column] = float(corr)

    for column in datetime_columns:
        series = pd.to_datetime(feature_df[column], errors="coerce")
        numeric_time = series.map(lambda value: value.timestamp() if pd.notna(value) else pd.NA)
        numeric_time = pd.to_numeric(numeric_time, errors="coerce")
        if int(numeric_time.notna().sum()) < MIN_NON_NULL_FOR_ANALYSIS:
            continue
        corr = numeric_time.corr(clean["_target_numeric"])
        if pd.notna(corr):
            correlation_map[column] = float(corr)

    ordered_correlations = [
        CorrelationItem(feature=column, correlation=correlation_map.get(column))
        for column in selected_columns
        if column in feature_df.columns
    ]
    return ordered_correlations, numeric_columns, categorical_columns, datetime_columns


def build_feature_importance_with_selection(
    df: pd.DataFrame,
    target_column: str | None,
    selected_columns: list[str],
) -> list[FeatureImportanceItem]:
    target = build_target_series(df, target_column)
    if target.empty:
        return []

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty or clean["_target_numeric"].nunique() < 2:
        return []

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(
        clean,
        target_column,
        selected_columns,
    )
    if feature_df.empty:
        return []

    prepared = pd.DataFrame(index=feature_df.index)
    for column in numeric_columns:
        prepared[column] = coerce_numeric_series(feature_df[column])
    for column in datetime_columns:
        dt_series = pd.to_datetime(feature_df[column], errors="coerce")
        prepared[column] = dt_series.map(lambda value: value.timestamp() if pd.notna(value) else None)
    for column in categorical_columns:
        prepared[column] = feature_df[column].astype("string").fillna("__missing__")

    if prepared.empty:
        return []

    categorical_feature_indices = [prepared.columns.get_loc(column) for column in categorical_columns]
    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(prepared, clean["_target_numeric"], cat_features=categorical_feature_indices))

    items = [
        FeatureImportanceItem(feature=feature, importance=float(importance))
        for feature, importance in zip(prepared.columns.tolist(), model.get_feature_importance(), strict=False)
    ]
    importance_map = {item.feature: item.importance for item in items}
    return [
        FeatureImportanceItem(feature=column, importance=float(importance_map.get(column, 0.0)))
        for column in selected_columns
        if column in prepared.columns
    ]


def resolve_forecast_columns(dataset: Dataset, requested_columns: list[str], df: pd.DataFrame) -> list[str]:
    available_columns = [column for column in list(dataset.columns_json) if column in df.columns and column != dataset.target_column]
    resolved: list[str] = []
    for column in requested_columns:
        resolved_column = resolve_target_column(column, available_columns)
        if resolved_column and resolved_column not in resolved:
            resolved.append(resolved_column)
    if resolved:
        return resolved

    selected_columns = get_analysis_columns(dataset, df)
    return [column for column in selected_columns if column in available_columns]


def prepare_forecast_frame(
    df: pd.DataFrame,
    target_column: str | None,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    target = build_target_series(df, target_column)
    if target.empty:
        return pd.DataFrame(), [], pd.Series(dtype=float)

    clean = df.copy()
    clean["_target_numeric"] = target
    clean = clean[clean["_target_numeric"].notna()].copy()
    if clean.empty or clean["_target_numeric"].nunique() < 2:
        return pd.DataFrame(), [], pd.Series(dtype=float)

    feature_df, numeric_columns, categorical_columns, datetime_columns = classify_columns(
        clean,
        target_column,
        feature_columns,
    )
    prepared = pd.DataFrame(index=feature_df.index)
    numeric_feature_columns: list[str] = []

    for column in numeric_columns:
        prepared[column] = coerce_numeric_series(feature_df[column])
        numeric_feature_columns.append(column)

    for column in datetime_columns:
        dt_series = pd.to_datetime(feature_df[column], errors="coerce")
        prepared[column] = dt_series.map(lambda value: value.timestamp() if pd.notna(value) else None)
        numeric_feature_columns.append(column)

    for column in categorical_columns:
        prepared[column] = feature_df[column].astype("string").fillna("__missing__")

    prepared = prepared.dropna(axis=0, how="all")
    if prepared.empty:
        return pd.DataFrame(), numeric_feature_columns, pd.Series(dtype=float)

    target_series = clean.loc[prepared.index, "_target_numeric"]
    return prepared, numeric_feature_columns, target_series


def build_forecast_metrics(actual: pd.Series, predicted: np.ndarray) -> ForecastMetrics:
    residual = actual.to_numpy(dtype=float) - predicted
    rmse = float(np.sqrt(np.mean(np.square(residual)))) if len(actual) else None
    mae = float(np.mean(np.abs(residual))) if len(actual) else None
    if len(actual) and actual.nunique() > 1:
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(actual.to_numpy(dtype=float) - actual.mean())))
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot else None
    else:
        r2 = None

    return ForecastMetrics(
        train_rows=0,
        test_rows=int(len(actual)),
        rmse=rmse,
        mae=mae,
        r2=r2,
    )


def train_forecast_model(
    dataset: Dataset,
    records: list[Record],
    requested_columns: list[str],
    test_fraction: float = 0.2,
    random_seed: int = 42,
):
    df = records_to_dataframe(records, list(dataset.columns_json))
    feature_columns = resolve_forecast_columns(dataset, requested_columns, df)
    prepared, numeric_feature_columns, target_series = prepare_forecast_frame(df, dataset.target_column, feature_columns)
    if prepared.empty or target_series.empty:
        return None

    test_fraction = min(max(test_fraction, 0.05), 0.4)
    total_rows = len(prepared)
    test_rows = max(1, int(total_rows * test_fraction)) if total_rows > 4 else max(1, total_rows // 3)
    test_rows = min(test_rows, max(total_rows - 1, 1))
    shuffled_indices = prepared.sample(frac=1.0, random_state=random_seed).index.tolist()
    test_index = shuffled_indices[:test_rows]
    train_index = shuffled_indices[test_rows:] or shuffled_indices[: max(total_rows - test_rows, 1)]

    train_frame = prepared.loc[train_index]
    train_target = target_series.loc[train_index]
    test_frame = prepared.loc[test_index]
    test_target = target_series.loc[test_index]

    categorical_columns = [column for column in prepared.columns if column not in numeric_feature_columns]
    categorical_feature_indices = [prepared.columns.get_loc(column) for column in categorical_columns]

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(train_frame, train_target, cat_features=categorical_feature_indices))

    test_predictions = model.predict(test_frame)
    metrics = build_forecast_metrics(test_target, test_predictions)
    metrics.train_rows = int(len(train_frame))
    metrics.test_rows = int(len(test_frame))

    baseline_values: dict[str, str | float] = {}
    for column in prepared.columns:
        if column in numeric_feature_columns:
            baseline_values[column] = float(pd.to_numeric(prepared[column], errors="coerce").median())
        else:
            mode = prepared[column].mode(dropna=True)
            baseline_values[column] = str(mode.iloc[0]) if not mode.empty else "__missing__"

    return {
        "model": model,
        "prepared": prepared,
        "feature_columns": prepared.columns.tolist(),
        "numeric_feature_columns": numeric_feature_columns,
        "baseline_values": baseline_values,
        "metrics": metrics,
    }


def predict_forecast_rows(
    trained: dict,
    rows: list[dict],
) -> list[float | None]:
    feature_columns = trained["feature_columns"]
    numeric_feature_columns = set(trained["numeric_feature_columns"])
    baseline_values = trained["baseline_values"]
    prediction_frame = pd.DataFrame(index=range(len(rows)))

    for column in feature_columns:
        source_values = [row.get(column, baseline_values.get(column)) for row in rows]
        series = pd.Series(source_values, dtype="object")
        if column in numeric_feature_columns:
            prediction_frame[column] = coerce_numeric_series(series)
        else:
            prediction_frame[column] = series.astype("string").fillna("__missing__")

    result: list[float | None] = [None] * len(rows)
    valid_mask = prediction_frame.notna().any(axis=1)
    if valid_mask.any():
        predicted = trained["model"].predict(prediction_frame.loc[valid_mask])
        for index, value in zip(prediction_frame.loc[valid_mask].index.tolist(), predicted, strict=False):
            result[index] = float(value)
    return result


def build_forecast_contour(
    trained: dict,
    x_feature: str | None,
    y_feature: str | None,
    slice_overrides: dict[str, str | float | int | None] | None = None,
    contour_resolution: int = 24,
) -> ForecastContourResponse | None:
    if not x_feature or not y_feature:
        return None
    if x_feature == y_feature:
        return None

    prepared = trained["prepared"]
    numeric_feature_columns = set(trained["numeric_feature_columns"])
    if x_feature not in numeric_feature_columns or y_feature not in numeric_feature_columns:
        return None

    x_series = pd.to_numeric(prepared[x_feature], errors="coerce").dropna()
    y_series = pd.to_numeric(prepared[y_feature], errors="coerce").dropna()
    if x_series.empty or y_series.empty:
        return None

    contour_resolution = int(min(max(contour_resolution, 8), 40))
    x_values = np.linspace(float(x_series.min()), float(x_series.max()), contour_resolution)
    y_values = np.linspace(float(y_series.min()), float(y_series.max()), contour_resolution)
    baseline_values = trained["baseline_values"]
    feature_columns = trained["feature_columns"]

    grid_rows: list[dict] = []
    for y_value in y_values:
        for x_value in x_values:
            row = {column: baseline_values.get(column) for column in feature_columns}
            if slice_overrides:
                for column, value in slice_overrides.items():
                    if column in row:
                        row[column] = value
            row[x_feature] = float(x_value)
            row[y_feature] = float(y_value)
            grid_rows.append(row)

    predictions = predict_forecast_rows(trained, grid_rows)
    z_values: list[list[float | None]] = []
    offset = 0
    for _ in y_values:
        row_values = predictions[offset : offset + len(x_values)]
        z_values.append(row_values)
        offset += len(x_values)

    return ForecastContourResponse(
        x_feature=x_feature,
        y_feature=y_feature,
        x_values=[float(value) for value in x_values],
        y_values=[float(value) for value in y_values],
        z_values=z_values,
    )


def build_forecast_model_response(
    dataset: Dataset,
    records: list[Record],
    requested_columns: list[str],
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> ForecastModelResponse:
    trained = train_forecast_model(dataset, records, requested_columns, test_fraction=test_fraction, random_seed=random_seed)
    if trained is None:
        return ForecastModelResponse(
            dataset=dataset_to_summary(dataset),
            target_column=dataset.target_column,
            feature_columns=[],
            metrics=ForecastMetrics(train_rows=0, test_rows=0, rmse=None, mae=None, r2=None),
            numeric_feature_columns=[],
        )

    return ForecastModelResponse(
        dataset=dataset_to_summary(dataset),
        target_column=dataset.target_column,
        feature_columns=trained["feature_columns"],
        metrics=trained["metrics"],
        numeric_feature_columns=list(trained["numeric_feature_columns"]),
    )


def build_forecast_predict_response(
    dataset: Dataset,
    records: list[Record],
    requested_columns: list[str],
    rows: list[dict],
    x_feature: str | None = None,
    y_feature: str | None = None,
    slice_overrides: dict[str, str | float | int | None] | None = None,
    contour_resolution: int = 24,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> ForecastPredictResponse:
    trained = train_forecast_model(dataset, records, requested_columns, test_fraction=test_fraction, random_seed=random_seed)
    if trained is None:
        return ForecastPredictResponse(
            dataset=dataset_to_summary(dataset),
            target_column=dataset.target_column,
            feature_columns=[],
            metrics=ForecastMetrics(train_rows=0, test_rows=0, rmse=None, mae=None, r2=None),
            predictions=[None for _ in rows],
            contour=None,
        )

    return ForecastPredictResponse(
        dataset=dataset_to_summary(dataset),
        target_column=dataset.target_column,
        feature_columns=trained["feature_columns"],
        metrics=trained["metrics"],
        predictions=predict_forecast_rows(trained, rows),
        contour=build_forecast_contour(
            trained,
            x_feature,
            y_feature,
            slice_overrides=slice_overrides,
            contour_resolution=contour_resolution,
        ),
    )
