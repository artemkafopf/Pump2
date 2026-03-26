from __future__ import annotations

from io import BytesIO

import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.db.models import Dataset, Record
from app.schemas.analysis import (
    AnalysisResponse,
    CorrelationItem,
    DatasetDetail,
    DatasetSummary,
    FeatureImportanceItem,
    Overview,
    UploadResponse,
)


NUMERIC_PARSE_THRESHOLD = 0.8
DATETIME_PARSE_THRESHOLD = 0.8
MIN_NON_NULL_FOR_ANALYSIS = 5


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
        target_column=dataset.target_column,
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
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def classify_columns(
    df: pd.DataFrame,
    target_column: str | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    feature_df = df.drop(columns=[target_column], errors="ignore").copy()
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


def build_overview(df: pd.DataFrame, target_column: str | None) -> Overview:
    target = build_target_series(df, target_column)
    _, numeric_columns, categorical_columns, datetime_columns = classify_columns(df, target_column)
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
    overview = build_overview(df, dataset.target_column)
    correlations, numeric_columns, categorical_columns, datetime_columns = build_correlations(df, dataset.target_column)
    notes: list[str] = []

    if dataset.target_column is None:
        notes.append("Target column was not set.")
    elif overview.valid_target_rows == 0:
        notes.append("Target column does not contain enough numeric values for CatBoost analysis.")
    if overview.total_rows < 20:
        notes.append("Dataset is small, so relationships may be unstable.")

    return AnalysisResponse(
        dataset=dataset_to_summary(dataset),
        overview=overview,
        correlations=correlations,
        feature_importance=build_feature_importance(df, dataset.target_column),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        notes=notes,
    )
