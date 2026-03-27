from datetime import datetime

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    id: int
    name: str
    original_filename: str
    target_column: str | None
    selected_features: list[str]
    row_count: int
    columns: list[str]
    created_at: datetime


class DatasetDetail(BaseModel):
    dataset: DatasetSummary
    rows: list[dict]


class UploadResponse(BaseModel):
    dataset: DatasetSummary


class ColumnSelectionUpdate(BaseModel):
    target_column: str | None = None
    selected_features: list[str] = []


class Overview(BaseModel):
    total_rows: int
    total_columns: int
    valid_target_rows: int
    missing_values_share: float
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int
    target_mean: float | None
    target_median: float | None
    target_min: float | None
    target_max: float | None


class CorrelationItem(BaseModel):
    feature: str
    correlation: float | None


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class AnalysisResponse(BaseModel):
    dataset: DatasetSummary
    overview: Overview
    correlations: list[CorrelationItem]
    feature_importance: list[FeatureImportanceItem]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    notes: list[str]


class ForecastTrainRequest(BaseModel):
    feature_columns: list[str]
    test_fraction: float = 0.2
    random_seed: int = 42


class ForecastMetrics(BaseModel):
    train_rows: int
    test_rows: int
    rmse: float | None
    mae: float | None
    r2: float | None


class ForecastModelResponse(BaseModel):
    dataset: DatasetSummary
    target_column: str | None
    feature_columns: list[str]
    metrics: ForecastMetrics
    numeric_feature_columns: list[str]


class ForecastPredictRequest(BaseModel):
    feature_columns: list[str]
    rows: list[dict]
    x_feature: str | None = None
    y_feature: str | None = None
    slice_overrides: dict[str, str | float | int | None] = {}
    contour_resolution: int = 24
    test_fraction: float = 0.2
    random_seed: int = 42


class ForecastContourResponse(BaseModel):
    x_feature: str
    y_feature: str
    x_values: list[float]
    y_values: list[float]
    z_values: list[list[float | None]]


class ForecastPredictResponse(BaseModel):
    dataset: DatasetSummary
    target_column: str | None
    feature_columns: list[str]
    metrics: ForecastMetrics
    predictions: list[float | None]
    contour: ForecastContourResponse | None
