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
    correlation: float


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
