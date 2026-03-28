from datetime import datetime

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    id: int
    name: str
    original_filename: str
    storage_section: str
    storage_version: int
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


class SavedModelSummary(BaseModel):
    id: int
    dataset_id: int
    name: str
    target_column: str | None
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    metrics: ForecastMetrics
    is_active: bool
    created_at: datetime


class SaveForecastModelRequest(BaseModel):
    name: str | None = None
    feature_columns: list[str]
    test_fraction: float = 0.2
    random_seed: int = 42


class ForecastPredictRequest(BaseModel):
    model_id: int | None = None
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


class LLMStatusResponse(BaseModel):
    available: bool
    model: str
    installed: bool
    models: list[str]
    error: str | None = None


class CanonicalVariableSummary(BaseModel):
    id: int
    canonical_name: str
    description: str | None = None
    aliases: list[str]


class DatasetColumnMatchSummary(BaseModel):
    id: int | None = None
    source_column: str
    canonical_name: str
    canonical_variable_id: int | None = None
    confidence: float
    reasoning: str | None = None
    status: str
    llm_used: bool


class VariableReconcileRequest(BaseModel):
    persist: bool = True
    use_llm: bool = True


class VariableReconcileResponse(BaseModel):
    dataset: DatasetSummary
    matches: list[DatasetColumnMatchSummary]
    unresolved_columns: list[str]
    llm_used: bool
    notes: list[str]


class ReportGenerateRequest(BaseModel):
    report_type: str = "analysis_forecast"
    model_id: int | None = None
    use_llm: bool = True


class GeneratedReportSummary(BaseModel):
    id: int
    dataset_id: int
    trained_model_id: int | None = None
    report_type: str
    title: str
    content: str
    llm_used: bool
    created_at: datetime
