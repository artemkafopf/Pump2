from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.database import get_db
from app.db.models import Dataset, Record
from app.schemas.analysis import (
    AnalysisResponse,
    ColumnSelectionUpdate,
    DatasetDetail,
    DatasetSummary,
    ForecastModelResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
    ForecastTrainRequest,
    UploadResponse,
)
from app.services.analysis import (
    build_analysis_response,
    build_dataset_detail,
    build_forecast_model_response,
    build_forecast_predict_response,
    build_upload_response,
    dataset_to_summary,
    read_excel_to_dataframe,
    resolve_target_column,
    to_json_safe,
)

router = APIRouter()


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [dataset_to_summary(dataset) for dataset in datasets]


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    dataset_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    filename = file.filename or ""
    lower_name = filename.lower()
    if not lower_name.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported.")

    contents = await file.read()
    try:
        df = read_excel_to_dataframe(contents, filename=filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Excel file is empty.")

    original_columns = [str(column) for column in df.columns.tolist()]
    df.columns = original_columns
    dataset = Dataset(
        name=dataset_name.strip(),
        original_filename=filename,
        target_column=None,
        selected_features_json=[],
        row_count=int(len(df)),
        columns_json=original_columns,
    )
    db.add(dataset)
    db.flush()

    for row_index, row in enumerate(df.to_dict(orient="records")):
        payload = {str(key): to_json_safe(value) for key, value in row.items()}
        db.add(
            Record(
                dataset_id=dataset.id,
                row_index=row_index,
                payload=payload,
            )
        )

    db.commit()
    db.refresh(dataset)
    return build_upload_response(dataset)


@router.patch("/datasets/{dataset_id}/selection", response_model=DatasetSummary)
def update_dataset_selection(dataset_id: int, payload: ColumnSelectionUpdate, db: Session = Depends(get_db)):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    target_column = None
    if payload.target_column:
        target_column = resolve_target_column(payload.target_column, list(dataset.columns_json))
        if target_column is None:
            raise HTTPException(status_code=400, detail="Selected target column was not found.")

    selected_features: list[str] = []
    for column in payload.selected_features:
        resolved = resolve_target_column(column, list(dataset.columns_json))
        if resolved and resolved != target_column and resolved not in selected_features:
            selected_features.append(resolved)

    dataset.target_column = target_column
    dataset.selected_features_json = selected_features
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset_to_summary(dataset)


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return build_dataset_detail(dataset, dataset.records)


@router.get("/datasets/{dataset_id}/analysis", response_model=AnalysisResponse)
def get_dataset_analysis(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return build_analysis_response(dataset, dataset.records)


@router.post("/datasets/{dataset_id}/forecast/train", response_model=ForecastModelResponse)
def train_dataset_forecast(dataset_id: int, payload: ForecastTrainRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return build_forecast_model_response(
        dataset,
        dataset.records,
        payload.feature_columns,
        test_fraction=payload.test_fraction,
        random_seed=payload.random_seed,
    )


@router.post("/datasets/{dataset_id}/forecast/predict", response_model=ForecastPredictResponse)
def predict_dataset_forecast(dataset_id: int, payload: ForecastPredictRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return build_forecast_predict_response(
        dataset,
        dataset.records,
        payload.feature_columns,
        payload.rows,
        x_feature=payload.x_feature,
        y_feature=payload.y_feature,
        slice_overrides=payload.slice_overrides,
        contour_resolution=payload.contour_resolution,
        test_fraction=payload.test_fraction,
        random_seed=payload.random_seed,
    )
