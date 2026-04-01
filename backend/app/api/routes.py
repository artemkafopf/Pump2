from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.database import get_db
from app.db.models import Dataset, Record, TrainedModel
from app.schemas.analysis import (
    AnalysisResponse,
    CanonicalEntitySummary,
    CanonicalVariableSummary,
    ColumnSelectionUpdate,
    DatasetDetail,
    DatasetColumnMatchSummary,
    DatasetEntityMatchSummary,
    DatasetSummary,
    DictionaryClearResponse,
    EntityReconcileRequest,
    EntityReconcileResponse,
    ForecastModelResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
    ForecastTrainRequest,
    GeneratedReportSummary,
    LLMStatusResponse,
    ManualEntityMatchRequest,
    ManualVariableMatchRequest,
    RepairForecastRequest,
    RepairForecastResponse,
    ReportGenerateRequest,
    SaveForecastModelRequest,
    SavedModelSummary,
    VariableReconcileRequest,
    VariableReconcileResponse,
    UploadResponse,
)
from app.services.analysis import (
    build_analysis_response,
    build_dataset_detail,
    build_forecast_model_response,
    build_forecast_contour_excel_bytes,
    build_forecast_predict_response,
    build_upload_response,
    dataset_to_summary,
    list_saved_models,
    read_excel_to_dataframe,
    resolve_target_column,
    save_trained_forecast_model,
    saved_model_to_summary,
    train_forecast_model,
    to_json_safe,
)
from app.services.llm_client import llm_client
from app.services.reporting import generate_report, list_reports
from app.services.repair_forecast import build_repair_forecast
from app.services.row_mapping import (
    entity_dictionary_snapshot,
    get_dataset_entity_matches,
    reconcile_dataset_entities,
    save_manual_entity_matches,
)
from app.services.variable_mapping import (
    clear_variable_dictionary,
    dictionary_snapshot,
    get_dataset_matches,
    reconcile_dataset_columns,
    save_manual_dataset_matches,
)

router = APIRouter()


@router.get("/llm/status", response_model=LLMStatusResponse)
def get_llm_status():
    return LLMStatusResponse(**llm_client.status())


@router.get("/variables/dictionary", response_model=list[CanonicalVariableSummary])
def get_variable_dictionary(db: Session = Depends(get_db)):
    return [CanonicalVariableSummary(**item) for item in dictionary_snapshot(db)]


@router.delete("/variables/dictionary", response_model=DictionaryClearResponse)
def reset_variable_dictionary(db: Session = Depends(get_db)):
    return DictionaryClearResponse(**clear_variable_dictionary(db))


@router.get("/entities/dictionary", response_model=list[CanonicalEntitySummary])
def get_entity_dictionary_items(db: Session = Depends(get_db)):
    return [CanonicalEntitySummary(**item) for item in entity_dictionary_snapshot(db)]


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [dataset_to_summary(dataset) for dataset in datasets]


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    dataset_name: str = Form(...),
    storage_section: str = Form("fact_epu"),
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
    normalized_section = (storage_section or "fact_epu").strip() or "fact_epu"
    latest_version = db.scalar(
        select(Dataset.storage_version)
        .where(Dataset.storage_section == normalized_section)
        .order_by(Dataset.storage_version.desc())
        .limit(1)
    )

    dataset = Dataset(
        name=dataset_name.strip(),
        original_filename=filename,
        storage_section=normalized_section,
        storage_version=int(latest_version or 0) + 1,
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


@router.get("/datasets/{dataset_id}/variables/matches", response_model=list[DatasetColumnMatchSummary])
def get_dataset_variable_matches(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return [
        DatasetColumnMatchSummary(
            id=item.id,
            source_column=item.source_column,
            canonical_name=item.canonical_name,
            canonical_variable_id=item.canonical_variable_id,
            confidence=float(item.confidence),
            reasoning=item.reasoning,
            status=item.status,
            llm_used=bool(item.llm_used),
        )
        for item in get_dataset_matches(db, dataset_id)
    ]


@router.post("/datasets/{dataset_id}/variables/reconcile", response_model=VariableReconcileResponse)
def reconcile_variables(dataset_id: int, payload: VariableReconcileRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    result = reconcile_dataset_columns(
        db,
        dataset,
        persist=payload.persist,
        use_llm=payload.use_llm,
        columns=payload.columns,
    )
    return VariableReconcileResponse(
        dataset=dataset_to_summary(dataset),
        matches=[
            DatasetColumnMatchSummary(
                id=item.id,
                source_column=item.source_column,
                canonical_name=item.canonical_name,
                canonical_variable_id=item.canonical_variable_id,
                confidence=float(item.confidence),
                reasoning=item.reasoning,
                status=item.status,
                llm_used=bool(item.llm_used),
            )
            for item in result["matches"]
        ],
        unresolved_columns=result["unresolved_columns"],
        llm_used=bool(result["llm_used"]),
        notes=result["notes"],
    )


@router.post("/datasets/{dataset_id}/variables/manual", response_model=VariableReconcileResponse)
def save_manual_variable_matches(dataset_id: int, payload: ManualVariableMatchRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    matches = save_manual_dataset_matches(
        db,
        dataset,
        [item.model_dump() for item in payload.matches],
    )
    return VariableReconcileResponse(
        dataset=dataset_to_summary(dataset),
        matches=[
            DatasetColumnMatchSummary(
                id=item.id,
                source_column=item.source_column,
                canonical_name=item.canonical_name,
                canonical_variable_id=item.canonical_variable_id,
                confidence=float(item.confidence),
                reasoning=item.reasoning,
                status=item.status,
                llm_used=bool(item.llm_used),
            )
            for item in matches
        ],
        unresolved_columns=[],
        llm_used=False,
        notes=["Manual mapping saved."],
    )


@router.get("/datasets/{dataset_id}/entities/matches", response_model=list[DatasetEntityMatchSummary])
def get_dataset_entities(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.entity_matches))
        .where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return [
        DatasetEntityMatchSummary(
            id=item.id,
            entity_type=item.entity_type,
            source_value=item.source_value,
            canonical_value=item.canonical_value,
            canonical_entity_id=item.canonical_entity_id,
            confidence=float(item.confidence),
            reasoning=item.reasoning,
            status=item.status,
        )
        for item in get_dataset_entity_matches(db, dataset_id)
    ]


@router.post("/datasets/{dataset_id}/entities/reconcile", response_model=EntityReconcileResponse)
def reconcile_entities(dataset_id: int, payload: EntityReconcileRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.records), selectinload(Dataset.column_matches))
        .where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    result = reconcile_dataset_entities(db, dataset, persist=payload.persist)
    return EntityReconcileResponse(
        dataset=dataset_to_summary(dataset),
        matches=[
            DatasetEntityMatchSummary(
                id=item.id,
                entity_type=item.entity_type,
                source_value=item.source_value,
                canonical_value=item.canonical_value,
                canonical_entity_id=item.canonical_entity_id,
                confidence=float(item.confidence),
                reasoning=item.reasoning,
                status=item.status,
            )
            for item in result["matches"]
        ],
        unresolved_values=result["unresolved_values"],
        resolved_columns=result["resolved_columns"],
    )


@router.post("/datasets/{dataset_id}/entities/manual", response_model=EntityReconcileResponse)
def save_manual_entities(dataset_id: int, payload: ManualEntityMatchRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.records), selectinload(Dataset.column_matches))
        .where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    matches = save_manual_entity_matches(
        db,
        dataset,
        [item.model_dump() for item in payload.matches],
    )
    return EntityReconcileResponse(
        dataset=dataset_to_summary(dataset),
        matches=[
            DatasetEntityMatchSummary(
                id=item.id,
                entity_type=item.entity_type,
                source_value=item.source_value,
                canonical_value=item.canonical_value,
                canonical_entity_id=item.canonical_entity_id,
                confidence=float(item.confidence),
                reasoning=item.reasoning,
                status=item.status,
            )
            for item in matches
        ],
        unresolved_values={},
        resolved_columns=reconcile_dataset_entities(db, dataset, persist=False)["resolved_columns"],
    )


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


@router.get("/datasets/{dataset_id}/forecast/models", response_model=list[SavedModelSummary])
def get_saved_forecast_models(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.trained_models)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return list_saved_models(dataset)


@router.post("/datasets/{dataset_id}/forecast/models", response_model=SavedModelSummary)
def save_dataset_forecast_model(dataset_id: int, payload: SaveForecastModelRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records), selectinload(Dataset.trained_models)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    trained = train_forecast_model(
        dataset,
        dataset.records,
        payload.feature_columns,
        test_fraction=payload.test_fraction,
        random_seed=payload.random_seed,
    )
    if trained is None:
        raise HTTPException(status_code=400, detail="Model could not be trained on the selected columns.")

    for model in dataset.trained_models:
        model.is_active = 0
        db.add(model)

    saved_model = save_trained_forecast_model(dataset, trained, model_name=payload.name)
    db.add(saved_model)
    db.commit()
    db.refresh(saved_model)
    return saved_model_to_summary(saved_model)


@router.post("/datasets/{dataset_id}/forecast/predict", response_model=ForecastPredictResponse)
def predict_dataset_forecast(dataset_id: int, payload: ForecastPredictRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records), selectinload(Dataset.trained_models)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    selected_model = None
    if payload.model_id is not None:
        selected_model = next((model for model in dataset.trained_models if model.id == payload.model_id), None)
        if selected_model is None:
            raise HTTPException(status_code=404, detail="Saved model not found.")
    return build_forecast_predict_response(
        dataset,
        dataset.records,
        payload.feature_columns,
        payload.rows,
        model=selected_model,
        x_feature=payload.x_feature,
        y_feature=payload.y_feature,
        slice_overrides=payload.slice_overrides,
        contour_resolution=payload.contour_resolution,
        test_fraction=payload.test_fraction,
        random_seed=payload.random_seed,
    )


@router.post("/datasets/{dataset_id}/forecast/contour-export")
def export_forecast_contour(dataset_id: int, payload: ForecastPredictRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records), selectinload(Dataset.trained_models)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    selected_model = None
    if payload.model_id is not None:
        selected_model = next((model for model in dataset.trained_models if model.id == payload.model_id), None)
        if selected_model is None:
            raise HTTPException(status_code=404, detail="Saved model not found.")

    file_bytes = build_forecast_contour_excel_bytes(
        dataset,
        dataset.records,
        payload.feature_columns,
        model=selected_model,
        x_feature=payload.x_feature,
        y_feature=payload.y_feature,
        slice_overrides=payload.slice_overrides,
        test_fraction=payload.test_fraction,
        random_seed=payload.random_seed,
    )
    if file_bytes is None:
        raise HTTPException(status_code=400, detail="Contour export could not be built for the selected axes.")

    filename = f"nomogram_dataset_{dataset.id}.xlsx"
    return StreamingResponse(
        BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/datasets/{dataset_id}/repair-forecast", response_model=RepairForecastResponse)
def calculate_repair_forecast(dataset_id: int, payload: RepairForecastRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.records), selectinload(Dataset.trained_models), selectinload(Dataset.column_matches))
        .where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if payload.source_dataset_id is None:
        raise HTTPException(status_code=400, detail="Source dataset id is required for repair forecast.")

    source_dataset = db.scalar(
        select(Dataset)
        .options(selectinload(Dataset.records), selectinload(Dataset.column_matches))
        .where(Dataset.id == payload.source_dataset_id)
    )
    if source_dataset is None:
        raise HTTPException(status_code=404, detail="Source dataset not found.")
    if dataset.storage_section != "fact_epu":
        raise HTTPException(status_code=400, detail="Repair forecast can only be calculated from the 'Факт ЭПУ' dataset.")

    try:
        return build_repair_forecast(
            db,
            dataset,
            source_dataset,
            model_id=payload.model_id,
            base_failure_coefficient=payload.base_failure_coefficient,
            nominal_gap_coefficient=payload.nominal_gap_coefficient,
            manual_feature_values=payload.manual_feature_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/reports", response_model=list[GeneratedReportSummary])
def get_dataset_reports(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return [
        GeneratedReportSummary(
            id=report.id,
            dataset_id=report.dataset_id,
            trained_model_id=report.trained_model_id,
            report_type=report.report_type,
            title=report.title,
            content=report.content,
            llm_used=bool(report.llm_used),
            created_at=report.created_at,
        )
        for report in list_reports(db, dataset_id)
    ]


@router.post("/datasets/{dataset_id}/reports/generate", response_model=GeneratedReportSummary)
def generate_dataset_report(dataset_id: int, payload: ReportGenerateRequest, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).options(selectinload(Dataset.records)).where(Dataset.id == dataset_id)
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    model = None
    if payload.model_id is not None:
        model = db.scalar(select(TrainedModel).where(TrainedModel.id == payload.model_id))
        if model is None or model.dataset_id != dataset_id:
            raise HTTPException(status_code=404, detail="Saved model not found.")

    report = generate_report(
        db,
        dataset,
        dataset.records,
        report_type=payload.report_type,
        model=model,
        use_llm=payload.use_llm,
    )
    return GeneratedReportSummary(
        id=report.id,
        dataset_id=report.dataset_id,
        trained_model_id=report.trained_model_id,
        report_type=report.report_type,
        title=report.title,
        content=report.content,
        llm_used=bool(report.llm_used),
        created_at=report.created_at,
    )
