from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset, GeneratedReport, TrainedModel
from app.services.analysis import build_analysis_response, saved_model_to_summary, to_json_safe
from app.services.llm_client import llm_client
from app.services.variable_mapping import get_dataset_matches, summarize_matches_by_canonical


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return to_json_safe(value)


def build_llm_prompt_payload(payload: dict) -> dict:
    analysis = payload.get("analysis") or {}
    semantic_mapping = payload.get("semantic_mapping") or {}
    return {
        "dataset": payload.get("dataset"),
        "analysis": {
            "overview": analysis.get("overview"),
            "correlations": list(analysis.get("correlations") or [])[:6],
            "feature_importance": list(analysis.get("feature_importance") or [])[:6],
            "notes": list(analysis.get("notes") or [])[:8],
        },
        "semantic_mapping": dict(list(semantic_mapping.items())[:8]),
        "forecast_model": payload.get("forecast_model"),
    }
def build_report_payload_with_session(
    session: Session,
    dataset: Dataset,
    records: list,
    model: TrainedModel | None = None,
) -> dict:
    analysis = build_analysis_response(dataset, records)
    column_matches = get_dataset_matches(session, dataset.id)
    active_model = model

    if active_model is None:
        active_model = session.scalar(
            select(TrainedModel)
            .where(TrainedModel.dataset_id == dataset.id)
            .order_by(TrainedModel.is_active.desc(), TrainedModel.created_at.desc())
            .limit(1)
        )

    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "storage_section": dataset.storage_section,
            "storage_version": dataset.storage_version,
            "target_column": dataset.target_column,
            "selected_features": list(dataset.selected_features_json or []),
            "row_count": dataset.row_count,
        },
        "analysis": {
            "overview": analysis.overview.model_dump(),
            "correlations": [item.model_dump() for item in analysis.correlations[:12]],
            "feature_importance": [item.model_dump() for item in analysis.feature_importance[:12]],
            "notes": analysis.notes,
        },
        "semantic_mapping": summarize_matches_by_canonical(column_matches),
        "forecast_model": saved_model_to_summary(active_model).model_dump() if active_model else None,
    }


def fallback_report_text(payload: dict, report_type: str) -> str:
    dataset = payload["dataset"]
    analysis = payload["analysis"]
    top_features = analysis["feature_importance"][:5]
    top_correlations = analysis["correlations"][:5]
    mapping = payload.get("semantic_mapping") or {}
    model_info = payload.get("forecast_model")

    lines = [
        f"Отчет: {report_type}",
        f"Набор данных: {dataset['name']} ({dataset['storage_section']} v{dataset['storage_version']})",
        f"Целевая переменная: {dataset.get('target_column') or 'не выбрана'}",
        f"Строк: {dataset['row_count']}",
        "",
        "Ключевые факторы CatBoost:",
    ]
    if top_features:
        lines.extend(
            [f"- {item['feature']}: importance {item['importance']:.3f}" for item in top_features]
        )
    else:
        lines.append("- Недостаточно данных для расчета важности факторов.")

    lines.append("")
    lines.append("Ключевые корреляции:")
    if top_correlations:
        for item in top_correlations:
            correlation = item.get("correlation")
            correlation_text = "n/a" if correlation is None else f"{correlation:.3f}"
            lines.append(f"- {item['feature']}: {correlation_text}")
    else:
        lines.append("- Корреляции не определены.")

    if mapping:
        lines.append("")
        lines.append("Согласованные канонические переменные:")
        for canonical_name, aliases in list(mapping.items())[:8]:
            lines.append(f"- {canonical_name}: {', '.join(aliases)}")

    if model_info:
        metrics = model_info["metrics"]
        lines.append("")
        lines.append("Состояние модели прогноза:")
        lines.append(
            f"- RMSE: {metrics['rmse'] if metrics['rmse'] is not None else 'n/a'}, "
            f"MAE: {metrics['mae'] if metrics['mae'] is not None else 'n/a'}, "
            f"R2: {metrics['r2'] if metrics['r2'] is not None else 'n/a'}"
        )

    notes = analysis.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Замечания:")
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines)


def llm_report_text(payload: dict, report_type: str) -> tuple[str | None, bool, str | None]:
    prompt_payload = build_llm_prompt_payload(payload)
    system_prompt = (
        "Ты аналитик производственных данных по насосам и отказам. "
        "Пиши краткий, структурированный отчет на русском языке. "
        "Не выдумывай показатели, используй только входные данные."
    )
    user_prompt = (
        f"Тип отчета: {report_type}\n"
        "Сформируй отчет с разделами: 1) Сводка 2) Главные факторы 3) Риски/аномалии 4) Рекомендации.\n"
        f"Данные:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    return llm_client.chat_text(system_prompt, user_prompt)


def generate_report(
    session: Session,
    dataset: Dataset,
    records: list,
    report_type: str = "analysis_forecast",
    model: TrainedModel | None = None,
    use_llm: bool = True,
) -> GeneratedReport:
    payload = make_json_safe(build_report_payload_with_session(session, dataset, records, model=model))
    llm_used = 0

    if use_llm:
        content, ok, error = llm_report_text(payload, report_type)
        if ok and content:
            report_content = content
            llm_used = 1
        else:
            report_content = fallback_report_text(payload, report_type)
            if error:
                report_content += f"\n\n[LLM fallback reason: {error}]"
    else:
        report_content = fallback_report_text(payload, report_type)

    report = GeneratedReport(
        dataset_id=dataset.id,
        trained_model_id=model.id if model else None,
        report_type=report_type,
        title=f"{dataset.name}: {report_type}",
        content=report_content,
        source_payload_json=payload,
        llm_used=llm_used,
    )
    session.add(report)
    session.commit()
    session.refresh(report)
    return report


def list_reports(session: Session, dataset_id: int) -> list[GeneratedReport]:
    return list(
        session.scalars(
            select(GeneratedReport)
            .where(GeneratedReport.dataset_id == dataset_id)
            .order_by(GeneratedReport.created_at.desc())
        ).all()
    )
