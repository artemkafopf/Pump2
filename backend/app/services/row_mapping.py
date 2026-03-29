from __future__ import annotations

import difflib
import re
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    CanonicalEntityValue,
    Dataset,
    DatasetColumnMatch,
    DatasetEntityMatch,
    EntityValueAlias,
    Record,
)


ENTITY_TYPES = ("license_area", "cluster", "well")
ENTITY_LABELS = {
    "license_area": "Участок недр",
    "cluster": "Куст",
    "well": "Скважина",
}
ENTITY_COLUMN_PATTERNS = {
    "license_area": ("участок недр", "участ", "license", "area", "ун"),
    "cluster": ("куст", "cluster", "pad"),
    "well": ("скваж", "well", "well id", "id скваж"),
}


def normalize_entity_value(value: str) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_entity_dictionary(session: Session, entity_type: str | None = None) -> list[CanonicalEntityValue]:
    query = select(CanonicalEntityValue).options(selectinload(CanonicalEntityValue.aliases))
    if entity_type:
        query = query.where(CanonicalEntityValue.entity_type == entity_type)
    query = query.order_by(CanonicalEntityValue.entity_type.asc(), CanonicalEntityValue.canonical_value.asc())
    return list(session.scalars(query).all())


def get_dataset_entity_matches(session: Session, dataset_id: int) -> list[DatasetEntityMatch]:
    return list(
        session.scalars(
            select(DatasetEntityMatch)
            .options(selectinload(DatasetEntityMatch.canonical_entity))
            .where(DatasetEntityMatch.dataset_id == dataset_id)
            .order_by(DatasetEntityMatch.entity_type.asc(), DatasetEntityMatch.source_value.asc())
        ).all()
    )


def score_similarity(left: str, right: str) -> float:
    normalized_left = normalize_entity_value(left)
    normalized_right = normalize_entity_value(right)
    ratio = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(ratio, overlap)


def get_or_create_canonical_entity(session: Session, entity_type: str, canonical_value: str) -> CanonicalEntityValue:
    normalized = normalize_entity_value(canonical_value)
    existing = session.scalar(
        select(CanonicalEntityValue).where(CanonicalEntityValue.normalized_value == normalized)
    )
    if existing is not None:
        return existing

    created = CanonicalEntityValue(
        entity_type=entity_type,
        canonical_value=canonical_value.strip(),
        normalized_value=normalized,
    )
    session.add(created)
    session.flush()
    return created


def ensure_entity_alias(session: Session, canonical_entity: CanonicalEntityValue, alias_value: str) -> None:
    normalized = normalize_entity_value(alias_value)
    if not normalized:
        return
    existing = session.scalar(select(EntityValueAlias).where(EntityValueAlias.normalized_alias == normalized))
    if existing is not None:
        return
    session.add(
        EntityValueAlias(
            canonical_entity_id=canonical_entity.id,
            alias_value=alias_value,
            normalized_alias=normalized,
        )
    )


def resolve_entity_columns(dataset: Dataset) -> dict[str, str | None]:
    match_map = {item.source_column: item.canonical_name for item in dataset.column_matches}
    resolved: dict[str, str | None] = {entity_type: None for entity_type in ENTITY_TYPES}

    for column in list(dataset.columns_json or []):
        searchable = " ".join(filter(None, [column, match_map.get(column)])).casefold()
        for entity_type, patterns in ENTITY_COLUMN_PATTERNS.items():
            if resolved[entity_type] is not None:
                continue
            if any(pattern in searchable for pattern in patterns):
                resolved[entity_type] = column

    return resolved


def collect_entity_values(dataset: Dataset) -> dict[str, list[str]]:
    resolved_columns = resolve_entity_columns(dataset)
    rows = [record.payload for record in dataset.records]
    values_by_type: dict[str, list[str]] = {}

    for entity_type, column in resolved_columns.items():
        if not column:
            values_by_type[entity_type] = []
            continue
        seen: set[str] = set()
        values: list[str] = []
        for row in rows:
            value = str(row.get(column) or "").strip()
            normalized = normalize_entity_value(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(value)
        values_by_type[entity_type] = values
    return values_by_type


def heuristic_entity_match(
    entity_type: str,
    source_value: str,
    dictionary: list[CanonicalEntityValue],
) -> tuple[str, float, str, CanonicalEntityValue | None]:
    best_entity = None
    best_score = 0.0
    best_basis = None
    for item in dictionary:
        names = [item.canonical_value, *[alias.alias_value for alias in item.aliases]]
        for candidate in names:
            score = score_similarity(source_value, candidate)
            if score > best_score:
                best_score = score
                best_basis = candidate
                best_entity = item

    if best_entity is not None and best_score >= 0.8:
        return best_entity.canonical_value, best_score, f"Matched against '{best_basis}'.", best_entity

    return source_value.strip(), max(best_score, 0.55), f"Created candidate {ENTITY_LABELS[entity_type]} from source value.", None


def reconcile_dataset_entities(session: Session, dataset: Dataset, persist: bool = True) -> dict:
    values_by_type = collect_entity_values(dataset)
    if persist:
        for item in get_dataset_entity_matches(session, dataset.id):
            session.delete(item)
        session.flush()

    match_rows: list[DatasetEntityMatch] = []
    unresolved: dict[str, list[str]] = defaultdict(list)

    for entity_type in ENTITY_TYPES:
        dictionary = get_entity_dictionary(session, entity_type=entity_type)
        for source_value in values_by_type.get(entity_type, []):
            canonical_value, confidence, reasoning, entity = heuristic_entity_match(entity_type, source_value, dictionary)
            if confidence < 0.65:
                unresolved[entity_type].append(source_value)
            status = "matched" if confidence >= 0.8 else "review"

            if persist:
                if entity is None:
                    entity = get_or_create_canonical_entity(session, entity_type, canonical_value)
                ensure_entity_alias(session, entity, source_value)
                match = DatasetEntityMatch(
                    dataset_id=dataset.id,
                    entity_type=entity_type,
                    source_value=source_value,
                    canonical_entity_id=entity.id,
                    canonical_value=entity.canonical_value,
                    confidence=confidence,
                    reasoning=reasoning,
                    status=status,
                )
                session.add(match)
            else:
                match = DatasetEntityMatch(
                    dataset_id=dataset.id,
                    entity_type=entity_type,
                    source_value=source_value,
                    canonical_entity_id=entity.id if entity else None,
                    canonical_value=canonical_value,
                    confidence=confidence,
                    reasoning=reasoning,
                    status=status,
                )
            match_rows.append(match)

    if persist:
        session.commit()
        for item in match_rows:
            session.refresh(item)

    return {
        "matches": match_rows,
        "unresolved_values": dict(unresolved),
        "resolved_columns": resolve_entity_columns(dataset),
    }


def save_manual_entity_matches(session: Session, dataset: Dataset, manual_matches: list[dict]) -> list[DatasetEntityMatch]:
    for item in get_dataset_entity_matches(session, dataset.id):
        session.delete(item)
    session.flush()

    saved: list[DatasetEntityMatch] = []
    valid_values = collect_entity_values(dataset)

    for item in manual_matches:
        entity_type = str(item.get("entity_type") or "").strip()
        source_value = str(item.get("source_value") or "").strip()
        canonical_value = str(item.get("canonical_value") or "").strip()
        if entity_type not in ENTITY_TYPES or not source_value or not canonical_value:
            continue
        if source_value not in valid_values.get(entity_type, []):
            continue

        entity = get_or_create_canonical_entity(session, entity_type, canonical_value)
        ensure_entity_alias(session, entity, source_value)
        match = DatasetEntityMatch(
            dataset_id=dataset.id,
            entity_type=entity_type,
            source_value=source_value,
            canonical_entity_id=entity.id,
            canonical_value=entity.canonical_value,
            confidence=1.0,
            reasoning="Manually assigned by user.",
            status="matched",
        )
        session.add(match)
        saved.append(match)

    session.commit()
    for item in saved:
        session.refresh(item)
    return saved


def entity_dictionary_snapshot(session: Session) -> list[dict]:
    result = []
    for entity in get_entity_dictionary(session):
        result.append(
            {
                "id": entity.id,
                "entity_type": entity.entity_type,
                "canonical_value": entity.canonical_value,
                "aliases": [alias.alias_value for alias in entity.aliases],
            }
        )
    return result
