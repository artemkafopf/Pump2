from __future__ import annotations

import difflib
import re
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import CanonicalVariable, Dataset, DatasetColumnMatch, VariableAlias
from app.services.llm_client import llm_client


STOP_TOKENS = {
    "and",
    "or",
    "for",
    "the",
    "of",
    "value",
    "rate",
    "param",
    "parameter",
}

TOKEN_REPLACEMENTS = {
    "deb.": "debit",
    "deb": "debit",
    "davl.": "pressure",
    "zab.": "bottomhole",
    "qn": "oil rate",
    "qzh": "fluid rate",
    "qj": "fluid rate",
    "priem": "injectivity",
    "watercut": "water cut",
}


def normalize_column_name(value: str) -> str:
    text = str(value or "").strip().casefold()
    for source, target in TOKEN_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"[^0-9a-z\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: str) -> set[str]:
    normalized = normalize_column_name(value)
    return {token for token in normalized.split() if token and token not in STOP_TOKENS}


def score_similarity(left: str, right: str) -> float:
    normalized_left = normalize_column_name(left)
    normalized_right = normalize_column_name(right)
    ratio = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
    tokens_left = tokenize(left)
    tokens_right = tokenize(right)
    overlap = len(tokens_left & tokens_right) / max(len(tokens_left | tokens_right), 1)
    return max(ratio, overlap)


def canonicalize_name(value: str) -> str:
    normalized = normalize_column_name(value)
    if not normalized:
        return str(value or "").strip()
    return " ".join(part.capitalize() for part in normalized.split())


def get_dictionary(session: Session) -> list[CanonicalVariable]:
    return list(
        session.scalars(
            select(CanonicalVariable)
            .options(selectinload(CanonicalVariable.aliases))
            .order_by(CanonicalVariable.canonical_name.asc())
        ).all()
    )


def get_dataset_matches(session: Session, dataset_id: int) -> list[DatasetColumnMatch]:
    return list(
        session.scalars(
            select(DatasetColumnMatch)
            .options(selectinload(DatasetColumnMatch.canonical_variable))
            .where(DatasetColumnMatch.dataset_id == dataset_id)
            .order_by(DatasetColumnMatch.source_column.asc())
        ).all()
    )


def get_or_create_canonical_variable(
    session: Session,
    canonical_name: str,
    description: str | None = None,
) -> CanonicalVariable:
    existing = session.scalar(
        select(CanonicalVariable).where(CanonicalVariable.canonical_name == canonical_name)
    )
    if existing is not None:
        if description and not existing.description:
            existing.description = description
            session.add(existing)
        return existing

    created = CanonicalVariable(
        canonical_name=canonical_name,
        description=description,
    )
    session.add(created)
    session.flush()
    return created


def ensure_alias(session: Session, canonical_variable: CanonicalVariable, alias_name: str) -> None:
    normalized = normalize_column_name(alias_name)
    if not normalized:
        return

    existing = session.scalar(
        select(VariableAlias).where(VariableAlias.normalized_alias == normalized)
    )
    if existing is not None:
        return

    session.add(
        VariableAlias(
            canonical_variable_id=canonical_variable.id,
            alias_name=alias_name,
            normalized_alias=normalized,
        )
    )


def heuristic_match(
    source_column: str,
    dictionary: list[CanonicalVariable],
) -> tuple[str, float, str, CanonicalVariable | None]:
    if not dictionary:
        canonical_name = canonicalize_name(source_column)
        return canonical_name, 0.55, "New canonical variable created from source column.", None

    best_variable: CanonicalVariable | None = None
    best_score = 0.0
    best_basis = ""

    for variable in dictionary:
        names = [variable.canonical_name, *[alias.alias_name for alias in variable.aliases]]
        for candidate in names:
            score = score_similarity(source_column, candidate)
            if score > best_score:
                best_score = score
                best_variable = variable
                best_basis = candidate

    if best_variable is not None and best_score >= 0.8:
        return (
            best_variable.canonical_name,
            best_score,
            f"Matched by heuristic similarity against '{best_basis}'.",
            best_variable,
        )

    canonical_name = canonicalize_name(source_column)
    return canonical_name, max(best_score, 0.55), "Heuristic fallback created a new canonical variable.", None


def llm_reconcile(columns: list[str]) -> tuple[dict[str, dict], bool, str | None]:
    system_prompt = (
        "You reconcile spreadsheet column names into canonical variables for pump and oilfield analytics. "
        "Return strict JSON with key 'matches'. Each item must include source_column, canonical_name, "
        "confidence, description, reasoning."
    )
    user_prompt = (
        "Columns:\n"
        + "\n".join(f"- {column}" for column in columns)
        + "\n\nGroup semantically equivalent headers under one canonical variable when appropriate."
    )
    payload, ok, error = llm_client.chat_json(system_prompt, user_prompt)
    if not ok or not payload:
        return {}, False, error

    result: dict[str, dict] = {}
    for item in payload.get("matches", []):
        source = str(item.get("source_column") or "").strip()
        if not source:
            continue
        result[source] = {
            "canonical_name": canonicalize_name(item.get("canonical_name") or source),
            "confidence": float(item.get("confidence") or 0.0),
            "description": str(item.get("description") or "").strip() or None,
            "reasoning": str(item.get("reasoning") or "").strip() or "Matched by local LLaMA.",
        }
    return result, True, None


def reconcile_dataset_columns(
    session: Session,
    dataset: Dataset,
    persist: bool = True,
    use_llm: bool = True,
) -> dict:
    dictionary = get_dictionary(session)
    llm_matches: dict[str, dict] = {}
    llm_used = False
    notes: list[str] = []

    if use_llm:
        llm_matches, llm_used, llm_error = llm_reconcile(list(dataset.columns_json))
        if llm_error:
            notes.append(f"LLaMA fallback to heuristics: {llm_error}")

    if persist:
        for item in get_dataset_matches(session, dataset.id):
            session.delete(item)
        session.flush()

    match_rows: list[DatasetColumnMatch] = []
    unresolved: list[str] = []

    for column in list(dataset.columns_json):
        if column in llm_matches:
            llm_item = llm_matches[column]
            canonical_name = canonicalize_name(llm_item["canonical_name"])
            confidence = float(min(max(llm_item["confidence"], 0.0), 1.0))
            reasoning = llm_item.get("reasoning") or "Matched by local LLaMA."
            variable = session.scalar(
                select(CanonicalVariable).where(CanonicalVariable.canonical_name == canonical_name)
            )
            description = llm_item.get("description")
        else:
            canonical_name, confidence, reasoning, variable = heuristic_match(column, dictionary)
            description = None

        status = "matched" if confidence >= 0.8 else "review"
        if confidence < 0.65:
            unresolved.append(column)

        if persist:
            if variable is None:
                variable = get_or_create_canonical_variable(session, canonical_name, description=description)
            ensure_alias(session, variable, column)
            match = DatasetColumnMatch(
                dataset_id=dataset.id,
                canonical_variable_id=variable.id,
                source_column=column,
                canonical_name=canonical_name,
                confidence=confidence,
                reasoning=reasoning,
                status=status,
                llm_used=1 if column in llm_matches else 0,
            )
            session.add(match)
            match_rows.append(match)
        else:
            match_rows.append(
                DatasetColumnMatch(
                    dataset_id=dataset.id,
                    canonical_variable_id=variable.id if variable else None,
                    source_column=column,
                    canonical_name=canonical_name,
                    confidence=confidence,
                    reasoning=reasoning,
                    status=status,
                    llm_used=1 if column in llm_matches else 0,
                )
            )

    if persist:
        session.commit()
        for match in match_rows:
            session.refresh(match)

    return {
        "matches": match_rows,
        "unresolved_columns": unresolved,
        "llm_used": llm_used,
        "notes": notes,
    }


def seed_default_dictionary(session: Session) -> None:
    if session.scalar(select(CanonicalVariable.id).limit(1)) is not None:
        return

    defaults = {
        "Oil Rate": ["oil rate", "debit nefti", "qn"],
        "Fluid Rate": ["fluid rate", "debit zhidkosti", "qzh"],
        "Bottomhole Pressure": ["bottomhole pressure", "davl zab"],
        "Injectivity": ["injectivity", "coefficient injectivity", "priemistost"],
        "Water Cut": ["water cut", "water percent"],
        "Date": ["date", "timestamp", "time"],
        "Year": ["year"],
    }

    for canonical_name, aliases in defaults.items():
        variable = get_or_create_canonical_variable(session, canonical_name)
        seen: set[str] = set()
        for alias in aliases:
            normalized = normalize_column_name(alias)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ensure_alias(session, variable, alias)
    session.commit()


def dictionary_snapshot(session: Session) -> list[dict]:
    result = []
    for variable in get_dictionary(session):
        result.append(
            {
                "id": variable.id,
                "canonical_name": variable.canonical_name,
                "description": variable.description,
                "aliases": [alias.alias_name for alias in variable.aliases],
            }
        )
    return result


def summarize_matches_by_canonical(matches: list[DatasetColumnMatch]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for match in matches:
        grouped[match.canonical_name].append(match.source_column)
    return dict(grouped)
