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

DOMAIN_CANONICAL_RULES = [
    {
        "canonical_name": (
            "Прирост дебита жидкости, м3/сут; Прирост дебита газа, тыс. м3/сут; "
            "Прирост приемистости воды, м3/сут; Закачка газа, тыс м3/сут; "
            "(ВГВ Прирост приемистости газа, тыс. м3/сут)"
        ),
        "required_tokens": {
            "прирост",
            "дебита",
            "жидкости",
            "газа",
            "приемистости",
            "воды",
            "закачка",
        },
        "reasoning": "Matched by explicit GTM composite increment rule.",
    },
]


def normalize_column_name(value: str) -> str:
    text = str(value or "").strip().casefold()
    for source, target in TOKEN_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
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


def get_prior_column_examples(session: Session, dataset: Dataset, limit: int = 160) -> list[dict]:
    datasets = list(
        session.scalars(
            select(Dataset)
            .options(selectinload(Dataset.column_matches))
            .where(Dataset.id != dataset.id)
            .order_by(Dataset.created_at.desc())
            .limit(50)
        ).all()
    )

    examples: list[dict] = []
    seen_headers: set[str] = set()

    for item in datasets:
        match_map = {match.source_column: match.canonical_name for match in item.column_matches}
        for header in list(item.columns_json or []):
            normalized = normalize_column_name(header)
            if not normalized or normalized in seen_headers:
                continue
            seen_headers.add(normalized)
            examples.append(
                {
                    "header": header,
                    "canonical_name": match_map.get(header) or canonicalize_name(header),
                    "dataset_name": item.name,
                    "storage_section": item.storage_section,
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


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
    prior_examples: list[dict] | None = None,
) -> tuple[str, float, str, CanonicalVariable | None]:
    source_tokens = tokenize(source_column)
    for rule in DOMAIN_CANONICAL_RULES:
        if rule["required_tokens"].issubset(source_tokens):
            variable = next(
                (item for item in dictionary if item.canonical_name == rule["canonical_name"]),
                None,
            )
            return (
                rule["canonical_name"],
                0.98,
                rule["reasoning"],
                variable,
            )

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

    best_example = None
    best_example_score = 0.0
    for example in prior_examples or []:
        score = score_similarity(source_column, example["header"])
        if score > best_example_score:
            best_example_score = score
            best_example = example

    if best_example is not None and best_example_score >= 0.78:
        canonical_name = canonicalize_name(best_example["canonical_name"])
        return (
            canonical_name,
            best_example_score,
            (
                "Matched by similarity against previous dataset header "
                f"'{best_example['header']}' from '{best_example['dataset_name']}'."
            ),
            None,
        )

    canonical_name = canonicalize_name(source_column)
    return canonical_name, max(best_score, 0.55), "Heuristic fallback created a new canonical variable.", None


def llm_reconcile(
    columns: list[str],
    dictionary: list[CanonicalVariable],
    prior_examples: list[dict],
) -> tuple[dict[str, dict], bool, str | None]:
    candidate_lines: list[str] = []
    for column in columns:
        variable_candidates: list[tuple[float, str]] = []
        for variable in dictionary:
            aliases = [variable.canonical_name, *[alias.alias_name for alias in variable.aliases]]
            score = max((score_similarity(column, alias) for alias in aliases), default=0.0)
            if score >= 0.35:
                variable_candidates.append(
                    (
                        score,
                        f"canonical={variable.canonical_name} | aliases={', '.join(aliases[:6])}",
                    )
                )

        prior_candidates: list[tuple[float, str]] = []
        for example in prior_examples:
            score = score_similarity(column, example["header"])
            if score >= 0.35:
                prior_candidates.append(
                    (
                        score,
                        f"header={example['header']} => canonical={example['canonical_name']} | dataset={example['dataset_name']}",
                    )
                )

        top_variable_candidates = [item for _, item in sorted(variable_candidates, key=lambda item: item[0], reverse=True)[:3]]
        top_prior_candidates = [item for _, item in sorted(prior_candidates, key=lambda item: item[0], reverse=True)[:3]]
        candidate_lines.append(
            f"- {column}\n"
            + (
                "  possible canonical variables:\n    - "
                + "\n    - ".join(top_variable_candidates)
                if top_variable_candidates else
                "  possible canonical variables:\n    - none"
            )
            + "\n"
            + (
                "  similar previous headers:\n    - "
                + "\n    - ".join(top_prior_candidates)
                if top_prior_candidates else
                "  similar previous headers:\n    - none"
            )
        )

    system_prompt = (
        "You reconcile spreadsheet column names into canonical variables for pump and oilfield analytics. "
        "Return strict JSON with key 'matches'. Each item must include source_column, canonical_name, "
        "confidence, description, reasoning. Prefer canonical names that already exist in known variables "
        "or closely match headers from previous datasets when appropriate."
    )
    user_prompt = (
        "Columns to reconcile with candidate context:\n"
        + "\n".join(candidate_lines)
        + "\n\nGroup semantically equivalent headers under one canonical variable when appropriate. "
        "Reuse an existing canonical variable whenever the meaning matches."
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
    columns: list[str] | None = None,
) -> dict:
    dictionary = get_dictionary(session)
    prior_examples = get_prior_column_examples(session, dataset)
    dataset_columns = list(dataset.columns_json)
    requested_columns = [column for column in (columns or dataset_columns) if column in dataset_columns]
    if not requested_columns:
        requested_columns = dataset_columns
    llm_matches: dict[str, dict] = {}
    llm_used = False
    notes: list[str] = []

    if use_llm:
        llm_matches, llm_used, llm_error = llm_reconcile(requested_columns, dictionary, prior_examples)
        if llm_error:
            notes.append(f"LLaMA fallback to heuristics: {llm_error}")

    if persist:
        existing_matches = get_dataset_matches(session, dataset.id)
        target_columns = set(requested_columns)
        for item in existing_matches:
            if item.source_column in target_columns:
                session.delete(item)
        session.flush()

    match_rows: list[DatasetColumnMatch] = []
    unresolved: list[str] = []

    for column in requested_columns:
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
            canonical_name, confidence, reasoning, variable = heuristic_match(column, dictionary, prior_examples)
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


def clear_variable_dictionary(session: Session) -> dict[str, int]:
    removed_matches = session.query(DatasetColumnMatch).delete()
    removed_aliases = session.query(VariableAlias).delete()
    removed_variables = session.query(CanonicalVariable).delete()
    session.commit()
    return {
        "removed_matches": int(removed_matches or 0),
        "removed_aliases": int(removed_aliases or 0),
        "removed_variables": int(removed_variables or 0),
    }


def summarize_matches_by_canonical(matches: list[DatasetColumnMatch]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for match in matches:
        grouped[match.canonical_name].append(match.source_column)
    return dict(grouped)


def save_manual_dataset_matches(
    session: Session,
    dataset: Dataset,
    manual_matches: list[dict],
) -> list[DatasetColumnMatch]:
    existing_matches = get_dataset_matches(session, dataset.id)
    for item in existing_matches:
        session.delete(item)
    session.flush()

    saved_matches: list[DatasetColumnMatch] = []
    available_columns = set(dataset.columns_json or [])

    for item in manual_matches:
        source_column = str(item.get("source_column") or "").strip()
        canonical_name = canonicalize_name(item.get("canonical_name") or source_column)
        if not source_column or source_column not in available_columns or not canonical_name:
            continue

        variable = get_or_create_canonical_variable(session, canonical_name)
        ensure_alias(session, variable, source_column)
        match = DatasetColumnMatch(
            dataset_id=dataset.id,
            canonical_variable_id=variable.id,
            source_column=source_column,
            canonical_name=canonical_name,
            confidence=1.0,
            reasoning="Manually assigned by user.",
            status="matched",
            llm_used=0,
        )
        session.add(match)
        saved_matches.append(match)

    session.commit()
    for match in saved_matches:
        session.refresh(match)
    return saved_matches
