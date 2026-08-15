"""LLM-assisted worksheet column reconciliation with deterministic fallback."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .operating_data import CANONICAL_ALIASES, normalize_alias_token


STOP_TOKENS = {
    "and",
    "or",
    "for",
    "the",
    "of",
    "value",
    "rate",
    "parameter",
    "column",
    "field",
}

TOKEN_REPLACEMENTS = {
    "deb.": "debit",
    "deb": "debit",
    "дав.": "давление",
    "заб.": "забой",
    "qж": "qliq",
    "qжид": "qliq",
    "qgas": "qgas",
    "гжф": "glf",
    "рзаб": "pbhp",
}

JsonResolver = Callable[[str, str], tuple[dict[str, Any] | None, bool, str | None]]


@dataclass(frozen=True)
class ReconciliationMatch:
    source_column: str
    canonical_name: str
    confidence: float
    description: str | None
    reasoning: str
    method: str


@dataclass(frozen=True)
class ReconciliationResult:
    matches: tuple[ReconciliationMatch, ...]
    llm_used: bool
    llm_error: str | None

    @property
    def mapping(self) -> dict[str, str]:
        return {item.source_column: item.canonical_name for item in self.matches}


def normalize_reconciliation_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for source, target in TOKEN_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[_/\\\-|]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_reconciliation_name(value: Any) -> set[str]:
    normalized = normalize_reconciliation_name(value)
    return {token for token in normalized.split() if token and token not in STOP_TOKENS}


def score_column_similarity(left: str, right: str) -> float:
    normalized_left = normalize_reconciliation_name(left)
    normalized_right = normalize_reconciliation_name(right)
    sequence_ratio = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
    tokens_left = tokenize_reconciliation_name(left)
    tokens_right = tokenize_reconciliation_name(right)
    token_overlap = len(tokens_left & tokens_right) / max(len(tokens_left | tokens_right), 1)
    return max(sequence_ratio, token_overlap)


def canonicalize_header_name(value: Any) -> str:
    normalized = normalize_reconciliation_name(value)
    if not normalized:
        return str(value or "").strip()
    return " ".join(part.capitalize() for part in normalized.split())


def build_canonical_dictionary(
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    alias_map = aliases or CANONICAL_ALIASES
    dictionary: list[dict[str, Any]] = []
    for canonical_name, alias_values in alias_map.items():
        dictionary.append(
            {
                "canonical_name": canonical_name,
                "description": None,
                "aliases": list(dict.fromkeys([canonical_name, *alias_values])),
            }
        )
    return dictionary


def _best_dictionary_candidates(
    source_column: str,
    dictionary: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in dictionary:
        aliases = [str(alias) for alias in item.get("aliases", []) if str(alias).strip()]
        best_score = max((score_column_similarity(source_column, alias) for alias in aliases), default=0.0)
        if best_score >= threshold:
            candidates.append(
                {
                    "canonical_name": str(item["canonical_name"]),
                    "aliases": aliases,
                    "score": best_score,
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def _best_prior_examples(
    source_column: str,
    prior_examples: Sequence[Mapping[str, Any]] | None,
    *,
    limit: int = 3,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in prior_examples or []:
        header = str(item.get("header") or "").strip()
        canonical_name = str(item.get("canonical_name") or "").strip()
        if not header or not canonical_name:
            continue
        score = score_column_similarity(source_column, header)
        if score >= threshold:
            candidates.append(
                {
                    "header": header,
                    "canonical_name": canonical_name,
                    "dataset_name": item.get("dataset_name"),
                    "score": score,
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def build_reconciliation_prompt(
    columns: Sequence[str],
    *,
    dictionary: Sequence[Mapping[str, Any]],
    prior_examples: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str]:
    candidate_lines: list[str] = []
    for column in columns:
        dictionary_candidates = _best_dictionary_candidates(column, dictionary)
        prior_candidates = _best_prior_examples(column, prior_examples)
        candidate_lines.append(
            f"- {column}\n"
            + (
                "  possible canonical variables:\n    - "
                + "\n    - ".join(
                    f"canonical={item['canonical_name']} | aliases={', '.join(item['aliases'][:6])}"
                    for item in dictionary_candidates
                )
                if dictionary_candidates
                else "  possible canonical variables:\n    - none"
            )
            + "\n"
            + (
                "  similar previous headers:\n    - "
                + "\n    - ".join(
                    f"header={item['header']} => canonical={item['canonical_name']}"
                    + (
                        f" | dataset={item['dataset_name']}"
                        if item.get("dataset_name")
                        else ""
                    )
                    for item in prior_candidates
                )
                if prior_candidates
                else "  similar previous headers:\n    - none"
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
    return system_prompt, user_prompt


def heuristic_reconcile_column(
    source_column: str,
    *,
    dictionary: Sequence[Mapping[str, Any]],
    prior_examples: Sequence[Mapping[str, Any]] | None = None,
) -> ReconciliationMatch:
    best_dictionary = _best_dictionary_candidates(source_column, dictionary, limit=1, threshold=0.0)
    if best_dictionary and best_dictionary[0]["score"] >= 0.8:
        item = best_dictionary[0]
        return ReconciliationMatch(
            source_column=source_column,
            canonical_name=item["canonical_name"],
            confidence=float(item["score"]),
            description=None,
            reasoning=f"Matched by heuristic similarity against known aliases for '{item['canonical_name']}'.",
            method="heuristic_dictionary",
        )

    best_prior = _best_prior_examples(source_column, prior_examples, limit=1, threshold=0.0)
    if best_prior and best_prior[0]["score"] >= 0.78:
        item = best_prior[0]
        return ReconciliationMatch(
            source_column=source_column,
            canonical_name=str(item["canonical_name"]),
            confidence=float(item["score"]),
            description=None,
            reasoning=(
                f"Matched by similarity against previous header '{item['header']}'"
                + (f" from '{item['dataset_name']}'." if item.get("dataset_name") else ".")
            ),
            method="heuristic_prior_example",
        )

    best_score = best_dictionary[0]["score"] if best_dictionary else 0.0
    fallback_name = canonicalize_header_name(source_column)
    return ReconciliationMatch(
        source_column=source_column,
        canonical_name=fallback_name,
        confidence=float(max(best_score, 0.55)),
        description=None,
        reasoning="Heuristic fallback created a new canonical variable candidate.",
        method="heuristic_new",
    )


def parse_llm_reconciliation_payload(payload: Mapping[str, Any]) -> dict[str, ReconciliationMatch]:
    matches: dict[str, ReconciliationMatch] = {}
    for item in payload.get("matches", []):
        source_column = str(item.get("source_column") or "").strip()
        if not source_column:
            continue
        canonical_name = str(item.get("canonical_name") or "").strip() or canonicalize_header_name(source_column)
        match = ReconciliationMatch(
            source_column=source_column,
            canonical_name=canonical_name,
            confidence=float(item.get("confidence") or 0.0),
            description=str(item.get("description") or "").strip() or None,
            reasoning=str(item.get("reasoning") or "").strip() or "Matched by local LLaMA.",
            method="llm",
        )
        matches[source_column] = match
    return matches


def reconcile_worksheet_columns(
    columns: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    prior_examples: Sequence[Mapping[str, Any]] | None = None,
    llm_resolver: JsonResolver | None = None,
    use_llm: bool = True,
) -> ReconciliationResult:
    dictionary = build_canonical_dictionary(aliases=aliases)
    llm_matches: dict[str, ReconciliationMatch] = {}
    llm_used = False
    llm_error: str | None = None

    if use_llm and llm_resolver is not None:
        system_prompt, user_prompt = build_reconciliation_prompt(
            columns,
            dictionary=dictionary,
            prior_examples=prior_examples,
        )
        payload, ok, error = llm_resolver(system_prompt, user_prompt)
        if ok and payload:
            llm_matches = parse_llm_reconciliation_payload(payload)
            llm_used = True
        else:
            llm_error = error or "Unknown LLM reconciliation failure."

    matches: list[ReconciliationMatch] = []
    for column in columns:
        if column in llm_matches:
            matches.append(llm_matches[column])
        else:
            matches.append(
                heuristic_reconcile_column(
                    column,
                    dictionary=dictionary,
                    prior_examples=prior_examples,
                )
            )
    return ReconciliationResult(
        matches=tuple(matches),
        llm_used=llm_used,
        llm_error=llm_error,
    )


def build_prior_examples_from_mappings(
    mapping_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in mapping_rows:
        header = str(row.get("header") or row.get("source_column") or "").strip()
        canonical_name = str(row.get("canonical_name") or "").strip()
        if not header or not canonical_name:
            continue
        normalized = normalize_alias_token(header)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        examples.append(
            {
                "header": header,
                "canonical_name": canonical_name,
                "dataset_name": row.get("dataset_name"),
            }
        )
    return examples


def reconciliation_result_to_json(result: ReconciliationResult) -> str:
    return json.dumps(
        {
            "llm_used": result.llm_used,
            "llm_error": result.llm_error,
            "matches": [
                {
                    "source_column": item.source_column,
                    "canonical_name": item.canonical_name,
                    "confidence": item.confidence,
                    "description": item.description,
                    "reasoning": item.reasoning,
                    "method": item.method,
                }
                for item in result.matches
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
