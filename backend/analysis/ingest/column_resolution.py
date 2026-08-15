"""Unified, source-agnostic column resolution.

Ingestion boundaries across this repo each re-invented column matching: exact
Russian-header lookups here, substring anchors there, positional guesses
elsewhere. When a source export renames a column (an extra dot, a reworded
label, a reordering) the ad-hoc paths silently drop the data.

This module centralizes the decision. Given the headers of a source file and a
spec of ``{canonical: (synonyms, required?)}``, it resolves each header to a
canonical name through three ordered strategies -- exact, synonym, then a
bounded fuzzy match -- and returns a :class:`ColumnMappingReport` that records
*how* every header was matched (or that it was left UNMAPPED). Required columns
that fail to resolve raise a clear error naming the file and the nearest misses;
optional columns that fail to resolve warn.

The fuzzy strategy reuses the scored matcher already implemented in
:mod:`analysis.ingest.worksheet_reconciliation` (difflib ratio + token overlap) and
requires a *unique* best match at or above the threshold -- ambiguity is treated
as unresolved rather than guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

from .normalize import strip_header_prefix


logger = logging.getLogger(__name__)


def score_column_similarity(left: str, right: str) -> float:
    """Scored header similarity, imported lazily to avoid an import cycle.

    ``worksheet_reconciliation`` pulls in ``operating_data`` at module load,
    which imports the ``failure_update`` package -- which imports this module.
    Deferring the import to call time breaks that cycle while still reusing the
    single scoring implementation.
    """
    from .worksheet_reconciliation import score_column_similarity as _score

    return _score(left, right)


def _token_count(value: str) -> int:
    from .worksheet_reconciliation import tokenize_reconciliation_name

    return len(tokenize_reconciliation_name(value))


def _same_token_shape(header: str, term: str) -> bool:
    """True when header and term have the same number of content tokens.

    Fuzzy matching is meant to absorb typos, spacing and punctuation drift
    ("Скв ." -> "Скв.", "останова" -> "остановки") -- not a header that carries
    an *extra* qualifying word which changes its meaning ("Тип скважины" is not
    "Скважина"). Requiring equal content-token counts rejects the latter while
    still accepting same-shape rewordings.
    """
    return _token_count(header) == _token_count(term)

# Fuzzy score at/above which a unique best match is accepted.
DEFAULT_FUZZY_THRESHOLD = 0.8
# Small epsilon so two fuzzy candidates within this band count as a tie
# (ambiguous -> unresolved) rather than an arbitrary winner.
_TIE_EPSILON = 1e-6
UNMAPPED = "UNMAPPED"


class ResolutionMethod(str, Enum):
    """How a source header was matched to its canonical name."""

    EXACT = "exact"
    SYNONYM = "synonym"
    FUZZY = "fuzzy"
    UNMAPPED = "unmapped"


@dataclass(frozen=True)
class ColumnSpec:
    """A single canonical column and the source headers that denote it."""

    canonical: str
    synonyms: tuple[str, ...] = ()
    required: bool = False

    @property
    def match_terms(self) -> tuple[str, ...]:
        """Canonical name plus synonyms, de-duplicated, canonical first."""
        seen: list[str] = []
        for term in (self.canonical, *self.synonyms):
            if term is None:
                continue
            if term not in seen:
                seen.append(term)
        return tuple(seen)


@dataclass(frozen=True)
class ColumnResolution:
    """The outcome for one source header."""

    source_header: str
    canonical: Optional[str]
    method: ResolutionMethod
    score: float
    candidates: tuple[tuple[str, float], ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.canonical is not None

    @property
    def is_exact(self) -> bool:
        return self.method is ResolutionMethod.EXACT


class MissingRequiredColumnError(ValueError):
    """Raised when a required canonical column cannot be resolved."""


def _method_rank(method: ResolutionMethod) -> int:
    return {
        ResolutionMethod.EXACT: 3,
        ResolutionMethod.SYNONYM: 2,
        ResolutionMethod.FUZZY: 1,
        ResolutionMethod.UNMAPPED: 0,
    }[method]


@dataclass
class ColumnMappingReport:
    """Per-header resolution outcomes for one source file."""

    resolutions: tuple[ColumnResolution, ...]
    specs: Mapping[str, ColumnSpec]
    dataset: Optional[str] = None
    source_file: Optional[str] = None

    @property
    def resolved_canonicals(self) -> set[str]:
        return {r.canonical for r in self.resolutions if r.canonical is not None}

    @property
    def mapping(self) -> dict[str, str]:
        """source header -> canonical, for every resolved header."""
        return {
            r.source_header: r.canonical
            for r in self.resolutions
            if r.canonical is not None
        }

    def unmapped_headers(self) -> list[ColumnResolution]:
        return [r for r in self.resolutions if r.canonical is None]

    def non_exact_resolutions(self) -> list[ColumnResolution]:
        return [r for r in self.resolutions if not r.is_exact]

    def rename_map(self) -> dict[str, str]:
        """A collision-safe ``df.rename`` map: source header -> canonical.

        When several headers resolve to the same canonical, only the single best
        (exact > synonym > fuzzy, then score, preferring a header already equal to
        the canonical) is renamed; the rest are left untouched so the rename can
        never create duplicate columns. Headers whose source already equals the
        canonical are omitted (nothing to rename).
        """
        best: dict[str, ColumnResolution] = {}
        for res in self.resolutions:
            if res.canonical is None:
                continue
            current = best.get(res.canonical)
            if current is None or _priority(res) > _priority(current):
                best[res.canonical] = res
        return {
            res.source_header: canonical
            for canonical, res in best.items()
            if res.source_header != canonical
        }

    def apply(self, frame):
        """Return ``frame`` with columns renamed to canonical names."""
        rename = self.rename_map()
        return frame.rename(columns=rename) if rename else frame

    def missing_required(self) -> list[tuple[ColumnSpec, list[tuple[str, float]]]]:
        """Required specs that did not resolve, with nearest source-header misses."""
        resolved = self.resolved_canonicals
        headers = [r.source_header for r in self.resolutions]
        missing: list[tuple[ColumnSpec, list[tuple[str, float]]]] = []
        for spec in self.specs.values():
            if not spec.required or spec.canonical in resolved:
                continue
            missing.append((spec, _nearest_headers(spec, headers)))
        return missing

    def raise_if_required_missing(self) -> None:
        missing = self.missing_required()
        if not missing:
            return
        where = self.source_file or self.dataset or "source"
        parts: list[str] = []
        for spec, candidates in missing:
            if candidates:
                nearest = ", ".join(f"'{header}' ({score:.2f})" for header, score in candidates)
            else:
                nearest = "no similar headers found"
            parts.append(f"'{spec.canonical}' (nearest: {nearest})")
        raise MissingRequiredColumnError(
            f"{where}: could not resolve required column(s): " + "; ".join(parts)
        )

    def warn_unresolved_optional(self) -> None:
        """Warn (once per header) about optional columns that did not resolve.

        Only headers with a plausible near-miss are worth surfacing; a completely
        unrelated extra column is expected noise and is left to the report sheet.
        """
        where = self.source_file or self.dataset or "source"
        for res in self.unmapped_headers():
            if res.candidates and res.candidates[0][1] >= 0.5:
                nearest = ", ".join(f"'{name}' ({score:.2f})" for name, score in res.candidates)
                logger.warning(
                    "%s: column '%s' left UNMAPPED (nearest: %s)",
                    where,
                    res.source_header,
                    nearest,
                )

    def format_lines(self) -> list[str]:
        """One human-readable line per non-exact resolution (for logs)."""
        where = self.source_file or self.dataset or "source"
        lines: list[str] = []
        for res in self.non_exact_resolutions():
            if res.canonical is not None:
                lines.append(
                    f"  [{where}] '{res.source_header}' -> '{res.canonical}' "
                    f"({res.method.value}, score={res.score:.2f})"
                )
            else:
                nearest = (
                    ", ".join(f"'{name}' {score:.2f}" for name, score in res.candidates)
                    if res.candidates
                    else "none"
                )
                lines.append(
                    f"  [{where}] UNMAPPED '{res.source_header}' (nearest: {nearest})"
                )
        return lines

    def log(self) -> None:
        """Print the per-file mapping report (non-exact resolutions only)."""
        for line in self.format_lines():
            print(line)

    def to_records(self) -> list[dict]:
        """Flat records for the audit report."""
        records: list[dict] = []
        for res in self.resolutions:
            records.append(
                {
                    "dataset": self.dataset,
                    "source_file": self.source_file,
                    "source_header": res.source_header,
                    "canonical": res.canonical or UNMAPPED,
                    "method": res.method.value,
                    "score": round(res.score, 3),
                }
            )
        return records


def _priority(res: ColumnResolution) -> tuple[int, float, int]:
    identity = 1 if res.canonical is not None and res.source_header == res.canonical else 0
    return (_method_rank(res.method), res.score, identity)


def _nearest_headers(
    spec: ColumnSpec, headers: Sequence[str], *, limit: int = 3
) -> list[tuple[str, float]]:
    scored = [
        (header, max((score_column_similarity(header, term) for term in spec.match_terms), default=0.0))
        for header in headers
    ]
    scored = [item for item in scored if item[1] > 0.0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def build_specs(
    canonical_to_synonyms: Mapping[str, Sequence[str]],
    *,
    required: Iterable[str] = (),
) -> dict[str, ColumnSpec]:
    """Build a ``{canonical: ColumnSpec}`` map from a synonym table."""
    required_set = set(required)
    specs: dict[str, ColumnSpec] = {}
    for canonical, synonyms in canonical_to_synonyms.items():
        specs[canonical] = ColumnSpec(
            canonical=canonical,
            synonyms=tuple(synonyms or ()),
            required=canonical in required_set,
        )
    return specs


def resolve_columns(
    headers: Sequence[str],
    specs: Mapping[str, ColumnSpec],
    *,
    dataset: Optional[str] = None,
    source_file: Optional[str] = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> ColumnMappingReport:
    """Resolve source ``headers`` against ``specs``.

    Each header is matched by, in order: exact (normalized equality to a
    canonical name), synonym (normalized equality to a listed synonym), then a
    bounded fuzzy match requiring a unique best canonical at/above
    ``fuzzy_threshold``. Ambiguous fuzzy matches are left UNMAPPED.
    """
    spec_list = list(specs.values())
    # Pre-compute normalized match terms per canonical.
    normalized_terms: dict[str, set[str]] = {}
    normalized_canonical: dict[str, str] = {}
    for spec in spec_list:
        normalized_canonical[spec.canonical] = strip_header_prefix(spec.canonical)
        normalized_terms[spec.canonical] = {strip_header_prefix(term) for term in spec.match_terms}

    resolutions: list[ColumnResolution] = []
    for header in headers:
        norm_header = strip_header_prefix(header)

        exact_hits = [
            spec.canonical for spec in spec_list if norm_header == normalized_canonical[spec.canonical]
        ]
        if len(exact_hits) >= 1:
            resolutions.append(
                ColumnResolution(str(header), exact_hits[0], ResolutionMethod.EXACT, 1.0)
            )
            continue

        synonym_hits = [
            spec.canonical for spec in spec_list if norm_header in normalized_terms[spec.canonical]
        ]
        if len(set(synonym_hits)) == 1:
            resolutions.append(
                ColumnResolution(str(header), synonym_hits[0], ResolutionMethod.SYNONYM, 1.0)
            )
            continue

        # Fuzzy: per canonical, the best similarity across its terms plus whether
        # any at/above-threshold term also passes the token-shape gate.
        scored: list[tuple[str, float, bool]] = []
        for spec in spec_list:
            best_score = 0.0
            gate_ok = False
            for term in spec.match_terms:
                similarity = score_column_similarity(str(header), term)
                if similarity > best_score:
                    best_score = similarity
                if similarity >= fuzzy_threshold and _same_token_shape(str(header), term):
                    gate_ok = True
            scored.append((spec.canonical, best_score, gate_ok))

        scored.sort(key=lambda item: item[1], reverse=True)
        candidates = tuple((c, s) for c, s, _ in scored[:3] if s > 0.0)

        accepted = sorted(
            [(c, s) for c, s, ok in scored if ok], key=lambda item: item[1], reverse=True
        )
        if accepted:
            best_canonical, best_score = accepted[0]
            tied = [c for c, s in accepted if abs(s - best_score) <= _TIE_EPSILON]
            if len(set(tied)) == 1:
                resolutions.append(
                    ColumnResolution(str(header), best_canonical, ResolutionMethod.FUZZY, best_score, candidates)
                )
                continue

        resolutions.append(
            ColumnResolution(str(header), None, ResolutionMethod.UNMAPPED, 0.0, candidates)
        )

    return ColumnMappingReport(
        resolutions=tuple(resolutions),
        specs=dict(specs),
        dataset=dataset,
        source_file=source_file,
    )


__all__ = [
    "ColumnMappingReport",
    "ColumnResolution",
    "ColumnSpec",
    "MissingRequiredColumnError",
    "ResolutionMethod",
    "UNMAPPED",
    "build_specs",
    "resolve_columns",
]
