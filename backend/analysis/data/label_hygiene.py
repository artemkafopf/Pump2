"""Label hygiene helpers for the survival stack (Phase A T5).

Three concerns, all reusable and side-effect-free:

1. ``trim_failed_node`` — TRIM (and case-fold the "no failure" sentinels) so that
   ``'НКТ '`` with a trailing space stops being a separate failure category.
2. ``normalize_field`` / ``FIELD_ALIAS_MAP`` — collapse raw Cyrillic field names onto
   their coded twins.  **Anything unmapped is returned as ``None``** so the caller can
   route it to a ``needs_review`` list instead of guessing silently.
3. ``count_nonvt_sour`` — apply the 10 mg/l H₂S threshold to **all** fields and report
   how many non-Vt runs would flip to sour.  This is *reporting only*; the active
   stratum definition stays Vt-only until the user reviews the count (T5.3).

These are pure functions; wiring them into ``scripts/pipeline.py`` (ingest-time
normalisation) is a separate, reviewable step.
"""
from __future__ import annotations

import pandas as pd

# H₂S sour threshold (mg/l), consistent with chemistry_run_features.
H2S_SOUR_THRESHOLD = 10.0

# Confirmed field aliases → codes. Only mappings supported by the review doc
# (coded twin explicitly named) are included; everything else → needs_review.
FIELD_ALIAS_MAP: dict[str, str] = {
    "Ичёдинское нефтяное месторождение": "Ic",  # review: n=7 vs coded 'Ic'
    "АЗЛУ": "Az",                               # review: n=1 alongside 'Az'
}

# Two-letter (Latin) field codes already in canonical form.
_CANONICAL_CODE_MAXLEN = 4

# "No failure" sentinels that differ only by case/whitespace.
_NO_FAILURE_SENTINELS = {"нет"}


def trim_failed_node(value: object) -> str | None:
    """Normalise a ``failed_node`` label: strip surrounding whitespace and collapse
    the case/whitespace variants of the "no failure" sentinel to lowercase ``'нет'``.

    Real node labels keep their original casing (only whitespace is trimmed) so that
    ``'НКТ '`` → ``'НКТ'`` merges with ``'НКТ'`` without altering distinct categories.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.strip().lower() in _NO_FAILURE_SENTINELS:
        return "нет"
    return s


def normalize_field(name: object) -> str | None:
    """Return the canonical field code for ``name``.

    * Known aliases → their code (via ``FIELD_ALIAS_MAP``).
    * Already-canonical short Latin codes (e.g. ``Ya``, ``Vt``, ``Az``) → unchanged.
    * Anything else (unmapped Cyrillic full names) → ``None`` (route to needs_review).
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    if s in FIELD_ALIAS_MAP:
        return FIELD_ALIAS_MAP[s]
    # Canonical codes are short and ASCII (Latin letters + optional lowercase tail).
    if len(s) <= _CANONICAL_CODE_MAXLEN and s.isascii() and s[:1].isalpha():
        return s
    return None


def field_alias_report(fields: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (mapping_applied, needs_review) tables from a Series of raw field names.

    ``mapping_applied`` lists each distinct raw value, its resolved code (or blank),
    and run count.  ``needs_review`` is the subset that resolved to ``None``.
    """
    vc = fields.value_counts(dropna=False)
    rows = []
    for raw, n in vc.items():
        code = normalize_field(raw)
        rows.append({
            "field_alias": raw,
            "field_code": code if code is not None else "",
            "resolved": code is not None,
            "n_runs": int(n),
        })
    applied = pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)
    needs_review = applied[~applied["resolved"]].reset_index(drop=True)
    return applied, needs_review


def count_nonvt_sour(
    df: pd.DataFrame,
    field_col: str = "field",
    h2s_col: str = "h2s_proxy_mg_l",
    threshold: float = H2S_SOUR_THRESHOLD,
) -> pd.DataFrame:
    """Apply the sour threshold to **all** fields; report per-field sour counts.

    Reporting only — does NOT change stratum definitions.  Returns one row per field
    with ``n_runs``, ``n_sour_if_threshold_applied`` and ``would_flip`` (non-Vt sour).
    """
    h2s = pd.to_numeric(df[h2s_col], errors="coerce")
    is_sour = h2s > threshold
    rows = []
    for field, idx in df.groupby(field_col).groups.items():
        sub = is_sour.loc[idx]
        n_sour = int(sub.sum())
        rows.append({
            "field": field,
            "n_runs": int(len(idx)),
            "n_sour_if_threshold_applied": n_sour,
            "currently_stratified_sour": field == "Vt",
            "would_flip_to_sour": n_sour if field != "Vt" else 0,
        })
    out = pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)
    return out


__all__ = [
    "trim_failed_node",
    "normalize_field",
    "field_alias_report",
    "count_nonvt_sour",
    "FIELD_ALIAS_MAP",
    "H2S_SOUR_THRESHOLD",
]
