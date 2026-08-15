"""Duplicate detection and matching logic.

Matching is index-based: the target register is indexed **once** by its
suffix-preserving well key (:func:`well_identity.normalize_well_key`) into a
dict of ``well_key -> [(failure_date, install_date, nno, row_index)]``. Every
candidate is then reconciled by a cheap lookup on that dict instead of an
``O(new x target)`` per-candidate ``iterrows`` scan of the whole target.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from ..normalize import normalize_well, parse_date, parse_number, normalize_text
from .well_identity import normalize_well_key


class DuplicateStatus(Enum):
    """Status of duplicate detection"""
    EXACT_EXISTING = "exact_existing"
    LIKELY_EXISTING = "likely_existing"
    NEW = "new"


class MatchResult:
    """Result of matching operation"""
    def __init__(
        self,
        status: DuplicateStatus,
        confidence: float = 0.0,
        matching_row_idx: Optional[int] = None,
        match_reason: Optional[str] = None
    ):
        self.status = status
        self.confidence = confidence
        self.matching_row_idx = matching_row_idx
        self.match_reason = match_reason


def is_same_well(well1: str, well2: str) -> bool:
    """
    Check if two well identifiers refer to the same well.
    
    Args:
        well1: First well ID
        well2: Second well ID
        
    Returns:
        True if wells match
    """
    w1 = normalize_well(well1)
    w2 = normalize_well(well2)
    return w1 == w2 and w1 != ""


def is_same_date(date1, date2, tolerance_days: int = 0) -> bool:
    """
    Check if two dates are the same or within tolerance.
    
    Args:
        date1: First date
        date2: Second date
        tolerance_days: Tolerance in days
        
    Returns:
        True if dates match
    """
    d1 = parse_date(date1)
    d2 = parse_date(date2)
    
    if d1 is None or d2 is None:
        return False
    
    diff = abs((d1 - d2).days)
    return diff <= tolerance_days


# A single indexed target run: parsed dates + runtime, and its target row index.
TargetRun = Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[float], int]


def build_target_index(
    target_df: pd.DataFrame,
    well_col: str,
    failure_date_col: Optional[str] = None,
    install_date_col: Optional[str] = None,
    nno_col: Optional[str] = None,
) -> Dict[str, List[TargetRun]]:
    """Index the target register once by suffix-preserving well key.

    Returns ``{well_key: [(failure_date, install_date, nno, row_index), ...]}``.
    The key is :func:`normalize_well_key` (bore-suffix preserving) so distinct
    bores of one physical well never merge. All reconciliation reads this dict;
    nothing scans ``target_df`` per candidate.
    """
    index: Dict[str, List[TargetRun]] = {}
    for row_index, row in target_df.iterrows():
        well_key = normalize_well_key(row.get(well_col))
        if not well_key:
            continue
        failure_date = parse_date(row.get(failure_date_col)) if failure_date_col else None
        install_date = parse_date(row.get(install_date_col)) if install_date_col else None
        nno = parse_number(row.get(nno_col)) if nno_col else None
        index.setdefault(well_key, []).append((failure_date, install_date, nno, row_index))
    return index


def match_candidate(
    well_key: str,
    failure_date: Optional[pd.Timestamp],
    install_date: Optional[pd.Timestamp],
    index: Dict[str, List[TargetRun]],
    *,
    nno: Optional[float] = None,
    tolerance_days: int = 2,
    nno_tolerance: float = 5.0,
) -> MatchResult:
    """Reconcile one candidate against the target index. Pure, O(runs-for-well).

    Tiers, in order:
      1. exact — same well + same failure date, or (running well, no failure
         date) same install date -> ``EXACT_EXISTING``.
      2. tolerance — same well + failure date within ``tolerance_days``, or
         install date within ``tolerance_days``, or same install date + NNO
         within ``nno_tolerance`` -> ``LIKELY_EXISTING`` (needs review).
      3. no match -> ``NEW``.
    """
    runs = index.get(well_key) if well_key else None
    if not runs:
        return MatchResult(DuplicateStatus.NEW)

    # Tier 1: exact failure-date match.
    if failure_date is not None:
        for target_failure, _target_install, _nno, row_index in runs:
            if target_failure is not None and (failure_date - target_failure).days == 0:
                return MatchResult(
                    DuplicateStatus.EXACT_EXISTING,
                    confidence=1.00,
                    matching_row_idx=row_index,
                    match_reason="exact_well_exact_date",
                )

    # Tier 1 (running wells): no failure date -> exact install-date match.
    if failure_date is None and install_date is not None:
        for _target_failure, target_install, _nno, row_index in runs:
            if target_install is not None and (install_date - target_install).days == 0:
                return MatchResult(
                    DuplicateStatus.EXACT_EXISTING,
                    confidence=0.95,
                    matching_row_idx=row_index,
                    match_reason="exact_well_install_date",
                )

    # Tier 2: failure date within tolerance.
    if failure_date is not None:
        for target_failure, _target_install, _nno, row_index in runs:
            if target_failure is not None and abs((failure_date - target_failure).days) <= tolerance_days:
                return MatchResult(
                    DuplicateStatus.LIKELY_EXISTING,
                    confidence=0.90,
                    matching_row_idx=row_index,
                    match_reason="well_failure_date_within_tolerance",
                )

    # Tier 2: install date within tolerance, or same install date + NNO within tolerance.
    if install_date is not None:
        for _target_failure, target_install, target_nno, row_index in runs:
            if target_install is None:
                continue
            install_diff = abs((install_date - target_install).days)
            if install_diff <= tolerance_days:
                return MatchResult(
                    DuplicateStatus.LIKELY_EXISTING,
                    confidence=0.85,
                    matching_row_idx=row_index,
                    match_reason="well_install_date_within_tolerance",
                )
            if install_diff == 0 and nno is not None and target_nno is not None and abs(nno - target_nno) <= nno_tolerance:
                return MatchResult(
                    DuplicateStatus.LIKELY_EXISTING,
                    confidence=0.80,
                    matching_row_idx=row_index,
                    match_reason="well_install_date_nno_within_tolerance",
                )

    return MatchResult(DuplicateStatus.NEW)


def detect_duplicates(
    new_records: pd.DataFrame,
    target_df: pd.DataFrame,
    well_col: str,
    failure_date_col: str,
    install_date_col: Optional[str] = None,
    nno_col: Optional[str] = None,
    new_well_col: str = "well",
    new_failure_date_col: str = "failure_date",
    new_install_date_col: str = "installation_date",
    new_nno_col: str = "runtime_nno",
    tolerance_days: int = 2,
) -> Dict[int, MatchResult]:
    """Reconcile every new record against the target via a single shared index.

    The index is built once; each candidate is matched by a dict lookup. A NEW
    candidate is added back into the index so a later identical row in the same
    batch is caught as a duplicate (its ``matching_row_idx`` is ``None`` — an
    in-batch echo, not a target row).
    """
    index = build_target_index(target_df, well_col, failure_date_col, install_date_col, nno_col)

    results: Dict[int, MatchResult] = {}
    for idx, row in new_records.iterrows():
        well_value = row.get(new_well_col, row.get("normalized_well"))
        well_key = normalize_well_key(well_value)
        failure_date = parse_date(row.get(new_failure_date_col))
        install_date = parse_date(row.get(new_install_date_col, row.get("Дата монтажа")))
        nno = parse_number(row.get(new_nno_col, row.get("Наработка (сут)")))

        result = match_candidate(
            well_key,
            failure_date,
            install_date,
            index,
            nno=nno,
            tolerance_days=tolerance_days,
        )
        results[idx] = result

        if result.status == DuplicateStatus.NEW and well_key:
            # Register the candidate so a later identical row in this batch is
            # recognised as an in-batch duplicate (no target row to point at).
            index.setdefault(well_key, []).append((failure_date, install_date, nno, None))

    return results
