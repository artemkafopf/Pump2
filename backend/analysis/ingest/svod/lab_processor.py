"""Lab chemistry dataset processing.

A thin ``failure_update``-facing wrapper over the shared lab loader and SQLite
store in :mod:`analysis.ingest.lab`. When a lab SQLite store resolves from the
given source it is used for lookups; otherwise the Excel workbooks are read
directly. Both paths share the same loader and window/aggregation helpers, so
results are identical regardless of the source kind.
"""

import pandas as pd

from ..config import LAB_DATA_PROCESSING
from ..config import LAB_CHEMISTRY_COLUMNS, LAB_COLUMN_RULES
from ..normalize import normalize_well
from ..lab.processor import (
    WELL_COLUMN,
    aggregate_lab_column,
    collapse_duplicate_lab_columns,
    load_lab_export,
    resolve_lab_files,
    resolve_sample_date_column,
    select_lab_rows_for_window,
)
from ..lab.sqlite_store import (
    aggregate_lab_chemistry_sqlite,
    resolve_lab_sqlite_path,
)


def enrich_lab_chemistry(records: pd.DataFrame) -> pd.DataFrame:
    """Ensure lab chemistry columns exist and use '-' for missing values."""
    records = records.copy()
    for col in LAB_CHEMISTRY_COLUMNS:
        if col not in records.columns:
            records[col] = None
        records[col] = records[col].where(records[col].notna(), "-")
    return records


def _ensure_lab_columns(records: pd.DataFrame) -> pd.DataFrame:
    for col in LAB_CHEMISTRY_COLUMNS:
        if col not in records.columns:
            records[col] = None
    return records


def _record_well_value(record, well_col: str):
    well_value = record.get(well_col, record.get("Скв."))
    if pd.isna(well_value) or well_value is None:
        return None
    return well_value


def _record_failure_date(record, failure_date_col: str):
    return record.get(failure_date_col, record.get("Дата остановки"))


def _enrich_from_lab_sqlite(
    records: pd.DataFrame,
    sqlite_path,
    av_window: int,
    well_col: str,
    failure_date_col: str,
) -> pd.DataFrame:
    """Enrich records from a prebuilt lab SQLite store."""
    records = _ensure_lab_columns(records)
    print(f"  Using lab SQLite store {sqlite_path}")
    for idx, record in records.iterrows():
        well_value = _record_well_value(record, well_col)
        if well_value is None:
            continue
        values = aggregate_lab_chemistry_sqlite(
            sqlite_path,
            well_value,
            failure_date=_record_failure_date(record, failure_date_col),
            av_window=av_window,
            column_rules=LAB_COLUMN_RULES,
        )
        for target_col, value in values.items():
            records.loc[idx, target_col] = value
    return records


def _enrich_from_lab_files(
    records: pd.DataFrame,
    lab_path,
    av_window: int,
    well_col: str,
    failure_date_col: str,
) -> pd.DataFrame:
    """Enrich records by reading lab workbooks directly."""
    lab_frames = []
    for file_path in resolve_lab_files(lab_path):
        try:
            lab_df = load_lab_export(file_path)
            lab_frames.append(lab_df)
            print(f"  Loaded lab file {file_path.name} | rows={len(lab_df)}")
        except Exception as exc:
            print(f"  Skipped malformed lab file {file_path.name}: {exc}")
    if not lab_frames:
        return enrich_lab_chemistry(records)

    lab_df = pd.concat(lab_frames, ignore_index=True, sort=False)
    lab_df = collapse_duplicate_lab_columns(lab_df)
    if WELL_COLUMN not in lab_df.columns:
        return enrich_lab_chemistry(records)

    sample_date_header = resolve_sample_date_column(lab_df.columns)
    lab_df["normalized_well"] = lab_df[WELL_COLUMN].apply(normalize_well)
    records = _ensure_lab_columns(records)

    for idx, record in records.iterrows():
        well_value = _record_well_value(record, well_col)
        if well_value is None:
            continue
        matching_rows = lab_df[lab_df["normalized_well"] == normalize_well(well_value)]
        if matching_rows.empty:
            continue
        selected_rows = select_lab_rows_for_window(
            matching_rows,
            _record_failure_date(record, failure_date_col),
            av_window=av_window,
            date_col=sample_date_header if sample_date_header is not None else "",
        )
        for target_col, source_col in LAB_COLUMN_RULES.items():
            if source_col not in lab_df.columns:
                continue
            value = aggregate_lab_column(selected_rows, source_col)
            if value is not None:
                records.loc[idx, target_col] = value
    return records


def enrich_from_lab(
    records: pd.DataFrame,
    lab_path,
    av_window: int = None,
    well_col: str = "well",
    failure_date_col: str = "failure_date",
    prefer_sqlite: bool = True,
) -> pd.DataFrame:
    """Enrich records from a lab SQLite store or lab workbooks.

    When ``prefer_sqlite`` is set and a ``lab.sqlite`` store resolves from
    ``lab_path``, values are read from the store; otherwise the Excel workbooks
    are read directly. Both paths apply the same window/averaging rules.
    """
    records = records.copy()
    av_window = LAB_DATA_PROCESSING.get("av_window", 1) if av_window is None else av_window

    sqlite_path = resolve_lab_sqlite_path(lab_path) if prefer_sqlite else None
    if sqlite_path is not None:
        return _enrich_from_lab_sqlite(records, sqlite_path, av_window, well_col, failure_date_col)
    return _enrich_from_lab_files(records, lab_path, av_window, well_col, failure_date_col)


__all__ = [
    "collapse_duplicate_lab_columns",
    "select_lab_rows_for_window",
    "aggregate_lab_column",
    "enrich_lab_chemistry",
    "enrich_from_lab",
]
