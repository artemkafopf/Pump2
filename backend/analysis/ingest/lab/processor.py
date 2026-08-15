"""Lab chemistry export loading and merging helpers.

Lab workbooks carry a three-row header (``header=[1, 2, 3]``) which is flattened
to its last meaningful fragment and de-duplicated. The loader here is the single
source of truth shared by :mod:`analysis.ingest.lab.sqlite_store` and the
``failure_update`` lab enrichment path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from analysis.paths import resolve_source_dir

from ..column_resolution import build_specs, resolve_columns
from ..excel_io import read_excel
from ..normalize import parse_date


DEFAULT_SOURCE_DIR = resolve_source_dir("lab")
WELL_COLUMN = "Скважина"
SAMPLE_DATE_COLUMN = "Дата время отбора"
SAMPLE_DATE_FALLBACK_COLUMN = "Дата анализа"
LAB_HEADER_ROWS = [1, 2, 3]
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}

# Anchor columns the lab loader keys on. Each stays a distinct canonical (no
# cross-mapping between the two date columns) so a lightly renamed header -- an
# extra space/dot -- still resolves back to the exact name the store expects.
LAB_ANCHOR_SPECS = build_specs(
    {
        WELL_COLUMN: ["скважина", "скв", "скв.", "№ скважины", "well"],
        SAMPLE_DATE_COLUMN: ["дата время отбора", "дата отбора", "дата и время отбора"],
        SAMPLE_DATE_FALLBACK_COLUMN: ["дата анализа", "дата проведения анализа"],
    }
)


def flatten_lab_column_label(column_label) -> Optional[str]:
    """Collapse a lab workbook column tuple to the last meaningful label."""
    if isinstance(column_label, tuple):
        parts = []
        for part in column_label:
            if pd.isna(part):
                continue
            part_str = str(part).strip()
            if not part_str or part_str.lower().startswith("unnamed:"):
                continue
            parts.append(part_str)
        return parts[-1] if parts else None
    if pd.isna(column_label):
        return None
    text = str(column_label).strip()
    return text or None


def collapse_duplicate_lab_columns(lab_df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate flattened lab columns into a single canonical column."""
    if lab_df.empty or lab_df.columns.is_unique:
        return lab_df

    collapsed = pd.DataFrame(index=lab_df.index)
    seen: List[str] = []
    for column_name in lab_df.columns:
        if column_name in seen:
            continue
        seen.append(column_name)
        selected = lab_df.loc[:, lab_df.columns == column_name]
        if isinstance(selected, pd.Series) or selected.shape[1] == 1:
            collapsed[column_name] = selected if isinstance(selected, pd.Series) else selected.iloc[:, 0]
            continue
        # Prefer the first non-null value across duplicate columns.
        collapsed[column_name] = selected.bfill(axis=1).iloc[:, 0]
    return collapsed


def detect_lab_header_top(file_path: Path, max_scan_rows: int = 8) -> int:
    """Return the 0-based row index where the lab header block starts.

    Lab workbooks carry a three-row header but its top row varies per file
    (some start at row 0, others at row 1). We locate it by finding the first
    row that contains the well column (``Скважина``); the two rows beneath it
    complete the header block. Falls back to the first row of ``LAB_HEADER_ROWS``
    when the marker is not found.
    """
    from ..normalize import normalize_text

    raw = read_excel(str(file_path), sheet_name=0, header=None, nrows=max_scan_rows)
    well_marker = normalize_text(WELL_COLUMN)
    for row_index in range(len(raw)):
        row_values = {normalize_text(value) for value in raw.iloc[row_index].tolist()}
        if well_marker in row_values:
            return row_index
    return LAB_HEADER_ROWS[0]


def load_lab_export(file_path: Path, header=None) -> pd.DataFrame:
    """Load one lab workbook into a flat, de-duplicated dataframe.

    Detects the three-row header block (its top row varies per file), flattens
    each column to its last meaningful fragment, drops columns with no usable
    label, and collapses duplicate columns. Adds ``_source_file`` /
    ``_source_path`` provenance columns. Pass ``header`` explicitly to override
    detection.
    """
    file_path = Path(file_path)
    if header is None:
        top = detect_lab_header_top(file_path)
        header = [top, top + 1, top + 2]
    lab_df = read_excel(str(file_path), sheet_name=0, header=header)
    flattened_columns = [flatten_lab_column_label(col) for col in lab_df.columns]
    lab_df.columns = flattened_columns
    lab_df = lab_df.loc[:, [col is not None for col in lab_df.columns]].copy()
    lab_df = collapse_duplicate_lab_columns(lab_df)
    # Normalize the well / sample-date anchor headers to their canonical names so
    # a renamed source header still feeds the exact-string lookups downstream.
    anchor_report = resolve_columns(
        [str(column) for column in lab_df.columns],
        LAB_ANCHOR_SPECS,
        dataset="lab",
        source_file=file_path.name,
    )
    lab_df = anchor_report.apply(lab_df)
    lab_df = lab_df.dropna(how="all").reset_index(drop=True)
    lab_df["_source_file"] = file_path.name
    lab_df["_source_path"] = str(file_path)
    return lab_df


def resolve_lab_files(source: Path | str | Sequence[Path | str]) -> List[Path]:
    """Resolve one or more files/directories into concrete lab files."""
    if isinstance(source, (str, Path)):
        items: Iterable[Path | str] = [source]
    else:
        items = source

    discovered: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in EXCEL_SUFFIXES and not child.name.startswith("~$"):
                    discovered.append(child.resolve())
        elif path.is_file() and path.suffix.lower() in EXCEL_SUFFIXES and not path.name.startswith("~$"):
            discovered.append(path.resolve())

    unique_files = sorted(set(discovered), key=lambda value: (value.name.lower(), str(value).lower()))
    return [Path(path) for path in unique_files]


def merge_lab_exports(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read and concatenate all discovered lab export files."""
    files = resolve_lab_files(source)
    if not files:
        return pd.DataFrame()

    if verbose:
        print(f"Loading {len(files)} lab file(s)")

    frames = []
    for index, file_path in enumerate(files, start=1):
        try:
            frames.append(load_lab_export(file_path))
            if verbose:
                print(f"  [{index}/{len(files)}] {file_path.name}")
        except Exception as exc:  # pragma: no cover - defensive, mirrors legacy loader
            print(f"  Skipped malformed lab file {file_path.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def resolve_sample_date_column(columns: Iterable[object]) -> Optional[str]:
    """Return the sample-date column name present in ``columns``, if any."""
    column_set = {str(column) for column in columns}
    if SAMPLE_DATE_COLUMN in column_set:
        return SAMPLE_DATE_COLUMN
    if SAMPLE_DATE_FALLBACK_COLUMN in column_set:
        return SAMPLE_DATE_FALLBACK_COLUMN
    return None


def select_lab_rows_for_window(
    matching_rows: pd.DataFrame,
    failure_date,
    av_window: int,
    date_col: str = SAMPLE_DATE_COLUMN,
) -> pd.DataFrame:
    """Select lab rows to aggregate for a failure record.

    ``av_window`` semantics (matching the legacy ``lab_processor`` behaviour):
    ``-1`` uses every prior sample, ``1`` uses only the latest prior sample, and
    ``N`` uses samples within ``N`` days before the failure date.
    """
    if matching_rows.empty:
        return matching_rows
    lab_rows = matching_rows.copy()
    if date_col not in lab_rows.columns:
        return lab_rows.tail(1)
    lab_rows["_lab_date"] = lab_rows[date_col].apply(parse_date)
    lab_rows = lab_rows[lab_rows["_lab_date"].notna()].copy()
    if lab_rows.empty:
        return matching_rows.tail(1)
    lab_rows = lab_rows.sort_values("_lab_date")
    fail_ts = parse_date(failure_date)
    if av_window == -1:
        return lab_rows
    if fail_ts is None:
        return lab_rows.tail(1)
    prior_rows = lab_rows[lab_rows["_lab_date"] <= fail_ts].copy()
    fallback_rows = prior_rows if not prior_rows.empty else lab_rows
    if av_window == 1:
        return fallback_rows.tail(1)
    window_start = fail_ts - pd.Timedelta(days=av_window)
    window_rows = prior_rows[prior_rows["_lab_date"] >= window_start].copy()
    if not window_rows.empty:
        return window_rows
    return fallback_rows.tail(1)


def aggregate_lab_column(selected_rows: pd.DataFrame, source_col: str):
    """Average numeric lab values in ``source_col``, ignoring NaNs."""
    if source_col not in selected_rows.columns or selected_rows.empty:
        return None
    numeric_values = pd.to_numeric(selected_rows[source_col], errors="coerce")
    if numeric_values.notna().any():
        return float(numeric_values.mean())
    return None
