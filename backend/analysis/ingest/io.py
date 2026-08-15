"""Input/Output operations for Excel files"""

import datetime as _dt

import pandas as pd
import openpyxl
from copy import copy
from openpyxl.styles import Alignment, PatternFill
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Sequence, Union

from .config import COLUMN_SYNONYMS, HEADER_DETECTION_KEYWORDS, SOURCE_FILE_PATTERNS
from .detect_headers import auto_detect_workbook_header, clean_header_names, detect_header_row
from .excel_io import read_excel


SourceInput = Union[str, Path, Sequence[Union[str, Path]]]


def normalize_source_input(source: SourceInput) -> List[Path]:
    """Normalize a file, directory, or file list into concrete paths."""
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return [Path(source)]
    return [Path(item) for item in source if item is not None]


def discover_source_files(source: SourceInput, dataset_type: str) -> List[Path]:
    """Discover dataset files from a path, folder, or explicit list.

    For ordinary datasets, directory discovery is intentionally non-recursive:
    only files directly inside the provided parent directory are considered.
    Dataset-specific logic such as techregime month-folder traversal is handled
    outside this helper.
    """
    discovered: List[Path] = []
    patterns = SOURCE_FILE_PATTERNS.get(dataset_type, ("*.xlsx", "*.xlsm", "*.xls"))
    for item in normalize_source_input(source):
        if item.is_dir():
            for pattern in patterns:
                discovered.extend(
                    path for path in item.glob(pattern)
                    if not path.name.startswith("~$")
                )
        elif item.exists() and item.is_file():
            if not item.name.startswith("~$"):
                discovered.append(item)
    unique_files = sorted({path.resolve() for path in discovered}, key=lambda path: (path.name.lower(), str(path).lower()))
    return [Path(path) for path in unique_files]


def load_dataset_frames(
    source: SourceInput,
    dataset_type: str,
    sheet_name: Optional[str] = None,
    required_keywords: Optional[List[str]] = None,
) -> List[Tuple[Path, pd.DataFrame, str, int]]:
    """Load all readable files for a dataset and return per-file frames."""
    files = discover_source_files(source, dataset_type)
    if not files:
        print(f"  No {dataset_type} files found from source: {source}")
        return []

    print(f"  Reading {len(files)} {dataset_type} file(s)")
    frames: List[Tuple[Path, pd.DataFrame, str, int]] = []
    keywords = required_keywords or HEADER_DETECTION_KEYWORDS.get(dataset_type)

    for file_path in files:
        try:
            if sheet_name is None:
                actual_sheet, header_row, df = auto_detect_workbook_header(
                    str(file_path),
                    required_keywords=keywords,
                )
            else:
                header_row, df = detect_header_row(
                    str(file_path),
                    sheet_name=sheet_name,
                    required_keywords=keywords,
                )
                actual_sheet = sheet_name
            df = clean_header_names(df)
            frames.append((file_path, df, actual_sheet, header_row))
            print(f"    Loaded {file_path.name} | sheet={actual_sheet} | header_row={header_row} | rows={len(df)}")
        except Exception as exc:
            print(f"    Skipped malformed {dataset_type} file {file_path.name}: {exc}")

    return frames


def load_excel_file(
    file_path: str,
    sheet_name: Optional[str] = None,
    header: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str], int]:
    """
    Load Excel file with automatic header detection if needed.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name (auto-detect if None)
        header: Header row index (auto-detect if None)
        
    Returns:
        Tuple of (dataframe, actual_sheet_name, header_row_index)
    """
    file_path = str(file_path)
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get sheet names, releasing the file handle promptly on Windows so the
    # workbook can later be deleted (e.g. a temporary seed schema workbook).
    with pd.ExcelFile(file_path) as xls:
        sheet_names = list(xls.sheet_names)

    if sheet_name is None:
        # Use first sheet with content
        sheet_name = sheet_names[0]

    if header is None:
        # Auto-detect header
        header, df = detect_header_row(file_path, sheet_name=sheet_name)
    else:
        # Load with specified header (read-only -> fast engine when available).
        df = read_excel(file_path, sheet_name=sheet_name, header=header)
    
    # Clean headers
    df = clean_header_names(df)
    
    return df, sheet_name, header


def load_workbook_sheet(
    file_path: str,
    sheet_name: str = None
) -> Tuple[pd.DataFrame, str, openpyxl.Workbook, openpyxl.worksheet.worksheet.Worksheet]:
    """
    Load Excel sheet with openpyxl workbook for writing.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name (uses first if None)
        
    Returns:
        Tuple of (dataframe, sheet_name, workbook, worksheet)
    """
    file_path = str(file_path)
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load with pandas first for data
    df, actual_sheet, _ = load_excel_file(file_path, sheet_name=sheet_name)
    
    # Load with openpyxl for formatting/writing
    wb = openpyxl.load_workbook(file_path)
    ws = wb[actual_sheet]
    
    return df, actual_sheet, wb, ws


def get_last_row_with_data(ws: openpyxl.worksheet.worksheet.Worksheet) -> int:
    """
    Get index of last non-empty row in worksheet.
    
    Args:
        ws: openpyxl Worksheet
        
    Returns:
        Row index (1-based)
    """
    return ws.max_row


def get_column_index(ws: openpyxl.worksheet.worksheet.Worksheet, column_name: str) -> Optional[int]:
    """
    Find column index by name in first row.
    
    Args:
        ws: openpyxl Worksheet
        column_name: Column header name
        
    Returns:
        Column index (1-based) or None
    """
    for col_idx, cell in enumerate(ws[1], 1):
        if cell.value and str(cell.value).strip().lower() == column_name.lower():
            return col_idx
    return None


def copy_cell_style(source_cell, target_cell):
    """
    Copy formatting from source cell to target cell.
    
    Args:
        source_cell: Source cell
        target_cell: Target cell
    """
    if source_cell.has_style:
        target_cell.font = copy(source_cell.font)
        target_cell.border = copy(source_cell.border)
        target_cell.fill = copy(source_cell.fill)
        target_cell.number_format = source_cell.number_format
        target_cell.protection = copy(source_cell.protection)
        target_cell.alignment = copy(source_cell.alignment)


def highlight_runtime_cell_if_needed(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    row_idx: int,
    runtime_column_name: str,
    runtime_value,
    header_row: int = 1,
    threshold_days: float = 90,
):
    """
    Highlight the runtime cell in red when runtime is below the threshold.

    Args:
        ws: openpyxl Worksheet
        row_idx: Target row index (1-based)
        runtime_column_name: Header text of the runtime column
        runtime_value: Runtime value to evaluate
        header_row: Row number containing headers
        threshold_days: Highlight values strictly below this threshold
    """
    try:
        runtime_number = float(runtime_value)
    except (TypeError, ValueError):
        return

    if runtime_number >= threshold_days:
        return

    runtime_col_idx = get_column_index(ws, runtime_column_name)
    if runtime_col_idx is None:
        return

    ws.cell(row=row_idx, column=runtime_col_idx).fill = PatternFill(
        fill_type="solid",
        start_color="FFFF0000",
        end_color="FFFF0000",
    )


def append_row_to_worksheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    row_data: Dict[str, any],
    header_row: int = 1,
    copy_format_from_row: Optional[int] = None
) -> int:
    """
    Append a row to worksheet with data and optional formatting.
    
    Args:
        ws: openpyxl Worksheet
        row_data: Dictionary of {column_name: value}
        header_row: Row number containing headers (1-based)
        copy_format_from_row: Row to copy formatting from (1-based)
        
    Returns:
        Row index where data was inserted (1-based)
    """
    # Determine target row
    last_row = ws.max_row
    target_row = last_row + 1
    
    # Write data
    for col_idx, cell in enumerate(ws[header_row], 1):
        col_name = str(cell.value).strip() if cell.value else None
        
        if col_name and col_name in row_data:
            value = row_data[col_name]
            target_cell = ws.cell(row=target_row, column=col_idx)
            target_cell.value = value
            
            # Copy format if specified
            if copy_format_from_row:
                source_cell = ws.cell(row=copy_format_from_row, column=col_idx)
                copy_cell_style(source_cell, target_cell)
    
    return target_row


def append_column_to_worksheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    column_name: str,
    *,
    header_row: int = 1,
    copy_format_from_column: Optional[int] = None,
) -> int:
    """
    Append a new column to a worksheet and optionally copy styling from another column.

    Args:
        ws: openpyxl Worksheet
        column_name: Header text for the new column
        header_row: Row number containing headers (1-based)
        copy_format_from_column: Source column index to copy styles from

    Returns:
        Column index where the header was inserted (1-based)
    """
    target_col = ws.max_column + 1
    header_cell = ws.cell(row=header_row, column=target_col)
    header_cell.value = column_name

    if copy_format_from_column:
        for row_idx in range(1, ws.max_row + 1):
            source_cell = ws.cell(row=row_idx, column=copy_format_from_column)
            target_cell = ws.cell(row=row_idx, column=target_col)
            copy_cell_style(source_cell, target_cell)
        ws.cell(row=header_row, column=target_col).value = column_name

    return target_col


DATE_NUMBER_FORMAT = "DD.MM.YYYY"


def _is_date_header(value) -> bool:
    """True when a header names a date column (so its cells show date only)."""
    return "дата" in str(value).strip().lower() if value is not None else False


def _displayed_length(value, *, is_date_column: bool) -> int:
    """Length of a cell's *displayed* text, used to size columns."""
    if value is None:
        return 0
    if is_date_column and isinstance(value, (_dt.datetime, _dt.date)):
        return len("00.00.0000")
    # Longest line drives width for multi-line header text.
    return max((len(line) for line in str(value).splitlines()), default=0)


def apply_autofilter_and_autofit(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    *,
    header_row: int = 1,
    freeze_header: bool = True,
    right_align_data: bool = True,
    min_width: float = 9.0,
    max_width: float = 60.0,
) -> None:
    """Format a register worksheet for review: filter, autofit, dates, alignment.

    - Enables an AutoFilter over the used range so the header row gets filter
      dropdowns, and freezes the header row.
    - Renders date columns (header contains "дата") as **date only** -- the value
      is coerced to a ``date`` and given a ``DD.MM.YYYY`` number format, dropping
      the ``00:00:00`` time component.
    - Right-aligns every data cell (the header row keeps its default alignment).
    - Approximates Excel "autofit": each column is sized to its widest *displayed*
      cell (dates count as ``DD.MM.YYYY``), plus room for the filter dropdown,
      clamped to ``[min_width, max_width]``.
    """
    if ws.max_row < header_row or ws.max_column < 1:
        return

    ws.auto_filter.ref = ws.dimensions
    if freeze_header:
        ws.freeze_panes = f"A{header_row + 1}"

    date_columns = {
        cell.column_letter: _is_date_header(cell.value)
        for cell in ws[header_row]
    }
    right_alignment = Alignment(horizontal="right")

    widest: dict[str, float] = {}
    for row in ws.iter_rows():
        for cell in row:
            column_letter = cell.column_letter
            is_date_column = date_columns.get(column_letter, False)

            if cell.row != header_row:
                if is_date_column and isinstance(cell.value, _dt.datetime):
                    cell.value = cell.value.date()
                if is_date_column and isinstance(cell.value, _dt.date):
                    cell.number_format = DATE_NUMBER_FORMAT
                if right_align_data:
                    cell.alignment = right_alignment

            length = _displayed_length(cell.value, is_date_column=is_date_column)
            if length > widest.get(column_letter, 0):
                widest[column_letter] = length

    for column_letter, length in widest.items():
        # +3 leaves room for the AutoFilter dropdown arrow on the header cell.
        width = max(min_width, min(max_width, length + 3))
        ws.column_dimensions[column_letter].width = width


def save_workbook(wb: openpyxl.Workbook, output_path: str):
    """
    Save openpyxl workbook.

    Args:
        wb: Workbook
        output_path: Output file path
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def create_audit_sheet(wb: openpyxl.Workbook, sheet_name: str) -> openpyxl.worksheet.worksheet.Worksheet:
    """
    Create new sheet in workbook.
    
    Args:
        wb: Workbook
        sheet_name: New sheet name
        
    Returns:
        New worksheet
    """
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        # Clear existing data
        for row in ws.iter_rows():
            for cell in row:
                cell.value = None
    else:
        ws = wb.create_sheet(sheet_name)
    
    return ws


def write_dataframe_to_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    df: pd.DataFrame,
    start_row: int = 1,
    start_col: int = 1,
    include_header: bool = True
):
    """
    Write DataFrame to openpyxl worksheet.
    
    Args:
        ws: Worksheet
        df: DataFrame
        start_row: Starting row (1-based)
        start_col: Starting column (1-based)
        include_header: Whether to write column headers
    """
    # Write headers
    if include_header:
        for col_idx, col_name in enumerate(df.columns, start_col):
            ws.cell(row=start_row, column=col_idx, value=col_name)
        start_row += 1
    
    # Write data
    for row_idx, (_, row) in enumerate(df.iterrows(), start_row):
        for col_idx, value in enumerate(row, start_col):
            ws.cell(row=row_idx, column=col_idx, value=value)
