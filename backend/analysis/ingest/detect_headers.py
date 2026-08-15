"""Header detection for Excel files."""

from typing import List, Tuple

import pandas as pd

from .excel_io import read_excel
from .normalize import normalize_text


def _score_header_candidate(columns, required_keywords: List[str]) -> float:
    """Score a detected header, preferring semantic names over Unnamed columns."""
    normalized_columns = [normalize_text(str(column)) for column in columns]
    normalized_keywords = [normalize_text(keyword) for keyword in (required_keywords or [])]

    score = 0.0
    for keyword in normalized_keywords:
        if keyword and any(keyword in column for column in normalized_columns):
            score += 2.0

    non_empty_columns = [column for column in normalized_columns if column]
    score += len(non_empty_columns) * 0.1
    score -= sum(1 for column in normalized_columns if column.startswith("unnamed")) * 0.75
    score -= sum(1 for column in normalized_columns if column.startswith("col_")) * 0.25
    return score


def detect_header_row(
    file_path: str,
    sheet_name=0,
    required_keywords: List[str] = None,
    max_rows_to_scan: int = 20,
) -> Tuple[int, pd.DataFrame]:
    """
    Detect header row in Excel file by scanning for keyword matches.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index
        required_keywords: Keywords that should appear in header
        max_rows_to_scan: Maximum rows to scan for header
        
    Returns:
        Tuple of (header_row_index, dataframe_with_correct_header)
    """
    # Read the sheet ONCE without a header, pick the header row from that raw
    # frame, then slice/rename it in memory -- no second full read of the file.
    df_raw = read_excel(file_path, sheet_name=sheet_name, header=None)

    if len(df_raw) == 0:
        return 0, df_raw

    best_row = _pick_header_row(df_raw, required_keywords or [], max_rows_to_scan)
    df = _slice_with_header(df_raw, best_row)
    return best_row, df


def _pick_header_row(df_raw: pd.DataFrame, required_keywords: List[str], max_rows_to_scan: int) -> int:
    """Return the index of the best header row within the raw (header=None) frame."""
    best_score = float("-inf")
    best_row = 0
    scan_limit = min(max_rows_to_scan, len(df_raw))

    for row_idx in range(scan_limit):
        row_values = [normalize_text(v) for v in df_raw.iloc[row_idx]]
        row_text = " ".join(row_values)

        score = 0.0
        for keyword in required_keywords:
            if normalize_text(keyword) in row_text:
                score += 1

        # Bonus for non-empty cells, but penalize placeholder header rows.
        score += sum(1 for v in row_values if v) * 0.1
        score -= sum(1 for v in row_values if v.startswith("unnamed")) * 0.75

        if score > best_score:
            best_score = score
            best_row = row_idx

    return best_row


def _slice_with_header(df_raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    """Slice the raw frame at ``header_row``: use that row as the header.

    ``infer_objects`` recovers per-column dtypes (numbers/dates that read as
    object under ``header=None``) so the result matches a ``header=header_row``
    read without touching the file again.
    """
    columns = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = columns
    df = df.reset_index(drop=True)
    return df.infer_objects()


def auto_detect_workbook_header(
    file_path: str,
    required_keywords: List[str] = None,
) -> Tuple[str, int, pd.DataFrame]:
    """
    Auto-detect sheet name and header in workbook.
    
    Tries to find sheet with most data and detects header row.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Tuple of (sheet_name, header_row_index, dataframe)
    """
    # Get all sheet names and release the file handle promptly on Windows.
    with pd.ExcelFile(file_path) as xls:
        sheet_names = list(xls.sheet_names)
    
    best_sheet = None
    best_header = 0
    best_df = None
    best_score = float("-inf")
    
    for sheet in sheet_names:
        # Skip sheets with obvious non-data names
        sheet_lower = sheet.lower()
        if any(skip in sheet_lower for skip in ["сводка", "примечание", "legend", "notes"]):
            continue
        
        try:
            header_row, df = detect_header_row(
                file_path,
                sheet_name=sheet,
                required_keywords=required_keywords,
            )

            # Prefer sheets that both contain data and expose meaningful headers.
            score = (len(df.columns) * min(len(df), 1000)) + _score_header_candidate(df.columns, required_keywords)
            
            if score > best_score:
                best_score = score
                best_sheet = sheet
                best_df = df
                best_header = header_row
        except Exception:
            continue
    
    if best_sheet is None:
        # Fallback to first sheet
        best_sheet = sheet_names[0]
        best_header, best_df = detect_header_row(
            file_path,
            sheet_name=best_sheet,
            required_keywords=required_keywords,
        )
    
    return best_sheet, best_header, best_df


def find_data_start_column(df: pd.DataFrame) -> int:
    """
    Find first non-empty column (useful for left-aligned sheets).
    
    Args:
        df: DataFrame
        
    Returns:
        Index of first non-empty column
    """
    for i in range(len(df.columns)):
        if df.iloc[:, i].notna().sum() > 0:
            return i
    return 0


def clean_header_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column header names.
    
    Args:
        df: DataFrame with headers
        
    Returns:
        DataFrame with cleaned headers
    """
    df.columns = [
        str(col).strip() if pd.notna(col) else f"col_{i}"
        for i, col in enumerate(df.columns)
    ]
    return df
