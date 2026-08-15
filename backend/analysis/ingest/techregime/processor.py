"""Merge TechRegime exports, write partitioned outputs, and report anomalies."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

import pandas as pd

from analysis.paths import resolve_source_dir

from ..excel_io import read_excel
from ._normalize import normalize_text, parse_date


DEFAULT_SOURCE_DIR = resolve_source_dir("techregime")
DEFAULT_OUTPUT_DIRNAME = "split"
DATE_COLUMN = "Дата"
FIELD_COLUMN = "М/р"
WELL_ID_COLUMN = "ID скважины"
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
QUALITY_REPORT_FILENAME = "techregime_data_quality_report.txt"
PartitionMode = Literal["year", "field", "field_year", "well_field"]


@dataclass(frozen=True)
class PartitionOutput:
    """Metadata about one written partition workbook."""

    mode: PartitionMode
    partition_key: str
    rows: int
    output_path: Path


@dataclass(frozen=True)
class PartitionResult:
    """Summary of the merge-and-partition operation."""

    source_files: tuple[Path, ...]
    total_rows: int
    invalid_date_rows: int
    invalid_field_rows: int
    invalid_well_field_rows: int
    quality_report_path: Path | None
    flagged_date_count: int
    missing_dates: tuple[str, ...]
    outputs: tuple[PartitionOutput, ...]


@dataclass(frozen=True)
class DateQualityStat:
    """Completeness metrics for one report date."""

    date_key: str
    entry_count: int
    filled_row_count: int
    filled_row_ratio: float
    avg_filled_values_per_entry: float
    flagged_reasons: tuple[str, ...]


@dataclass(frozen=True)
class YearlyOutput:
    """Backward-compatible metadata about one yearly workbook."""

    year: int
    rows: int
    output_path: Path


@dataclass(frozen=True)
class MergeByYearResult:
    """Backward-compatible summary of the year-only split operation."""

    source_files: tuple[Path, ...]
    total_rows: int
    invalid_date_rows: int
    yearly_outputs: tuple[YearlyOutput, ...]


def _header_fragment(value) -> str:
    """Return a clean header fragment or an empty string."""
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed:"):
        return ""
    return text


def flatten_export_column_label(column_label) -> str:
    """Flatten a one- or two-row TechRegime export column label."""
    if isinstance(column_label, tuple):
        parts = [_header_fragment(part) for part in column_label]
        parts = [part for part in parts if part]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if parts[0] == parts[-1]:
            return parts[0]
        return " | ".join(parts)
    return _header_fragment(column_label)


def _make_unique_columns(columns: Sequence[str]) -> List[str]:
    """Preserve readable column names while disambiguating duplicates."""
    counts: dict[str, int] = {}
    unique_columns: List[str] = []
    for index, column in enumerate(columns, start=1):
        base_name = column or f"column_{index}"
        counts[base_name] = counts.get(base_name, 0) + 1
        if counts[base_name] == 1:
            unique_columns.append(base_name)
        else:
            unique_columns.append(f"{base_name}__{counts[base_name]}")
    return unique_columns


def _flatten_columns(columns: Sequence[object]) -> List[str]:
    return _make_unique_columns([flatten_export_column_label(column) for column in columns])


def _slice_raw_with_headers(raw: pd.DataFrame, header_rows: List[int]) -> pd.DataFrame:
    """Build a frame from a header-less raw frame using ``header_rows`` as headers.

    For a two-row header the columns become tuples (as a MultiIndex read would),
    which ``flatten_export_column_label`` already handles; empty/merged cells
    read as NaN, which ``_header_fragment`` discards exactly like a pandas
    ``Unnamed:`` placeholder.
    """
    top = max(header_rows)
    if top >= len(raw):
        return raw.iloc[0:0]
    if len(header_rows) == 1:
        columns = list(raw.iloc[header_rows[0]])
    else:
        columns = list(zip(*(list(raw.iloc[r]) for r in header_rows)))
    data = raw.iloc[top + 1:].copy()
    data.columns = columns
    return data.reset_index(drop=True).infer_objects()


def load_techregime_export(file_path: Path, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Load one current-format TechRegime export into a flat dataframe.

    Reads the sheet ONCE (header=None) and tries a two-row then a one-row header
    against that in-memory frame, instead of re-reading the whole file per attempt.
    """
    file_path = Path(file_path)
    raw = read_excel(file_path, sheet_name=0, header=None)

    for header_rows in ([0, 1], [0]):
        df = _slice_raw_with_headers(raw, header_rows)
        df.columns = _flatten_columns(df.columns)
        df = df.dropna(how="all").reset_index(drop=True)
        if date_column in df.columns:
            df["_source_file"] = file_path.name
            df["_source_path"] = str(file_path)
            return df

    raise ValueError(f"Expected '{date_column}' column in {file_path}")


def resolve_techregime_files(source: Path | str | Sequence[Path | str]) -> List[Path]:
    """Resolve one or more files/directories into concrete TechRegime files."""
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


def merge_techregime_exports(
    source: Path | str | Sequence[Path | str],
    date_column: str = DATE_COLUMN,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read and concatenate all discovered TechRegime export files."""
    files = resolve_techregime_files(source)
    if not files:
        return pd.DataFrame()
    _verbose_print(f"Loading {len(files)} TechRegime file(s)", verbose=verbose)

    frames = []
    for index, file_path in enumerate(files, start=1):
        frames.append(load_techregime_export(file_path, date_column=date_column))
        _render_progress("Loading files", index, len(files), verbose=verbose)
    return pd.concat(frames, ignore_index=True, sort=False)


def _default_output_dir(source_files: Sequence[Path]) -> Path:
    if not source_files:
        return DEFAULT_SOURCE_DIR / DEFAULT_OUTPUT_DIRNAME
    common_parent = source_files[0].parent
    return common_parent / DEFAULT_OUTPUT_DIRNAME


def _sanitize_filename_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "unknown"


def _render_progress(label: str, current: int, total: int, *, verbose: bool) -> None:
    """Render a lightweight progress bar for long-running operations."""
    if not verbose:
        return

    safe_total = max(total, 1)
    width = 24
    filled = int(width * current / safe_total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total}", end=end, flush=True, file=sys.stdout)


def _verbose_print(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stdout)


def _is_filled_value(value) -> bool:
    if pd.isna(value) or value is None:
        return False
    if isinstance(value, str):
        return normalize_text(value) not in {"", "nan", "none"}
    return True


def _derive_well_field(value) -> str | None:
    """Extract the prefix before `_` from `ID скважины`."""
    if pd.isna(value) or value is None:
        return None

    text = str(value).strip()
    if not text or "_" not in text:
        return None

    prefix = text.split("_", 1)[0].strip()
    return prefix or None


def _prepare_partition_columns(
    merged: pd.DataFrame,
    date_column: str,
    field_column: str,
) -> pd.DataFrame:
    prepared = merged.copy()
    prepared["_parsed_date"] = prepared[date_column].apply(parse_date)
    prepared["_parsed_year"] = prepared["_parsed_date"].apply(
        lambda value: None if value is None else int(value.year)
    )

    if field_column in prepared.columns:
        prepared["_field_value"] = prepared[field_column].apply(
            lambda value: None if normalize_text(value) == "" else str(value).strip()
        )
    else:
        prepared["_field_value"] = None

    if WELL_ID_COLUMN in prepared.columns:
        prepared["_well_field_value"] = prepared[WELL_ID_COLUMN].apply(_derive_well_field)
    else:
        prepared["_well_field_value"] = None

    return prepared


def _compute_date_quality_stats(
    prepared: pd.DataFrame,
    date_column: str,
) -> tuple[List[DateQualityStat], int]:
    valid_dates = prepared[prepared["_parsed_date"].notna()].copy()
    if valid_dates.empty:
        return [], 0

    content_columns = [
        column
        for column in prepared.columns
        if not str(column).startswith("_") and column != date_column
    ]
    if not content_columns:
        content_columns = [date_column]

    filled_values_per_row = valid_dates[content_columns].apply(
        lambda column: column.map(_is_filled_value)
    ).sum(axis=1)
    dataset_row_fill_median = float(filled_values_per_row.median()) if not filled_values_per_row.empty else 0.0
    filled_row_threshold = max(1, int(math.floor(dataset_row_fill_median * 0.75)))

    valid_dates["_filled_values_per_row"] = filled_values_per_row
    valid_dates["_date_key"] = valid_dates["_parsed_date"].dt.strftime("%Y-%m-%d")

    per_date_entries = valid_dates.groupby("_date_key").size().astype(float)
    valid_dates["_is_filled_row"] = valid_dates["_filled_values_per_row"] >= filled_row_threshold
    per_date_filled_ratio = valid_dates.groupby("_date_key")["_is_filled_row"].mean().astype(float)
    per_date_avg_fill = valid_dates.groupby("_date_key")["_filled_values_per_row"].mean().astype(float)

    median_entries = float(per_date_entries.median()) if not per_date_entries.empty else 0.0
    q1_entries = float(per_date_entries.quantile(0.25)) if not per_date_entries.empty else 0.0
    q3_entries = float(per_date_entries.quantile(0.75)) if not per_date_entries.empty else 0.0
    iqr_entries = q3_entries - q1_entries
    lower_entry_bound = max(
        1.0,
        median_entries * 0.6,
        q1_entries - 1.5 * iqr_entries if iqr_entries > 0 else 0.0,
    )
    lower_entry_bound = min(lower_entry_bound, median_entries) if median_entries > 0 else lower_entry_bound

    median_filled_ratio = float(per_date_filled_ratio.median()) if not per_date_filled_ratio.empty else 0.0
    median_avg_fill = float(per_date_avg_fill.median()) if not per_date_avg_fill.empty else 0.0

    date_stats: List[DateQualityStat] = []
    flagged_count = 0
    for date_key, frame in valid_dates.groupby("_date_key", sort=True):
        entry_count = int(len(frame))
        filled_row_count = int((frame["_filled_values_per_row"] >= filled_row_threshold).sum())
        filled_row_ratio = filled_row_count / entry_count if entry_count else 0.0
        avg_filled_values_per_entry = float(frame["_filled_values_per_row"].mean()) if entry_count else 0.0

        reasons: List[str] = []
        if median_entries > 0 and entry_count < lower_entry_bound:
            reasons.append(f"entries unusually low ({entry_count} vs median {median_entries:.1f})")
        if median_filled_ratio > 0 and filled_row_ratio < median_filled_ratio * 0.85 and filled_row_ratio < 0.95:
            reasons.append(
                f"filled-row ratio low ({filled_row_ratio:.0%} vs median {median_filled_ratio:.0%})"
            )
        if median_avg_fill > 0 and avg_filled_values_per_entry < median_avg_fill * 0.75:
            reasons.append(
                "average filled values per entry low "
                f"({avg_filled_values_per_entry:.1f} vs median {median_avg_fill:.1f})"
            )

        if reasons:
            flagged_count += 1
        date_stats.append(
            DateQualityStat(
                date_key=str(date_key),
                entry_count=entry_count,
                filled_row_count=filled_row_count,
                filled_row_ratio=filled_row_ratio,
                avg_filled_values_per_entry=avg_filled_values_per_entry,
                flagged_reasons=tuple(reasons),
            )
        )

    return date_stats, flagged_count


def _find_missing_dates(prepared: pd.DataFrame) -> List[str]:
    valid_dates = prepared[prepared["_parsed_date"].notna()]["_parsed_date"]
    if valid_dates.empty:
        return []

    normalized_dates = sorted(pd.Timestamp(value).normalize() for value in valid_dates.unique())
    if len(normalized_dates) <= 1:
        return []

    full_range = pd.date_range(start=normalized_dates[0], end=normalized_dates[-1], freq="D")
    existing = {date.strftime("%Y-%m-%d") for date in normalized_dates}
    return [date.strftime("%Y-%m-%d") for date in full_range if date.strftime("%Y-%m-%d") not in existing]


def _write_quality_report(
    output_root: Path,
    source_files: Sequence[Path],
    total_rows: int,
    invalid_date_rows: int,
    date_stats: Sequence[DateQualityStat],
    missing_dates: Sequence[str],
) -> Path:
    report_path = output_root / QUALITY_REPORT_FILENAME
    flagged_dates = [stat for stat in date_stats if stat.flagged_reasons]
    lines = [
        "TechRegime Data Quality Report",
        "",
        f"Files analyzed: {len(source_files)}",
        f"Total merged rows: {total_rows}",
        f"Rows with invalid dates: {invalid_date_rows}",
        f"Analyzed dates: {len(date_stats)}",
        f"Potentially incomplete dates: {len(flagged_dates)}",
        f"Missing dates in sequence: {len(missing_dates)}",
        "",
        "Heuristic:",
        "- Compare entry count by date against the dataset-wide median.",
        "- Compare row completeness by date using filled-value coverage across the row.",
        "- Dates below the usual range are marked as potentially missing data.",
        "- Dates absent from the daily sequence are listed separately as missing dates.",
        "",
        "Per-date summary:",
    ]

    for stat in date_stats:
        status = "POTENTIALLY_MISSING" if stat.flagged_reasons else "OK"
        lines.append(
            f"- {stat.date_key}: status={status}; entries={stat.entry_count}; "
            f"filled_rows={stat.filled_row_count}; filled_row_ratio={stat.filled_row_ratio:.0%}; "
            f"avg_filled_values_per_entry={stat.avg_filled_values_per_entry:.1f}"
        )
        for reason in stat.flagged_reasons:
            lines.append(f"  reason: {reason}")

    if not flagged_dates:
        lines.extend(
            [
                "",
                "No dates were flagged as potentially missing data.",
            ]
        )

    lines.append("")
    if missing_dates:
        lines.append("Missing dates:")
        for missing_date in missing_dates:
            lines.append(f"- {missing_date}")
    else:
        lines.append("No missing dates were identified in the daily sequence.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _partition_frame(
    prepared: pd.DataFrame,
    mode: PartitionMode,
) -> pd.DataFrame:
    if mode == "year":
        return prepared[prepared["_parsed_year"].notna()].copy()
    if mode == "field":
        return prepared[prepared["_field_value"].notna()].copy()
    if mode == "field_year":
        return prepared[
            prepared["_field_value"].notna() & prepared["_parsed_year"].notna()
        ].copy()
    if mode == "well_field":
        return prepared[prepared["_well_field_value"].notna()].copy()
    raise ValueError(f"Unsupported partition mode: {mode}")


def _build_partition_key(row: pd.Series, mode: PartitionMode) -> str:
    if mode == "year":
        return str(int(row["_parsed_year"]))
    if mode == "field":
        return str(row["_field_value"])
    if mode == "field_year":
        return f"{row['_field_value']}_{int(row['_parsed_year'])}"
    if mode == "well_field":
        return str(row["_well_field_value"])
    raise ValueError(f"Unsupported partition mode: {mode}")


def _build_output_filename(mode: PartitionMode, partition_key: str) -> str:
    return f"techregime_{mode}_{_sanitize_filename_part(partition_key)}.xlsx"


def merge_techregime_exports_partitioned(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    output_dir: Path | str | None = None,
    modes: Sequence[PartitionMode] = ("year",),
    date_column: str = DATE_COLUMN,
    field_column: str = FIELD_COLUMN,
    verbose: bool = False,
) -> PartitionResult:
    """Merge TechRegime exports and write one workbook per requested partition."""
    normalized_modes = tuple(dict.fromkeys(modes))
    source_files = tuple(resolve_techregime_files(source))
    merged = merge_techregime_exports(source_files, date_column=date_column, verbose=verbose)

    if merged.empty:
        return PartitionResult(
            source_files=source_files,
            total_rows=0,
            invalid_date_rows=0,
            invalid_field_rows=0,
            invalid_well_field_rows=0,
            quality_report_path=None,
            flagged_date_count=0,
            missing_dates=(),
            outputs=(),
        )

    prepared = _prepare_partition_columns(
        merged,
        date_column=date_column,
        field_column=field_column,
    )
    invalid_date_rows = int(prepared["_parsed_year"].isna().sum())
    invalid_field_rows = int(prepared["_field_value"].isna().sum())
    invalid_well_field_rows = int(prepared["_well_field_value"].isna().sum())

    output_root = Path(output_dir) if output_dir is not None else _default_output_dir(source_files)
    output_root.mkdir(parents=True, exist_ok=True)
    _verbose_print(f"Writing outputs into {output_root}", verbose=verbose)

    date_stats, flagged_date_count = _compute_date_quality_stats(
        prepared,
        date_column=date_column,
    )
    missing_dates = tuple(_find_missing_dates(prepared))
    quality_report_path = _write_quality_report(
        output_root=output_root,
        source_files=source_files,
        total_rows=len(merged),
        invalid_date_rows=invalid_date_rows,
        date_stats=date_stats,
        missing_dates=missing_dates,
    )
    _verbose_print("Quality report generated", verbose=verbose)

    partition_total = 0
    for mode in normalized_modes:
        mode_frame = _partition_frame(prepared, mode)
        if mode_frame.empty:
            continue
        mode_keys = mode_frame.apply(lambda row: _build_partition_key(row, mode), axis=1)
        partition_total += int(mode_keys.nunique())

    outputs: List[PartitionOutput] = []
    partition_index = 0
    for mode in normalized_modes:
        mode_root = output_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)

        mode_frame = _partition_frame(prepared, mode)
        if mode_frame.empty:
            continue

        mode_frame = mode_frame.copy()
        mode_frame["_partition_key"] = mode_frame.apply(
            lambda row: _build_partition_key(row, mode),
            axis=1,
        )

        for partition_key, partition_frame in mode_frame.groupby("_partition_key", sort=True):
            output_path = mode_root / _build_output_filename(mode, partition_key)
            export_frame = partition_frame.drop(
                columns=["_parsed_date", "_parsed_year", "_field_value", "_well_field_value", "_partition_key"]
            )
            export_frame.to_excel(output_path, index=False)
            partition_index += 1
            _render_progress("Writing partitions", partition_index, partition_total, verbose=verbose)
            outputs.append(
                PartitionOutput(
                    mode=mode,
                    partition_key=str(partition_key),
                    rows=len(export_frame),
                    output_path=output_path,
                )
            )

    return PartitionResult(
        source_files=source_files,
        total_rows=len(merged),
        invalid_date_rows=invalid_date_rows,
        invalid_field_rows=invalid_field_rows,
        invalid_well_field_rows=invalid_well_field_rows,
        quality_report_path=quality_report_path,
        flagged_date_count=flagged_date_count,
        missing_dates=missing_dates,
        outputs=tuple(outputs),
    )


def merge_techregime_exports_by_year(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    output_dir: Path | str | None = None,
    date_column: str = DATE_COLUMN,
    verbose: bool = False,
) -> MergeByYearResult:
    """Backward-compatible wrapper for the year-only split."""
    result = merge_techregime_exports_partitioned(
        source=source,
        output_dir=output_dir,
        modes=("year",),
        date_column=date_column,
        verbose=verbose,
    )
    yearly_outputs = tuple(
        YearlyOutput(
            year=int(output.partition_key),
            rows=output.rows,
            output_path=output.output_path,
        )
        for output in result.outputs
        if output.mode == "year"
    )
    return MergeByYearResult(
        source_files=result.source_files,
        total_rows=result.total_rows,
        invalid_date_rows=result.invalid_date_rows,
        yearly_outputs=yearly_outputs,
    )
