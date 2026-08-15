"""Telemetry export loading and merging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from analysis.paths import resolve_telemetry_source_dir

from ..io import load_excel_file


#: Newest dated export drop -- see ``analysis.paths.resolve_telemetry_source_dir``.
DEFAULT_SOURCE_DIR = resolve_telemetry_source_dir()
WELL_COLUMN = "Скважина"
DATE_COLUMN = "Дата"
TIMESTAMP_COLUMN = "Дата"
FIELD_COLUMN = "Месторождение"
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}


def load_telemetry_export(file_path: Path) -> pd.DataFrame:
    """Load one telemetry export into a flat dataframe."""
    file_path = Path(file_path)
    frame, _, _ = load_excel_file(str(file_path))
    frame = frame.copy()
    frame = frame.dropna(how="all").reset_index(drop=True)
    frame["_source_file"] = file_path.name
    frame["_source_path"] = str(file_path)
    return frame


def resolve_telemetry_files(source: Path | str | Sequence[Path | str]) -> List[Path]:
    """Resolve one or more files/directories into concrete telemetry files."""
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


def merge_telemetry_exports(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read and concatenate all discovered telemetry export files."""
    files = resolve_telemetry_files(source)
    if not files:
        return pd.DataFrame()

    if verbose:
        print(f"Loading {len(files)} telemetry file(s)")

    frames = []
    for index, file_path in enumerate(files, start=1):
        frames.append(load_telemetry_export(file_path))
        if verbose:
            print(f"  [{index}/{len(files)}] {file_path.name}")
    return pd.concat(frames, ignore_index=True, sort=False)
