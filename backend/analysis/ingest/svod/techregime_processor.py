"""Techregime dataset processing."""

import re
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..config import TECHREGIME_COLUMN_RULES
from ..io import discover_source_files, normalize_source_input
from ..normalize import normalize_text, normalize_well, parse_date, parse_number
from .enrich import (
    build_techregime_columns_from_header_rows,
    get_first_non_empty_value_from_files,
    normalize_techregime_header,
)
from ..techregime.processor import (
    DATE_COLUMN as FLAT_DATE_COLUMN,
    WELL_ID_COLUMN as FLAT_WELL_ID_COLUMN,
    load_techregime_export,
    resolve_techregime_files as resolve_flat_techregime_files,
)
from ..techregime.extract import (
    DEFAULT_QUERY_INTERVAL,
    extract_techregime_values_from_frame,
)


SQLITE_TECHREGIME_SUFFIXES = {".sqlite", ".db"}
WELL_NUMBER_COLUMN = "№ скважины"
TECHREGIME_FIELD_COLUMN = "М/р"
TECHREGIME_SQLITE_FIELD_FALLBACK = "_meta_field"
TECHREGIME_FIELD_TARGET_COLUMNS = ("field", "Месторождение")
TECHREGIME_DERIVED_TARGET_COLUMNS = ("ГЖФ",)
TECHREGIME_VALUE_ALIASES = {
    "Тип ствола скв": "wellbore_type",
    "Дебит жидк.": "liquid_rate",
    "Дебит нефти": "oil_rate",
    "Дебит газа": "gas_rate",
    "Газовый фактор": "gas_factor",
    "ГЖФ": "gas_liquid_ratio",
    "Кпрод.": "productivity_index",
    "Обводненность": "water_cut",
    "Плотн. нефти": "oil_density",
    "Плотн. Воды": "water_density",
    "Рпл.": "reservoir_pressure",
    "Рзаб": "bottomhole_pressure",
    "Частота": "frequency",
    "Загр, Двиг,": "motor_load",
    "Ртр": "rtr",
    "Рзт": "rzt",
    "Рлин": "rlin",
}
TECHREGIME_ZERO_SAFE_TARGET_COLUMNS = (
    "Дебит жидк.",
    "Дебит нефти",
    "Дебит газа",
    "Газовый фактор",
    "Кпрод.",
    "Обводненность",
    "Плотн. нефти",
    "Плотн. Воды",
    "Рпл.",
    "Рзаб",
    "Частота",
    "Загр, Двиг,",
    "Ртр",
    "Рзт",
    "Рлин",
)
TECHREGIME_STATUS_ALLOWED_VALUES = {"в работе"}
TECHREGIME_HISTORY_LOOKBACK = pd.Timedelta(days=365)
TECHREGIME_RUNTIME_COLUMNS = ("runtime_nno", "Наработка (сут)", "I. Наработка (сут)")
TECHREGIME_OPTIONAL_EXCLUDED_SOURCE_COLUMNS = {
    FLAT_DATE_COLUMN,
    FLAT_WELL_ID_COLUMN,
    WELL_NUMBER_COLUMN,
}


def detect_techregime_header_rows(file_path: str, sheet_name=0) -> Optional[Tuple[int, int, int]]:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=10)
    for row_idx in range(1, len(raw) - 1):
        row_values = [normalize_techregime_header(v) for v in raw.iloc[row_idx].tolist()[:10]]
        if "м/р" in row_values or "месторождение" in row_values:
            return (row_idx - 1, row_idx, row_idx + 1)
    return None


def flatten_techregime_column(column_label: tuple) -> Tuple[str, str]:
    if not isinstance(column_label, tuple):
        field = normalize_techregime_header(column_label)
        return "", field
    parts = [normalize_techregime_header(part) for part in column_label]
    parts = [part for part in parts if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[-2], parts[-1]


def parse_techregime_file_date(file_path: Path):
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", file_path.name)
    if not match:
        return None
    return parse_date(match.group(1))


def find_techregime_month_folder(root_path: Path, failure_date) -> Optional[Path]:
    fail_ts = parse_date(failure_date)
    if fail_ts is None:
        return None
    year_dir = root_path / f"{fail_ts.year}"
    if not year_dir.exists():
        return None
    month_prefix = f"{fail_ts.month:02d} "
    for child in year_dir.iterdir():
        if child.is_dir() and child.name.startswith(month_prefix):
            return child
    return None


def list_techregime_candidate_files(root_path, failure_date) -> List[Path]:
    discovered_files = discover_source_files(root_path, "techregime")
    if not discovered_files:
        return []
    if len(discovered_files) == 1 and discovered_files[0].is_file():
        return discovered_files

    directory_roots = [path for path in normalize_source_input(root_path) if path.exists() and path.is_dir()]
    if not directory_roots:
        filtered = []
        fail_ts = parse_date(failure_date)
        for file_path in discovered_files:
            file_date = parse_techregime_file_date(file_path)
            if fail_ts is None or file_date is None or (file_date.year == fail_ts.year and file_date.month == fail_ts.month):
                filtered.append(file_path)
        return sorted(filtered or discovered_files, key=lambda path: ((parse_techregime_file_date(path) or pd.Timestamp.min).value, path.name), reverse=True)

    month_dir = None
    for source_path in directory_roots:
        month_dir = find_techregime_month_folder(source_path, failure_date)
        if month_dir is not None:
            break
    if month_dir is None:
        return []

    files = []
    for file_path in month_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.name.startswith("~$"):
            continue
        if file_path.suffix.lower() not in {".xls", ".xlsx", ".xlsm"}:
            continue
        file_name_norm = normalize_text(file_path.name)
        if not file_name_norm.startswith("тр_"):
            continue
        if "газ" in file_name_norm or "ппд" in file_name_norm:
            continue
        files.append(file_path)
    return sorted(files, key=lambda path: ((parse_techregime_file_date(path) or pd.Timestamp.min).value, path.name), reverse=True)


def load_techregime_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    header_rows = detect_techregime_header_rows(str(file_path))
    if header_rows is None:
        return None
    try:
        raw = pd.read_excel(str(file_path), header=None)
    except Exception:
        return None
    header_frame = raw.iloc[list(header_rows)].copy()
    columns = build_techregime_columns_from_header_rows(header_frame)
    if not columns:
        return None
    data_start_row = header_rows[-1] + 1
    df = raw.iloc[data_start_row:].reset_index(drop=True)
    if df.empty:
        return None
    df.columns = pd.MultiIndex.from_tuples(columns, names=["group", "field"])
    return df.sort_index(axis=1)


def resolve_techregime_sqlite_path(source) -> Optional[Path]:
    candidates: list[Path] = []
    for path in normalize_source_input(source):
        if path.exists() and path.is_file() and path.suffix.lower() in SQLITE_TECHREGIME_SUFFIXES:
            candidates.append(path.resolve())
            continue
        if path.exists() and path.is_dir():
            for suffix in sorted(SQLITE_TECHREGIME_SUFFIXES):
                candidates.extend(sorted(item.resolve() for item in path.glob(f"*{suffix}") if item.is_file()))

    unique_candidates = sorted(set(candidates), key=lambda item: (item.name.lower(), str(item).lower()))
    if not unique_candidates:
        return None
    preferred = [path for path in unique_candidates if path.stem.lower() == "techregime"]
    if len(preferred) == 1:
        return preferred[0]
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    latest_candidate = max(unique_candidates, key=lambda item: item.stat().st_mtime)
    return latest_candidate


def flatten_sqlite_techregime_rule(source_col: tuple[str | None, str]) -> str:
    if source_col[0]:
        return f"{source_col[0]} | {source_col[1]}"
    return source_col[1]


def get_default_techregime_selected_columns() -> list[str]:
    return [
        "Месторождение",
        *TECHREGIME_COLUMN_RULES.keys(),
        *TECHREGIME_DERIVED_TARGET_COLUMNS,
    ]


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_techregime_request_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(str(value))
    return unique_values


def _normalize_selected_techregime_columns(selected_columns: Sequence[str] | None) -> list[str]:
    if selected_columns is None:
        return get_default_techregime_selected_columns()
    return _dedupe_preserving_order([str(column).strip() for column in selected_columns if str(column).strip()])


def _iter_builtin_source_column_map() -> Iterable[tuple[str, str]]:
    yield ("Месторождение", TECHREGIME_FIELD_COLUMN)
    for target_col, source_col in TECHREGIME_COLUMN_RULES.items():
        yield (target_col, flatten_sqlite_techregime_rule(source_col))


def _builtin_target_from_source_column(source_column: str) -> str | None:
    normalized_source = _normalized_techregime_request_name(source_column)
    for target_col, builtin_source in _iter_builtin_source_column_map():
        if _normalized_techregime_request_name(builtin_source) == normalized_source:
            return target_col
    return None


def _strip_techregime_top_level(source_column: str) -> str:
    text = str(source_column).strip()
    if "|" not in text:
        return text
    return text.split("|", 1)[1].strip() or text


def _target_column_to_source_request_name(target_col: str) -> str | None:
    normalized_target = _normalized_techregime_request_name(target_col)
    if normalized_target in {"месторождение", "field"}:
        return TECHREGIME_FIELD_COLUMN
    if target_col in TECHREGIME_COLUMN_RULES:
        return flatten_sqlite_techregime_rule(TECHREGIME_COLUMN_RULES[target_col])
    if target_col in TECHREGIME_DERIVED_TARGET_COLUMNS:
        return None
    return _strip_techregime_top_level(target_col)


def _selected_includes_column(selected_columns: Sequence[str], column_name: str) -> bool:
    normalized_target = _normalized_techregime_request_name(column_name)
    return any(
        _normalized_techregime_request_name(selected_column) == normalized_target
        for selected_column in selected_columns
    )


def _legacy_request_tuple_from_target_column(target_col: str) -> tuple[str, str] | None:
    if target_col in TECHREGIME_COLUMN_RULES:
        source_group, source_field = TECHREGIME_COLUMN_RULES[target_col]
        return (
            normalize_techregime_header(source_group) if source_group else "",
            normalize_techregime_header(source_field),
        )

    source_request = _target_column_to_source_request_name(target_col)
    if not source_request:
        return None
    if "|" in source_request:
        group_name, field_name = [part.strip() for part in source_request.split("|", 1)]
        return (
            normalize_techregime_header(group_name),
            normalize_techregime_header(field_name),
        )
    return ("", normalize_techregime_header(source_request))


def _legacy_request_tuple_from_matching_rows(
    target_col: str,
    matching_rows_by_file: Sequence[pd.DataFrame],
) -> tuple[str, str] | None:
    direct_tuple = _legacy_request_tuple_from_target_column(target_col)
    if target_col in TECHREGIME_COLUMN_RULES or direct_tuple is None:
        return direct_tuple

    normalized_target = _normalized_techregime_request_name(target_col)
    resolved_matches: list[tuple[str, str]] = []
    for rows in matching_rows_by_file:
        for column in rows.columns:
            if not isinstance(column, tuple):
                continue
            group_name = normalize_techregime_header(column[0]) if column[0] else ""
            field_name = normalize_techregime_header(column[1]) if len(column) > 1 else ""
            source_name = flatten_sqlite_techregime_rule((group_name or None, field_name))
            if _normalized_techregime_request_name(_strip_techregime_top_level(source_name)) != normalized_target:
                continue
            candidate = (group_name, field_name)
            if candidate not in resolved_matches:
                resolved_matches.append(candidate)
    if len(resolved_matches) == 1:
        return resolved_matches[0]
    return direct_tuple


def _discover_legacy_selectable_columns(source) -> list[str]:
    candidate_files = list_techregime_candidate_files(source, None)
    if not candidate_files:
        return []
    techregime_df = load_techregime_dataframe(candidate_files[0])
    if techregime_df is None or techregime_df.empty:
        return []

    discovered: list[str] = []
    for column in techregime_df.columns:
        if not isinstance(column, tuple):
            source_name = str(column)
        else:
            group_name = normalize_techregime_header(column[0]) if column[0] else ""
            field_name = normalize_techregime_header(column[1]) if len(column) > 1 else ""
            source_name = flatten_sqlite_techregime_rule((group_name or None, field_name))
        builtin_name = _builtin_target_from_source_column(source_name)
        discovered.append(builtin_name or _strip_techregime_top_level(source_name))
    return discovered


def discover_available_techregime_columns(source, *, prefer_sqlite: bool = True) -> list[str]:
    discovered: list[str] = []

    sqlite_path = resolve_techregime_sqlite_path(source) if prefer_sqlite else None
    if sqlite_path is not None:
        source_columns = _load_sqlite_original_columns(sqlite_path)
        discovered.extend(source_columns)
    else:
        flat_frame = _prepare_flat_techregime_frame(source)
        if not flat_frame.empty:
            discovered.extend(str(column) for column in flat_frame.columns if not str(column).startswith("_"))
        else:
            discovered.extend(_discover_legacy_selectable_columns(source))

    selectable_columns: list[str] = []
    for column_name in discovered:
        if str(column_name).startswith("_"):
            continue
        builtin_name = _builtin_target_from_source_column(str(column_name))
        if builtin_name is not None:
            selectable_columns.append(builtin_name)
            continue
        if column_name in TECHREGIME_OPTIONAL_EXCLUDED_SOURCE_COLUMNS:
            continue
        selectable_columns.append(_strip_techregime_top_level(str(column_name)))

    if "Дебит жидк." in selectable_columns and "Дебит газа" in selectable_columns:
        selectable_columns.append("ГЖФ")

    return _dedupe_preserving_order(selectable_columns)


def get_first_non_empty_sqlite_value(rows: pd.DataFrame, column_name: str):
    if rows.empty or column_name not in rows.columns:
        return None
    for value in rows[column_name].tolist():
        if pd.isna(value) or value in ["", "-", "–"]:
            continue
        return value
    return None


def get_first_non_empty_flat_value(rows: pd.DataFrame, column_name: str):
    if rows.empty or column_name not in rows.columns:
        return None
    for value in rows[column_name].tolist():
        if pd.isna(value) or value in ["", "-", "–"]:
            continue
        return value
    return None


def find_techregime_well_rows(df: pd.DataFrame, well_value) -> pd.DataFrame:
    well_norm = normalize_well(well_value)
    candidates = []
    for col in [("", "id скважины"), ("", "№ скважины")]:
        if col in df.columns:
            candidates.append(col)
    if not candidates:
        return df.iloc[0:0].copy()
    mask = pd.Series(False, index=df.index)
    for col in candidates:
        selected = df[col]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        mask = mask | selected.apply(lambda x: normalize_well(x) == well_norm)
    return df[mask].copy()


def _prepare_flat_techregime_frame(source) -> pd.DataFrame:
    flat_files = tuple(resolve_flat_techregime_files(source))
    if not flat_files:
        return pd.DataFrame()

    frames = [load_techregime_export(file_path) for file_path in flat_files]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    prepared = merged.copy()
    prepared["_techregime_parsed_date"] = prepared[FLAT_DATE_COLUMN].apply(parse_date)

    if FLAT_WELL_ID_COLUMN in prepared.columns:
        prepared["_techregime_normalized_well_id"] = prepared[FLAT_WELL_ID_COLUMN].apply(normalize_well)
    else:
        prepared["_techregime_normalized_well_id"] = ""

    if WELL_NUMBER_COLUMN in prepared.columns:
        prepared["_techregime_normalized_well_number"] = prepared[WELL_NUMBER_COLUMN].apply(normalize_well)
    else:
        prepared["_techregime_normalized_well_number"] = ""

    return prepared


def _ensure_flat_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "_techregime_parsed_date" not in prepared.columns and FLAT_DATE_COLUMN in prepared.columns:
        prepared["_techregime_parsed_date"] = prepared[FLAT_DATE_COLUMN].apply(parse_date)
    if "_techregime_normalized_well_id" not in prepared.columns:
        if FLAT_WELL_ID_COLUMN in prepared.columns:
            prepared["_techregime_normalized_well_id"] = prepared[FLAT_WELL_ID_COLUMN].apply(normalize_well)
        else:
            prepared["_techregime_normalized_well_id"] = ""
    if "_techregime_normalized_well_number" not in prepared.columns:
        if WELL_NUMBER_COLUMN in prepared.columns:
            prepared["_techregime_normalized_well_number"] = prepared[WELL_NUMBER_COLUMN].apply(normalize_well)
        else:
            prepared["_techregime_normalized_well_number"] = ""
    return prepared


def _find_flat_techregime_well_rows(df: pd.DataFrame, well_value, failure_date) -> pd.DataFrame:
    if df.empty:
        return df

    prepared = _ensure_flat_helper_columns(df)

    well_norm = normalize_well(well_value)
    fail_ts = parse_date(failure_date)
    mask = pd.Series(True, index=prepared.index)

    if well_norm:
        mask = mask & (
            (prepared["_techregime_normalized_well_id"] == well_norm)
            | (prepared["_techregime_normalized_well_number"] == well_norm)
        )

    if fail_ts is not None and "_techregime_parsed_date" in prepared.columns:
        mask = mask & prepared["_techregime_parsed_date"].notna()
        mask = mask & (prepared["_techregime_parsed_date"] <= fail_ts)

    rows = prepared.loc[mask].copy()
    if rows.empty:
        return rows
    if "_techregime_parsed_date" in rows.columns:
        return rows.sort_values("_techregime_parsed_date", ascending=False, na_position="last")
    return rows


def _is_present_value(value) -> bool:
    value = _normalize_techregime_scalar_value(value)
    if value is None:
        return False
    if isinstance(value, str):
        return normalize_text(value) not in {"", "nan", "none", "-", "–"}
    return not pd.isna(value)


def _normalize_techregime_scalar_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value if normalize_text(value) not in {"", "nan", "none", "-", "–"} else None
    if isinstance(value, (list, tuple, set, pd.Series, pd.Index)):
        flattened_values = []
        for item in list(value):
            normalized_item = _normalize_techregime_scalar_value(item)
            if normalized_item is not None:
                flattened_values.append(normalized_item)
        unique_values = []
        seen_markers = set()
        for item in flattened_values:
            if isinstance(item, str):
                marker = f"str:{normalize_text(item)}"
            else:
                marker = f"obj:{repr(item)}"
            if marker in seen_markers:
                continue
            seen_markers.add(marker)
            unique_values.append(item)
        if len(unique_values) == 1:
            return unique_values[0]
        return None
    if pd.isna(value) or value is None:
        return None
    return value


def _ensure_techregime_columns(
    records: pd.DataFrame,
    *,
    selected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    for target_col in _iter_techregime_target_columns(selected_columns):
        if target_col not in records.columns:
            records[target_col] = None
    return records


def _assign_techregime_value(records: pd.DataFrame, row_idx: int, target_col: str, value) -> None:
    value = _normalize_techregime_scalar_value(value)
    if not _is_present_value(value):
        return
    records.loc[row_idx, target_col] = value
    alias = TECHREGIME_VALUE_ALIASES.get(target_col)
    if alias:
        records.loc[row_idx, alias] = value


def _clear_techregime_value(records: pd.DataFrame, row_idx: int, target_col: str) -> None:
    if target_col in records.columns:
        records.loc[row_idx, target_col] = None
    alias = TECHREGIME_VALUE_ALIASES.get(target_col)
    if alias and alias in records.columns:
        records.loc[row_idx, alias] = None


def _assign_techregime_field(records: pd.DataFrame, row_idx: int, field_value) -> None:
    field_value = _normalize_techregime_scalar_value(field_value)
    if not _is_present_value(field_value):
        return
    for target_col in TECHREGIME_FIELD_TARGET_COLUMNS:
        records.loc[row_idx, target_col] = field_value


def _iter_techregime_target_columns(selected_columns: Sequence[str] | None = None) -> Iterable[str]:
    normalized_selected = _normalize_selected_techregime_columns(selected_columns)
    for target_col in normalized_selected:
        if _normalized_techregime_request_name(target_col) in {"месторождение", "field"}:
            yield from TECHREGIME_FIELD_TARGET_COLUMNS
            continue
        yield target_col
        alias = TECHREGIME_VALUE_ALIASES.get(target_col)
        if alias:
            yield alias


def _apply_techregime_derived_values(
    records: pd.DataFrame,
    row_idx: int,
    *,
    selected_columns: Sequence[str] | None = None,
) -> None:
    selected_target_columns = _normalize_selected_techregime_columns(selected_columns)
    if not _selected_includes_column(selected_target_columns, "ГЖФ"):
        return
    liquid_rate = parse_number(records.loc[row_idx].get("Дебит жидк.", records.loc[row_idx].get("liquid_rate")))
    gas_rate = parse_number(records.loc[row_idx].get("Дебит газа", records.loc[row_idx].get("gas_rate")))
    if liquid_rate is None or gas_rate is None or liquid_rate <= 0 or gas_rate <= 0:
        _clear_techregime_value(records, row_idx, "ГЖФ")
        return
    _assign_techregime_value(records, row_idx, "ГЖФ", gas_rate / liquid_rate)


def _normalized_techregime_request_name(column_name: str) -> str:
    return normalize_text(column_name)


def _build_techregime_request_map(
    selected_columns: Sequence[str] | None = None,
) -> dict[str, str]:
    request_map: dict[str, str] = {}
    for target_col in _normalize_selected_techregime_columns(selected_columns):
        source_name = _target_column_to_source_request_name(target_col)
        if source_name is None:
            continue
        if _normalized_techregime_request_name(target_col) in {"месторождение", "field"}:
            request_map["__field__"] = TECHREGIME_FIELD_COLUMN
            continue
        request_map[target_col] = source_name
    if "Месторождение" not in request_map and "__field__" not in request_map:
        request_map["__field__"] = TECHREGIME_FIELD_COLUMN
    return request_map


def _filter_available_request_map(
    request_map: dict[str, str],
    available_columns: Iterable[object],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    available_column_list = [column for column in available_columns if column is not None]
    for target_col, column_name in request_map.items():
        matching_column = _find_matching_column_name(available_column_list, column_name)
        if matching_column is not None:
            resolved[target_col] = matching_column
    return resolved


def _load_sqlite_original_columns(sqlite_path: Path) -> list[str]:
    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(
            "SELECT original_name FROM techregime_column_map ORDER BY column_order",
            connection,
        )
    return frame["original_name"].astype(str).tolist()


def _parse_interval_value(interval) -> pd.Timedelta:
    if interval is None:
        return DEFAULT_QUERY_INTERVAL
    if isinstance(interval, pd.Timedelta):
        return interval
    if isinstance(interval, timedelta):
        return pd.Timedelta(interval)
    if isinstance(interval, (int, float)):
        return pd.Timedelta(days=float(interval))
    parsed = pd.to_timedelta(interval)
    if pd.isna(parsed) or parsed < pd.Timedelta(0):
        raise ValueError("Interval must be a non-negative duration.")
    return parsed


def _resolve_record_interval(record, default_interval) -> pd.Timedelta:
    for column_name in TECHREGIME_RUNTIME_COLUMNS:
        runtime_days = parse_number(record.get(column_name))
        if runtime_days is not None and runtime_days > 0:
            return pd.Timedelta(days=float(runtime_days))
    return _parse_interval_value(default_interval)


def _find_matching_column_name(columns: Iterable[object], requested_name: str) -> str | None:
    normalized_requested = _normalized_techregime_request_name(requested_name)
    suffix_matches: list[str] = []
    for column in columns:
        column_text = str(column)
        if _normalized_techregime_request_name(column_text) == normalized_requested:
            return column_text
        if "|" in column_text and _normalized_techregime_request_name(_strip_techregime_top_level(column_text)) == normalized_requested:
            suffix_matches.append(column_text)
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def _find_status_column_name(columns: Iterable[object]) -> str | None:
    for column in columns:
        normalized = _normalized_techregime_request_name(str(column))
        if normalized == "статус" or normalized.endswith("| статус"):
            return str(column)
    return None


def _prepare_history_rows(rows: pd.DataFrame) -> pd.DataFrame:
    prepared = rows.copy()
    if FLAT_DATE_COLUMN in prepared.columns:
        prepared["_history_date"] = prepared[FLAT_DATE_COLUMN].apply(parse_date)
    elif "_meta_report_date" in prepared.columns:
        prepared["_history_date"] = prepared["_meta_report_date"].apply(parse_date)
    else:
        prepared["_history_date"] = pd.NaT
    return prepared[prepared["_history_date"].notna()].copy()


def _slice_history_lookback(rows: pd.DataFrame, failure_date, interval=TECHREGIME_HISTORY_LOOKBACK) -> pd.DataFrame:
    prepared = _prepare_history_rows(rows)
    failure_ts = parse_date(failure_date)
    if prepared.empty:
        return prepared
    if failure_ts is None:
        return prepared.sort_values("_history_date", ascending=False, na_position="last")

    interval_end = failure_ts.normalize()
    interval_start = interval_end - _parse_interval_value(interval)
    sliced = prepared[
        (prepared["_history_date"].dt.normalize() >= interval_start)
        & (prepared["_history_date"].dt.normalize() <= interval_end)
    ].copy()
    return sliced.sort_values("_history_date", ascending=False, na_position="last")


def _filter_status_rows(rows: pd.DataFrame) -> pd.DataFrame:
    status_column = _find_status_column_name(rows.columns)
    if status_column is None:
        return rows
    return rows[
        rows[status_column].apply(
            lambda value: _normalized_techregime_request_name(value) in TECHREGIME_STATUS_ALLOWED_VALUES
        )
    ].copy()


def _latest_non_empty_history_value(rows: pd.DataFrame, *column_names: str):
    if rows.empty:
        return None
    for column_name in column_names:
        if column_name not in rows.columns:
            continue
        for value in rows[column_name].tolist():
            if _is_present_value(value):
                return value
    return None


def _parse_positive_numeric_value(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    return float(numeric)


def _positive_numeric_mean(values: pd.Series):
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[numeric.notna() & (numeric > 0)]
    if valid.empty:
        return None
    result = float(valid.mean())
    if result <= 0:
        return None
    return result


def _average_zero_safe_interval(
    rows: pd.DataFrame,
    column_name: str,
    *,
    interval_end,
    interval,
):
    if rows.empty or column_name not in rows.columns:
        return None
    interval_end_ts = parse_date(interval_end)
    if interval_end_ts is None:
        return None

    interval_delta = _parse_interval_value(interval)
    interval_end_norm = interval_end_ts.normalize()
    interval_start = interval_end_norm - interval_delta
    window_rows = rows[
        (rows["_history_date"].dt.normalize() >= interval_start)
        & (rows["_history_date"].dt.normalize() <= interval_end_norm)
    ].copy()
    window_rows = _filter_status_rows(window_rows)
    return _positive_numeric_mean(window_rows[column_name])


def _extract_zero_safe_value_from_history(
    rows: pd.DataFrame,
    target_col: str,
    *,
    interval_end,
    interval,
):
    column_name = _find_matching_column_name(
        rows.columns,
        flatten_sqlite_techregime_rule(TECHREGIME_COLUMN_RULES[target_col]),
    )
    if column_name is None:
        return None

    primary_value = _average_zero_safe_interval(
        rows,
        column_name,
        interval_end=interval_end,
        interval=interval,
    )
    return primary_value


def _apply_zero_safe_history_values(
    records: pd.DataFrame,
    row_idx: int,
    history_rows: pd.DataFrame,
    *,
    failure_date,
    interval,
    selected_columns: Sequence[str] | None = None,
) -> None:
    prepared_history = _prepare_history_rows(history_rows)
    if prepared_history.empty:
        return
    selected_target_columns = _normalize_selected_techregime_columns(selected_columns)
    zero_safe_targets = [
        target_col for target_col in TECHREGIME_ZERO_SAFE_TARGET_COLUMNS
        if _selected_includes_column(selected_target_columns, target_col)
    ]
    for target_col in zero_safe_targets:
        value = _extract_zero_safe_value_from_history(
            prepared_history,
            target_col,
            interval_end=failure_date,
            interval=interval,
        )
        if value is not None:
            _assign_techregime_value(records, row_idx, target_col, value)


def _apply_techregime_consistency_checks(
    records: pd.DataFrame,
    row_idx: int,
    *,
    history_rows: pd.DataFrame | None = None,
) -> None:
    """Run extendable post-enrichment consistency checks for TechRegime fields."""
    row = records.loc[row_idx]
    bottomhole_pressure = _parse_positive_numeric_value(
        row.get("Рзаб", row.get("bottomhole_pressure"))
    )
    if bottomhole_pressure is None:
        _clear_techregime_value(records, row_idx, "Рзаб")
        return

    well_type = normalize_text(
        row.get(
            "Тип ствола скв",
            row.get(
                "wellbore_type",
                row.get("Тип скв"),
            ),
        )
    )
    if not well_type and history_rows is not None and not history_rows.empty:
        well_type = normalize_text(
            _latest_non_empty_history_value(history_rows, "Тип ствола скв", "Тип скв")
        )
    reservoir_pressure = _parse_positive_numeric_value(
        row.get("Рпл.", row.get("reservoir_pressure"))
    )

    if well_type == "добывающая" and reservoir_pressure is not None and bottomhole_pressure > reservoir_pressure:
        _clear_techregime_value(records, row_idx, "Рзаб")


def _load_sqlite_well_history(sqlite_path: Path, well_value, failure_date, interval=TECHREGIME_HISTORY_LOOKBACK) -> pd.DataFrame:
    normalized_well = normalize_well(well_value)
    failure_ts = parse_date(failure_date)
    if not normalized_well:
        return pd.DataFrame()

    with sqlite3.connect(sqlite_path) as connection:
        column_map = pd.read_sql_query(
            "SELECT storage_name, original_name FROM techregime_column_map ORDER BY column_order",
            connection,
        )
        storage_columns = [f'"{row["storage_name"]}"' for _, row in column_map.iterrows()]
        select_columns = [
            *storage_columns,
            "_meta_report_date",
            "_meta_normalized_well_id",
            "_meta_normalized_well_number",
            "_meta_field",
            "_meta_row_id",
        ]
        where_clauses = [
            "(_meta_normalized_well_id = ? OR _meta_normalized_well_number = ?)",
        ]
        params: list[object] = [normalized_well, normalized_well]
        if failure_ts is not None:
            lookback_delta = _parse_interval_value(interval)
            where_clauses.append("_meta_report_date >= ?")
            where_clauses.append("_meta_report_date <= ?")
            params.extend(
                [
                    (failure_ts.normalize() - lookback_delta).strftime("%Y-%m-%d"),
                    failure_ts.strftime("%Y-%m-%d"),
                ]
            )
        sql = (
            f"SELECT {', '.join(select_columns)} FROM techregime_records "
            f"WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY _meta_report_date ASC, _meta_row_id ASC"
        )
        rows = pd.read_sql_query(sql, connection, params=params)

    if rows.empty:
        return rows

    renamed_columns = {
        str(row["storage_name"]): str(row["original_name"])
        for _, row in column_map.iterrows()
    }
    restored = rows.rename(columns=renamed_columns)
    if FLAT_DATE_COLUMN not in restored.columns and "_meta_report_date" in restored.columns:
        restored[FLAT_DATE_COLUMN] = restored["_meta_report_date"]
    if TECHREGIME_FIELD_COLUMN not in restored.columns and "_meta_field" in restored.columns:
        restored[TECHREGIME_FIELD_COLUMN] = restored["_meta_field"]
    return restored


def _build_flat_well_history_index(df: pd.DataFrame) -> dict[str, list[int]]:
    prepared = _ensure_flat_helper_columns(df)
    index_map: dict[str, set[int]] = {}
    for column_name in ("_techregime_normalized_well_id", "_techregime_normalized_well_number"):
        if column_name not in prepared.columns:
            continue
        grouped = prepared.groupby(column_name).groups
        for normalized_well, row_indexes in grouped.items():
            if not normalized_well:
                continue
            index_map.setdefault(str(normalized_well), set()).update(int(index) for index in row_indexes)
    return {
        normalized_well: sorted(row_indexes)
        for normalized_well, row_indexes in index_map.items()
    }


def _get_flat_well_history_rows(
    df: pd.DataFrame,
    history_index: dict[str, list[int]],
    well_value,
    failure_date,
    interval=TECHREGIME_HISTORY_LOOKBACK,
) -> pd.DataFrame:
    well_norm = normalize_well(well_value)
    if not well_norm:
        return df.iloc[0:0].copy()
    row_indexes = history_index.get(well_norm)
    if not row_indexes:
        return df.iloc[0:0].copy()
    rows = df.loc[row_indexes].copy()
    return _slice_history_lookback(rows, failure_date, interval=interval)


def refresh_records_from_techregime(
    records: pd.DataFrame,
    techregime_path,
    well_col: str = "well",
    failure_date_col: str = "failure_date",
    overwrite: bool = True,
    prefer_sqlite: bool = True,
    interval=DEFAULT_QUERY_INTERVAL,
    selected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Refresh TechRegime-managed fields on an existing dataframe."""
    refreshed = enrich_from_techregime(
        records,
        techregime_path=techregime_path,
        well_col=well_col,
        failure_date_col=failure_date_col,
        prefer_sqlite=prefer_sqlite,
        interval=interval,
        selected_columns=selected_columns,
    )
    if overwrite:
        return refreshed

    result = records.copy()
    for column in _iter_techregime_target_columns(selected_columns):
        if column not in refreshed.columns:
            continue
        if column not in result.columns:
            result[column] = refreshed[column]
            continue
        existing = result[column]
        result[column] = existing.where(existing.apply(_is_present_value), refreshed[column])
    if "techregime_match_confidence" in refreshed.columns:
        result["techregime_match_confidence"] = refreshed["techregime_match_confidence"]
    return result


def enrich_from_techregime(
    records: pd.DataFrame,
    techregime_path,
    well_col: str = "well",
    failure_date_col: str = "failure_date",
    prefer_sqlite: bool = True,
    interval=DEFAULT_QUERY_INTERVAL,
    selected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Enrich from TechRegime SQLite, flat new-style Excel files, or legacy month folders."""
    records = records.copy()
    file_cache: Dict[Path, Optional[pd.DataFrame]] = {}
    selected_target_columns = _normalize_selected_techregime_columns(selected_columns)
    records = _ensure_techregime_columns(records, selected_columns=selected_target_columns)
    request_map = _build_techregime_request_map(selected_target_columns)
    sqlite_path = resolve_techregime_sqlite_path(techregime_path) if prefer_sqlite else None

    if sqlite_path is not None:
        for idx, record in records.iterrows():
            if well_col not in record or pd.isna(record[well_col]):
                continue
            record_interval = _resolve_record_interval(record, interval)

            try:
                history_rows = _slice_history_lookback(
                    _load_sqlite_well_history(
                        sqlite_path,
                        record[well_col],
                        record.get(failure_date_col),
                        interval=record_interval,
                    ),
                    record.get(failure_date_col),
                    interval=record_interval,
                )
            except Exception:
                continue
            if history_rows.empty:
                continue

            available_request_map = _filter_available_request_map(request_map, history_rows.columns)
            requested_columns = list(available_request_map.values())
            extract_result = None
            if requested_columns:
                extract_result = extract_techregime_values_from_frame(
                    history_rows,
                    well_id=record[well_col],
                    query_date=record.get(failure_date_col),
                    normalized_columns=requested_columns,
                    interval=record_interval,
                    source_kind="sqlite_history",
                )
            if extract_result is not None and extract_result.matched_rows > 0:
                for target_col, column_name in available_request_map.items():
                    if target_col == "__field__" or target_col in TECHREGIME_ZERO_SAFE_TARGET_COLUMNS:
                        continue
                    value = extract_result.values.get(_normalized_techregime_request_name(column_name))
                    _assign_techregime_value(records, idx, target_col, value)

            field_value = _latest_non_empty_history_value(
                history_rows,
                TECHREGIME_FIELD_COLUMN,
                TECHREGIME_SQLITE_FIELD_FALLBACK,
            )
            _assign_techregime_field(records, idx, field_value)

            if "techregime_match_confidence" not in records.columns:
                records["techregime_match_confidence"] = None
            records.loc[idx, "techregime_match_confidence"] = 0.8
            _apply_zero_safe_history_values(
                records,
                idx,
                history_rows,
                failure_date=record.get(failure_date_col),
                interval=record_interval,
                selected_columns=selected_target_columns,
            )
            _apply_techregime_consistency_checks(records, idx, history_rows=history_rows)
            _apply_techregime_derived_values(records, idx, selected_columns=selected_target_columns)
        return records

    flat_techregime = _prepare_flat_techregime_frame(techregime_path)
    if not flat_techregime.empty:
        flat_techregime = _ensure_flat_helper_columns(flat_techregime)
        history_index = _build_flat_well_history_index(flat_techregime)
        available_request_map = _filter_available_request_map(request_map, flat_techregime.columns)
        requested_columns = list(available_request_map.values())
        for idx, record in records.iterrows():
            if well_col not in record or pd.isna(record[well_col]):
                continue
            record_interval = _resolve_record_interval(record, interval)
            history_rows = _get_flat_well_history_rows(
                flat_techregime,
                history_index,
                record[well_col],
                record.get(failure_date_col),
                interval=record_interval,
            )
            if history_rows.empty:
                continue

            extract_result = None
            if requested_columns:
                extract_result = extract_techregime_values_from_frame(
                    history_rows,
                    well_id=record[well_col],
                    query_date=record.get(failure_date_col),
                    normalized_columns=requested_columns,
                    interval=record_interval,
                    source_kind="excel",
                )
            if extract_result is not None and extract_result.matched_rows > 0:
                for target_col, column_name in available_request_map.items():
                    if target_col == "__field__" or target_col in TECHREGIME_ZERO_SAFE_TARGET_COLUMNS:
                        continue
                    value = extract_result.values.get(_normalized_techregime_request_name(column_name))
                    _assign_techregime_value(records, idx, target_col, value)

            field_value = _latest_non_empty_history_value(history_rows, TECHREGIME_FIELD_COLUMN)
            _assign_techregime_field(records, idx, field_value)

            if "techregime_match_confidence" not in records.columns:
                records["techregime_match_confidence"] = None
            records.loc[idx, "techregime_match_confidence"] = 0.78
            _apply_zero_safe_history_values(
                records,
                idx,
                history_rows,
                failure_date=record.get(failure_date_col),
                interval=record_interval,
                selected_columns=selected_target_columns,
            )
            _apply_techregime_consistency_checks(records, idx, history_rows=history_rows)
            _apply_techregime_derived_values(records, idx, selected_columns=selected_target_columns)
        return records

    for idx, record in records.iterrows():
        if well_col not in record or pd.isna(record[well_col]):
            continue
        candidate_files = list_techregime_candidate_files(techregime_path, record.get(failure_date_col))
        if not candidate_files:
            continue

        matching_rows_by_file: List[pd.DataFrame] = []
        for file_path in candidate_files:
            if file_path not in file_cache:
                file_cache[file_path] = load_techregime_dataframe(file_path)
            tech_df = file_cache[file_path]
            if tech_df is None:
                continue
            matching_rows = find_techregime_well_rows(tech_df, record[well_col])
            if not matching_rows.empty:
                matching_rows_by_file.append(matching_rows)
        if not matching_rows_by_file:
            continue

        for target_col in selected_target_columns:
            # ⚠ На legacy-ветке (месячные папки Excel, sqlite нет) zero-safe
            # колонки НЕ пропускаются. «Zero-safe» означает «усреднить по окну
            # истории, отбросив нули простоя», но у legacy-кадра нет ни даты, ни
            # истории — усреднять не по чему, и `_apply_zero_safe_history_values`
            # молча ничего не пишет. Пропуск здесь оставлял пустыми 15 из 16
            # колонок ТехРежима: весь режим, кроме «Тип ствола скв». Правило
            # «первое непустое по файлам от свежего к старому» здесь и есть
            # верное. Дефект не проявлялся в проде только потому, что рядом с
            # выгрузкой лежит techregime.sqlite и ветка не выбиралась.
            if _normalized_techregime_request_name(target_col) in {"месторождение", "field", "гжф"}:
                continue
            normalized_source = _legacy_request_tuple_from_matching_rows(target_col, matching_rows_by_file)
            if normalized_source is None:
                continue
            value = get_first_non_empty_value_from_files(matching_rows_by_file, normalized_source)
            _assign_techregime_value(records, idx, target_col, value)

        if _selected_includes_column(selected_target_columns, "Месторождение"):
            field_value = get_first_non_empty_value_from_files(
                matching_rows_by_file,
                ("", normalize_techregime_header(TECHREGIME_FIELD_COLUMN)),
            )
            _assign_techregime_field(records, idx, field_value)

        if "techregime_match_confidence" not in records.columns:
            records["techregime_match_confidence"] = None
        records.loc[idx, "techregime_match_confidence"] = 0.75
        _apply_techregime_consistency_checks(
            records,
            idx,
            history_rows=pd.concat(matching_rows_by_file, ignore_index=False, sort=False),
        )
        _apply_techregime_derived_values(records, idx, selected_columns=selected_target_columns)
    return records


__all__ = [
    "detect_techregime_header_rows",
    "flatten_techregime_column",
    "parse_techregime_file_date",
    "find_techregime_month_folder",
    "list_techregime_candidate_files",
    "load_techregime_dataframe",
    "find_techregime_well_rows",
    "discover_available_techregime_columns",
    "enrich_from_techregime",
    "get_default_techregime_selected_columns",
    "refresh_records_from_techregime",
]
