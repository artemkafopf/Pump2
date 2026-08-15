"""Shared derivations for the Свод build, plus the telemetry enricher.

⚠ Порядок наведён при переносе. В ``db_builder`` этот модуль нёс ВТОРУЮ, мёртвую
копию пяти живых процессоров — ``extract_new_failures_from_pdk``,
``enrich_from_big``, ``enrich_from_techregime``, ``enrich_from_opz``,
``enrich_from_lab`` — и копии эти разошлись с живыми. Мёртвый
``extract_new_failures_from_pdk`` фильтровал ГТМ/ППР/Прочие ДО схлопывания
дублей и не знал ни про схлопывание одного пуска, ни про починку well_id, ни про
свободный текст; мёртвый ``enrich_from_big`` не умел сопоставлять по дате
монтажа. Импортируй кто-нибудь из ``enrich`` вместо процессора — получил бы
прошлогоднюю логику и не заметил. Дубликаты удалены; здесь остались только
предикаты, которые процессоры действительно делят, и телеметрия.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import PDK_CLASSIFIER_ALIASES
from ..io import discover_source_files, load_excel_file
from ..normalize import normalize_text, normalize_well, parse_date, parse_number
from ..telemetry import load_telemetry_sqlite_frame, resolve_telemetry_sqlite_path


# --------------------------------------------------------------------------- #
# Identity / classification predicates
# --------------------------------------------------------------------------- #

def derive_field_code_from_well(well_value) -> Optional[str]:
    """Derive field code from all leading letters immediately before an underscore."""
    if pd.isna(well_value) or well_value is None:
        return None

    match = re.search(r"([A-Za-zА-Яа-яЁё]+)\s*_", str(well_value))
    if not match:
        return None

    return match.group(1)


def normalize_contractor(value):
    """Collapse contractor / ownership variants to a canonical short name.

    The Big register stores ownership as long corporate strings such as
    ``Новые-технологии ООО «ИНК»``, ``Новые-технологии ООО «ИНК-НГГ»`` and
    ``Новые-технологии АО «ИНК-Запад»`` -- all the same operator. They are folded
    to a single canonical ``Новые технологии``.
    """
    if pd.isna(value) or value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    normalized = normalize_text(text)
    if "новые" in normalized and "технолог" in normalized:
        return "Новые технологии"
    return text


def derive_acid_type(field_value) -> Optional[str]:
    """Derive acid/non-acid label from current field text."""
    field_text = normalize_text(field_value)
    if not field_text:
        return "Некислый"
    if "некисл" in field_text:
        return "Некислый"
    if "кисл" in field_text:
        return "Кислый"
    return "Некислый"


def is_gtm_marker(value) -> bool:
    """True when a source cell contains the GTM marker."""
    return normalize_text(value) == "гтм"


def is_non_failure_reason_marker(value) -> bool:
    """True when a stop reason represents a non-failure operational event."""
    return normalize_text(value) in {"гтм", "ппр", "прочие"}


def is_missing_failed_component(value) -> bool:
    """True when failed-node / failed-element is effectively empty."""
    normalized = normalize_text(value)
    return normalized in {"", "нет", "-", "н/д", "nan", "none"}


def is_kdmnu_marker(value) -> bool:
    """True when a source cell identifies a KDMNU record."""
    return "кдмну" in normalize_text(value)


def is_mech_production_purpose(value) -> bool:
    """True when an artificial-lift run purpose is mechanized (ESP) oil production.

    This mirrors the PDK oil-well (`Тип скважины == НФ`) filter for the
    artificial-lift running-well path: only `Цель спуска == Мех. добыча` runs
    are ESP oil production. Injection (`Нагнетательная`), water intake
    (`водозаборная`), flowing (`Фонтанная`), piezometric, gaslift and other
    purposes are not oil-production runs and must be excluded.
    """
    normalized = normalize_text(value).replace(".", "").replace(" ", "")
    return normalized.startswith("мехдобыч")


def run_has_esp_gno(value) -> bool:
    """True when a ``Тип ГНО`` value denotes a downhole pump (ESP / ВНН screw).

    ``ЭЦН…``/``ВНН…`` are pumps (artificial lift); ``Воронка`` (tubing funnel),
    ``УГРП`` (frac string) and blanks are not. Used to rescue the rare flowing
    (``Фонтанная``) completion that still carries an ESP.
    """
    normalized = normalize_text(value).replace(".", "").replace(" ", "")
    return "эцн" in normalized or normalized.startswith("внн")


def is_esp_oil_run(purpose, gno_type=None) -> bool:
    """True for an artificial-lift oil run.

    ``Цель спуска == Мех. добыча`` always qualifies (regardless of the ГНО type —
    most such rows carry no ESP marker in ``Тип ГНО``). A ``Фонтанная`` (flowing)
    completion qualifies only when it carries an ESP in ``Тип ГНО`` — a small but
    real set of flowing wells that still run a pump.
    """
    if is_mech_production_purpose(purpose):
        return True
    normalized = normalize_text(purpose).replace(".", "").replace(" ", "")
    return normalized.startswith("фонтан") and run_has_esp_gno(gno_type)


def derive_runtime_group(runtime_value) -> Optional[str]:
    """Map runtime in days to the target workbook's runtime-group buckets."""
    runtime = parse_number(runtime_value)
    if runtime is None:
        return None
    if runtime <= 3:
        return "0-3 суток"
    if runtime <= 30:
        return "4-30 суток"
    if runtime <= 180:
        return "31-180 сут"
    if runtime <= 365:
        return "181-365 сут"
    if runtime <= 765:
        return "366-765 сут"
    return "больше 766 сут"


def normalize_oil_well_classifier(value) -> str:
    """Normalize PDK well-purpose markers to a coarse oil-well intent."""
    text = normalize_text(value)
    text = text.replace(".", "").replace(",", "")
    if text in {"нф", "неф", "нефть", "нефт", "нефтяная", "нефтяные"}:
        return "oil"
    if "неф" in text or text.startswith("нф"):
        return "oil"
    return text


def is_oil_well_row(row: pd.Series) -> bool:
    """True when a PDK row belongs to an oil well by any known classifier field."""
    for source_col in PDK_CLASSIFIER_ALIASES["well_type_class"]:
        if source_col not in row.index:
            continue
        normalized = normalize_oil_well_classifier(row.get(source_col))
        if normalized == "oil":
            return True
    return False


def combine_dataset_frames(frames: List[Tuple[Path, pd.DataFrame, str, int]]) -> pd.DataFrame:
    """Concatenate loaded dataset frames and preserve source lineage."""
    normalized_frames = []
    for file_path, df, sheet_name, header_row in frames:
        frame = df.copy()
        frame["_source_file"] = file_path.name
        frame["_source_path"] = str(file_path)
        frame["_source_sheet"] = sheet_name
        frame["_source_header_row"] = header_row
        normalized_frames.append(frame)
    if not normalized_frames:
        return pd.DataFrame()
    return pd.concat(normalized_frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------- #
# Multi-row header flattening (Big / TechRegime workbooks)
# --------------------------------------------------------------------------- #

def normalize_techregime_header(value) -> str:
    """Normalize techregime header text while ignoring unit rows."""
    text = normalize_text(value)
    if text.startswith("unnamed:"):
        return ""
    return text


def build_techregime_columns_from_header_rows(header_frame: pd.DataFrame) -> List[Tuple[str, str]]:
    """Build normalized `(group, field)` tuples from raw techregime header rows."""
    if header_frame.empty:
        return []

    filled = header_frame.copy().ffill(axis=1)
    columns: List[Tuple[str, str]] = []

    for col_idx in range(filled.shape[1]):
        group = normalize_techregime_header(filled.iloc[0, col_idx])
        field = normalize_techregime_header(filled.iloc[1, col_idx])
        if not field:
            field = normalize_techregime_header(filled.iloc[2, col_idx])
        columns.append((group, field))

    return columns


def normalize_big_header(value) -> str:
    """Normalize Big workbook header text while ignoring unnamed placeholders."""
    text = normalize_text(value)
    if text.startswith("unnamed:"):
        return ""
    return text


def build_big_columns_from_header_rows(header_frame: pd.DataFrame) -> List[Tuple[str, str]]:
    """Build normalized `(section, field)` tuples from raw Big workbook header rows."""
    if header_frame.empty:
        return []

    filled = header_frame.copy().ffill(axis=1)
    columns: List[Tuple[str, str]] = []

    for col_idx in range(filled.shape[1]):
        section = normalize_big_header(filled.iloc[0, col_idx])
        field = normalize_big_header(filled.iloc[1, col_idx])
        if not field:
            field = section
            section = ""
        columns.append((section, field))

    return columns


def get_first_matching_column(df: pd.DataFrame, column_key):
    """Return the first matching column as a Series even if headers are duplicated."""
    selected = df[column_key]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[:, 0]
    return selected


def get_first_matching_value(row: pd.Series, column_key):
    """Return the first matching scalar value from a row with possible duplicate headers."""
    value = row[column_key]
    if isinstance(value, pd.Series):
        for item in value.tolist():
            if pd.notna(item):
                return item
        return value.iloc[0] if not value.empty else None
    return value


def get_first_non_empty_value_from_files(
    matching_rows_by_file: List[pd.DataFrame],
    source_column: Tuple[str, str],
):
    """Return the first non-empty value across files ordered newest to oldest."""
    for rows in matching_rows_by_file:
        if source_column not in rows.columns:
            continue
        selected = rows[source_column]
        if isinstance(selected, pd.DataFrame):
            candidate_values = selected.to_numpy().flatten().tolist()
        else:
            candidate_values = selected.tolist()
        for value in candidate_values:
            if pd.isna(value) or value in ["", "-", "–"]:
                continue
            return value
    return None


# --------------------------------------------------------------------------- #
# Telemetry enrichment
# --------------------------------------------------------------------------- #

#: A runtime beyond this many days cannot be a real ESP run and would overflow
#: the window arithmetic; such rows fall back to the last reading instead.
_MAX_PLAUSIBLE_RUNTIME_DAYS = 40_000


def enrich_from_telemetry(
    records: pd.DataFrame,
    telemetry_dir,
    field_col: str = "field",
    well_col: str = "well",
    failure_date_col: str = "failure_date",
    *,
    sqlite_path=None,
) -> pd.DataFrame:
    """Average per-run operating values out of telemetry onto each register row.

    Prefers the prebuilt telemetry SQLite store (one indexed pass over the whole
    daily table) and falls back to reading the per-field Excel exports.

    ⚠ ``sqlite_path`` is explicit. The store does **not** live inside the export
    drop folder — the drop is one dated sub-folder per export, the store sits
    beside them — so resolving it from ``telemetry_dir`` silently returned
    ``None`` and demoted every build to the slow per-file Excel path.
    """
    records = records.copy()
    telemetry_cache: Dict[str, pd.DataFrame] = {}
    runtime_columns = ("runtime_nno", "Наработка (сут)", "I. Наработка (сут)")

    def is_missing_value(value) -> bool:
        if value is None or pd.isna(value):
            return True
        if isinstance(value, str):
            return normalize_text(value) in {"", "nan", "none", "-", "–"}
        return False

    def resolve_runtime_days(record: pd.Series) -> Optional[float]:
        for column_name in runtime_columns:
            runtime_days = parse_number(record.get(column_name))
            if runtime_days is not None and 0 < runtime_days <= _MAX_PLAUSIBLE_RUNTIME_DAYS:
                return float(runtime_days)
        return None

    def average_positive_numeric(series: pd.Series) -> Optional[float]:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric[numeric.notna() & (numeric > 0)]
        if valid.empty:
            return None
        return float(valid.mean())

    def should_backfill(row_idx, alias_col: str, target_col: Optional[str] = None) -> bool:
        alias_value = records.loc[row_idx, alias_col] if alias_col in records.columns else None
        if not is_missing_value(alias_value):
            return False
        if target_col and target_col in records.columns:
            target_value = records.loc[row_idx, target_col]
            if not is_missing_value(target_value):
                return False
        return True

    telemetry_files = discover_source_files(telemetry_dir, "telemetry")
    if sqlite_path is None:
        sqlite_path = resolve_telemetry_sqlite_path(telemetry_dir)
    telemetry_daily_df = None
    telemetry_by_well: Dict[str, pd.DataFrame] = {}
    if sqlite_path is not None and Path(sqlite_path).exists():
        try:
            telemetry_daily_df = load_telemetry_sqlite_frame(sqlite_path, daily=True)
        except Exception as exc:
            print(f"  Telemetry store {sqlite_path} unreadable ({exc}); falling back to exports")
            telemetry_daily_df = None
        # Normalize wells and index by well once, instead of scanning the whole
        # (multi-million row) telemetry table per record. `well` normalization is
        # the single most expensive operation here, so it must run only once.
        if telemetry_daily_df is not None and not telemetry_daily_df.empty:
            telemetry_daily_df = telemetry_daily_df.copy()
            telemetry_daily_df["_norm_well"] = telemetry_daily_df["well"].map(normalize_well)
            telemetry_daily_df["tel_date"] = pd.to_datetime(telemetry_daily_df["date"], errors="coerce")
            telemetry_by_well = {
                str(well_key): group
                for well_key, group in telemetry_daily_df.groupby("_norm_well", sort=False)
            }
            print(f"  Telemetry store: {len(telemetry_daily_df)} daily row(s) over {len(telemetry_by_well)} well(s)")
        else:
            telemetry_daily_df = None

    def resolve_telemetry_file(field_code: str) -> Optional[Path]:
        if not telemetry_files:
            return None
        if len(telemetry_files) == 1 and telemetry_files[0].is_file():
            return telemetry_files[0]
        field_norm = normalize_text(field_code)
        candidates = []
        for file_path in telemetry_files:
            name_norm = normalize_text(file_path.stem)
            if f"_{field_norm}" in name_norm or name_norm.endswith(field_norm) or field_norm in name_norm:
                candidates.append(file_path)
        return sorted(candidates)[0] if candidates else None

    def load_telemetry_df(field_code: str) -> Optional[pd.DataFrame]:
        if field_code in telemetry_cache:
            return telemetry_cache[field_code]
        file_path = resolve_telemetry_file(field_code)
        if file_path is None:
            telemetry_cache[field_code] = None
            return None
        try:
            tel_df, _, _ = load_excel_file(str(file_path))
        except Exception:
            tel_df = None
        telemetry_cache[field_code] = tel_df
        return tel_df

    telemetry_rules = {
        "frequency": (("Частота вращения двиг., Гц", "frequency_hz"), "Частота"),
        "motor_load": (("Загрузка, %", "motor_load_percent"), "Загр, Двиг,"),
        "intake_pressure": (("Р на приеме насоса, атм", "P_intake_atm"), None),
    }

    for idx, record in records.iterrows():
        well_norm = normalize_well(record.get(well_col))
        fail_date = parse_date(record.get(failure_date_col))
        runtime_days = resolve_runtime_days(record)

        matching_tel = None
        telemetry_window = None
        best_tel = None
        use_sqlite_daily = telemetry_daily_df is not None

        if use_sqlite_daily:
            well_history = telemetry_by_well.get(well_norm)
            if well_history is None or well_history.empty:
                continue
            matching_tel = well_history
            if fail_date is not None:
                matching_tel = matching_tel[matching_tel["tel_date"] <= fail_date]
            if matching_tel.empty:
                continue
            matching_tel = matching_tel.copy()
        else:
            field = record.get(field_col)
            if not field:
                continue
            tel_df = load_telemetry_df(str(field))
            if tel_df is None:
                continue

            tel_col = next((c for c in tel_df.columns if normalize_text(c) == "скважина"), None)
            date_col = next((c for c in tel_df.columns if normalize_text(c) == "дата"), None)
            if tel_col is None or date_col is None:
                continue

            matching_tel = tel_df[tel_df[tel_col].apply(lambda x: normalize_well(x) == well_norm)]
            if matching_tel.empty:
                continue

            matching_tel = matching_tel.copy()
            matching_tel["tel_date"] = matching_tel[date_col].apply(parse_date)
            matching_tel = matching_tel[matching_tel["tel_date"].notna()]
            if fail_date is not None:
                matching_tel = matching_tel[matching_tel["tel_date"] <= fail_date]
            if matching_tel.empty:
                continue

        if runtime_days is not None and fail_date is not None:
            window_start = fail_date.normalize() - pd.Timedelta(days=runtime_days)
            telemetry_window = matching_tel[
                (matching_tel["tel_date"].dt.normalize() >= window_start)
                & (matching_tel["tel_date"].dt.normalize() <= fail_date.normalize())
            ].copy()
        else:
            telemetry_window = matching_tel.copy()
        best_tel = matching_tel.loc[matching_tel["tel_date"].idxmax()]

        for alias_col, (source_columns, target_col) in telemetry_rules.items():
            source_col = next(
                (column_name for column_name in source_columns if column_name in matching_tel.columns),
                None,
            )
            if source_col is None:
                continue
            if not should_backfill(idx, alias_col, target_col):
                continue
            if alias_col not in records.columns:
                records[alias_col] = None
            if target_col and target_col not in records.columns:
                records[target_col] = None

            if runtime_days is not None and fail_date is not None:
                value = average_positive_numeric(telemetry_window[source_col])
            else:
                value = parse_number(best_tel[source_col])
                if value is None and not is_missing_value(best_tel[source_col]):
                    value = best_tel[source_col]

            if value is None:
                continue

            records.loc[idx, alias_col] = value
            if target_col and is_missing_value(records.loc[idx, target_col]):
                records.loc[idx, target_col] = value

    return records


__all__ = [
    "build_big_columns_from_header_rows",
    "build_techregime_columns_from_header_rows",
    "combine_dataset_frames",
    "derive_acid_type",
    "derive_field_code_from_well",
    "derive_runtime_group",
    "enrich_from_telemetry",
    "get_first_matching_column",
    "get_first_matching_value",
    "get_first_non_empty_value_from_files",
    "is_esp_oil_run",
    "is_gtm_marker",
    "is_kdmnu_marker",
    "is_mech_production_purpose",
    "is_missing_failed_component",
    "is_non_failure_reason_marker",
    "is_oil_well_row",
    "normalize_big_header",
    "normalize_contractor",
    "normalize_oil_well_classifier",
    "normalize_techregime_header",
    "run_has_esp_gno",
]
