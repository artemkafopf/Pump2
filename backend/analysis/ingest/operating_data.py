"""Telemetry-first operating-data processing for ESP analysis datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .normalize import (
    normalize_text,
    normalize_well,
    parse_date,
    parse_date_series,
    parse_number,
)
from .telemetry import load_telemetry_sqlite_frame, resolve_telemetry_sqlite_path
from .techregime import load_techregime_export


NUMERIC_OPERATING_COLUMNS = (
    "Qliq_m3d",
    "Qgas_m3d",
    "watercut_percent",
    "GLF_m3m3",
    "P_bhp_atm",
    "P_intake_atm",
    # ⚠ Рпл (пластовое) и Рнас (насыщения) — РАЗНЫЕ величины. В db_builder «рпл»
    # стоял в алиасах ``Pbubble_atm``, и на источнике ТехРежима (где Рпл есть, а
    # Рнас нет) пластовое давление уезжало под именем давления насыщения — вместе
    # со всеми производными от него ``pressure_ratio_*``. На телеметрии дефект был
    # латентным: там нет ни того ни другого, ``Pbubble_atm`` пуст во всех 1 319 833
    # суточных строках. Разведено.
    "Pbubble_atm",
    "P_reservoir_atm",
    "gas_factor_m3t",
    "frequency_hz",
    "current_a",
    "voltage_v",
    "motor_load_percent",
    "pump_nominal_rate_m3d",
    "reference_frequency_hz",
    "motor_power_kw",
)
DERIVED_COLUMNS = (
    "Kpod",
    "Kpod_freq",
    "pressure_ratio_bhp",
    "pressure_ratio_intake",
    "qliq_per_kw",
    "motor_load_per_hz",
)
TABULAR_SUFFIXES = {".xls", ".xlsx", ".xlsm", ".csv", ".txt"}
WINDOWS = ("full_run", "30d", "prev30d")
AGGREGATION_STATS = ("mean", "median", "min", "max", "std", "range", "last")
THRESHOLD_FEATURES = (
    ("lt_040", lambda series: series < 0.40),
    ("lt_055", lambda series: series < 0.55),
    ("lt_070", lambda series: series < 0.70),
    ("gt_085", lambda series: series > 0.85),
    ("gt_100", lambda series: series > 1.00),
)

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "well": ("well", "скважина", "скв", "скв.", "id скважины", "№ скважины", "номер скважины"),
    "run_id": ("run id", "run_id", "ид запуска", "ид пуска", "id run", "pump run id"),
    "field": ("field", "месторождение", "м/р", "принадлежность", "well field"),
    "contractor": ("contractor", "подрядчик", "собственник оборудования", "vendor"),
    "date": ("date", "дата", "datetime", "дата замера", "дата отчета"),
    "install_date": ("install date", "дата монтажа", "монтаж", "installation date"),
    "start_date": ("start date", "дата запуска", "launch date", "дата старта"),
    "stop_date": ("stop date", "дата остановки", "дата демонтажа", "shutdown date"),
    "failure_date": ("failure date", "дата отказа", "отказ дата", "дата аварии"),
    "TTF_days": ("ttf", "ttf days", "наработка (сут)", "наработка, сут", "нно", "runtime days"),
    "event": ("event", "event flag", "failure flag", "флаг отказа", "событие", "признак отказа"),
    # ⚠⚠ Первым стоит ПОЛНОЕ имя колонки телеметрии. Выгрузка от 2026-08-15
    # добавила блок ОЗНА (групповая замерная установка), и «Дебит жидкости
    # (ОЗНА), м3/сут» стал вторым кандидатом на подстроку «дебитжидкости».
    # Резолвер принимает только ЕДИНСТВЕННОЕ частичное совпадение, поэтому ничья
    # оставила ``Qliq_m3d`` пустым во всех 1 414 192 строках — при том, что в
    # июньской сборке он был заполнен на 1 168 492. Точный алиас снимает ничью в
    # пользу ШАХ (товарная шахматка), а не прибора ОЗНА.
    "Qliq_m3d": (
        "дебит жидкости (шах), м3/сут",
        "дебит жидк.", "дебит жидкости", "qliq", "liquid rate", "дебит ж", "дебит жидкости м3/сут",
    ),
    "Qgas_m3d": ("дебит газа", "qgas", "gas rate", "газ м3/сут"),
    "watercut_percent": ("обводненность", "watercut", "water cut", "обв %", "обводненность %"),
    "GLF_m3m3": ("гжф", "glf", "газожидкостный фактор", "gas liquid factor"),
    # ⚠⚠ ГФ (м³/т, газ на ТОННУ НЕФТИ) и ГЖФ (м³/м³, газ на КУБ ЖИДКОСТИ) — РАЗНЫЕ
    # величины, отличающиеся тем сильнее, чем выше обводнённость. Канона под ГФ не
    # было вовсе, поэтому модельная колонка `gas_factor` брала из телеметрии ГЖФ, а
    # из техрежима ГФ — и склеивала их в одну. На пересечении это дало r = 0.236 при
    # НУЛЕ строк, совпадающих в пределах 1 %, и систематическом сдвиге уровня в
    # 2.5–3 раза (медианы 75–105 против 160–277). Точный алиас снимает и ничью с
    # блоком ОЗНА.
    "gas_factor_m3t": (
        "газовый фактор, м3/т",
        "газовый фактор",
        "гф",
        "gas oil ratio",
        "gor",
    ),
    "P_bhp_atm": ("рзаб", "p_bhp", "bhp", "bottomhole pressure", "забойное давление"),
    # ⚠ "дав. нас" is the intake-pressure header ("давление на насосе"), not
    # "давление насыщения" — the two collide once normalized, so bubble point is
    # matched only by its unambiguous spellings.
    "P_intake_atm": ("дав. нас", "давление насоса", "p intake", "intake pressure", "р на приеме насоса"),
    "Pbubble_atm": ("рнас", "рнас.", "pbubble", "bubble point pressure", "давление насыщения"),
    "P_reservoir_atm": ("рпл", "рпл.", "p reservoir", "reservoir pressure", "пластовое давление"),
    "frequency_hz": ("частота", "частота вращения двиг., гц", "frequency", "freq", "hz"),
    "current_a": ("ток", "current", "ток а", "current a"),
    "voltage_v": ("напряжение", "voltage", "voltage v", "u"),
    "motor_load_percent": ("загрузка, %", "загрузка", "загр, двиг,", "motor load", "load percent"),
    "pump_nominal_rate_m3d": ("ном. произв. м₃/сут", "ном. произв. м3/сут", "номинальная производительность", "pump nominal rate"),
    "reference_frequency_hz": ("базовая частота", "опорная частота", "reference frequency", "50 гц", "50hz"),
    "motor_power_kw": ("мощность двигателя", "motor power", "power kw", "квт"),
}


@dataclass(frozen=True)
class OperatingDataResult:
    """Telemetry-first operating-data output for run-level analytics."""

    features: pd.DataFrame
    telemetry_daily: pd.DataFrame
    techregime_daily: pd.DataFrame
    unified_daily: pd.DataFrame
    source_column_maps: dict[str, dict[str, str]]
    unsafe_feature_columns: tuple[str, ...]


def normalize_alias_token(value: Any) -> str:
    """Normalize alias tokens for case/punctuation-insensitive matching."""
    text = normalize_text(value)
    return re.sub(r"[^0-9a-zа-я]+", "", text, flags=re.IGNORECASE)


def _candidate_tokens(values: Iterable[Any]) -> list[str]:
    return [token for token in (normalize_alias_token(value) for value in values) if token]


def resolve_canonical_columns(
    frame: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Resolve dataframe columns to canonical names via exact-then-safe-partial alias matching."""
    alias_map = aliases or CANONICAL_ALIASES
    normalized_columns = {
        str(column): normalize_alias_token(column)
        for column in frame.columns
    }
    resolved: dict[str, str] = {}
    used_columns: set[str] = set()

    for canonical_name, alias_values in alias_map.items():
        alias_tokens = _candidate_tokens((canonical_name, *alias_values))
        partial_tokens = _candidate_tokens(alias_values)
        exact_matches = [
            column_name
            for column_name, normalized_column in normalized_columns.items()
            if column_name not in used_columns and normalized_column in alias_tokens
        ]
        if len(exact_matches) == 1:
            resolved[canonical_name] = exact_matches[0]
            used_columns.add(exact_matches[0])
            continue

        partial_matches = []
        for column_name, normalized_column in normalized_columns.items():
            if column_name in used_columns:
                continue
            if not normalized_column or len(normalized_column) < 4:
                continue
            if any(
                alias_token and len(alias_token) >= 5
                and alias_token in normalized_column
                for alias_token in partial_tokens
            ):
                partial_matches.append(column_name)
        if len(partial_matches) == 1:
            resolved[canonical_name] = partial_matches[0]
            used_columns.add(partial_matches[0])

    return resolved


def ambiguous_canonical_columns(
    frame: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Canonical names that a source left unresolved *because of a tie*.

    ⚠⚠ Silence is the hazard here, not the tie. :func:`resolve_canonical_columns`
    accepts a partial match only when it is unique — sound, but a source that
    grows a second similarly-named column then drops the metric with no error at
    all. That is exactly how the 2026-08-15 telemetry drop emptied ``Qliq_m3d``:
    an added ОЗНА block gave «дебитжидкости» two candidates, and the store built
    "successfully" with the liquid rate gone. Reporting the tie makes the next
    such export loud instead of lossy.
    """
    alias_map = aliases or CANONICAL_ALIASES
    resolved = resolve_canonical_columns(frame, aliases=alias_map)
    normalized_columns = {str(column): normalize_alias_token(column) for column in frame.columns}

    ties: dict[str, list[str]] = {}
    for canonical_name, alias_values in alias_map.items():
        if canonical_name in resolved:
            continue
        partial_tokens = _candidate_tokens(alias_values)
        candidates = [
            column_name
            for column_name, normalized_column in normalized_columns.items()
            if normalized_column
            and len(normalized_column) >= 4
            and any(
                token and len(token) >= 5 and token in normalized_column
                for token in partial_tokens
            )
        ]
        if len(candidates) > 1:
            ties[canonical_name] = sorted(candidates)
    return ties


def _coerce_dates(series: pd.Series) -> pd.Series:
    """Parse a whole date column, vectorized where that is safe.

    ⚠ A bare number in a date column is an **Excel serial**, and only the
    per-cell :func:`parse_date` reads it as one — it bounds-checks the value to a
    plausible date range first. ``pd.to_datetime`` instead reads the integer as
    nanoseconds since the epoch and silently returns 1970. Numeric cells are
    therefore routed to ``parse_date``; the textual majority is parsed in one
    vectorized pass, which on the 1.3 M-row telemetry drop replaces 1.3 M Python
    calls.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    present = series.notna()
    numeric_like = pd.to_numeric(series, errors="coerce").notna() & present

    if numeric_like.any():
        result.loc[numeric_like] = pd.to_datetime(
            series[numeric_like].apply(parse_date), errors="coerce"
        )
    textual = present & ~numeric_like
    if textual.any():
        result.loc[textual] = parse_date_series(series[textual])
    return result


#: Cells that :func:`parse_number` reads as "no value" rather than as text.
_NUMERIC_NULL_TOKENS = {"", "-", "н/д", "нет"}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Parse a whole numeric column, vectorized.

    Matches :func:`parse_number` cell-for-cell — comma decimal separator, spaces
    as thousands separators, ``-``/``н/д``/``нет`` as missing — but does it with
    string ops instead of 1.3 M per-cell calls.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    text = text.mask(text.str.lower().isin(_NUMERIC_NULL_TOKENS))
    text = text.str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    # ``to_numeric`` over a nullable ``string`` column yields nullable ``Float64``;
    # cast back to plain float64 so missing values stay NaN and no downstream
    # arithmetic trips over ``pd.NA``.
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _first_non_empty(values: pd.Series) -> Any:
    for value in values:
        if pd.notna(value) and normalize_text(value) not in {"", "nan", "none", "-", "–"}:
            return value
    return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    valid = denominator_numeric.notna() & numerator_numeric.notna() & (denominator_numeric != 0)
    result = pd.Series(index=numerator.index, dtype=float)
    result.loc[valid] = numerator_numeric.loc[valid] / denominator_numeric.loc[valid]
    return result


def _iter_source_items(source: Any) -> list[Path]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        items = [Path(source)]
    elif isinstance(source, Sequence) and not isinstance(source, pd.DataFrame):
        items = [Path(item) for item in source if isinstance(item, (str, Path))]
    else:
        return []

    discovered: list[Path] = []
    for item in items:
        if item.is_dir():
            for child in sorted(item.iterdir()):
                if child.is_file() and child.suffix.lower() in TABULAR_SUFFIXES and not child.name.startswith("~$"):
                    discovered.append(child)
        elif item.is_file() and item.suffix.lower() in TABULAR_SUFFIXES and not item.name.startswith("~$"):
            discovered.append(item)
    return sorted(set(path.resolve() for path in discovered), key=lambda path: (path.name.lower(), str(path).lower()))


def load_operating_source(source: pd.DataFrame | str | Path | Sequence[str | Path] | None, source_kind: str) -> pd.DataFrame:
    """Load telemetry or techregime data from a dataframe or local file."""
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if source_kind == "telemetry":
        sqlite_path = resolve_telemetry_sqlite_path(source)
        if sqlite_path is not None:
            return load_telemetry_sqlite_frame(sqlite_path, daily=True)
    frames: list[pd.DataFrame] = []
    for source_path in _iter_source_items(source):
        if source_kind == "techregime":
            frames.append(load_techregime_export(source_path))
        elif source_path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
            frames.append(pd.read_excel(source_path))
        elif source_path.suffix.lower() in {".csv", ".txt"}:
            frames.append(pd.read_csv(source_path))
        else:
            raise ValueError(f"Unsupported {source_kind} source: {source_path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def canonicalize_operating_source(
    source: pd.DataFrame | str | Path,
    *,
    source_kind: str,
    target_runs: pd.DataFrame | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    join_preference: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Canonicalize one operating-data source into a daily time series."""
    raw = load_operating_source(source, source_kind=source_kind)
    if raw.empty:
        empty_daily = pd.DataFrame(
            columns=["_join_key", "date", "well", "run_id", "field", "contractor", "event", *NUMERIC_OPERATING_COLUMNS, "source_kind"]
        )
        return empty_daily, {}
    column_map = resolve_canonical_columns(raw, aliases=aliases)
    for canonical_name, candidates in ambiguous_canonical_columns(raw, aliases=aliases).items():
        print(
            f"  ⚠⚠ [{source_kind}] «{canonical_name}» НЕ разрешён: подходят "
            f"{len(candidates)} колонки — {', '.join(candidates)}. Колонка останется "
            "ПУСТОЙ. Добавьте точный алиас в CANONICAL_ALIASES."
        )
    renamed = raw.rename(columns={value: key for key, value in column_map.items()}).copy()

    for column_name in ("well", "run_id", "field", "contractor", "event"):
        if column_name not in renamed.columns:
            renamed[column_name] = None
    if "date" not in renamed.columns:
        raise ValueError(f"{source_kind} source must contain a recognizable date column.")

    renamed["well"] = renamed["well"].apply(normalize_well) if "well" in renamed.columns else ""
    renamed["run_id"] = renamed["run_id"].apply(lambda value: normalize_text(value) or None)
    renamed["date"] = _coerce_dates(renamed["date"]).dt.normalize()

    for column_name in NUMERIC_OPERATING_COLUMNS:
        if column_name not in renamed.columns:
            renamed[column_name] = None
        renamed[column_name] = _coerce_numeric(renamed[column_name])

    renamed = renamed[renamed["date"].notna()].copy()

    join_key = determine_join_key(target_runs, telemetry=renamed if source_kind == "telemetry" else None, techregime=renamed if source_kind == "techregime" else None, join_preference=join_preference)
    renamed["_join_key"] = renamed[join_key].fillna("")
    renamed = renamed[renamed["_join_key"].astype(str).ne("")].copy()

    if target_runs is not None and not target_runs.empty:
        target_keys = set(target_runs[join_key].fillna("").astype(str))
        renamed = renamed[renamed["_join_key"].astype(str).isin(target_keys)].copy()

    aggregations: dict[str, Any] = {
        "_join_key": "first",
        "well": _first_non_empty,
        "run_id": _first_non_empty,
        "field": _first_non_empty,
        "contractor": _first_non_empty,
        "event": _first_non_empty,
    }
    for column_name in NUMERIC_OPERATING_COLUMNS:
        aggregations[column_name] = "mean"

    daily = (
        renamed.sort_values(["_join_key", "date"])
        .groupby(["_join_key", "date"], dropna=False, as_index=False)
        .agg(aggregations)
    )
    daily["source_kind"] = source_kind
    return daily, column_map


def canonicalize_target_runs(
    target_runs: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Canonicalize the target runs table used for feature generation."""
    column_map = resolve_canonical_columns(target_runs, aliases=aliases)
    renamed = target_runs.rename(columns={value: key for key, value in column_map.items()}).copy()

    for column_name in ("well", "run_id", "field", "contractor", "event"):
        if column_name not in renamed.columns:
            renamed[column_name] = None
    for column_name in ("install_date", "start_date", "stop_date", "failure_date"):
        if column_name not in renamed.columns:
            renamed[column_name] = pd.NaT
        renamed[column_name] = _coerce_dates(renamed[column_name])
    if "TTF_days" not in renamed.columns:
        renamed["TTF_days"] = None
    renamed["TTF_days"] = _coerce_numeric(renamed["TTF_days"])
    renamed["well"] = renamed["well"].apply(normalize_well) if "well" in renamed.columns else ""
    renamed["run_id"] = renamed["run_id"].apply(lambda value: normalize_text(value) or None)
    return renamed, column_map


def determine_join_key(
    target_runs: pd.DataFrame | None,
    *,
    telemetry: pd.DataFrame | None = None,
    techregime: pd.DataFrame | None = None,
    join_preference: str | None = None,
) -> str:
    """Use run_id when present on both target and a source; otherwise use well."""
    if join_preference in {"run_id", "well"}:
        return join_preference
    if target_runs is None or target_runs.empty:
        return "well"
    if "run_id" not in target_runs.columns or target_runs["run_id"].dropna().empty:
        return "well"
    for source_frame in (telemetry, techregime):
        if source_frame is not None and "run_id" in source_frame.columns and source_frame["run_id"].dropna().astype(str).str.len().gt(0).any():
            return "run_id"
    return "well"


def build_unified_daily(
    telemetry_daily: pd.DataFrame,
    techregime_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Combine telemetry and techregime daily rows with telemetry-first metric fallback."""
    merge_columns = ["_join_key", "date"]
    telemetry_frame = telemetry_daily.copy().add_prefix("telemetry_")
    techregime_frame = techregime_daily.copy().add_prefix("regime_")
    merged = telemetry_frame.merge(
        techregime_frame,
        how="outer",
        left_on=["telemetry__join_key", "telemetry_date"],
        right_on=["regime__join_key", "regime_date"],
    )

    merged["_join_key"] = merged["telemetry__join_key"].combine_first(merged["regime__join_key"])
    merged["date"] = merged["telemetry_date"].combine_first(merged["regime_date"])

    def merged_column(name: str) -> pd.Series:
        if name in merged.columns:
            return merged[name]
        return pd.Series([None] * len(merged), index=merged.index, dtype=object)

    for column_name in ("well", "run_id", "field", "contractor", "event"):
        telemetry_series = merged_column(f"telemetry_{column_name}")
        regime_series = merged_column(f"regime_{column_name}")
        merged[column_name] = telemetry_series.where(telemetry_series.notna(), regime_series)

    for column_name in NUMERIC_OPERATING_COLUMNS:
        telemetry_column = f"telemetry_{column_name}"
        regime_column = f"regime_{column_name}"
        telemetry_series = merged_column(telemetry_column)
        regime_series = merged_column(regime_column)
        merged[column_name] = telemetry_series.where(telemetry_series.notna(), regime_series)
        merged[f"{column_name}_source"] = telemetry_series.notna().map({True: "telemetry", False: None})
        fallback_mask = merged[f"{column_name}_source"].isna() & regime_series.notna()
        merged.loc[fallback_mask, f"{column_name}_source"] = "techregime"
        merged[f"{column_name}_source"] = merged[f"{column_name}_source"].fillna("missing")

    unified = merged[["_join_key", "date", "well", "run_id", "field", "contractor", "event", *NUMERIC_OPERATING_COLUMNS, *[f"{column}_source" for column in NUMERIC_OPERATING_COLUMNS]]].copy()
    return add_derived_daily_features(unified)


def add_derived_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute derived daily operating features and provenance."""
    result = frame.copy()
    result["Kpod"] = _safe_divide(result["Qliq_m3d"], result["pump_nominal_rate_m3d"])
    result["qliq_per_kw"] = _safe_divide(result["Qliq_m3d"], result["motor_power_kw"])
    result["motor_load_per_hz"] = _safe_divide(result["motor_load_percent"], result["frequency_hz"])
    result["pressure_ratio_bhp"] = _safe_divide(result["P_bhp_atm"], result["Pbubble_atm"])
    result["pressure_ratio_intake"] = _safe_divide(result["P_intake_atm"], result["Pbubble_atm"])

    frequency_non_zero = pd.to_numeric(result["frequency_hz"], errors="coerce")
    reference_frequency = pd.to_numeric(result["reference_frequency_hz"], errors="coerce")
    kpod = pd.to_numeric(result["Kpod"], errors="coerce")
    valid_mask = kpod.notna() & reference_frequency.notna() & frequency_non_zero.notna() & frequency_non_zero.ne(0)
    result["Kpod_freq"] = None
    result.loc[valid_mask, "Kpod_freq"] = (
        kpod.loc[valid_mask] * reference_frequency.loc[valid_mask] / frequency_non_zero.loc[valid_mask]
    )

    derived_sources = {
        "Kpod": ("Qliq_m3d_source", "pump_nominal_rate_m3d_source"),
        "Kpod_freq": ("Kpod_source", "reference_frequency_hz_source", "frequency_hz_source"),
        "pressure_ratio_bhp": ("P_bhp_atm_source", "Pbubble_atm_source"),
        "pressure_ratio_intake": ("P_intake_atm_source", "Pbubble_atm_source"),
        "qliq_per_kw": ("Qliq_m3d_source", "motor_power_kw_source"),
        "motor_load_per_hz": ("motor_load_percent_source", "frequency_hz_source"),
    }
    for derived_name, component_sources in derived_sources.items():
        source_values = []
        for source_column in component_sources:
            if source_column not in result.columns and source_column == "Kpod_source":
                source_values.append(result["Kpod_source"])
            else:
                source_values.append(result[source_column])
        source_frame = pd.concat(source_values, axis=1)
        result[f"{derived_name}_source"] = source_frame.apply(_combine_source_labels, axis=1)

    return result


def _combine_source_labels(values: pd.Series) -> str:
    labels = [normalize_text(value) for value in values if normalize_text(value) not in {"", "missing"}]
    if not labels:
        return "missing"
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]
    return "mixed"


def _window_mask(dates: pd.Series, window_name: str, start_date: pd.Timestamp | None, anchor_date: pd.Timestamp | None) -> pd.Series:
    if anchor_date is None:
        return pd.Series(False, index=dates.index)
    if window_name == "full_run":
        if start_date is None:
            return dates <= anchor_date
        return (dates >= start_date.normalize()) & (dates <= anchor_date.normalize())
    if window_name == "30d":
        return (dates >= anchor_date.normalize() - pd.Timedelta(days=29)) & (dates <= anchor_date.normalize())
    if window_name == "prev30d":
        window_end = anchor_date.normalize() - pd.Timedelta(days=30)
        window_start = anchor_date.normalize() - pd.Timedelta(days=59)
        return (dates >= window_start) & (dates <= window_end)
    raise ValueError(f"Unsupported window: {window_name}")


def _aggregate_numeric_series(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.notna()]
    if numeric.empty:
        return {stat: None for stat in AGGREGATION_STATS}
    maximum = float(numeric.max())
    minimum = float(numeric.min())
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": minimum,
        "max": maximum,
        "std": float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0,
        "range": maximum - minimum,
        "last": float(numeric.iloc[-1]),
    }


def _window_aggregate(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    prefix: str,
    start_date: pd.Timestamp | None,
    anchor_date: pd.Timestamp | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if frame.empty:
        for metric in metrics:
            for window_name in WINDOWS:
                for stat in AGGREGATION_STATS:
                    row[f"{prefix}{metric}_{window_name}_{stat}"] = None
                row[f"{prefix}{metric}_{window_name}_n_days"] = 0
        return row

    dated = frame.sort_values("date").copy()
    for metric in metrics:
        for window_name in WINDOWS:
            window_rows = dated[_window_mask(dated["date"], window_name, start_date, anchor_date)]
            row[f"{prefix}{metric}_{window_name}_n_days"] = int(window_rows[metric].notna().sum()) if metric in window_rows.columns else 0
            stats = _aggregate_numeric_series(window_rows[metric]) if metric in window_rows.columns else {stat: None for stat in AGGREGATION_STATS}
            for stat_name, value in stats.items():
                row[f"{prefix}{metric}_{window_name}_{stat_name}"] = value
    return row


def _add_ratio_features(row: dict[str, Any], metrics: Sequence[str], *, prefix: str = "") -> None:
    for metric in metrics:
        last30_mean = row.get(f"{prefix}{metric}_30d_mean")
        full_run_mean = row.get(f"{prefix}{metric}_full_run_mean")
        previous30_mean = row.get(f"{prefix}{metric}_prev30d_mean")
        row[f"{prefix}{metric}_30d_to_full_run_mean_ratio"] = (
            None if full_run_mean in (None, 0) or pd.isna(full_run_mean) or last30_mean is None else float(last30_mean) / float(full_run_mean)
        )
        row[f"{prefix}{metric}_30d_to_prev30d_mean_ratio"] = (
            None if previous30_mean in (None, 0) or pd.isna(previous30_mean) or last30_mean is None else float(last30_mean) / float(previous30_mean)
        )


def _add_last30_threshold_features(row: dict[str, Any], window_rows: pd.DataFrame) -> None:
    kpod = pd.to_numeric(window_rows.get("Kpod"), errors="coerce") if "Kpod" in window_rows.columns else pd.Series(dtype=float)
    kpod = kpod[kpod.notna()]
    total_days = int(len(kpod))
    row["Kpod_30d_n_days"] = total_days
    for suffix, predicate in THRESHOLD_FEATURES:
        count = int(predicate(kpod).sum()) if total_days else 0
        row[f"Kpod_30d_days_{suffix}"] = count
        row[f"Kpod_30d_share_{suffix}"] = None if total_days == 0 else count / total_days
    row["Kpod_30d_days_lt_070_minus_gt_085"] = row["Kpod_30d_days_lt_070"] - row["Kpod_30d_days_gt_085"]


def _aggregate_feature_sources(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    start_date: pd.Timestamp | None,
    anchor_date: pd.Timestamp | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric in metrics:
        source_column = f"{metric}_source"
        if source_column not in frame.columns:
            continue
        for window_name in WINDOWS:
            window_rows = frame[_window_mask(frame["date"], window_name, start_date, anchor_date)]
            row[f"{metric}_{window_name}_mean_source"] = _combine_source_labels(window_rows[source_column])
            row[f"{metric}_{window_name}_last_source"] = _combine_source_labels(window_rows[source_column].tail(1))
    return row


def _choose_final_metric_values(row: dict[str, Any], metric: str) -> None:
    for window_name in WINDOWS:
        for stat_name in AGGREGATION_STATS:
            telemetry_key = f"telemetry_{metric}_{window_name}_{stat_name}"
            regime_key = f"regime_{metric}_{window_name}_{stat_name}"
            target_key = f"{metric}_{window_name}_{stat_name}"
            source_key = f"{metric}_{window_name}_{stat_name}_source"
            telemetry_value = row.get(telemetry_key)
            regime_value = row.get(regime_key)
            if telemetry_value is not None and not pd.isna(telemetry_value):
                row[target_key] = telemetry_value
                row[source_key] = "telemetry"
            elif regime_value is not None and not pd.isna(regime_value):
                row[target_key] = regime_value
                row[source_key] = "techregime"
            else:
                row[target_key] = None
                row[source_key] = "missing"
        telemetry_days = row.get(f"telemetry_{metric}_{window_name}_n_days", 0)
        regime_days = row.get(f"regime_{metric}_{window_name}_n_days", 0)
        row[f"{metric}_{window_name}_n_days"] = telemetry_days or regime_days or 0


def _anchor_date_for_run(run: pd.Series) -> pd.Timestamp | None:
    for column_name in ("failure_date", "stop_date", "start_date", "install_date"):
        value = run.get(column_name)
        if pd.notna(value):
            return pd.Timestamp(value)
    return None


def _start_date_for_run(run: pd.Series) -> pd.Timestamp | None:
    for column_name in ("start_date", "install_date"):
        value = run.get(column_name)
        if pd.notna(value):
            return pd.Timestamp(value)
    anchor_date = _anchor_date_for_run(run)
    runtime_days = run.get("TTF_days")
    runtime_numeric = parse_number(runtime_days)
    if anchor_date is not None and runtime_numeric is not None:
        return anchor_date - pd.Timedelta(days=float(runtime_numeric))
    return None


def _index_daily_by_join_key(daily: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Group a daily frame by its string join key: ``{join_value: sub_frame}``.

    Returns the group dict and an empty same-schema frame to serve lookups for
    runs with no matching daily rows. Within-group row order is preserved, so the
    result matches the previous ``daily[daily["_join_key"] == v]`` filter.
    """
    empty = daily.iloc[0:0]
    if daily.empty or "_join_key" not in daily.columns:
        return {}, empty
    keys = daily["_join_key"].astype(str)
    return {str(key): sub for key, sub in daily.groupby(keys, sort=False)}, empty


def build_operating_features(
    target_runs: pd.DataFrame,
    *,
    telemetry: pd.DataFrame | str | Path | Sequence[str | Path] | None,
    techregime: pd.DataFrame | str | Path | Sequence[str | Path] | None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    join_preference: str | None = None,
) -> OperatingDataResult:
    """Build telemetry-first operating features for the target runs table."""
    target_frame, target_column_map = canonicalize_target_runs(target_runs, aliases=aliases)

    telemetry_daily, telemetry_column_map = canonicalize_operating_source(
        telemetry,
        source_kind="telemetry",
        target_runs=target_frame,
        aliases=aliases,
        join_preference=join_preference,
    )
    techregime_daily, techregime_column_map = canonicalize_operating_source(
        techregime,
        source_kind="techregime",
        target_runs=target_frame,
        aliases=aliases,
        join_preference=join_preference,
    )

    join_key = determine_join_key(target_frame, telemetry=telemetry_daily, techregime=techregime_daily, join_preference=join_preference)
    unified_daily = build_unified_daily(telemetry_daily, techregime_daily)

    # Index each daily frame by join key ONCE (dict lookup per run) instead of
    # re-filtering the whole frame with `daily["_join_key"] == v` inside the loop
    # (previously O(runs x daily-rows) -- the worst hotspot in the build).
    telemetry_groups, telemetry_empty = _index_daily_by_join_key(telemetry_daily)
    techregime_groups, techregime_empty = _index_daily_by_join_key(techregime_daily)
    unified_groups, unified_empty = _index_daily_by_join_key(unified_daily)

    features_rows: list[dict[str, Any]] = []
    metrics = [*NUMERIC_OPERATING_COLUMNS, *DERIVED_COLUMNS]
    for _, run in target_frame.iterrows():
        join_value = run.get(join_key)
        if pd.isna(join_value) or normalize_text(join_value) == "":
            continue

        join_value = str(join_value)
        start_date = _start_date_for_run(run)
        anchor_date = _anchor_date_for_run(run)

        telemetry_rows = telemetry_groups.get(join_value, telemetry_empty).copy()
        techregime_rows = techregime_groups.get(join_value, techregime_empty).copy()
        unified_rows = unified_groups.get(join_value, unified_empty).copy()

        unified_rows = add_derived_daily_features(unified_rows)
        row = {
            "well": run.get("well"),
            "run_id": run.get("run_id"),
            "field": run.get("field"),
            "contractor": run.get("contractor"),
            "install_date": run.get("install_date"),
            "start_date": run.get("start_date"),
            "stop_date": run.get("stop_date"),
            "failure_date": run.get("failure_date"),
            "TTF_days": run.get("TTF_days"),
            "event": run.get("event"),
            "_join_key": join_value,
        }
        row.update(_window_aggregate(telemetry_rows, NUMERIC_OPERATING_COLUMNS, prefix="telemetry_", start_date=start_date, anchor_date=anchor_date))
        row.update(_window_aggregate(techregime_rows, NUMERIC_OPERATING_COLUMNS, prefix="regime_", start_date=start_date, anchor_date=anchor_date))
        row.update(_window_aggregate(unified_rows, metrics, prefix="", start_date=start_date, anchor_date=anchor_date))
        row.update(_aggregate_feature_sources(unified_rows, metrics, start_date=start_date, anchor_date=anchor_date))

        for metric in NUMERIC_OPERATING_COLUMNS:
            _choose_final_metric_values(row, metric)
        _add_ratio_features(row, metrics)

        last30_rows = unified_rows[_window_mask(unified_rows["date"], "30d", start_date, anchor_date)]
        _add_last30_threshold_features(row, last30_rows)
        features_rows.append(row)

    features = pd.DataFrame(features_rows)
    unsafe_columns = tuple(sorted(column for column in features.columns if "_full_run_" in column))
    return OperatingDataResult(
        features=features,
        telemetry_daily=telemetry_daily,
        techregime_daily=techregime_daily,
        unified_daily=unified_daily,
        source_column_maps={
            "target_runs": target_column_map,
            "telemetry": telemetry_column_map,
            "techregime": techregime_column_map,
        },
        unsafe_feature_columns=unsafe_columns,
    )
