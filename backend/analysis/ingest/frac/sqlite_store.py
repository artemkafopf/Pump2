"""SQLite-хранилище реестра ГРП и его выдача в расчёт.

Три таблицы, по образцу телеметрии:

``frac_raw``      все 280 колонок выгрузки как есть, под позиционными именами
                  ``col_0001…`` плюс карта в ``frac_column_map`` — реестр нужен
                  и другим проектам, поэтому ничего не выбрасывается;
``frac_stages``   типизированная стадия: скважина, дата, № стадии, пласт, ГТМ,
                  подрядчик;
``frac_wells``    строка на скважину: был ли ГРП, сколько стадий и обработок,
                  первая и последняя дата.

Расчёт ЭЦН потребляет отсюда :func:`frac_flags_for_runs` — признак «ГРП был до
монтажа этого насоса». ⚠ Именно ДО монтажа: ГРП, сделанный после подъёма насоса,
о его наработке ничего сказать не может, а join по одной скважине втащил бы
будущее в прошлое.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis.paths import resolve_frac_db_path

from ..normalize import normalize_well, parse_date
from ..store_sync import dedup_across_files, guard_nonempty_source
from .processor import (
    DEFAULT_SOURCE_DIR,
    build_frac_stages,
    build_frac_wells,
    frac_column_map,
    merge_frac_registers,
    resolve_frac_files,
)

SQLITE_DEFAULT_PATH = resolve_frac_db_path()
SQLITE_RAW_TABLE = "frac_raw"
SQLITE_STAGES_TABLE = "frac_stages"
SQLITE_WELLS_TABLE = "frac_wells"
SQLITE_COLUMN_MAP_TABLE = "frac_column_map"
SQLITE_SUFFIXES = {".sqlite", ".db"}

SQLITE_METADATA_COLUMNS = {
    "_meta_row_id",
    "_meta_frac_date",
    "_meta_normalized_well",
    "_meta_source_file",
    "_meta_source_path",
}


@dataclass(frozen=True)
class FracSQLiteBuildResult:
    """Summary of a frac SQLite build."""

    sqlite_path: Path
    source_files: tuple[Path, ...]
    raw_rows: int
    stage_rows: int
    well_rows: int


def _storage_column_name(index: int) -> str:
    return f"col_{index:04d}"


def _frac_dedup_key(stages: pd.DataFrame) -> pd.Series:
    """Natural key per stage: ``(well, frac_date, stage_no)``.

    Stages of one treatment share a well and a date, so the stage number has to
    be part of the key — without it a 15-stage МГРП would collapse to one row the
    moment two exports overlap.
    """
    if stages.empty:
        return pd.Series(dtype="string")
    well = stages["well_key"].astype(str)
    date = stages["frac_date"].apply(
        lambda value: value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else ""
    )
    stage = stages["stage_no"].apply(
        lambda value: "" if value is None or pd.isna(value) else f"{float(value):g}"
    )
    key = well.str.cat([date.astype(str), stage.astype(str)], sep="|")
    return key.mask(well.eq("") & date.eq(""), "")


def _prepare_raw_storage_frame(merged: pd.DataFrame, stages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Built with a single concat: inserting 280 columns one at a time fragments
    # the frame badly enough that pandas warns about it.
    storage = merged.copy()
    storage.columns = [_storage_column_name(index) for index in range(1, len(merged.columns) + 1)]

    if "_raw_index" in stages.columns and not stages.empty:
        aligned = stages.drop_duplicates(subset=["_raw_index"]).set_index("_raw_index")
        frac_date = aligned["frac_date"].reindex(merged.index).apply(
            lambda value: value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
        )
        normalized_well = aligned["well_key"].reindex(merged.index)
    else:
        frac_date = pd.Series([None] * len(merged), index=merged.index, dtype=object)
        normalized_well = pd.Series([None] * len(merged), index=merged.index, dtype=object)

    meta = pd.DataFrame(
        {
            "_meta_row_id": range(1, len(merged) + 1),
            "_meta_frac_date": frac_date,
            "_meta_normalized_well": normalized_well,
            "_meta_source_file": merged.get("_source_file"),
            "_meta_source_path": merged.get("_source_path"),
        },
        index=merged.index,
    )
    return pd.concat([storage, meta], axis=1), frac_column_map(merged)


def _dates_to_text(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """SQLite has no date type; store ISO text so range predicates still sort."""
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].apply(
                lambda value: value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None
            )
    return result


def build_frac_sqlite(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    sqlite_path: Path | str = SQLITE_DEFAULT_PATH,
    verbose: bool = False,
) -> FracSQLiteBuildResult:
    """Import the frac register(s) into a SQLite lookup store."""
    sqlite_path = Path(sqlite_path)
    source_files = tuple(Path(path) for path in resolve_frac_files(source))
    # Refuse to wipe a populated store when no source files were found.
    guard_nonempty_source(source_files, sqlite_path, SQLITE_STAGES_TABLE, label="frac")

    merged = merge_frac_registers(source_files, verbose=verbose)
    if merged.empty:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            for table in (SQLITE_RAW_TABLE, SQLITE_STAGES_TABLE, SQLITE_WELLS_TABLE, SQLITE_COLUMN_MAP_TABLE):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
        return FracSQLiteBuildResult(sqlite_path, source_files, 0, 0, 0)

    merged = merged.reset_index(drop=True)
    stages = build_frac_stages(merged.assign(_raw_index=merged.index))
    if "_raw_index" in merged.columns:
        merged = merged.drop(columns=["_raw_index"])

    if not stages.empty:
        stages, dropped = dedup_across_files(stages, _frac_dedup_key(stages))
        if dropped:
            print(f"[frac] dedup dropped {dropped} stage(s) duplicated across files (kept newest)")

    wells = build_frac_wells(stages)

    raw_storage, column_map = _prepare_raw_storage_frame(merged, stages)
    stages_storage = _dates_to_text(
        stages.drop(columns=[c for c in ("_raw_index",) if c in stages.columns]),
        ("frac_date", "refrac_previous_date"),
    )
    wells_storage = _dates_to_text(wells, ("first_frac_date", "last_frac_date"))

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Building frac SQLite store at {sqlite_path}")

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        for table in (SQLITE_RAW_TABLE, SQLITE_STAGES_TABLE, SQLITE_WELLS_TABLE, SQLITE_COLUMN_MAP_TABLE):
            connection.execute(f"DROP TABLE IF EXISTS {table}")

        raw_storage.to_sql(SQLITE_RAW_TABLE, connection, if_exists="append", index=False)
        stages_storage.to_sql(SQLITE_STAGES_TABLE, connection, if_exists="append", index=False)
        wells_storage.to_sql(SQLITE_WELLS_TABLE, connection, if_exists="append", index=False)
        column_map.to_sql(SQLITE_COLUMN_MAP_TABLE, connection, if_exists="replace", index=False)

        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{SQLITE_STAGES_TABLE}_well_date "
            f"ON {SQLITE_STAGES_TABLE} (well_key, frac_date)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{SQLITE_WELLS_TABLE}_well "
            f"ON {SQLITE_WELLS_TABLE} (well_key)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{SQLITE_RAW_TABLE}_well_date "
            f"ON {SQLITE_RAW_TABLE} (_meta_normalized_well, _meta_frac_date)"
        )
        connection.commit()

    return FracSQLiteBuildResult(
        sqlite_path=sqlite_path,
        source_files=source_files,
        raw_rows=len(raw_storage),
        stage_rows=len(stages_storage),
        well_rows=len(wells_storage),
    )


def resolve_frac_sqlite_path(source: Path | str | Sequence[Path | str]) -> Path | None:
    """Resolve a frac SQLite store from a file, directory, or list."""
    candidates: list[Path] = []
    items = [source] if isinstance(source, (str, Path)) else list(source)
    for item in items:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in SQLITE_SUFFIXES:
            candidates.append(path.resolve())
        elif path.is_dir():
            for suffix in sorted(SQLITE_SUFFIXES):
                candidates.extend(sorted(child.resolve() for child in path.glob(f"*{suffix}") if child.is_file()))
    if not candidates:
        return None
    preferred = [path for path in candidates if path.stem.lower() == "frac"]
    if preferred:
        return preferred[0]
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_table(sqlite_path: Path | str, table: str, date_columns: Sequence[str]) -> pd.DataFrame:
    with sqlite3.connect(str(sqlite_path)) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {table}", connection)
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_frac_stages(sqlite_path: Path | str = SQLITE_DEFAULT_PATH) -> pd.DataFrame:
    """Typed stage table from the store."""
    return _read_table(sqlite_path, SQLITE_STAGES_TABLE, ("frac_date", "refrac_previous_date"))


def load_frac_wells(sqlite_path: Path | str = SQLITE_DEFAULT_PATH) -> pd.DataFrame:
    """Per-well summary from the store."""
    return _read_table(sqlite_path, SQLITE_WELLS_TABLE, ("first_frac_date", "last_frac_date"))


def load_frac_raw(sqlite_path: Path | str = SQLITE_DEFAULT_PATH) -> pd.DataFrame:
    """Full register with its original export headers restored."""
    with sqlite3.connect(str(sqlite_path)) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {SQLITE_RAW_TABLE}", connection)
        column_map = pd.read_sql_query(
            f"SELECT storage_label, original_name, column_order FROM {SQLITE_COLUMN_MAP_TABLE} ORDER BY column_order",
            connection,
        )
    if frame.empty:
        return frame
    mapping = {
        _storage_column_name(int(row["column_order"])): str(row["storage_label"])
        for _, row in column_map.iterrows()
    }
    restored = frame.rename(columns=mapping)
    ordered = [mapping[key] for key in sorted(mapping) if mapping[key] in restored.columns]
    metadata = [column for column in restored.columns if column.startswith("_meta_")]
    return restored[ordered + metadata]


def frac_flags_for_runs(
    runs: pd.DataFrame,
    *,
    stages: pd.DataFrame | None = None,
    sqlite_path: Path | str = SQLITE_DEFAULT_PATH,
    well_col: str = "well",
    install_col: str = "installation_date",
) -> pd.DataFrame:
    """Per-run fracturing flags, joined on ``(well, дата монтажа)``.

    Returns a frame index-aligned with ``runs`` carrying:

    ``ГРП``                    1 when the well has any recorded frac, else 0;
    ``ГРП стадий``             stages recorded on the well before this mount;
    ``Дата последнего ГРП``    latest frac date **at or before** the mount;
    ``ГРП до монтажа``         1 when that date exists.

    ⚠ Стыковка по ``(скважина, дата монтажа)``, а не по позиционному ``run`` —
    стоячее правило проекта. ⚠ «до монтажа» отсекает будущее: ГРП, проведённый
    после спуска насоса, не может объяснить его наработку.
    """
    result = pd.DataFrame(index=runs.index)
    result["ГРП"] = 0
    result["ГРП стадий"] = 0
    result["Дата последнего ГРП"] = pd.NaT
    result["ГРП до монтажа"] = 0

    if stages is None:
        if not Path(sqlite_path).exists():
            return result
        stages = load_frac_stages(sqlite_path)
    if stages is None or stages.empty or runs.empty:
        return result

    by_well: dict[str, pd.Series] = {
        str(well_key): group["frac_date"].dropna().sort_values()
        for well_key, group in stages.groupby("well_key", sort=False)
    }

    for idx, row in runs.iterrows():
        well_key = normalize_well(row.get(well_col, row.get("Скв.")))
        if not well_key:
            continue
        dates = by_well.get(well_key)
        if dates is None or dates.empty:
            continue
        result.at[idx, "ГРП"] = 1
        mount = parse_date(row.get(install_col, row.get("Дата монтажа")))
        if mount is None:
            continue
        prior = dates[dates <= mount.normalize()]
        if prior.empty:
            continue
        result.at[idx, "ГРП стадий"] = int(len(prior))
        result.at[idx, "Дата последнего ГРП"] = prior.max()
        result.at[idx, "ГРП до монтажа"] = 1

    return result


__all__ = [
    "FracSQLiteBuildResult",
    "SQLITE_COLUMN_MAP_TABLE",
    "SQLITE_DEFAULT_PATH",
    "SQLITE_RAW_TABLE",
    "SQLITE_STAGES_TABLE",
    "SQLITE_WELLS_TABLE",
    "build_frac_sqlite",
    "frac_flags_for_runs",
    "load_frac_raw",
    "load_frac_stages",
    "load_frac_wells",
    "resolve_frac_sqlite_path",
]
