"""Пересобрать SQLite-хранилище телеметрии из свежей выгрузки.

Тонкая обёртка над :mod:`analysis.ingest.telemetry`. По умолчанию берёт САМУЮ
СВЕЖУЮ датированную папку выгрузки (``analysis.paths.resolve_telemetry_source_dir``)
и пишет в НОВЫЙ файл рядом с текущим хранилищем — прежнее не затирается, пока вы
не сравните и не подмените осознанно.

⚠ Каждая выгрузка — ПОЛНАЯ история (2019-02-28 … дата выгрузки), поэтому свежая
папка полностью заменяет предыдущую. Сливать их не надо: дедупликация всё равно
выбросит старые строки, только сначала потратит время на их чтение.

    python scripts/ingest/build_telemetry_db.py --verbose
    python scripts/ingest/build_telemetry_db.py --output data/sqlite/telemetry.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from analysis.ingest.telemetry import build_telemetry_sqlite
from analysis.paths import LOCAL_SQLITE_DIR, parse_version_date, resolve_telemetry_source_dir


def _drop_version(source_dir: Path) -> str | None:
    """Дата актуализации выгрузки — из имён файлов внутри неё, а не из имени папки.

    ⚠ Папки выгрузок названы непоследовательно: ``20261508`` это YYYY-DD-MM, а
    файлы внутри — ``…20260815.xlsx``, YYYY-MM-DD. Имя папки как версия не
    разбирается (месяц 15), поэтому хранилище, названное по папке, выпадало из
    конвенции датированных версий и резолвер молча брал ПРЕДЫДУЩУЮ базу.
    Версия берётся из файлов.
    """
    versions = [parse_version_date(child) for child in source_dir.glob("*.xls*")]
    versions = [version for version in versions if version]
    if versions:
        return max(versions).strftime("%Y%m%d")
    return parse_version_date(source_dir) and parse_version_date(source_dir).strftime("%Y%m%d")


def _default_output(source_dir: Path) -> Path:
    """``telemetry <YYYYMMDD>.sqlite`` рядом с текущей базой, никогда поверх неё."""
    version = _drop_version(source_dir)
    stem = f"telemetry {version}" if version else "telemetry (новая)"
    return LOCAL_SQLITE_DIR / f"{stem}.sqlite"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None, help="Папка выгрузки (по умолчанию — свежая датированная)")
    parser.add_argument("--output", default=None, help="Путь к telemetry.sqlite (по умолчанию — новый файл)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # Line buffering matters as much as the encoding here: a build that
            # dies partway must not take its own progress log down with it.
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    args = parse_args(argv)
    source = Path(args.source) if args.source else resolve_telemetry_source_dir()
    output = Path(args.output) if args.output else _default_output(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Выгрузка:  {source}")
    print(f"Хранилище: {output}")
    if output.exists():
        print(f"  ⚠ файл существует ({output.stat().st_size / 2**20:.0f} МБ) и будет перезаписан")

    result = build_telemetry_sqlite(source, sqlite_path=output, verbose=args.verbose)
    print(
        f"\nГотово: {result.total_rows} исходных строк, {result.raw_indexed_rows} в сыром слое, "
        f"{result.daily_rows} суточных"
    )

    with sqlite3.connect(str(output)) as connection:
        span = pd.read_sql_query(
            "SELECT MIN(_meta_record_date) a, MAX(_meta_record_date) b, "
            "COUNT(DISTINCT _meta_normalized_well) w FROM telemetry_daily",
            connection,
        )
    print(f"  период: {span['a'][0]} … {span['b'][0]}; скважин: {span['w'][0]}")
    print(f"  файлов: {len(result.source_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
