"""Собрать SQLite-хранилище реестра ГРП из «Свода ГРП».

Тонкая обёртка над :mod:`analysis.ingest.frac`. Пишет три таблицы: полную
выгрузку как есть (``frac_raw`` + карта колонок), типизированные стадии
(``frac_stages``) и сводку по скважинам (``frac_wells``).

    python scripts/ingest/build_frac_db.py            # в data/sqlite/frac.sqlite
    python scripts/ingest/build_frac_db.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.ingest.frac import build_frac_sqlite, load_frac_stages, load_frac_wells
from analysis.paths import LOCAL_SQLITE_DIR, resolve_source_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None, help="Файл или папка «Свод ГРП» (по умолчанию — analysis.paths)")
    parser.add_argument("--output", default=None, help="Путь к frac.sqlite (по умолчанию data/sqlite/frac.sqlite)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # Line buffering matters as much as the encoding here: a build that
            # dies partway must not take its own progress log down with it.
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    args = parse_args(argv)
    source = Path(args.source) if args.source else resolve_source_dir("frac")
    output = Path(args.output) if args.output else LOCAL_SQLITE_DIR / "frac.sqlite"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Источник ГРП: {source}")
    print(f"Хранилище:    {output}")

    result = build_frac_sqlite(source, sqlite_path=output, verbose=args.verbose)
    print(
        f"\nГотово: {result.raw_rows} строк выгрузки, {result.stage_rows} стадий, "
        f"{result.well_rows} скважин"
    )

    stages = load_frac_stages(output)
    wells = load_frac_wells(output)
    if not stages.empty:
        print(f"  период ГРП: {stages['frac_date'].min():%Y-%m-%d} … {stages['frac_date'].max():%Y-%m-%d}")
        print(f"  обработок (скважина+дата): {stages.drop_duplicates(['well_key', 'frac_date']).shape[0]}")
        by_field = wells["field"].value_counts()
        print("  скважин с ГРП по месторождениям: " + ", ".join(f"{k} {v}" for k, v in by_field.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
