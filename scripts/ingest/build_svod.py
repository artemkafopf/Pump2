"""Собрать Свод ЭЦН с нуля из сырых выгрузок и сравнить с действующим.

Тонкая обёртка над :mod:`analysis.ingest.svod`. Результат — в
``results/svod_build_full/<дата>/``: сам регистр, аудит сборщика и построчный диф
против того Свода, который сейчас читает модель. Живой вход
``data/inputs/Отказы свод с анализом.xlsx`` НЕ трогается: промоут — отдельное
осознанное действие после просмотра дифа.

    python scripts/ingest/build_svod.py
    python scripts/ingest/build_svod.py --no-techregime --no-telemetry   # быстрый прогон
    python scripts/ingest/build_svod.py --telemetry-db data/sqlite/telemetry_20261508.sqlite
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from analysis.ingest.config import EVENT_SCOPE_OPTIONS, DEFAULT_EVENT_SCOPE
from analysis.ingest.svod.build_svod import (
    build_svod_from_scratch,
    build_svod_incrementally,
    default_sources,
    use_utf8_console,
)
from analysis.ingest.svod.diff import compare_registers
from analysis.ingest.svod.report import DEFECTS_FIXED, collect_stats, write_report
from analysis.ingest.svod.pdk_processor import pdk_observation_horizon
from analysis.ingest.svod.versioning import decide_build_mode
from analysis.paths import (
    LOCAL_INPUT_DIR,
    SVOD_DOCUMENT_NAME,
    results_dir,
    resolve_svod_main_path,
)

SLUG = "svod_build_full"
#: Полный Свод — ВСЁ, что доехало, с пометкой пригодности. ⚠⚠ Остаётся в results/,
#: РЯДОМ С ВХОДОМ РАСЧЁТА ему не место: файл с непригодными строками в одной папке
#: с моделью — это ровно та ошибка, от которой его пометки и защищают.
FULL_OUTPUT_NAME = "Свод полный (не для расчёта).xlsx"


def _output_name(horizon) -> str:
    """``Свод ЭЦН <YYYYMMDD>.xlsx`` — дата АКТУАЛЬНОСТИ, а не дня сборки.

    Регистр не может сказать ничего правее горизонта наблюдения ПДК, поэтому именно
    горизонт и есть его дата актуальности: две сборки в разные дни из одной выгрузки
    ПДК содержат одно и то же. День сборки записан на листе «Метаданные сборки».
    """
    stem = Path(SVOD_DOCUMENT_NAME).stem
    if horizon is None:
        return SVOD_DOCUMENT_NAME
    return f"{stem} {pd.Timestamp(horizon):%Y%m%d}.xlsx"


def _peek_horizon(pdk_path):
    """Горизонт ПДК до сборки — им решается, есть ли что дописывать."""
    from analysis.ingest.svod.pdk_processor import extract_new_failures_from_pdk

    try:
        return pdk_observation_horizon(extract_new_failures_from_pdk(pdk_path, None))
    except Exception as exc:
        print(f"  ⚠ Горизонт ПДК заранее не определился ({exc}); режим выберется без него")
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="Путь к собранному регистру")
    parser.add_argument("--telemetry-db", default=None, help="Путь к telemetry.sqlite для обогащения")
    parser.add_argument("--frac-db", default=None, help="Путь к frac.sqlite")
    parser.add_argument("--compare-with", default=None, help="Свод для дифа (по умолчанию — действующий вход модели)")
    parser.add_argument("--no-compare", action="store_true", help="Не строить диф")
    parser.add_argument("--no-techregime", dest="include_techregime", action="store_false")
    parser.add_argument("--no-telemetry", dest="include_telemetry", action="store_false")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Горизонт наблюдения. По умолчанию — последняя дата остановки в ПДК: "
             "правее неё отказ записать нечем, поэтому экспозиция туда не идёт.",
    )
    parser.add_argument(
        "--extend",
        default=None,
        help="Дописать этот регистр вместо полной пересборки. По умолчанию сборщик "
             "решает сам: перестраивает, только если изменился ПОДХОД к формированию.",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Перестроить с нуля, даже если подход не менялся.",
    )
    parser.add_argument(
        "--full-output",
        default=None,
        help="Путь к ПОЛНОМУ Своду (всё, что доехало, с пометкой пригодности и причиной).",
    )
    parser.add_argument("--event-scope", choices=EVENT_SCOPE_OPTIONS, default=DEFAULT_EVENT_SCOPE)
    parser.set_defaults(include_techregime=True, include_telemetry=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    use_utf8_console()
    args = parse_args(argv)

    out_dir = results_dir(SLUG)
    sources = default_sources()
    horizon = _peek_horizon(sources.pdk)
    # ⚠ Регистр пишется В ВХОД МОДЕЛИ (`data/inputs/`), но версией, а не поверх:
    # прежний файл остаётся на месте, а резолвер сам берёт самую свежую версию.
    output = Path(args.output) if args.output else LOCAL_INPUT_DIR / _output_name(horizon)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Источники:")
    print(sources.describe())
    print(f"\nВыход: {output}\n")

    full_output = Path(args.full_output) if args.full_output else out_dir / "tables" / FULL_OUTPUT_NAME

    # Дописывать или перестраивать — решается по ПОДХОДУ, а не по приросту данных.
    existing = Path(args.extend) if args.extend else Path(resolve_svod_main_path())
    decision = decide_build_mode(existing, horizon=horizon)
    if args.full_rebuild:
        decision = type(decision)(True, "запрошено явно (--full-rebuild)", decision.previous)
    print(decision.describe())
    if decision.previous is not None:
        print(f"  Прежняя сборка: {decision.previous.built_at}, горизонт {decision.previous.horizon}")

    common = dict(
        audit_xlsx=str(out_dir / "logs" / "svod_audit.xlsx"),
        audit_md=str(out_dir / "logs" / "svod_audit.md"),
        telemetry_db_path=args.telemetry_db or str(sources.telemetry_db),
        frac_path=args.frac_db or str(sources.frac_db),
    )
    if decision.full_rebuild:
        workflow = build_svod_from_scratch(
            output_workbook=str(output),
            full_output_workbook=str(full_output),
            seed_path=str(out_dir / "logs" / "_svod_seed.xlsx"),
            include_techregime=args.include_techregime,
            include_telemetry=args.include_telemetry,
            event_scope=args.event_scope,
            as_of=args.as_of,
            **common,
        )
    else:
        workflow = build_svod_incrementally(
            existing,
            output_workbook=str(output),
            full_output_workbook=str(full_output),
            as_of=args.as_of,
            **common,
        )
    print(f"\nСобрано строк: {len(workflow.new_failures)}")

    described = {
        "ПДК": str(sources.pdk),
        "паспорт оборудования (Big)": str(sources.artificial_lift),
        "ОПЗ": str(sources.opz),
        "лаборатория": str(sources.lab),
        "ТехРежим": str(sources.techregime),
        "телеметрия (выгрузка)": str(sources.telemetry_dir),
        "телеметрия (хранилище)": args.telemetry_db or str(sources.telemetry_db),
        "ГРП (хранилище)": args.frac_db or str(sources.frac_db),
    }
    stats = collect_stats(output, described)
    print(
        f"  отказов {stats.failures}, работает {stats.running}, "
        f"категория узла {stats.node_coverage_all:.1%} (2018+: {stats.node_coverage_2018:.1%})"
    )

    diff = None
    tables = out_dir / "tables"
    if not args.no_compare:
        reference = Path(args.compare_with) if args.compare_with else Path(resolve_svod_main_path())
        if reference.exists():
            print(f"\nСравнение с {reference}")
            diff = compare_registers(output, reference)
            print(diff.describe())

            for name, frame in diff.to_frames().items():
                if frame.empty:
                    continue
                path = tables / f"svod_diff_{name}.csv"
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"  {path.name}: {len(frame)} строк")

            summary_path = tables / "svod_diff_summary.csv"
            pd.DataFrame(
                [{"показатель": key, "значение": value} for key, value in diff.summary.items()]
            ).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"  {summary_path.name}")
        else:
            print(f"\n⚠ Нечего сравнивать: {reference} не найден")

    report_path = write_report(
        out_dir / "reports" / "svod_build_report.html",
        stats,
        diff,
        DEFECTS_FIXED,
        built_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    print(f"\nОтчёт (RU+EN): {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
