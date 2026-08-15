"""Сборка Свода ЭЦН с нуля из сырых выгрузок.

Регистр строится не «дополнением» существующего файла, а с чистого листа: пустая
книга со схемой :func:`analysis.ingest.config.build_svod_target_columns`
заполняется прогоном рабочего процесса в режиме 2 (rebuild from scratch).

Источники берутся из :mod:`analysis.paths`; результат — в
``results_dir("svod_build_full")``, а не поверх живого входа модели. Промоут в
``data/inputs/`` — отдельное осознанное действие, после того как посмотрели диф.

Пропуск ТехРежима / телеметрии (``include_techregime=False`` /
``include_telemetry=False``) не меняет схему выхода: колонки остаются, просто
пустые — они приходят из схемы-затравки, а не из этапа обогащения.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from openpyxl import Workbook

from analysis.paths import (
    resolve_equipment_big_path,
    resolve_frac_db_path,
    resolve_source_dir,
    resolve_telemetry_db_path,
    resolve_telemetry_source_dir,
)

from ..config import (
    DEFAULT_EVENT_SCOPE,
    DEFAULT_OPERATING_DATA_PRIMARY_SOURCE,
    DEFAULT_TECHREGIME_INTERVAL_DAYS,
    DEFAULT_TECHREGIME_USE_SQL_IF_AVAILABLE,
    SVOD_SHEET_NAME,
    build_svod_target_columns,
)
from .exclusions import write_full_register
from .main import FailureUpdateWorkflow


@dataclass(frozen=True)
class SvodSources:
    """Every input the register is built from, already resolved to a path."""

    pdk: Path
    artificial_lift: Path
    opz: Path
    lab: Path
    techregime: Path
    telemetry_dir: Path
    telemetry_db: Path
    frac_db: Path

    def describe(self) -> str:
        return "\n".join(
            f"  {name:16s} {value}" for name, value in (
                ("ПДК", self.pdk),
                ("паспорт (Big)", self.artificial_lift),
                ("ОПЗ", self.opz),
                ("лаборатория", self.lab),
                ("ТехРежим", self.techregime),
                ("телеметрия", self.telemetry_dir),
                ("телеметрия БД", self.telemetry_db),
                ("ГРП БД", self.frac_db),
            )
        )


def default_sources() -> SvodSources:
    """Resolve every register source through :mod:`analysis.paths`."""
    return SvodSources(
        pdk=resolve_source_dir("pdk"),
        artificial_lift=resolve_equipment_big_path(),
        opz=resolve_source_dir("opz"),
        lab=resolve_source_dir("lab"),
        techregime=resolve_source_dir("techregime"),
        telemetry_dir=resolve_telemetry_source_dir(),
        telemetry_db=resolve_telemetry_db_path(),
        frac_db=resolve_frac_db_path(),
    )


def write_svod_seed_workbook(
    path,
    columns: Optional[Sequence[str]] = None,
    sheet_name: str = SVOD_SHEET_NAME,
) -> str:
    """Write a header-only workbook carrying the canonical Свод schema.

    This seed defines the register's columns so the rebuild-from-scratch
    workflow emits a stable schema even when some enrichment sources are absent.
    """
    columns = list(columns) if columns is not None else build_svod_target_columns()
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    for col_idx, name in enumerate(columns, start=1):
        worksheet.cell(row=1, column=col_idx, value=name)
    path = str(path)
    workbook.save(path)
    return path


def build_svod_from_scratch(
    pdk_path=None,
    *,
    artificial_lift_path=None,
    artificial_lift_manual_path=None,
    opz_clean_path=None,
    opz_standard_path=None,
    lab_path=None,
    techregime_path=None,
    telemetry_dir=None,
    telemetry_db_path=None,
    telemetry_vt_path=None,
    frac_path=None,
    as_of=None,
    include_techregime: bool = True,
    include_telemetry: bool = True,
    output_workbook: Optional[str] = None,
    full_output_workbook: Optional[str] = None,
    audit_xlsx: Optional[str] = None,
    audit_md: Optional[str] = None,
    seed_path: Optional[str] = None,
    keep_seed: bool = False,
    columns: Optional[Sequence[str]] = None,
    event_scope: str = DEFAULT_EVENT_SCOPE,
    techregime_use_sql_if_available: bool = DEFAULT_TECHREGIME_USE_SQL_IF_AVAILABLE,
    techregime_interval=DEFAULT_TECHREGIME_INTERVAL_DAYS,
    operating_data_primary_source: str = DEFAULT_OPERATING_DATA_PRIMARY_SOURCE,
    include_operating_time_derived: bool = False,
    apply_presentation: bool = True,
    dry_run: bool = False,
) -> FailureUpdateWorkflow:
    """Build the Свод register from raw sources, with no target workbook.

    Any source left as ``None`` is resolved from :func:`default_sources`.
    Returns the executed :class:`FailureUpdateWorkflow` for inspection.
    """
    sources = default_sources()
    pdk_path = pdk_path if pdk_path is not None else sources.pdk
    artificial_lift_path = artificial_lift_path if artificial_lift_path is not None else sources.artificial_lift
    opz_clean_path = opz_clean_path if opz_clean_path is not None else sources.opz
    lab_path = lab_path if lab_path is not None else sources.lab
    techregime_path = techregime_path if techregime_path is not None else sources.techregime
    telemetry_dir = telemetry_dir if telemetry_dir is not None else sources.telemetry_dir
    telemetry_db_path = telemetry_db_path if telemetry_db_path is not None else sources.telemetry_db
    frac_path = frac_path if frac_path is not None else sources.frac_db

    if output_workbook is None:
        raise ValueError(
            "build_svod_from_scratch: output_workbook is required — write the "
            'register under results_dir("svod_build_full"), never over the live input.'
        )
    output_dir = Path(output_workbook).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_path = seed_path or str(output_dir / f"_svod_seed_{Path(output_workbook).stem}.xlsx")
    write_svod_seed_workbook(seed_path, columns=columns)

    if not include_techregime:
        techregime_path = None
    if not include_telemetry:
        telemetry_dir = None
        telemetry_vt_path = None
        telemetry_db_path = None

    audit_xlsx = audit_xlsx or str(Path(output_workbook).with_name(f"{Path(output_workbook).stem}_audit.xlsx"))
    audit_md = audit_md or str(Path(output_workbook).with_name(f"{Path(output_workbook).stem}_audit.md"))

    try:
        workflow = FailureUpdateWorkflow(
            target_path=seed_path,
            pdk_path=pdk_path,
            artificial_lift_path=artificial_lift_path,
            artificial_lift_manual_path=artificial_lift_manual_path,
            techregime_path=techregime_path,
            opz_clean_path=opz_clean_path,
            opz_standard_path=opz_standard_path,
            telemetry_vt_path=telemetry_vt_path,
            telemetry_dir=telemetry_dir,
            telemetry_db_path=telemetry_db_path,
            lab_path=lab_path,
            frac_path=frac_path,
            as_of=as_of,
            update_mode=2,  # rebuild from scratch
            output_workbook=output_workbook,
            audit_xlsx=audit_xlsx,
            audit_md=audit_md,
            dry_run=dry_run,
            event_scope=event_scope,
            techregime_use_sql_if_available=techregime_use_sql_if_available,
            techregime_interval=techregime_interval,
            operating_data_primary_source=operating_data_primary_source,
            include_operating_time_derived=include_operating_time_derived,
            include_event_failure_flag=False,
            apply_presentation=apply_presentation,
            normalize_svod_fields=True,
        )
        workflow.execute()
    finally:
        if not keep_seed:
            try:
                Path(seed_path).unlink()
            except OSError:
                pass

    # Полный Свод — ОТДЕЛЬНЫМ файлом. Всё, что доехало из источников, включая
    # непригодное, с пометкой и причиной. Не вход расчёта, а картина «что есть».
    if full_output_workbook and not dry_run and workflow.new_failures is not None:
        write_full_register(
            full_output_workbook,
            workflow.new_failures,
            workflow.exclusions,
            columns=list(columns) if columns else build_svod_target_columns(),
            horizon=workflow.observation_horizon,
        )
        print(f"  Полный Свод (со служебными пометками): {full_output_workbook}")
        print(workflow.exclusions.describe())

    return workflow


def build_svod_incrementally(
    existing_register,
    *,
    output_workbook: str,
    full_output_workbook: Optional[str] = None,
    **kwargs,
) -> FailureUpdateWorkflow:
    """Дописать существующий регистр вместо полной пересборки.

    Прогон идёт в режиме 1 (сверка по ключу): совпавшая по ``(скважина, дата
    монтажа)`` строка обновляется НА МЕСТЕ, новая — дописывается, а спорная уходит
    в лист needs-review, а не молча заменяет собой прежнюю.

    ⚠ Обновление на месте здесь обязательно, а не только дописывание: когда ПДК
    доезжает до новой даты, часть прежде ЦЕНЗУРИРОВАННЫХ пусков успела отказать, и
    их строки должны закрыться. Чистый append оставил бы их вечно живыми.
    """
    sources = default_sources()
    resolved = {
        "pdk_path": sources.pdk,
        "artificial_lift_path": sources.artificial_lift,
        "opz_clean_path": sources.opz,
        "lab_path": sources.lab,
        "techregime_path": sources.techregime,
        "telemetry_dir": sources.telemetry_dir,
        "telemetry_db_path": sources.telemetry_db,
        "frac_path": sources.frac_db,
    }
    resolved.update({key: value for key, value in kwargs.items() if value is not None})

    workflow = FailureUpdateWorkflow(
        target_path=str(existing_register),
        update_mode=1,          # key-based reconciliation: update / needs-review / append
        output_workbook=output_workbook,
        include_event_failure_flag=False,
        apply_presentation=True,
        normalize_svod_fields=True,
        **resolved,
    )
    workflow.execute()

    if full_output_workbook and workflow.new_failures is not None:
        write_full_register(
            full_output_workbook,
            workflow.new_failures,
            workflow.exclusions,
            columns=build_svod_target_columns(),
            horizon=workflow.observation_horizon,
        )
        print(f"  Полный Свод (со служебными пометками): {full_output_workbook}")
        print(workflow.exclusions.describe())
    return workflow


def use_utf8_console() -> None:
    """Ask for UTF-8 **and line buffering** on stdout/stderr.

    Two separate hazards, both already hit:

    * a Windows console at code page 866/1251 cannot encode the progress output,
      so the run dies on the final "✓ Complete!" *after* the register has been
      built — which reads as a failed build;
    * when stdout is a pipe it is block-buffered, so a build that dies partway
      loses every line it printed and the log shows only the traceback. Line
      buffering keeps the progress that led up to the failure.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


__all__ = [
    "SvodSources",
    "build_svod_from_scratch",
    "build_svod_incrementally",
    "default_sources",
    "use_utf8_console",
    "write_svod_seed_workbook",
]
