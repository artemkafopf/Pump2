"""Отпечаток правил сборки и решение «дописать или перестроить».

Свод не надо перестраивать каждый раз. Источники прирастают вперёд: ПДК доезжает
до новой даты, паспорт добавляет спуски — и правильная реакция на это «дописать
новый интервал», а не «собрать всё заново». Перестраивать целиком нужно ровно
тогда, когда меняется **подход к формированию**: схема колонок, правила разметки
причин, классификатор узла, набор источников.

⚠⚠ «Подход изменился» нельзя оставлять на память человека. Правки в таблицы
правил — обычное дело, а последствие у них редкое и неочевидное: старые строки
регистра остаются размеченными ПО-СТАРОМУ, новые приезжают по-новому, и в одном
файле молча живут две разные разметки. Поэтому здесь считается **отпечаток**
декларативных таблиц, он пишется в регистр, и сборщик сам решает, что делать.

Отпечаток берётся с ТАБЛИЦ, а не с текста модулей: правка комментария или
опечатки в докстринге не должна требовать часовой пересборки, а изменение хоть
одного правила — должно.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from analysis.paths import parse_version_date

METADATA_SHEET = "Метаданные сборки"

#: Ручная версия ПОДХОДА. Поднимается, когда меняется что-то, чего отпечаток
#: таблиц не видит: порядок этапов, семантика горизонта, правило схлопывания.
BUILD_RULES_VERSION = "2026.08.15"


def rules_fingerprint() -> str:
    """SHA-256 по декларативным таблицам, задающим форму регистра."""
    from ..config import (
        BIG_COLUMN_RULES,
        LAB_CHEMISTRY_COLUMNS,
        OPZ_COLUMNS,
        TECHREGIME_COLUMN_RULES,
        build_svod_target_columns,
    )
    from .causes import (
        CAUSE_RULES,
        DISPUTED_RULES,
        NODE_CATEGORIES,
        TEXT_RULES,
        UNSPECIFIED_REASONS,
    )

    payload = {
        "version": BUILD_RULES_VERSION,
        "columns": build_svod_target_columns(),
        "cause_rules": [list(rule) for rule in CAUSE_RULES],
        "disputed_rules": list(DISPUTED_RULES),
        "text_rules": [list(rule) for rule in TEXT_RULES],
        "unspecified": sorted(UNSPECIFIED_REASONS),
        "node_categories": list(NODE_CATEGORIES),
        "big_columns": {k: list(v) for k, v in BIG_COLUMN_RULES.items()},
        "techregime_columns": {k: [str(part) for part in v] for k, v in TECHREGIME_COLUMN_RULES.items()},
        "opz_columns": dict(OPZ_COLUMNS),
        "lab_columns": list(LAB_CHEMISTRY_COLUMNS),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class BuildMetadata:
    """Паспорт сборки, который лежит в самом регистре."""

    rules_version: str
    rules_fingerprint: str
    horizon: Optional[str]
    built_at: str
    sources: Mapping[str, str]

    @classmethod
    def current(cls, horizon, sources: Mapping[str, Any]) -> "BuildMetadata":
        return cls(
            rules_version=BUILD_RULES_VERSION,
            rules_fingerprint=rules_fingerprint(),
            horizon=str(pd.Timestamp(horizon).date()) if horizon is not None else None,
            built_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sources={str(name): describe_source(value) for name, value in sources.items()},
        )

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"параметр": "версия правил", "значение": self.rules_version},
            {"параметр": "отпечаток правил", "значение": self.rules_fingerprint},
            {"параметр": "горизонт наблюдения", "значение": self.horizon or ""},
            {"параметр": "собран", "значение": self.built_at},
        ]
        rows += [{"параметр": f"источник: {name}", "значение": value} for name, value in self.sources.items()]
        return pd.DataFrame(rows)

    @classmethod
    def read(cls, path) -> Optional["BuildMetadata"]:
        """Прочитать паспорт из регистра; ``None`` — если его там нет."""
        try:
            frame = pd.read_excel(path, sheet_name=METADATA_SHEET)
        except Exception:
            return None
        values = dict(zip(frame["параметр"].astype(str), frame["значение"].astype(str)))
        if "отпечаток правил" not in values:
            return None
        return cls(
            rules_version=values.get("версия правил", ""),
            rules_fingerprint=values.get("отпечаток правил", ""),
            horizon=values.get("горизонт наблюдения") or None,
            built_at=values.get("собран", ""),
            sources={
                key.split(": ", 1)[1]: value
                for key, value in values.items()
                if key.startswith("источник: ")
            },
        )

    def as_dict(self) -> dict:
        return asdict(self)


def describe_source(value: Any) -> str:
    """Имя источника с датой актуализации, если она есть в имени файла."""
    path = Path(str(value))
    version = parse_version_date(path)
    label = path.name if path.suffix else str(value)
    return f"{label} (версия {version})" if version else label


@dataclass(frozen=True)
class BuildDecision:
    """Что делать с существующим регистром и почему."""

    full_rebuild: bool
    reason: str
    previous: Optional[BuildMetadata] = None

    def describe(self) -> str:
        mode = "ПОЛНАЯ ПЕРЕСБОРКА" if self.full_rebuild else "ДОПИСЫВАНИЕ"
        return f"  Режим: {mode} — {self.reason}"


def decide_build_mode(existing_register, horizon=None) -> BuildDecision:
    """Решить, дописывать существующий регистр или строить заново.

    Полная пересборка — только когда изменился ПОДХОД (или регистра ещё нет):
    прирост данных сам по себе основанием не является.
    """
    path = Path(existing_register) if existing_register else None
    if path is None or not path.exists():
        return BuildDecision(True, "существующего регистра нет")

    previous = BuildMetadata.read(path)
    if previous is None:
        return BuildDecision(
            True,
            "в регистре нет паспорта сборки: он собран прежним сборщиком, "
            "и чем размечены его строки — неизвестно",
        )

    current_fingerprint = rules_fingerprint()
    if previous.rules_fingerprint != current_fingerprint:
        return BuildDecision(
            True,
            f"изменился подход к формированию (отпечаток правил "
            f"{previous.rules_fingerprint} → {current_fingerprint}): дописывать нельзя, "
            "иначе в одном файле окажутся две разные разметки",
            previous,
        )
    if previous.rules_version != BUILD_RULES_VERSION:
        return BuildDecision(
            True,
            f"изменилась версия подхода ({previous.rules_version} → {BUILD_RULES_VERSION})",
            previous,
        )

    if horizon is not None and previous.horizon:
        if str(pd.Timestamp(horizon).date()) <= previous.horizon:
            return BuildDecision(
                False,
                f"правила прежние, горизонт не сдвинулся ({previous.horizon}) — "
                "дописывать нечего, прогон будет холостым",
                previous,
            )
        return BuildDecision(
            False,
            f"правила прежние, горизонт сдвинулся {previous.horizon} → "
            f"{pd.Timestamp(horizon).date()}",
            previous,
        )
    return BuildDecision(False, "правила прежние", previous)


def write_metadata_sheet(workbook, metadata: BuildMetadata) -> None:
    """Добавить лист-паспорт в открытую книгу openpyxl."""
    from openpyxl.styles import Font

    if METADATA_SHEET in workbook.sheetnames:
        del workbook[METADATA_SHEET]
    sheet = workbook.create_sheet(METADATA_SHEET)
    frame = metadata.to_frame()
    for index, title in enumerate(frame.columns, start=1):
        cell = sheet.cell(row=1, column=index, value=str(title))
        cell.font = Font(bold=True)
    for row_index, row in enumerate(frame.itertuples(index=False), start=2):
        for col_index, value in enumerate(row, start=1):
            sheet.cell(row=row_index, column=col_index, value=str(value))
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 64
    sheet.freeze_panes = "A2"


__all__ = [
    "BUILD_RULES_VERSION",
    "METADATA_SHEET",
    "BuildDecision",
    "BuildMetadata",
    "decide_build_mode",
    "describe_source",
    "rules_fingerprint",
    "write_metadata_sheet",
]
