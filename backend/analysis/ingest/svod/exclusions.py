"""Реестр отсева: что не попало в выходной Свод и ПОЧЕМУ.

Сборщик отбрасывает строки на семи разных этапах, и до сих пор все они уходили в
`print` со счётчиком: «dropped 1448 active run(s)». Кто именно, какая скважина,
по какому правилу — нигде. Проверить решение было нельзя, оспорить тоже.

Здесь это исправлено. Каждый отсев регистрируется строкой с причиной, и на выходе
получаются ДВА регистра:

``Отказы свод с анализом.xlsx``
    выходной. Только пригодные строки — то, что уходит дальше в расчёт;

``Свод полный.xlsx``
    всё, что есть на текущий момент, включая непригодное, с колонками
    ``Пригодно для расчёта`` (0/1), ``Причина непригодности`` и ``Этап отсева``.

⚠⚠ Второй регистр — **не вход расчёта**. Он для человека: посмотреть, что уже
доехало из источников, но пока не может быть использовано, и по какой причине.
Строка «непригодно» почти всегда означает не «мусор», а «данные опережают ПДК»:
пуск уже смонтирован или уже закрылся, а отказы за этот период ещё не выгружены.
Как только ПДК догонит, эта же строка станет пригодной сама собой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

USABLE_COLUMN = "Пригодно для расчёта"
REASON_COLUMN = "Причина непригодности"
STAGE_COLUMN = "Этап отсева"


@dataclass(frozen=True)
class Reason:
    """Одна причина непригодности: короткий код и объяснение для человека."""

    code: str
    stage: str
    explanation: str

    def __str__(self) -> str:  # pragma: no cover - удобство отладки
        return self.code


#: ⚠ «Данные опережают ПДК» — самая частая причина, и она ВРЕМЕННАЯ.
MOUNTED_AFTER_HORIZON = Reason(
    "монтаж после горизонта ПДК",
    "горизонт наблюдения",
    "Пуск смонтирован позже последней даты ПДК, поэтому его исход записать нечем: "
    "в выборке он работал бы чистым разбавителем и тянул выживаемость вверх в ранней "
    "полосе. Станет пригодным, когда ПДК догонит паспорт.",
)
CLOSED_AFTER_HORIZON = Reason(
    "закрытие после горизонта ПДК",
    "горизонт наблюдения",
    "Паспорт показывает отказ или демонтаж позже последней даты ПДК. Само событие в "
    "выходной Свод не берётся (ПДК его не подтверждает), но прожитая жизнь пуска "
    "сохраняется — строка уходит в расчёт цензурированной на горизонте.",
)
RUNTIME_BEYOND_HORIZON = Reason(
    "наработка за горизонтом ПДК",
    "горизонт наблюдения",
    "Наработка живого пуска считалась по календарю дальше последней даты ПДК. "
    "Излишек — экспозиция без возможности наблюдать отказ, он обрезан.",
)
NON_OIL_PURPOSE = Reason(
    "не нефтяная добыча",
    "нефтяной фильтр",
    "Назначение ствола — нагнетание, пьезометрия, водозабор, консервация или "
    "техоперации. Регистр ведётся по добывающим ЭЦН.",
)
BRINE_BORE = Reason(
    "рассольный ствол (рс)",
    "нефтяной фильтр",
    "Рассольный ствол физически отдельная скважина от нефтяной с тем же номером; "
    "их слияние порождает фантомный «живой насос».",
)
STALE_OPEN_RUN = Reason(
    "устаревшая открытая строка",
    "живые пуски",
    "Открытая строка, чей монтаж предшествует последнему ЗАКРЫТОМУ пуску того же "
    "ствола: пуск давно закончился, строка в паспорте не закрыта.",
)
SUPERSEDED_OPEN_RUN = Reason(
    "не последний открытый пуск скважины",
    "живые пуски",
    "У скважины несколько открытых строк; живым может быть только последний спуск.",
)
KDMNU_MARKER = Reason(
    "КДМНУ",
    "ПДК",
    "Запись относится к КДМНУ, а не к фонду ЭЦН.",
)

ALL_REASONS = (
    MOUNTED_AFTER_HORIZON,
    CLOSED_AFTER_HORIZON,
    RUNTIME_BEYOND_HORIZON,
    NON_OIL_PURPOSE,
    BRINE_BORE,
    STALE_OPEN_RUN,
    SUPERSEDED_OPEN_RUN,
    KDMNU_MARKER,
)


@dataclass
class ExclusionLedger:
    """Собирает отброшенные строки вместе с причиной.

    Хранит сами строки, а не только счётчики: из них потом собирается полный Свод,
    в котором видно и что отсеяли, и почему.
    """

    records: list[pd.DataFrame] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, rows: pd.DataFrame, reason: Reason) -> None:
        """Записать отброшенные строки под указанной причиной."""
        if rows is None or rows.empty:
            return
        marked = rows.copy()
        marked[USABLE_COLUMN] = 0
        marked[REASON_COLUMN] = reason.code
        marked[STAGE_COLUMN] = reason.stage
        self.records.append(marked)
        self.counts[reason.code] = self.counts.get(reason.code, 0) + len(marked)

    def note(self, reason: Reason, count: int) -> None:
        """Отметить причину, у которой строки остались в выборке (например обрезка)."""
        if count:
            self.counts[reason.code] = self.counts.get(reason.code, 0) + int(count)

    @property
    def total(self) -> int:
        return sum(len(frame) for frame in self.records)

    def frame(self) -> pd.DataFrame:
        """Все отброшенные строки одним кадром."""
        if not self.records:
            return pd.DataFrame()
        return pd.concat(self.records, ignore_index=True, sort=False)

    def summary(self) -> pd.DataFrame:
        """Причина → сколько строк, с объяснением."""
        by_code = {reason.code: reason for reason in ALL_REASONS}
        rows = [
            {
                "причина": code,
                "строк": count,
                "этап": by_code[code].stage if code in by_code else "",
                "объяснение": by_code[code].explanation if code in by_code else "",
            }
            for code, count in sorted(self.counts.items(), key=lambda item: -item[1])
        ]
        return pd.DataFrame(rows)

    def describe(self) -> str:
        if not self.counts:
            return "  Отсев: пусто"
        parts = ", ".join(f"{code}: {count}" for code, count in sorted(self.counts.items(), key=lambda i: -i[1]))
        return f"  Отсев ({self.total} строк отброшено): {parts}"


def build_full_register(usable: pd.DataFrame, ledger: ExclusionLedger) -> pd.DataFrame:
    """Полный Свод: пригодные строки плюс отсев, всё с пометкой и причиной."""
    usable = usable.copy()
    usable[USABLE_COLUMN] = 1
    usable[REASON_COLUMN] = None
    usable[STAGE_COLUMN] = None

    excluded = ledger.frame()
    if excluded.empty:
        return usable
    combined = pd.concat([usable, excluded], ignore_index=True, sort=False)
    # Отсев — в конец, чтобы пригодная часть читалась ровно как выходной Свод.
    return combined.sort_values(USABLE_COLUMN, ascending=False, kind="stable").reset_index(drop=True)


FULL_REGISTER_SHEET = "Свод полный"
EXCLUSION_SUMMARY_SHEET = "Причины отсева"

FULL_REGISTER_BANNER = (
    "⚠⚠ ЭТО НЕ ВХОД РАСЧЁТА. Здесь всё, что доехало из источников на момент сборки, "
    "включая непригодное. Строки с «{usable} = 0» в выходной Свод НЕ входят — причина "
    "в колонке «{reason}». Чаще всего это не мусор, а данные, опередившие ПДК: отказы "
    "за этот период ещё не выгружены, и как только ПДК догонит, строка станет пригодной."
).format(usable=USABLE_COLUMN, reason=REASON_COLUMN)


def write_full_register(
    path,
    usable: pd.DataFrame,
    ledger: "ExclusionLedger",
    *,
    columns: Optional[Iterable[str]] = None,
    horizon=None,
) -> None:
    """Записать полный Свод: пригодное плюс отсев, с пометкой и причиной.

    Отдельный ФАЙЛ, а не второй лист выходного Свода — чтобы его нельзя было
    прочитать в расчёт по ошибке, промахнувшись листом.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    combined = build_full_register(usable, ledger)
    ordered = list(columns) if columns else [c for c in combined.columns if not str(c).startswith("_")]
    for marker in (USABLE_COLUMN, REASON_COLUMN, STAGE_COLUMN):
        if marker in ordered:
            ordered.remove(marker)
    # Пометки — ПЕРВЫМИ колонками: их должно быть видно, не прокручивая лист.
    ordered = [USABLE_COLUMN, REASON_COLUMN, STAGE_COLUMN] + [c for c in ordered if c in combined.columns]

    summary = ledger.summary()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        combined[ordered].to_excel(writer, sheet_name=FULL_REGISTER_SHEET, index=False, startrow=1)
        if not summary.empty:
            summary.to_excel(writer, sheet_name=EXCLUSION_SUMMARY_SHEET, index=False)

        sheet = writer.sheets[FULL_REGISTER_SHEET]
        banner = FULL_REGISTER_BANNER
        if horizon is not None:
            banner += f" Горизонт наблюдения ПДК: {pd.Timestamp(horizon).date()}."
        cell = sheet.cell(row=1, column=1, value=banner)
        cell.font = Font(bold=True, color="FF9C0006")
        cell.fill = PatternFill(fill_type="solid", start_color="FFFFC7CE", end_color="FFFFC7CE")
        cell.alignment = Alignment(vertical="center")
        sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(ordered), 1))
        sheet.row_dimensions[1].height = 34
        sheet.freeze_panes = "D3"
        for index, width in enumerate((10, 30, 22), start=1):
            sheet.column_dimensions[get_column_letter(index)].width = width

        if not summary.empty:
            summary_sheet = writer.sheets[EXCLUSION_SUMMARY_SHEET]
            for letter, width in zip("ABCD", (32, 10, 22, 100)):
                summary_sheet.column_dimensions[letter].width = width
            for row in summary_sheet.iter_rows(min_row=2, min_col=4, max_col=4):
                for summary_cell in row:
                    summary_cell.alignment = Alignment(wrap_text=True, vertical="top")


def mark_reason(frame: pd.DataFrame, mask, reason: Reason) -> None:
    """Проставить причину на строках, которые остаются в выборке изменёнными.

    Используется там, где строка не выбрасывается, а правится (цензурирование по
    горизонту, обрезка наработки): в полном Своде должно быть видно, что с ней
    сделали и почему.
    """
    if frame is None or frame.empty or mask is None or not mask.any():
        return
    for column in (REASON_COLUMN, STAGE_COLUMN):
        if column not in frame.columns:
            frame[column] = None
    frame.loc[mask, REASON_COLUMN] = reason.code
    frame.loc[mask, STAGE_COLUMN] = reason.stage


__all__ = [
    "ALL_REASONS",
    "EXCLUSION_SUMMARY_SHEET",
    "FULL_REGISTER_BANNER",
    "FULL_REGISTER_SHEET",
    "write_full_register",
    "BRINE_BORE",
    "CLOSED_AFTER_HORIZON",
    "ExclusionLedger",
    "KDMNU_MARKER",
    "MOUNTED_AFTER_HORIZON",
    "NON_OIL_PURPOSE",
    "REASON_COLUMN",
    "RUNTIME_BEYOND_HORIZON",
    "Reason",
    "STAGE_COLUMN",
    "STALE_OPEN_RUN",
    "SUPERSEDED_OPEN_RUN",
    "USABLE_COLUMN",
    "build_full_register",
    "mark_reason",
]
