"""Построчный диф двух Сводов: что изменилось при пересборке.

Приёмка требует именно построчного отчёта, а не итога: «сколько строк
добавилось, у скольких появился диагноз, у скольких изменилась группа».

⚠⚠ Стыковка ТОЛЬКО по ``(скважина, дата монтажа)``. Колонка ``run`` — позиционный
индекс, у разных кадров разный; на этом однажды 669 пусков из 2308 получили чужой
уровень. Позиция строки в файле здесь не значит ничего.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..config import SVOD_SHEET_NAME
from ..normalize import normalize_well, parse_date
from .causes import (
    CAUSE_GROUP_COLUMN,
    NODE_CATEGORY_COLUMN,
    classify_cause_group,
    classify_failure_row,
    has_node_diagnosis,
)

KEY_COLUMNS = ("well_key", "mount_date")

#: Fields whose value changes are worth listing one by one.
TRACKED_FIELDS = (
    "Дата остановки",
    "Дата демонтажа",
    "Наработка (сут)",
    "Причина остановки",
    "Отказавший узел",
    "Отказавший элемент",
    "Характер неисправности",
    "Причина отказа УЭЦН",
    "Кислый/Некислый",
    "Тип УЭЦН",
    "Ном. Произв. м₃/сут",
)


@dataclass
class SvodDiff:
    """Row-level comparison of two registers."""

    added: pd.DataFrame = field(default_factory=pd.DataFrame)
    removed: pd.DataFrame = field(default_factory=pd.DataFrame)
    gained_diagnosis: pd.DataFrame = field(default_factory=pd.DataFrame)
    lost_diagnosis: pd.DataFrame = field(default_factory=pd.DataFrame)
    changed_group: pd.DataFrame = field(default_factory=pd.DataFrame)
    changed_fields: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "added": self.added,
            "removed": self.removed,
            "gained_diagnosis": self.gained_diagnosis,
            "lost_diagnosis": self.lost_diagnosis,
            "changed_group": self.changed_group,
            "changed_fields": self.changed_fields,
        }

    def describe(self) -> str:
        lines = ["Диф против прежнего Свода:"]
        for key, value in self.summary.items():
            lines.append(f"  {key:44s} {value}")
        return "\n".join(lines)


def load_register(path: Path | str, sheet_name: str = SVOD_SHEET_NAME) -> pd.DataFrame:
    """Read a Свод sheet and attach the join key plus derived cause labels.

    The old register carries no cause columns, so they are derived here with the
    same rules the builder now applies — otherwise "группа изменилась" could not
    be answered for rows that existed before.
    """
    frame = pd.read_excel(path, sheet_name=sheet_name, header=0)
    frame["well_key"] = frame["Скв."].map(normalize_well)
    frame["mount_date"] = frame["Дата монтажа"].map(parse_date)

    if CAUSE_GROUP_COLUMN not in frame.columns:
        frame[CAUSE_GROUP_COLUMN] = [
            classify_cause_group(
                row.get("Причина отказа УЭЦН"),
                row.get("Примечание"),
                row.get("Осложнения при ТКРС, ДЖ, Рекомендации ТКРС"),
            )
            for _, row in frame.iterrows()
        ]
    if NODE_CATEGORY_COLUMN not in frame.columns:
        frame[NODE_CATEGORY_COLUMN] = [
            classify_failure_row(row, default=None) for _, row in frame.iterrows()
        ]
    frame["_has_diagnosis"] = [
        has_node_diagnosis(row.get("Отказавший узел"), row.get("Отказавший элемент"))
        for _, row in frame.iterrows()
    ]
    return frame


def _index(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(well_key, mount_date)``, diagnosed rows winning ties.

    A full register carries several rows for one run — a ГТМ pull beside the
    failure row. Sorting the diagnosed row first means the deduplication keeps
    evidence rather than whichever row happened to come first.
    """
    keyed = frame[frame["well_key"].ne("") & frame["mount_date"].notna()].copy()
    keyed = keyed.sort_values("_has_diagnosis", ascending=False, kind="stable")
    return keyed.drop_duplicates(subset=list(KEY_COLUMNS), keep="first").set_index(list(KEY_COLUMNS))


def _identity(frame: pd.DataFrame, index) -> pd.DataFrame:
    columns = [c for c in ("Скв.", "Месторождение", "Дата монтажа", "Дата остановки", "Наработка (сут)") if c in frame.columns]
    return frame.loc[index, columns].reset_index()


def compare_registers(
    new_path: Path | str,
    old_path: Path | str,
    *,
    tracked_fields: Sequence[str] = TRACKED_FIELDS,
) -> SvodDiff:
    """Compare a freshly built register against the register in use."""
    new_frame = load_register(new_path)
    old_frame = load_register(old_path)
    new_index = _index(new_frame)
    old_index = _index(old_frame)

    new_keys = new_index.index
    old_keys = old_index.index
    added_keys = new_keys.difference(old_keys)
    removed_keys = old_keys.difference(new_keys)
    common_keys = new_keys.intersection(old_keys)

    diff = SvodDiff()
    diff.added = _identity(new_index, added_keys)
    diff.removed = _identity(old_index, removed_keys)

    new_common = new_index.loc[common_keys]
    old_common = old_index.loc[common_keys]

    gained = common_keys[new_common["_has_diagnosis"].to_numpy() & ~old_common["_has_diagnosis"].to_numpy()]
    lost = common_keys[~new_common["_has_diagnosis"].to_numpy() & old_common["_has_diagnosis"].to_numpy()]
    diff.gained_diagnosis = pd.DataFrame(
        {
            "Скв.": new_index.loc[gained, "Скв."].to_numpy(),
            "Дата монтажа": new_index.loc[gained, "Дата монтажа"].to_numpy(),
            "узел_новый": new_index.loc[gained, "Отказавший узел"].to_numpy(),
            "категория_новая": new_index.loc[gained, NODE_CATEGORY_COLUMN].to_numpy(),
        }
    )
    diff.lost_diagnosis = pd.DataFrame(
        {
            "Скв.": old_index.loc[lost, "Скв."].to_numpy(),
            "Дата монтажа": old_index.loc[lost, "Дата монтажа"].to_numpy(),
            "узел_прежний": old_index.loc[lost, "Отказавший узел"].to_numpy(),
        }
    )

    new_group = new_common[CAUSE_GROUP_COLUMN].astype("string").fillna("")
    old_group = old_common[CAUSE_GROUP_COLUMN].astype("string").fillna("")
    changed_group_mask = (new_group.to_numpy() != old_group.to_numpy())
    changed_keys = common_keys[changed_group_mask]
    diff.changed_group = pd.DataFrame(
        {
            "Скв.": new_index.loc[changed_keys, "Скв."].to_numpy(),
            "Дата монтажа": new_index.loc[changed_keys, "Дата монтажа"].to_numpy(),
            "группа_прежняя": old_group.to_numpy()[changed_group_mask],
            "группа_новая": new_group.to_numpy()[changed_group_mask],
            "причина": new_index.loc[changed_keys, "Причина отказа УЭЦН"].to_numpy(),
        }
    )

    changes: list[dict] = []
    for column in tracked_fields:
        if column not in new_index.columns or column not in old_index.columns:
            continue
        new_values = new_common[column]
        old_values = old_common[column]
        differing = _values_differ(old_values, new_values)
        for key in common_keys[differing.to_numpy()]:
            changes.append(
                {
                    "Скв.": new_index.loc[key, "Скв."],
                    "Дата монтажа": new_index.loc[key, "Дата монтажа"],
                    "поле": column,
                    "было": old_index.loc[key, column],
                    "стало": new_index.loc[key, column],
                }
            )
    diff.changed_fields = pd.DataFrame(changes)

    diff.summary = {
        "строк в новом регистре": len(new_index),
        "строк в прежнем регистре": len(old_index),
        "добавилось пусков": len(added_keys),
        "исчезло пусков": len(removed_keys),
        "совпало по ключу": len(common_keys),
        "появился диагноз": len(gained),
        "пропал диагноз": len(lost),
        "изменилась группа причины": len(changed_keys),
        "изменений в отслеживаемых полях": len(diff.changed_fields),
    }
    return diff


def _values_differ(old: pd.Series, new: pd.Series) -> pd.Series:
    """Elementwise inequality tolerant of numeric/date/text type drift."""
    old_numeric = pd.to_numeric(old, errors="coerce")
    new_numeric = pd.to_numeric(new, errors="coerce")
    both_numeric = old_numeric.notna() & new_numeric.notna()

    # Every tracked field is probed as a date, including the text ones, so pandas
    # cannot infer a single format and says so once per column. That is expected
    # here — the probe is deliberately heterogeneous — and the warning otherwise
    # buries the diff summary it is printed alongside.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        old_dates = pd.to_datetime(old, errors="coerce")
        new_dates = pd.to_datetime(new, errors="coerce")
    both_dates = old_dates.notna() & new_dates.notna() & ~both_numeric

    old_text = old.astype("string").str.strip().str.casefold().fillna("")
    new_text = new.astype("string").str.strip().str.casefold().fillna("")

    differ = old_text.to_numpy() != new_text.to_numpy()
    differ = pd.Series(differ, index=old.index)
    differ.loc[both_numeric] = (old_numeric[both_numeric] - new_numeric[both_numeric]).abs() > 1e-9
    differ.loc[both_dates] = old_dates[both_dates].dt.normalize() != new_dates[both_dates].dt.normalize()
    return differ


__all__ = ["KEY_COLUMNS", "TRACKED_FIELDS", "SvodDiff", "compare_registers", "load_register"]
