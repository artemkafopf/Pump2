"""Реестр ГРП («Свод ГРП») — чтение, выравнивание заголовка, канонический кадр.

Выгрузка приходит одним листом ``FracturingSummary`` на 280 колонок: паспорт
стадии от свойств пласта до фактических параметров трещины. Нас в моделях
интересует прежде всего «был ли ГРП на скважине и когда», но тянуть надо всё —
реестр используется и в других проектах, а второй частичный парсер рядом с
полным это ровно тот разрыв, ради устранения которого затевался перенос.

⚠⚠ **Заголовок сдвинут на одну колонку.** Строка заголовка несёт лишний ведущий
``№``, которого в строках данных нет, поэтому наивное ``header=0`` подписывает
`Скважина` значениями `Месторождение`, `Дата ГРП` — номером стадии и так далее
по всем 280 колонкам, молча. Сдвиг здесь не зашит константой, а **определяется
по якорям** (:func:`detect_column_shift`): подпись принимается только если под
ней действительно лежат коды скважин, разбираемые даты и типы ГТМ. Поменяется
выгрузка — процессор скажет об этом вслух, а не подпишет колонки заново неверно.

⚠ Метки колонок в выгрузке НЕ уникальны: ``Скважина`` встречается дважды
(внутренний числовой ид и код вида ``Vt_8412``), ``№ партии`` — пять раз. Кадр с
неуникальными метками отдаёт по имени DataFrame, а не Series, и любой
downstream-код на этом спотыкается. :func:`load_frac_register` дедуплицирует
метки суффиксом ``␟2``, ``␟3`` … сохраняя исходные в ``frac_column_map``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from analysis.paths import resolve_source_dir

from ..excel_io import read_excel
from ..normalize import normalize_text, normalize_well, parse_date, parse_number

DEFAULT_SOURCE_DIR = resolve_source_dir("frac")
FRAC_SHEET_NAME = "FracturingSummary"
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}

#: Separator used to make duplicate header labels unique (``Скважина␟2``). A unit
#: separator cannot occur in a real header, so the original label is recoverable
#: by splitting on it.
DUPLICATE_LABEL_SEPARATOR = "␟"

#: Header labels of the columns the canonical stage frame is built from.
WELL_CODE_LABEL = "скважина"
FRAC_DATE_LABEL = "дата грп"
GTM_TYPE_LABEL = "вид гтм"

_WELL_CODE_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+_\S+")
_GTM_RE = re.compile(r"грп|пвлг|опз", re.IGNORECASE)


@dataclass(frozen=True)
class FracLayout:
    """Where the header, the numbering row and the data begin, and their offset."""

    header_row: int
    data_start_row: int
    column_shift: int
    n_columns: int


def resolve_frac_files(source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR) -> List[Path]:
    """Resolve one or more files/directories into concrete frac workbooks."""
    if isinstance(source, (str, Path)):
        items: Iterable[Path | str] = [source]
    else:
        items = source

    discovered: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in EXCEL_SUFFIXES and not child.name.startswith("~$"):
                    discovered.append(child.resolve())
        elif path.is_file() and path.suffix.lower() in EXCEL_SUFFIXES and not path.name.startswith("~$"):
            discovered.append(path.resolve())

    unique_files = sorted(set(discovered), key=lambda value: (value.name.lower(), str(value).lower()))
    return [Path(path) for path in unique_files]


def _looks_like_well_codes(values: pd.Series) -> bool:
    text = values.dropna().astype(str).str.strip()
    if text.empty:
        return False
    return float(text.str.match(_WELL_CODE_RE).mean()) >= 0.8


def _looks_like_dates(values: pd.Series) -> bool:
    parsed = values.dropna().apply(parse_date)
    if parsed.empty:
        return False
    valid = parsed.notna()
    if float(valid.mean()) < 0.8:
        return False
    # A stage number column also "parses" once pandas is generous enough; require
    # the values to land in a plausible frac era rather than 1900.
    years = parsed[valid].apply(lambda value: value.year)
    return bool(((years >= 1990) & (years <= 2100)).mean() >= 0.9)


def _looks_like_gtm(values: pd.Series) -> bool:
    text = values.dropna().astype(str)
    if text.empty:
        return False
    return float(text.str.contains(_GTM_RE, na=False).mean()) >= 0.8


_ANCHORS = (
    (WELL_CODE_LABEL, _looks_like_well_codes),
    (FRAC_DATE_LABEL, _looks_like_dates),
    (GTM_TYPE_LABEL, _looks_like_gtm),
)


def find_header_row(raw: pd.DataFrame, *, max_scan: int = 12) -> int:
    """Row index carrying the column labels (the one naming ``Дата ГРП``)."""
    for row_idx in range(min(max_scan, len(raw))):
        labels = {normalize_text(value) for value in raw.iloc[row_idx]}
        if FRAC_DATE_LABEL in labels and WELL_CODE_LABEL in labels:
            return row_idx
    raise ValueError(
        "frac: не найдена строка заголовка — ни в одной из первых "
        f"{max_scan} строк нет одновременно «Дата ГРП» и «Скважина»"
    )


def _is_numbering_row(values: Sequence) -> bool:
    """True for the export's ordinal row (``0, 1, 2 …`` with a blank lead-in)."""
    numbers = [parse_number(value) for value in values]
    present = [number for number in numbers if number is not None]
    if len(present) < 10:
        return False
    expected = list(range(len(present)))
    return [int(number) for number in present] == expected


def detect_column_shift(
    raw: pd.DataFrame,
    header_row: int,
    data_start_row: int,
    *,
    probe_rows: int = 200,
    candidates: Sequence[int] = (0, -1, 1, -2, 2),
) -> int:
    """Offset from a header position to the data column it actually labels.

    Returns ``0`` for a well-formed sheet and ``-1`` for the current export,
    whose header row carries one extra leading ``№`` cell. Every candidate offset
    must satisfy **all** anchors (well codes, frac dates, ГТМ types), so a wrong
    offset cannot be accepted just because one column happens to fit.
    """
    labels = [normalize_text(value) for value in raw.iloc[header_row]]
    probe = raw.iloc[data_start_row : data_start_row + probe_rows]
    if probe.empty:
        raise ValueError("frac: в выгрузке нет строк данных")

    for shift in candidates:
        satisfied = 0
        for label, predicate in _ANCHORS:
            positions = [index for index, value in enumerate(labels) if value == label]
            if not positions:
                continue
            if any(
                0 <= position + shift < probe.shape[1]
                and predicate(probe.iloc[:, position + shift])
                for position in positions
            ):
                satisfied += 1
        if satisfied == len(_ANCHORS):
            return shift

    raise ValueError(
        "frac: не удалось выровнять заголовок с данными — ни один сдвиг из "
        f"{list(candidates)} не даёт одновременно коды скважин, даты ГРП и типы ГТМ. "
        "Формат выгрузки изменился; проверьте лист перед тем, как править сдвиг."
    )


def detect_layout(raw: pd.DataFrame) -> FracLayout:
    """Locate the header, the first data row and the header→data column offset."""
    header_row = find_header_row(raw)
    data_start_row = header_row + 1
    if data_start_row < len(raw) and _is_numbering_row(raw.iloc[data_start_row]):
        data_start_row += 1
    shift = detect_column_shift(raw, header_row, data_start_row)
    n_columns = raw.shape[1] - abs(shift)
    return FracLayout(
        header_row=header_row,
        data_start_row=data_start_row,
        column_shift=shift,
        n_columns=n_columns,
    )


def deduplicate_labels(labels: Sequence[str]) -> List[str]:
    """Make header labels unique, keeping the first occurrence untouched."""
    seen: dict[str, int] = {}
    unique: List[str] = []
    for label in labels:
        text = str(label).strip()
        count = seen.get(text, 0) + 1
        seen[text] = count
        unique.append(text if count == 1 else f"{text}{DUPLICATE_LABEL_SEPARATOR}{count}")
    return unique


def original_label(label: str) -> str:
    """Recover the export's own header from a de-duplicated label."""
    return str(label).split(DUPLICATE_LABEL_SEPARATOR, 1)[0]


def load_frac_register(file_path: Path | str, sheet_name: str = FRAC_SHEET_NAME) -> pd.DataFrame:
    """Read one frac workbook into a flat frame with corrected, unique headers.

    Every source column is preserved — the register feeds other projects too, so
    nothing is dropped here. Lineage columns ``_source_file`` / ``_source_path``
    are appended.
    """
    file_path = Path(file_path)
    raw = read_excel(str(file_path), sheet_name=sheet_name, header=None)
    if raw.empty:
        return pd.DataFrame()

    layout = detect_layout(raw)
    shift = layout.column_shift

    # Header position ``p`` labels data position ``p + shift``. Keep only the
    # positions where both sides exist, so a shifted export loses the header's
    # dangling lead-in cell (and the data's dangling trailing one), never a
    # populated column.
    header_positions = [
        position
        for position in range(raw.shape[1])
        if 0 <= position + shift < raw.shape[1]
    ]
    labels = [str(raw.iloc[layout.header_row, position]).strip() for position in header_positions]
    data_positions = [position + shift for position in header_positions]

    frame = raw.iloc[layout.data_start_row :, data_positions].reset_index(drop=True)
    frame.columns = deduplicate_labels(labels)

    dropped = [
        position for position in range(raw.shape[1]) if position not in set(data_positions)
    ]
    for position in dropped:
        values = raw.iloc[layout.data_start_row :, position]
        if values.notna().any():
            raise ValueError(
                f"frac: выравнивание заголовка (сдвиг {shift}) отбросило бы "
                f"{int(values.notna().sum())} непустых значений в колонке {position}. "
                "Формат изменился — проверьте лист."
            )

    frame = frame.dropna(how="all").reset_index(drop=True)
    frame["_source_file"] = file_path.name
    frame["_source_path"] = str(file_path)
    return frame


def frac_column_map(frame: pd.DataFrame) -> pd.DataFrame:
    """``{storage label -> original export header}`` for the loaded frame."""
    return pd.DataFrame(
        [
            {
                "column_order": order,
                "storage_label": str(column),
                "original_name": original_label(column),
            }
            for order, column in enumerate(frame.columns, start=1)
        ]
    )


# --------------------------------------------------------------------------- #
# Canonical stage frame
# --------------------------------------------------------------------------- #

#: canonical name -> the de-duplicated header it comes from.
#: ``Скважина`` appears twice: the operator's internal numeric id first, the well
#: code (``Vt_8412``) second — the code is the one that joins to the register.
CANONICAL_STAGE_COLUMNS = {
    "well": f"Скважина{DUPLICATE_LABEL_SEPARATOR}2",
    "well_source_id": "Скважина",
    "operator": "Недропользователь",
    "field_name": "Месторождение",
    "cluster": "Куст",
    "layer": "Пласт",
    "saturation": "Насыщение",
    "fund": "Фонд",
    "gtm_type": "Вид ГТМ",
    "contractor": "Исполнитель",
    "frac_date": "Дата ГРП",
    "stage_no": "№ стадии",
    "refrac_previous_date": "Refrac: дата предыдущего ГРП",
}

_DATE_FIELDS = ("frac_date", "refrac_previous_date")
_NUMERIC_FIELDS = ("stage_no",)


def build_frac_stages(frame: pd.DataFrame) -> pd.DataFrame:
    """One typed row per frac **stage**, keyed by ``(well_key, frac_date, stage_no)``."""
    if frame.empty:
        return pd.DataFrame(columns=[*CANONICAL_STAGE_COLUMNS, "well_key", "field"])

    stages = pd.DataFrame(index=frame.index)
    for canonical, source in CANONICAL_STAGE_COLUMNS.items():
        stages[canonical] = frame[source] if source in frame.columns else None

    for column in _DATE_FIELDS:
        stages[column] = stages[column].apply(parse_date)
    for column in _NUMERIC_FIELDS:
        stages[column] = stages[column].apply(parse_number)

    stages["well"] = stages["well"].astype(str).str.strip()
    stages["well_key"] = stages["well"].apply(normalize_well)
    # Field code from the well prefix, exactly as the register derives it — the
    # `Месторождение` cell here is a licence-area name ("Большетирский участок"),
    # not the Vt/Ya/Bt code the rest of the project keys on.
    stages["field"] = stages["well"].str.extract(r"^([A-Za-zА-Яа-яЁё]+)_", expand=False)

    for column in ("gtm_type", "contractor", "layer", "saturation", "fund", "cluster", "field_name"):
        stages[column] = stages[column].apply(
            lambda value: str(value).strip() if pd.notna(value) and str(value).strip() else None
        )

    stages = stages[stages["well_key"].ne("") & stages["frac_date"].notna()].copy()
    # ``_raw_index`` links a stage back to its row in the raw frame, so the raw
    # table can carry the parsed well/date in its ``_meta_*`` columns.
    for column in ("_source_file", "_source_path", "_raw_index"):
        if column in frame.columns:
            stages[column] = frame.loc[stages.index, column]
    return stages.reset_index(drop=True)


def build_frac_wells(stages: pd.DataFrame) -> pd.DataFrame:
    """One row per well: whether it was fracced, how much, and when.

    ``treatments`` counts distinct ``(well, date)`` jobs — a multi-stage МГРП is
    one treatment, not fifteen.
    """
    columns = [
        "well_key", "well", "field", "has_frac", "stages", "treatments",
        "first_frac_date", "last_frac_date", "max_stages_per_treatment", "gtm_types",
    ]
    if stages.empty:
        return pd.DataFrame(columns=columns)

    grouped = stages.groupby("well_key", sort=True)
    records = []
    for well_key, group in grouped:
        treatments = group.drop_duplicates(subset=["well_key", "frac_date"])
        per_treatment = group.groupby("frac_date").size()
        gtm_types = sorted({value for value in group["gtm_type"].dropna().tolist()})
        records.append(
            {
                "well_key": well_key,
                "well": group["well"].iloc[0],
                "field": group["field"].dropna().iloc[0] if group["field"].notna().any() else None,
                "has_frac": 1,
                "stages": int(len(group)),
                "treatments": int(len(treatments)),
                "first_frac_date": group["frac_date"].min(),
                "last_frac_date": group["frac_date"].max(),
                "max_stages_per_treatment": int(per_treatment.max()),
                "gtm_types": "; ".join(gtm_types) if gtm_types else None,
            }
        )
    return pd.DataFrame(records, columns=columns)


def merge_frac_registers(
    source: Path | str | Sequence[Path | str] = DEFAULT_SOURCE_DIR,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read and concatenate every discovered frac workbook."""
    files = resolve_frac_files(source)
    if not files:
        return pd.DataFrame()
    if verbose:
        print(f"Loading {len(files)} frac register file(s)")
    frames = []
    for index, file_path in enumerate(files, start=1):
        frames.append(load_frac_register(file_path))
        if verbose:
            print(f"  [{index}/{len(files)}] {file_path.name}")
    return pd.concat(frames, ignore_index=True, sort=False)


__all__ = [
    "CANONICAL_STAGE_COLUMNS",
    "DEFAULT_SOURCE_DIR",
    "DUPLICATE_LABEL_SEPARATOR",
    "FRAC_SHEET_NAME",
    "FracLayout",
    "build_frac_stages",
    "build_frac_wells",
    "deduplicate_labels",
    "detect_column_shift",
    "detect_layout",
    "find_header_row",
    "frac_column_map",
    "load_frac_register",
    "merge_frac_registers",
    "original_label",
    "resolve_frac_files",
]
