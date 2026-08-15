"""PDK source ingestion and failure extraction."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..column_resolution import build_specs, resolve_columns
from ..config import COLUMN_SYNONYMS, FIELD_NAME_TO_CODE
from ..io import load_dataset_frames
from ..normalize import get_column_by_synonym, normalize_text, normalize_well, parse_date, parse_number
from .enrich import (
    combine_dataset_frames,
    derive_acid_type,
    derive_field_code_from_well,
    derive_runtime_group,
    is_kdmnu_marker,
    is_missing_failed_component,
    is_non_failure_reason_marker,
    is_oil_well_row,
)
from .well_identity import is_rassol_bore


# Canonical PDK -> register header for every field the extractor consumes. Used
# both to rename incoming columns (via the column resolver, so a renamed source
# header still lands in the expected Russian column) and to drive the alias
# assignment loop below.
PDK_ALIAS_TO_HEADER = {
    "field": "Месторождение",
    "cluster": "Куст",
    "well": "Скв.",
    "contractor": "Принадлежность",
    "esp_type": "Тип УЭЦН",
    "descent_depth": "Нспуска",
    "installation_date": "Дата монтажа",
    "failure_date": "Дата остановки",
    "runtime_nno": "Наработка (сут)",
    "failure_reason": "Причина остановки",
    "failure_flag": "Флаг отказа",
    "failure_marker": "Признак отказа",
    "dismantling_date": "Дата демонтажа",
    "failed_node": "Отказавший узел",
    "failed_element": "Отказавший элемент",
    "malfunction_character": "Характер неисправности",
    "esp_failure_reason": "Причина отказа УЭЦН",
    "responsible_party": "Виновная сторона",
    "responsible_person": "Виновное лицо",
    # Free text. Filled on 92 % (note) and 78 % (complications) of PDK rows and,
    # until now, dropped at this boundary -- which is why 321 stops that carry no
    # ``Отказавший узел`` but do describe the fault in words reached the register
    # as "не определено" with nothing left to read.
    "note": "Примечание",
    "workover_complications": "Осложнения при ТКРС, ДЖ, Рекомендации ТКРС",
    "runtime_group": "группа наработок",
    "gabarit": "Габарит",
    "productivity": "Производительность",
}

# The well id is the one column nothing downstream can proceed without.
PDK_REQUIRED_HEADERS = {"Скв."}

# Values of ``Отказавший узел`` that name an operation rather than a failed piece
# of equipment. They are diagnoses of the *job*, not of the pump, so they must not
# promote an otherwise unclassified stop to a confirmed failure.
NON_EQUIPMENT_NODES = {"непроход уэцн"}


def build_pdk_column_specs():
    """Resolver specs keyed by canonical register header, synonyms from config."""
    canonical_to_synonyms = {
        header: list(COLUMN_SYNONYMS.get(alias, []))
        for alias, header in PDK_ALIAS_TO_HEADER.items()
    }
    return build_specs(canonical_to_synonyms, required=PDK_REQUIRED_HEADERS)


PDK_COLUMN_SPECS = build_pdk_column_specs()


def _has_field_prefix(value) -> bool:
    """True when a well id already carries a ``<field>_`` prefix (e.g. ``Ya_724``)."""
    if pd.isna(value) or value is None:
        return False
    return bool(re.match(r"^\s*[A-Za-zА-Яа-яЁё]+_", str(value).strip()))


def _well_tail_needing_prefix(value) -> Optional[str]:
    """Return the id tail to prefix for a well missing its field prefix, else None.

    Handles bare numbers (``5``), numbers with a bore/branch marker (``2ш``) and
    hyphenated ids (``8-4``) -- anything that lacks a leading ``<field>_``. Excel's
    trailing ``.0`` on numeric ids is stripped.
    """
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip()
    if not text or _has_field_prefix(text):
        return None
    text = re.sub(r"\.0+$", "", text)
    return text or None


def _extract_well_prefix(value) -> Optional[str]:
    if pd.isna(value) or value is None:
        return None
    match = re.match(r"^\s*([A-Za-zА-Яа-яЁё]+)_", str(value).strip())
    if not match:
        return None
    prefix = match.group(1).strip()
    return prefix or None


def _clean_pad_value(value):
    """Blank a pad (Куст) value that does not fit the general cluster pattern.

    Pads are cluster identifiers and always carry a number (e.g. ``19``,
    ``Куст 8``). Stray non-numeric values such as a lone ``р`` are data noise
    and are dropped to a blank rather than carried into the register.
    """
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "–"}:
        return None
    if not re.search(r"\d", text):
        return None
    return text


def _normalize_cluster_key(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = re.findall(r"\d+", text)
    if digits:
        return ".".join(digits)
    return normalize_text(text)


def _most_common_non_empty(series: pd.Series) -> Optional[str]:
    cleaned = [str(value).strip() for value in series if pd.notna(value) and str(value).strip()]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return str(counts.index[0]) if not counts.empty else None


def repair_pdk_well_ids(source_df: pd.DataFrame) -> pd.DataFrame:
    """Prefix PDK well IDs that are missing a ``<field>_`` prefix.

    Un-prefixed ids -- bare numbers (``5``), branch markers (``2ш``) and
    hyphenated ids (``8-4``) -- are completed with the field's most common well
    prefix (inferred per field, and per field+cluster when available), so they
    become valid ids like ``Ya_5`` / ``Ya_2ш`` whose field code can be derived.
    """
    repaired = source_df.copy()
    if "Скв." not in repaired.columns:
        return repaired

    repaired["_pdk_well_tail"] = repaired["Скв."].apply(_well_tail_needing_prefix)
    repaired["_pdk_well_prefix"] = repaired["Скв."].apply(_extract_well_prefix)
    repaired["_pdk_cluster_key"] = repaired["Куст"].apply(_normalize_cluster_key) if "Куст" in repaired.columns else ""

    field_cluster_prefix = (
        repaired[repaired["_pdk_well_prefix"].notna()]
        .groupby(["Месторождение", "_pdk_cluster_key"])["_pdk_well_prefix"]
        .agg(_most_common_non_empty)
        .to_dict()
        if "Месторождение" in repaired.columns
        else {}
    )
    field_prefix = (
        repaired[repaired["_pdk_well_prefix"].notna()]
        .groupby("Месторождение")["_pdk_well_prefix"]
        .agg(_most_common_non_empty)
        .to_dict()
        if "Месторождение" in repaired.columns
        else {}
    )

    def repair_row(row: pd.Series):
        current_well = row.get("Скв.")
        tail = row.get("_pdk_well_tail")
        # A column mixing ``None`` and strings surfaces the blanks as ``NaN`` (a
        # truthy float), so guard on the expected string type rather than falsiness.
        if not isinstance(tail, str) or not tail:
            return current_well
        field_value = row.get("Месторождение")
        cluster_key = row.get("_pdk_cluster_key", "")
        prefix = field_cluster_prefix.get((field_value, cluster_key))
        if not prefix:
            prefix = field_prefix.get(field_value)
        if not prefix:
            # Fields whose wells never carry a prefix cannot be learned from the
            # data -- fall back to the explicit field-name -> code map.
            prefix = FIELD_NAME_TO_CODE.get(normalize_text(field_value))
        if not prefix:
            return current_well
        return f"{prefix}_{tail}"

    repaired["Скв."] = repaired.apply(repair_row, axis=1)
    repaired = repaired.drop(columns=["_pdk_well_tail", "_pdk_well_prefix", "_pdk_cluster_key"], errors="ignore")
    return repaired


def pdk_observation_horizon(extracted: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Последняя дата, до которой ПДК наблюдает отказы.

    ⚠⚠ Это **горизонт наблюдения всего регистра**, а не служебная деталь. Отказы
    приходят ТОЛЬКО из ПДК, а живые пуски — из паспорта оборудования, который
    обновляется отдельно. Всё, что лежит правее этой даты, — экспозиция, в которой
    событие наблюдать нечем:

    * пуск, смонтированный после горизонта, разбавляет выборку запуском, чей исход
      принципиально не может быть записан;
    * живой пуск, цензурированный сегодняшним днём, а не горизонтом, получает
      бесплатную наработку без риска отказа.

    И то и другое смещает выживаемость вверх, причём сильнее всего в ранней полосе,
    где как раз и решается вопрос о детской смертности. Отсюда и правило: сборка
    идёт НА ПОСЛЕДНЮЮ ДАТУ ПДК — это одна из причин, по которой построение вообще
    начинается с ПДК.
    """
    if extracted is None or extracted.empty or "failure_date" not in extracted.columns:
        return None
    dates = pd.to_datetime(extracted["failure_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def describe_horizon_completeness(extracted: pd.DataFrame, horizon: pd.Timestamp) -> Optional[str]:
    """Предупреждение, когда ПДК у горизонта заполняется РЕЖЕ обычного.

    ⚠⚠ Сравнивается ИНТЕНСИВНОСТЬ (остановок в сутки), а не месячная сумма.
    Выгрузка почти всегда снимается в середине месяца, поэтому последний месяц
    всегда «недобирает» по сумме — сравнение сумм давало бы тревогу на КАЖДОЙ
    сборке, а предупреждение, которое горит всегда, перестают читать. На выгрузке
    2026-08 такая проверка обвинила июль (24 остановки против 40 типичных), хотя
    его темп 2.0/сут против июньских 2.1/сут — месяц не редкий, он просто неполный,
    и горизонт 2026-07-12 совершенно нормален.

    Настоящая беда выглядит иначе: у самого горизонта записи РЕДЕЮТ, потому что
    отчёты за последние дни ещё не доехали. Это и ловится.
    """
    if extracted is None or extracted.empty or horizon is None:
        return None
    dates = pd.to_datetime(extracted["failure_date"], errors="coerce").dropna()
    if dates.empty:
        return None

    monthly = dates.dt.to_period("M").value_counts().sort_index()
    if len(monthly) < 4:
        return None

    final_period = monthly.index[-1]
    elapsed_days = max((horizon - final_period.to_timestamp()).days + 1, 1)
    final_rate = int(monthly.iloc[-1]) / elapsed_days

    previous = monthly.iloc[-7:-1]
    previous_rates = [
        count / period.days_in_month for period, count in previous.items()
    ]
    typical_rate = float(pd.Series(previous_rates).median()) if previous_rates else 0.0
    if typical_rate <= 0 or final_rate >= typical_rate * 0.6:
        return None
    return (
        f"у горизонта ПДК записи редеют: {final_rate:.2f} остановок/сут за "
        f"{elapsed_days} дн. {final_period} против типичных {typical_rate:.2f}/сут. "
        f"Отчёты за последние дни, похоже, ещё не доехали — горизонт "
        f"{horizon.date()} оптимистичен, задайте --as-of раньше."
    )


def derive_failure_flag(failure_reason, failed_node) -> int:
    """Classify the stop as failure, non-failure, or ambiguous.

    The stop reason is the primary signal. A genuine failure reason marks a
    confirmed failure (``1``) even when the failed component has not been
    diagnosed yet -- this is common for recent events where ``Отказавший узел``
    is filled in only after investigation. ``ГТМ``/``ППР`` markers with no
    failed component are non-failure operational events (``0``). Rows with no
    usable reason fall back to the failed component, and stay ambiguous
    (``-1``) only when neither reason nor failed component is available.

    ``Прочие`` is handled apart from ``ГТМ``/``ППР``. The latter two name a
    planned operation; ``Прочие`` names nothing at all -- it is the catch-all the
    operator picks when the reason box does not fit the event. Treating it like a
    planned pull sent 107 rows that *already carry a diagnosed component* (72 of
    them a cable line) to ``-1``, and ``failures_only`` / ``exclude_ambiguous_events``
    then dropped the diagnosis with the row. When the component is known it is the
    strongest evidence on the row, so it decides.
    """
    failure_reason_text = str(failure_reason).strip() if failure_reason is not None else ""
    failed_node_missing = is_missing_failed_component(failed_node)
    reason_missing = is_missing_failed_component(failure_reason_text)

    if is_non_failure_reason_marker(failure_reason_text):
        if normalize_text(failure_reason_text) in {"гтм", "ппр"}:
            return 0 if failed_node_missing else -1
        # ``Прочие``: the component decides, unless it names an operation rather
        # than a piece of equipment ("непроход УЭЦН" is a run-in obstruction).
        if failed_node_missing or normalize_text(failed_node) in NON_EQUIPMENT_NODES:
            return -1
        return 1

    # A genuine failure reason confirms a failure on its own, regardless of
    # whether the failed component has been recorded yet.
    if not reason_missing:
        return 1

    # No usable reason: fall back to the failed component as evidence.
    if not failed_node_missing:
        return 1
    return -1


def _collapse_same_pump_runs(extracted: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate PDK rows that refer to the same pump run.

    Rows are grouped by normalized well ID plus installation date. When a group
    contains at least one confirmed failure (`failure_flag == 1`), keep the
    latest such failure row. Otherwise keep the latest row in the group and
    carry over the maximum observed runtime as a lower bound for the ongoing run.
    """
    if extracted.empty or "well" not in extracted.columns or "installation_date" not in extracted.columns:
        return extracted

    prepared = extracted.copy()
    prepared["_group_well"] = prepared["well"].apply(normalize_well)
    prepared["_group_installation_date"] = prepared["installation_date"].apply(parse_date)
    valid_group_mask = prepared["_group_well"].ne("") & prepared["_group_installation_date"].notna()
    if not valid_group_mask.any():
        return extracted

    def runtime_sort_value(value):
        parsed = parse_number(value)
        if parsed is None or pd.isna(parsed):
            return float("-inf")
        return float(parsed)

    selected_indexes: list[int] = []
    grouped = prepared[valid_group_mask].groupby(["_group_well", "_group_installation_date"], sort=False)
    for _, group in grouped:
        candidate_group = group.copy()

        # ⚠⚠ Предпочтение подтверждённому отказу — это ТАЙ-БРЕЙК, а не фильтр.
        #
        # Раньше здесь строки с ``failure_flag != 1`` ВЫБРАСЫВАЛИСЬ, если в группе
        # нашёлся хоть один отказ. Пока «Прочие» с заполненным узлом считались
        # неоднозначными, это было почти безобидно. Как только «Прочие» стали
        # подтверждённым отказом, отбор начал уничтожать длинные пуски:
        #
        #   Ya_622, монтаж 2019-07-30 — в ПДК две строки:
        #     · остановка 2019-07-30, ННО 0, «Прочие» / Кабельная линия  → флаг 1
        #     · остановка 2023-04-21, ННО 1359, ГТМ                      → флаг 0
        #   Это РАЗНЫЕ события: происшествие с кабелем в день спуска и плановый
        #   подъём через 3.7 года. Фильтр оставлял первое, перенос максимальной
        #   наработки писал на него 1359, а санация видела «наработка ≫ календарь»
        #   и обрезала её до НУЛЯ. Пуск на 1359 суток превращался в отказ на 0.
        #
        # Направление ошибки — худшее из возможных: она фабрикует события в полосе
        # 0–5 суток, то есть ровно там, где решается вопрос о детской смертности.
        # Дата остановки решает первой; флаг отказа разводит только строки, у
        # которых эта дата совпала.
        if "failure_flag" in candidate_group.columns:
            candidate_group["_sort_is_failure"] = (
                pd.to_numeric(candidate_group["failure_flag"], errors="coerce") == 1
            ).astype(int)
        else:
            candidate_group["_sort_is_failure"] = 0

        if "failure_date" in candidate_group.columns:
            candidate_group["_sort_failure_date"] = candidate_group["failure_date"].apply(parse_date)
        else:
            candidate_group["_sort_failure_date"] = pd.NaT
        if "runtime_nno" in candidate_group.columns:
            candidate_group["_sort_runtime"] = candidate_group["runtime_nno"].apply(runtime_sort_value)
        else:
            candidate_group["_sort_runtime"] = float("-inf")

        # Tie-break on evidence. PDK routinely records one stop twice -- same well,
        # same installation, same failure date -- with the component filled in on
        # one row and left as "Нет" on the other. Date and runtime cannot separate
        # those, so the empty row used to win on sort order alone and the diagnosis
        # was lost. Sorting a diagnosed row last makes it the one kept.
        #
        # This deliberately only breaks *ties*: when the diagnosed row belongs to an
        # earlier, genuinely different event sharing an installation date, its date
        # still decides and the diagnosis is not carried onto a later pull.
        if "failed_node" in candidate_group.columns:
            candidate_group["_sort_has_node"] = (
                ~candidate_group["failed_node"].apply(is_missing_failed_component)
            ).astype(int)
        else:
            candidate_group["_sort_has_node"] = 0

        selected_row = candidate_group.sort_values(
            by=["_sort_failure_date", "_sort_is_failure", "_sort_runtime", "_sort_has_node"],
            ascending=[True, True, True, True],
            na_position="last",
        ).iloc[-1]
        selected_index = int(selected_row.name)

        if "runtime_nno" in prepared.columns:
            max_runtime = pd.to_numeric(group["runtime_nno"], errors="coerce").dropna()
            if not max_runtime.empty:
                prepared.at[selected_index, "runtime_nno"] = float(max_runtime.max())
                if "Наработка (сут)" in prepared.columns:
                    prepared.at[selected_index, "Наработка (сут)"] = float(max_runtime.max())

        selected_indexes.append(selected_index)

    invalid_indexes = prepared.index[~valid_group_mask].tolist()
    kept_indexes = invalid_indexes + selected_indexes
    collapsed = prepared.loc[kept_indexes].copy()
    collapsed = collapsed.drop(
        columns=["_group_well", "_group_installation_date"],
        errors="ignore",
    )
    return collapsed.sort_index().copy()


def extract_new_failures_from_pdk(
    pdk_path,
    cutoff_date: Optional[pd.Timestamp] = None,
    *,
    audit=None,
) -> pd.DataFrame:
    """Extract new failure records from one or more PDK sources."""
    frames = load_dataset_frames(pdk_path, "pdk")
    pdk_df = combine_dataset_frames(frames)
    if pdk_df.empty:
        return pdk_df

    # Resolve incoming headers to canonical register names *before* any of the
    # header-hardcoded logic below (repair_pdk_well_ids, the brine-bore filter,
    # the shared_target_headers extraction) so a renamed source column -- an
    # extra dot, a reordering -- still lands in the column that logic expects.
    mapping_report = resolve_columns(
        [str(column) for column in pdk_df.columns],
        PDK_COLUMN_SPECS,
        dataset="pdk",
        source_file=str(pdk_path),
    )
    mapping_report.raise_if_required_missing()
    mapping_report.warn_unresolved_optional()
    mapping_report.log()
    if audit is not None:
        audit.add_column_mapping(mapping_report)
    pdk_df = mapping_report.apply(pdk_df)

    before_filter_count = len(pdk_df)
    oil_mask = pdk_df.apply(is_oil_well_row, axis=1)
    pdk_df = pdk_df[oil_mask].copy()
    print(f"  PDK oil-well filter kept {len(pdk_df)} of {before_filter_count} row(s)")

    # Belt-and-suspenders: even inside the oil-well class, a brine (``рс``) bore
    # is a distinct non-oil wellbore and must never enter the register. The
    # bore identity is decided by suffix, not by the bare well number.
    if "Скв." in pdk_df.columns:
        rassol_mask = pdk_df["Скв."].apply(is_rassol_bore)
        if rassol_mask.any():
            print(f"  PDK brine-bore (рс) filter dropped {int(rassol_mask.sum())} row(s)")
            pdk_df = pdk_df[~rassol_mask].copy()

    pdk_df = repair_pdk_well_ids(pdk_df)

    shared_target_headers = [
        "Месторождение",
        "Куст",
        "Скв.",
        "Принадлежность",
        "Тип УЭЦН",
        "Нспуска",
        "Дата монтажа",
        "Дата остановки",
        "Наработка (сут)",
        "Причина остановки",
        "Флаг отказа",
        "Дата демонтажа",
        "Признак отказа",
        "Отказавший узел",
        "Отказавший элемент",
        "Характер неисправности",
        "Причина отказа УЭЦН",
        "Виновная сторона",
        "Виновное лицо",
        "Примечание",
        "Осложнения при ТКРС, ДЖ, Рекомендации ТКРС",
        "группа наработок",
        "Габарит",
        "Производительность",
    ]

    extracted = pd.DataFrame(index=pdk_df.index)
    for header in shared_target_headers:
        if header in pdk_df.columns:
            extracted[header] = pdk_df[header].values

    alias_map = PDK_ALIAS_TO_HEADER

    synonym_fallbacks = {
        "field": "field",
        "cluster": "cluster",
        "well": "well",
        "failure_date": "failure_date",
        "runtime_nno": "runtime_nno",
        "failure_reason": "failure_reason",
        "failure_marker": "failure_marker",
        "failed_node": "failed_node",
        "failed_element": "failed_element",
        "malfunction_character": "malfunction_character",
        "responsible_party": "responsible_party",
        "esp_type": "esp_type",
        "contractor": "contractor",
        "installation_date": "installation_date",
        "dismantling_date": "dismantling_date",
    }

    for alias, raw_header in alias_map.items():
        if raw_header in extracted.columns:
            extracted[alias] = extracted[raw_header]
            continue
        fallback_name = synonym_fallbacks.get(alias)
        if fallback_name:
            series = get_column_by_synonym(pdk_df, fallback_name, COLUMN_SYNONYMS)
            if series is not None:
                extracted[alias] = series.values

    for raw_date_col in ["Дата монтажа", "Дата демонтажа"]:
        if raw_date_col in extracted.columns:
            extracted[raw_date_col] = extracted[raw_date_col].apply(parse_date)

    if "installation_date" in extracted.columns:
        extracted["installation_date"] = extracted["installation_date"].apply(parse_date)
    if "dismantling_date" in extracted.columns:
        extracted["dismantling_date"] = extracted["dismantling_date"].apply(parse_date)
    if "failure_date" in extracted.columns:
        extracted["failure_date"] = extracted["failure_date"].apply(parse_date)
        extracted["Дата остановки"] = extracted["failure_date"]
        if cutoff_date is not None:
            # Mode-0 fast path: rows dated on/before the cutoff are excluded. Warn
            # with the count so this silent loss (late-arriving corrections with
            # older dates) becomes visible.
            older = int((extracted["failure_date"].notna() & (extracted["failure_date"] <= cutoff_date)).sum())
            if older:
                print(
                    f"  WARNING: {older} incoming row(s) dated on/before the cutoff "
                    f"{cutoff_date.date()} were excluded (mode 0 fast path). Use "
                    f"update mode 1 to reconcile late-arriving rows."
                )
            extracted = extracted[extracted["failure_date"] > cutoff_date].copy()

    if "well" in extracted.columns:
        extracted["normalized_well"] = extracted["well"].apply(normalize_well)

    if "runtime_nno" in extracted.columns:
        extracted["runtime_nno_raw"] = extracted["runtime_nno"]
        extracted["runtime_nno"] = extracted["runtime_nno"].apply(parse_number)
        extracted["Наработка (сут)"] = extracted["runtime_nno"]

    if "failure_reason" in extracted.columns:
        extracted["failure_reason_raw"] = extracted["failure_reason"]
    if "failure_reason" in extracted.columns:
        failed_node_series = (
            extracted["failed_node"]
            if "failed_node" in extracted.columns
            else pd.Series(index=extracted.index, dtype=object)
        )
        extracted["failure_flag"] = [
            derive_failure_flag(reason, failed_node)
            for reason, failed_node in zip(extracted["failure_reason"], failed_node_series)
        ]
        extracted["Флаг отказа"] = extracted["failure_flag"]
    extracted = _collapse_same_pump_runs(extracted)

    if "well" in extracted.columns:
        extracted["field_derived"] = extracted["well"].apply(derive_field_code_from_well)
    if "field" in extracted.columns:
        extracted["acid_type"] = extracted["field"].apply(derive_acid_type)

    kdmnu_mask = pd.Series(False, index=extracted.index)
    if "contractor" in extracted.columns:
        kdmnu_mask = kdmnu_mask | extracted["contractor"].apply(is_kdmnu_marker)
    if "Принадлежность" in extracted.columns:
        kdmnu_mask = kdmnu_mask | extracted["Принадлежность"].apply(is_kdmnu_marker)
    if "well" in extracted.columns:
        kdmnu_mask = kdmnu_mask | extracted["well"].apply(is_kdmnu_marker)
    if "Скв." in extracted.columns:
        kdmnu_mask = kdmnu_mask | extracted["Скв."].apply(is_kdmnu_marker)
    extracted = extracted[~kdmnu_mask].copy()

    if "field_derived" in extracted.columns:
        existing_field = extracted["field"] if "field" in extracted.columns else pd.Series(index=extracted.index, dtype=object)
        extracted["field"] = extracted["field_derived"].combine_first(existing_field)
        extracted["Месторождение"] = extracted["field"]

    if "runtime_nno" in extracted.columns:
        derived_runtime_group = extracted["runtime_nno"].apply(derive_runtime_group)
        if "группа наработок" in extracted.columns:
            extracted["runtime_group"] = extracted["группа наработок"].where(
                extracted["группа наработок"].notna(),
                derived_runtime_group,
            )
            extracted["группа наработок"] = extracted["runtime_group"]
        else:
            extracted["runtime_group"] = derived_runtime_group
            extracted["группа наработок"] = extracted["runtime_group"]

    for pad_column in ("Куст", "cluster"):
        if pad_column in extracted.columns:
            extracted[pad_column] = extracted[pad_column].apply(_clean_pad_value)

    extracted["source_record_id"] = range(len(extracted))
    extracted["raw_source"] = "PDK"
    return extracted


__all__ = [
    "derive_failure_flag",
    "describe_horizon_completeness",
    "extract_new_failures_from_pdk",
    "pdk_observation_horizon",
]
