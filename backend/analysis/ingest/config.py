"""Configuration for the Свод (ESP failure register) build chain.

Merged from ``db_builder``'s two config modules (``db_builder.config`` and
``db_builder.failure_update.config``), which had drifted into holding overlapping
notions of "where the data is". There is now one table of source folders, and it
is resolved through :mod:`analysis.paths` rather than hardcoded — so an operator
who moves the export drop changes one resolver, not two config files in two
repositories.
"""

from __future__ import annotations

import pandas as pd

from analysis.paths import resolve_source_dir, resolve_telemetry_source_dir


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

# Resolved lazily: importing this module must not depend on the export drop
# existing (tests run without it).
def default_source(kind: str):
    """Default source folder for one ingest ``kind``, via :mod:`analysis.paths`."""
    if kind == "telemetry":
        return str(resolve_telemetry_source_dir())
    return str(resolve_source_dir(kind))


DEFAULT_SOURCE_KINDS = (
    "pdk",
    "artificial_lift",
    "opz",
    "telemetry",
    "techregime",
    "lab",
    "frac",
)

# Filenames that name a specific workbook inside a source folder.
ARTIFICIAL_LIFT_BIG_FILENAME = "WellsArtificialLiftBig.xlsx"
ARTIFICIAL_LIFT_MANUAL_FILENAME = "WellsArtificialLiftManual.xlsx"

DEFAULT_OUTPUT = {
    "workbook": "Отказы свод с анализом.xlsx",
    "audit_xlsx": "svod_audit.xlsx",
    "audit_md": "svod_audit.md",
}

# Lab averaging window: -1 == use every sample for the well (no time window).
LAB_DATA_PROCESSING = {
    "av_window": -1,
}

DEFAULT_TECHREGIME_INTERVAL_DAYS = 7
DEFAULT_TECHREGIME_USE_SQL_IF_AVAILABLE = True
DEFAULT_EVENT_SCOPE = "all"
# Always materialize the event failure-flag column when the incoming target does
# not already carry one, so downstream survival/CatBoost workflows can tell
# failures (1) from censored/non-failure rows (0/-1) without re-deriving.
DEFAULT_INCLUDE_EVENT_FAILURE_FLAG = True
DEFAULT_OPERATING_DATA_PRIMARY_SOURCE = "telemetry"
OPERATING_DATA_PRIMARY_SOURCE_OPTIONS = ("telemetry", "techregime")
EVENT_SCOPE_OPTIONS = ("all", "failures_only")
UPDATE_MODE_LABELS = {
    0: "Update from records newer than the latest target failure date",
    1: "Full update with duplicate check against the target register",
    2: "Rebuild target register from scratch",
}

DEFAULT_OPERATING_TIME_DERIVED_COLUMNS = (
    "Qliq_m3d_30d_mean",
    "Qliq_m3d_prev30d_mean",
    "Qliq_m3d_30d_to_prev30d_mean_ratio",
    "frequency_hz_30d_mean",
    "frequency_hz_prev30d_mean",
    "frequency_hz_30d_to_prev30d_mean_ratio",
    "motor_load_percent_30d_mean",
    "P_intake_atm_30d_mean",
    "P_bhp_atm_30d_mean",
    "Pbubble_atm_30d_mean",
    "Kpod_30d_mean",
    "Kpod_freq_30d_mean",
    "Kpod_30d_days_lt_070",
    "Kpod_30d_share_lt_070",
    "Kpod_30d_days_gt_085",
    "Kpod_30d_share_gt_085",
    "Qliq_m3d_30d_mean_source",
    "frequency_hz_30d_mean_source",
    "motor_load_percent_30d_mean_source",
    "P_intake_atm_30d_mean_source",
    "P_bhp_atm_30d_mean_source",
    "Pbubble_atm_30d_mean_source",
    "Kpod_30d_mean_source",
    "Kpod_freq_30d_mean_source",
)

# Column synonyms for various sources
COLUMN_SYNONYMS = {
    "well": ["скважина", "скв", "скв.", "№ скважины", "номер скважины", "well"],
    "field": ["месторождение", "м/р", "мр", "field"],
    "cluster": ["куст", "кп", "cluster", "pad"],
    "failure_date": ["дата отказа", "дата остановки", "дата аварии", "failure_date"],
    "runtime_nno": ["нно", "наработка", "наработка (сут)", "наработка сут", "наработка, сут", "runtime", "nno"],
    "failure_flag": ["флаг отказа", "failure_flag"],
    "failure_marker": ["признак отказа", "failure_marker"],
    "failure_reason": ["причина отказа", "причина остановки", "reason", "failure_reason"],
    "failed_node": ["отказавший узел", "узел", "failed_node", "node"],
    "failed_element": ["отказавший элемент", "элемент", "element", "failed_element"],
    "malfunction_character": ["характер неисправности", "неисправность", "character", "malfunction"],
    "responsible_party": ["виновная сторона", "ответственный", "responsible", "party"],
    "esp_type": ["тип уэцн", "тип esp", "esp_type", "pump_type"],
    "pump_model": ["модель насоса", "pump_model", "model"],
    "motor_model": ["модель двигателя", "motor_model"],
    "protector": ["протектор", "protector"],
    "contractor": ["принадлежность", "contractor"],
    "launch_date": ["дата запуска", "launch_date"],
    "installation_date": ["дата монтажа", "installation_date"],
    "dismantling_date": ["дата демонтажа", "dismantling_date"],
    # Fields that previously had no synonyms and vanished silently on a rename.
    "descent_depth": ["нспуска", "н спуска", "глубина спуска", "descent_depth"],
    "esp_failure_reason": [
        "причина отказа уэцн",
        "причина отказа эцн",
        "причина отказа уэцн (эцн)",
        "esp_failure_reason",
    ],
    "responsible_person": ["виновное лицо", "ответственное лицо", "responsible_person"],
    # Free text. The complications header is long and has already drifted once
    # ("...ДЖ, Рек" vs "...ДЖ, Рекомендации ТКРС"), and a 0.79 fuzzy score was not
    # enough to resolve it -- so both spellings are named explicitly.
    "note": ["примечание", "примечания", "комментарий пдк", "note"],
    "workover_complications": [
        "осложнения при ткрс, дж, рекомендации ткрс",
        "осложнения при ткрс, дж, рек",
        "осложнения при ткрс",
        "осложнения",
    ],
    "runtime_group": ["группа наработок", "группа наработки", "группа наработок сут", "runtime_group"],
    "gabarit": ["габарит", "габарит уэцн", "gabarit"],
    "productivity": ["производительность", "productivity"],
}

# Lab chemistry columns
LAB_CHEMISTRY_COLUMNS = [
    "Массовая концентрация хлористых солей в нефти, мг/дм³",
    "Массовая доля механических примесей в нефти, мг/дм³",
    "Массовая доля сероводорода, мг/дм³",
    "Cl⁻, мг/л",
    "SO₄²⁻, мг/л",
    "HCO₃⁻, мг/л",
    "Ca₂⁺, мг/л",
    "Mg₂⁺, мг/л",
    "Na + K, мг/л",
    "Общая минерализация, г/л",
    "Катионный коэффициент",
    "pH",
    "Механические примеси (КВЧ), мг/дм³",
]

LAB_COLUMN_RULES = {column: column for column in LAB_CHEMISTRY_COLUMNS}

# OPZ-related target columns
OPZ_COLUMNS = {
    "opz": "ОПЗ",
    "sko_opz": "СКО ОПЗ",
    "gk_opz": "ГК ОПЗ",
    "gko_opz": "ГКО ОПЗ",
    "sko_esp": "СКО УЭЦН",
    "gko_esp": "ГКО УЭЦН",
    "two_stage": "2-х этапка",
}

# OPZ lookback period (days)
OPZ_LOOKBACK_DAYS = 365

# Header detection keywords per source
HEADER_DETECTION_KEYWORDS = {
    "target": ["скв", "дата", "отказ", "нно", "месторождение", "куст", "скважина"],
    "pdk": ["скважина", "отказ", "причина", "нно", "дата", "скв"],
    "big": ["скважина", "дата запуска", "дата отказа", "нно", "уэцн", "скв"],
    "lab": ["скважина", "дата время отбора", "дата анализа", "ph", "состав воды"],
    "techregime": ["скважина", "тип ствола", "дебит", "частота", "рзаб", "скв"],
    "opz": ["скважина", "дата", "опз", "ско", "гко", "скв"],
    "telemetry": ["скважина", "дата", "частота", "загрузка", "приеме насоса"],
    "frac": ["скважина", "дата грп", "пласт", "проппант", "стадии"],
}

SOURCE_FILE_PATTERNS = {
    "pdk": ("*.xlsx", "*.xlsm", "*.xls"),
    "big": ("*.xlsx", "*.xlsm", "*.xls"),
    "lab": ("*.xlsx", "*.xlsm", "*.xls"),
    "opz": ("*.xlsx", "*.xlsm", "*.xls"),
    "techregime": ("*.xlsx", "*.xlsm", "*.xls"),
    "telemetry": ("*.xlsx", "*.xlsm", "*.xls"),
    "frac": ("*.xlsx", "*.xlsm", "*.xls"),
}

# Explicit field-name -> code map for fields whose wells never carry a prefix,
# so the code cannot be learned from the data. Keys are normalized (lowercased,
# ё→е, single-spaced) to match `normalize_text`. Extend as new fields appear.
FIELD_NAME_TO_CODE = {
    "гораздинское": "Gor",
    "гораздинское месторождение": "Gor",
    "гораздинское нефтяное месторождение": "Gor",
    "ичединское": "Ic",
    "ичединское нефтяное месторождение": "Ic",
    "инм": "Ic",
    "янгкм": "Ya",
}

PDK_CLASSIFIER_ALIASES = {
    "well_type_class": [
        "Тип скважины",
        "тип скважины",
        "По назначению",
        "по назначению",
        "Назначение",
        "назначение",
    ],
}

TECHREGIME_COLUMN_RULES = {
    "Тип ствола скв": (None, "Тип ствола скв"),
    "Дебит жидк.": ("Текущий режим работы скважины", "Дебит жидк."),
    "Дебит нефти": ("Текущий режим работы скважины", "Дебит нефти"),
    "Дебит газа": ("Текущий режим работы скважины", "Дебит газа"),
    "Газовый фактор": ("Текущий режим работы скважины", "Газовый фактор"),
    "Кпрод.": ("Текущий режим работы скважины", "Кпрод."),
    "Обводненность": ("Текущий режим работы скважины", "Обводненность"),
    "Плотн. нефти": ("Свойства флюида - PVT", "Плотн. нефти"),
    "Плотн. Воды": ("Свойства флюида - PVT", "Плотн. Воды"),
    "Рпл.": ("Текущий режим работы скважины", "Рпл."),
    "Рзаб": ("Текущий режим работы скважины", "Рзаб"),
    "Частота": ("Текущий режим работы скважины", "Частота"),
    "Загр, Двиг,": ("Текущий режим работы скважины", "Загр. Двиг."),
    "Ртр": ("Текущий режим работы скважины", "Ртр"),
    "Рзт": ("Текущий режим работы скважины", "Рзт"),
    "Рлин": ("Текущий режим работы скважины", "Рлин"),
}

BIG_COLUMN_RULES = {
    "Габарит УЭЦН": ("Насос (50Гц)", "Габарит УЭЦН"),
    "Длина УЭЦН/метр": ("Насос (50Гц)", "Длина УЭЦН/метр"),
    "Работа в кривизне": ("Насос (50Гц)", "Работа в кривизне"),
    "Ном. Произв. м₃/сут": ("Насос (50Гц)", "Ном. Произв. м3/сут"),
    "Ном.напор (50Гц)": ("Насос (50Гц)", "Ном.напор (50Гц)"),
    "Номинальная частота, Гц": ("Насос (50Гц)", "Номинальная частота, Гц"),
    "Кол.ступеней": ("Насос (50Гц)", "Кол.ступеней"),
    "Группа исполнения УЭЦН": ("Насос (50Гц)", "Группа исполнения УЭЦН"),
    "Коррозионная стойкость": ("Насос (50Гц)", "Коррозионная стойкость"),
    "Мощность, кВт": ("ПЭД", "Мощность, кВт"),
    "Ном. напряжение/В": ("ПЭД", "Ном. напряжение/В"),
    "Ном. ток/ A": ("ПЭД", "Ном. ток/ A"),
    "Ток x.x": ("ПЭД", "Ток x.x"),
    "Глубина спуска УЭЦН, по НКТ": ("ФА/НКТ", "Глубина спуска УЭЦН, по НКТ"),
    "ВГ, м": ("ФА/НКТ", "ВГ, м"),
    "Диаметр НКТ": ("ФА/НКТ", "Диаметр НКТ"),
    "Марка НКТ": ("ФА/НКТ", "Марка НКТ"),
}

# --------------------------------------------------------------------------- #
# Свод (failure register) canonical schema -- used to build the register from
# scratch when no target workbook is supplied. The column order below is the
# register's own layout: identity/run core, cause taxonomy, then equipment (Big),
# derived, operating regime (TechRegime), stimulation (OPZ), lab chemistry,
# reference telemetry, event flag.
# --------------------------------------------------------------------------- #
SVOD_SHEET_NAME = "Свод"
SVOD_EVENT_FLAG_COLUMN = "Failure Flag"
# Free-text provenance for values the builder filled in (e.g. nominal params
# derived from the ESP base, chemistry taken as a pad mean).
SVOD_COMMENT_COLUMN = "Комментарий"
# ⚠⚠ `event == 0` ≠ «работает»: под нулём лежат ещё и подъёмы по ГТМ и ППР.
# Работающий пуск определяется ПУСТОЙ датой окончания, и только так.
SVOD_RUNNING_COLUMN = "Работает"

SVOD_CORE_COLUMNS = [
    "Скв.",
    "Месторождение",
    "Куст",
    "Принадлежность",
    "Кислый/Некислый",
    "Тип УЭЦН",
    "Нспуска",
    "Дата монтажа",
    "Дата запуска",
    "Дата остановки",
    "Дата демонтажа",
    "Наработка (сут)",
    "группа наработок",
    "Причина остановки",
    "Признак отказа",
    "Флаг отказа",
    "Работает",
    "Отказавший узел",
    "Отказавший элемент",
    "Характер неисправности",
    "Причина отказа УЭЦН",
    "Виновная сторона",
    "Виновное лицо",
    # Free text carried through from PDK: the only record of the fault for stops
    # where ``Отказавший узел`` was never filled in.
    "Примечание",
    "Осложнения при ТКРС, ДЖ, Рекомендации ТКРС",
    "Габарит",
    "Производительность",
]

# Cause taxonomy, derived by ``svod.causes`` — see that module for the rules.
SVOD_CAUSE_COLUMNS = [
    "группа_причины",
    "спорная_зона",
    "узел_категория",
]

# Computed register columns filled after enrichment.
SVOD_DERIVED_COLUMNS = [
    "Дельта Дебита Ж и номинала, м3/сут",
    "ГЖФ",
]

# Fracturing (ГРП) columns joined from the frac register.
SVOD_FRAC_COLUMNS = [
    "ГРП",
    "ГРП стадий",
    "Дата последнего ГРП",
    "ГРП до монтажа",
]

# --------------------------------------------------------------------------- #
# Reference-only operating block.
#
# The customer asked to keep per-run operating averages visible to whoever reads
# the register, but they must not be mistaken for input to the *dynamic* forecast
# — that one builds its own monthly frame straight from the telemetry store.
#
# ⚠⚠ They are marked, NOT renamed. Six Pump2 modules read these headers verbatim
# out of the register (`production_risk.freq_bins.SVOD_COVARIATES`,
# `svod_nno_decomposition`, `svod_ttf_by_design`, `svod_ttf_vs_ql`,
# `esp_survival.data`, `production_risk.crosswalk`), so a `СПР.` prefix would
# silently empty their covariates instead of protecting anything. The marking is
# therefore carried by presentation and by a machine-readable legend sheet:
#   * the block is emitted contiguously,
#   * its header cells get a distinct fill plus an explanatory cell note,
#   * every column is listed on the «Справка о колонках» sheet with its source
#     and its ``справочная`` / ``расчётная`` role.
# --------------------------------------------------------------------------- #
SVOD_LEGEND_SHEET_NAME = "Справка о колонках"

#: Register columns that are operating snapshots, not run identity or design.
SVOD_REFERENCE_COLUMNS = list(TECHREGIME_COLUMN_RULES.keys()) + ["ГЖФ"]

SVOD_REFERENCE_NOTE = (
    "Справочное значение: усреднённый режим по скважине на дату остановки. "
    "В динамическом прогнозе НЕ участвует — там свой месячный кадр из телеметрии."
)

#: Roles shown on the legend sheet.
SVOD_COLUMN_ROLES = {
    "идентификация": SVOD_CORE_COLUMNS,
    "причина (разметка)": SVOD_CAUSE_COLUMNS,
    "оборудование": list(BIG_COLUMN_RULES.keys()),
    "расчётная": SVOD_DERIVED_COLUMNS[:1],
    "справочная": SVOD_REFERENCE_COLUMNS,
    "ОПЗ": list(OPZ_COLUMNS.values()),
    "ГРП": SVOD_FRAC_COLUMNS,
    "химия": LAB_CHEMISTRY_COLUMNS,
    "провенанс": [SVOD_COMMENT_COLUMN],
}


def build_svod_target_columns() -> list:
    """Assemble the full, de-duplicated Свод column schema in register order.

    The schema is composed from the same rule tables the workflow uses to
    populate the register, so operating-regime (TechRegime) and equipment (Big)
    columns are always emitted even when their extraction is skipped -- they are
    simply left empty.
    """
    columns: list = []

    def _add(names):
        for name in names:
            if name not in columns:
                columns.append(name)

    _add(SVOD_CORE_COLUMNS)
    _add(SVOD_CAUSE_COLUMNS)
    _add(BIG_COLUMN_RULES.keys())
    _add(SVOD_DERIVED_COLUMNS)
    _add(TECHREGIME_COLUMN_RULES.keys())
    _add(OPZ_COLUMNS.values())
    _add(SVOD_FRAC_COLUMNS)
    _add(LAB_CHEMISTRY_COLUMNS)
    # The event flag lives in the Russian `Флаг отказа` column (part of
    # SVOD_CORE_COLUMNS); the English `Failure Flag` would duplicate it, so it is
    # intentionally not added here.
    _add([SVOD_COMMENT_COLUMN])
    return columns


# Matching rules
MATCHING_RULES = {
    "exact_well_exact_date": True,
    "well_date_within_days": 2,
}

# Kept for the mode-0 fast path only (records newer than this date).
NEW_FAILURE_CUTOFF = pd.Timestamp("2025-08-28")
