"""Разметка причин отказа и категории отказавшего узла — в самом Своде.

До переноса разметка жила в аналитическом скрипте
(``results/reporting/models/v7_pdk_causes.py``), а классификатор узла — в
``vt_failure/data.py``. Обе теперь здесь: Свод приезжает в расчёт уже
размеченным, и расчётчику нечего доделывать руками.

Критерий деления — **заказчика**: *в эксплуатацию уходит всё, на что мы не можем
повлиять на начальном этапе.*

===========================  ==============================================
группа                       что входит
===========================  ==============================================
``монтаж/завод``             брак монтажа, подбора, комплектации, заводской
                             брак узла, брак ремонта, скрытый дефект,
                             конструктивный недостаток, **механическое
                             повреждение кабеля**
``скважина/организация``     брак подготовки скважины, конструкция скважины,
                             кривизна, смещение и негерметичность ЭК,
                             организационные причины любой стороны,
                             отсутствие оборудования, необоснованный подъём
``эксплуатация``             засорение, солеотложения, H₂S, коррозия,
                             выработка ресурса, старение изоляции, брак
                             эксплуатации, **необеспечен приток, влияние
                             газа**
``не указана``               причина пуста или ГТМ/ППР без текста
``прочее``                   причина есть, но ни одно правило не сработало
===========================  ==============================================

⚠ В ``прочее`` осознанно остаются три формулировки — **«Полет» (4), «Разобран» (2),
«Перенос» (2)**. Они называют не причину, а то, ЧТО произошло с оборудованием;
«полёт» бывает и следствием качества резьбы (монтаж), и следствием коррозии НКТ
(эксплуатация). Догадка здесь двигала бы β без основания, поэтому строки честно
лежат отдельной группой, а не разносятся по вкусу.

⚠ **«Механическое повреждение кабеля»** (145 строк ПДК, медиана 0 суток, 74 % в
первом месяце) раньше числилось эксплуатацией — это монтаж. Перенесено.

⚠ **«Необеспечен приток»** и **«влияние газа»** идут в эксплуатацию по критерию
заказчика (повлиять на старте нельзя), хотя это пласт. Проверено: без этого β
газовой группы переворачивалась 1.085 → 0.931.

⚠⚠ **Граница «брак \\<узел\\>» против «брак монтажа» КОММЕРЧЕСКАЯ** — она решает,
кто платит, а не что сломалось. Поэтому жёсткого отнесения нет: такие строки
получают ``спорная_зона = 1``, расчёт гоняется вилкой, и вилка двигает β на 0.07.

⛔ **«Виновная сторона» ковариатой не годится**: 863 из 1484 записаны на самого
оператора — это результат согласования, а не измерение. Колонка в Своде
сохраняется, но в разметку не входит.

⚠ Разметка известна только ПОСЛЕ подъёма ⇒ законна как **фильтр популяции**, не
как ковариата прогноза.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from ..normalize import normalize_text

# --------------------------------------------------------------------------- #
# Группы причин
# --------------------------------------------------------------------------- #

GROUP_BUILD = "монтаж/завод"
GROUP_WELL = "скважина/организация"
GROUP_OPERATION = "эксплуатация"
GROUP_UNSPECIFIED = "не указана"
GROUP_OTHER = "прочее"

CAUSE_GROUPS = (GROUP_BUILD, GROUP_WELL, GROUP_OPERATION, GROUP_UNSPECIFIED, GROUP_OTHER)

#: Значения «Причина отказа УЭЦН», которые причиной не являются: пусто, плановая
#: операция, «на дорасследовании». Нормализованы под :func:`normalize_text`.
UNSPECIFIED_REASONS = frozenset({
    "", "nan", "none", "нет", "-", "–", "н/д",
    "гтм", "гтм ппр", "ппр",
    "дорасследование", "опи",
    "смотреть с цднг и огт", "смотреть с ткрс и огт",
})

#: Подстроки причины → группа. Порядок важен: первое совпадение выигрывает,
#: поэтому более узкие правила стоят выше более общих.
#: Проверено на всех 101 различных значениях «Причина отказа УЭЦН» в обеих
#: выгрузках ПДК (архив 2018-2022 + текущая 2023-2026).
CAUSE_RULES: tuple[tuple[str, str], ...] = (
    # --- монтаж/завод -----------------------------------------------------
    # «Механическое повреждение кабеля» — перенесено сюда из эксплуатации.
    ("механическое повреждение кабел", GROUP_BUILD),
    ("брак монтажа", GROUP_BUILD),
    ("брак подбора", GROUP_BUILD),
    ("брак комплектации", GROUP_BUILD),
    ("брак ремонта", GROUP_BUILD),
    ("скрытый дефект", GROUP_BUILD),
    ("скрытый деффект", GROUP_BUILD),          # опечатка встречается в ПДК
    ("конструктивный недостаток", GROUP_BUILD),
    # «брак <узел>» — заводской брак конкретного узла (спорная зона, см. ниже)
    ("брак пэд", GROUP_BUILD),
    ("брак кабел", GROUP_BUILD),
    ("брак гидрозащиты", GROUP_BUILD),
    ("брак гз", GROUP_BUILD),
    ("брак гс", GROUP_BUILD),
    ("брак эцн", GROUP_BUILD),
    ("брак нкт", GROUP_BUILD),
    ("брак удлинителя", GROUP_BUILD),
    ("брак сростка", GROUP_BUILD),
    ("брак патрубка", GROUP_BUILD),
    ("брак модуля", GROUP_BUILD),
    ("брак дополнительного", GROUP_BUILD),
    ("дополнительное оборудование", GROUP_BUILD),
    ("брак тмсп", GROUP_BUILD),
    ("брак цф", GROUP_BUILD),
    # --- скважина/организация ---------------------------------------------
    ("подготовки скважины", GROUP_WELL),
    ("конструкция скважины", GROUP_WELL),
    ("работа в кривизне", GROUP_WELL),
    ("смещение эксплуатационной", GROUP_WELL),
    ("смещение эк", GROUP_WELL),
    ("негерметичность эксплуатационной", GROUP_WELL),
    ("негерметичность эк", GROUP_WELL),
    ("организационные", GROUP_WELL),
    ("отсутствие необходимого", GROUP_WELL),
    ("невозвратное оборудование", GROUP_WELL),
    ("необоснованный подъем", GROUP_WELL),
    ("не проход в эк", GROUP_WELL),
    ("непроход в эк", GROUP_WELL),
    ("нестабильное энергоснабжение", GROUP_WELL),
    ("нефтегазопроявление", GROUP_WELL),
    # Подрядчик ТКРС — сторона, а не наш узел: и его оборудование, и качество его
    # работ отвечают за подъём так же, как «организационные причины ТКРС».
    ("оборудования ткрс", GROUP_WELL),
    ("операций ткрс", GROUP_WELL),
    # --- эксплуатация ------------------------------------------------------
    # ⚠ пласт, но по критерию заказчика — эксплуатация
    ("необеспечен приток", GROUP_OPERATION),
    ("не обеспечен приток", GROUP_OPERATION),
    ("влияние газа", GROUP_OPERATION),
    ("засорение", GROUP_OPERATION),
    # Единственное и множественное число встречаются оба («Солеотложение» ×1,
    # «Солеотложения» ×180) — правило пишется по общему корню.
    ("солеотложени", GROUP_OPERATION),
    ("парафиноотложени", GROUP_OPERATION),
    ("гидратоотложени", GROUP_OPERATION),
    ("сероводород", GROUP_OPERATION),
    ("коррози", GROUP_OPERATION),
    ("выработка ресурса", GROUP_OPERATION),
    # «Старение изоляции кабеля» и «Старение обмотки ПЭД» — одно и то же старение.
    ("старение", GROUP_OPERATION),
    ("брак эксплуатации", GROUP_OPERATION),
    ("негерметичность нкт", GROUP_OPERATION),
    ("негерметичность подвески", GROUP_OPERATION),
    ("прочие по вине оборудования", GROUP_OPERATION),
    ("прочее по вине оборудования", GROUP_OPERATION),
    ("прочие по оборудованию", GROUP_OPERATION),
)

#: Причины, чьё отнесение решает, КТО ПЛАТИТ, а не что сломалось. Расчёт гоняется
#: вилкой по этому флагу; вилка двигает β на 0.07, поэтому жёстко относить их в
#: одну группу нельзя.
DISPUTED_RULES: tuple[str, ...] = (
    "брак пэд", "брак кабел", "брак гидрозащиты", "брак гз", "брак гс",
    "брак эцн", "брак нкт", "брак удлинителя", "брак сростка", "брак патрубка",
    "брак модуля", "брак дополнительного", "брак тмсп", "брак цф",
    "брак ремонта", "скрытый дефект", "скрытый деффект",
    "конструктивный недостаток",
    "прочие по вине оборудования", "прочее по вине оборудования",
    "прочие по оборудованию",
)

#: Свободный текст читается ТОЛЬКО у строк без структурной причины — там же лежат
#: «непроход», «уронили», «полетел кабель при СПО».
TEXT_RULES: tuple[tuple[str, str], ...] = (
    ("неквалифицированн", GROUP_BUILD),
    ("дефект монтажа", GROUP_BUILD),
    ("при монтаже", GROUP_BUILD),
    ("заводск", GROUP_BUILD),
    ("брак", GROUP_BUILD),
    ("непроход", GROUP_WELL),
    ("не прошел", GROUP_WELL),
    ("уронил", GROUP_WELL),
    ("прихват", GROUP_WELL),
    ("посадка", GROUP_WELL),
    ("при спуске", GROUP_WELL),
    ("при подъеме", GROUP_WELL),
)


def classify_cause_group(reason, note=None, complications=None) -> str:
    """Группа причины по «Причина отказа УЭЦН», с добором из свободного текста.

    ``note`` / ``complications`` — «Примечание» и «Осложнения при ТКРС»; они
    читаются только когда структурной причины нет, иначе редкое слово в
    примечании перебило бы записанную причину.
    """
    normalized = normalize_text(reason)
    if normalized in UNSPECIFIED_REASONS:
        text = f"{normalize_text(note)} {normalize_text(complications)}".strip()
        for marker, group in TEXT_RULES:
            if marker in text:
                return group
        return GROUP_UNSPECIFIED

    for marker, group in CAUSE_RULES:
        if marker in normalized:
            return group
    return GROUP_OTHER


def is_disputed_cause(reason) -> int:
    """1 для причин, чьё отнесение — коммерческая граница, а не техническая."""
    normalized = normalize_text(reason)
    if normalized in UNSPECIFIED_REASONS:
        return 0
    return int(any(marker in normalized for marker in DISPUTED_RULES))


def unmatched_causes(reasons: Iterable) -> pd.Series:
    """Значения причины, попавшие в ``прочее`` — с частотами.

    Новая формулировка в ПДК не должна растворяться в общей куче: сборщик
    печатает этот список, чтобы правило можно было дописать осознанно.
    """
    values = [str(value).strip() for value in reasons if str(value).strip()]
    if not values:
        return pd.Series(dtype="int64")
    frame = pd.Series(values)
    other = frame[[classify_cause_group(value) == GROUP_OTHER for value in frame]]
    return other.value_counts()


# --------------------------------------------------------------------------- #
# Категория отказавшего узла
# --------------------------------------------------------------------------- #

#: Те же 7 категорий, что несёт ``vt_failure.config.FAILURE_CATEGORIES``
#: (равенство закреплено тестом), плюс пустая — когда диагноза нет.
NODE_CATEGORIES = (
    "КЛ (R-0)",
    "ПЭД (R-0)",
    "Слом вала",
    "Засорение РО",
    "НКТ",
    "Износ РО",
    "Износ/негермет.гидрозащиты",
)

_EMPTY_NODE_VALUES = frozenset({"", "нет", "-", "—", "н/д", "nan", "<na>", "none", "not"})

_CABLE_ELEMENTS = frozenset({
    "кабельный удлинитель", "основная длина", "кабельный сросток",
    "кабельная муфта", "термовставка", "сальниковая разделка",
})
_MOTOR_ELEMENTS = frozenset({
    "статор с обмоткой", "верхнее лобовое", "ротор", "выводные концы", "колодка токоввода",
})
_NKT_NODES = frozenset({
    "нкт", "клапан сливной", "подвесной патрубок", "клапан обратный",
    "мандрель", "переводник",
})
_CLOG_WORDS = ("засорен", "твердые отложения", "солеотложени")
_WEAR_WORDS = ("разрушен", "износ", "осевой", "пар трения", "радиальный", "промыв", "трещин", "эрозион")
_PUMP_NODES = frozenset({"эцн", "газосепаратор", "диспергатор", "входной модуль"})


def _text(value) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def has_node_diagnosis(failed_node, failed_element=None) -> bool:
    """True когда в строке вообще записан диагноз (узел или элемент)."""
    node = _text(failed_node)
    element = _text(failed_element)
    return not (node in _EMPTY_NODE_VALUES and element in _EMPTY_NODE_VALUES)


def classify_node_category(
    failed_node,
    failed_element=None,
    malfunction_character=None,
    esp_failure_reason=None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Категория отказавшего узла по четырём полям Свода.

    ⚠⚠ ``default`` — то, что возвращается, когда НИ ОДНО правило не сработало.
    Исторически это было «Износ РО», и на выгрузке ОТКАЗОВ (где узел заполнен у
    всех строк) оно почти не срабатывало. На ПОЛНОМ регистре срабатывает
    постоянно, и тогда категория износа набирается строками, про которые ничего
    не известно, — а износ мы по ней потом меряем: на нашей популяции это было
    278 отказов вместо 169, то есть **+109 выдуманных износов**. Поэтому здесь
    дефолт — ``None``, и повышать его до строки можно только осознанно.
    """
    node = _text(failed_node)
    element = _text(failed_element)
    character = _text(malfunction_character)
    reason = _text(esp_failure_reason)

    if not has_node_diagnosis(failed_node, failed_element):
        return None

    if node in ("кабельная линия", "тмс"):
        return "КЛ (R-0)"
    if element in _CABLE_ELEMENTS and any(
        word in character for word in ("изоляц", "прогар", "оплавл", "механич", "разрушен")
    ):
        return "КЛ (R-0)"
    if node == "пэд" and element not in ("узел пяты", "шлицевая муфта"):
        return "ПЭД (R-0)"
    if element in _MOTOR_ELEMENTS and any(
        word in character for word in ("замыкание", "электропробой", "прогар", "перегрев", "изоляц")
    ):
        return "ПЭД (R-0)"
    if node == "гидрозащита":
        return "Износ/негермет.гидрозащиты"
    if "вал" in element and "слом" in character:
        return "Слом вала"
    if "шлицевая муфта" in element and any(w in character for w in ("слом", "разрушен")) and node != "гидрозащита":
        return "Слом вала"
    if "корпус" in element and "слом" in character:
        return "Слом вала"
    if node in _NKT_NODES or "нкт" in node:
        return "НКТ"
    if "подвеска нкт" in element:
        return "НКТ"

    if node in _PUMP_NODES:
        if "рабочие органы" in element:
            if any(word in character for word in _WEAR_WORDS):
                return "Износ РО"
            if any(word in character for word in _CLOG_WORDS):
                return "Засорение РО"
            if any(word in reason for word in ("засорен", "солеотложени")):
                return "Засорение РО"
            return "Засорение РО"
        if "вал" in element:
            return "Слом вала"
        if any(word in character for word in _WEAR_WORDS) or any(
            word in reason for word in ("коррозия", "эрозион")
        ):
            return "Износ РО"
        if any(word in character for word in _CLOG_WORDS) or any(
            word in reason for word in ("засорен", "солеотложени")
        ):
            return "Засорение РО"
        return default

    if node == "пэд" and "узел пяты" in element:
        return "Износ РО"
    if node == "пэд":
        return "ПЭД (R-0)"

    scores = {category: 0 for category in NODE_CATEGORIES}
    if element in _CABLE_ELEMENTS:
        scores["КЛ (R-0)"] += 2
    if any(word in character for word in ("кабел", "изоляц")):
        scores["КЛ (R-0)"] += 1
    if element in _MOTOR_ELEMENTS:
        scores["ПЭД (R-0)"] += 2
    if "замыкание" in character:
        scores["ПЭД (R-0)"] += 2
    if "вал" in element:
        scores["Слом вала"] += 2
    if "слом" in character:
        scores["Слом вала"] += 2
    if "рабочие органы" in element:
        if any(word in character for word in _CLOG_WORDS):
            scores["Засорение РО"] += 3
        if any(word in character for word in _WEAR_WORDS):
            scores["Износ РО"] += 3
    if any(word in character for word in _CLOG_WORDS):
        scores["Засорение РО"] += 1
    if any(word in character for word in _WEAR_WORDS):
        scores["Износ РО"] += 1
    if "нкт" in node or "подвеска нкт" in element:
        scores["НКТ"] += 2
    if "обрыв" in character:
        scores["НКТ"] += 1
    if any(word in element for word in ("уплотнен", "пяты")) or "пята" in character:
        scores["Износ/негермет.гидрозащиты"] += 2
    if "негермет" in character:
        scores["Износ/негермет.гидрозащиты"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


def classify_failure_row(row, default: Optional[str] = None) -> Optional[str]:
    """Row-wise wrapper over :func:`classify_node_category` (Свод column names)."""
    return classify_node_category(
        row.get("Отказавший узел", ""),
        row.get("Отказавший элемент", ""),
        row.get("Характер неисправности", ""),
        row.get("Причина отказа УЭЦН", ""),
        default=default,
    )


# --------------------------------------------------------------------------- #
# Применение к регистру
# --------------------------------------------------------------------------- #

CAUSE_GROUP_COLUMN = "группа_причины"
DISPUTED_COLUMN = "спорная_зона"
NODE_CATEGORY_COLUMN = "узел_категория"


def attach_cause_columns(records: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Добавить в кадр три колонки разметки.

    Возвращает копию с ``группа_причины``, ``спорная_зона`` (0/1) и
    ``узел_категория`` (пустая, а не «Износ РО», когда диагноза нет).
    """
    records = records.copy()
    if records.empty:
        for column in (CAUSE_GROUP_COLUMN, DISPUTED_COLUMN, NODE_CATEGORY_COLUMN):
            records[column] = None
        return records

    def _column(name: str) -> pd.Series:
        if name in records.columns:
            return records[name]
        return pd.Series([None] * len(records), index=records.index, dtype=object)

    reason = _column("Причина отказа УЭЦН")
    note = _column("Примечание")
    complications = _column("Осложнения при ТКРС, ДЖ, Рекомендации ТКРС")

    records[CAUSE_GROUP_COLUMN] = [
        classify_cause_group(reason_value, note_value, complications_value)
        for reason_value, note_value, complications_value in zip(reason, note, complications)
    ]
    records[DISPUTED_COLUMN] = [is_disputed_cause(value) for value in reason]
    records[NODE_CATEGORY_COLUMN] = [
        classify_failure_row(row, default=None) for _, row in records.iterrows()
    ]

    if verbose:
        counts = records[CAUSE_GROUP_COLUMN].value_counts()
        summary = ", ".join(f"{group}: {count}" for group, count in counts.items())
        disputed = int(pd.to_numeric(records[DISPUTED_COLUMN], errors="coerce").fillna(0).sum())
        categorized = int(records[NODE_CATEGORY_COLUMN].notna().sum())
        print(f"  Cause taxonomy: {summary}")
        print(f"  Disputed (commercial boundary) rows: {disputed}")
        print(f"  Node category filled on {categorized} of {len(records)} row(s)")
        leftovers = unmatched_causes(reason.dropna())
        if not leftovers.empty:
            preview = ", ".join(f"«{value}» ×{count}" for value, count in leftovers.head(8).items())
            print(f"  ⚠ Причины без правила ({GROUP_OTHER}): {preview}")

    return records


__all__ = [
    "CAUSE_GROUPS",
    "CAUSE_GROUP_COLUMN",
    "CAUSE_RULES",
    "DISPUTED_COLUMN",
    "DISPUTED_RULES",
    "GROUP_BUILD",
    "GROUP_OPERATION",
    "GROUP_OTHER",
    "GROUP_UNSPECIFIED",
    "GROUP_WELL",
    "NODE_CATEGORIES",
    "NODE_CATEGORY_COLUMN",
    "TEXT_RULES",
    "UNSPECIFIED_REASONS",
    "attach_cause_columns",
    "classify_cause_group",
    "classify_failure_row",
    "classify_node_category",
    "has_node_diagnosis",
    "is_disputed_cause",
    "unmatched_causes",
]
