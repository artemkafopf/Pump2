"""Кандидаты в СТАТИЧЕСКИЙ слой: заканчивание, оборудование, ГРП, история скважины.

Статический слой модели отказов — это сейчас ровно один член ``θ_Qном = (Qном/250)^c``.
Здесь собраны величины, которые известны **на момент спуска** и потому годятся в прогноз, а
не только в объяснение: кривизна и тип ствола (заканчивание), число ступеней и мощность ПЭД
(оборудование), факт ГРП до монтажа, история отказов самой скважины по узлам.

Ключ стыковки — ``(well_key, дата монтажа)``
--------------------------------------------
⚠⚠ Позиционный ``run`` в этом проекте уже стоил 669 пусков из 2308, получивших чужой уровень
(FINDINGS §57, [[project_v64_corrected_v63]]). Здесь всё стыкуется только по содержательному
ключу. Свод и Big описывают одни и те же спуски, но расходятся в дате монтажа на несколько
суток, поэтому Big подтягивается ``merge_asof`` с допуском — ровно как в
:func:`analysis.workflows.production_risk.esp_population.attach_equipment`.

⚠⚠ «Работа в кривизне»: ПРОЧЕРК — это ЗНАЧЕНИЕ, а не пропуск
------------------------------------------------------------
В Своде колонка заполнена тремя способами: число (1857 строк, 0.30–15.4), **прочерк ``-``**
(1502) и пустая ячейка (65). То есть прочерк ставится намеренно и означает «насос не работает
в кривизне», а настоящий пропуск — это отдельные 1.9 % строк. Заполнять прочерк нулём нельзя
дважды: во-первых, у числовых значений нет нуля вовсе (минимум 0.30), то есть ноль был бы вне
носителя; во-вторых, пустая ячейка тогда слилась бы с прочерком. Поэтому кривизна кодируется
СОСТОЯНИЕМ (:data:`CURV_LEVELS`), а не числом с заполнением.

Единицы кривизны установлены по распределению, а не по названию колонки: q05 0.32, медиана
0.55, q95 1.58 при максимуме 15.4 — это **градусы на 10 м** (при градусах полного угла
наклона распределение лежало бы в диапазоне 0–90).

⚠ ГРП берётся ТОЛЬКО выполненный ДО монтажа текущего пуска. «ГРП во время пуска» — не
ковариата, а конкурирующее событие: он заканчивает пуск подъёмом по ГТМ, и признак означал бы
обусловливание на будущее.

⚠ История отказов строится строго по пускам, ЗАКОНЧИВШИМСЯ до даты монтажа текущего, и у
первого пуска скважины она отсутствует — это ОТДЕЛЬНОЕ состояние, а не ноль: «истории нет» и
«история чистая» — разные вещи.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import resolve_frac_db_path

#: Колонки Свода, из которых собирается статический паспорт пуска.
SVOD_STATIC_COLUMNS = {
    "Работа в кривизне": "curvature_raw",
    "Тип ствола скв": "wellbore_type_raw",
    "Кол.ступеней": "stages",
    "Мощность, кВт": "ped_power_kw",
    "Ном.напор (50Гц)": "head_nom_m",
    "Глубина спуска УЭЦН, по НКТ": "pump_depth_m",
}

#: Колонки паспорта, которые остаются ТЕКСТОМ (разбираются отдельными парсерами).
_TEXT_COLUMNS = {"curvature_raw", "wellbore_type_raw"}

#: Состояния кривизны. ``нет`` — прочерк в Своде, ``неизвестно`` — пустая ячейка.
CURV_LEVELS = ("нет", "слабая", "сильная", "неизвестно")

#: Граница «слабая/сильная» — медиана числовых значений, ЗАДАНА ЗАРАНЕЕ (не подбиралась).
CURV_SPLIT = 0.55

#: Тип ствола. Вертикальные (23 пуска) и многоствольные (10) слишком редки для своих уровней —
#: вертикальные идут к наклонно-направленным (общее «не горизонтальная»), многоствольные к
#: горизонтальным. ⚠ Это решение, а не данность; оно печатается в шаге 0.
WELLBORE_LEVELS = ("гориз", "негориз", "неизвестно")
_WELLBORE_MAP = {
    "горизонтальная": "гориз",
    "многостовольная": "гориз",
    "многоствольная": "гориз",
    "наклонно-направленная": "негориз",
    "вертикальная": "негориз",
}

#: Узлы отказа, по которым строится история скважины (порядок фиксирован для воспроизводимости).
HISTORY_NODES = (
    "Засорение РО",
    "Износ РО",
    "КЛ (R-0)",
    "ПЭД (R-0)",
    "Слом вала",
    "Износ/негермет.гидрозащиты",
    "НКТ",
)

#: Допуск стыковки Свод↔Big по дате монтажа, суток (как в ``esp_population.attach_equipment``).
BIG_TOLERANCE_DAYS = 10


# ---------------------------------------------------------------------------
# Нормализаторы — чистые функции, покрыты тестами
# ---------------------------------------------------------------------------
def norm_key(value: object) -> str | None:
    """Канонический ключ скважины: ``Ya_149`` / ``ya_149 `` -> ``ya_149``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = " ".join(str(value).split()).strip().casefold().replace(" ", "")
    return text or None


def parse_curvature(value: object) -> tuple[str, float | None]:
    """(состояние, число) из ячейки «Работа в кривизне».

    Прочерк — намеренная запись «не работает в кривизне», пустая ячейка — пропуск. Всё, что
    не разбирается в число и не прочерк, считается пропуском, а не нулём.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "неизвестно", None
    text = " ".join(str(value).split()).strip()
    if not text:
        return "неизвестно", None
    if text in {"-", "—", "–", "нет"}:
        return "нет", None
    try:
        num = float(text.replace(",", ".").replace(" ", ""))
    except ValueError:
        return "неизвестно", None
    if not np.isfinite(num) or num <= 0:
        return "неизвестно", None
    return ("слабая" if num < CURV_SPLIT else "сильная"), num


def parse_wellbore(value: object) -> str:
    """«Тип ствола скв» -> один из :data:`WELLBORE_LEVELS`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "неизвестно"
    text = " ".join(str(value).split()).strip().casefold()
    return _WELLBORE_MAP.get(text, "неизвестно" if not text else "негориз")


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------
def load_svod_static(prediction_workbook_path: Path | None = None) -> pd.DataFrame:
    """Статический паспорт пуска из Свода: одна строка на ``(well_key, install)``."""
    from analysis.data.equipment_big import to_num
    from analysis.workflows.production_risk import crosswalk

    src = Path(prediction_workbook_path or crosswalk.resolve_prediction_workbook_path())
    sv = pd.read_excel(src, sheet_name="Свод")
    out = pd.DataFrame({
        "well_key": sv["Скв."].map(norm_well_key),
        "install": pd.to_datetime(sv["Дата монтажа"], errors="coerce").dt.normalize(),
    })
    for src_col, dst in SVOD_STATIC_COLUMNS.items():
        if src_col not in sv.columns:
            out[dst] = np.nan
            continue
        # ⚠ Числовые колонки Свода приходят ТЕКСТОМ с неразрывными пробелами-разделителями
        # («2 680.00»): наивный `to_numeric` даёт 3 % покрытия вместо 100 %, молча.
        out[dst] = sv[src_col] if dst in _TEXT_COLUMNS else to_num(sv[src_col])
    out = out.dropna(subset=["well_key", "install"])
    return out.drop_duplicates(["well_key", "install"], keep="first").reset_index(drop=True)


def norm_well_key(value: object) -> str | None:
    """Ключ скважины Свода в написании популяции (``ya_149``)."""
    from analysis.workflows.production_risk import crosswalk

    code = crosswalk.norm_well(value)
    return norm_key(code)


def load_big_static(equipment_big_path: Path | None = None) -> pd.DataFrame:
    """Тот же паспорт из ``WellsArtificialLiftBig`` — для пусков, которых в Своде нет."""
    from analysis.data.equipment_big import load_equipment_big

    big = load_equipment_big(equipment_big_path) if equipment_big_path else load_equipment_big()
    big = big[big["is_esp_strict"] == True].copy()  # noqa: E712 — pandas mask
    out = pd.DataFrame({
        "well_key": big["well_key"].map(norm_key),
        "install": pd.to_datetime(big["install_date"], errors="coerce").dt.normalize(),
        "curvature_raw": big.get("curvature"),
        "wellbore_type_raw": np.nan,
        "stages": pd.to_numeric(big.get("stages"), errors="coerce"),
        "ped_power_kw": pd.to_numeric(big.get("ped_power_kw"), errors="coerce"),
        "head_nom_m": pd.to_numeric(big.get("head_nom_m"), errors="coerce"),
        "pump_depth_m": pd.to_numeric(big.get("pump_depth_m"), errors="coerce"),
    })
    out = out.dropna(subset=["well_key", "install"])
    return out.drop_duplicates(["well_key", "install"], keep="first").reset_index(drop=True)


def load_frac_stages(frac_db_path: Path | None = None) -> pd.DataFrame:
    """Стадии ГРП из реестра «Свод ГРП»: ``well_key``, ``frac_date``, ``gtm_type``.

    ⚠ Реестр перенесён в репозиторий процессором, который закрыл две ловушки выгрузки —
    сдвиг заголовка на колонку и неуникальные метки колонок. Проверка распределения этим не
    заменяется: она делается на шаге 0 (:func:`frac_coverage`).
    """
    path = Path(frac_db_path or resolve_frac_db_path())
    with sqlite3.connect(str(path)) as con:
        stages = pd.read_sql(
            "SELECT well_key, frac_date, stage_no, gtm_type FROM frac_stages", con)
    stages["well_key"] = stages["well_key"].map(norm_key)
    stages["frac_date"] = pd.to_datetime(stages["frac_date"], errors="coerce")
    return stages.dropna(subset=["well_key", "frac_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Приклейка к популяции
# ---------------------------------------------------------------------------
def _keyed(pop: pd.DataFrame, well_col: str, install_col: str) -> pd.DataFrame:
    out = pop.copy()
    out["_wk"] = out[well_col].map(norm_key)
    out["_inst"] = pd.to_datetime(out[install_col], errors="coerce").dt.normalize()
    return out


def attach_passport(pop: pd.DataFrame, *, well_col: str = "code",
                    install_col: str = "install",
                    svod: pd.DataFrame | None = None,
                    big: pd.DataFrame | None = None) -> pd.DataFrame:
    """Кривизна, тип ствола, ступени, мощность на каждый пуск популяции.

    Свод стыкуется ТОЧНО по ``(well_key, дата монтажа)`` — популяция из него и построена.
    Big подтягивается ``merge_asof`` с допуском :data:`BIG_TOLERANCE_DAYS`, потому что даты
    монтажа в двух регистрах расходятся на несколько суток, и только там, где Свода нет.
    Колонка ``passport_source`` говорит, откуда приехало значение, — без неё покрытие
    невозможно отчитать, а оно не случайно.
    """
    svod = load_svod_static() if svod is None else svod
    big = load_big_static() if big is None else big
    left = _keyed(pop, well_col, install_col)

    cols = ["curvature_raw", "wellbore_type_raw", "stages", "ped_power_kw",
            "head_nom_m", "pump_depth_m"]
    exact = left.merge(svod.rename(columns={"well_key": "_wk", "install": "_inst"}),
                       on=["_wk", "_inst"], how="left")
    exact["passport_source"] = np.where(exact["curvature_raw"].notna()
                                        | exact["wellbore_type_raw"].notna()
                                        | exact["stages"].notna(), "svod", "нет")

    need = exact["passport_source"].eq("нет").to_numpy()
    if need.any() and len(big):
        right = big.rename(columns={"well_key": "_wk", "install": "_inst"})
        sub = (exact.loc[need, ["_wk", "_inst"]].reset_index()
               .dropna(subset=["_inst"]).sort_values("_inst"))
        got = pd.merge_asof(sub, right.sort_values("_inst"), on="_inst", by="_wk",
                            tolerance=pd.Timedelta(days=BIG_TOLERANCE_DAYS),
                            direction="nearest")
        got = got.set_index("index")
        for c in cols:
            if c in got.columns:
                fill = exact.loc[got.index, c].isna().to_numpy()
                exact.loc[got.index[fill], c] = got.loc[got.index[fill], c]
        hit = got[[c for c in cols if c in got.columns]].notna().any(axis=1)
        exact.loc[hit[hit].index, "passport_source"] = "big"

    parsed = [parse_curvature(v) for v in exact["curvature_raw"]]
    exact["curv_state"] = [p[0] for p in parsed]
    exact["curv_deg10m"] = [p[1] for p in parsed]
    exact["wellbore"] = [parse_wellbore(v) for v in exact["wellbore_type_raw"]]
    for c in ("stages", "ped_power_kw", "head_nom_m", "pump_depth_m"):
        exact[c] = pd.to_numeric(exact[c], errors="coerce")
    return exact.drop(columns=["_wk", "_inst"])


def attach_frac(pop: pd.DataFrame, *, well_col: str = "code", install_col: str = "install",
                stages: pd.DataFrame | None = None) -> pd.DataFrame:
    """Признаки ГРП, выполненного СТРОГО ДО монтажа пуска.

    ``frac_before`` — был ли; ``frac_stages_before`` — сколько стадий; ``frac_age_days`` —
    сколько суток прошло от последней стадии до монтажа. Скважины, у которых ГРП есть, но
    ПОЗЖЕ монтажа, помечаются ``frac_later`` — они не «без ГРП», и смешивать их с
    нетронутыми нельзя.
    """
    stages = load_frac_stages() if stages is None else stages
    left = _keyed(pop, well_col, install_col)
    by_well: dict[str, np.ndarray] = {
        k: np.sort(g["frac_date"].to_numpy("datetime64[ns]"))
        for k, g in stages.groupby("well_key", sort=False)
    }
    n_before, last_before, later = [], [], []
    for wk, inst in zip(left["_wk"], left["_inst"]):
        dates = by_well.get(wk)
        if dates is None or pd.isna(inst):
            n_before.append(0); last_before.append(pd.NaT); later.append(False)
            continue
        cut = np.datetime64(inst)
        before = dates[dates < cut]
        n_before.append(int(before.size))
        last_before.append(pd.Timestamp(before[-1]) if before.size else pd.NaT)
        later.append(bool(before.size == 0 and dates.size))
    out = pop.copy()
    out["frac_stages_before"] = n_before
    out["frac_last_before"] = last_before
    out["frac_before"] = np.asarray(n_before) > 0
    out["frac_later"] = later
    inst = pd.to_datetime(out[install_col], errors="coerce")
    out["frac_age_days"] = (inst - out["frac_last_before"]).dt.days.astype("float")
    return out


def attach_failure_history(pop: pd.DataFrame, *, well_col: str = "code",
                           install_col: str = "install", clock_col: str = "t1",
                           event_col: str = "event", cause_col: str = "cause",
                           nodes: tuple[str, ...] = HISTORY_NODES,
                           counts_as_failure=None, counts_as_run=None,
                           prefix: str = "hist") -> pd.DataFrame:
    """История отказов скважины по пускам, ЗАКОНЧИВШИМСЯ до монтажа текущего.

    Для каждого пуска считается ``hist_n`` — сколько прошлых пусков этой скважины уже
    закончилось, ``hist_fail_share`` — какая их доля отказала (любым узлом), и
    ``hist_<узел>_share`` — доля, отказавшая ЭТИМ узлом.

    ``counts_as_failure`` — булева маска по строкам ``pop``, сужающая понятие «отказ» для
    ИСТОРИИ. ⚠⚠ Это не косметика: по умолчанию в историю идёт ЛЮБОЙ отказ, а среди них
    эксплуатационных только 42 % — остальное монтаж/завод, скважина/организация и, главное,
    архивные пуски до 2018, у которых причины нет вовсе. «Был отказ» и «был отказ ПО ИЗНОСУ» —
    разные признаки, и разница видна только если считать обе истории и сравнить вне выборки.
    ``prefix`` разводит колонки, чтобы обе истории жили в одном кадре.

    ``counts_as_run`` — маска, ограничивающая, какие пуски вообще могут БЫТЬ историей.
    ⚠⚠ Без неё история когорты 2018+ на 30 % состоит из архивных пусков до 2018, у которых
    причина отказа неизвестна в 79 % случаев. Тогда «в истории был отказ с известной
    причиной» означает попросту «предыдущий пуск был свежий» — эпоха, а не физика скважины.
    Ограничение истории той же когортой, что и событие, снимает это по построению.

    ⚠⚠ Утечки из будущего здесь нет по построению: берутся только пуски, чей конец строго
    раньше монтажа текущего. Конец считается как ``монтаж + наработка`` — в популяции v6.4
    часы календарные.
    ⚠ У первого пуска скважины истории НЕТ, и это отдельное состояние (``hist_n == 0``), а не
    «чистая история»: доли у таких пусков остаются NaN, чтобы их нельзя было принять за нули.
    """
    left = _keyed(pop, well_col, install_col)
    end = left["_inst"] + pd.to_timedelta(
        pd.to_numeric(left[clock_col], errors="coerce").fillna(0.0), unit="D")
    ev = pd.to_numeric(left[event_col], errors="coerce").fillna(0.0).to_numpy() > 0
    if counts_as_failure is not None:
        ev = ev & np.asarray(counts_as_failure, bool)
    usable = (np.ones(len(left), bool) if counts_as_run is None
              else np.asarray(counts_as_run, bool))
    cause = left[cause_col].astype(str).to_numpy() if cause_col in left.columns else None

    order = np.argsort(left["_inst"].to_numpy("datetime64[ns]"), kind="stable")
    idx_by_well: dict[str, list[int]] = {}
    for i in order:
        idx_by_well.setdefault(left["_wk"].iloc[i], []).append(int(i))

    n_prev = np.zeros(len(left)); n_fail = np.zeros(len(left))
    node_fail = {c: np.zeros(len(left)) for c in nodes}
    ends = end.to_numpy("datetime64[ns]")
    insts = left["_inst"].to_numpy("datetime64[ns]")
    for rows in idx_by_well.values():
        for pos, i in enumerate(rows):
            for j in rows[:pos]:
                if not (ends[j] < insts[i]) or not usable[j]:
                    continue
                n_prev[i] += 1
                if ev[j]:
                    n_fail[i] += 1
                    if cause is not None and cause[j] in node_fail:
                        node_fail[cause[j]][i] += 1

    out = pop.copy()
    out[f"{prefix}_n"] = n_prev.astype(int)
    has = n_prev > 0
    out[f"{prefix}_fail_share"] = np.where(has, n_fail / np.maximum(n_prev, 1), np.nan)
    out[f"{prefix}_any"] = np.where(has, (n_fail > 0).astype(float), np.nan)
    for c in nodes:
        key = f"{prefix}_{_node_slug(c)}_share"
        out[key] = np.where(has, node_fail[c] / np.maximum(n_prev, 1), np.nan)
        out[f"{prefix}_{_node_slug(c)}_any"] = np.where(has, (node_fail[c] > 0).astype(float),
                                                        np.nan)
    return out


_NODE_SLUGS = {
    "Засорение РО": "clog",
    "Износ РО": "wear",
    "КЛ (R-0)": "cable",
    "ПЭД (R-0)": "motor",
    "Слом вала": "shaft",
    "Износ/негермет.гидрозащиты": "protector",
    "НКТ": "tubing",
    "не определено": "unknown",
}


def _node_slug(cause: str) -> str:
    return _NODE_SLUGS.get(cause, "".join(ch for ch in cause.casefold() if ch.isalnum())[:10])


# ---------------------------------------------------------------------------
# Отчёт покрытия — шаг 0 требует его для КАЖДОГО кандидата
# ---------------------------------------------------------------------------
def coverage(frame: pd.DataFrame, column: str, by: str = "stratum") -> pd.DataFrame:
    """Покрытие и распределение ковариаты по стратам — доля непустых и число уровней."""
    d = frame[[by, column]].copy()
    present = d[column].notna()
    rows = []
    for s, g in d.groupby(by, sort=True):
        p = g[column].notna()
        rows.append({by: s, "пусков": len(g), "непустых": int(p.sum()),
                     "доля": round(float(p.mean()), 3),
                     "уникальных": int(g.loc[p, column].nunique())})
    rows.append({by: "ФОНД", "пусков": len(d), "непустых": int(present.sum()),
                 "доля": round(float(present.mean()), 3),
                 "уникальных": int(d.loc[present, column].nunique())})
    return pd.DataFrame(rows)


def frac_coverage(pop: pd.DataFrame, by: str = "stratum") -> pd.DataFrame:
    """Проверка распределения признака ГРП против ожидаемого — обязательна перед фитом."""
    rows = []
    for s, g in pop.groupby(by, sort=True):
        rows.append({by: s, "пусков": len(g),
                     "ГРП_до": int(g["frac_before"].sum()),
                     "доля_до": round(float(g["frac_before"].mean()), 3),
                     "ГРП_позже": int(g["frac_later"].sum()),
                     "стадий_медиана": float(
                         g.loc[g["frac_before"], "frac_stages_before"].median())
                     if g["frac_before"].any() else np.nan})
    return pd.DataFrame(rows)
