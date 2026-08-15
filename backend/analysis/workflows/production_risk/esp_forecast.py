"""Прогноз отказов по Мирнинскому УН (Mc+Mr) — ожидаемые числа, не медианные даты.

Зачем это существует
--------------------
Правило «поднимать на медианном остаточном ресурсе» СОЗДАЁТ ком. Измерено на живом
парке Mc (58 насосов): МРП(t) лежит в диапазоне 122..233 сут при возрастах 2..763 сут,
то есть детерминированное правило обязано сложить все 58 подъёмов в окно ~4 месяца
(пик 18 в одном месяце). Реальные отказы разнесены РАСПРЕДЕЛЕНИЕМ, а точечная оценка
этот разброс выбрасывает. Слой Ql раздвигает окно лишь до 116..253 (пик 18->15) —
он не лечит механизм, потому что причина не в ковариатах, а в самом правиле.

Поэтому здесь месячная вероятность, а не дата:

  ИСТОРИЯ (сверка с фактом) — обновления НЕТ. Даты монтажа известны, поэтому
      ожидание за месяц = сумма по пробегам, ДОЖИВШИМ до начала месяца, величины
      1 - S(a+d)/S(a). Это прогноз на шаг вперёд, сопоставимый с фактом.
  ПРОГНОЗ — обновление ЕСТЬ. Будущие монтажи неизвестны, поэтому отказавший насос
      немедленно заменяется и новый входит в возраст 0 (процесс восстановления).

Часы
----
Модель обучена на `tte` = Свод «Наработка (сут)» ~= КАЛЕНДАРЬ (медиана tte/календарь
= 0.986). Поэтому здесь календарные сутки без пересчёта на наработку.

Слой Ql — честный статус
------------------------
`QL_BETA_LANDMARK` = ln(1.253) измерен landmark-схемой (ковариата из окна [0,30),
риск-набор — дожившие до 30 сут), глобально по 19 стратам, n=1741, p<1e-4. Он не
совпадает с развёрнутым `config.QL_HAZARD_BETA = 0.07`, и это РАЗНЫЕ ОЦЕНИВАЕМЫЕ
ВЕЛИЧИНЫ: 0.07 — внутрипробеговый прямой causal-эффект из lag/lead-проверки
(lead t+3 beta=+0.84 vs lag t-3 beta=+0.09), пригодный как «что если»; landmark
меряет МЕЖСКВАЖИННУЮ связь — она годится для РАНЖИРОВАНИЯ, но частично несёт
неизмеренные свойства скважины и не является рычагом.

Опорную точку из `config.QL_HAZARD_FIELD_REF_LOG['Mc']` (=3.782, ~43 м3/сут) НЕЛЬЗЯ
брать для этого парка: живые Mc дают медиану ~144 м3/сут, 14 из 58 упираются в цап
(+-log5) и слой вырождается в константу. Опора считается по самой популяции.

Слой включается только через `c4_gate()` — гейт из Workstream C (обучение <= 2024-12,
оценка 2025-01..2026-06). Пока гейт не пройден, слой остаётся сценарным.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as P

# c0: НЕ режем ранние отказы — стартовую массу дня-0 держит первая компонента смеси
# (w1~3% при eta1 на полу 0.5). Это выбранная модель «cal c0 k2»: она БЕЗУСЛОВНАЯ —
# описывает жизнь ЛЮБОГО смонтированного насоса, включая ~3%, отказывающих при запуске.
#
# Было 3.0 (c3). c3 условен на дожитии до дня 3, т.е. выбрасывает стартовые отказы и
# потому систематически ОПТИМИСТИЧЕН: против факта KM он идёт выше на +0.02..+0.06
# на всём горизонте, MRL 645 против 572 у c0 k2. Для прогноза отказов «от монтажа»
# нужен безусловный c0.  Сетка всех 36 подгонок: results/production_risk_mc_weibull_grid/.
CUT_DAYS = 0.0

# Отображаемый ОР = RESOURCE_BLEND_W·RMST + (1-RESOURCE_BLEND_W)·MRL.
#
# Это ТОЛЬКО ОТОБРАЖАЕМАЯ цифра: прогноз отказов её не использует (проверено —
# статистика берёт из модели суточную опасность q[a]=1-S(a+1)/S(a) и даёт те же
# 52.03 ожидаемых отказа при любом определении ОР, вплоть до абсурдного 999).
#
# Почему смесь, а не чистые RMST/MRL:
#   RMST(a) при фиксированном tau падает к 0 при a->tau => 6 из 58 живых насосов Mc
#     (500..763 сут) получили бы «отказ сегодня». Это артефакт закрытия окна.
#   MRL(a) не схлопывается (573->413), но даёт старому насосу 413 сут остатка —
#     основание «не трогать ещё год с лишним» для насоса, отработавшего 3 года.
#   w=0.9 => старые получают ~40 сут («дай доработать квартал»), а не 0 и не 413.
# Ограничение, которое надо знать: за tau смесь вырождается в (1-w)·MRL, поэтому для
# 500/700/1000 сут выходит 44/41/38 — по возрасту почти не различает.
RESOURCE_BLEND_W = 0.9
QL_BETA_LANDMARK = float(np.log(1.253))
QL_CAP = C.QL_HAZARD_CAP_LOG_RATIO  # log(5) — переиспользуем развёрнутый цап
OOS_FIT_LAST = "2024-12"
OOS_SCORE_FIRST = "2025-01"
MIN_REL_GAIN = 0.05                 # запас, ниже которого выигрыш MAE — шум


@dataclass(frozen=True)
class SurvivalModel:
    w1: float
    beta1: float
    eta1: float
    beta2: float
    eta2: float
    n_runs: int
    n_events: int
    n_censored: int
    n_cut: int
    cut_days: float
    tau: float = 475.0        # горизонт ограничения RMST = 95-й перцентиль наблюдённой наработки

    def as_dict(self) -> dict:
        return asdict(self)

    def survival(self, t) -> np.ndarray:
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return (self.w1 * np.exp(-((t / self.eta1) ** self.beta1))
                + (1.0 - self.w1) * np.exp(-((t / self.eta2) ** self.beta2)))

    def rmst(self, age: float = 0.0, tau: float | None = None, theta: float = 1.0) -> float:
        """Условный RMST = ожидаемый ОСТАТОЧНЫЙ ресурс в окне (age, tau].

        RMST(a) = ∫_a^tau S(u)du / S(a) — это и есть RUL: ожидаемая оставшаяся
        наработка при условии дожития до возраста a.  ГЛАВНАЯ величина: она привязана
        к наблюдённому горизонту и почти не зависит от выбора подгонки (по всем шести
        хорошим фитам расходится на единицы суток), в отличие от медианы (B50 гуляет
        418..601) и от безусловного среднего (гуляет 517..662).

        Убывает к нулю при a -> tau ПО ПОСТРОЕНИЮ (окно закрывается) — это не износ.
        """
        tau = self.tau if tau is None else float(tau)
        if age >= tau:
            return 0.0
        g = np.linspace(age, tau, 4000)
        s = np.asarray(self.survival(g), dtype=float) ** theta
        s_a = float(self.survival(age)) ** theta
        return float(np.trapezoid(s, g)) / max(s_a, 1e-12)

    def mrl(self, age: float = 0.0, theta: float = 1.0) -> float:
        """Безусловный MRL = ∫_a^inf S(u)du / S(a).

        Показывает ФОРМУ: при beta>1 убывает с возрастом (износ), при beta<1 растёт.
        Уровень экстраполирует хвост за tau, где данных нет (для Mc за 475 сут живо
        ещё 53% насосов) — доверять НАПРАВЛЕНИЮ, не абсолюту.
        """
        top = max(self.eta2 * 30.0, self.tau * 10.0)
        g = np.linspace(age, top, 60000)
        s = np.asarray(self.survival(g), dtype=float) ** theta
        s_a = float(self.survival(age)) ** theta
        return float(np.trapezoid(s, g)) / max(s_a, 1e-12)

    def resource(self, age: float = 0.0, theta: float = 1.0,
                 w: float | None = None) -> float:
        """ОТОБРАЖАЕМЫЙ остаточный ресурс = w·RMST(age) + (1-w)·MRL(age).

        Единственная цифра, которую мы называем человеку. Прогноз отказов её НЕ
        использует. Обоснование выбора w — у RESOURCE_BLEND_W.
        """
        w = RESOURCE_BLEND_W if w is None else float(w)
        return w * self.rmst(age, theta=theta) + (1.0 - w) * self.mrl(age, theta=theta)

    def b50(self) -> float:
        """Медиана. НЕ использовать как ресурс — см. resource()/rmst(). Для диагностики."""
        from scipy.optimize import brentq
        return float(brentq(lambda x: self.survival(x) - 0.5, 1e-6, 1e6))

    def daily_fail_prob(self, max_age: int) -> np.ndarray:
        """q[a] = P(отказ на сутках a | дожил до a) для возрастов 0..max_age."""
        s = self.survival(np.arange(max_age + 2, dtype=float))
        q = 1.0 - np.divide(s[1:], s[:-1], out=np.ones_like(s[:-1]), where=s[:-1] > 0)
        return np.clip(q, 0.0, 1.0)


def fit_mc_model(pop: pd.DataFrame | None = None, cut_days: float = CUT_DAYS,
                 all_cause: bool = False) -> SurvivalModel:
    """Смесь Вейбулла на Mc (монтажи 2024+).

    ``all_cause=False`` (по умолчанию) — событие = ОТКАЗ; ГТМ и работающие насосы в
    цензуру. Это оценка для ПРОГНОЗА ОТКАЗОВ (ННО, B50~520).
    ``all_cause=True`` — событие = ЛЮБОЙ подъём (отказ+ГТМ). Это оценка для
    ПЛАНИРОВАНИЯ РЕМОНТОВ (МРП, B50~240) — отдельная задача, см.
    [[project_workover_planning_model]]: две задачи требуют двух оценок.

    Раньше здесь было зашито all-cause, из-за чего «прогноз отказов» на деле считал
    подъёмы и давал 10.5%/мес вместо 4.6%.

    Ограничения beta2 > max(1, beta1) и eta2 >= 2*eta1 берутся из
    `fit_latent_weibull_em` — без них подгонка меняет местами компоненты (проверено:
    при c=3 без ограничений выходило beta1=5.08 > beta2=0.954).
    """
    from analysis.models.survival.weibull_em import fit_latent_weibull_em

    if pop is None:
        pop = P.build("2026-07-01")
    mc = P.select(pop, "Mc")
    mc = mc[mc["tte"] > 0].copy()
    mc["event"] = (mc["end"].notna().astype(int) if all_cause
                   else (mc["event"] == 1).astype(int))
    early = (mc["event"] == 1) & (mc["tte"] < cut_days)
    fitset = mc[~early]
    r = fit_latent_weibull_em(fitset["tte"].to_numpy(float),
                              fitset["event"].to_numpy(int), num_starts=60)
    m = r.model
    return SurvivalModel(
        w1=float(m.weight_1),
        beta1=float(m.component_1.beta), eta1=float(m.component_1.eta),
        beta2=float(m.component_2.beta), eta2=float(m.component_2.eta),
        n_runs=int(len(fitset)), n_events=int(fitset["event"].sum()),
        n_censored=int((fitset["event"] == 0).sum()), n_cut=int(early.sum()),
        cut_days=float(cut_days),
        # tau для RMST — 95-й перцентиль НАБЛЮДЁННОЙ наработки: дальше него кривая
        # держится на считанных насосах, и интегрировать туда значит выдумывать хвост.
        tau=float(np.percentile(fitset["tte"].to_numpy(float), 95)),
    )


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"))


def ql_theta(ql_m3d: float, ref_log: float, beta: float = QL_BETA_LANDMARK) -> float:
    """theta = exp(beta * clip(log1p(Ql) - ref, +-log5)) — форма развёрнутого дила."""
    if ql_m3d is None or not np.isfinite(ql_m3d) or ql_m3d < 0:
        return 1.0
    z = float(np.clip(np.log1p(float(ql_m3d)) - float(ref_log), -QL_CAP, QL_CAP))
    return float(np.exp(beta * z))


def active_fleet_by_month(codes: set[str] | None = None,
                          master_path=None) -> dict[str, set[str]]:
    """Активный ДОБЫВАЮЩИЙ парк по месяцам из плана ПП: месяц -> {скважины}.

    Критерий тот же, что у `failure_rate._active_producing_rows`:
    «Отработанное время» > 0 И (нефть > 0 ИЛИ жидкость > 0).

    Зачем: «есть смонтированный непонятый насос» != «скважина добывала». Насос может
    стоять в скважине, которая месяц не работала (простой, ожидание, ППД, перевод), —
    план такой месяц показывает нулём. Замер на Мирнинском 2024-10: пробегов по
    Свод/Big — 36 скважин, а добывающих по плану — 19. Знаменатель был завышен вдвое,
    т.е. интенсивность отказов занижена вдвое.
    """
    from analysis.workflows.production_risk import crosswalk as _cw

    plan = _cw.load_plan(master_path=master_path) if master_path else _cw.load_plan()
    op, oil, liq = plan.op_days_raw, plan.oil_volume, plan.liquid_volume
    months = list(op.columns)
    oil = oil.reindex(index=op.index, columns=months, fill_value=0.0)
    liq = liq.reindex(index=op.index, columns=months, fill_value=0.0)
    active = (op > 0) & ((oil > 0) | (liq > 0))
    out: dict[str, set[str]] = {}
    for month in months:
        wells = set(active.index[active[month]].astype(str))
        out[str(month)] = (wells & codes) if codes is not None else wells
    return out


def plan_entrants(live_codes: set[str], codes: set[str], asof: pd.Timestamp, months: int,
                  master_path=None, prefixes: tuple[str, ...] = ("MC_", "MR_", "NE_"),
                  ) -> pd.DataFrame:
    """Скважины плана, которым нужна строка со СВЕЖИМ насосом. code/entry/start_day/age/category.

    Две группы, обе обязаны быть в прогнозе, иначе он внутренне противоречив:

    * **ВНС** (`category="ВНС"`) — скважины, которых ещё НЕТ в нашей ЭЦН-истории
      (`code not in codes`). Парк Мирнинского по плану РАСТЁТ 67 -> 82 за 18 мес за счёт
      ввода новых скважин; без них либо % от парка занижен (отказы новых не в
      числителе), либо знаменатель держится на 58-63 и показывает «падение парка»,
      которого нет.
    * **БАЗА между насосами** (`category="БАЗА"`) — скважины, которые В нашей истории
      есть (`code in codes`), план их активно добывает, но живого насоса на ``asof`` НЕТ
      (последний подъём прошёл прямо перед датой расчёта). Их пропускали ОБА фильтра:
      не в `live` (нет открытого пробега) и не среди ВНС (они в `codes`). На Mc это
      MC_201/MC_306/MC_608/MR_1904 — реальные скважины с гарантированным будущим
      отказом, молча выпадавшие из прогноза.

    Кто входит: активная добывающая скважина плана (см. `active_fleet_by_month`) с
    префиксом Mc/Mr, БЕЗ живого насоса на asof (`code not in live_codes`), с датой
    появления свежего насоса внутри горизонта.

    Дата ввода и возраст:
    * ВНС с первым активным месяцем >= asof — вводится вперёд: entry = этот месяц,
      age = 0, start_day = дни от asof.
    * ВНС, уже добывающая до asof (лаг отчётности: план показал добычу, пробега ещё
      нет) — насос уже работает: entry = asof, age = asof - первый_активный, start_day 0.
    * БАЗА между насосами — свежий насос ставится СЕЙЧАС: entry = asof, age = 0,
      start_day 0. Возраст 0, а не остаточный возраст прошлого пробега: пробег закрыт,
      новый насос начинает с нуля (и со своей детской смертностью).

    Дата ввода — из плана (`active_fleet_by_month`), а не из паспорта: парк считается по
    плану, и вход обязан считаться тем же критерием.

    Допущение — «новая добывающая скважина Мирнинского = УЭЦН»: способа эксплуатации в
    плане нет. Проверено на истории: с 2024-01 по 2026-03 ВСЕ добывающие по плану
    скважины Mc есть в нашей строгой ЭЦН-популяции (промахов 0 из 14/32/44/45/58).
    Для другого УН это надо перепроверять, а не переносить.

    ``prefixes`` включает "NE_" (Нежданинское): это ДРУГОЕ месторождение того же
    Мирнинского УН, 3 добывающие ВНС (NE_4001/4002/4003), которых нет в нашей Mc-истории.
    Модель cal c0 k2 подогнана на Mc/Mr, поэтому по решению пользователя они включаются,
    НО их источник в форме помечается отдельно (`pred_source`), чтобы было явно видно:
    параметры перенесены с Mc на другой пласт (допущение сильнее, чем «новая Mc = ЭЦН»).
    Nlt-скважины Мирнинского в прогноз не попадают сами — у них нулевая добыча по плану.

    Подход взят из конвейера, а не выдуман: `layers.build_well_states` тоже брал
    вселенную из плана (`crosswalk.load_plan().producers`), а не из пробегов, и ставил
    новой скважине age_pmf={0:1.0} («new_start»).

    ⚠ Чего у конвейера НЕ надо брать: гейт `include_primary` при
    esp_scope_policy="conservative" выбрасывал ВНС без ЭЦН-следа (нет строки ГТМ, нет
    техрежима) как "excluded_non_esp" — а у не введённой скважины следа и не может
    быть. Из 49 новых Mc в прогноз доезжали 18, а 31 молча выпадала ИЗ ЧИСЛИТЕЛЯ,
    оставаясь в знаменателе (парк-то плановый). Здесь берутся все: история говорит,
    что добывающая Mc — это ЭЦН.
    """
    fleet = active_fleet_by_month(master_path=master_path)
    asof_per = pd.Period(asof, "M")
    first_seen: dict[str, pd.Period] = {}
    for month, wells in sorted(fleet.items()):
        per = pd.Period(month, "M")
        for code in wells:
            if str(code).startswith(prefixes) and code not in live_codes:
                first_seen.setdefault(str(code), per)
    # первый активный месяц НЕ РАНЬШЕ asof — момент постановки свежего насоса
    first_from_asof: dict[str, pd.Period] = {}
    for month, wells in sorted(fleet.items()):
        per = pd.Period(month, "M")
        if per < asof_per:
            continue
        for code in wells:
            if str(code).startswith(prefixes) and code not in live_codes:
                first_from_asof.setdefault(str(code), per)
    last_per = asof_per + (months - 1)
    rows = []
    for code, comm in sorted(first_seen.items()):
        entry_per = first_from_asof.get(code)
        if entry_per is None or entry_per > last_per:
            continue                                  # свежий насос — за горизонтом
        existing = code in codes
        if existing or comm >= asof_per:
            # БАЗА между насосами, либо ВНС, вводимая на/после asof: свежий насос
            entry = entry_per.start_time
            start_day = max(0, int((entry - asof.normalize()).days))
            age = 0.0
        else:
            # ВНС, уже добывавшая до asof: насос работает, у него есть возраст
            entry = asof.normalize()
            start_day = 0
            age = float(max(0, int((asof.normalize() - comm.start_time).days)))
        rows.append({"code": code, "entry": entry, "start_day": start_day,
                     "age": age, "category": "БАЗА" if existing else "ВНС"})
    return pd.DataFrame(rows, columns=["code", "entry", "start_day", "age", "category"])


def live_fleet(pop: pd.DataFrame, asof: pd.Timestamp, field: str = "Mc") -> pd.DataFrame:
    """Насосы, РАБОТАЮЩИЕ на ``asof``: code/install/age/category, готово для payload.

    Живой = нет записанного подъёма ИЛИ подъём ПОЗЖЕ asof. Второе важно: `build(asof)`
    оставляет в `end` реальную дату подъёма, даже если она на несколько дней позже asof
    (для дожития такой пробег цензурируется на asof, но `end` не стирается). Фильтр
    `end.isna()` в одиночку выбрасывал 3 насоса Mc, поднятых через 1-5 дней после asof
    (MC_514, MC_5104, MR_1707) — на дату расчёта они ещё работают.

    Когорта 2024+ здесь НЕ применяется: это правило ПОДГОНКИ, а прогнозировать надо
    любой работающий насос. На Mc живых пре-2024 монтажей всё равно нет (61 = 61).
    """
    mc = pop[pop["field"] == field]
    live = mc[mc["end"].isna() | (mc["end"] > asof)].copy()
    live["age"] = (asof - live["install"]).dt.days.astype(float)
    live["raw_id"] = live["code"]
    live["category"] = "БАЗА"
    return live.reset_index(drop=True)


def plan_metadata(codes, asof: pd.Timestamp, master_path=None) -> pd.DataFrame:
    """Куст и ДебН (т/сут) из плана ПП для набора скважин: code/cluster/oil_rate.

    Куст = `producer_meta["pad"]`. ДебН = `registry_oil_rate` в месяц ``asof`` (суточный
    дебит нефти, т/сут); если в этот месяц ноль/нет — последний ненулевой до asof, иначе
    первый ненулевой после (у ВНС добыча только впереди).
    """
    from analysis.workflows.production_risk import crosswalk as _cw

    plan = _cw.load_plan(master_path=master_path) if master_path else _cw.load_plan()
    pm = plan.producer_meta
    ro = plan.registry_oil_rate
    asof_m = pd.Period(asof, "M").strftime("%Y-%m")
    cols = list(ro.columns)
    rows = []
    for code in codes:
        pad = str(pm.loc[code, "pad"]).strip() if code in pm.index else ""
        oil = None
        if code in ro.index:
            s = ro.loc[code]
            if asof_m in s.index and s.get(asof_m, 0) > 0:
                oil = float(s[asof_m])
            else:
                past = [c for c in cols if c <= asof_m and s.get(c, 0) > 0]
                fut = [c for c in cols if c > asof_m and s.get(c, 0) > 0]
                if past:
                    oil = float(s[past[-1]])
                elif fut:
                    oil = float(s[fut[0]])
        rows.append({"code": code, "cluster": pad or None, "oil_rate": oil})
    return pd.DataFrame(rows, columns=["code", "cluster", "oil_rate"])


def history_match(model: SurvivalModel, pop: pd.DataFrame, *, first: str, last: str,
                  theta_by_code: dict[str, float] | None = None,
                  all_cause: bool = False, cohort_only: bool = False,
                  fleet_by_month: dict[str, set[str]] | None = None,
                  field: str = "Mc") -> pd.DataFrame:
    """Помесячно: ФАКТ против ожидания модели + % от парка. Обновления НЕТ.

    ``cohort_only=False`` (по умолчанию) — берутся ВСЕ пробеги Mc, включая монтажи до
    2024. Когорта 2024+ — правило для ПОДГОНКИ модели (чтобы аномалия до-2023 её не
    смещала), а ФАКТ это наблюдение: пробег, смонтированный в 2023 и отказавший в
    2024, отказал по-настоящему и обязан попасть в факт. Фильтр когорты съедал 3 из 47
    отказов (6%): 2024-02, 2024-10, 2025-04 — из-за этого факт расходился с прежними
    релизами. Знаменатель «в работе» считается по тем же пробегам, иначе % поедет.

    ``all_cause`` должен совпадать с тем, на чём подогнан ``model``: False = отказы
    (ГТМ в цензуру), True = любые подъёмы. Иначе сверяется одно с другим.

    Ожидание за месяц = сумма по пробегам, живым на начало месяца, величины
    1 - (S(a+d)/S(a))^theta, где a — возраст на начало месяца, d — сутки экспозиции
    внутри месяца. Условие «жив на начало месяца» берётся из ФАКТА, поэтому это
    сверка на шаг вперёд, а не самостоятельный прогноз.

    Назначение — УБЕДИТЬСЯ ГЛАЗАМИ, что модель считает то же, что и факт; это не
    метрика калибровки (честная — бэктест с фактической экспозицией, O/E 0.98).

    «В работе» = АКТИВНЫЙ ДОБЫВАЮЩИЙ парк из плана ПП (``fleet_by_month``, см.
    `active_fleet_by_month`), а не число пробегов с непонятым насосом. Смонтированный
    насос != добывающая скважина: на Мирнинском 2024-10 пробегов было 36, а добывающих
    по плану — 19, т.е. знаменатель завышался вдвое и интенсивность занижалась вдвое.
    Если ``fleet_by_month`` не передан, парк считается по скважинам с живым пробегом
    (старое поведение) — только для тестов и отладки без плана.
    """
    # ``field`` по умолчанию "Mc" — обратная совместимость; `vt_forecast` передаёт "Vt".
    # Когорта 2024+ (`cohort_only`) — правило Мирнинского, на другие поля не переносится.
    if cohort_only and field != "Mc":
        raise ValueError("cohort_only — правило Мирнинского, для %s недопустимо" % field)
    mc = (P.select(pop, "Mc") if cohort_only
          else pop[pop["field"] == field]).copy()
    mc["is_event"] = (mc["end"].notna().astype(int) if all_cause
                      else (mc["event"] == 1).astype(int))
    rows = []
    for per in _months(pd.Period(first, "M").start_time, pd.Period(last, "M").start_time):
        m0, m1 = per.start_time, per.end_time.normalize()
        exp_n = 0.0
        fact_n = 0
        wells: set[str] = set()
        for r in mc.itertuples():
            if r.install > m1:
                continue
            end = r.end if pd.notna(r.end) else None
            if end is not None and end < m0:
                continue                                  # уже поднят до месяца
            a0 = float((m0 - r.install).days)
            if a0 < 0:
                a0 = 0.0
            stop = min(m1, end) if end is not None else m1
            d = float((stop - max(m0, r.install)).days) + 1.0
            if d <= 0:
                continue
            wells.add(str(r.code))
            th = 1.0 if theta_by_code is None else float(theta_by_code.get(r.code, 1.0))
            s0 = float(model.survival(a0)) ** th
            s1 = float(model.survival(a0 + d)) ** th
            exp_n += (1.0 - s1 / s0) if s0 > 0 else 0.0
            if end is not None and m0 <= end <= m1 and r.is_event == 1:
                fact_n += 1
        fleet = (len(fleet_by_month.get(str(per), ()))
                 if fleet_by_month is not None else len(wells))
        rows.append({"месяц": str(per), "в работе": fleet,
                     "факт": fact_n, "модель ожидает": exp_n})
    df = pd.DataFrame(rows)
    df["факт, % от парка"] = (100.0 * df["факт"] / df["в работе"].replace(0, np.nan)).round(1)
    df["модель, % от парка"] = (100.0 * df["модель ожидает"] / df["в работе"].replace(0, np.nan)).round(1)
    df["факт/модель"] = df["факт"] / df["модель ожидает"].replace(0, np.nan)
    return df


def _sim_frame(live: pd.DataFrame, entrants: pd.DataFrame | None) -> pd.DataFrame:
    """Живые насосы + ВНС в одной таблице симуляции: code / age / start_day / новая."""
    sim = live[["code", "age"]].copy()
    sim["start_day"] = 0
    sim["новая"] = False
    if entrants is not None and len(entrants):
        e = entrants[["code", "age", "start_day"]].copy()
        e["новая"] = True
        sim = pd.concat([sim, e], ignore_index=True)
    return sim


def forecast(model: SurvivalModel, live: pd.DataFrame, *, asof: pd.Timestamp, months: int = 18,
             theta_by_code: dict[str, float] | None = None,
             fleet_by_month: dict[str, set[str]] | None = None,
             entrants: pd.DataFrame | None = None,
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ожидаемые СОБЫТИЯ вперёд, С обновлением (отказавший насос меняется сразу).

    Что именно за событие — задано моделью: `fit_mc_model(all_cause=False)` ⇒ отказы,
    `all_cause=True` ⇒ любые подъёмы. Модель здесь этого не знает, поэтому колонка
    названа нейтрально.

    ``entrants`` (см. `plan_entrants`) — ВНС: новый насос встаёт в симуляцию со своего
    месяца ввода в возрасте 0 и дальше живёт по той же модели. Без них числитель и
    знаменатель считаются по РАЗНЫМ популяциям, и прогноз внутренне противоречив.
    Новый насос — не мелочь: у cal c0 k2 масса дня-0 ~3%, т.е. каждая ВНС приносит
    детскую смертность, которой у старого парка уже нет.

    «Парк» = активный ДОБЫВАЮЩИЙ парк из плана ПП (``fleet_by_month``, тот же критерий
    что в истории: «Отработанное время»>0 И (нефть>0 ИЛИ жидкость>0)). Он ОБЯЗАН
    включать те же ВНС, что и ``entrants``, иначе снова разъедется с числителем.
    Без него берётся размер симуляции — это завышает парк (насос стоит и в скважине,
    которая месяц не добывает) и потому занижает % интенсивности.

    Возвращает (помесячно, по скважинам-помесячно).
    """
    horizon = int(months * 31 + 5)
    sim = _sim_frame(live, entrants)
    max_age = int(sim["age"].max()) + horizon + 2
    q = model.daily_fail_prob(max_age)          # len == max_age + 1 (возрасты 0..max_age)
    per_list = _months(asof, asof + pd.DateOffset(months=months - 1))
    day_month = {}
    for i in range(horizon):
        day_month[i] = pd.Period(asof + pd.Timedelta(days=i), "M")

    per_well = []
    for r in sim.itertuples():
        th = 1.0 if theta_by_code is None else float(theta_by_code.get(r.code, 1.0))
        q_eff = 1.0 - np.power(1.0 - q, th)
        u = np.zeros(len(q_eff), dtype=float)
        sd = int(r.start_day)
        if sd <= 0:
            u[int(r.age)] = 1.0
        acc: dict[pd.Period, float] = {}
        surv = u
        for i in range(horizon):
            if sd > 0 and i == sd:
                u[0] = 1.0                      # ВНС: насос появляется в возрасте 0
            fails = u * q_eff
            n_fail = float(fails.sum())
            surv = u - fails
            # Сдвиг возраста явно, БЕЗ np.roll: roll заворачивает самый старый бин в
            # нулевой, где его затирает обновление -> масса молча теряется.
            nxt = np.zeros_like(u)
            nxt[1:] = surv[:-1]
            nxt[0] = n_fail                     # обновление: новый насос в возрасте 0
            u = nxt
            per = day_month[i]
            acc[per] = acc.get(per, 0.0) + n_fail
        assert float(surv[-1]) < 1e-9, "горизонт возрастов мал: масса дошла до края"
        row = {"скв.": r.code, "возраст, сут": int(r.age), "новая": bool(r.новая)}
        row.update({str(p): acc.get(p, 0.0) for p in per_list})
        row["всего за горизонт"] = float(sum(acc.get(p, 0.0) for p in per_list))
        per_well.append(row)
    pw = pd.DataFrame(per_well)
    months_str = [str(p) for p in per_list]
    if fleet_by_month is not None:
        # план кончается раньше горизонта -> держим последний известный парк
        known = [len(fleet_by_month[mo]) for mo in months_str if mo in fleet_by_month]
        tail = known[-1] if known else int(len(sim))
        fleet_col = [len(fleet_by_month.get(mo, ())) or tail for mo in months_str]
    else:
        fleet_col = [int(len(sim))] * len(months_str)
    monthly = pd.DataFrame({
        "месяц": months_str,
        "в работе": fleet_col,
        "ожидаемые события": [float(pw[str(p)].sum()) for p in per_list],
    })
    monthly["модель, % от парка"] = (
        100.0 * monthly["ожидаемые события"] / monthly["в работе"].replace(0, np.nan)).round(1)
    monthly["нарастающим итогом"] = monthly["ожидаемые события"].cumsum()
    return monthly, pw


def residual_median(model: SurvivalModel, age: float, theta: float = 1.0) -> float:
    """Условная медиана остаточного ресурса (ОР) при возрасте `age`.

    Решаем S(age+x)^theta = 0.5 * S(age)^theta. Это ТОЧЕЧНАЯ оценка — именно она
    создаёт ком, если по ней ставить даты (см. модуль-док). Нужна только для колонки
    планировщика ОР/ННО; ресурсный счёт берётся из `forecast()`.
    """
    from scipy.optimize import brentq

    age = max(float(age), 0.0)
    s_a = float(model.survival(age)) ** theta
    if s_a <= 0:
        return float("nan")
    f = lambda x: float(model.survival(x)) ** theta - 0.5 * s_a
    hi = max(age * 4.0, model.b50() * 8.0, 100.0)
    try:
        return float(brentq(f, age + 1e-9, hi)) - age
    except Exception:
        return float("nan")


def well_table(model: SurvivalModel, live: pd.DataFrame, *, asof: pd.Timestamp,
               theta_by_code: dict[str, float] | None = None) -> pd.DataFrame:
    """Строки планировщика по каждому живому насосу — ОР через RMST, не медиану.

    ОР = **RMST(возраст)** = ожидаемый остаточный ресурс в окне (возраст, tau].
    ННО = возраст + ОР = полная ожидаемая наработка на момент отказа (не остаток).
    MRL даётся справочно: его НАПРАВЛЕНИЕ показывает износ, но уровень
    экстраполирует хвост за tau.

    Медиана (`residual_median`) больше НЕ используется: она сильно зависит от выбора
    подгонки (B50 по хорошим фитам гуляет 418..601), тогда как RMST сходится в
    единицы суток. Часы календарные — модель обучена на «Наработке» Свода.
    """
    rows = []
    for r in live.itertuples():
        th = 1.0 if theta_by_code is None else float(theta_by_code.get(r.code, 1.0))
        age = float(r.age)
        orr = model.resource(age, theta=th)
        rows.append({
            "скв.": r.code,
            "дата монтажа": pd.Timestamp(r.install).date(),
            "возраст на сегодня, сут": int(age),
            "Ql, м3/сут": getattr(r, "ql", np.nan),
            "источник Ql": getattr(r, "ql_src", ""),
            "theta (Ql)": round(th, 4),
            "ОР до отказа, сут": round(orr) if np.isfinite(orr) else np.nan,
            "ННО (возраст + ОР), сут": round(age + orr) if np.isfinite(orr) else np.nan,
            "RMST (справочно), сут": round(model.rmst(age, theta=th)),
            "MRL (справочно), сут": round(model.mrl(age, theta=th)),
            "дата отказа": (asof + pd.Timedelta(days=float(orr))).date() if np.isfinite(orr) else None,
        })
    return pd.DataFrame(rows).sort_values("дата отказа").reset_index(drop=True)


def _daily_dates(asof: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    end = (pd.Period(asof, "M") + (months - 1)).end_time.normalize()
    n = int((end - asof.normalize()).days) + 1
    return [asof.normalize() + pd.Timedelta(days=i) for i in range(n)]


def _month_start_on_grid(d: pd.Timestamp, grid_start: pd.Timestamp) -> pd.Timestamp:
    """Snap a date to the first of its month, clamped to the forecast grid start.

    The statistical model yields a monthly failure PROBABILITY, not a dated event —
    the day within a month carries no information.  Anchoring every predicted failure
    at month start states exactly that, instead of implying a spurious day via
    inverse-CDF spreading (that spread was a repair-SCHEDULE nicety; this deliverable
    forecasts FAILURES, a separate task).
    """
    return max(d.replace(day=1), grid_start)


def _payload_row(r, event_dates: list[pd.Timestamp], date_index: dict, n_days: int,
                 asof: pd.Timestamp, model: "SurvivalModel | None" = None,
                 entry: pd.Timestamp | None = None) -> dict:
    statuses = [1] * n_days
    firsts = [d for d in event_dates if d in date_index]
    for d in firsts:
        statuses[date_index[d]] = 0
    age = float(r.age)
    d2w = float((min(firsts) - asof).days) if firsts else None
    if d2w is not None:
        prob_nno = round(age + d2w)
    elif model is not None:
        # Событий не досталось (их 52 на 58 скважин — эти в очереди по риску последние),
        # но ННО назвать надо. Берём age + MRL: отказ ложится ЗА горизонт, что и
        # означает «в сетке нуля нет».  Смесь w=0.9 сюда не годится — она даёт
        # age+ОР = 391..396 сут при горизонте 548, т.е. противоречила бы сетке.
        prob_nno = round(age + model.mrl(age))
    else:
        prob_nno = None
    # Дата активации = дата постановки ТЕКУЩЕГО насоса: для живого — его монтаж, для
    # новой строки (ВНС/БАЗА между насосами) — ``entry`` (месяц ввода из плана).
    if entry is not None:
        activation = pd.Timestamp(entry)
    else:
        inst = getattr(r, "install", None)
        activation = pd.Timestamp(inst) if inst is not None and pd.notna(inst) else None
    # «Наработка факт» = текущая наработка. У свежего насоса (ВНС / БАЗА между насосами,
    # возраст 0) наработка и есть 0 — по требованию пользователя показываем 0, а не
    # пусто: насос ещё не отработал ни дня, и ОР = Вероятностный прогноз − 0 = полный ННО.
    fresh = entry is not None and age <= 0
    return {
        "category": getattr(r, "category", "") or "",
        "license_area": getattr(r, "license_area", "Мирнинский УН"),
        "cluster_name": getattr(r, "cluster", None),
        "well_name": getattr(r, "raw_id", r.code) or r.code,
        "catboost_nno": None,
        "probabilistic_nno": prob_nno,       # наработка на подъём = возраст + расстояние
        "days_to_workover": round(d2w) if d2w is not None else None,
        "actual_nno": 0 if fresh else round(age),
        # обычно "survival"; для скважин ВНЕ охвата модели (другое месторождение)
        # источник помечается явно, чтобы было видно, что параметры перенесены с Mc.
        "used_prediction_source": getattr(r, "pred_source", None) or "survival",
        "oil_rate": getattr(r, "oil_rate", None),
        "activation_date": activation.date().isoformat() if activation is not None else None,
        "event_dates": [d.date().isoformat() for d in event_dates],
        "statuses": statuses,
    }


def debt_to_date(model: "SurvivalModel", age: float, theta: float = 1.0,
                 step: float = 30.0) -> float:
    """ДОЛГ, накопленный насосом ДО даты расчёта: сумма risk_30d от возраста 0 до `age`.

    Считается ТОЙ ЖЕ арифметикой, что и forward-накопление в `allocate_events`
    (сумма помесячных `1 - S(t+step)/S(t)`), иначе seed и приросты были бы в разных
    единицах.

    Зачем seed вообще нужен (решение пользователя 2026-07-20). Без него все насосы
    входят в прогноз с нулём, и очередь определяется ТОЛЬКО скоростью накопления. На
    страте с beta<1 (Vt_nonsour) или beta≈1 (Vt_sour) это ставит СВЕЖИЕ насосы в начало
    очереди: на Vt 11 из 12 строк с ННО <= 31 сут — насосы возраста 0, одна с ННО = 0
    (отказ в день монтажа). Такое в план не ставится. Seed по истории делает очередь
    монотонной по возрасту на любой форме опасности.

    ⚠ Цена решения: это ранжирование по НАКОПЛЕННОЙ опасности H(a) = -ln S(a), которая
    растёт с возрастом ВСЕГДА. Оно жёстко задаёт «старые первыми» независимо от beta,
    т.е. на страте с убывающей опасностью противоречит самой модели (свежий насос
    Vt_nonsour объективно рискованнее: 0.0509/мес против 0.0252 у 2137-суточного).
    Поэтому seed меняет ПОРЯДОК, но НЕ месячные итоги: сколько отказов в месяце
    по-прежнему задаёт E_m, и суммарный счёт не двигается.
    """
    age = max(0.0, float(age))
    if age <= 0:
        return 0.0
    total = 0.0
    t = 0.0
    while t < age:
        d = min(step, age - t)
        s0 = float(model.survival(t)) ** theta
        s1 = float(model.survival(t + d)) ** theta
        if s0 > 0:
            total += 1.0 - s1 / s0
        t += d
    return total


GUARD_MIN_AGE_DAYS = 90.0
"""Насос моложе этого возраста не ставится в сетку — см. `allocate_events`."""


def allocate_events(monthly: pd.DataFrame, per_well: pd.DataFrame,
                    initial_debt: dict[str, float] | None = None,
                    min_age_days: float = GUARD_MIN_AGE_DAYS,
                    entry_month: dict[str, int] | None = None,
                    ) -> dict[str, list[pd.Period]]:
    """Кто и когда отказывает. Оба входа — выход `forecast()`, другого счёта здесь нет.

    СКОЛЬКО. Месячное ожидание E_m дробное (2.78, 3.12, ...), а сетка ставит целые.
    Идём по месяцам и переносим остаток: n_m = floor(E_m + carry), carry -= n_m.
    Флотская сумма сохраняется (75 при ΣE=75.3, теряется хвостовой остаток).

    КТО. Две величины, их легко перепутать — держим имена раздельно:

    * ``risk_30d`` — ПРИРОСТ за месяц: ожидание отказов насоса за этот месяц, ПРЯМО ИЗ
      ``per_well``. Это ожидаемое ЧИСЛО событий (в `forecast` отказавшая масса
      возвращается в возраст 0), а не P(хотя бы один отказ); при 0.03-0.17 разница
      1-9%, но она есть.
    * ``debt`` — ДОЛГ: накопленная сумма ``risk_30d`` **с даты расчёта**. У ВСЕХ
      насосов стартует с нуля: прошлая наработка в долг не входит.

    Каждый месяц долг растёт на risk_30d; событие достаётся насосам с наибольшим
    накопленным долгом, у получившего долг уменьшается на 1. Так выданное каждому
    насосу идёт за его собственным ожиданием (замер на парке Mc: |выдано - ожидание|
    <= 0.59 по каждой скважине), а месячный итог точен: Σ приростов за месяц = E_m по
    построению (`forecast` считает monthly как сумму per_well).

    Предел целочисленности — не дефект, а арифметика. За 18 мес ожидание СТАРОГО насоса
    0.90 отказа, ВНС — 0.47; ни у кого не выходит 1. Значит 75 целых событий на 107
    скважин обязаны лечь как «75 скважин по одному отказу», и по величине ожидания это
    все 58 старых + 17 новых. Поэтому в разрезе групп сетка даёт 58/17 против ожидания
    52.0/23.2: 0.90 нельзя выдать, можно выдать только 0 или 1. Флотская сумма при этом
    точна (75 при ΣE=75.3), и именно она идёт на лист интенсивности.

    Почему risk_30d берётся из ``per_well``, а не «самый рискованный сейчас» по возрасту.
    Прежнее
    правило имело ДВА дефекта, и оба вскрылись на ВНС:

    1. Строгая очередь по риску. Старый насос всегда выигрывает у молодого (месячный
       риск 6.4% при 730 сут против 4.7% при 0), поэтому 49 новых скважин получали 9
       событий — вдвое меньше даже целочисленного предела (17). ВНС молча выпадали из
       сетки, оставаясь в знаменателе интенсивности.
    2. Свой, второй счёт. Раскладка трекала возраст ТОЧКОЙ (age += 30.4, при отказе 0),
       а `forecast` ведёт РАСПРЕДЕЛЕНИЕ массы по возрастам. Две разные динамики из одной
       модели неизбежно расходятся; единственный способ этого не допустить — не считать
       второй раз, а брать ожидания оттуда же, откуда их берёт график.

    ``min_age_days`` (guard, по умолчанию 90) — насос моложе порога не ставится в сетку.
    Без него на страте с beta <= 1 свежий насос попадает в первый же месяц и в план
    уходит строка с ННО = 0..31 сут (замер на Vt: 12 таких строк из 254, одна с ННО = 0 —
    отказ в день монтажа). Цена guard по инвариантности ничтожна: max|O/E-1| меняется на
    0.00-0.05, при этом все строки с ННО <= 90 исчезают. ``entry_month`` даёт индекс
    месяца ввода для ВНС (у живых насосов 0), иначе их возраст считался бы от начала
    горизонта, а не от монтажа.

    ⚠ ``initial_debt`` (сеяние долга историей насоса) ПРОВЕРЕНО И ОТКЛОНЕНО 2026-07-20.
    Оно чинит порядок и ННО, но проваливает тест инвариантности: если взять выданную
    сетку как данные и переоценить опасность, сеяние даёт max|O/E-1| = 12.4 (Ya) /
    **38.7 (Vt)** / 14.0 (Mc) против 3.6 / 3.4 / 14.0 у несеяного правила, т.е. фабрикует
    износ, которого модель не утверждает. Параметр оставлен только для воспроизведения
    эксперимента; в поставке НЕ использовать. Разбор — `agents/analyses/
    event_allocation_rule_research.md`.

    Порядок по возрасту правило НЕ задаёт — его задаёт форма опасности: при beta > 1 (Mc)
    «старые первыми» (корреляция Спирмена возраст↔дата на `Сводпрогноз_Mc.xlsx` = -0.992),
    при beta < 1 (Vt nonsour) первыми идут МОЛОДЫЕ, при beta ≈ 1 порядка нет вовсе.

    ⚠ Порядок задаёт НЕ это правило, а ФОРМА ОПАСНОСТИ модели. Долг у всех стартует с
    нуля, поэтому решают только приросты внутри горизонта. У Mc (`cal c0 k2`, beta~1.32)
    risk_30d РАСТЁТ с возрастом (0.0257 при 30 сут -> 0.0391 при 150), отсюда «старые
    первыми». Накопленный за 5 мес долг: возраст 30 -> 0.165, возраст 750 -> 0.325.
    На страте с beta<1 тот же код даёт ОБРАТНЫЙ порядок (Vt_nonsour_Pooled: возраст 30
    -> 0.286, возраст 750 -> 0.171 — молодые первыми), а при beta≈1 порядка нет вовсе
    (Vt_sour_Pooled: 0.844 против 0.857, разброс 1.6%). Переносить это правило на другую
    страту без проверки знака beta нельзя.

    Зачем это заменило прежнюю раскладку (`repair_compat._allocate_event_dates`:
    обратная CDF + хеш-фаза): та ставила дату ЖЕРЕБЬЁВКОЙ, и результат был не просто
    случайным, а анти-физичным. Замер на живом парке Mc: корреляция Спирмена
    возраст↔дата = **-0.177** (шум), из-за чего MC_010 (90 сут, риск 3.4%/мес) получал
    отказ СЕГОДНЯ, а MC_007 (696 сут, риск 6.3%/мес — вдвое выше) — только через 123
    дня.

    ⚠ Это ДЕФОЛТНЫЙ ПОРЯДОК, а не предсказание. Модель даёт по скважинам близкие
    вероятности, и правило разворачивает их в очередь. Это защитимо (объяснимо,
    согласуется с beta2>1 и с ожиданиями модели поскважинно), но очередь — свойство
    ПРАВИЛА, а не сила сигнала в данных.
    """
    months = [pd.Period(x, "M") for x in monthly["месяц"]]
    mcols = [str(p) for p in months]
    E = monthly["ожидаемые события"].to_numpy(dtype=float)
    exp = per_well.set_index("скв.")[mcols]
    out: dict[str, list[pd.Period]] = {str(c): [] for c in exp.index}
    seed = initial_debt or {}
    debt: dict[str, float] = {str(c): float(seed.get(str(c), 0.0)) for c in exp.index}
    age0 = {str(r["скв."]): float(r.get("возраст, сут", 0.0) or 0.0)
            for _, r in per_well.iterrows()}
    entry = entry_month or {}
    carry = 0.0
    for j, per in enumerate(months):
        risk_30d = exp[mcols[j]]     # прирост за месяц; ДОЛГ — это его накопленная сумма
        for code in debt:
            debt[code] += float(risk_30d[code])
        total = E[j] + carry
        n_m = int(np.floor(total))
        carry = total - n_m
        if n_m <= 0:
            continue
        if min_age_days > 0:
            elig = [c for c in debt
                    if age0.get(c, 0.0) + 30.0 * (j - entry.get(c, 0)) >= min_age_days]
        else:
            elig = list(debt)
        for code in sorted(elig, key=lambda c: debt[c], reverse=True)[:n_m]:
            out[code].append(per)
            debt[code] -= 1.0
    return out


def repair_payload(live: pd.DataFrame, model: SurvivalModel, *, asof: pd.Timestamp,
                   horizon_months: int, method: str = "statistical",
                   theta_by_code: dict[str, float] | None = None,
                   entrants: pd.DataFrame | None = None) -> dict:
    """Полезная нагрузка формы «Прогноз_ремонтов» для одного из двух методов.

    Это прогноз ОТКАЗОВ (планирование ремонтов — отдельная задача).

    Метод один — СТАТИСТИЧЕСКИЙ: месячные вероятности отказа из survival-модели.
    Сколько отказов в месяце — перенос дробного остатка; кто отказывает — самые
    рискованные насосы (см. `allocate_events`). Отказ ставится на 1-е ЧИСЛО своего
    месяца: день внутри месяца модель не определяет.

    Детерминированный вариант (0 на возраст+ОР, дальше обновления) УДАЛЁН: точечная
    оценка складывала подъёмы в ком (пик 15-18/мес против 6 у статистики, 7-10 занятых
    месяцев из 18) и целила в наименее рисковые насосы. Это была политика планирования
    ремонтов — отдельная задача, а здесь прогнозируются отказы.

    Обе версии дают идентичную форму (строка на скважину + суточные 0/1), поэтому
    пишутся одним repair_compat.write_excel.
    """
    dates = _daily_dates(asof, horizon_months)
    date_index = {d: i for i, d in enumerate(dates)}
    asof_n = asof.normalize()
    events_by_code: dict[str, list[pd.Timestamp]] = {}
    if method == "statistical":
        monthly, per_well = forecast(model, live, asof=asof, months=horizon_months,
                                     theta_by_code=theta_by_code, entrants=entrants)
        # Сеяния долга НЕТ (проверено и отклонено — см. `allocate_events`); абсурдные
        # ННО закрывает guard по возрасту, который не трогает инвариантность.
        entry_j = {}
        if entrants is not None and len(entrants):
            base = pd.Period(asof, "M")
            entry_j = {str(r.code): max(0, (pd.Period(pd.Timestamp(r.entry), "M") - base).n)
                       for r in entrants.itertuples()}
        by_period = allocate_events(monthly, per_well, entry_month=entry_j)
        for code, periods in by_period.items():
            snapped = [_month_start_on_grid(p.start_time, dates[0]) for p in periods]
            events_by_code[code] = [d for d in snapped if dates[0] <= d <= dates[-1]]
    else:
        raise ValueError("method must be 'statistical' (детерминированный удалён)")

    rows = [_payload_row(r, events_by_code.get(r.code, []), date_index, len(dates), asof_n, model)
            for r in live.itertuples()]
    # ВНС — такие же строки формы: у них есть отказы, и без них сетка противоречит
    # интенсивности (парк по плану растёт 67 -> 82, а отказывали бы только старые).
    if entrants is not None and len(entrants):
        rows += [_payload_row(r, events_by_code.get(r.code, []), date_index, len(dates),
                              asof_n, model, entry=pd.Timestamp(r.entry))
                 for r in entrants.itertuples()]
    label = "статистический (месячная вероятность отказа, отметка на 1-е число месяца)"
    # Форма обязана нести саму модель: без параметров распределения цифры в сетке
    # невоспроизводимы и неотличимы от любой другой подгонки.
    model_line = (
        "Модель cal c%.0f k2: смесь двух Вейбуллов, "
        "S(t) = w1·exp(-(t/η1)^β1) + (1-w1)·exp(-(t/η2)^β2). "
        "w1=%.4f · β1=%.3f η1=%.2f · β2=%.3f η2=%.2f."
        % (model.cut_days, model.w1, model.beta1, model.eta1, model.beta2, model.eta2)
    )
    resource_line = (
        "Ресурс считается через RMST, НЕ через медиану: RMST(0)=%.0f сут (ожидаемая "
        "наработка до отказа в окне до tau=%.0f сут), MRL(0)=%.0f сут (справочно, "
        "экстраполирует хвост). Медиана B50=%.0f сут — только для диагностики."
        % (model.rmst(0.0), model.tau, model.mrl(0.0), model.b50())
    )
    fit_line = (
        "Подгонка: Мирнинский (Mc+Mr), монтажи с %s, событие = ОТКАЗ (ГТМ и работающие "
        "насосы — цензура), левое усечение c=%.0f сут; пробегов %d, отказов %d, "
        "цензурировано %d, отброшено ранних %d. Часы: «Наработка (сут)» ≈ календарь."
        % (C.MC_INSTALL_COHORT_START.isoformat(), model.cut_days, model.n_runs,
           model.n_events, model.n_censored, model.n_cut)
    )
    return {
        "start_date": dates[0].date().isoformat(),
        "end_date": dates[-1].date().isoformat(),
        "dates": [d.date().isoformat() for d in dates],
        "used_feature_columns": ["stratum", "operating_age"],
        "missing_feature_columns": [],
        "rows": rows,
        "notes": [
            "Мирнинский УН (Mc + Mr), метод: %s." % label,
            model_line,
            resource_line,
            fit_line,
            "«Вероятностный прогноз» = наработка сегодня + расстояние до отказа (полная ННО).",
            "0 — день предсказанного ОТКАЗА, 1 — остальные дни.",
            "Это прогноз ОТКАЗОВ. Планирование ремонтов (все подъёмы, МРП) — отдельная задача.",
        ],
    }


def c4_gate(model: SurvivalModel, pop: pd.DataFrame, theta_by_code: dict[str, float],
            *, first: str = "2024-01", last: str = "2026-06") -> pd.DataFrame:
    """Гейт Workstream C: даёт ли слой Ql выигрыш ВНЕ выборки (2025-01..2026-06)?

    Сравниваем MAE месячного счёта и суммарное факт/модель на окне оценки для
    базовой модели и базовой+Ql. Слой промотируется в базовый прогноз, только если
    MAE падает и факт/модель не ломается.
    """
    base = history_match(model, pop, first=first, last=last)
    withql = history_match(model, pop, first=first, last=last, theta_by_code=theta_by_code)
    rows = []
    for lab, df in (("базовая", base), ("базовая + Ql", withql)):
        oos = df[df["месяц"] >= OOS_SCORE_FIRST]
        ins = df[df["месяц"] <= OOS_FIT_LAST]
        err = (oos["факт"] - oos["модель ожидает"]).abs()
        rows.append({
            "модель": lab,
            "MAE вне выборки, подъёмов/мес": float(err.mean()),
            "факт/модель вне выборки": float(oos["факт"].sum() / oos["модель ожидает"].sum()),
            "факт/модель в выборке": float(ins["факт"].sum() / ins["модель ожидает"].sum()),
            "месяцев в оценке": int(len(oos)),
        })
    out = pd.DataFrame(rows)
    base_mae = float(out.loc[0, "MAE вне выборки, подъёмов/мес"])
    gain = (base_mae - float(out.loc[1, "MAE вне выборки, подъёмов/мес"])) / base_mae
    out["выигрыш MAE, %"] = ["", round(100.0 * gain, 2)]
    # Порог, а не просто «ниже». Критерий «MAE строго ниже» — штамп: любой
    # коэффициент с верным знаком чуть сдвинет MAE вниз, а на 18 месяцах малых
    # счётов сдвиг в 1% неотличим от шума. Слой промотируется только с запасом.
    out["вердикт"] = [
        "",
        ("ПРОШЁЛ (выигрыш %.1f%% > %.0f%%)" % (100 * gain, 100 * MIN_REL_GAIN))
        if gain > MIN_REL_GAIN else
        ("НЕ ПРОШЁЛ (выигрыш %.1f%% <= %.0f%% — в пределах шума) -> слой остаётся сценарным"
         % (100 * gain, 100 * MIN_REL_GAIN)),
    ]
    return out
