"""Прогноз отказов по Верхнетирскому (Vt) — та же машинерия, что у Мирнинского,
но модель НЕ одна.

Почему нужен отдельный модуль
-----------------------------
`esp_forecast.forecast` / `repair_payload` / `history_match` берут ОДНУ `SurvivalModel` на весь
парк. Для Mc это верно: там одна страта (`Mc_nonsour_Pooled`). У Vt страт восемь —
sour/nonsour x brt/slb/oth (+Pooled), и они различаются не только масштабом, но и ФОРМОЙ:

    Vt_sour_*     beta ~= 1.01-1.09  -> опасность ПОСТОЯННА, возраст не несёт порядка
    Vt_nonsour_*  beta ~= 0.73-0.84  -> опасность УБЫВАЕТ, порядок ОБРАТНЫЙ к Mc

Пулить их нельзя (жизнь различается втрое, см. `project_vt_weibull_scan`), поэтому здесь
прогноз считается ПО СТРАТАМ и сшивается:

1. каждая скважина резолвится в свою страту реестра (`StrataModel.resolve`);
2. `esp_forecast.forecast` гоняется отдельно на каждой группе — своя кривая, свой
   `daily_fail_prob`;
3. `per_well` склеивается, `monthly` пересчитывается СУММОЙ по склейке, а знаменатель
   «в работе» берётся общий (парк плана), иначе интенсивность считается по разным
   популяциям;
4. `allocate_events` запускается ОДИН раз на склейке — перенос дробного остатка обязан
   быть ФЛОТСКИМ, иначе каждая страта теряет свой хвост и тотал разъезжается.

⚠ Что отсюда НЕ следует переносить на Vt механически
----------------------------------------------------
Правило долга переносимо, а вывод «старые первыми» — НЕТ: он держится на beta>1 у Mc.
На `Vt_nonsour_*` тот же код ставит первыми МОЛОДЫЕ насосы (накопленный за 5 мес долг:
возраст 30 -> 0.286 против возраст 750 -> 0.171), а на `Vt_sour_*` порядка нет вовсе
(0.844 против 0.857, разброс 1.6%). Это свойство модели, не дефект раскладки; см.
`agents/analyses/mc_prognosis_debt_ordering_handoff.md` §5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk.survival import StrataModel

VT_PLAN_FIELD = "Верхнетирский УН"
VT_PREFIXES = ("VT_", "VTI_", "VTB_", "VTBB_")
# tau = 95-й перцентиль наблюдённой наработки Vt (t_cal), для RMST; у Mc это 475.
VT_TAU = 760.0


def model_from_registry(params: dict[str, float], *, tau: float = VT_TAU) -> M.SurvivalModel:
    """Строка реестра -> SurvivalModel, чтобы переиспользовать всю машинерию esp_forecast."""
    return M.SurvivalModel(
        w1=float(params["w1"]), beta1=float(params["beta1"]), eta1=float(params["eta1"]),
        beta2=float(params["beta2"]), eta2=float(params["eta2"]),
        n_runs=0, n_events=0, n_censored=0, n_cut=0, cut_days=0.0, tau=tau,
    )


def stratum_of(pop: pd.DataFrame) -> pd.Series:
    """code -> (h2s_class, contractor_group) по популяции Vt.

    Сера — свойство ФЛЮИДА скважины, подрядчик — свойство пробега; для скважины берём
    последний известный пробег (он же тот, что сейчас в земле).
    """
    vt = pop[pop["field"] == "Vt"].sort_values("install")
    last = vt.groupby("code").tail(1).set_index("code")
    return last[["h2s_class", "contractor_group"]]


def resolve_models(codes, strata: pd.DataFrame, registry: StrataModel,
                   *, default: tuple[str, str] = ("nonsour", "Pooled"),
                   ) -> tuple[dict[str, M.SurvivalModel], pd.DataFrame]:
    """Каждой скважине — её модель. Возвращает (code -> SurvivalModel, таблица резолва).

    Скважины без истории (ВНС) не имеют ни серы, ни подрядчика ⇒ идут в ``default``.
    Это ЯВНОЕ допущение, а не молчаливый фолбэк: колонка `источник` в таблице резолва
    помечает такие строки, чтобы их доля была видна в книге.
    """
    out: dict[str, M.SurvivalModel] = {}
    rows = []
    cache: dict[str, M.SurvivalModel] = {}
    for code in codes:
        if code in strata.index:
            h2s = str(strata.at[code, "h2s_class"])
            ctr = str(strata.at[code, "contractor_group"])
            src = "история"
        else:
            h2s, ctr = default
            src = "ВНС: страта по умолчанию"
        params, key = registry.resolve("Vt", h2s, ctr)
        if key not in cache:
            cache[key] = model_from_registry(params)
        out[code] = cache[key]
        rows.append({"скв.": code, "страта": key, "источник": src,
                     "w1": round(params["w1"], 4), "beta1": round(params["beta1"], 4),
                     "eta1": round(params["eta1"], 2), "beta2": round(params["beta2"], 4),
                     "eta2": round(params["eta2"], 2)})
    return out, pd.DataFrame(rows)


def _group_by_model(codes, models: dict[str, M.SurvivalModel]) -> dict[M.SurvivalModel, list[str]]:
    groups: dict[M.SurvivalModel, list[str]] = {}
    for c in codes:
        groups.setdefault(models[c], []).append(c)
    return groups


def forecast_multi(models: dict[str, M.SurvivalModel], live: pd.DataFrame, *,
                   asof: pd.Timestamp, months: int,
                   theta_by_code: dict[str, float] | None = None,
                   fleet_by_month: dict[str, set[str]] | None = None,
                   entrants: pd.DataFrame | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`esp_forecast.forecast` по стратам + сшивка. Сигнатура совместима с оригиналом.

    Знаменатель «в работе» берётся ОБЩИЙ (не по стратам): интенсивность — свойство
    парка, а не модели. Если `fleet_by_month` не передан, берётся размер симуляции.
    """
    per_parts = []
    all_codes = list(live["code"])
    if entrants is not None and len(entrants):
        all_codes += list(entrants["code"])

    for model, codes in _group_by_model(all_codes, models).items():
        s = set(codes)
        lv = live[live["code"].isin(s)]
        en = entrants[entrants["code"].isin(s)] if entrants is not None and len(entrants) else None
        if lv.empty and (en is None or en.empty):
            continue
        if lv.empty:
            # forecast требует непустой live-фрейм: подставляем пустой с теми же колонками
            lv = live.iloc[0:0]
        _, pw = M.forecast(model, lv, asof=asof, months=months,
                           theta_by_code=theta_by_code, entrants=en)
        per_parts.append(pw)

    per_well = pd.concat(per_parts, ignore_index=True)
    month_cols = [c for c in per_well.columns
                  if c not in ("скв.", "возраст, сут", "новая", "всего за горизонт")]
    fleet = (
        [len(fleet_by_month.get(m, ())) or len(per_well) for m in month_cols]
        if fleet_by_month is not None else [len(per_well)] * len(month_cols)
    )
    monthly = pd.DataFrame({
        "месяц": month_cols,
        "в работе": fleet,
        "ожидаемые события": [float(per_well[m].sum()) for m in month_cols],
    })
    monthly["модель, % от парка"] = (
        100.0 * monthly["ожидаемые события"] / monthly["в работе"].replace(0, np.nan)).round(1)
    monthly["нарастающим итогом"] = monthly["ожидаемые события"].cumsum()
    return monthly, per_well


def repair_payload_multi(live: pd.DataFrame, models: dict[str, M.SurvivalModel], *,
                         asof: pd.Timestamp, horizon_months: int,
                         theta_by_code: dict[str, float] | None = None,
                         entrants: pd.DataFrame | None = None,
                         fleet_by_month: dict[str, set[str]] | None = None,
                         model_note: str = "") -> dict:
    """Форма «Прогноз_ремонтов» на многострановой модели. Аналог `M.repair_payload`.

    Раскладка `allocate_events` — ОДНА на весь парк (флотский перенос остатка), но
    `Вероятностный прогноз` для скважин без события считается по ЕЁ модели (age + MRL),
    иначе у sour и nonsour получился бы один и тот же остаточный ресурс.
    """
    dates = M._daily_dates(asof, horizon_months)
    date_index = {d: i for i, d in enumerate(dates)}
    asof_n = asof.normalize()

    monthly, per_well = forecast_multi(models, live, asof=asof, months=horizon_months,
                                       theta_by_code=theta_by_code,
                                       fleet_by_month=fleet_by_month, entrants=entrants)
    # Сеяния долга НЕТ (проверено и отклонено: на Vt max|O/E-1| = 38.7 против 3.4 —
    # см. `M.allocate_events`). Абсурдные ННО закрывает guard по возрасту.
    base = pd.Period(asof, "M")
    entry_j = ({str(r.code): max(0, (pd.Period(pd.Timestamp(r.entry), "M") - base).n)
                for r in entrants.itertuples()}
               if entrants is not None and len(entrants) else {})
    by_period = M.allocate_events(monthly, per_well, entry_month=entry_j)
    events_by_code: dict[str, list[pd.Timestamp]] = {}
    for code, periods in by_period.items():
        snapped = [M._month_start_on_grid(p.start_time, dates[0]) for p in periods]
        events_by_code[code] = [d for d in snapped if dates[0] <= d <= dates[-1]]

    rows = [M._payload_row(r, events_by_code.get(r.code, []), date_index, len(dates),
                           asof_n, models.get(r.code))
            for r in live.itertuples()]
    if entrants is not None and len(entrants):
        rows += [M._payload_row(r, events_by_code.get(r.code, []), date_index, len(dates),
                                asof_n, models.get(r.code), entry=pd.Timestamp(r.entry))
                 for r in entrants.itertuples()]

    return {
        "as_of": asof_n,
        "dates": dates,
        "rows": rows,
        "method": "statistical",
        "method_label": "статистический (месячная вероятность отказа, отметка на 1-е число месяца)",
        "notes": [
            "Верхнетирский УН: модель СВОЯ У КАЖДОЙ СТРАТЫ (sour/nonsour x подрядчик) — "
            "пулить их нельзя, жизнь различается втрое.",
            model_note,
            "«Вероятностный прогноз» = наработка сегодня + расстояние до отказа (полная ННО).",
            "0 — день предсказанного ОТКАЗА, 1 — остальные дни.",
            "Это прогноз ОТКАЗОВ. Планирование ремонтов (все подъёмы, МРП) — отдельная задача.",
            "⚠ Порядок отказов задаёт ФОРМА опасности: у Vt_sour beta≈1 (порядка по возрасту "
            "нет), у Vt_nonsour beta<1 (первыми идут МОЛОДЫЕ) — в отличие от Mc, где beta>1.",
        ],
    }


def history_match_multi(models: dict[str, M.SurvivalModel], pop: pd.DataFrame, *,
                        first: str, last: str,
                        theta_by_code: dict[str, float] | None = None,
                        fleet_by_month: dict[str, set[str]] | None = None,
                        ) -> pd.DataFrame:
    """Помесячный факт-vs-модель по Vt: `M.history_match` по стратам, затем сумма.

    Факт и ожидание складываются по стратам; «в работе» берётся общий парк, поэтому
    проценты считаются ПОСЛЕ сшивки, а не усредняются по стратам.
    """
    parts = []
    vt = pop[pop["field"] == "Vt"]
    for (h2s, ctr), g in vt.groupby(["h2s_class", "contractor_group"]):
        # ⚠ Группировать надо ПРОБЕГИ, а не скважины: подрядчик может смениться между
        # пробегами одной скважины, и фильтр `pop[code.isin(codes)]` тогда затягивает в
        # страту чужие пробеги. Именно так факт раздувался 323 -> 491 (в 1.5 раза).
        mdl = None
        for c in g["code"]:
            if c in models:
                mdl = models[c]
                break
        if mdl is None:
            continue
        h = M.history_match(mdl, g, first=first, last=last, field="Vt",
                            theta_by_code=theta_by_code, fleet_by_month=None)
        parts.append(h[["месяц", "факт", "модель ожидает"]])

    if not parts:
        raise ValueError("history_match_multi: ни одна страта Vt не собралась")
    out = parts[0].copy()
    for p in parts[1:]:
        out = out.merge(p, on="месяц", how="outer", suffixes=("", "_r"))
        out["факт"] = out["факт"].fillna(0) + out.pop("факт_r").fillna(0)
        out["модель ожидает"] = out["модель ожидает"].fillna(0) + out.pop("модель ожидает_r").fillna(0)
    out = out.sort_values("месяц").reset_index(drop=True)
    if fleet_by_month is not None:
        out["в работе"] = [len(fleet_by_month.get(m, ())) for m in out["месяц"]]
    else:
        out["в работе"] = np.nan
    out["факт, % от парка"] = (100.0 * out["факт"] / out["в работе"].replace(0, np.nan)).round(1)
    out["модель, % от парка"] = (
        100.0 * out["модель ожидает"] / out["в работе"].replace(0, np.nan)).round(1)
    return out[["месяц", "в работе", "факт", "модель ожидает",
                "факт, % от парка", "модель, % от парка"]]


def c4_gate_multi(models: dict[str, M.SurvivalModel], pop: pd.DataFrame,
                  theta_by_code: dict[str, float], *,
                  first: str = "2024-01", last: str = "2026-06",
                  fleet_by_month: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Даёт ли слой Ql выигрыш ВНЕ выборки? Порт `M.c4_gate` на многострановую модель.

    Окно оценки — `M.OOS_SCORE_FIRST`..last. Слой промотируется в базу, только если MAE
    месячного счёта падает не меньше чем на `M.MIN_REL_GAIN`, и суммарное факт/модель
    при этом не ломается. Ниже порога — слой остаётся СЦЕНАРНЫМ.
    """
    base = history_match_multi(models, pop, first=first, last=last, fleet_by_month=fleet_by_month)
    withql = history_match_multi(models, pop, first=first, last=last,
                                 theta_by_code=theta_by_code, fleet_by_month=fleet_by_month)
    oos = base["месяц"] >= M.OOS_SCORE_FIRST
    rows = []
    for name, df in (("база", base), ("база + Ql", withql)):
        d = df[oos]
        mae = float((d["факт"] - d["модель ожидает"]).abs().mean())
        rows.append({"модель": name, "окно": f"{M.OOS_SCORE_FIRST}..{last}",
                     "MAE мес.": round(mae, 4),
                     "факт": float(d["факт"].sum()),
                     "модель": round(float(d["модель ожидает"].sum()), 1),
                     "O/E": round(float(d["факт"].sum() / max(d["модель ожидает"].sum(), 1e-9)), 3)})
    gain = (rows[0]["MAE мес."] - rows[1]["MAE мес."]) / max(rows[0]["MAE мес."], 1e-9)
    verdict = ("ПРОШЁЛ: слой можно промотировать в базу"
               if gain >= M.MIN_REL_GAIN else
               "НЕ ПРОШЁЛ: слой остаётся СЦЕНАРНЫМ")
    out = pd.DataFrame(rows)
    out["выигрыш MAE"] = ["", "%.1f%% (порог %.0f%%)" % (100 * gain, 100 * M.MIN_REL_GAIN)]
    out["вердикт"] = ["", verdict]
    return out


def live_fleet(pop: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Работающие насосы Vt на asof. Тот же критерий, что у Mc (`end` позже asof тоже живой)."""
    return M.live_fleet(pop, asof, field="Vt")
