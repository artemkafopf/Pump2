"""ПРОГНОЗ ОТКАЗОВ в форме «Прогноз_ремонтов», ТОЛЬКО по Мирнинскому УН (Mc + Mr).

Модель — `esp_forecast.fit_mc_model` = **cal c0 k2**: смесь двух Вейбуллов на
монтажах 2024+, событие = ОТКАЗ (ГТМ и работающие насосы — цензура), БЕЗ усечения
ранних (c=0), стартовую массу дня-0 держит первая компонента. Параметры печатаются
в лист «Summary» самой книгой (см. `esp_forecast.repair_payload`).

⚠ Раньше здесь бралcя projection штатного конвейера (`run.run`), т.е. **другая
модель** — shipped-Вейбулл из бандла `esp_models.csv`, — и раскладка событий обратной
CDF внутри месяца. Теперь книга строится тем же `esp_forecast.repair_payload`, что и
приложение `production_risk_mc_repair.py`, поэтому обе выгрузки согласованы:
одна модель, отметка отказа на 1-е число месяца.

ВСЕ листы считаются одной моделью. Раньше книга была гибридом: сетка на cal c0 k2, а
«Интенсивность отказов» — из конвейера на shipped-Вейбулле бандла (`Mc_nonsour_Pooled`:
k1, beta=1.252, b50=503, окно 2023+, ГТМ считается отказом). Две модели в одной книге
показывать нельзя, поэтому конвейер убран: интенсивность строится из `history_match`
(факт vs модель) и `forecast` — то же, что показывает вкладка приложения.

Побочно: конвейер и так ничего не давал сетке — «ДебН», «Категория», «Куст» приходили
пустыми (живой парк их не несёт). Эти колонки остаются пустыми и сейчас.

ВНС. Вместе с конвейером сначала уехало и полезное — вселенная скважин ИЗ ПЛАНА
(`plan.producers`), а не из пробегов ЭЦН. Без неё прогноз был противоречив: парк
Мирнинского по плану РАСТЁТ 67 -> 82 за счёт ввода новых скважин, а отказы считались
только по 58 живым насосам. Вселенная возвращена (`esp_forecast.plan_entrants`), новый
насос входит в возрасте 0 со своего планового месяца — как и делал конвейер
(age_pmf={0:1.0}). ВНС дают +23.2 отказа к 52.0, т.е. треть прогноза.

Чего у конвейера НЕ взято: гейт include_primary при esp_scope_policy="conservative"
выбрасывал ВНС без ЭЦН-следа (нет строки ГТМ, нет техрежима) — а у не введённой
скважины следа и быть не может. Из 49 новых Mc он пропускал 18, а 31 выкидывал из
ЧИСЛИТЕЛЯ, оставляя в знаменателе (парк-то плановый). Здесь берутся все 49: на истории
2024-01..2026-03 ВСЕ добывающие по плану скважины Mc есть в строгой ЭЦН-популяции.

Запуск:  python scripts/run/export_mc_repair_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import repair_compat

MC_PLAN_FIELD = "Мирнинский УН"
HIST_FIRST, HIST_LAST = "2024-01", "2026-06"
HORIZON_MONTHS = 18
METHOD = "statistical"


def _live_fleet(pop: pd.DataFrame, asof: pd.Timestamp, meta: pd.DataFrame) -> pd.DataFrame:
    """Работающие насосы Mc на дату расчёта + метаданные (Куст/ДебН) — вход для payload."""
    live = M.live_fleet(pop, asof, field="Mc")
    live["license_area"] = MC_PLAN_FIELD
    live = live.merge(meta, on="code", how="left")
    return live


def _intensity_sheet(hist: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Интенсивность отказов = отказы / парк в работе, %. История + прогноз, одна модель.

    Знаменатель в обеих половинах — СКВАЖИНЫ (не пробеги), иначе проценты
    несопоставимы: скважина с подъёмом и переустановкой в одном месяце даёт 2 пробега.
    """
    h = hist.rename(columns={"модель ожидает": "модель"})[
        ["месяц", "в работе", "факт", "модель", "факт, % от парка", "модель, % от парка"]].copy()
    h["период"] = "история"
    f = fc.rename(columns={"ожидаемые события": "модель"})[
        ["месяц", "в работе", "модель", "модель, % от парка"]].copy()
    # float, а не pd.NA: у concat с all-NA колонками dtype меняется (FutureWarning),
    # и «факт» уехал бы в object — в Excel это текст вместо чисел.
    f["факт"] = float("nan")
    f["факт, % от парка"] = float("nan")
    f["период"] = "прогноз"
    out = pd.concat([h, f[h.columns]], ignore_index=True)
    out["модель"] = out["модель"].astype(float).round(2)
    return out


def _add_intensity_chart(path: Path, intensity: pd.DataFrame) -> None:
    """График интенсивности на лист — факт vs модель, % от парка, история+прогноз.

    Раньше график рисовал `failure_rate.write_sheets`; вместе с конвейером он ушёл, и
    лист остался голой таблицей. Рисуем сами — тем же LineChart, но по НАШЕЙ модели.
    """
    import openpyxl
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.axis import ChartLines

    wb = openpyxl.load_workbook(path)
    ws = wb["Интенсивность отказов"]
    cols = list(intensity.columns)
    n = len(intensity)
    c_fact = cols.index("факт, % от парка") + 1
    c_model = cols.index("модель, % от парка") + 1

    chart = LineChart()
    chart.title = "Интенсивность отказов УЭЦН — Мирнинский (cal c0 k2): факт vs модель"
    chart.y_axis.title = "% от парка в работе"
    chart.x_axis.title = "Месяц"
    chart.y_axis.majorGridlines = ChartLines()
    chart.height, chart.width = 9, 28
    for col in (c_fact, c_model):
        chart.add_data(Reference(ws, min_col=col, min_row=1, max_row=n + 1), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=n + 1))
    # факт — точки без линии: он рваный (0..50%) и линия между месяцами вводит в
    # заблуждение; модель — сплошная.
    s_fact, s_model = chart.series[0], chart.series[1]
    s_fact.graphicalProperties.line.noFill = True
    s_fact.marker.symbol = "circle"
    s_fact.marker.size = 5
    s_model.graphicalProperties.line.width = 22000
    ws.add_chart(chart, "J2")
    wb.save(path)


def main() -> None:
    cfg = C.RunConfig()
    asof = pd.Timestamp(str(cfg.forecast_start))
    pop = P.build(str(cfg.forecast_start))
    model = M.fit_mc_model(pop)

    # Свежие насосы из плана: ВНС (новых скважин нет в истории) + БАЗА между насосами
    # (скважина есть, но насос поднят прямо перед asof). Парк Мирнинского по плану
    # РАСТЁТ 67 -> 82; без ввода числитель (отказы) и знаменатель (парк) считались бы по
    # разным популяциям. Вселенная — из плана, как в конвейере (`plan.producers`).
    codes = set(pop[pop["field"] == "Mc"]["code"])
    live0 = M.live_fleet(pop, asof, field="Mc")
    live_codes = set(live0["code"])
    entrants = M.plan_entrants(live_codes, codes, asof, HORIZON_MONTHS)

    meta = M.plan_metadata(list(codes | set(entrants["code"])), asof)
    live = _live_fleet(pop, asof, meta)
    ent = entrants.merge(meta, on="code", how="left")
    ent["license_area"] = MC_PLAN_FIELD
    ent["raw_id"] = ent["code"]
    # NE_ (Нежданинское) — другое месторождение того же УН, модель перенесена с Mc;
    # помечаем источник, чтобы это было явно видно в форме.
    ent["pred_source"] = ent["code"].str.startswith("NE_").map(
        {True: "survival (Mc-модель, другое м-е)", False: "survival"})

    n_vns = int((entrants["category"] == "ВНС").sum())
    n_ren = int((entrants["category"] == "БАЗА").sum())
    n_ne = int(entrants["code"].str.startswith("NE_").sum())
    print("Мирнинский: работающих насосов = %d | ВНС %d + БАЗА между насосами %d = %d свежих (в т.ч. NE %d)"
          % (len(live), n_vns, n_ren, len(entrants), n_ne))
    print("модель c=%.0f: w1=%.4f β2=%.3f η2=%.1f B50=%.0f сут"
          % (model.cut_days, model.w1, model.beta2, model.eta2, model.b50()))
    if live.empty:
        raise SystemExit("Работающие насосы Mc не найдены — проверить популяцию")

    payload = M.repair_payload(live, model, asof=asof, horizon_months=HORIZON_MONTHS,
                               method=METHOD, entrants=ent)

    # Интенсивность — из ТОЙ ЖЕ модели: история (сверка с фактом) + прогноз.
    # Парк — активный ДОБЫВАЮЩИЙ из плана ПП, а не «есть непонятый насос»: на
    # 2024-10 это 19 скважин против 36 пробегов, т.е. вдвое. Парк ОБЯЗАН включать те
    # же свежие насосы, что и симуляция, иначе снова разъедется с числителем.
    # Знаменатель — ВСЕ скважины Mc/Mr/NE, а не только живые сейчас: в истории
    # (2024) добывали скважины, чьи насосы давно подняты. `live_codes` их резал —
    # 2024-10 падал 19 -> 15. Берём полную популяцию + свежие из плана.
    fleet = M.active_fleet_by_month(codes=codes | set(entrants["code"]))
    hist = M.history_match(model, pop, first=HIST_FIRST, last=HIST_LAST,
                           fleet_by_month=fleet)
    fc, _ = M.forecast(model, live, asof=asof, months=HORIZON_MONTHS,
                       fleet_by_month=fleet, entrants=ent)
    print("свежих насосов за %d мес: %d (ввод %s..%s)"
          % (HORIZON_MONTHS, len(entrants),
             entrants["entry"].min().date(), entrants["entry"].max().date()))
    intensity = _intensity_sheet(hist, fc)

    out = results_dir("production_risk_mc_forecast")
    path = out / "tables" / "Прогноз_ремонтов_Mc.xlsx"
    written = repair_compat.write_excel(path, payload)          # без failure_rate конвейера
    with pd.ExcelWriter(written, engine="openpyxl", mode="a") as xl:
        intensity.to_excel(xl, sheet_name="Интенсивность отказов", index=False)
        hist.to_excel(xl, sheet_name="История_факт_модель", index=False)
    _add_intensity_chart(written, intensity)

    rows = repair_compat.rows_frame(payload)
    print("записано:", written)
    print("строк:", len(rows), "| событий:", int(rows["event_count"].sum()))
    cols = ["well_name", "actual_nno", "days_to_workover", "probabilistic_nno", "event_count"]
    print(rows[cols].head(12).to_string(index=False))
    bad = rows.dropna(subset=["actual_nno", "probabilistic_nno"])
    bad = bad[bad["probabilistic_nno"] < bad["actual_nno"]]
    print("строк, где «Вероятностный прогноз» < «Факт ННО»: %d (должно быть 0)" % len(bad))
    print()
    print(hist.to_string(index=False))


if __name__ == "__main__":
    main()
