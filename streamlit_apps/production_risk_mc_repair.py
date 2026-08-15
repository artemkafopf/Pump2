"""ПРОГНОЗ ОТКАЗОВ по Мирнинскому УН (Mc + Mr) — статистический.

Модель cal c0 k2: смесь двух Вейбуллов, монтажи 2024+, событие = ОТКАЗ (ГТМ и
работающие насосы — цензура). Месячная вероятность отказа; каждый прогнозный отказ
ставится на 1-е число своего месяца (день внутри месяца модель не определяет).

Детерминированный вариант (0 на возраст+ОР) УДАЛЁН: он складывал подъёмы в ком и был
политикой ПЛАНИРОВАНИЯ РЕМОНТОВ — это отдельная задача, здесь прогнозируются отказы.

Пишется в форму «Прогноз_ремонтов» (строка на скважину + суточные 0/1) через
repair_compat.write_excel. «Вероятностный прогноз» = возраст + расстояние до отказа.

Запуск:  .venv/Scripts/python.exe -m streamlit run streamlit_apps/production_risk_mc_repair.py
"""
from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd
import streamlit as st

from analysis.paths import WAREHOUSE_DIR
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import repair_compat

st.set_page_config(page_title="Прогноз ремонтов — Мирнинский", layout="wide")
ASOF_DEFAULT = pd.Timestamp("2026-07-01")


@st.cache_data(show_spinner="Строю популяцию Mc и подгоняю модель ...")
def _load(asof_iso: str):
    asof = pd.Timestamp(asof_iso)
    pop = P.build(asof_iso)
    model = M.fit_mc_model(pop)
    codes = set(pop[pop["field"] == "Mc"]["code"])
    live = M.live_fleet(pop, asof, field="Mc")
    live["license_area"] = "Мирнинский УН"
    live_codes = set(live["code"])

    with sqlite3.connect(WAREHOUSE_DIR / "pump2.db") as con:
        dm = pd.read_sql("SELECT well_key, dt, qliq FROM proc__daily_merged", con, parse_dates=["dt"])
    dm["code"] = dm["well_key"].map(crosswalk.norm_well)
    dm["qliq"] = pd.to_numeric(dm["qliq"], errors="coerce")
    rec = dm[(dm["dt"] >= asof - pd.Timedelta(days=90)) & (dm["qliq"] > 0)].groupby("code")["qliq"].median()
    live["ql"] = live["code"].map(rec)
    live["ql_src"] = np.where(live["ql"].notna(), "телеметрия", "импутация")
    med = float(live["ql"].median())
    live["ql"] = live["ql"].fillna(med)
    ref_log = float(np.median(np.log1p(live["ql"])))
    theta = {r.code: (M.ql_theta(float(r.ql), ref_log) if r.ql_src == "телеметрия" else 1.0)
             for r in live.itertuples()}
    # Парк = активный ДОБЫВАЮЩИЙ из плана ПП («Отработанное время»>0 и добыча>0), а
    # не «стоит непонятый насос»: на 2024-10 это 19 против 36, знаменатель вдвое.
    # Свежие насосы (ВНС + БАЗА между насосами) входят и в парк, и в симуляцию — иначе
    # числитель и знаменатель по разным популяциям (парк растёт 67 -> 82).
    entrants = M.plan_entrants(live_codes, codes, asof, 24)
    meta = M.plan_metadata(list(codes | set(entrants["code"])), asof)
    live = live.merge(meta, on="code", how="left")
    entrants = entrants.merge(meta, on="code", how="left")
    entrants["license_area"] = "Мирнинский УН"
    entrants["raw_id"] = entrants["code"]
    # NE_ (Нежданинское) — другое месторождение того же УН, модель перенесена с Mc.
    entrants["pred_source"] = entrants["code"].str.startswith("NE_").map(
        {True: "survival (Mc-модель, другое м-е)", False: "survival"})
    # Знаменатель — ВСЕ Mc/Mr/NE, а не только живые: в истории добывали и те скважины,
    # чьи насосы уже подняты (иначе 2024-10 падает 19 -> 15).
    fleet = M.active_fleet_by_month(codes=codes | set(entrants["code"]))
    hist = M.history_match(model, pop, first="2024-01", last="2026-06",
                           fleet_by_month=fleet)
    return asof, model, live.reset_index(drop=True), theta, ref_log, med, hist, fleet, entrants


def _peak_by_month(payload: dict) -> pd.Series:
    by: dict[str, int] = {}
    for r in payload["rows"]:
        for i, s in enumerate(r["statuses"]):
            if s == 0:
                mk = payload["dates"][i][:7]
                by[mk] = by.get(mk, 0) + 1
    return pd.Series(by).sort_index()


def _to_excel_bytes(payload: dict, hist: pd.DataFrame) -> bytes:
    tmp = Path(st.session_state["_scratch"]) / "mc_repair_tmp.xlsx"
    repair_compat.write_excel(tmp, payload)
    with pd.ExcelWriter(tmp, engine="openpyxl", mode="a") as xl:
        hist.to_excel(xl, sheet_name="История_факт_модель", index=False)
    return tmp.read_bytes()


st.title("Прогноз ремонтов — Мирнинский УН (Mc + Mr)")
st.session_state.setdefault("_scratch", str(REPO_ROOT / "results" / "_scratch_mc_repair"))
Path(st.session_state["_scratch"]).mkdir(parents=True, exist_ok=True)

c1, c2, c3 = st.columns(3)
asof_iso = c1.text_input("Дата расчёта", ASOF_DEFAULT.date().isoformat())
horizon = int(c2.number_input("Горизонт, мес", 6, 36, 18, 1))
use_ql = c3.checkbox("Слой Ql (сценарно)", value=False,
                     help="Гейт C4 его не промотировал (выигрыш ~1%). Справочно.")

asof, model, live, theta, ref_log, med, hist, fleet, entrants = _load(asof_iso)
theta_arg = theta if use_ql else None
ent_h = entrants[entrants["start_day"] <= horizon * 31].reset_index(drop=True)

st.caption(
    f"Модель **cal c{model.cut_days:.0f} k2** (монтажи с {C.MC_INSTALL_COHORT_START.year}), "
    "событие = ОТКАЗ (ГТМ и работающие — цензура): "
    f"w1={model.w1:.4f} · β1={model.beta1:.3f} η1={model.eta1:.2f} · "
    f"β2={model.beta2:.3f} η2={model.eta2:.0f} · **B50={model.b50():.0f} сут** · "
    f"пробегов {model.n_runs} / отказов {model.n_events} · живых насосов {len(live)} "
    f"+ свежих по плану {len(ent_h)} "
    f"(ВНС {int((ent_h['category']=='ВНС').sum())} / БАЗА между насосами "
    f"{int((ent_h['category']=='БАЗА').sum())}) · "
    f"опора Ql {np.expm1(ref_log):.0f} м³/сут · "
    f"факт/модель за историю {hist['факт'].sum()/hist['модель ожидает'].sum():.2f}"
)
st.caption(
    "c0 = ранние отказы НЕ вырезаны: стартовую массу дня-0 (~3%, η1 на полу 0.5) держит "
    "первая компонента. Модель БЕЗУСЛОВНАЯ — жизнь любого смонтированного насоса. "
    "Вариант c3 условен на дожитии до дня 3 и потому оптимистичен (MRL 645 против 572)."
)

pay_stat = M.repair_payload(live, model, asof=asof, horizon_months=horizon,
                            theta_by_code=theta_arg, entrants=ent_h)
peak_stat = _peak_by_month(pay_stat)

fc_monthly, fc_pw = M.forecast(model, live, asof=asof, months=horizon, theta_by_code=theta_arg,
                               fleet_by_month=fleet, entrants=ent_h)

tab0, tab1, tab_w = st.tabs(
    ["Интенсивность отказов", "Прогноз отказов (форма)", "По скважинам"])

for tab, name, payload, peak, why in [
    (tab1, "статистический", pay_stat, peak_stat,
     "Месячная вероятность отказа из модели. Счёт событий на скважину сохраняет "
     "флотскую сумму (Σ ожидаемых отказов); каждый отказ стоит на 1-м числе своего "
     "месяца — день внутри месяца модель не определяет."),
]:
    with tab:
        st.markdown(f"**{why}**")
        rows = repair_compat.rows_frame(payload)
        m1, m2, m3 = st.columns(3)
        m1.metric("Событий за горизонт", int(rows["event_count"].sum()))
        m2.metric("Пик отказов/мес", int(peak.max()) if len(peak) else 0)
        m3.metric("Месяцев занято", int((peak > 0).sum()) if len(peak) else 0)
        st.bar_chart(peak.rename("отказов"))
        show = rows[["well_name", "actual_nno", "days_to_workover", "probabilistic_nno", "event_count"]]
        st.dataframe(show.rename(columns={
            "well_name": "Скважина", "actual_nno": "Факт ННО",
            "days_to_workover": "Дней до отказа", "probabilistic_nno": "Вероятностный прогноз (ННО)",
            "event_count": "Событий"}), use_container_width=True, height=360)
        st.download_button(
            f"⬇ Excel «Прогноз_ремонтов» — {name}", data=_to_excel_bytes(payload, hist),
            file_name=f"Прогноз_ремонтов_Mc_{name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

with tab0:
    st.markdown(
        "**Интенсивность отказов = отказы / парк в работе, %.** История (факт vs модель) "
        "и прогноз на одной шкале — так видно, сшивается ли модель с фактом."
    )
    h = hist.rename(columns={"модель ожидает": "модель"})
    h_pct = h[["месяц", "факт, % от парка", "модель, % от парка"]].set_index("месяц")
    f_pct = (fc_monthly[["месяц", "модель, % от парка"]].set_index("месяц")
             .rename(columns={"модель, % от парка": "модель, % (прогноз)"}))
    st.line_chart(pd.concat([h_pct, f_pct], axis=0), height=320)

    k1, k2, k3 = st.columns(3)
    k1.metric("Факт, % / мес (история)", f"{h['факт, % от парка'].mean():.1f}%")
    k2.metric("Модель, % / мес (история)", f"{h['модель, % от парка'].mean():.1f}%")
    k3.metric("Модель, % / мес (прогноз)", f"{fc_monthly['модель, % от парка'].mean():.1f}%")
    st.caption(
        f"Факт/модель за историю = **{h['факт'].sum() / h['модель'].sum():.2f}**. "
        f"Парк по плану РАСТЁТ {fc_monthly['в работе'].iloc[0]:.0f} → "
        f"{fc_monthly['в работе'].iloc[-1]:.0f} за счёт свежих насосов ({len(ent_h)} шт.: "
        f"ВНС + БАЗА между насосами); они смоделированы (насос с возраста 0) и дают "
        f"**{fc_pw[fc_pw['новая']]['всего за горизонт'].sum():.1f}** отказов из "
        f"{fc_monthly['ожидаемые события'].sum():.1f} — у свежего насоса своя детская "
        "смертность. Прогноз выше истории не из-за смещения: история считается БЕЗ "
        "обновления (поднятый насос не порождает новый), прогноз — С обновлением. "
        "ГТМ в прогнозе нет — это верхняя оценка (с ГТМ как конкурирующим риском было "
        "бы ≈ −0.6 п.п.). Простой между отказом и запуском на % НЕ влияет: он уходит и "
        "из числителя, и из знаменателя."
    )

    st.markdown("**Число отказов по месяцам**")
    hist_cnt = h[["месяц", "в работе", "факт", "модель"]].copy()
    hist_cnt["период"] = "история"
    fc_cnt = fc_monthly.rename(columns={"ожидаемые события": "модель"})[
        ["месяц", "в работе", "модель"]].copy()
    fc_cnt["факт"] = np.nan
    fc_cnt["период"] = "прогноз"
    cnt = pd.concat([hist_cnt, fc_cnt[["месяц", "в работе", "факт", "модель", "период"]]],
                    ignore_index=True)
    cnt["модель"] = cnt["модель"].round(2)
    st.dataframe(cnt, use_container_width=True, hide_index=True, height=420)
    st.caption(
        f"Итого: факт за историю **{int(h['факт'].sum())}**, модель ожидала "
        f"**{h['модель'].sum():.1f}**; прогноз на {horizon} мес — "
        f"**{fc_monthly['ожидаемые события'].sum():.0f}** отказов "
        f"(с обновлением: отказавший насос сразу меняется)."
    )

with tab_w:
    st.markdown(
        f"**ОР = {M.RESOURCE_BLEND_W:.1f}·RMST + {1-M.RESOURCE_BLEND_W:.1f}·MRL** — "
        "это ТОЛЬКО отображаемая цифра, прогноз отказов её не использует "
        "(проверено: статистика даёт те же Σожид при любом определении ОР)."
    )
    wt = M.well_table(model, live, asof=asof, theta_by_code=theta_arg)
    st.dataframe(wt, use_container_width=True, hide_index=True, height=520)
    st.caption(
        "Чистый RMST дал бы 0 насосам старше tau (%d из %d живых) — артефакт закрытия "
        "окна. Чистый MRL дал бы им ~413 сут («не трогать ещё год»). Смесь w=%.1f даёт "
        "~40 сут. За tau=%.0f смесь вырождается в (1-w)·MRL, поэтому по возрасту там "
        "почти не различает — это известное ограничение."
        % (int((live["age"] >= model.tau).sum()), len(live), M.RESOURCE_BLEND_W, model.tau)
    )
