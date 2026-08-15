"""Two-layer workover model — NORMAL (schedule) + INFANT (demand uplift).

The premise, from the 2026-07-17 review: a schedule is a resource commitment, so it
must not contain "this new pump fails on day 8".  But those failures consume rigs,
pumps and oil, so they belong in DEMAND.  One curve cannot do both — the shipped
Weibull only matches the fleet total by smearing an infant spike it cannot represent
into a plateau it then over-states, which is why bolting an infant multiplier on top
double-counts.

So the OUTPUT is split, not the fit:

  вкладка 2  ГРАФИК     — normal layer only.  What you commit crews and rigs to.
  вкладка 3  ПОТРЕБНОСТЬ — normal + infant uplift.  Pumps, rig-days, oil loss.
  вкладка 4  ПРОВЕРКА   — schedule + infant must reproduce observed TOTAL failures.
                          That is the check that the split did not lose anything.

Read the caveats on вкладка 1 before choosing a window: the cut is NOT innocent.

Run: .venv/Scripts/python.exe -m streamlit run streamlit_apps/production_risk_two_layer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import (
    crosswalk,
    esp_population as P,
    failure_rate as FR,
    layers as L,
    repair_compat,
    two_layer as TL,
)
from analysis.workflows.production_risk.survival import HazardLayer, StrataModel

st.set_page_config(page_title="Двухслойная модель ремонтов", layout="wide")

FIELDS = {"Мирнинский УН": "Mc", "Ярактинский УН": "Ya", "Верхнетирский УН": "Vt"}
NORMAL_C = "#1F5C99"
INFANT_C = "#FF8C00"
OBS_C = "#1A2744"


@st.cache_resource(show_spinner="Загрузка плана ПП, Свод, ТР ... (~40 с, один раз)")
def load_context():
    cfg = C.RunConfig()
    plan = crosswalk.load_plan(cfg.forecast_start, cfg.horizon_end, cfg.pp_master_path)
    esp = crosswalk.load_esp_source(cfg.bundle_date, cfg.prediction_workbook_path)
    gtm = crosswalk.load_gtm(cfg.gtm_schedule_path)
    tr = crosswalk.load_current_techregime_status(cfg.techregime_workbook_path)
    downtime = crosswalk.derive_downtime_quantiles(esp)
    well_field = FR.build_well_field(plan)
    return cfg, plan, esp, gtm, tr, downtime, well_field


@st.cache_data(show_spinner="Разложение на слои ...")
def fit_layers(model_field: str, h2s: str, cut: float) -> dict:
    pop = P.build("2026-07-01")
    g = P.select(pop, model_field, h2s=h2s)
    return TL.fit(g, cut).as_dict()


def field_strata(model_field: str, bundle_date: str) -> list[str]:
    m = StrataModel(bundle_date=bundle_date)
    return [s for s in m.by_stratum if s.startswith(f"{model_field}_")]


@st.cache_data(show_spinner="Проекция по НОРМАЛЬНОМУ слою ...")
def project(un: str, model_field: str, cut: float):
    """Schedule from the normal layer alone, plus the installs the infant layer needs.

    Fits EACH stratum of the field separately.  Vt_sour and Vt_nonsour are different
    populations (B50 98 vs 438) — pooling them and applying one curve to both
    under-predicted Vt by 38%.
    """
    cfg, plan, esp, gtm, tr, (gd, fd), well_field = load_context()
    model = StrataModel(bundle_date=cfg.bundle_date)
    for stratum in field_strata(model_field, cfg.bundle_date):
        h2s = stratum.split("_")[1]
        try:
            lay = TL.Layers(**fit_layers(model_field, h2s, float(cut)))
        except Exception:
            continue  # too few events past the cut — keep the shipped row
        row = dict(model.by_stratum[stratum])
        row.update(TL.normal_registry_params(lay))
        row["uptime_factor"] = 1.0
        model.by_stratum[stratum] = row
    hazard = HazardLayer(bundle_date=cfg.bundle_date)

    states, _ = L.build_well_states(plan, esp, gtm, model, cfg, current_status=tr)
    scoped = [s for s in states if well_field.get(s.code) == un]
    specs = tuple(s for s in cfg.scenarios if s.scenario_id == C.PRIMARY_SCENARIO_ID)
    proj = L.run_projection(scoped, specs, model, hazard, gd, fd, plan.fwd_months,
                            cfg.changeout_p90, downtime_override_days=cfg.downtime_override_days)
    payload = repair_compat.build(proj, scoped, cfg, scenario_id=C.PRIMARY_SCENARIO_ID)
    rate = FR.compute(plan, esp, proj, cfg, scenario_id=C.PRIMARY_SCENARIO_ID,
                      model=model, hazard=hazard, only_fields={un})
    monthly = rate.monthly[rate.monthly["field"] == un].sort_values("month").copy()

    econ = (proj.groupby("wid")
            .agg(E=("expected_failures", "sum"), oil=("planned_oil_t", "sum"), op=("planned_op_days", "sum"))
            .reset_index())
    econ["oil_rate"] = np.where(econ["op"] > 0, econ["oil"] / econ["op"], 0.0)
    econ["risk"] = econ["E"] * econ["oil_rate"]
    # Every commissioning is an install, and so is every scheduled workover.
    n_new = sum(1 for s in scoped if str(s.state_label).startswith("new_"))
    rows = pd.DataFrame([
        {"Скважина": r["well_name"], "Категория": r["category"],
         "Факт ННО, сут": r["actual_nno"], "Дней до ремонта": r.get("days_to_workover"),
         "Дата ремонта": (r["event_dates"] or [None])[0],
         "Ремонтов": len(r["event_dates"] or []), "wid": r["well_name"]}
        for r in payload["rows"]
    ])
    return payload, rows, econ, monthly, rate.forecast_first_month, n_new


st.title("Двухслойная модель ремонтов — НОРМА (график) + МЛАДЕНЧЕСТВО (потребность)")

with st.sidebar:
    st.header("Область")
    un = st.selectbox("Участок недр", list(FIELDS), index=0)
    fld = FIELDS[un]
    st.header("Окно младенчества")
    cut = st.select_slider("Граница, сут", options=list(TL.CUT_CHOICES), value=30.0)
    st.caption(
        "Отказ на 89-е сутки может быть обычным — поэтому граница это ПАРАМЕТР, "
        "а не факт. Смотрите вкладку 1: у Ya избыток выходит на плато к 30 сут, "
        "а у Za/Az растёт и дальше — там граница уже НАЗНАЧАЕТ, а не измеряет."
    )
    st.header("Окно установок для МРП")
    regime = st.selectbox(
        "Считать МРП по установкам", ["2024+", "вся история"], index=0,
        help="Фильтр по ДАТЕ УСТАНОВКИ, а не по экспозиции: усечение экспозиции "
             "оставляет старым пробегам их свежее время и прячет, что недавние "
             "пробеги короче (Mc: ГТМ 295 -> 182 сут).",
    )
    install_from = "2024-01-01" if regime == "2024+" else None
    lay = fit_layers(fld, "nonsour", float(cut))
    st.header("Слой НОРМА")
    st.caption("Показана основная страта (nonsour); каждая страта поля фитится отдельно.")
    st.metric("beta", f"{lay['beta']:.3f}",
              help="beta<1: интенсивность падает с возрастом. >1 был бы износ.")
    st.metric("B50, сут", f"{lay['b50']:.0f}")
    st.header("Слой МЛАДЕНЧЕСТВО")
    st.metric("p на установку", f"{lay['p_infant']:.4f}")
    st.metric("избыток, отказов", f"{lay['excess']:.1f}",
              delta=f"{100 * lay['excess'] / max(lay['n_events_total'], 1):.0f}% от всех")

payload, rows, econ, monthly, fcst_first, n_new = project(un, fld, float(cut))

fwd = monthly[monthly["month"] >= fcst_first]
n_sched = int(sum(len(r["event_dates"] or []) for r in payload["rows"]))
installs = n_sched + n_new
n_infant = TL.infant_uplift(installs, TL.Layers(**lay))

t1, t2, t3, t4 = st.tabs(["1 · Слои", "2 · ГРАФИК (норма)", "3 · ПОТРЕБНОСТЬ (норма+младенч.)", "4 · Проверка"])

with t1:
    st.subheader("Разложение при разных границах окна")
    tbl = {}
    for c in TL.CUT_CHOICES:
        try:
            tbl[f"{c:.0f} сут"] = TL.Layers(**fit_layers(fld, "nonsour", float(c)))
        except Exception:
            pass
    st.dataframe(TL.summary_table(tbl), use_container_width=True, hide_index=True)
    st.warning(
        "**Граница не невинна.** Раздвигая окно, вы одновременно поднимаете beta нормы "
        "и «избыток» — это две стороны одной перепараметризации. Честная граница там, "
        "где избыток ВЫХОДИТ НА ПЛАТО (Ya: 77 отказов на 30 сут и дальше не растёт). "
        "Там, где он растёт монотонно (Za, Az), окно уже назначает ответ."
    )
    grid = np.arange(1, 1500.0)
    h = (lay["beta"] / lay["eta"]) * ((grid / lay["eta"]) ** (lay["beta"] - 1)) * 30.4
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=h, mode="lines", line=dict(color=NORMAL_C, width=3),
                             name=f"НОРМА: beta={lay['beta']:.2f}"))
    fig.add_vrect(x0=0, x1=cut, fillcolor=INFANT_C, opacity=0.15, line_width=0,
                  annotation_text="окно младенчества", annotation_position="top left")
    fig.update_layout(height=380, xaxis_title="Наработка, сут", yaxis_title="Интенсивность, отк/мес",
                      yaxis=dict(range=[0, float(np.nanpercentile(h, 97))]), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**НОРМА — не плато и не износ.** beta={lay['beta']:.3f} < 1: это медленно "
        "убывающая интенсивность. Износ (beta>1) проверен четырьмя независимыми "
        "способами и НЕ найден на хорошо обеспеченных полях — в т.ч. моделью с "
        "gamma-frailty по скважинам (Ya: 0.717 → 0.740 при theta=0.11, p=2.5e-05), "
        "так что неоднородность скважин износ не маскирует. Исключение — Ic (beta≈1.13).\n\n"
        "**Следствие для планирования:** при beta<1 остаточный ресурс МЕДЛЕННО РАСТЁТ "
        "с возрастом, т.е. самые старые насосы наименее срочные — обратное правилу "
        "«меняем самые старые». Приоритет — по риску (вкладка 3)."
    )

with t2:
    st.subheader("График ремонтов — только НОРМАЛЬНЫЙ слой")
    mode = st.radio(
        "Правило постановки ремонта",
        ["Статистический (по интенсивности)", "Конвенциональный (по ресурсу МРП)"],
        index=1, horizontal=True,
    )
    horizon_days = len(payload["dates"])

    if mode.startswith("Статист"):
        st.info(
            "**Прогноз.** Число ремонтов верно (±20%), но КОНКРЕТНАЯ ДАТА по скважине — "
            "детерминированный псевдослучайный розыгрыш: при beta<1 остаточный ресурс не "
            "зависит от возраста, поэтому предсказать «когда» нельзя в принципе. Отсюда "
            "Mc_1608: наработка 17 сут → ремонт через 37, а насос с 700 сут — без ремонта. "
            "Воспроизводимо ≠ предсказуемо."
        )
        view = rows.drop(columns=["wid"]).copy()
        n_show = n_sched
    else:
        mrps = TL.mrp_days(P.build("2026-07-01"), install_from=install_from)
        mrp_fld = np.median([v for k, v in mrps.items() if k.startswith(f"{fld}_")] or [np.nan])
        st.info(
            f"**Конвенция, а не прогноз.** Ремонт каждые МРП={mrp_fld:.0f} сут наработки: "
            f"следующий через `МРП − наработка`; насос старше МРП — «просрочен», день 0. "
            "Строки монотонны и объяснимы (старый раньше нового) — но данные говорят "
            "ОБРАТНОЕ: остаточный ресурс РАСТЁТ с возрастом, а 33–35% насосов переживают "
            "среднее по определению. Это правило целит бригады в наименее рисковые насосы."
        )
        recs = []
        for _, r in rows.iterrows():
            age = float(r["Факт ННО, сут"]) if pd.notna(r["Факт ННО, сут"]) else 0.0
            ev = TL.conventional_events(age, float(mrp_fld), horizon_days)
            recs.append({
                "Скважина": r["Скважина"], "Категория": r["Категория"],
                "Факт ННО, сут": r["Факт ННО, сут"],
                "Дней до ремонта": ev[0] if ev else None,
                "Дата ремонта": (pd.Timestamp(payload["dates"][0]) + pd.Timedelta(days=ev[0])).date().isoformat() if ev else None,
                "Ремонтов": len(ev),
                "Просрочен": "да" if (pd.notna(r["Факт ННО, сут"]) and float(r["Факт ННО, сут"]) > mrp_fld) else "нет",
            })
        view = pd.DataFrame(recs)
        n_show = int(view["Ремонтов"].sum())

    m = st.columns(4)
    m[0].metric("Скважин", f"{len(payload['rows'])}")
    m[1].metric("Ремонтов в графике", f"{n_show}")
    m[2].metric("Ремонтов в месяц", f"{n_show / max(horizon_days / 30.4, 1):.1f}")
    m[3].metric("Скважин с ремонтом", f"{int((view['Ремонтов'] > 0).sum())}")
    if mode.startswith("Конвенц"):
        st.metric("Из них уже просрочены", f"{int((view.get('Просрочен', pd.Series(dtype=str)) == 'да').sum())}",
                  help="наработка уже больше МРП — по конвенции ремонт немедленно")
    st.caption(
        "Это то, подо что бронируются бригады. Младенческие отказы сюда НЕ входят — "
        "их нельзя запланировать пер-скважинно; они во вкладке 3."
    )
    if st.checkbox("Только скважины с ремонтом", value=False):
        view = view[view["Ремонтов"] > 0]
    st.dataframe(view.sort_values("Дней до ремонта", na_position="last"),
                 use_container_width=True, hide_index=True, height=430)
    st.caption(
        "«Дней до ремонта» = ОСТАТОЧНЫЙ ресурс от начала прогноза, а не полная наработка — "
        "с «Факт ННО» (возрастом на сегодня) он не сравним и противоречить ему не может."
    )

with t3:
    st.subheader("Потребность = график + младенческая надбавка")
    a = st.columns(4)
    a[0].metric("Установок за горизонт", f"{installs}", help=f"{n_sched} ремонтов + {n_new} новых скважин")
    a[1].metric("Ремонтов по графику", f"{n_sched}")
    a[2].metric("Младенческих (надбавка)", f"{n_infant:.1f}",
                delta=f"+{100 * n_infant / max(n_sched, 1):.0f}% к графику")
    a[3].metric("ВСЕГО потребность", f"{n_sched + n_infant:.0f}")
    st.caption(
        "Надбавка рекурсивна: младенческий отказ — сам по себе ремонт, а его сменный "
        "насос несёт тот же риск, отсюда геометрическая сумма."
    )
    oil_lost = float((econ["risk"].sum() / max(econ["E"].sum(), 1e-9)) * n_infant) if len(econ) else 0.0
    b = st.columns(2)
    b[0].metric("Доп. простой, бригадо-суток", f"{n_infant * repair_compat._downtime_days(pd.DataFrame({'downtime_days': [5]})):.0f}",
                help="надбавка × медианный простой на ремонт")
    b[1].metric("Доп. потери нефти, т", f"{oil_lost:,.0f}",
                help="надбавка × средний дебит на ожидаемый отказ")
    st.subheader("Приоритет — по РИСКУ (не по вероятности)")
    st.caption("Вероятность по скважинам почти одинакова (CV≈0.27), дебит различается в ~79 раз.")
    e = econ[econ["oil_rate"] > 0].sort_values("risk", ascending=False).copy()
    e["Приоритет"] = range(1, len(e) + 1)
    st.dataframe(
        e.rename(columns={"wid": "Скважина", "E": "Ожид. отказов", "oil_rate": "ДебН, т/сут", "risk": "Риск"})
         [["Приоритет", "Скважина", "Ожид. отказов", "ДебН, т/сут", "Риск"]].round(2),
        use_container_width=True, hide_index=True, height=380,
    )

with t4:
    st.subheader("Проверка ПОМЕСЯЧНО: график + младенчество vs ФАКТ")
    hist = monthly[(monthly["month"] < fcst_first) & (monthly["month"] >= "2024-01")].copy()
    hist["младенч_надбавка"] = hist["predicted_failures"] * lay["p_infant"] / max(1 - lay["p_infant"], 1e-9)
    hist["всего_модель"] = hist["predicted_failures"] + hist["младенч_надбавка"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist["month"], y=hist["predicted_failures"], name="Норма (график)", marker_color=NORMAL_C))
    fig.add_trace(go.Bar(x=hist["month"], y=hist["младенч_надбавка"], name="Младенчество", marker_color=INFANT_C))
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["observed_failures"], mode="lines+markers", name="ФАКТ",
                             line=dict(color=OBS_C, width=2), marker=dict(size=7, symbol="diamond")))
    fig.update_layout(barmode="stack", height=400, xaxis_title="Месяц", yaxis_title="Отказов в месяц",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    cum = hist.assign(
        факт_нараст=hist["observed_failures"].cumsum(),
        модель_нараст=hist["всего_модель"].cumsum(),
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cum["month"], y=cum["факт_нараст"], mode="lines", name="ФАКТ (нарастающим)",
                              line=dict(color=OBS_C, width=3)))
    fig2.add_trace(go.Scatter(x=cum["month"], y=cum["модель_нараст"], mode="lines", name="Модель (нарастающим)",
                              line=dict(color=NORMAL_C, width=3, dash="dash")))
    fig2.update_layout(height=320, xaxis_title="Месяц", yaxis_title="Отказов, нарастающим итогом",
                       hovermode="x unified", legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Нарастающий итог — честнее помесячного: месячные числа малы (2–10), и их "
        "разброс это в основном пуассоновский шум, а не ошибка модели. Смотрите, "
        "расходятся ли линии систематически."
    )

    tbl = hist[["month", "observed_failures", "predicted_failures", "младенч_надбавка", "всего_модель"]].copy()
    tbl["всего/факт"] = (tbl["всего_модель"] / tbl["observed_failures"].replace(0, np.nan)).round(2)
    tbl.columns = ["Месяц", "ФАКТ", "Норма (график)", "Младенчество", "Всего модель", "всего/факт"]
    for c in ("Норма (график)", "Младенчество", "Всего модель"):
        tbl[c] = tbl[c].round(2)
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)
    st.caption(
        "«всего/факт» ≈ 1.0 — разложение ничего не потеряло. «Норма» < факта ожидаемо "
        "и это НЕ ошибка: график намеренно не содержит младенческих отказов. "
        "Помесячное отношение шумное — итог по периоду информативнее."
    )
    tot_f = hist["observed_failures"].sum(); tot_m = hist["всего_модель"].sum()
    st.metric("Итого за период: всего/факт", f"{tot_m / max(tot_f, 1e-9):.2f}",
              help=f"факт {tot_f:.0f} vs модель {tot_m:.1f}")
