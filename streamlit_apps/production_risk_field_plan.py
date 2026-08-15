"""Workover-plan review app — Мирнинский / Верхнетирский / Ярактинский.

Compares the shipped registry fit (option A) against a spike + constant-hazard
refit (option B) for the selected УН, so the choice is reviewable rather than
asserted.  Params are injected at runtime; **`esp_models.csv` is untouched.**

Option B exists because the empirical hazard is "infant spike (~30 d) then FLAT,
no wear-out arm", which a single Weibull cannot express — its MLE compromises at
`beta<1`, a curve that decays forever.  19 of the registry's 26 strata are fit
that way (`model_kind = k1`, w1 = 0); the 7 `k2` strata already carry a mixture
and are NOT expected to improve.  So option B does not win everywhere, and the
per-field notes below say where it does.

What is and is not trustworthy — read before using the per-well tab:
  * the monthly COUNT is solid (±20%);
  * the per-well DATE is a seeded pseudo-random draw, not a prediction (constant
    hazard ⇒ memoryless ⇒ remaining life is independent of age);
  * priority should be read from RISK (E × oil), not probability: failure
    probability is near-flat across wells while oil rate spans ~79x.

Evidence: `docs/notes/production_risk_mc_weibull_shape_findings.md` (Mc, form),
`results/production_risk_vt_hazard_fit/` (Vt, per-stratum audit).

Run:  .venv/Scripts/python.exe -m streamlit run streamlit_apps/production_risk_field_plan.py
      (the venv has streamlit 1.48.1 — `width='stretch'` is NOT supported)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
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
    esp_hazard_fit as H,
    esp_population as P,
    failure_rate as FR,
    layers,
    repair_compat,
)
from analysis.workflows.production_risk.survival import HazardLayer, StrataModel

st.set_page_config(page_title="План ремонтов — обзор", layout="wide")

PARAM_KEYS = ("w1", "beta1", "eta1", "beta2", "eta2")
OBS_COLOR = "#1A2744"
PRED_COLOR = "#FF8C00"
FCST_COLOR = "#1F5C99"


@dataclass(frozen=True)
class FieldSpec:
    code: str                  # model field code used by StrataModel / esp_population
    un: str                    # «УН» as it appears in the ПП plan
    default_option: str        # "A" or "B" — whichever this field's calibration supports
    note: str                  # what option B is and is not worth on this field


FIELDS: dict[str, FieldSpec] = {
    "Мирнинский": FieldSpec(
        code="Mc",
        un="Мирнинский УН",
        # Mc is fitted on the installs-2024+ COHORT (config.MC_INSTALL_COHORT_START),
        # applied by esp_population.select().  Pre-2023 is the anomaly — 2 failures vs
        # 11.8 expected on 21 runs.  This replaces the earlier left-truncated-exposure
        # variant, which kept pre-2024 runs' later exposure and hid that recent runs
        # are shorter.
        default_option="B",
        note=(
            "**Одна страта** (`Mc_nonsour_Pooled`), **когорта монтажей 2024+** "
            "(164 пробега / 44 отказа). Бандл — одиночный Вейбулл (k1) с `beta=1.252`: "
            "считает молодой насос НАДЁЖНЕЕ старого. Это противоречит эмпирике и "
            "**остаётся единственным аргументом за B** — младенческий пик реален (на Ya "
            "бандл даёт 0.73–0.78 от факта на 0–30 сут), а ~58 скважин Mc стартуют с "
            "возраста 0 внутри горизонта.\n\n"
            "Прогноз: B **45** против A **50**. Калибровка (когорта 2024+, 2026-07-17): "
            "B `1.19 / 0.94 / 1.39`, A `1.06 / 0.86 / 1.33` — **по уровню A ЛУЧШЕ B** "
            "(ср. |откл| 0.53 против 0.64). Разница между вариантами — в ФОРМЕ, не в "
            "уровне.\n\n"
            "**Слабое место B:** `beta1/eta1` упираются в границы — пик схлопывается в "
            "точечную массу на дне 0.5, его величина задана ограничением, а не данными.\n\n"
            "⚠️ Аргументы Mc-сессии «A экстраполирует износ» и «Вейбулл завышает плато "
            "в 1.4–1.8×» опирались на популяцию без Big-only закрытых пробегов и "
            "**не подтвердились** — на исправленной популяции плато откалибровано "
            "(0.96–1.29). Довод про младенческий пик устоял; довод про плато — нет."
        ),
    ),
    "Верхнетирский": FieldSpec(
        code="Vt",
        un="Верхнетирский УН",
        # Measured on the corrected population (2026-07-17): A and B are practically
        # identical — 261 vs 257 workovers, 1.09/0.96/0.95 vs 1.09/0.96/0.94.  There is
        # no reason to deviate from the shipped bundle.
        default_option="A",
        note=(
            "**8 страт** (кислые/некислые × подрядчик), 318 отказов. Популяция сходится "
            "с реестром (209 отказов против 198 в `Vt_nonsour_Pooled`, 110 против 111 "
            "в `Vt_sour_Pooled`).\n\n"
            "**A и B здесь практически неразличимы:** 261 против 257 ремонтов, "
            "модель/факт `1.09 / 0.96 / 0.95` против `1.09 / 0.96 / 0.94`. По KM бандл "
            "даже лучше: `Vt_nonsour_Pooled` 0.047 против 0.088 у B, `Vt_sour_Pooled` "
            "0.060 против 0.054. Медианное отношение к эмпирике: бандл 1.17 / 1.04, "
            "B 0.97 / 1.12. **Бандл на Vt в порядке — B ничего не добавляет, "
            "по умолчанию A.**\n\n"
            "⚠️ Более ранний вывод «бандл завышает кислые в 1.84×, KM 0.168» был "
            "**артефактом популяции** (53 задвоенных цензурированных пробега раздували "
            "знаменатель экспозиции). После исправления дефекта нет."
        ),
    ),
    "Ярактинский": FieldSpec(
        code="Ya",
        un="Ярактинский УН",
        # Corrected population (2026-07-17) added Ya's 619 Big-only closed runs, which
        # a Свод-only population had been missing — a third of Ya's events.  A and B are
        # now within a few percent (133 vs 137); no reason to deviate from the bundle.
        default_option="A",
        note=(
            "**4 страты, 1226 отказов** — самая надёжная оценка в реестре (сходится с "
            "реестровыми 1192 после добавления 619 закрытых Big-only пробегов).\n\n"
            "**A и B почти неразличимы:** 133 против 137 ремонтов, модель/факт "
            "`0.94 / 1.18 / 1.11` против `0.98 / 1.27 / 1.25`. **По умолчанию A.**\n\n"
            "Единственный реальный дефект формы бандла — **младенческий пик**: на 0–30 "
            "сут модель даёт 0.73–0.78 от факта (эмпирика `Ya_nonsour_brt`: **0.0989**/мес "
            "на 0–30 против 0.024–0.050 на плато). Участка износа нет — на 1100–1500 "
            "интенсивность (0.021) НИЖЕ, чем на 30–90 (0.050).\n\n"
            "⚠️ Более ранние выводы «B занижает Ya» (0.70/0.87/0.84) и «Вейбулл завышает "
            "плато в 1.4–1.8×» были **артефактами популяции**. На исправленной популяции "
            "отношение модель/эмпирика по плато = **0.96–1.29**, т.е. бандл откалиброван."
        ),
    ),
}


def parse_stratum(name: str) -> tuple[str, str, str | None]:
    """`Vt_sour_brt` → (Vt, sour, brt); `Vt_sour_Pooled` → (Vt, sour, None)."""
    field, h2s, ctr = name.split("_")
    return field, h2s, (None if ctr == "Pooled" else ctr)


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


@st.cache_data(show_spinner=False)
def field_strata(bundle_date: str, code: str) -> tuple[str, ...]:
    df = pd.read_csv(C.model_registry_path(bundle_date), encoding="utf-8-sig")
    return tuple(df.loc[df["stratum"].str.startswith(f"{code}_"), "stratum"])


@st.cache_data(show_spinner="Подгонка спайк+константа по стратам ...")
def fit_option_b(code: str, strata: tuple[str, ...]) -> dict:
    """Fit spike+constant on each stratum's OWN runs.

    Every stratum is fitted separately rather than borrowing a donor: the Mc review
    measured that borrowing Ya's shape under-predicts by 20%, and that a field's own
    events beat a donor even at 49 events.

    `P.select` applies the Мирнинский installs-2024+ cohort itself, so Mc needs no
    special-casing here — and specifically no left truncation, which is what this used
    to do and which hid that recent Mc runs are shorter.
    """
    pop = P.build("2026-07-01")
    out: dict[str, dict] = {}
    for name in strata:
        field, h2s, ctr = parse_stratum(name)
        g = P.select(pop, field, h2s, ctr)
        if len(g) == 0 or int(g["event"].sum()) < 3:
            continue
        fitted = H.fit(g["tte"].to_numpy(float), g["event"].to_numpy(int))
        out[name] = {
            **H.registry_params(fitted),
            "n_runs": int(len(g)),
            "n_failures": int(g["event"].sum()),
        }
    return out


@st.cache_data(show_spinner=False)
def option_a_params(bundle_date: str, strata: tuple[str, ...]) -> dict:
    df = pd.read_csv(C.model_registry_path(bundle_date), encoding="utf-8-sig").set_index("stratum")
    return {
        name: {
            **{k: float(df.loc[name, k]) for k in PARAM_KEYS},
            "n_runs": int(df.loc[name, "n_runs"]),
            "n_failures": int(df.loc[name, "n_failures"]),
        }
        for name in strata if name in df.index
    }


@st.cache_data(show_spinner="Расчёт плана ...")
def build_plan(un: str, params_items: tuple):
    """Project the selected УН under the given per-stratum params."""
    cfg, plan, esp, gtm, tr, (gd, fd), well_field = load_context()

    model = StrataModel(bundle_date=cfg.bundle_date)
    for name, values in params_items:
        row = dict(model.by_stratum[name])
        row.update(dict(zip(PARAM_KEYS, values)))
        model.by_stratum[name] = row
    hazard = HazardLayer(bundle_date=cfg.bundle_date)

    states, _ = layers.build_well_states(plan, esp, gtm, model, cfg, current_status=tr)
    scoped = [s for s in states if well_field.get(s.code) == un]
    specs = tuple(s for s in cfg.scenarios if s.scenario_id == C.PRIMARY_SCENARIO_ID)
    projection = layers.run_projection(
        scoped, specs, model, hazard, gd, fd, plan.fwd_months,
        cfg.changeout_p90, downtime_override_days=cfg.downtime_override_days,
    )
    payload = repair_compat.build(projection, scoped, cfg, scenario_id=C.PRIMARY_SCENARIO_ID)
    rate = FR.compute(
        plan, esp, projection, cfg, scenario_id=C.PRIMARY_SCENARIO_ID,
        model=model, hazard=hazard, only_fields={un},
    )

    econ = (
        projection.groupby("wid")
        .agg(E=("expected_failures", "sum"), oil=("planned_oil_t", "sum"),
             op=("planned_op_days", "sum"))
        .reset_index()
    )
    econ["oil_rate"] = np.where(econ["op"] > 0, econ["oil"] / econ["op"], 0.0)
    econ["risk"] = econ["E"] * econ["oil_rate"]

    rows = pd.DataFrame([
        {
            "Скважина": r["well_name"],
            "Категория": r["category"],
            "Факт ННО, сут": r["actual_nno"],
            "Дней до ремонта": r.get("days_to_workover"),
            "Дата ремонта": (r["event_dates"] or [None])[0],
            "Ремонтов в горизонте": len(r["event_dates"] or []),
            "wid": r["well_name"],
        }
        for r in payload["rows"]
    ])
    monthly = rate.monthly[rate.monthly["field"] == un].sort_values("month").copy()
    return payload, rows, econ, monthly, rate.forecast_first_month


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Объект")
    field_name = st.selectbox("УН", list(FIELDS), index=0)
    spec = FIELDS[field_name]

    cfg0 = C.RunConfig()
    strata = field_strata(cfg0.bundle_date, spec.code)

    st.header("Модель отказов")
    cohort = spec.code in C.MC_COHORT_FIELDS
    b_label = "B — спайк+константа" + (
        f" (монтажи {C.MC_INSTALL_COHORT_START.year}+)" if cohort else ""
    )
    options = [b_label, "A — текущий бандл"]
    # The default follows each field's own measured calibration, not a house preference:
    # B wins on Mc and on Vt's sour strata, but under-predicts Ya.
    choice = st.radio("Вариант", options, index=0 if spec.default_option == "B" else 1)

    if choice.startswith("B"):
        fits = fit_option_b(spec.code, strata)
    else:
        fits = option_a_params(cfg0.bundle_date, strata)

    if not fits:
        st.error("Не удалось подогнать ни одной страты (нужно ≥3 отказа).")
        st.stop()

    # Option B is fitted on esp_population (Свод + Big OPEN runs); the registry was fit on
    # Свод + ALL Big-only runs.  Where those disagree materially, B is not comparable to A
    # and the gap must be visible rather than buried in a note.
    if choice.startswith("B"):
        reg_ev = option_a_params(cfg0.bundle_date, strata)
        fit_ev = sum(p["n_failures"] for p in fits.values())
        ref_ev = sum(p["n_failures"] for n, p in reg_ev.items() if n in fits)
        if ref_ev and fit_ev < 0.9 * ref_ev:
            st.error(
                f"**Популяция подгонки неполная:** {fit_ev} отказов против {ref_ev} в "
                f"реестре ({fit_ev / ref_ev:.0%}). B на этом объекте занижает — сравнение "
                "с A некорректно."
            )

    st.caption(spec.note)

    st.subheader(f"Параметры по стратам ({len(fits)})")
    st.dataframe(
        pd.DataFrame([
            {
                "страта": name.replace(f"{spec.code}_", ""),
                "пробегов": p["n_runs"],
                "отказов": p["n_failures"],
                "B50": round(H.quantile(0.5, p), 0),
                "h(0)": round(H.monthly_hazard(0, p), 3),
                "h(300)": round(H.monthly_hazard(300, p), 3),
            }
            for name, p in fits.items()
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "`h(0)` / `h(300)` — вероятность отказа за месяц для насоса возрастом 0 и 300 "
        "сут. Эмпирика: новый насос отказывает ЧАЩЕ. Если `h(0) < h(300)`, форма кривой "
        "противоречит данным."
    )

    with st.expander("Полные параметры"):
        st.dataframe(
            pd.DataFrame([{"страта": n, **{k: round(p[k], 4) for k in PARAM_KEYS}}
                          for n, p in fits.items()]),
            use_container_width=True, hide_index=True,
        )

st.title(f"{field_name} УН — план ремонтов")

params_items = tuple((name, tuple(float(p[k]) for k in PARAM_KEYS)) for name, p in fits.items())
payload, rows, econ, monthly, fcst_first = build_plan(spec.un, params_items)

tab_sum, tab_rate, tab_plan, tab_prio = st.tabs(
    ["1 · Сводка", "2 · Интенсивность отказов (факт vs модель)", "3 · План ремонтов", "4 · Приоритет"]
)

# ── 1. summary ────────────────────────────────────────────────────────────────
with tab_sum:
    fwd = monthly[monthly["month"] >= fcst_first]
    hist = monthly[monthly["month"] < fcst_first]
    n_events = int(sum(len(r["event_dates"] or []) for r in payload["rows"]))

    m = st.columns(5)
    m[0].metric("Скважин в плане", f"{len(payload['rows'])}")
    m[1].metric("Ремонтов в горизонте", f"{n_events}")
    m[2].metric("Ремонтов в месяц", f"{fwd['predicted_failures'].sum() / max(len(fwd), 1):.1f}")
    m[3].metric("Горизонт", f"{payload['start_date']} → {payload['end_date']}")
    m[4].metric("Скважин с ремонтом", f"{int((rows['Ремонтов в горизонте'] > 0).sum())}")

    st.subheader("Калибровка по истории: факт vs модель")
    years = hist.assign(year=hist["month"].str[:4]).groupby("year").agg(
        факт=("observed_failures", "sum"),
        модель=("predicted_failures", "sum"),
        фонд_ср=("fleet_size", "mean"),
    ).reset_index()
    years["модель/факт"] = (years["модель"] / years["факт"].replace(0, np.nan)).round(2)
    years["модель"] = years["модель"].round(1)
    years["фонд_ср"] = years["фонд_ср"].round(1)
    st.dataframe(years, use_container_width=True, hide_index=True)
    st.caption(
        "«модель/факт» = 1.0 — попадание. Агрегатная калибровка НЕ различает A и B: они "
        "отличаются ФОРМОЙ, а не уровнем. Разница видна в прогнозе и в приоритете, а не "
        "в этой таблице."
    )

    st.info(
        "**Чему здесь можно верить:** месячному КОЛИЧЕСТВУ ремонтов (±20%) и приоритету "
        "по риску (вкладка 4). **Чему нельзя:** конкретной ДАТЕ по скважине — при "
        "постоянной интенсивности остаточный ресурс не зависит от возраста, поэтому дата "
        "в плане это детерминированный псевдослучайный розыгрыш, а не прогноз."
    )

# ── 2. failure rate ───────────────────────────────────────────────────────────
with tab_rate:
    view = monthly[monthly["month"] >= "2024-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["month"], y=view["observed_failures"], mode="lines+markers",
        line=dict(color=OBS_COLOR, width=2.5), marker=dict(size=6), name="Факт (отказов)",
    ))
    fig.add_trace(go.Scatter(
        x=view["month"], y=view["predicted_failures"], mode="lines",
        line=dict(color=PRED_COLOR, width=2.5), name="Модель",
    ))
    fig.add_vline(x=fcst_first, line=dict(color="#888", width=1, dash="dash"))
    fig.add_annotation(x=fcst_first, y=1.02, yref="paper", showarrow=False,
                       text="прогноз →", font=dict(size=11, color="#888"))
    fig.update_layout(height=420, hovermode="x unified", xaxis_title="Месяц",
                      yaxis_title="Отказов в месяц", legend=dict(orientation="h", y=-0.22),
                      margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=view["month"], y=view["observed_rate"], mode="lines+markers",
                              line=dict(color=OBS_COLOR, width=2), name="Факт"))
    fig2.add_trace(go.Scatter(x=view["month"], y=view["predicted_rate"], mode="lines",
                              line=dict(color=PRED_COLOR, width=2), name="Модель"))
    fig2.add_vline(x=fcst_first, line=dict(color="#888", width=1, dash="dash"))
    fig2.update_layout(height=340, hovermode="x unified", xaxis_title="Месяц",
                       yaxis_title="Отказов на скважину в месяц",
                       legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Помесячная таблица"):
        st.dataframe(
            view[["month", "fleet_size", "observed_failures", "predicted_failures",
                  "observed_rate", "predicted_rate"]].round(4),
            use_container_width=True, hide_index=True,
        )

# ── 3. plan ───────────────────────────────────────────────────────────────────
with tab_plan:
    st.caption(
        "«Дней до ремонта» = расстояние от начала прогноза до первого 0 в строке "
        "скважины. Это ОСТАТОЧНЫЙ ресурс от сегодня, а не полный ННО — поэтому он "
        "не сравним с «Факт ННО» (возраст насоса) и не может ему противоречить. "
        "Пусто = ремонт в горизонте не прогнозируется."
    )
    show = rows.drop(columns=["wid"]).copy()
    only = st.checkbox("Только скважины с ремонтом", value=False)
    if only:
        show = show[show["Ремонтов в горизонте"] > 0]
    st.dataframe(
        show.sort_values("Дней до ремонта", na_position="last"),
        use_container_width=True, hide_index=True, height=520,
    )

    hist_days = rows["Дней до ремонта"].dropna()
    if len(hist_days):
        fig = go.Figure(go.Histogram(x=hist_days, nbinsx=30, marker_color=FCST_COLOR))
        fig.update_layout(height=300, xaxis_title="Дней до ремонта", yaxis_title="Скважин",
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Медиана {hist_days.median():.0f} сут при середине горизонта "
            f"{len(payload['dates']) / 2:.0f} сут — распределение почти равномерное, "
            "что и ожидается при постоянной интенсивности."
        )

# ── 4. priority ───────────────────────────────────────────────────────────────
with tab_prio:
    st.caption(
        "Вероятность отказа по скважинам почти одинакова, а дебит различается в десятки "
        "раз — поэтому осмысленный приоритет даёт РИСК = ожидаемые отказы × дебит нефти, "
        "а не вероятность."
    )
    e = econ.merge(rows[["wid", "Дней до ремонта", "Ремонтов в горизонте"]], on="wid", how="left")
    e = e[e["oil_rate"] > 0].sort_values("risk", ascending=False)
    e["Приоритет"] = range(1, len(e) + 1)
    out = e.rename(columns={
        "wid": "Скважина", "E": "Ожид. отказов", "oil_rate": "ДебН, т/сут",
        "risk": "Риск (отказы×дебит)",
    })[["Приоритет", "Скважина", "Ожид. отказов", "ДебН, т/сут", "Риск (отказы×дебит)",
        "Дней до ремонта", "Ремонтов в горизонте"]]
    st.dataframe(out.round(2), use_container_width=True, hide_index=True, height=520)

    top = e.head(20)
    fig = go.Figure(go.Bar(x=top["wid"], y=top["risk"], marker_color=FCST_COLOR))
    fig.update_layout(height=340, xaxis_title="", yaxis_title="Риск = ожид. отказы × дебит",
                      margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
